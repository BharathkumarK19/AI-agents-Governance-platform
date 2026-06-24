import json
import logging
import os
import re
import time
import traceback
import argparse
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv
from openai import AuthenticationError
from langchain_openai import ChatOpenAI

from agents.async_utils import resolve_agent_result
from agents.crewai_runtime import get_crewai_components
from agents.tools import build_source_bundle, search_tavily


load_dotenv()

MODEL_NAME = "openrouter/auto"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MAX_TOKENS = 2048


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "agent_name": getattr(record, "agent_name", None),
            "task_name": getattr(record, "task_name", None),
            "message": record.getMessage(),
        }
        for key in (
            "input_data",
            "output_data",
            "error",
            "confidence_score",
            "hallucination_flag",
            "trace_id",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["stack_trace"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logger():
    logger = logging.getLogger("crewai_observability")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = configure_logger()


class Trace:
    def __init__(self):
        self.trace_id = str(uuid4())
        self.steps = []

    def add_span(
        self,
        step_name,
        input_data,
        output_data,
        start_time,
        end_time,
        status="success",
        error=None,
        hallucination_flag=False,
        evaluation_score=None,
        agent_responsible=None,
        issues=None,
    ):
        self.steps.append(
            {
                "step_name": step_name,
                "input_data": input_data,
                "output_data": output_data,
                "start_time": start_time,
                "end_time": end_time,
                "duration": round(end_time - start_time, 6),
                "status": status,
                "error": error,
                "hallucination_flag": hallucination_flag,
                "evaluation_score": evaluation_score,
                "agent_responsible": agent_responsible,
                "issues": issues or [],
            }
        )

    def annotate_span(self, step_name, hallucination_flag, evaluation_score, agent_responsible, issues):
        for span in self.steps:
            if span["step_name"] == step_name:
                span["hallucination_flag"] = hallucination_flag
                span["evaluation_score"] = evaluation_score
                span["agent_responsible"] = agent_responsible
                span["issues"] = issues
                if hallucination_flag:
                    span["status"] = "failed"
                break

    def to_dict(self):
        return {"trace_id": self.trace_id, "steps": self.steps}

    def replay(self):
        return self.to_dict()


def require_env(name):
    value = os.getenv(name)
    if value is not None:
        value = value.strip()
    if not value:
        raise ValueError(f"{name} is not set.")
    return value


def format_runtime_error(exc):
    message = str(exc)
    lowered = message.lower()

    if isinstance(exc, AuthenticationError) or "401" in lowered or "user not found" in lowered:
        return (
            "OpenRouter authentication failed. Check `OPENROUTER_API_KEY` in `.env` "
            "and replace it with a valid active key."
        )

    if "402" in lowered and "max_tokens" in lowered and "credits" in lowered:
        return (
            "OpenRouter rejected the request because the token budget is too high for the "
            "available credits. Lower `OPENROUTER_MAX_TOKENS` in `.env` to something like "
            "`1024` or `2048`, or add more OpenRouter credits."
        )

    return message


def resolve_max_tokens():
    raw_value = os.getenv("OPENROUTER_MAX_TOKENS", "").strip()
    if not raw_value:
        return DEFAULT_MAX_TOKENS

    try:
        parsed = int(raw_value)
    except ValueError:
        return DEFAULT_MAX_TOKENS

    return parsed if parsed > 0 else DEFAULT_MAX_TOKENS


def build_llms():
    _, _, LLM, _, _ = get_crewai_components()
    api_key = require_env("OPENROUTER_API_KEY")
    max_tokens = resolve_max_tokens()
    langchain_llm = ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    crewai_llm = LLM(
        model=MODEL_NAME,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return langchain_llm, crewai_llm


def create_agents(llm):
    Agent, _, _, _, _ = get_crewai_components()
    research_agent = Agent(
        role="Research Agent",
        goal="Find accurate and relevant information from the internet.",
        backstory="Expert researcher who gathers reliable and up-to-date data using tools.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    analysis_agent = Agent(
        role="Analysis Agent",
        goal="Analyze research data and extract meaningful insights.",
        backstory="Skilled at identifying patterns, trends, and key insights.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    summary_agent = Agent(
        role="Summary Agent",
        goal="Create clear, structured, and presentation-ready reports.",
        backstory="Expert communicator who simplifies complex insights.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    evaluation_agent = Agent(
        role="Fact Checker / Evaluation Agent",
        goal="Verify factual correctness of outputs and detect hallucinations.",
        backstory="Expert evaluator trained to validate outputs against evidence.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    return research_agent, analysis_agent, summary_agent, evaluation_agent


def create_step_inputs(query, source_bundle, research_output=None, analysis_output=None):
    research_input = (
        f"User query: {query}\n\n"
        f"Live web findings:\n{source_bundle}\n\n"
        "Create a detailed research report focused on trends, facts, and real-world data. "
        "Use the provided findings as evidence."
    )
    analysis_input = (
        f"User query: {query}\n\n"
        f"Research output:\n{research_output}\n\n"
        "Analyze the research output and extract key insights, patterns, trends, and "
        "conclusions. Return concise bullet-point insights."
    )
    summary_input = (
        f"User query: {query}\n\n"
        f"Analysis output:\n{analysis_output}\n\n"
        "Convert the analysis into a final structured report. Include headings, bullet "
        "points, and make it presentation-ready."
    )
    return {
        "Research": research_input,
        "Analysis": analysis_input,
        "Summary": summary_input,
    }


def build_evaluation_input(query, research_output, analysis_output, summary_output):
    schema = {
        "hallucination_detected": True,
        "confidence_score": 0.0,
        "issues": [
            {
                "claim": "string",
                "reason": "string",
                "severity": "low|medium|high",
            }
        ],
        "verdict": "Reliable | Needs Review",
    }
    return (
        f"User query: {query}\n\n"
        "You are evaluating hallucinations in a multi-agent system.\n"
        "Follow this rubric strictly:\n"
        "1. Extract key claims from the summary.\n"
        "2. Compare each claim with the research data.\n"
        "3. Mark each claim as Supported, Partially supported, or Not supported.\n"
        "4. Detect unsupported claims, contradictions, or fabricated facts.\n"
        "5. Produce the final verdict.\n\n"
        f"Research output (source of truth):\n{research_output}\n\n"
        f"Analysis output:\n{analysis_output}\n\n"
        f"Final summary output:\n{summary_output}\n\n"
        "Return only valid JSON using this schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )


def normalize_output(result):
    for attribute in ("raw", "output", "content", "text"):
        value = getattr(result, attribute, None)
        if value:
            return value
    return str(result)


def extract_token_usage(result):
    for attribute in ("usage_metrics", "token_usage", "usage"):
        usage = getattr(result, attribute, None)
        if usage is None:
            continue
        if isinstance(usage, dict):
            for key in ("total_tokens", "tokens", "total"):
                if key in usage and usage[key] is not None:
                    return usage[key]
        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is not None:
            return total_tokens
    return None


def parse_json_block(text):
    if isinstance(text, dict):
        return text
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def log_event(level, message, agent_name, task_name, **kwargs):
    LOGGER.log(
        level,
        message,
        extra={"agent_name": agent_name, "task_name": task_name, **kwargs},
    )


def execute_agent_step(agent, agent_name, task_name, input_data, trace, metrics):
    start_time = time.perf_counter()
    log_event(
        logging.INFO,
        "agent_step_started",
        agent_name,
        task_name,
        input_data=input_data,
        trace_id=trace.trace_id,
    )
    try:
        result = resolve_agent_result(agent.kickoff(input_data))
        output_data = normalize_output(result)
        end_time = time.perf_counter()
        trace.add_span(
            step_name=task_name,
            input_data=input_data,
            output_data=output_data,
            start_time=start_time,
            end_time=end_time,
            agent_responsible=agent_name,
        )
        metrics["per_agent_execution_time"][agent_name] = round(end_time - start_time, 6)
        metrics["number_of_steps_executed"] += 1
        token_count = extract_token_usage(result)
        if token_count is not None:
            metrics["number_of_tokens"] += token_count
        log_event(
            logging.INFO,
            "agent_step_completed",
            agent_name,
            task_name,
            output_data=output_data,
            trace_id=trace.trace_id,
        )
        return output_data
    except Exception as exc:
        friendly_error = format_runtime_error(exc)
        end_time = time.perf_counter()
        trace.add_span(
            step_name=task_name,
            input_data=input_data,
            output_data=None,
            start_time=start_time,
            end_time=end_time,
            status="failed",
            error=friendly_error,
            hallucination_flag=False,
            evaluation_score=None,
            agent_responsible=agent_name,
        )
        metrics["per_agent_execution_time"][agent_name] = round(end_time - start_time, 6)
        metrics["number_of_steps_executed"] += 1
        metrics["error_count"] += 1
        LOGGER.exception(
            "agent_step_failed",
            extra={
                "agent_name": agent_name,
                "task_name": task_name,
                "input_data": input_data,
                "error": friendly_error,
                "trace_id": trace.trace_id,
            },
        )
        raise RuntimeError(friendly_error) from exc


def infer_agent_responsibility(issue, research_output, analysis_output, summary_output):
    claim = (issue.get("claim") or "").lower()
    reason = (issue.get("reason") or "").lower()
    research_text = (research_output or "").lower()
    analysis_text = (analysis_output or "").lower()
    summary_text = (summary_output or "").lower()

    if any(term in reason for term in ("fabricated", "not supported", "unsupported", "invented")):
        if claim and claim in research_text:
            return "Research Agent"
        if claim and claim in analysis_text and claim not in research_text:
            return "Analysis Agent"
        return "Summary Agent"

    if any(term in reason for term in ("misinterpret", "overstated", "incorrect conclusion", "distorted")):
        return "Analysis Agent"

    if claim and claim in summary_text and claim not in research_text and claim not in analysis_text:
        return "Summary Agent"

    if claim and claim in research_text:
        return "Research Agent"

    return "Summary Agent"


def evaluate_hallucination(
    evaluation_agent,
    query,
    research_output,
    analysis_output,
    summary_output,
    trace,
    metrics,
):
    input_data = build_evaluation_input(query, research_output, analysis_output, summary_output)
    start_time = time.perf_counter()
    log_event(
        logging.INFO,
        "evaluation_started",
        "Evaluation Agent",
        "Evaluation",
        input_data=input_data,
        trace_id=trace.trace_id,
    )
    try:
        result = resolve_agent_result(evaluation_agent.kickoff(input_data))
        output_text = normalize_output(result)
        try:
            evaluation = parse_json_block(output_text)
        except (TypeError, json.JSONDecodeError, ValueError):
            # Some model responses can be empty or non-JSON despite the schema instruction.
            evaluation = {
                "hallucination_detected": False,
                "confidence_score": 0.0,
                "issues": [],
                "verdict": "Needs Review",
                "explanation": (
                    "Evaluation agent returned non-JSON output, so strict hallucination "
                    "verification could not be completed."
                ),
                "raw_evaluator_output": (output_text or "")[:1200],
            }
        evaluation.setdefault("hallucination_detected", False)
        evaluation.setdefault("confidence_score", 0.0)
        evaluation.setdefault("issues", [])
        evaluation.setdefault("verdict", "Reliable")
        evaluation.setdefault(
            "explanation",
            "Evaluation completed against available research evidence.",
        )

        issues = evaluation["issues"]
        if not isinstance(issues, list):
            issues = []
            evaluation["issues"] = issues
        responsible_agents = []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            agent_responsible = infer_agent_responsibility(
                issue, research_output, analysis_output, summary_output
            )
            issue["agent_responsible"] = agent_responsible
            responsible_agents.append(agent_responsible)
            metrics["agent_error_breakdown"][agent_responsible] = (
                metrics["agent_error_breakdown"].get(agent_responsible, 0) + 1
            )

        try:
            confidence_score = float(evaluation["confidence_score"])
        except (TypeError, ValueError):
            confidence_score = 0.0
            evaluation["confidence_score"] = confidence_score
        hallucination_flag = bool(evaluation["hallucination_detected"])
        if hallucination_flag:
            metrics["number_of_flagged_outputs"] += 1

        metrics["confidence_scores"].append(confidence_score)

        agent_responsible = responsible_agents[0] if responsible_agents else None
        if hallucination_flag and agent_responsible:
            if agent_responsible == "Research Agent":
                trace.annotate_span("Research", True, confidence_score, agent_responsible, issues)
            elif agent_responsible == "Analysis Agent":
                trace.annotate_span("Analysis", True, confidence_score, agent_responsible, issues)
            else:
                trace.annotate_span("Summary", True, confidence_score, agent_responsible, issues)

        end_time = time.perf_counter()
        trace.add_span(
            step_name="Evaluation",
            input_data=input_data,
            output_data=evaluation,
            start_time=start_time,
            end_time=end_time,
            status="failed" if hallucination_flag else "success",
            hallucination_flag=hallucination_flag,
            evaluation_score=confidence_score,
            agent_responsible=agent_responsible,
            issues=issues,
        )
        metrics["per_agent_execution_time"]["Evaluation Agent"] = round(end_time - start_time, 6)
        metrics["number_of_steps_executed"] += 1
        token_count = extract_token_usage(result)
        if token_count is not None:
            metrics["number_of_tokens"] += token_count

        log_event(
            logging.INFO,
            "evaluation_completed",
            "Evaluation Agent",
            "Evaluation",
            output_data=evaluation,
            confidence_score=confidence_score,
            hallucination_flag=hallucination_flag,
            trace_id=trace.trace_id,
        )
        return evaluation
    except Exception as exc:
        friendly_error = format_runtime_error(exc)
        end_time = time.perf_counter()
        fallback_evaluation = {
            "hallucination_detected": False,
            "confidence_score": 0.0,
            "issues": [],
            "verdict": "Needs Review",
            "explanation": (
                "Hallucination verification could not be completed because the "
                f"evaluation agent failed: {friendly_error}"
            ),
            "suggested_failed_agents": [],
            "raw_evaluator_output": "",
        }
        trace.add_span(
            step_name="Evaluation",
            input_data=input_data,
            output_data=fallback_evaluation,
            start_time=start_time,
            end_time=end_time,
            status="failed",
            error=friendly_error,
            hallucination_flag=False,
            evaluation_score=None,
            agent_responsible="Evaluation Agent",
        )
        metrics["per_agent_execution_time"]["Evaluation Agent"] = round(end_time - start_time, 6)
        metrics["number_of_steps_executed"] += 1
        metrics["error_count"] += 1
        metrics["confidence_scores"].append(0.0)
        LOGGER.exception(
            "evaluation_failed",
            extra={
                "agent_name": "Evaluation Agent",
                "task_name": "Evaluation",
                "input_data": input_data,
                "error": friendly_error,
                "trace_id": trace.trace_id,
            },
        )
        return fallback_evaluation


def retry_summary_if_needed(summary_agent, query, analysis_output, evaluation, trace, metrics):
    if not evaluation.get("hallucination_detected"):
        return None

    issues = evaluation.get("issues", [])
    retry_input = (
        f"User query: {query}\n\n"
        f"Previous analysis output:\n{analysis_output}\n\n"
        f"Hallucination issues found:\n{json.dumps(issues, indent=2)}\n\n"
        "Rewrite the final answer conservatively. Remove unsupported claims, avoid fabricated "
        "facts, and explicitly note uncertainty where evidence is incomplete."
    )
    retry_output = execute_agent_step(
        summary_agent,
        agent_name="Summary Agent",
        task_name="Summary Retry",
        input_data=retry_input,
        trace=trace,
        metrics=metrics,
    )
    return (
        "This response may contain uncertain information.\n\n"
        f"Safer fallback summary:\n{retry_output}"
    )


def build_metrics():
    return {
        "total_execution_time": 0.0,
        "per_agent_execution_time": {},
        "number_of_tokens": 0,
        "number_of_steps_executed": 0,
        "error_count": 0,
        "hallucination_rate": 0.0,
        "average_confidence_score": 0.0,
        "number_of_flagged_outputs": 0,
        "agent_error_breakdown": {},
        "confidence_scores": [],
    }


def finalize_metrics(metrics):
    confidence_scores = metrics.pop("confidence_scores", [])
    if confidence_scores:
        metrics["average_confidence_score"] = round(
            sum(confidence_scores) / len(confidence_scores), 6
        )
    if metrics["number_of_steps_executed"]:
        metrics["hallucination_rate"] = round(
            metrics["number_of_flagged_outputs"] / metrics["number_of_steps_executed"], 6
        )
    return metrics


def run_system(query):
    total_start = time.perf_counter()
    trace = Trace()
    metrics = build_metrics()
    final_result = None
    hallucination_report = None

    try:
        _, crewai_llm = build_llms()
        (
            research_agent,
            analysis_agent,
            summary_agent,
            evaluation_agent,
        ) = create_agents(crewai_llm)

        results = search_tavily(query)
        source_bundle = build_source_bundle(results)

        step_inputs = create_step_inputs(query, source_bundle)
        research_output = execute_agent_step(
            research_agent,
            agent_name="Research Agent",
            task_name="Research",
            input_data=step_inputs["Research"],
            trace=trace,
            metrics=metrics,
        )

        step_inputs = create_step_inputs(
            query,
            source_bundle,
            research_output=research_output,
        )
        analysis_output = execute_agent_step(
            analysis_agent,
            agent_name="Analysis Agent",
            task_name="Analysis",
            input_data=step_inputs["Analysis"],
            trace=trace,
            metrics=metrics,
        )

        step_inputs = create_step_inputs(
            query,
            source_bundle,
            research_output=research_output,
            analysis_output=analysis_output,
        )
        summary_output = execute_agent_step(
            summary_agent,
            agent_name="Summary Agent",
            task_name="Summary",
            input_data=step_inputs["Summary"],
            trace=trace,
            metrics=metrics,
        )

        hallucination_report = evaluate_hallucination(
            evaluation_agent,
            query,
            research_output,
            analysis_output,
            summary_output,
            trace,
            metrics,
        )

        if hallucination_report.get("hallucination_detected"):
            responsible_agents = sorted(
                {issue.get("agent_responsible", "Summary Agent") for issue in hallucination_report.get("issues", [])}
            )
            hallucination_report["explanation"] = (
                "Hallucination was detected because one or more claims in the final summary "
                "were unsupported, contradictory, or fabricated relative to the research evidence."
            )
            hallucination_report["suggested_failed_agents"] = responsible_agents
            retried_summary = retry_summary_if_needed(
                summary_agent,
                query,
                analysis_output,
                hallucination_report,
                trace,
                metrics,
            )
            final_result = retried_summary or "This response may contain uncertain information."
        else:
            if hallucination_report.get("verdict") == "Reliable":
                hallucination_report["explanation"] = (
                    "No major unsupported or fabricated claims were detected against the research evidence."
                )
            hallucination_report.setdefault("suggested_failed_agents", [])
            final_result = summary_output
    except Exception as exc:
        metrics["error_count"] += 1
        friendly_error = format_runtime_error(exc)
        LOGGER.exception(
            "system_execution_failed",
            extra={
                "agent_name": "System",
                "task_name": "Run",
                "input_data": query,
                "error": friendly_error,
                "trace_id": trace.trace_id,
            },
        )
        final_result = {
            "status": "failed",
            "error": friendly_error,
            "stack_trace": traceback.format_exc(),
        }
        hallucination_report = {
            "hallucination_detected": False,
            "confidence_score": 0.0,
            "issues": [],
            "verdict": "Needs Review",
            "explanation": "Evaluation could not be completed because the system execution failed.",
            "suggested_failed_agents": [],
        }
    finally:
        metrics["total_execution_time"] = round(time.perf_counter() - total_start, 6)
        metrics = finalize_metrics(metrics)

    return final_result, hallucination_report, trace, metrics


def main():
    parser = argparse.ArgumentParser(description="Run the governance research pipeline.")
    parser.add_argument("--query", type=str, default=None, help="Run without interactive prompt.")
    args = parser.parse_args()

    if args.query is not None:
        query = args.query.strip()
    else:
        try:
            query = input("Enter your query: ").strip()
        except EOFError:
            print('No input received. Pass a query with --query "..." for non-interactive runs.')
            return

    if not query:
        print("A query is required.")
        return

    final_result, hallucination_report, trace, metrics = run_system(query)
    final_payload = {
        "final_answer": final_result,
        "hallucination_report": hallucination_report,
        "trace": trace.to_dict(),
        "metrics_summary": metrics,
        "replay": trace.replay(),
    }
    print(json.dumps(final_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
