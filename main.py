import json
import logging
import os
import re
import time
import traceback
import ast
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv
from openai import AuthenticationError
from langchain_openai import ChatOpenAI
import yaml

from agents.crewai_runtime import get_crewai_components
from agents.tools import build_source_bundle, search_tavily
from hallucination_report_engine import HallucinationReportEngine
from self_healing_pipeline import SelfHealingPipeline


load_dotenv()

MODEL_NAME = "openrouter/auto"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MAX_TOKENS = 1024


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
        return json.dumps(payload, ensure_ascii=False, default=str)


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


def _max_tokens() -> int:
    raw = os.getenv("OPENROUTER_MAX_TOKENS", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_MAX_TOKENS


def format_runtime_error(exc):
    message = str(exc)
    lowered = message.lower()

    if isinstance(exc, AuthenticationError) or "401" in lowered or "user not found" in lowered:
        return (
            "OpenRouter authentication failed. Check `OPENROUTER_API_KEY` in `.env` "
            "and replace it with a valid active key."
        )

    return message


def log_event(level, message, agent_name, task_name, **kwargs):
    LOGGER.log(
        level,
        message,
        extra={"agent_name": agent_name, "task_name": task_name, **kwargs},
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


def build_llms():
    _, _, LLM, _, _ = get_crewai_components()
    api_key = require_env("OPENROUTER_API_KEY")
    langchain_llm = ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=_max_tokens(),
    )
    crewai_llm = LLM(
        model=MODEL_NAME,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        temperature=0.2,
        max_tokens=_max_tokens(),
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
            block = match.group(0)
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                return ast.literal_eval(block)
        return ast.literal_eval(cleaned)


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
        result = agent.kickoff(input_data)
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
        result = evaluation_agent.kickoff(input_data)
        output_text = normalize_output(result)
        evaluation = parse_json_block(output_text)
        evaluation.setdefault("hallucination_detected", False)
        evaluation.setdefault("confidence_score", 0.0)
        evaluation.setdefault("issues", [])
        evaluation.setdefault("verdict", "Reliable")

        issues = evaluation["issues"]
        responsible_agents = []
        for issue in issues:
            agent_responsible = infer_agent_responsibility(
                issue, research_output, analysis_output, summary_output
            )
            issue["agent_responsible"] = agent_responsible
            responsible_agents.append(agent_responsible)
            metrics["agent_error_breakdown"][agent_responsible] = (
                metrics["agent_error_breakdown"].get(agent_responsible, 0) + 1
            )

        confidence_score = float(evaluation["confidence_score"])
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
        trace.add_span(
            step_name="Evaluation",
            input_data=input_data,
            output_data=None,
            start_time=start_time,
            end_time=end_time,
            status="failed",
            error=friendly_error,
            agent_responsible="Evaluation Agent",
        )
        metrics["per_agent_execution_time"]["Evaluation Agent"] = round(end_time - start_time, 6)
        metrics["number_of_steps_executed"] += 1
        metrics["error_count"] += 1
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
        raise RuntimeError(friendly_error) from exc


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
    return "This response may contain uncertain information.\n\n" f"Safer fallback summary:\n{retry_output}"


def run_research_system(query):
    total_start = time.perf_counter()
    trace = Trace()
    metrics = build_metrics()
    final_result = None
    hallucination_report = None
    source_results = []

    try:
        _, crewai_llm = build_llms()
        research_agent, analysis_agent, summary_agent, evaluation_agent = create_agents(crewai_llm)

        source_results = search_tavily(query)
        source_bundle = build_source_bundle(source_results)

        step_inputs = create_step_inputs(query, source_bundle)
        research_output = execute_agent_step(
            research_agent,
            agent_name="Research Agent",
            task_name="Research",
            input_data=step_inputs["Research"],
            trace=trace,
            metrics=metrics,
        )

        step_inputs = create_step_inputs(query, source_bundle, research_output=research_output)
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
            hallucination_report["explanation"] = (
                "No major unsupported or fabricated claims were detected against the research evidence."
            )
            hallucination_report["suggested_failed_agents"] = []
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

    return final_result, hallucination_report, trace, metrics, source_results


def _prompt_with_default(prompt: str, env_name: str | None = None) -> str:
    default_value = os.getenv(env_name, "").strip() if env_name else ""
    suffix = f" [{default_value}]" if default_value else ""
    value = input(f"{prompt}{suffix}: ").strip()
    if value:
        return value
    if default_value:
        return default_value
    raise ValueError(f"{prompt} is required.")


def _load_datafetch_config(config_path: str = "datafetch.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}

    if not isinstance(data, dict):
        raise ValueError("`datafetch.yaml` must contain a top-level mapping.")

    return data


def _get_runtime_setting(config: dict, key: str, env_name: str | None = None, prompt: str | None = None) -> str:
    value = config.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()

    if env_name:
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            return env_value

    if prompt:
        return _prompt_with_default(prompt, env_name)

    raise ValueError(f"{key} is required.")


def _apply_runtime_env(config: dict) -> None:
    google_service_account_file = config.get("google_service_account_file")
    if isinstance(google_service_account_file, str) and google_service_account_file.strip():
        os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"] = google_service_account_file.strip()


def _save_and_print_advanced_report(
    query,
    response,
    context,
    tools_used,
    intermediate_steps,
    reference_sources=None,
    show_verified_sources=True,
):
    report_engine = HallucinationReportEngine()
    report = report_engine.generate_report(
        query=query,
        response=response,
        context=context,
        tools_used=tools_used,
        intermediate_steps=intermediate_steps,
        reference_sources=reference_sources,
    )
    saved_report = report_engine.save_report(report)
    report_payload = {
        "report_reference_id": saved_report["report_reference_id"],
        "report_file_path": saved_report["report_file_path"],
        "latest_report_file_path": saved_report["latest_report_file_path"],
        "hallucination_report": saved_report["report"],
    }
    print(json.dumps(report_payload, indent=2, ensure_ascii=False, default=str))
    final_label = saved_report["report"].get("final_verdict", {}).get("label", "UNKNOWN")
    if show_verified_sources:
        compared_resources = saved_report["report"].get("evidence_analysis", {}).get("compared_resources", [])
        verified_sources = saved_report["report"].get("evidence_analysis", {}).get("verified_sources", [])
        resources_to_show = verified_sources or compared_resources
        if resources_to_show:
            print("Compared / verified resources:")
            for source in resources_to_show[:10]:
                title = source.get("title") or source.get("source_name") or "source"
                url = source.get("url", "")
                summary = source.get("summary", "").strip()
                print(f"- {title}: {url}")
                if summary:
                    print(f"  Compared using this resource: {summary[:220]}")
    print(f"Verification verdict: {'HALLUCINATED' if final_label == 'HALLUCINATED' else 'NOT HALLUCINATED'}")
    return saved_report


def _build_research_intermediate_steps(trace: Trace) -> list[dict]:
    steps = []
    for index, span in enumerate(trace.steps, start=1):
        steps.append(
            {
                "agent": span.get("agent_responsible") or span.get("step_name") or "Unknown Agent",
                "action": span.get("step_name", "Executed step"),
                "output_summary": _truncate_text(span.get("output_data")),
                "source_used": span.get("step_name") == "Research",
                "basis": "retrieved context + prior analysis" if span.get("step_name") != "Research" else "retrieved context",
            }
        )
    return steps


def _build_business_intermediate_steps(pipeline_result, metrics, user_answers, insight):
    return [
        {
            "agent": "Sheet Ingestion",
            "action": "Fetched and synchronized transaction rows",
            "output_summary": json.dumps(pipeline_result, ensure_ascii=False, default=str),
            "source_used": True,
            "basis": "google sheets + neon",
        },
        {
            "agent": "Query Engine",
            "action": "Computed performance metrics",
            "output_summary": json.dumps(metrics, ensure_ascii=False, default=str),
            "source_used": True,
            "basis": "neon aggregation",
        },
        {
            "agent": "Insight Generator",
            "action": "Generated governance insight",
            "output_summary": _truncate_text(insight),
            "source_used": True,
            "basis": f"metrics + user answers {json.dumps(user_answers, ensure_ascii=False, default=str)}",
        },
    ]


def _verify_business_query(user_query: str, data_profile: dict) -> dict:
    normalized = user_query.lower()
    available_dimensions = set(data_profile.get("available_dimensions", []))
    supported_keywords = {
        "customer": "customer_id",
        "customers": "customer_id",
        "buyer": "customer_id",
        "item": "item_name",
        "items": "item_name",
        "product": "item_name",
        "sku": "item_id",
        "price": "price",
        "pricing": "price",
        "discount": "discount",
        "gst": "gst",
        "tax": "gst",
        "total": "total_price",
        "revenue": "total_price",
        "sales": "total_price",
        "sale": "total_price",
        "profit": "profit",
        "margin": "margin",
        "date": "timestamp",
        "time": "timestamp",
        "trend": "timestamp",
    }

    matched_dimensions = sorted(
        {
            dimension
            for keyword, dimension in supported_keywords.items()
            if keyword in normalized and dimension in available_dimensions
        }
    )
    matched_items = [
        item["item_name"]
        for item in data_profile.get("top_items", [])
        if item.get("item_name") and item["item_name"].lower() in normalized
    ]
    matched_customers = [
        customer["customer_id"]
        for customer in data_profile.get("top_customers", [])
        if customer.get("customer_id") and customer["customer_id"].lower() in normalized
    ]

    supported = bool(matched_dimensions or matched_items or matched_customers)
    if not supported:
        reason = (
            "The question does not map to the available transaction fields or known entities in the ingested business data."
        )
        proof_points = [
            f"Available fields: {', '.join(data_profile.get('available_dimensions', []))}",
            f"Known sample items: {', '.join(item['item_name'] for item in data_profile.get('top_items', [])[:5]) or 'none'}",
            f"Known sample customers: {', '.join(customer['customer_id'] for customer in data_profile.get('top_customers', [])[:5]) or 'none'}",
        ]
        summary = "Question is hallucinated for this business dataset because no relevant data field or known entity supports answering it."
    else:
        reason = "The question maps to fields/entities that exist in the ingested business data."
        proof_points = [
            f"Matched dimensions: {', '.join(matched_dimensions) or 'none'}",
            f"Matched items: {', '.join(matched_items) or 'none'}",
            f"Matched customers: {', '.join(matched_customers) or 'none'}",
            f"Dataset coverage: {data_profile.get('transaction_count', 0)} transactions, {data_profile.get('item_count', 0)} items, {data_profile.get('customer_count', 0)} customers",
        ]
        summary = "Question is not hallucinated for this business dataset because it matches available business fields or known entities."

    return {
        "hallucinated": not supported,
        "reason": reason,
        "proof_points": proof_points,
        "summary": summary,
        "matched_dimensions": matched_dimensions,
        "matched_items": matched_items,
        "matched_customers": matched_customers,
    }


def _verify_business_response(
    user_query: str,
    insight: str,
    metrics: dict,
    data_profile: dict,
    user_answers: dict,
) -> dict:
    from hallucination_detector import HallucinationDetector

    detector = HallucinationDetector()
    verification_context = json.dumps(
        {
            "metrics": metrics,
            "data_profile": data_profile,
            "user_answers": user_answers,
        },
        ensure_ascii=False,
        default=str,
    )
    analysis = detector.analyze(
        query=user_query,
        response=insight,
        context=verification_context,
        tools_used=["google_sheets", "neon_postgresql"],
    )

    response_numbers = sorted(set(re.findall(r"\b\d+(?:\.\d+)?\b", insight)))
    context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", verification_context))
    unsupported_numbers = [value for value in response_numbers if value not in context_numbers][:8]

    proof_points = [
        f"Context overlap score: {analysis['overlap_score']}",
        f"Risk score: {analysis['risk_score']}",
        f"Unsupported claims detected: {len(analysis.get('unsupported_claims', []))}",
        f"Metrics used for verification: {json.dumps(metrics, ensure_ascii=False, default=str)}",
    ]
    if unsupported_numbers:
        proof_points.append(
            "Response introduced numbers not found in the business context: "
            + ", ".join(unsupported_numbers)
        )

    hallucinated = (
        analysis["label"] == "HALLUCINATED"
        or bool(unsupported_numbers)
        or analysis["overlap_score"] < 0.35
    )
    reason = (
        "The business answer introduced unsupported claims or numeric values that are not grounded in the computed metrics."
        if hallucinated
        else "The business answer stayed aligned with the computed metrics, dataset profile, and user context."
    )
    summary = (
        "Business answer is hallucinated because the generated insight is not sufficiently supported by the available metrics/data context."
        if hallucinated
        else "Business answer is not hallucinated because its claims remain grounded in the available metrics/data context."
    )

    return {
        "hallucinated": hallucinated,
        "reason": reason,
        "summary": summary,
        "proof_points": proof_points,
        "analysis": analysis,
        "unsupported_numbers": unsupported_numbers,
    }


def _print_short_business_verification(title: str, verification: dict) -> None:
    print(title)
    print(f"- Verdict: {'HALLUCINATED' if verification.get('hallucinated') else 'NOT HALLUCINATED'}")
    print(f"- Summary: {verification.get('summary', '')}")
    proof_points = verification.get("proof_points", [])[:3]
    for proof in proof_points:
        print(f"- Proof: {proof}")


def _truncate_text(value, limit: int = 500) -> str:
    text = str(value) if value is not None else ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def run_business_mode():
    from agents.llm import get_llm
    from db import connect_db
    from interactive_query import ask_followup_questions, generate_insight
    from pipeline_runner import run_pipeline
    from query_engine import get_business_data_profile, get_performance_data

    config = _load_datafetch_config()
    _apply_runtime_env(config)
    sheet_url = _get_runtime_setting(
        config,
        "google_sheet_url",
        "GOOGLE_SHEET_URL",
        "Enter Google Sheet URL",
    )
    db_connection_string = _get_runtime_setting(
        config,
        "neon_database_url",
        "NEON_DATABASE_URL",
        "Enter Neon PostgreSQL connection string",
    )

    conn = connect_db(db_connection_string)
    try:
        pipeline_result = run_pipeline(sheet_url, conn)
        print(json.dumps(pipeline_result, indent=2, default=str))

        if getattr(conn, "closed", 1) == 0:
            conn.close()
        conn = connect_db(db_connection_string)

        user_query = input("Ask your business question: ").strip()
        if not user_query:
            raise ValueError("A user query is required.")

        data_profile = get_business_data_profile(conn)
        verification = _verify_business_query(user_query, data_profile)
        _print_short_business_verification("Business query verification:", verification)

        if verification["hallucinated"]:
            response = (
                "This question appears unrelated to the ingested business data.\n\n"
                f"Reason: {verification['reason']}\n"
                f"Summary: {verification['summary']}"
            )
            print(response)
            _save_and_print_advanced_report(
                query=user_query,
                response=response,
                context={
                    "pipeline_result": pipeline_result,
                    "data_profile": data_profile,
                    "verification": verification,
                },
                tools_used=["google_sheets", "neon_postgresql"],
                intermediate_steps=[
                    {
                        "agent": "Business Verifier",
                        "action": "Validated question against dataset coverage",
                        "output_summary": json.dumps(verification, ensure_ascii=False, default=str),
                        "source_used": True,
                        "basis": "dataset schema + known entities",
                    }
                ],
            )
            return

        user_answers = ask_followup_questions(user_query, data_profile=data_profile)
        days = int(user_answers.get("days") or os.getenv("PERFORMANCE_LOOKBACK_DAYS", "30"))
        metrics = get_performance_data(conn, days)
        llm = get_llm()
        insight = generate_insight(metrics, user_answers, llm)
        print(insight)
        response_verification = _verify_business_response(
            user_query=user_query,
            insight=insight,
            metrics=metrics,
            data_profile=data_profile,
            user_answers=user_answers,
        )
        _print_short_business_verification("Business answer verification:", response_verification)
        _save_and_print_advanced_report(
            query=user_query,
            response=insight,
            context={
                "pipeline_result": pipeline_result,
                "data_profile": data_profile,
                "verification": verification,
                "response_verification": response_verification,
                "metrics": metrics,
                "user_answers": user_answers,
            },
            tools_used=["google_sheets", "neon_postgresql"],
            intermediate_steps=_build_business_intermediate_steps(
                pipeline_result=pipeline_result,
                metrics=metrics,
                user_answers=user_answers,
                insight=insight,
            )
            + [
                {
                    "agent": "Business Answer Verifier",
                    "action": "Validated generated answer against business metrics and dataset context",
                    "output_summary": json.dumps(response_verification, ensure_ascii=False, default=str),
                    "source_used": True,
                    "basis": "metrics + dataset profile + user context",
                }
            ],
            show_verified_sources=False,
        )
    finally:
        conn.close()


def run_research_mode():
    query = input("Enter your research query: ").strip()
    if not query:
        raise ValueError("A query is required.")

    final_result, hallucination_report, trace, metrics, source_results = run_research_system(query)
    final_payload = {
        "final_answer": final_result,
        "hallucination_report": hallucination_report,
        "trace": trace.to_dict(),
        "metrics_summary": metrics,
        "replay": trace.replay(),
    }
    print(json.dumps(final_payload, indent=2, ensure_ascii=False, default=str))
    research_context = ""
    for span in trace.steps:
        if span.get("step_name") == "Research":
            research_context = span.get("input_data", "")
            break
    _save_and_print_advanced_report(
        query=query,
        response=final_result,
        context=research_context,
        tools_used=["tavily"],
        intermediate_steps=_build_research_intermediate_steps(trace),
        reference_sources=source_results,
        show_verified_sources=True,
    )

    if hallucination_report.get("hallucination_detected"):
        pipeline = SelfHealingPipeline(neon_db_url=os.getenv("NEON_DATABASE_URL", "").strip() or None)
        try:
            healed_result = pipeline.run(query)
            print(
                json.dumps(
                    {
                        "self_healing_triggered": True,
                        "self_healing_result": healed_result,
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )
            healed_label = healed_result.get("scores", {}).get("label", "UNKNOWN")
            print(f"Verification verdict: {'HALLUCINATED' if healed_label == 'HALLUCINATED' else 'NOT HALLUCINATED'}")
        finally:
            pipeline.close()


def main():
    mode = input(
        "Choose mode: A = normal research, B = business research: "
    ).strip().lower()

    if mode == "a":
        run_research_mode()
        return
    if mode == "b":
        run_business_mode()
        return

    raise ValueError("Invalid choice. Enter `A` for normal research or `B` for business research.")


if __name__ == "__main__":
    main()
