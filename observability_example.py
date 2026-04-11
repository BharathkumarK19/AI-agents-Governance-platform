import json
import os
from pathlib import Path

from dotenv import load_dotenv

from hallucination_detector import HallucinationDetector
from hallucination_report_engine import HallucinationReportEngine
from logger import ObservabilityLogger
import yaml


load_dotenv()


def run_observability_example(agent, query, retrieved_docs, tools_used, neon_db_url):
    context = _normalize_context(retrieved_docs)
    response = agent.run(query)
    intermediate_steps = [
        {
            "agent": "Search Agent",
            "action": "Retrieved documents",
            "output_summary": "Loaded retrieved documents for the query.",
            "source_used": bool(retrieved_docs),
            "basis": "retrieved context",
        },
        {
            "agent": "LLM",
            "action": "Generated response",
            "output_summary": response[:200],
            "source_used": bool(retrieved_docs),
            "basis": "context + prior knowledge",
        },
    ]

    detector = HallucinationDetector()
    analysis = detector.analyze(
        query=query,
        response=response,
        context=context,
        tools_used=tools_used,
    )
    report_engine = HallucinationReportEngine()
    report = report_engine.generate_report(
        query=query,
        response=response,
        context=context,
        tools_used=tools_used,
        intermediate_steps=intermediate_steps,
    )
    saved_report = report_engine.save_report(report)
    analysis["report_reference_id"] = saved_report["report_reference_id"]
    analysis["report_file_path"] = saved_report["report_file_path"]

    logger = ObservabilityLogger(
        neon_db_url=neon_db_url,
        local_path=str(Path("logs") / "observability_logs.json"),
    )
    try:
        logger.log(
            query=query,
            response=response,
            context=context,
            analysis=analysis,
            tools_used=tools_used,
        )
    finally:
        logger.close()

    return {
        "response": response,
        "hallucination_analysis": analysis,
        "hallucination_report": saved_report["report"],
    }


def _load_datafetch_config(config_path: str = "datafetch.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}

    if not isinstance(data, dict):
        return {}

    return data


class ExampleAgent:
    def run(self, query):
        return (
            "The latest results indicate strong performance, but this is likely to change "
            "if external market conditions shift."
        )


def _normalize_context(retrieved_docs):
    if isinstance(retrieved_docs, str):
        return retrieved_docs
    if isinstance(retrieved_docs, list):
        return "\n".join(_normalize_context(item) for item in retrieved_docs)
    if isinstance(retrieved_docs, dict):
        return json.dumps(retrieved_docs, ensure_ascii=False, default=str)
    return str(retrieved_docs)


if __name__ == "__main__":
    config = _load_datafetch_config()
    neon_db_url = os.getenv("NEON_DATABASE_URL", "").strip() or config.get("neon_database_url", "")

    result = run_observability_example(
        agent=ExampleAgent(),
        query="What are the latest market results?",
        retrieved_docs=[
            "Market summary says revenue grew 8% year-over-year.",
            "No live source confirms results from today.",
        ],
        tools_used=[],
        neon_db_url=neon_db_url,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
