import logging
import json
from functools import lru_cache
from importlib import import_module
from typing import Any

from db import create_transactions_table, find_existing_transaction, insert_transaction
from sheet_ingestion import fetch_sheet_data


@lru_cache(maxsize=1)
def _load_existing_agents() -> tuple[Any, Any, Any]:
    try:
        research_module = import_module("agents.research_agent")
        analysis_module = import_module("agents.analysis_agent")
        summary_module = import_module("agents.summary_agent")
        return (
            getattr(research_module, "research_agent"),
            getattr(analysis_module, "analysis_agent"),
            getattr(summary_module, "summary_agent"),
        )
    except ModuleNotFoundError:
        from agents.llm import get_crewai_llm
        from agents.research_pipeline import (
            create_analysis_agent,
            create_research_agent,
            create_summary_agent,
        )

        llm = get_crewai_llm()
        return (
            create_research_agent(llm),
            create_analysis_agent(llm),
            create_summary_agent(llm),
        )


def _call_agent(agent: Any, payload: Any) -> Any:
    if hasattr(agent, "kickoff") and callable(agent.kickoff):
        kickoff_payload = payload
        if isinstance(payload, dict):
            kickoff_payload = json.dumps(payload, default=str, ensure_ascii=False)
        elif not isinstance(payload, str):
            kickoff_payload = str(payload)
        return agent.kickoff(kickoff_payload)
    if callable(agent):
        return agent(payload)
    raise TypeError(f"Unsupported agent interface: {type(agent)!r}")


def _normalize_agent_output(result: Any) -> Any:
    if isinstance(result, (dict, list, str, int, float, bool)) or result is None:
        return result
    for attribute in ("raw", "output", "content", "text"):
        value = getattr(result, attribute, None)
        if value is not None:
            return value
    return str(result)


def _build_transaction_prompts(txn: dict[str, Any]) -> tuple[str, str, str]:
    transaction_json = json.dumps(txn, default=str, ensure_ascii=False, indent=2)
    research_prompt = (
        "You are reviewing a transaction record for a governance-oriented retail pipeline.\n"
        "Study the transaction and produce a concise factual research note.\n"
        "Focus on transaction context, pricing integrity signals, discount posture, tax fields, "
        "margin/profit observations, and any anomalies worth checking.\n\n"
        f"Transaction:\n{transaction_json}"
    )
    analysis_prompt_template = (
        "Analyze the following transaction research output.\n"
        "Extract the key performance implications, business risks, and operational insights.\n"
        "Be specific and grounded only in the provided content.\n\n"
        "Research Output:\n{research_output}"
    )
    summary_prompt_template = (
        "Create a short governance-style summary for the transaction below.\n"
        "Explain:\n"
        "1. What is happening\n"
        "2. What looks healthy or risky\n"
        "3. What should be checked next\n\n"
        "Analysis Output:\n{analysis_output}"
    )
    return research_prompt, analysis_prompt_template, summary_prompt_template


def _safe_log(level: int, message: str, task_name: str, **kwargs) -> None:
    try:
        from main import log_event

        log_event(level, message, "Batch Pipeline", task_name, **kwargs)
    except Exception:
        return


def process_transaction(txn: dict) -> dict:
    _safe_log(logging.INFO, "transaction_processing_started", "Process Transaction", input_data=txn)

    research_agent, analysis_agent, summary_agent = _load_existing_agents()
    research_prompt, analysis_prompt_template, summary_prompt_template = _build_transaction_prompts(txn)
    research_output = _normalize_agent_output(_call_agent(research_agent, research_prompt))
    analysis_output = _normalize_agent_output(
        _call_agent(
            analysis_agent,
            analysis_prompt_template.format(research_output=research_output),
        )
    )
    summary_output = _normalize_agent_output(
        _call_agent(
            summary_agent,
            summary_prompt_template.format(analysis_output=analysis_output),
        )
    )

    processed = {
        "input": txn,
        "research_output": research_output,
        "analysis_output": analysis_output,
        "summary_output": summary_output,
    }
    _safe_log(
        logging.INFO,
        "transaction_processing_completed",
        "Process Transaction",
        output_data=processed,
    )
    return processed


def run_pipeline(sheet_url, db_conn):
    create_transactions_table(db_conn)
    transactions = fetch_sheet_data(sheet_url)

    processed_count = 0
    reused_count = 0
    failed_transactions: list[dict[str, Any]] = []

    for txn in transactions:
        try:
            existing = find_existing_transaction(db_conn, txn)
            if existing and all(
                existing.get(key) not in (None, "", {})
                for key in ("research_output", "analysis_output", "summary_output")
            ):
                reused_count += 1
                _safe_log(
                    logging.INFO,
                    "transaction_reused_from_db",
                    "Run Pipeline",
                    input_data=txn,
                    output_data={"source": "postgres"},
                )
                continue

            processed = process_transaction(txn)
            insert_transaction(db_conn, processed)
            processed_count += 1
        except Exception as exc:
            failed_transactions.append(
                {
                    "transaction": txn,
                    "error": str(exc),
                }
            )
            _safe_log(
                logging.ERROR,
                "transaction_processing_failed",
                "Run Pipeline",
                input_data=txn,
                error=str(exc),
            )

    return {
        "fetched_count": len(transactions),
        "processed_count": processed_count,
        "reused_count": reused_count,
        "failed_count": len(failed_transactions),
        "failed_transactions": failed_transactions,
    }
