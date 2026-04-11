import json
import os
import re
import time
from datetime import datetime, timezone
from statistics import mean
from uuid import uuid4

import requests

from agents.llm import get_llm
from agents.tools import build_source_bundle, search_tavily
from db import connect_db
from logger import ObservabilityLogger
from sheet_ingestion import fetch_sheet_data
from verifier import Verifier


MAX_ATTEMPTS = 3
MAX_VERIFICATION_CLAIMS = 8
DEFAULT_MODEL_SEQUENCE = [
    "openrouter/auto",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
]


def _print_live_verification_event(event: str, payload: dict) -> None:
    if event == "claim_started":
        print(
            f"[self-healing verification] checking claim: "
            f"{str(payload.get('claim', ''))[:140]}"
        )
        return
    if event == "source_checked":
        result = payload.get("result", {})
        overlap = result.get("overlap")
        overlap_text = f", overlap={overlap}" if overlap is not None else ""
        print(
            f"[self-healing verification] source={payload.get('source_name')} "
            f"supported={result.get('supported', False)}{overlap_text}"
        )
        return
    if event == "claim_completed":
        print(
            f"[self-healing verification] status={payload.get('status')} "
            f"support_count={payload.get('support_count')}"
        )


def _load_model_sequence() -> list[str]:
    raw = os.getenv("SELF_HEALING_MODEL_SEQUENCE", "").strip()
    if raw:
        parsed = [item.strip() for item in raw.split(",") if item.strip()]
        if parsed:
            return parsed
    return DEFAULT_MODEL_SEQUENCE[:]


def generate_response(query, model_name, data_source):
    retrieval = retrieve_context(query, data_source)
    llm = get_llm(model=model_name)
    prompt = (
        "Answer the user query using only the retrieved evidence.\n"
        "If evidence is insufficient, say so explicitly.\n\n"
        f"Query: {query}\n\n"
        f"Retrieved Evidence:\n{retrieval['context']}\n"
    )
    response = llm.invoke(prompt)
    return {
        "response": getattr(response, "content", str(response)),
        "retrieved_context": retrieval["context"],
        "tools_used": retrieval["tools_used"],
        "raw_sources": retrieval["raw_sources"],
        "model_used": model_name,
    }


def extract_claims(response: str) -> list:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", response) if segment.strip()]
    claims = []
    for sentence in sentences:
        if not _is_verifiable_claim(sentence):
            continue
        clauses = re.split(r"\band\b", sentence, flags=re.IGNORECASE) if len(sentence.split()) > 16 else [sentence]
        for clause in clauses:
            cleaned = clause.strip(" -;:,")
            if _is_verifiable_claim(cleaned):
                claims.append(cleaned)
            if len(claims) >= MAX_VERIFICATION_CLAIMS:
                return claims
    return claims


def _is_verifiable_claim(text: str) -> bool:
    cleaned = text.strip(" -;:,*#|`")
    lowered = cleaned.lower()
    if len(cleaned.split()) < 6:
        return False
    if len(cleaned) > 220:
        return False
    skip_prefixes = (
        "report:",
        "prepared for:",
        "date:",
        "executive summary",
        "introduction",
        "conclusion",
        "recommendations",
        "references",
        "table",
    )
    if any(lowered.startswith(prefix) for prefix in skip_prefixes):
        return False
    if cleaned.startswith(("*", "#", "|", "-")):
        return False
    return True


def retrieve_context(query: str, data_source: str) -> dict:
    if data_source == "tavily":
        return _package_source_results(search_tavily(query), ["tavily"])
    if data_source == "serper":
        return _package_source_results(search_serper(query), ["serper"])
    if data_source == "hybrid":
        return _retrieve_hybrid_context(query)
    raise ValueError(f"Unsupported data source: {data_source}")


def _package_source_results(results: list[dict], tools_used: list[str]) -> dict:
    return {
        "context": build_source_bundle(results),
        "tools_used": tools_used,
        "raw_sources": results,
    }


def _retrieve_hybrid_context(query: str) -> dict:
    raw_sources: list[dict] = []
    tools_used: list[str] = []

    try:
        tavily_results = search_tavily(query)
        raw_sources.extend(_normalize_sources(tavily_results, default_source="tavily"))
        tools_used.append("tavily")
    except Exception:
        tavily_results = []

    try:
        serper_results = search_serper(query)
        raw_sources.extend(_normalize_sources(serper_results, default_source="serper"))
        tools_used.append("serper")
    except Exception:
        serper_results = []

    if _query_needs_internal_data(query):
        sheet_url = os.getenv("GOOGLE_SHEET_URL", "").strip()
        db_url = os.getenv("NEON_DATABASE_URL", "").strip()

        if sheet_url:
            try:
                rows = fetch_sheet_data(sheet_url)
                raw_sources.extend(_normalize_internal_rows(rows[:10], "google_sheets"))
                tools_used.append("google_sheets")
            except Exception:
                pass

        if db_url:
            try:
                conn = connect_db(db_url)
                try:
                    rows = _fetch_recent_transactions(conn, limit=10)
                finally:
                    if getattr(conn, "closed", 1) == 0:
                        conn.close()
                raw_sources.extend(_normalize_internal_rows(rows, "neon_postgresql"))
                tools_used.append("neon_postgresql")
            except Exception:
                pass

    return _package_source_results(raw_sources, _unique(tools_used))


def search_serper(query: str) -> list[dict]:
    api_key = os.getenv("SERPER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("SERPER_API_KEY is not set.")
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": 5},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    return [
        {
            "title": item.get("title", "Untitled result"),
            "url": item.get("link", ""),
            "content": item.get("snippet", ""),
        }
        for item in payload.get("organic", [])
    ]


def _query_needs_internal_data(query: str) -> bool:
    lowered = query.lower()
    keywords = (
        "customer",
        "customers",
        "transaction",
        "transactions",
        "sales",
        "revenue",
        "discount",
        "margin",
        "profit",
        "gst",
        "item",
        "purchase",
        "sheet",
        "database",
        "postgres",
        "neon",
    )
    return any(keyword in lowered for keyword in keywords)


def _fetch_recent_transactions(conn, limit: int = 10) -> list[dict]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT customer_id, item_name, item_id, price, discount, gst, total_price, profit, margin, timestamp
            FROM transactions
            ORDER BY timestamp DESC NULLS LAST, id DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cursor.fetchall()

    return [
        {
            "customer_id": row[0],
            "item_name": row[1],
            "item_id": row[2],
            "price": row[3],
            "discount": row[4],
            "gst": row[5],
            "total_price": row[6],
            "profit": row[7],
            "margin": row[8],
            "timestamp": row[9],
        }
        for row in rows
    ]


def _normalize_sources(results: list[dict], default_source: str) -> list[dict]:
    normalized = []
    for item in results:
        normalized.append(
            {
                "title": item.get("title", "Untitled result"),
                "url": item.get("url", ""),
                "content": item.get("content", "") or item.get("snippet", ""),
                "source": item.get("source", default_source),
            }
        )
    return normalized


def _normalize_internal_rows(rows: list[dict], source_name: str) -> list[dict]:
    normalized = []
    for index, row in enumerate(rows, start=1):
        normalized.append(
            {
                "title": f"{source_name} record {index}",
                "url": source_name,
                "content": json.dumps(row, ensure_ascii=False, default=str),
                "source": source_name,
            }
        )
    return normalized


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _summarize_sources(raw_sources: list[dict], limit: int = 5) -> list[dict]:
    preview = []
    for item in raw_sources[:limit]:
        preview.append(
            {
                "title": item.get("title", "Untitled result"),
                "url": item.get("url") or item.get("source") or "",
                "source": item.get("source", ""),
            }
        )
    return preview


def _build_comparison_summary(verification_results: list[dict]) -> dict:
    source_totals: dict[str, int] = {}
    for item in verification_results:
        for source_name, result in item.get("sources", {}).items():
            if result.get("supported"):
                source_totals[source_name] = source_totals.get(source_name, 0) + 1
    return {
        "supported_claims_by_source": source_totals,
        "verified_claims": sum(1 for item in verification_results if item["status"] == "VERIFIED"),
        "weakly_supported_claims": sum(
            1 for item in verification_results if item["status"] == "WEAKLY_SUPPORTED"
        ),
        "unsupported_claims": sum(1 for item in verification_results if item["status"] == "UNSUPPORTED"),
    }


def _build_stepwise_evolution_entry(
    attempt_number: int,
    query: str,
    generation: dict,
    verification_results: list[dict],
    scores: dict,
    decision: str,
    failure_reason: str | None,
    latency_ms: float,
    data_source: str,
) -> dict:
    return {
        "step": attempt_number,
        "query": query,
        "model_used": generation["model_used"],
        "data_source_used": data_source,
        "tools_used": generation["tools_used"],
        "tool_activity": [
            {
                "tool_name": tool_name,
                "status": "used",
            }
            for tool_name in generation["tools_used"]
        ],
        "retrieval": {
            "source_count": len(generation["raw_sources"]),
            "sources_preview": _summarize_sources(generation["raw_sources"]),
        },
        "generation": {
            "response_preview": generation["response"][:500],
            "claim_count": len(verification_results),
        },
        "comparison": {
            "summary": _build_comparison_summary(verification_results),
            "claims": verification_results,
        },
        "scores": scores,
        "confidence": {
            "confidence_score": scores["confidence_score"],
            "verification_score": scores["verification_score"],
            "consistency_score": scores["consistency_score"],
            "risk_score": scores["risk_score"],
        },
        "decision": decision,
        "failure_reason": failure_reason,
        "latency_ms": latency_ms,
    }


def compute_scores(verification_results: list[dict], responses: list[str] | None = None) -> dict:
    total_claims = len(verification_results)
    verified_claims = sum(1 for item in verification_results if item["status"] == "VERIFIED")
    weakly_supported_claims = sum(1 for item in verification_results if item["status"] == "WEAKLY_SUPPORTED")
    unsupported_claims = sum(1 for item in verification_results if item["status"] == "UNSUPPORTED")
    verification_score = verified_claims / total_claims if total_claims else 0.0
    confidence_score = (
        (verified_claims + 0.5 * weakly_supported_claims) / total_claims if total_claims else 0.0
    )
    consistency_score = _consistency_score(responses or [])

    risk_score = 0
    risk_score += unsupported_claims * 2
    if verification_score < 0.5:
        risk_score += 3
    if confidence_score < 0.6:
        risk_score += 2

    label = "HALLUCINATED" if risk_score >= 5 else "GROUNDED"
    return {
        "total_claims": total_claims,
        "verified_claims": verified_claims,
        "weakly_supported_claims": weakly_supported_claims,
        "unsupported_claims": unsupported_claims,
        "verification_score": round(verification_score, 4),
        "confidence_score": round(confidence_score, 4),
        "consistency_score": round(consistency_score, 4),
        "risk_score": risk_score,
        "label": label,
    }


class SelfHealingPipeline:
    def __init__(self, neon_db_url: str | None = None, base_log_dir: str = "observability_logs"):
        self.verifier = Verifier(progress_callback=_print_live_verification_event)
        self.logger = ObservabilityLogger(
            neon_db_url=neon_db_url,
            local_path=os.path.join(base_log_dir, "logs", "observability_logs.json"),
        )

    def run(self, query: str) -> dict:
        report_id = str(uuid4())
        decision_trace = []
        attempt_logs = []
        evolution_steps = []
        generated_responses = []
        final_payload = None
        started_at = time.perf_counter()
        model_sequence = _load_model_sequence()
        attempt_plan = [
            {"data_source": "tavily", "model": model_sequence[0]},
            {"data_source": "hybrid", "model": model_sequence[min(1, len(model_sequence) - 1)]},
            {"data_source": "serper", "model": model_sequence[min(2, len(model_sequence) - 1)]},
        ]

        for attempt in range(min(MAX_ATTEMPTS, len(attempt_plan))):
            current_plan = attempt_plan[attempt]
            data_source = current_plan["data_source"]
            model_name = current_plan["model"]
            attempt_started = time.perf_counter()
            generation = generate_response(query, model_name, data_source)
            response = generation["response"]
            claims = extract_claims(response)
            verification_results = self.verifier.verify_all(
                claims,
                query=query,
                reference_sources=generation["raw_sources"],
            )
            generated_responses.append(response)
            scores = compute_scores(verification_results, responses=generated_responses[-3:])

            failure_reason = None
            if scores["unsupported_claims"] > 0:
                failure_reason = "FACT_MISMATCH"
            elif scores["verification_score"] < 0.5:
                failure_reason = "LOW_VERIFICATION_SCORE"

            attempt_log = {
                "id": report_id,
                "attempt": attempt + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": query,
                "data_source_used": data_source,
                "model_used": model_name,
                "response": response,
                "claims": verification_results,
                "scores": scores,
                "failure_reason": failure_reason,
                "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 2),
                "tools_used": generation["tools_used"],
                "retrieval_source_count": len(generation["raw_sources"]),
                "retrieval_sources_preview": _summarize_sources(generation["raw_sources"]),
                "comparison_summary": _build_comparison_summary(verification_results),
            }
            attempt_logs.append(attempt_log)
            self.logger.save_attempt_log(attempt_log, report_id=report_id, attempt_number=attempt + 1)

            if scores["label"] == "GROUNDED":
                decision = f"Attempt {attempt + 1} accepted with {data_source} and {model_name}."
                decision_trace.append(decision)
                evolution_steps.append(
                    _build_stepwise_evolution_entry(
                        attempt_number=attempt + 1,
                        query=query,
                        generation=generation,
                        verification_results=verification_results,
                        scores=scores,
                        decision=decision,
                        failure_reason=failure_reason,
                        latency_ms=attempt_log["latency_ms"],
                        data_source=data_source,
                    )
                )
                final_payload = {
                    "response": response,
                    "claims": verification_results,
                    "scores": scores,
                    "data_source_used": data_source,
                    "model_used": model_name,
                    "retrieved_context": generation["retrieved_context"],
                    "tools_used": generation["tools_used"],
                }
                break

            if attempt == 0:
                decision = (
                    f"Attempt {attempt + 1} failed verification; switching to model "
                    f"{attempt_plan[1]['model']} and datasource {attempt_plan[1]['data_source']}."
                )
                decision_trace.append(decision)
            elif attempt == 1:
                decision = (
                    f"Attempt {attempt + 1} failed verification; switching to model "
                    f"{attempt_plan[2]['model']} and datasource {attempt_plan[2]['data_source']}."
                )
                decision_trace.append(decision)
            else:
                decision = "Final attempt failed verification. Returning best effort response."
                decision_trace.append(decision)
                final_payload = {
                    "response": response,
                    "claims": verification_results,
                    "scores": scores,
                    "data_source_used": data_source,
                    "model_used": model_name,
                    "retrieved_context": generation["retrieved_context"],
                    "tools_used": generation["tools_used"],
                }
            evolution_steps.append(
                _build_stepwise_evolution_entry(
                    attempt_number=attempt + 1,
                    query=query,
                    generation=generation,
                    verification_results=verification_results,
                    scores=scores,
                    decision=decision,
                    failure_reason=failure_reason,
                    latency_ms=attempt_log["latency_ms"],
                    data_source=data_source,
                )
            )

        if final_payload is None:
            final_payload = {
                "response": "",
                "claims": [],
                "scores": compute_scores([]),
                "data_source_used": attempt_plan[-1]["data_source"],
                "model_used": attempt_plan[-1]["model"],
                "retrieved_context": "",
                "tools_used": [],
            }

        report = {
            "id": report_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "data_source_used": final_payload["data_source_used"],
            "model_used": final_payload["model_used"],
            "response": final_payload["response"],
            "claims": [
                {
                    "text": item["claim"],
                    "status": item["status"],
                }
                for item in final_payload["claims"]
            ],
            "scores": {
                "verification_score": final_payload["scores"]["verification_score"],
                "confidence_score": final_payload["scores"]["confidence_score"],
                "risk_score": final_payload["scores"]["risk_score"],
            },
            "final_label": final_payload["scores"]["label"],
            "decision_trace": decision_trace,
            "evolution_steps": evolution_steps,
        }
        metrics = {
            "id": report_id,
            "total_claims": final_payload["scores"]["total_claims"],
            "verified": final_payload["scores"]["verified_claims"],
            "unsupported": final_payload["scores"]["unsupported_claims"],
            "confidence_score": final_payload["scores"]["confidence_score"],
            "verification_score": final_payload["scores"]["verification_score"],
            "consistency_score": final_payload["scores"]["consistency_score"],
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }

        report_path = self.logger.save_report(report)
        metrics_path = self.logger.save_metrics(metrics, report_id=report_id)
        self.logger.log(
            query=query,
            response=final_payload["response"],
            context=final_payload["retrieved_context"],
            analysis={
                "label": final_payload["scores"]["label"],
                "overlap_score": final_payload["scores"]["verification_score"],
                "risk_score": final_payload["scores"]["risk_score"],
                "reasoning": {
                    "verification": f"{final_payload['scores']['verified_claims']} of {final_payload['scores']['total_claims']} claims verified.",
                    "decision_trace": " ".join(decision_trace),
                },
                "root_cause": "Self-healing pipeline execution completed.",
                "prevention_suggestion": "Use verification-first acceptance and fallback retrieval as implemented.",
                "tools_used": final_payload["tools_used"],
                "report_reference_id": report_id,
                "report_file_path": report_path,
            },
            tools_used=final_payload["tools_used"],
        )

        return {
            "id": report_id,
            "query": query,
            "response": final_payload["response"],
            "claims": final_payload["claims"],
            "scores": final_payload["scores"],
            "data_source_used": final_payload["data_source_used"],
            "model_used": final_payload["model_used"],
            "decision_trace": decision_trace,
            "evolution_steps": evolution_steps,
            "report_path": report_path,
            "metrics_path": metrics_path,
            "attempts": attempt_logs,
        }

    def close(self) -> None:
        self.logger.close()


def _consistency_score(responses: list[str]) -> float:
    if len(responses) < 2:
        return 1.0 if responses else 0.0
    token_sets = [set(re.findall(r"[a-z0-9]+", response.lower())) for response in responses]
    pair_scores = []
    for index, left in enumerate(token_sets):
        for right in token_sets[index + 1 :]:
            union = left | right
            pair_scores.append(len(left & right) / len(union) if union else 1.0)
    return mean(pair_scores) if pair_scores else 1.0
