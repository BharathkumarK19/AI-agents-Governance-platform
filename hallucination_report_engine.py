import json
import math
import re
import ast
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from hallucination_detector import HallucinationDetector
from verifier import Verifier


RULE_WEIGHTS = {
    "NO_CONTEXT": 3,
    "LOW_CONTEXT_OVERLAP": 4,
    "TOOL_NOT_USED": 3,
    "UNSUPPORTED_CLAIMS": 5,
    "UNCERTAINTY_LANGUAGE": 2,
}
OVERLAP_THRESHOLD = 0.5
CLAIM_SUPPORT_THRESHOLD = 0.4
MAX_VERIFICATION_CLAIMS = 8


class HallucinationReportEngine:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.detector = HallucinationDetector()
        self.verifier = Verifier(progress_callback=self._emit_verification_event)

    def generate_report(
        self,
        query,
        response,
        context,
        tools_used,
        intermediate_steps,
        reference_sources: list[dict] | None = None,
    ) -> dict:
        query_text = self._to_text(query)
        response_text = self._to_text(response)
        context_text = self._to_text(context)
        normalized_tools = self._normalize_tools(tools_used)
        normalized_reference_sources = self._normalize_reference_sources(reference_sources)

        detector_analysis = self.detector.analyze(
            query=query_text,
            response=response_text,
            context=context_text,
            tools_used=normalized_tools,
        )

        context_sentences = self._split_context(context_text)
        claims = self._extract_claims(response_text)
        verified_claims, unsupported_claims = self._verify_claims(
            claims=claims,
            context_sentences=context_sentences,
            query=query_text,
            reference_sources=normalized_reference_sources,
        )
        supported_claims = sum(1 for item in verified_claims if item["supported_by_context"])
        independently_verified_claims = sum(
            1 for item in verified_claims if item["multi_source_verification"]["status"] == "VERIFIED"
        )

        tool_required = self.detector._query_requires_tools(query_text)
        overlap_score = float(detector_analysis["overlap_score"])
        uncertainty_hits = list(detector_analysis["uncertainty_words"])

        rules_triggered = self._rules_triggered(
            context_text=context_text,
            tool_required=tool_required,
            tools_used=normalized_tools,
            overlap_score=overlap_score,
            unsupported_claims=unsupported_claims,
            uncertainty_hits=uncertainty_hits,
        )
        scoring_breakdown = self._scoring_breakdown(rules_triggered)
        label = "HALLUCINATED" if scoring_breakdown["total_score"] <= -6 else "GROUNDED"
        confidence = self._confidence_score(
            overlap_score=overlap_score,
            unsupported_claims=unsupported_claims,
            rules_triggered=rules_triggered,
        )
        justification = self._build_justification(
            label=label,
            overlap_score=overlap_score,
            unsupported_claims=unsupported_claims,
            tool_required=tool_required,
            tools_used=normalized_tools,
            uncertainty_hits=uncertainty_hits,
        )

        report = {
            "schema_version": "2.0",
            "query": query_text,
            "response": response_text,
            "report_summary": {
                "response_generation_status": "completed",
                "claims_evaluated": len(claims),
                "supported_claims": supported_claims,
                "independently_verified_claims": independently_verified_claims,
                "unsupported_claims": len(unsupported_claims),
                "overall_outcome": label,
            },
            "generation_trace": {
                "process_description": (
                    "Execution steps were recorded first. The final response was then audited "
                    "using deterministic claim extraction, context matching, overlap scoring, "
                    "tool-usage checks, and uncertainty checks."
                ),
                "response_construction_basis": (
                    "retrieved context only" if context_text.strip() else "model output without retrieved context"
                ),
                "steps": self._build_generation_trace(intermediate_steps),
            },
            "evidence_analysis": {
                "context_provided": bool(context_text.strip()),
                "context_documents_count": len(context_sentences),
                "context_summary": self._summarize_context(context_sentences),
                "verification_method": {
                    "claim_extraction": "Response split into sentence-level and clause-level factual claims.",
                    "support_check": "Each claim compared against retrieved context snippets using cosine-style token overlap.",
                    "support_threshold": CLAIM_SUPPORT_THRESHOLD,
                    "multi_source_verification": (
                        "Each claim is checked against retrieved tool results first, then against Wikipedia, DuckDuckGo, and NewsAPI when realtime validation is relevant."
                    ),
                },
                "claims_in_response": verified_claims,
                "verified_sources": self._collect_verified_sources(verified_claims),
                "unsupported_claims": unsupported_claims,
                "missing_evidence_summary": self._missing_evidence_summary(
                    unsupported_claims=unsupported_claims,
                    context_text=context_text,
                ),
            },
            "validation_checks": {
                "tool_usage_check": {
                    "required": tool_required,
                    "used": bool(normalized_tools),
                    "tools_observed": normalized_tools,
                    "status": "PASSED" if (not tool_required or normalized_tools) else "FAILED",
                    "explanation": self._tool_usage_explanation(tool_required, normalized_tools),
                },
                "context_overlap_check": {
                    "score": round(overlap_score, 4),
                    "threshold": OVERLAP_THRESHOLD,
                    "status": "PASSED" if overlap_score >= OVERLAP_THRESHOLD else "FAILED",
                    "explanation": self._overlap_explanation(overlap_score),
                },
                "uncertainty_check": {
                    "uncertain_phrases_found": uncertainty_hits,
                    "status": "WARNING" if uncertainty_hits else "PASSED",
                    "explanation": self._uncertainty_explanation(uncertainty_hits),
                },
            },
            "decision_engine": {
                "rules_triggered": rules_triggered,
                "rule_evaluations": self._rule_evaluations(
                    rules_triggered=rules_triggered,
                    overlap_score=overlap_score,
                    tool_required=tool_required,
                    tools_used=normalized_tools,
                    unsupported_claims=unsupported_claims,
                    uncertainty_hits=uncertainty_hits,
                    context_text=context_text,
                ),
                "score_inputs": self._score_inputs(
                    context_text=context_text,
                    tool_required=tool_required,
                    tools_used=normalized_tools,
                    overlap_score=overlap_score,
                    unsupported_claims=unsupported_claims,
                    uncertainty_hits=uncertainty_hits,
                    total_claims=len(claims),
                    supported_claims=supported_claims,
                    independently_verified_claims=independently_verified_claims,
                ),
                "scoring_breakdown": scoring_breakdown,
                "decision_trace": self._decision_trace(
                    claims=claims,
                    supported_claims=supported_claims,
                    overlap_score=overlap_score,
                    rules_triggered=rules_triggered,
                    label=label,
                ),
            },
            "final_verdict": {
                "label": label,
                "confidence": confidence,
                "justification": justification,
            },
            "root_cause_analysis": {
                "primary_cause": self._primary_cause(rules_triggered, detector_analysis),
                "secondary_cause": self._secondary_cause(rules_triggered, detector_analysis),
                "explanation": self._root_cause_explanation(
                    rules_triggered=rules_triggered,
                    overlap_score=overlap_score,
                    unsupported_claims=unsupported_claims,
                ),
            },
            "prevention_recommendation": self._prevention_recommendations(rules_triggered),
        }
        return report

    def save_report(self, report: dict) -> dict:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        report_reference_id = str(uuid4())
        report_path = self.reports_dir / f"{timestamp}_report.json"
        latest_report_path = self.reports_dir / "latest_report.json"
        payload = dict(report)
        payload["report_reference_id"] = report_reference_id
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        serialized = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        report_path.write_text(serialized, encoding="utf-8")
        latest_report_path.write_text(serialized, encoding="utf-8")
        return {
            "report_reference_id": report_reference_id,
            "report_file_path": str(report_path),
            "latest_report_file_path": str(latest_report_path),
            "report": payload,
        }

    def _build_generation_trace(self, intermediate_steps: Any) -> list[dict]:
        if not intermediate_steps:
            return [
                {
                    "step": 1,
                    "agent": "LLM",
                    "action": "Generated response",
                    "output_summary": "Response generated without intermediate trace details.",
                    "source_used": False,
                    "basis": "prior knowledge or hidden runtime context",
                }
            ]

        normalized_steps = []
        steps = intermediate_steps if isinstance(intermediate_steps, list) else [intermediate_steps]
        for index, step in enumerate(steps, start=1):
            if isinstance(step, dict):
                normalized_steps.append(
                    {
                        "step": index,
                        "agent": step.get("agent", "Unknown Agent"),
                        "action": step.get("action", "Executed step"),
                        "output_summary": step.get("output_summary") or step.get("output") or "",
                        "source_used": bool(step.get("source_used", False)),
                        "basis": step.get("basis", "context + prior knowledge"),
                    }
                )
            else:
                normalized_steps.append(
                    {
                        "step": index,
                        "agent": "Unknown Agent",
                        "action": "Executed step",
                        "output_summary": self._to_text(step),
                        "source_used": False,
                        "basis": "context + prior knowledge",
                    }
                )
        return normalized_steps

    def _extract_claims(self, response: str) -> list[str]:
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", response) if segment.strip()]
        claims = []
        for sentence in sentences:
            if not self._is_verifiable_claim(sentence):
                continue
            clauses = [sentence]
            if " and " in sentence.lower() and len(sentence.split()) > 16:
                clauses = re.split(r"\band\b", sentence, flags=re.IGNORECASE)
            for clause in clauses:
                cleaned = clause.strip(" -;:,")
                if self._is_verifiable_claim(cleaned):
                    claims.append(cleaned)
                if len(claims) >= MAX_VERIFICATION_CLAIMS:
                    return claims
        return claims

    def _collect_verified_sources(self, verified_claims: list[dict]) -> list[dict]:
        collected = []
        seen = set()
        for item in verified_claims:
            for source_name, source_result in item.get("multi_source_verification", {}).get("sources", {}).items():
                source_url = source_result.get("source")
                if not source_url or source_url in seen:
                    continue
                top_sources = source_result.get("top_sources") or []
                if top_sources:
                    for top_source in top_sources:
                        url = top_source.get("url")
                        if url and url not in seen:
                            collected.append(
                                {
                                    "source_name": source_name,
                                    "title": top_source.get("title", ""),
                                    "url": url,
                                }
                            )
                            seen.add(url)
                else:
                    collected.append(
                        {
                            "source_name": source_name,
                            "title": "",
                            "url": source_url,
                        }
                    )
                    seen.add(source_url)
                if len(collected) >= 10:
                    return collected
        return collected

    def _is_verifiable_claim(self, text: str) -> bool:
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

    def _verify_claims(
        self,
        claims: list[str],
        context_sentences: list[str],
        query: str,
        reference_sources: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        verified_claims = []
        unsupported_claims = []
        for claim in claims:
            best_match = ""
            best_score = 0.0
            matched_terms = []
            external_verification = self.verifier.verify_claim(
                claim,
                query=query,
                reference_sources=reference_sources,
            )
            for context_sentence in context_sentences:
                score = self._similarity(claim, context_sentence)
                if score > best_score:
                    best_score = score
                    best_match = context_sentence
                    matched_terms = self._matched_terms(claim, context_sentence)

            supported = best_score >= CLAIM_SUPPORT_THRESHOLD
            verified_claims.append(
                {
                    "claim": claim,
                    "supported_by_context": supported,
                    "supporting_text": best_match,
                    "confidence": round(best_score, 2),
                    "verification_status": "SUPPORTED" if supported else "UNSUPPORTED",
                    "matched_terms": matched_terms,
                    "support_reason": (
                        f"Similarity {best_score:.2f} {'met' if supported else 'stayed below'} "
                        f"the {CLAIM_SUPPORT_THRESHOLD:.2f} threshold."
                    ),
                    "multi_source_verification": {
                        "status": external_verification["status"],
                        "support_count": external_verification["support_count"],
                        "sources_checked": ["Serper"],
                        "sources": external_verification["sources"],
                    },
                }
            )
            if not supported or external_verification["status"] == "UNSUPPORTED":
                unsupported_claims.append(
                    {
                        "claim": claim,
                        "reason": self._unsupported_reason(
                            context_supported=supported,
                            external_status=external_verification["status"],
                        ),
                        "best_available_context": best_match,
                        "observed_similarity": round(best_score, 2),
                        "multi_source_status": external_verification["status"],
                        "multi_source_support_count": external_verification["support_count"],
                    }
                )
        return verified_claims, unsupported_claims

    def _rules_triggered(
        self,
        context_text: str,
        tool_required: bool,
        tools_used: list[str],
        overlap_score: float,
        unsupported_claims: list[dict],
        uncertainty_hits: list[str],
    ) -> list[str]:
        rules = []
        if not context_text.strip():
            rules.append("NO_CONTEXT")
        if overlap_score < OVERLAP_THRESHOLD:
            rules.append("LOW_CONTEXT_OVERLAP")
        if tool_required and not tools_used:
            rules.append("TOOL_NOT_USED")
        if unsupported_claims:
            rules.append("UNSUPPORTED_CLAIMS")
        if uncertainty_hits:
            rules.append("UNCERTAINTY_LANGUAGE")
        return rules

    def _scoring_breakdown(self, rules_triggered: list[str]) -> dict:
        context_score = -RULE_WEIGHTS["LOW_CONTEXT_OVERLAP"] if "LOW_CONTEXT_OVERLAP" in rules_triggered else 0
        if "NO_CONTEXT" in rules_triggered:
            context_score -= RULE_WEIGHTS["NO_CONTEXT"]
        tool_penalty = -RULE_WEIGHTS["TOOL_NOT_USED"] if "TOOL_NOT_USED" in rules_triggered else 0
        uncertainty_penalty = -RULE_WEIGHTS["UNCERTAINTY_LANGUAGE"] if "UNCERTAINTY_LANGUAGE" in rules_triggered else 0
        unsupported_penalty = -RULE_WEIGHTS["UNSUPPORTED_CLAIMS"] if "UNSUPPORTED_CLAIMS" in rules_triggered else 0
        total_score = context_score + tool_penalty + uncertainty_penalty + unsupported_penalty
        return {
            "context_score": context_score,
            "tool_penalty": tool_penalty,
            "uncertainty_penalty": uncertainty_penalty,
            "unsupported_claims_penalty": unsupported_penalty,
            "total_score": total_score,
            "decision_threshold": -6,
        }

    def _confidence_score(self, overlap_score: float, unsupported_claims: list[dict], rules_triggered: list[str]) -> float:
        base = 0.55 + min(0.35, overlap_score * 0.35)
        penalty = min(0.35, len(unsupported_claims) * 0.08 + len(rules_triggered) * 0.05)
        confidence = base + penalty if rules_triggered else base
        return round(max(0.05, min(0.99, confidence)), 2)

    def _build_justification(
        self,
        label: str,
        overlap_score: float,
        unsupported_claims: list[dict],
        tool_required: bool,
        tools_used: list[str],
        uncertainty_hits: list[str],
    ) -> str:
        reasons = []
        if overlap_score < OVERLAP_THRESHOLD:
            reasons.append(f"context overlap is {overlap_score:.2f}, below the 0.50 threshold")
        if unsupported_claims:
            reasons.append(f"{len(unsupported_claims)} claims were not supported by retrieved context")
        if tool_required and not tools_used:
            reasons.append("a tool was required but not used")
        if uncertainty_hits:
            reasons.append(f"uncertainty language was found: {', '.join(uncertainty_hits)}")

        if label == "HALLUCINATED":
            return "Response is classified as hallucinated because " + "; ".join(reasons) + "."
        if reasons:
            return "Response is grounded overall, although minor warnings were noted: " + "; ".join(reasons) + "."
        return "Response is grounded because claims align with retrieved context and no failure rules were triggered."

    def _primary_cause(self, rules_triggered: list[str], detector_analysis: dict) -> str:
        if "TOOL_NOT_USED" in rules_triggered:
            return "Missing retrieval step"
        if "UNSUPPORTED_CLAIMS" in rules_triggered:
            return "Claims exceeded retrieved evidence"
        if "LOW_CONTEXT_OVERLAP" in rules_triggered or "NO_CONTEXT" in rules_triggered:
            return "Weak grounding context"
        return "Grounded generation path"

    def _secondary_cause(self, rules_triggered: list[str], detector_analysis: dict) -> str:
        if "UNCERTAINTY_LANGUAGE" in rules_triggered:
            return "Model relied on hedged language instead of evidence-backed statements"
        if detector_analysis.get("drift_detected"):
            return "Model relied on prior knowledge instead of data"
        return "No secondary failure cause detected"

    def _prevention_recommendations(self, rules_triggered: list[str]) -> list[str]:
        recommendations = []
        if "TOOL_NOT_USED" in rules_triggered:
            recommendations.append("Force tool usage for factual queries")
        if "LOW_CONTEXT_OVERLAP" in rules_triggered or "NO_CONTEXT" in rules_triggered:
            recommendations.append("Reject responses with low context overlap")
        if "UNSUPPORTED_CLAIMS" in rules_triggered:
            recommendations.append("Add claim verification layer")
        if "UNCERTAINTY_LANGUAGE" in rules_triggered:
            recommendations.append("Flag hedged factual statements for review")
        if not recommendations:
            recommendations.append("Keep the current retrieval and verification workflow")
        return recommendations

    def _score_inputs(
        self,
        context_text: str,
        tool_required: bool,
        tools_used: list[str],
        overlap_score: float,
        unsupported_claims: list[dict],
        uncertainty_hits: list[str],
        total_claims: int,
        supported_claims: int,
        independently_verified_claims: int,
    ) -> dict:
        return {
            "context_present": bool(context_text.strip()),
            "tool_required": tool_required,
            "tool_used": bool(tools_used),
            "context_overlap_score": round(overlap_score, 4),
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "independently_verified_claims": independently_verified_claims,
            "unsupported_claims": len(unsupported_claims),
            "uncertainty_terms_found": uncertainty_hits,
        }

    def _rule_evaluations(
        self,
        rules_triggered: list[str],
        overlap_score: float,
        tool_required: bool,
        tools_used: list[str],
        unsupported_claims: list[dict],
        uncertainty_hits: list[str],
        context_text: str,
    ) -> list[dict]:
        catalog = [
            (
                "NO_CONTEXT",
                not bool(context_text.strip()),
                RULE_WEIGHTS["NO_CONTEXT"],
                "No retrieved context was available during verification.",
            ),
            (
                "LOW_CONTEXT_OVERLAP",
                overlap_score < OVERLAP_THRESHOLD,
                RULE_WEIGHTS["LOW_CONTEXT_OVERLAP"],
                f"Overlap score {overlap_score:.2f} is below threshold {OVERLAP_THRESHOLD:.2f}.",
            ),
            (
                "TOOL_NOT_USED",
                tool_required and not tools_used,
                RULE_WEIGHTS["TOOL_NOT_USED"],
                "The query appears time-sensitive or factual, but no retrieval tool was recorded.",
            ),
            (
                "UNSUPPORTED_CLAIMS",
                bool(unsupported_claims),
                RULE_WEIGHTS["UNSUPPORTED_CLAIMS"],
                f"{len(unsupported_claims)} claims could not be mapped back to retrieved context.",
            ),
            (
                "UNCERTAINTY_LANGUAGE",
                bool(uncertainty_hits),
                RULE_WEIGHTS["UNCERTAINTY_LANGUAGE"],
                f"Uncertainty markers found: {', '.join(uncertainty_hits)}." if uncertainty_hits else "No uncertainty language found.",
            ),
        ]
        evaluations = []
        for rule_name, triggered, impact, evidence in catalog:
            evaluations.append(
                {
                    "rule": rule_name,
                    "triggered": triggered,
                    "impact_score": -impact if triggered else 0,
                    "evidence": evidence,
                }
            )
        return evaluations

    def _decision_trace(
        self,
        claims: list[str],
        supported_claims: int,
        overlap_score: float,
        rules_triggered: list[str],
        label: str,
    ) -> list[str]:
        trace = [
            f"Extracted {len(claims)} atomic claims from the response.",
            f"Verified {supported_claims} claims directly against retrieved context.",
            f"Measured overall context overlap at {overlap_score:.2f}.",
        ]
        if rules_triggered:
            trace.append("Triggered rules: " + ", ".join(rules_triggered) + ".")
        else:
            trace.append("No risk rules were triggered.")
        trace.append(f"Final classification set to {label}.")
        return trace

    def _missing_evidence_summary(self, unsupported_claims: list[dict], context_text: str) -> str:
        if not context_text.strip():
            return "No evidence was available, so every response claim lacked a retrievable support base."
        if not unsupported_claims:
            return "No major evidence gaps were found."
        return (
            f"{len(unsupported_claims)} claims lacked sufficient textual support in the retrieved context. "
            "Those claims should be backed by stronger retrieval or removed."
        )

    def _tool_usage_explanation(self, tool_required: bool, tools_used: list[str]) -> str:
        if tool_required and tools_used:
            return "Tool usage matched the query risk profile, so factual grounding was attempted."
        if tool_required and not tools_used:
            return "This check failed because the query pattern suggested live or factual validation but no tool call was recorded."
        if tools_used:
            return "Tools were used even though the query did not strictly demand them."
        return "Tool usage was not mandatory for this query pattern."

    def _overlap_explanation(self, overlap_score: float) -> str:
        if overlap_score >= OVERLAP_THRESHOLD:
            return "The final response remained sufficiently close to the retrieved evidence."
        return "The final response diverged materially from the retrieved evidence."

    def _uncertainty_explanation(self, uncertainty_hits: list[str]) -> str:
        if uncertainty_hits:
            return "Hedging language was detected, which can signal weak evidence or model uncertainty."
        return "No hedging language was detected."

    def _root_cause_explanation(self, rules_triggered: list[str], overlap_score: float, unsupported_claims: list[dict]) -> str:
        if "TOOL_NOT_USED" in rules_triggered:
            return "The answer path lacked external retrieval at the point where live evidence was needed."
        if "UNSUPPORTED_CLAIMS" in rules_triggered:
            return f"The model produced {len(unsupported_claims)} claims that could not be substantiated by the available context."
        if "LOW_CONTEXT_OVERLAP" in rules_triggered:
            return f"The response-context similarity score of {overlap_score:.2f} was too low for a grounded answer."
        return "The response stayed within the available evidence envelope."

    def _summarize_context(self, context_sentences: list[str]) -> str:
        if not context_sentences:
            return "No context was provided."
        return " ".join(context_sentences[:3])

    def _split_context(self, context_text: str) -> list[str]:
        return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", context_text) if segment.strip()]

    def _similarity(self, left: str, right: str) -> float:
        left_tokens = self._tokenize(left)
        right_tokens = self._tokenize(right)
        if not left_tokens or not right_tokens:
            return 0.0
        left_counts = Counter(left_tokens)
        right_counts = Counter(right_tokens)
        dot_product = sum(left_counts[token] * right_counts[token] for token in left_counts)
        left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
        right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot_product / (left_norm * right_norm)

    def _matched_terms(self, claim: str, context_sentence: str) -> list[str]:
        claim_tokens = set(self._tokenize(claim))
        context_tokens = set(self._tokenize(context_sentence))
        return sorted(claim_tokens & context_tokens)[:10]

    def _unsupported_reason(self, context_supported: bool, external_status: str) -> str:
        if not context_supported and external_status == "UNSUPPORTED":
            return (
                "Claim failed both context grounding and independent multi-source verification."
            )
        if not context_supported:
            return "Claim was not supported by retrieved context even though external support may exist."
        return "Claim matched local context weakly but did not receive enough independent external support."

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _normalize_tools(self, tools_used: Any) -> list[str]:
        if tools_used is None:
            return []
        if isinstance(tools_used, str):
            return [tools_used]
        if isinstance(tools_used, list):
            return [self._to_text(item) for item in tools_used if self._to_text(item)]
        return [self._to_text(tools_used)]

    def _normalize_reference_sources(self, reference_sources: Any) -> list[dict]:
        if not reference_sources:
            return []
        if isinstance(reference_sources, list):
            return [item for item in reference_sources if isinstance(item, dict)]
        return []

    def _emit_verification_event(self, event: str, payload: dict) -> None:
        if event == "claim_started":
            print(
                f"[report-verification] claim: {str(payload.get('claim', ''))[:140]}"
            )
            return
        if event == "source_checked":
            result = payload.get("result", {})
            overlap = result.get("overlap")
            overlap_text = f", overlap={overlap}" if overlap is not None else ""
            print(
                f"[report-verification] source={payload.get('source_name')} "
                f"supported={result.get('supported', False)}{overlap_text}"
            )
            return
        if event == "claim_completed":
            print(
                f"[report-verification] status={payload.get('status')} "
                f"support_count={payload.get('support_count')}"
            )

    def _to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(self._to_text(item) for item in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, default=str)
        return str(value)
