import math
import re
from collections import Counter
from typing import Any


UNCERTAINTY_WORDS = ("likely", "probably", "generally", "typically")
REALTIME_KEYWORDS = (
    "latest",
    "today",
    "current",
    "recent",
    "now",
    "price",
    "stock",
    "weather",
    "score",
    "news",
    "president",
    "ceo",
)
STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "is",
    "are",
    "was",
    "were",
    "for",
    "on",
    "with",
    "as",
    "by",
    "at",
    "from",
    "that",
    "this",
    "it",
}


class HallucinationDetector:
    def analyze(self, query, response, context, tools_used) -> dict:
        response_text = self._to_text(response)
        context_text = self._to_text(context)
        normalized_tools = self._normalize_tools(tools_used)

        overlap_score = self._semantic_similarity(response_text, context_text)
        requires_tools = self._query_requires_tools(query)
        tool_used = bool(normalized_tools)
        uncertainty_hits = self._find_uncertainty_words(response_text)
        unsupported_claims = self._find_unsupported_claims(response_text, context_text)
        self_reflection = self._self_reflection(
            response_text=response_text,
            context_text=context_text,
            unsupported_claims=unsupported_claims,
            overlap_score=overlap_score,
        )
        drift_analysis = self._detect_drift(response_text, context_text)

        risk_score = 0
        if not context_text.strip():
            risk_score += 3
        if requires_tools and not tool_used:
            risk_score += 3
        if overlap_score < 0.5:
            risk_score += 4
        if uncertainty_hits:
            risk_score += 2
        if unsupported_claims:
            risk_score += min(4, len(unsupported_claims))
        if drift_analysis["drift_detected"]:
            risk_score += 2

        label = "HALLUCINATED" if risk_score >= 6 else "GROUNDED"
        context_presence = (
            "Retrieved context was provided and used for comparison."
            if context_text.strip()
            else "No retrieved context was available, which weakens grounding."
        )
        tool_usage_reason = self._tool_usage_reasoning(requires_tools, tool_used, normalized_tools)
        overlap_reason = (
            f"Context overlap score is {overlap_score:.2f}. "
            + (
                "This suggests the response stays close to retrieved evidence."
                if overlap_score >= 0.5
                else "This suggests the response introduces content not well supported by context."
            )
        )
        uncertainty_reason = (
            f"Uncertainty markers found: {', '.join(uncertainty_hits)}."
            if uncertainty_hits
            else "No uncertainty markers were found in the response."
        )
        unsupported_reason = (
            "Unsupported claims detected: " + "; ".join(unsupported_claims)
            if unsupported_claims
            else "No clearly unsupported claims were detected by the heuristic claim check."
        )
        drift_reason = drift_analysis["reason"]

        root_cause = self._build_root_cause(
            context_text=context_text,
            requires_tools=requires_tools,
            tool_used=tool_used,
            overlap_score=overlap_score,
            unsupported_claims=unsupported_claims,
            drift_detected=drift_analysis["drift_detected"],
        )
        classification_reason = self._build_case_study_explanation(
            label=label,
            overlap_score=overlap_score,
            unsupported_claims=unsupported_claims,
            uncertainty_hits=uncertainty_hits,
            requires_tools=requires_tools,
            tool_used=tool_used,
            drift_analysis=drift_analysis,
        )
        prevention = self._build_prevention_suggestion(
            requires_tools=requires_tools,
            tool_used=tool_used,
            overlap_score=overlap_score,
            unsupported_claims=unsupported_claims,
            context_text=context_text,
        )

        return {
            "label": label,
            "overlap_score": round(overlap_score, 4),
            "risk_score": risk_score,
            "reasoning": {
                "context_presence": context_presence,
                "tool_usage": tool_usage_reason,
                "overlap_analysis": overlap_reason,
                "uncertainty_analysis": uncertainty_reason,
                "unsupported_claims_analysis": unsupported_reason,
                "self_reflection": self_reflection,
                "drift_detection": drift_reason,
            },
            "root_cause": root_cause,
            "why_this_is_hallucination_or_not": classification_reason,
            "prevention_suggestion": prevention,
            "unsupported_claims": unsupported_claims,
            "uncertainty_words": uncertainty_hits,
            "tools_used": normalized_tools,
            "drift_detected": drift_analysis["drift_detected"],
            "drift_details": drift_analysis["new_facts"],
            "self_reflection": self_reflection,
        }

    def _semantic_similarity(self, response: str, context: str) -> float:
        response_tokens = self._tokenize(response)
        context_tokens = self._tokenize(context)
        if not response_tokens or not context_tokens:
            return 0.0

        response_counts = Counter(response_tokens)
        context_counts = Counter(context_tokens)
        dot_product = sum(response_counts[token] * context_counts[token] for token in response_counts)
        response_norm = math.sqrt(sum(value * value for value in response_counts.values()))
        context_norm = math.sqrt(sum(value * value for value in context_counts.values()))
        if response_norm == 0 or context_norm == 0:
            return 0.0
        return dot_product / (response_norm * context_norm)

    def _query_requires_tools(self, query: str) -> bool:
        lowered = self._to_text(query).lower()
        return any(keyword in lowered for keyword in REALTIME_KEYWORDS)

    def _find_uncertainty_words(self, response: str) -> list[str]:
        lowered = response.lower()
        return [word for word in UNCERTAINTY_WORDS if word in lowered]

    def _find_unsupported_claims(self, response: str, context: str) -> list[str]:
        if not response.strip():
            return []

        context_tokens = set(self._tokenize(context))
        claims = self._split_claims(response)
        unsupported = []
        for claim in claims:
            claim_tokens = [token for token in self._tokenize(claim) if token not in STOP_WORDS]
            if len(claim_tokens) < 4:
                continue
            overlap = sum(1 for token in claim_tokens if token in context_tokens)
            if overlap / max(len(claim_tokens), 1) < 0.35:
                unsupported.append(claim.strip())
        return unsupported[:5]

    def _detect_drift(self, response: str, context: str) -> dict:
        response_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", response))
        context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", context))
        new_numbers = sorted(response_numbers - context_numbers)

        response_entities = {
            match.strip()
            for match in re.findall(r"\b[A-Z][a-zA-Z0-9&.-]*(?:\s+[A-Z][a-zA-Z0-9&.-]*)*\b", response)
        }
        context_entities = {
            match.strip()
            for match in re.findall(r"\b[A-Z][a-zA-Z0-9&.-]*(?:\s+[A-Z][a-zA-Z0-9&.-]*)*\b", context)
        }
        new_entities = sorted(entity for entity in response_entities - context_entities if len(entity) > 3)

        new_facts = (new_numbers + new_entities)[:8]
        drift_detected = bool(new_facts)
        reason = (
            "Drift detected because the response introduced facts or entities not seen in the retrieved context: "
            + ", ".join(new_facts)
            if drift_detected
            else "No significant drift detected between retrieved context and final response."
        )
        return {
            "drift_detected": drift_detected,
            "new_facts": new_facts,
            "reason": reason,
        }

    def _self_reflection(
        self,
        response_text: str,
        context_text: str,
        unsupported_claims: list[str],
        overlap_score: float,
    ) -> str:
        if not response_text.strip():
            return "No response was available to self-check."
        if not context_text.strip():
            return "No. The response cannot be confirmed because no grounding context was provided."
        if unsupported_claims or overlap_score < 0.5:
            return (
                "No. The response is not fully grounded because at least part of it is weakly supported "
                "or absent from the provided context."
            )
        return "Yes. The response appears to stay within the provided context."

    def _tool_usage_reasoning(self, requires_tools: bool, tool_used: bool, tools_used: list[str]) -> str:
        if requires_tools and tool_used:
            return f"Tool usage was appropriate for a fact-sensitive query. Tools used: {', '.join(tools_used)}."
        if requires_tools and not tool_used:
            return "The query appears fact-sensitive or time-sensitive, but no retrieval tool was used."
        if tool_used:
            return f"Tools were used even though they were not strictly required: {', '.join(tools_used)}."
        return "No tool usage was required based on the query pattern."

    def _build_root_cause(
        self,
        context_text: str,
        requires_tools: bool,
        tool_used: bool,
        overlap_score: float,
        unsupported_claims: list[str],
        drift_detected: bool,
    ) -> str:
        if not context_text.strip():
            return "The response was produced without retrieved grounding context."
        if requires_tools and not tool_used:
            return "The response answered a factual or time-sensitive query without calling a retrieval tool."
        if unsupported_claims:
            return "The response introduced claims that were not sufficiently supported by retrieved context."
        if drift_detected:
            return "The response drifted beyond the evidence and introduced new facts during synthesis."
        if overlap_score < 0.5:
            return "The response had weak semantic alignment with the retrieved context."
        return "The response remained grounded in the available context and tool evidence."

    def _build_case_study_explanation(
        self,
        label: str,
        overlap_score: float,
        unsupported_claims: list[str],
        uncertainty_hits: list[str],
        requires_tools: bool,
        tool_used: bool,
        drift_analysis: dict,
    ) -> str:
        if label == "HALLUCINATED":
            reasons = []
            if overlap_score < 0.5:
                reasons.append("low overlap with the retrieved context")
            if unsupported_claims:
                reasons.append("unsupported claims")
            if requires_tools and not tool_used:
                reasons.append("missing tool usage for a fact-sensitive query")
            if drift_analysis["drift_detected"]:
                reasons.append("response drift")
            if uncertainty_hits:
                reasons.append("uncertainty markers")
            return (
                "This response is classified as hallucinated because it shows "
                + ", ".join(reasons)
                + "."
            )
        return (
            "This response is classified as grounded because it stays aligned with retrieved context, "
            "does not rely on unsupported claims, and shows acceptable evidence overlap."
        )

    def _build_prevention_suggestion(
        self,
        requires_tools: bool,
        tool_used: bool,
        overlap_score: float,
        unsupported_claims: list[str],
        context_text: str,
    ) -> str:
        suggestions = []
        if not context_text.strip():
            suggestions.append("Require retrieval before answer generation.")
        if requires_tools and not tool_used:
            suggestions.append("Enforce a tool call policy for time-sensitive or factual prompts.")
        if overlap_score < 0.5:
            suggestions.append("Add a groundedness check that rejects low-overlap drafts.")
        if unsupported_claims:
            suggestions.append("Force the model to cite or map each claim back to retrieved context.")
        if not suggestions:
            suggestions.append("Keep the current retrieval-first workflow and retain the self-reflection guard.")
        return " ".join(suggestions)

    def _normalize_tools(self, tools_used: Any) -> list[str]:
        if tools_used is None:
            return []
        if isinstance(tools_used, str):
            return [tools_used]
        if isinstance(tools_used, list):
            return [self._to_text(tool) for tool in tools_used if self._to_text(tool)]
        return [self._to_text(tools_used)]

    def _split_claims(self, text: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]

    def _tokenize(self, text: str) -> list[str]:
        lowered = self._to_text(text).lower()
        return [token for token in re.findall(r"[a-z0-9]+", lowered) if token not in STOP_WORDS]

    def _to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(self._to_text(item) for item in value)
        if isinstance(value, dict):
            return "\n".join(f"{key}: {self._to_text(item)}" for key, item in value.items())
        return str(value)
