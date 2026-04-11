import os
from typing import Any

import requests


class Verifier:
    def __init__(
        self,
        timeout: int = 10,
        progress_callback=None,
        verbose: bool = True,
    ):
        self.serper_api_key = os.getenv("SERPER_API_KEY", "").strip()
        self.timeout = timeout
        self.progress_callback = progress_callback
        self.verbose = verbose

    def verify_claim(
        self,
        claim: str,
        query: str | None = None,
        reference_sources: list[dict[str, Any]] | None = None,
    ) -> dict:
        self._emit(
            "claim_started",
            claim=claim,
            query=query,
            reference_source_count=len(reference_sources or []),
        )
        serper_result = self._verify_with_serper(claim)
        self._emit("source_checked", claim=claim, source_name="serper", result=serper_result)

        support_count = 1 if serper_result["supported"] else 0
        status = "VERIFIED" if serper_result["supported"] else "UNSUPPORTED"

        result = {
            "claim": claim,
            "status": status,
            "support_count": support_count,
            "sources": {
                "serper": serper_result,
            },
        }
        self._emit(
            "claim_completed",
            claim=claim,
            status=status,
            support_count=support_count,
        )
        return result

    def verify_all(
        self,
        claims: list[str],
        query: str | None = None,
        reference_sources: list[dict[str, Any]] | None = None,
    ) -> list[dict]:
        return [
            self.verify_claim(claim, query=query, reference_sources=reference_sources)
            for claim in claims
        ]

    def _verify_with_serper(self, claim: str) -> dict:
        if not self.serper_api_key:
            return self._empty_source_result("Serper", "SERPER_API_KEY is not configured.")
        try:
            response = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self.serper_api_key, "Content-Type": "application/json"},
                json={"q": claim, "num": 5},
                timeout=self.timeout,
            )
            if response.status_code != 200:
                return self._empty_source_result("Serper", f"HTTP {response.status_code}")
            payload = response.json()
            results = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "content": item.get("snippet", ""),
                }
                for item in payload.get("organic", [])
            ]
            if not results:
                return self._empty_source_result("Serper", "No search results returned.")
            evidence = " ".join(
                f"{item.get('title', '').strip()} {item.get('content', '').strip()}".strip()
                for item in results
            ).strip()
            overlap = self._claim_overlap(claim, evidence)
            supported = overlap >= 0.45
            return {
                "supported": supported,
                "evidence": evidence[:500],
                "source": results[0].get("url") or "serper",
                "top_sources": [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "summary": item.get("content", "")[:220],
                    }
                    for item in results[:3]
                    if item.get("url")
                ],
                "results_checked": len(results),
                "overlap": round(overlap, 4),
            }
        except Exception as exc:
            return self._empty_source_result("Serper", str(exc))

    def _claim_supported(self, claim: str, evidence: str) -> bool:
        return self._claim_overlap(claim, evidence) >= 0.45

    def _claim_overlap(self, claim: str, evidence: str) -> float:
        if not evidence:
            return 0.0
        claim_tokens = set(self._tokenize(claim))
        evidence_tokens = set(self._tokenize(evidence))
        if not claim_tokens or not evidence_tokens:
            return 0.0
        return len(claim_tokens & evidence_tokens) / len(claim_tokens)

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in requests.utils.requote_uri(text).lower().replace("%20", " ").split() if token]

    def _empty_source_result(self, source_name: str, reason: str) -> dict:
        return {
            "supported": False,
            "evidence": reason,
            "source": source_name,
        }

    def _emit(self, event: str, **payload) -> None:
        if self.progress_callback is not None:
            try:
                self.progress_callback(event, payload)
            except Exception:
                return
            return

        if not self.verbose:
            return

        if event == "claim_started":
            print(
                f"[verification] checking claim: {payload.get('claim', '')[:140]} "
                f"(retrieved sources: {payload.get('reference_source_count', 0)})"
            )
            return

        if event == "source_checked":
            result = payload.get("result", {})
            overlap = result.get("overlap")
            overlap_text = f", overlap={overlap}" if overlap is not None else ""
            print(
                f"[verification] source={payload.get('source_name')} "
                f"supported={result.get('supported', False)}{overlap_text} "
                f"evidence={str(result.get('evidence', ''))[:120]}"
            )
            return

        if event == "claim_completed":
            print(
                f"[verification] claim result: status={payload.get('status')} "
                f"support_count={payload.get('support_count')}"
            )
