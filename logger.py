import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from db import connect_db, create_observability_logs_table, insert_observability_log


class ObservabilityLogger:
    def __init__(self, neon_db_url: str | None, local_path: str):
        self.neon_db_url = neon_db_url
        self.local_path = Path(local_path)
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.base_dir = self.local_path.parent.parent if self.local_path.parent.name == "logs" else self.local_path.parent
        self.reports_dir = self.base_dir / "reports"
        self.metrics_dir = self.base_dir / "metrics"
        self.attempts_dir = self.base_dir / "attempts"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.attempts_dir.mkdir(parents=True, exist_ok=True)
        self.conn = None
        if neon_db_url and str(neon_db_url).strip():
            self.conn = connect_db(neon_db_url)
            create_observability_logs_table(self.conn)

    def log(self, query, response, context, analysis, tools_used=None) -> dict:
        created_at = datetime.now(timezone.utc)
        record = {
            "id": str(uuid4()),
            "query": query,
            "response": response,
            "context": self._normalize_text(context),
            "tools_used": tools_used or analysis.get("tools_used", []),
            "overlap_score": float(analysis["overlap_score"]),
            "risk_score": int(analysis["risk_score"]),
            "label": analysis["label"],
            "reasoning": analysis["reasoning"],
            "root_cause": analysis["root_cause"],
            "prevention": analysis["prevention_suggestion"],
            "report_reference_id": analysis.get("report_reference_id"),
            "report_file_path": analysis.get("report_file_path"),
            "created_at": created_at,
        }
        if self.conn is not None:
            insert_observability_log(self.conn, record)
        self._append_to_local_file(
            {
                "timestamp": created_at.isoformat(),
                "query": query,
                "response": response,
                "retrieved_context": self._normalize_text(context),
                "tools_used": record["tools_used"],
                "report_reference_id": record["report_reference_id"],
                "report_file_path": record["report_file_path"],
                "hallucination_analysis": analysis,
            }
        )
        return record

    def save_report(self, report: dict) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        report_id = report.get("id", str(uuid4()))
        report_path = self.reports_dir / f"report_{report_id}_{timestamp}.json"
        latest_report_path = self.reports_dir / "latest_report.json"
        serialized = json.dumps(report, indent=2, ensure_ascii=False, default=str)
        report_path.write_text(serialized, encoding="utf-8")
        latest_report_path.write_text(serialized, encoding="utf-8")
        return str(report_path)

    def save_metrics(self, metrics: dict, report_id: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        metrics_path = self.metrics_dir / f"metrics_{report_id}_{timestamp}.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return str(metrics_path)

    def save_attempt_log(self, attempt_log: dict, report_id: str, attempt_number: int) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        attempt_path = self.attempts_dir / f"attempt_{report_id}_{attempt_number}_{timestamp}.json"
        attempt_path.write_text(
            json.dumps(attempt_log, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return str(attempt_path)

    def close(self) -> None:
        if getattr(self, "conn", None) is not None and not self.conn.closed:
            self.conn.close()

    def _append_to_local_file(self, entry: dict) -> None:
        payload = []
        if self.local_path.exists():
            existing = self.local_path.read_text(encoding="utf-8").strip()
            if existing:
                payload = json.loads(existing)
                if not isinstance(payload, list):
                    payload = [payload]
        payload.append(entry)
        self.local_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def _normalize_text(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(self._normalize_text(item) for item in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, default=str)
        return str(value)
