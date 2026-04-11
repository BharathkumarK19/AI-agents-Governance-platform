from typing import Any

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import psycopg2
from psycopg2.extras import Json


CREATE_TRANSACTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    customer_id TEXT,
    item_name TEXT,
    item_id TEXT,
    price FLOAT,
    discount FLOAT,
    gst FLOAT,
    total_price FLOAT,
    profit FLOAT,
    margin FLOAT,
    timestamp TIMESTAMP,
    research_output JSONB,
    analysis_output JSONB,
    summary_output JSONB
);
"""

CREATE_AGENT_OBSERVABILITY_LOGS_SQL = """
CREATE TABLE IF NOT EXISTS agent_observability_logs (
    id UUID PRIMARY KEY,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    context TEXT,
    tools_used JSONB NOT NULL DEFAULT '[]'::jsonb,
    overlap_score FLOAT NOT NULL,
    risk_score INT NOT NULL,
    label TEXT NOT NULL,
    reasoning JSONB NOT NULL,
    root_cause TEXT NOT NULL,
    prevention TEXT NOT NULL,
    report_reference_id TEXT,
    report_file_path TEXT,
    created_at TIMESTAMP NOT NULL
);
"""


def connect_db(connection_string: str):
    if not connection_string or not connection_string.strip():
        raise ValueError("A PostgreSQL connection string is required.")
    normalized_connection_string = _normalize_connection_string(connection_string.strip())
    return psycopg2.connect(
        normalized_connection_string,
        connect_timeout=10,
        application_name="agents_governance_pipeline",
    )


def create_transactions_table(conn) -> None:
    try:
        with conn.cursor() as cursor:
            cursor.execute(CREATE_TRANSACTIONS_TABLE_SQL)
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_transactions_lookup
                ON transactions (customer_id, item_id, timestamp)
                """
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def create_observability_logs_table(conn) -> None:
    try:
        with conn.cursor() as cursor:
            cursor.execute(CREATE_AGENT_OBSERVABILITY_LOGS_SQL)
            cursor.execute(
                """
                ALTER TABLE agent_observability_logs
                ADD COLUMN IF NOT EXISTS report_reference_id TEXT
                """
            )
            cursor.execute(
                """
                ALTER TABLE agent_observability_logs
                ADD COLUMN IF NOT EXISTS report_file_path TEXT
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_observability_logs_created_at
                ON agent_observability_logs (created_at)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_observability_logs_label
                ON agent_observability_logs (label)
                """
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def find_existing_transaction(conn, txn: dict) -> dict | None:
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    customer_id,
                    item_name,
                    item_id,
                    price,
                    discount,
                    gst,
                    total_price,
                    profit,
                    margin,
                    timestamp,
                    research_output,
                    analysis_output,
                    summary_output
                FROM transactions
                WHERE customer_id = %s
                  AND item_id = %s
                  AND timestamp = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (
                    txn.get("customer_id"),
                    txn.get("item_id"),
                    _normalize_timestamp(txn.get("timestamp")),
                ),
            )
            row = cursor.fetchone()
    except Exception:
        conn.rollback()
        raise

    if not row:
        return None

    return {
        "input": {
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
        },
        "research_output": row[10],
        "analysis_output": row[11],
        "summary_output": row[12],
    }


def insert_transaction(conn, processed_data: dict) -> None:
    txn = processed_data["input"]
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO transactions (
                    customer_id,
                    item_name,
                    item_id,
                    price,
                    discount,
                    gst,
                    total_price,
                    profit,
                    margin,
                    timestamp,
                    research_output,
                    analysis_output,
                    summary_output
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    txn.get("customer_id"),
                    txn.get("item_name"),
                    txn.get("item_id"),
                    txn.get("price"),
                    txn.get("discount"),
                    txn.get("gst"),
                    txn.get("total_price"),
                    txn.get("profit"),
                    txn.get("margin"),
                    _normalize_timestamp(txn.get("timestamp")),
                    Json(_json_ready(processed_data.get("research_output"))),
                    Json(_json_ready(processed_data.get("analysis_output"))),
                    Json(_json_ready(processed_data.get("summary_output"))),
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def insert_observability_log(conn, log_record: dict) -> None:
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO agent_observability_logs (
                    id,
                    query,
                    response,
                    context,
                    tools_used,
                    overlap_score,
                    risk_score,
                    label,
                    reasoning,
                    root_cause,
                    prevention,
                    report_reference_id,
                    report_file_path,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    log_record["id"],
                    log_record["query"],
                    log_record["response"],
                    log_record.get("context"),
                    Json(_json_ready(log_record.get("tools_used", []))),
                    log_record["overlap_score"],
                    log_record["risk_score"],
                    log_record["label"],
                    Json(_json_ready(log_record["reasoning"])),
                    log_record["root_cause"],
                    log_record["prevention"],
                    log_record.get("report_reference_id"),
                    log_record.get("report_file_path"),
                    _normalize_timestamp(log_record["created_at"]),
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _json_ready(value: Any) -> Any:
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return {"value": str(value)}


def _normalize_connection_string(connection_string: str) -> str:
    parsed = urlparse(connection_string)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))

    if not query.get("sslmode"):
        query["sslmode"] = "require"

    normalized_query = urlencode(query)
    return urlunparse(parsed._replace(query=normalized_query))


def _normalize_timestamp(value: Any) -> Any:
    if value is None:
        return None

    tzinfo = getattr(value, "tzinfo", None)
    if tzinfo is not None:
        return value.replace(tzinfo=None)

    return value
