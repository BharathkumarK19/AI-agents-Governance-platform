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
