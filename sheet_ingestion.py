import json
import os
from datetime import datetime
from typing import Any

import gspread
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials

load_dotenv()


HEADER_MAP = {
    "Customer ID": "customer_id",
    "Purchase Item Name": "item_name",
    "Item ID": "item_id",
    "Actual Price": "price",
    "Discount (%)": "discount",
    "GST Number": "gst_number",
    "GST (%)": "gst",
    "Total Price with GST": "total_price",
    "Profit": "profit",
    "Margin (%)": "margin",
    "Date & Time": "timestamp",
}


def _build_gspread_client() -> gspread.Client:
    creds = _load_credentials()
    client = gspread.authorize(creds)
    print("✅ Google Sheets client initialized successfully")
    return client


def _load_credentials() -> Credentials:
    json_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip()
    json_content = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

    if json_path and os.path.exists(json_path):
        return Credentials.from_service_account_file(json_path, scopes=scopes)
    if json_path:
        raise ValueError(
            f"GOOGLE_SERVICE_ACCOUNT_FILE is set but the file does not exist: {json_path}"
        )

    if json_content:
        try:
            creds_dict = json.loads(json_content)
        except json.JSONDecodeError as exc:
            raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON.") from exc
        return Credentials.from_service_account_info(creds_dict, scopes=scopes)

    raise ValueError(
        "Google Sheets credentials not configured. Set "
        "GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SERVICE_ACCOUNT_JSON"
    )


def _parse_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    cleaned = str(value).strip()
    if not cleaned:
        return 0.0

    cleaned = (
        cleaned.replace(",", "")
        .replace("%", "")
        .replace("₹", "")
        .replace("$", "")
    )
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value

    text = str(value).strip()
    if not text:
        return None

    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass

    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    return None


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "customer_id": str(row.get("Customer ID", "")).strip(),
        "item_name": str(row.get("Purchase Item Name", "")).strip(),
        "item_id": str(row.get("Item ID", "")).strip(),
        "price": _parse_float(row.get("Actual Price")),
        "discount": _parse_float(row.get("Discount (%)")),
        "gst_number": str(row.get("GST Number", "")).strip(),
        "gst": _parse_float(row.get("GST (%)")),
        "total_price": _parse_float(row.get("Total Price with GST")),
        "profit": _parse_float(row.get("Profit")),
        "margin": _parse_float(row.get("Margin (%)")),
        "timestamp": _parse_timestamp(row.get("Date & Time")),
    }


def _is_empty_row(row: dict[str, Any]) -> bool:
    return not any(str(value).strip() for value in row.values() if value is not None)


def fetch_sheet_data(sheet_url: str) -> list[dict]:
    if not sheet_url or not sheet_url.strip():
        raise ValueError("A Google Sheet URL is required.")

    client = _build_gspread_client()
    spreadsheet = client.open_by_url(sheet_url.strip())
    worksheet = spreadsheet.sheet1
    rows = worksheet.get_all_records(default_blank="")

    structured_data = []
    for row in rows:
        if not row or _is_empty_row(row):
            continue
        structured_data.append(_normalize_row(row))

    return structured_data
