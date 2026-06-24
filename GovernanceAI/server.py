import sys
import os
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import uuid
from typing import Any, Dict, List, Optional

# Resolve paths from this file location so startup works from any CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
CONVERSATIONS_FILE = PROJECT_ROOT / "GovernanceAI" / "conversations.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from main import run_system as _run_system

    def run_research_system(query):
        final_result, hallucination_report, trace, metrics = _run_system(query)
        return final_result, hallucination_report, trace, metrics, []

    print(f"[Backend] Successfully imported agent logic from {PROJECT_ROOT}")
except ImportError as e:
    print(f"[Backend] Failed to import from {PROJECT_ROOT}. Error: {e}")
    # Fallback to demo mode if backend is missing
    def run_research_system(query):
        return "Backend not found. Running in demo mode.", {"verdict": "SAFE"}, {"steps": []}, {}, []

app = FastAPI(title="Governance AI Backend Bridge")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_conversations() -> Dict[str, Dict[str, Any]]:
    if not CONVERSATIONS_FILE.exists():
        return {}

    try:
        with CONVERSATIONS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(data, dict):
        return {}

    return data


def _save_conversations() -> None:
    CONVERSATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CONVERSATIONS_FILE.open("w", encoding="utf-8") as file:
        json.dump(CONVERSATIONS, file, ensure_ascii=False, indent=2)


def _build_conversation_title(message: str) -> str:
    compact = " ".join((message or "").split()).strip()
    if not compact:
        return "Conversation"
    return compact[:57] + "..." if len(compact) > 60 else compact


def _ensure_conversation(conv_id: str, first_message: str) -> Dict[str, Any]:
    conversation = CONVERSATIONS.get(conv_id)
    now = _utc_now_iso()
    if conversation is None:
        conversation = {
            "id": conv_id,
            "title": _build_conversation_title(first_message),
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        CONVERSATIONS[conv_id] = conversation
    else:
        conversation["updated_at"] = now
    return conversation


def _serialize_message_content(final_result: Any) -> str:
    if isinstance(final_result, dict):
        if final_result.get("status") == "failed":
            error = final_result.get("error", "Unknown error")
            return (
                f"⚠️ **Run failed**\n\n"
                f"{error}\n\n"
                "Check your API keys, token budget, or provider credits and try again."
            )
        return json.dumps(final_result, ensure_ascii=False, indent=2)
    return str(final_result)


CONVERSATIONS: Dict[str, Dict[str, Any]] = _load_conversations()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str

class AgentInfo(BaseModel):
    id: str
    name: str
    role: str
    status: str
    description: str

# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "Governance AI Bridge"}

@app.get("/api/agents", response_model=List[AgentInfo])
async def get_agents():
    return [
        {"id": "research", "name": "Research Agent", "role": "Data Gathering", "status": "online", "description": "Expert at finding and verifying real-world information via Tavily Search."},
        {"id": "analysis", "name": "Analysis Agent", "role": "Pattern Recognition", "status": "online", "description": "Extracts insights and identifies governance trends from raw data."},
        {"id": "summary", "name": "Summary Agent", "role": "Communication", "status": "online", "description": "Translates complex findings into structured, presentation-ready reports."},
        {"id": "eval", "name": "Evaluation Agent", "role": "Quality Assurance", "status": "online", "description": "Monitors output for hallucinations and factual inconsistencies."},
    ]

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        query = request.message
        conv_id = request.conversation_id or str(uuid.uuid4())
        conversation = _ensure_conversation(conv_id, query)
        user_message = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": query,
            "timestamp": _utc_now_iso(),
        }
        conversation["messages"].append(user_message)
        
        # Run blocking model/tool work off the FastAPI event loop.
        final_result, hallucination_report, trace, metrics, sources = await asyncio.to_thread(
            run_research_system, query
        )
        assistant_message = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": _serialize_message_content(final_result),
            "timestamp": _utc_now_iso(),
            "trace": trace.to_dict() if hasattr(trace, "to_dict") else trace,
            "metrics": metrics,
            "hallucination_report": hallucination_report,
        }
        conversation["messages"].append(assistant_message)
        conversation["updated_at"] = assistant_message["timestamp"]
        _save_conversations()
        
        # Prepare response in the format expected by app.js
        return {
            "conversation_id": conv_id,
            "message": assistant_message,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations")
async def get_conversations():
    conversations = list(CONVERSATIONS.values())
    conversations.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return [
        {
            "id": item["id"],
            "title": item.get("title", "Conversation"),
            "updated_at": item.get("updated_at"),
            "created_at": item.get("created_at"),
        }
        for item in conversations
    ]


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    conversation = CONVERSATIONS.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    if conversation_id not in CONVERSATIONS:
        raise HTTPException(status_code=404, detail="Conversation not found")
    del CONVERSATIONS[conversation_id]
    _save_conversations()
    return {"status": "deleted", "conversation_id": conversation_id}

# ── Static Files ─────────────────────────────────────────────────────────────
# This serves the 'frontend' folder at the root URL
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    print(f"Warning: Frontend path {FRONTEND_DIR} not found.")

if __name__ == "__main__":
    print("Starting Governance AI Platform on http://localhost:8088")
    uvicorn.run(app, host="0.0.0.0", port=8088)
