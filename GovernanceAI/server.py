import sys
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import uuid
from typing import List, Optional

# Add the parent directory to sys.path so we can import the agent logic
backend_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(backend_path)

try:
    from main import run_research_system
    print(f"[Backend] Successfully imported agent logic from {backend_path}")
except ImportError as e:
    print(f"[Backend] Failed to import from {backend_path}. Error: {e}")
    # Fallback to demo mode if backend is missing
    def run_research_system(query):
        return "Backend not found. Running in demo mode.", {"verdict": "SAFE"}, {"steps": []}, {}, []

app = FastAPI(title="Governance AI Backend Bridge")

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
        
        # Run the real multi-agent pipeline
        final_result, hallucination_report, trace, metrics, sources = run_research_system(query)
        
        # Prepare response in the format expected by app.js
        return {
            "conversation_id": conv_id,
            "message": {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": str(final_result),
                "timestamp": uvicorn.config.datetime.now().isoformat() if hasattr(uvicorn.config, 'datetime') else "2024-01-01T00:00:00Z",
                "trace": trace.to_dict() if hasattr(trace, 'to_dict') else trace,
                "metrics": metrics,
                "hallucination_report": hallucination_report
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations")
async def get_conversations():
    # Placeholder for conversation history
    return []

# ── Static Files ─────────────────────────────────────────────────────────────
# This serves the 'frontend' folder at the root URL
frontend_path = os.path.join(os.getcwd(), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"Warning: Frontend path {frontend_path} not found.")

if __name__ == "__main__":
    print("Starting Governance AI Platform on http://localhost:8088")
    uvicorn.run(app, host="0.0.0.0", port=8088)
