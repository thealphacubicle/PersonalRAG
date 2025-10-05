# FastAPI API for PersonalRAG
import os
import sys
from pathlib import Path
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on sys.path for imports like `from src.rag import ...`
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

# Reuse RAG pipeline utilities
from src.agents import AgentController, EmailAgent, EmailService, SummarizerAgent
from src.rag import build_faiss_index, chunk_documents, load_github_json, load_pdfs, load_text_files

app = FastAPI(title="PersonalRAG API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://thealphacubicle.dev",
        "http://localhost:3000",
        "https://thealphacubicle.github.io",
    ],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=3, description="Conversation session identifier")
    query: str = Field(..., min_length=1, description="User question")


class ChatResponse(BaseModel):
    status_code: int = Field(..., ge=100, le=599, description="HTTP-like status code for the request outcome")
    answer: str = Field("", description="Answer text if available; empty on error")
    sources: List[str] = Field(default_factory=list, description="Unique list of source document identifiers")
    tools: List[str] = Field(default_factory=list, description="Ordered list of tools invoked during the turn")
    error: Optional[str] = Field(None, description="Error message if any; null when success")


class HealthResponse(BaseModel):
    status_code: int = Field(..., ge=100, le=599)
    ready: bool = Field(..., description="True when the service is fully operational")
    error: Optional[str] = Field(None, description="Error message if any")
    details: Optional[dict] = Field(None, description="Optional diagnostics like env/docs/index status")


# Global vectorstore, built once on startup
VECTORSTORE = None
AGENT_CONTROLLER: Optional[AgentController] = None
EMAIL_SERVICE: Optional[EmailService] = None
EMAIL_AGENT: Optional[EmailAgent] = None
SUMMARIZER_AGENT: Optional[SummarizerAgent] = None
DOCS_DIR = CURRENT_DIR.parent / "docs"
STARTUP_ERROR: Optional[str] = None


def _discover_files():
    pdfs: List[str] = []
    text_files: List[str] = []
    json_file: Optional[str] = None
    if not DOCS_DIR.exists():
        return pdfs, text_files, json_file
    for p in DOCS_DIR.iterdir():
        if p.suffix.lower() == ".pdf":
            pdfs.append(str(p))
        elif p.suffix.lower() == ".txt":
            text_files.append(str(p))
        elif p.suffix.lower() == ".json" and json_file is None:
            json_file = str(p)
    return pdfs, text_files, json_file


@app.on_event("startup")
def startup_build_index():
    global VECTORSTORE, STARTUP_ERROR, AGENT_CONTROLLER, EMAIL_SERVICE, EMAIL_AGENT, SUMMARIZER_AGENT

    try:
        # Basic env check
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set in environment variables.")

        pdf_files, text_files, json_file = _discover_files()
        if not pdf_files and not text_files and not json_file:
            raise RuntimeError("No documents found in docs directory.")

        all_docs = []
        if pdf_files:
            all_docs.extend(load_pdfs(pdf_files))
        if text_files:
            all_docs.extend(load_text_files(text_files))
        if json_file:
            all_docs.extend(load_github_json(json_file))

        if not all_docs:
            raise RuntimeError("No documents loaded for indexing.")

        chunked = chunk_documents(all_docs)
        VECTORSTORE = build_faiss_index(chunked)

        EMAIL_SERVICE = EmailService()
        EMAIL_AGENT = EmailAgent(EMAIL_SERVICE)
        SUMMARIZER_AGENT = SummarizerAgent()
        AGENT_CONTROLLER = AgentController(
            vectorstore=VECTORSTORE,
            email_agent=EMAIL_AGENT,
            summarizer_agent=SUMMARIZER_AGENT,
        )
        STARTUP_ERROR = None
    except Exception as e:
        # Don't crash the app; mark as unavailable and report via /v1/chat
        VECTORSTORE = None
        STARTUP_ERROR = f"Startup failed: {e}"


@app.get("/health", response_model=HealthResponse)
def health(response: Response):
    env_ok = bool(os.environ.get("OPENAI_API_KEY"))
    pdfs, text_files, json_file = _discover_files()
    docs_present = bool(pdfs or text_files or json_file)
    index_ready = VECTORSTORE is not None and not STARTUP_ERROR
    agent_ready = AGENT_CONTROLLER is not None and index_ready

    if not env_ok:
        code = 503
        msg = "OPENAI_API_KEY not set."
    elif STARTUP_ERROR:
        code = 503
        msg = STARTUP_ERROR
    elif not docs_present:
        code = 503
        msg = "No documents found in docs directory."
    elif not index_ready:
        code = 503
        msg = "Vector index not ready."
    elif not agent_ready:
        code = 503
        msg = "Agent controller not initialized."
    else:
        code = 200
        msg = None

    response.status_code = code
    return HealthResponse(
        status_code=code,
        ready=(code == 200),
        error=msg,
        details={
            "env": "ok" if env_ok else "missing",
            "pdf_count": len(pdfs),
            "text_count": len(text_files),
            "has_json": bool(json_file),
            "index_ready": index_ready,
            "agent_ready": agent_ready,
            "email_configured": bool(EMAIL_SERVICE and EMAIL_SERVICE.owner_address),
        },
    )


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest, response: Response):
    session_id = (req.session_id or "").strip()
    query = (req.query or "").strip()

    if not session_id:
        response.status_code = 422
        return ChatResponse(
            status_code=422,
            answer="",
            sources=[],
            tools=[],
            error="Session identifier cannot be empty.",
        )
    if not query:
        response.status_code = 422
        return ChatResponse(status_code=422, answer="", sources=[], tools=[], error="Query cannot be empty.")

    if STARTUP_ERROR:
        response.status_code = 503
        return ChatResponse(status_code=503, answer="", sources=[], error=STARTUP_ERROR)

    if VECTORSTORE is None:
        response.status_code = 503
        return ChatResponse(status_code=503, answer="", sources=[], error="Vector index not ready.")

    if AGENT_CONTROLLER is None:
        response.status_code = 503
        return ChatResponse(status_code=503, answer="", sources=[], error="Agent controller not ready.")

    try:
        result = AGENT_CONTROLLER.handle_message(session_id, query)
    except Exception as exc:  # noqa: BLE001
        response.status_code = 502
        return ChatResponse(
            status_code=502,
            answer="",
            sources=[],
            tools=[],
            error=f"Agent error: {exc}",
        )

    response.status_code = result.status_code
    return ChatResponse(
        status_code=result.status_code,
        answer=result.message,
        sources=result.sources,
        tools=result.tools,
        error=result.error if not result.ok else None,
    )


if __name__ == "__main__":
    # Optional local run: python -m src.app.api
    import uvicorn

    uvicorn.run("src.app.api:app", host="0.0.0.0", port=8000, reload=True)
