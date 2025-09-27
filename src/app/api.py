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
from src.rag import (
    build_faiss_index,
    chunk_documents,
    load_github_json,
    load_pdfs,
    run_query,
)

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
    query: str = Field(..., min_length=1, description="User question")


class ChatResponse(BaseModel):
    status_code: int = Field(..., ge=100, le=599, description="HTTP-like status code for the request outcome")
    answer: str = Field("", description="Answer text if available; empty on error")
    sources: List[str] = Field(default_factory=list, description="Unique list of source document identifiers")
    error: Optional[str] = Field(None, description="Error message if any; null when success")


class HealthResponse(BaseModel):
    status_code: int = Field(..., ge=100, le=599)
    ready: bool = Field(..., description="True when the service is fully operational")
    error: Optional[str] = Field(None, description="Error message if any")
    details: Optional[dict] = Field(None, description="Optional diagnostics like env/docs/index status")


# Global vectorstore, built once on startup
VECTORSTORE = None
DOCS_DIR = CURRENT_DIR.parent / "docs"
STARTUP_ERROR: Optional[str] = None


def _discover_files():
    pdfs: List[str] = []
    json_file: Optional[str] = None
    if not DOCS_DIR.exists():
        return pdfs, json_file
    for p in DOCS_DIR.iterdir():
        if p.suffix.lower() == ".pdf":
            pdfs.append(str(p))
        elif p.suffix.lower() == ".json" and json_file is None:
            json_file = str(p)
    return pdfs, json_file


@app.on_event("startup")
def startup_build_index():
    global VECTORSTORE, STARTUP_ERROR

    try:
        # Basic env check
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set in environment variables.")

        pdf_files, json_file = _discover_files()
        if not pdf_files and not json_file:
            raise RuntimeError("No PDF or JSON files found in docs directory.")

        all_docs = []
        if pdf_files:
            all_docs.extend(load_pdfs(pdf_files))
        if json_file:
            all_docs.extend(load_github_json(json_file))

        if not all_docs:
            raise RuntimeError("No documents loaded for indexing.")

        chunked = chunk_documents(all_docs)
        VECTORSTORE = build_faiss_index(chunked)
        STARTUP_ERROR = None
    except Exception as e:
        # Don't crash the app; mark as unavailable and report via /v1/chat
        VECTORSTORE = None
        STARTUP_ERROR = f"Startup failed: {e}"


@app.get("/health", response_model=HealthResponse)
def health(response: Response):
    env_ok = bool(os.environ.get("OPENAI_API_KEY"))
    pdfs, json_file = _discover_files()
    docs_present = bool(pdfs or json_file)
    index_ready = VECTORSTORE is not None and not STARTUP_ERROR

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
            "has_json": bool(json_file),
            "index_ready": index_ready,
        },
    )


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest, response: Response):
    # Normalize query and validate non-empty after trimming
    query = (req.query or "").strip()
    if not query:
        response.status_code = 422
        return ChatResponse(status_code=422, answer="", sources=[], error="Query cannot be empty.")

    # If startup failed or vector index not ready, return service unavailable
    if STARTUP_ERROR:
        response.status_code = 503
        return ChatResponse(status_code=503, answer="", sources=[], error=STARTUP_ERROR)

    if VECTORSTORE is None:
        response.status_code = 503
        return ChatResponse(status_code=503, answer="", sources=[], error="Vector index not ready.")

    try:
        result = run_query(VECTORSTORE, query, k=4)
        answer = result.get("result", "(No answer returned)")
        raw_sources = [d.metadata.get("source", "unknown") for d in result.get("source_documents", [])]
        # Deduplicate while preserving order
        seen = set()
        sources: List[str] = []
        for s in raw_sources:
            if s not in seen:
                seen.add(s)
                sources.append(s)
        response.status_code = 200
        return ChatResponse(status_code=200, answer=answer, sources=sources, error=None)
    except Exception as e:
        # Classify common upstream errors to help frontend handling
        err = str(e)
        err_l = err.lower()
        if any(k in err_l for k in ["api key", "unauthorized", "authentication"]):
            code = 401
        elif any(k in err_l for k in ["rate limit", "too many requests"]):
            code = 429
        elif "timeout" in err_l:
            code = 504
        else:
            code = 502
        response.status_code = code
        return ChatResponse(status_code=code, answer="", sources=[], error=f"Error generating answer: {err}")


if __name__ == "__main__":
    # Optional local run: python -m src.app.api
    import uvicorn

    uvicorn.run("src.app.api:app", host="0.0.0.0", port=8000, reload=True)
