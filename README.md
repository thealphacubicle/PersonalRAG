# Srihari Knowledge Chatbot (Personal RAG)

A minimal, production‑oriented Retrieval Augmented Generation (RAG) application that answers questions about **Srihari Raman** using local source documents (PDF resume, LinkedIn export, GitHub project metadata). It combines:

- Local document ingestion (PDF + structured JSON)
- Chunking & embedding (OpenAI embeddings)
- FAISS vector similarity search
- Positive, controlled system prompt (custom persona rules)
- LangChain RetrievalQA (non‑streaming path) or custom streaming UI (optional variant)
- Streamlit chat interface with source attribution & deduplicated citations

---
## Features
- ⚡ One‑time indexing cached via `st.cache_resource`
- 🔍 Semantic retrieval over curated profile knowledge
- 🧠 Structured system prompt enforcing tone, positivity, and fallback rules
- 🗂 Source documents cited (duplicates removed)
- 💬 Chat UI with expandable sources section
- 🧩 Modular pipeline (`main.py`) reusable outside the UI
- 🛡 Avoids hallucinations by grounding answers in retrieved chunks

---
## Architecture
```
                +---------------------+
                |  Source Documents   |
                |  (PDF + JSON)       |
                +----------+----------+
                           |
                    Load & Normalize
                           |
                     Chunk (LangChain)
                           |
                    Embed (OpenAI API)
                           |
                   +-------v-------+
                   |   FAISS Index |
                   +-------+-------+
                           |
                        Retrieval
                           |
User Query ---> Prompt Assembly (System Persona + Context) ---> OpenAI Chat Model ---> Formatted Answer + Sources
```

---
## Project Structure
```
src/
  main.py               # Core RAG build + query utilities
  github.py             # (If present) GitHub project extraction helper
  app/app.py            # Streamlit chat application
  docs/                 # Local knowledge base (PDFs, JSON)
    Srihari_Online_Resume.pdf
    Srihari_LinkedIn_Profile.pdf
    thealphacubicle_projects.json
pyproject.toml          # Poetry configuration
poetry.lock             # Locked dependency graph
```
> Do **not** commit secrets (`.env`).

---
## System Prompt Behavior
The assistant always:
- Speaks positively and respectfully about Srihari
- Blends retrieved facts with encouraging framing
- Never responds with a bare “I don't know”; instead provides constructive, positive context
- Supplies admiration/strengths if the fact is missing
- Avoids speculation beyond provided context

If retrieval returns little/no content, the fallback still produces a confidence‑preserving, positive answer (per the system prompt instructions).

---
## Prerequisites
- Python 3.13.x (or 3.12.x) but **< 3.15** (due to `faiss-cpu` wheel availability)
- Poetry (>=1.7)
- OpenAI API key (model: `gpt-4o-mini` + `text-embedding-3-small`)

---
## Installation
```bash
# Clone (example)
git clone <repo-url> personal-rag
cd personal-rag

# Install dependencies
poetry install

# (Optional) Force correct Python constraint if needed
# Edit pyproject.toml: python = ">=3.12,<3.15"
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-your-key
```
> Never commit `.env`.

---
## Indexing & CLI Test
You can exercise the pipeline directly:
```bash
poetry run python src/main.py
```
This will:
1. Load documents from `src/docs/`
2. Chunk & embed
3. Build FAISS index
4. Run a sample query

---
## Run the Streamlit App
```bash
poetry run streamlit run src/app/app.py
```
Then open the displayed local URL (typically http://localhost:8501).

First run builds the vector index (cached). Subsequent queries are fast.

---
## Adding / Updating Documents
Place additional PDFs or a new JSON metadata file into `src/docs/`. Then:
- Stop the app
- (Optional) Clear Streamlit cache: `streamlit cache clear`
- Restart the app to rebuild the index automatically

Recommended JSON shape (example excerpt):
```json
{
  "username": "thealphacubicle",
  "projects": [
    {
      "name": "project-name",
      "description": "...",
      "language": "Python",
      "stars": 42,
      "forks": 3,
      "readme": "Full README content or summary"
    }
  ]
}
```

---
## Retrieval Settings
| Parameter | Where | Purpose |
|-----------|-------|---------|
| k         | `run_query(..., k=4)` / app constant | Number of chunks retrieved |
| chunk_size| `chunk_documents` in `main.py`        | Larger = fewer, broader chunks |
| chunk_overlap | same                              | Helps maintain semantic continuity |

To tune recall vs speed, adjust `k` and `chunk_size`.

---
## Extending
| Goal | Suggested Change |
|------|------------------|
| Streaming answers | Replace RetrievalQA with manual retrieve + incremental OpenAI streaming (already prototyped in earlier variant) |
| Multi-file formats | Add loaders from `langchain_community.document_loaders` |
| Persist index | Use `FAISS.save_local()` / `load_local()` |
| Rerank stage | Insert Cohere / CrossEncoder reranker after retrieval |
| Eval harness | Add a small question → expected nugget set & measure hit rate |

---
## Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| `faiss-cpu forbidden` | Python constraint mismatch | Set `python = ">=3.12,<3.15"` in pyproject; recreate env |
| Empty answers | Docs missing / not loaded | Confirm PDF & JSON present in `src/docs/` |
| OpenAI auth error | Missing API key | Add to `.env` or export shell var |
| Slow first query | Embedding build | Normal; cached afterward |
| Duplicate sources | Now deduplicated in UI | If persists, clear cache |

---
## Security & Privacy
- All documents remain local; only embeddings and prompts leave the machine (to OpenAI).
- Do not upload proprietary or sensitive personal data unless you accept that embedding contents are processed by the API provider.
- Keep `.env` excluded via `.gitignore`.

---
## Potential Improvements (Roadmap)
- Add streaming UI version with typing animation (if not active)
- Add conversation memory while re‑grounding each turn
- Introduce guardrails for off‑topic queries
- Automated evaluation script
- Dockerfile for reproducible deployment

---
## License
Specify a license (e.g., MIT) here. Example:
```
MIT License – 2025 Your Name
```

---
## Quick Reference
```bash
poetry install                           # Install deps
poetry run python src/main.py            # CLI test
poetry run streamlit run src/app/app.py  # Launch UI
streamlit cache clear                    # Reset cached index (optional)
```

---
## Disclaimer
This assistant is tuned to remain **positive** by design; verify factual claims against the original source documents when precision is critical.

---
Happy exploring! Modify, extend, and adapt for broader personal knowledge bases.

