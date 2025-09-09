import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
from pathlib import Path
from typing import List

try:
    # Reuse pipeline functions from main module
    from src.main import (
        build_faiss_index,
        chunk_documents,
        load_github_json,
        load_pdfs,
        run_query,
    )
except ModuleNotFoundError as e:
    st.error(f"Required module not found: {e}. Try running `pip install langchain-openai` in your terminal.")
    st.stop()

# -----------------------------
# Paths & Data Loading Helpers
# -----------------------------
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def _discover_files():
    pdfs: List[str] = []
    json_file: str | None = None
    if not DOCS_DIR.exists():
        return pdfs, json_file
    for p in DOCS_DIR.iterdir():
        if p.suffix.lower() == ".pdf":
            pdfs.append(str(p))
        elif p.suffix.lower() == ".json" and json_file is None:
            json_file = str(p)
    return pdfs, json_file


@st.cache_resource(show_spinner=True)
def build_index():
    pdf_files, json_file = _discover_files()
    if not pdf_files and not json_file:
        raise FileNotFoundError("No PDF or JSON files found in docs directory.")

    all_docs = []
    if pdf_files:
        all_docs.extend(load_pdfs(pdf_files))
    if json_file:
        all_docs.extend(load_github_json(json_file))

    if not all_docs:
        raise ValueError("No documents loaded for indexing.")

    chunked = chunk_documents(all_docs)
    return build_faiss_index(chunked)


# -----------------------------
# UI Components
# -----------------------------
def render_header():
    st.set_page_config(page_title="Personal RAG Chatbot", page_icon="💬", layout="wide")
    st.title("💬 Personal Profile RAG Chatbot")
    st.caption("Interactive Retrieval-Augmented Generation over resume, LinkedIn profile, and GitHub projects.")

    with st.expander("ℹ️ About this app", expanded=False):
        st.write(
            """
            This chatbot uses:
            - FAISS vector store built from PDF and GitHub project data
            - OpenAI embeddings for semantic search
            - LangChain RetrievalQA for grounded answers
            Enter a question below about the profile or projects and get an evidence-backed answer.
            """
        )

def ensure_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY not found. Please set it as an environment variable.")
        return False
    return True


# -----------------------------
# Chat Logic
# -----------------------------
SYSTEM_PROMPT = (
    "You are a concise, professional assistant answering only from the provided documents. "
    "If unsure, say you don't have that information in the sources."
)

RESULTS_K = 4  # Number of results to retrieve per query


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome to the Personal Profile RAG Chatbot!"}
        ]
    if "vectorstore" not in st.session_state:
        try:
            with st.spinner("Building vector index (first time only)..."):
                st.session_state.vectorstore = build_index()
        except Exception as e:
            st.error(f"Failed to build index: {e}")
            st.stop()


def add_message(role: str, content: str, meta: dict | None = None):
    st.session_state.messages.append({"role": role, "content": content, "meta": meta or {}})


def render_chat_history():
    for m in st.session_state.messages:
        if m["role"] == "user":
            st.chat_message("user").markdown(m["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(m["content"])
                if m.get("meta", {}).get("sources"):
                    with st.expander("Sources"):
                        for s in m["meta"]["sources"]:
                            st.write(f"• {s}")


# -----------------------------
# Main App
# -----------------------------
def main():
    render_header()
    if not ensure_api_key():
        return

    init_session()
    render_chat_history()

    user_input = st.chat_input("Ask something about the profile or projects...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        add_message("user", user_input)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                try:
                    result = run_query(st.session_state.vectorstore, user_input, k=RESULTS_K)
                    answer = result.get("result", "(No answer returned)")
                    sources = [d.metadata.get("source", "unknown") for d in result.get("source_documents", [])]
                except Exception as e:
                    answer = f"Error: {e}"
                    sources = []
            add_message("assistant", answer, meta={"sources": sources})
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.write(f"• {s}")


if __name__ == "__main__":
    main()
