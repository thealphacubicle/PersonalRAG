"""Summarization helper that produces concise answers from retrieved context."""

from __future__ import annotations

from typing import Iterable, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class SummarizerAgent:
    """Generates concise answers grounded in retrieved snippets."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2) -> None:
        self._llm = ChatOpenAI(model=model_name, temperature=temperature)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You polish answers for Srihari Raman's assistant. Given a visitor question, trusted context "
                    "snippets, and an optional draft, return a final reply that is concise, natural, and grounded. Keep it "
                    "to at most three sentences, avoid bullet lists or section headers, and never paste large excerpts or raw "
                    "snippets. Mention only facts supported by the context, and if key information is missing, say so briefly "
                    "while staying positive. Output plain text only."
                ),
                (
                    "human",
                    "Question: {question}\n\nContext snippets:\n{context}\n\nDraft answer (can be empty):\n{draft}\n\nCitable sources:\n{sources}\n\nCompose the final reply now.",
                ),
            ]
        )

    def summarize(
        self,
        *,
        question: str,
        contexts: Iterable[str],
        draft: Optional[str] = None,
        sources: Optional[Iterable[str]] = None,
    ) -> str:
        context_text = "\n\n".join(contexts) if contexts else "(no context)"
        if len(context_text) > 2000:
            context_text = context_text[:2000] + "\n\n[context truncated]"
        draft_text = (draft or "").strip()
        sources_text = "\n".join(f"- {src}" for src in (sources or [])) or "(no sources)"
        result = (self._prompt | self._llm).invoke(
            {
                "question": question,
                "context": context_text,
                "draft": draft_text or "(no draft)",
                "sources": sources_text,
            }
        )
        return result.content.strip()


__all__ = ["SummarizerAgent"]
