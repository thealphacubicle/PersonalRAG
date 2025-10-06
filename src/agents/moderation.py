"""Moderation helpers used to guard user inputs and model outputs."""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PII_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}")
PII_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z]\w*\s)+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln)\b",
    re.IGNORECASE,
)


class ModerationVerdict(BaseModel):
    """Normalized moderation verdict returned by the guard model."""

    safe: bool = Field(..., description="True when the text is allowed to pass through as-is")
    action: str = Field(
        ..., description="One of: allow, block, sanitize. Indicates how the application should react"
    )
    reason: str = Field(..., description="Short explanation for logging and UX messaging")
    sanitized_text: Optional[str] = Field(
        None, description="If action is sanitize, the sanitized payload the client can safely show"
    )


def _remove_pii(text: str) -> str:
    """Redact common PII patterns from a string."""

    if not text:
        return text
    text = PII_PHONE_RE.sub("[phone redacted]", text)
    text = PII_ADDRESS_RE.sub("[address redacted]", text)
    return text


def _contains_pii(text: str) -> bool:
    if not text:
        return False
    return bool(PII_PHONE_RE.search(text) or PII_ADDRESS_RE.search(text))


class ModerationAgent:
    """Wraps an LLM-based moderation chain for both input and output checks."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self._llm = ChatOpenAI(model=model_name, temperature=temperature)

        self._parser = PydanticOutputParser(pydantic_object=ModerationVerdict)
        self._format_instructions = self._parser.get_format_instructions()

        base_system = (
            "You are a strict safety and hallucination guard for Srihari Raman's personal assistant. "
            "Allow visitors to share their own name, professional title, and email address when they "
            "request an introduction, but still block phone numbers, street addresses, hateful or "
            "violent content, self-harm coaching, or claims that contradict provided evidence. When "
            "you choose 'sanitize', replace only the unsafe fragments while keeping helpful details."
        )
        self._input_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    base_system
                    + "\n\nReturn your analysis in JSON using these instructions:\n{format_instructions}\n"
                    + "Focus on whether the USER input can be processed by the assistant."
                ),
                ("human", "User input:\n{text}"),
            ]
        )
        self._output_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    base_system
                    + "\n\nReturn your analysis in JSON using these instructions:\n{format_instructions}\n"
                    + "Focus on whether the ASSISTANT draft can be safely returned."
                ),
                (
                    "human",
                    "Assistant draft:\n{response}\n\nValidated context snippets:\n{context}\n",
                ),
            ]
        )

    def check_input(self, text: str) -> ModerationVerdict:
        """Run moderation against the raw user message."""

        if not text or not text.strip():
            return ModerationVerdict(safe=False, action="block", reason="Empty input rejected")

        # Quick heuristic to catch obvious PII harvesting before calling the model
        if _contains_pii(text):
            return ModerationVerdict(
                safe=False,
                action="block",
                reason="User input already contains sensitive personal information",
            )

        chain = self._input_prompt | self._llm | self._parser
        try:
            verdict = chain.invoke({"text": text, "format_instructions": self._format_instructions})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Moderation input check failed: %s", exc)
            return ModerationVerdict(
                safe=False,
                action="block",
                reason="Unable to validate user input for safety",
            )
        return verdict

    def check_output(self, draft: str, context: Optional[Iterable[str]] = None) -> ModerationVerdict:
        """Moderate the assistant draft with knowledge of trusted context."""

        context_str = "\n\n".join(context or []) or "(no retrieval context)"
        chain = self._output_prompt | self._llm | self._parser
        try:
            verdict = chain.invoke(
                {
                    "response": draft,
                    "context": context_str,
                    "format_instructions": self._format_instructions,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Moderation output check failed: %s", exc)
            sanitized = _remove_pii(draft)
            return ModerationVerdict(
                safe=False,
                action="block",
                reason="Unable to validate assistant draft for safety",
                sanitized_text=sanitized,
            )

        # Enforce PII sanitization even when the model thinks it's safe
        if verdict.safe and verdict.action == "allow" and _contains_pii(draft):
            sanitized = _remove_pii(draft)
            verdict = ModerationVerdict(
                safe=True,
                action="sanitize",
                reason="Direct PII removed from assistant draft",
                sanitized_text=sanitized,
            )

        if verdict.action == "sanitize" and not verdict.sanitized_text:
            verdict.sanitized_text = _remove_pii(draft)
        return verdict


__all__ = ["ModerationAgent", "ModerationVerdict"]
