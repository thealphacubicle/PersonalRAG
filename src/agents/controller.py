"""High-level LangChain controller orchestrating conversation agents."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import FAISS

from .email import EmailAgent, EmailAgentPayload, EmailAgentResult, EmailFlowState
from .moderation import ModerationAgent
from .summarizer import SummarizerAgent

logger = logging.getLogger(__name__)


@dataclass
class ControllerResult:
    ok: bool
    message: str
    sources: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    status_code: int = 200
    error: Optional[str] = None


@dataclass
class SessionState:
    chat_history: List[BaseMessage] = field(default_factory=list)
    email_flow: EmailFlowState = field(default_factory=EmailFlowState)
    last_contexts: List[str] = field(default_factory=list)
    last_sources: List[str] = field(default_factory=list)


class RagToolInput(BaseModel):
    query: str = Field(..., description="Search query scoped to Srihari Raman's content")
    k: int = Field(4, ge=1, le=8, description="Maximum number of chunks to retrieve")


SYSTEM_PROMPT = (
    "You are Srihari Raman's personal AI concierge. Use the available tools to answer questions "
    "faithfully, stay grounded in retrieved context, and keep responses encouraging yet factual. "
    "When you use information from the retrieval tool, cite sources inline as [source]. If the "
    "context does not contain the answer, say so and offer to help the visitor reach out via email. "
    "Call the email workflow tool whenever the visitor wants to contact or collaborate with Srihari. "
    "Never invent personal contact details or facts not supported by the retrieved context. "
    "Ask clarifying questions when requests are ambiguous before calling tools."
)


class AgentController:
    """Coordinates moderation, retrieval, and specialized tools."""

    def __init__(
        self,
        *,
        vectorstore: Optional[FAISS],
        moderation_agent: Optional[ModerationAgent] = None,
        email_agent: EmailAgent,
        model_name: str = "gpt-4o-mini",
        summarizer_agent: Optional[SummarizerAgent] = None,
    ) -> None:
        self.vectorstore = vectorstore
        self.moderation = moderation_agent
        self.email_agent = email_agent
        self.summarizer = summarizer_agent
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self.dialog_llm = ChatOpenAI(model=model_name, temperature=0.2)
        self.agent_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                MessagesPlaceholder("agent_scratchpad"),
                ("human", "{input}"),
            ]
        )

    def update_vectorstore(self, store: Optional[FAISS]) -> None:
        self.vectorstore = store

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions[session_id] = SessionState()

    def _get_session(self, session_id: str) -> SessionState:
        with self._lock:
            return self._sessions.setdefault(session_id, SessionState())

    def handle_message(self, session_id: str, user_message: str) -> ControllerResult:
        """Process a user turn through moderation and the agent executor."""

        session = self._get_session(session_id)

        if self.moderation:
            moderation_in = self.moderation.check_input(user_message)
            if not moderation_in.safe and moderation_in.action == "block":
                return ControllerResult(
                    ok=False,
                    message="I’m sorry, but I can’t help with that.",
                    status_code=403,
                    error=moderation_in.reason,
                )
            sanitized_user = (
                moderation_in.sanitized_text
                if moderation_in.action == "sanitize" and moderation_in.sanitized_text
                else user_message
            )
        else:
            sanitized_user = user_message

        tools_used: List[str] = []
        tools = self._build_tools(session, tools_used)
        agent = create_openai_functions_agent(self.dialog_llm, tools, self.agent_prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

        try:
            result = executor.invoke({"input": sanitized_user, "chat_history": session.chat_history})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent execution failed: %s", exc)
            return ControllerResult(
                ok=False,
                message="Something went wrong while I was working on that. Could we try again?",
                status_code=500,
                error=str(exc),
            )

        output_text = result.get("output", "")
        intermediate = result.get("intermediate_steps", [])
        sources, contexts = self._extract_sources(intermediate)
        if sources:
            session.last_sources = sources
        if contexts:
            session.last_contexts = contexts

        if self.summarizer and session.last_contexts and "rag_search" in tools_used:
            try:
                summary = self.summarizer.summarize(
                    question=sanitized_user,
                    contexts=session.last_contexts,
                    draft=output_text,
                    sources=session.last_sources,
                )
                if summary:
                    output_text = summary
                    if "summarizer" not in tools_used:
                        tools_used.append("summarizer")
            except (ValueError, ValidationError) as exc:
                logger.exception("Summarizer failed: %s", exc)

        if self.moderation:
            moderation_out = self.moderation.check_output(output_text, session.last_contexts)
            if not moderation_out.safe and moderation_out.action == "block":
                sanitized = moderation_out.sanitized_text or "I’m sorry, but I can’t share that."
                session.chat_history.append(HumanMessage(content=sanitized_user))
                session.chat_history.append(AIMessage(content=sanitized))
                return ControllerResult(
                    ok=False,
                    message=sanitized,
                    sources=[],
                    tools=tools_used,
                    status_code=403,
                    error=moderation_out.reason,
                )
            if moderation_out.action == "sanitize" and moderation_out.sanitized_text:
                output_text = moderation_out.sanitized_text

        session.chat_history.append(HumanMessage(content=sanitized_user))
        session.chat_history.append(AIMessage(content=output_text))

        return ControllerResult(
            ok=True,
            message=output_text,
            sources=session.last_sources,
            tools=tools_used,
            status_code=200,
        )

    def _build_tools(self, session: SessionState, tools_used: List[str]) -> List[StructuredTool]:
        """Create tool instances bound to the active session."""

        def rag_tool(query: str, k: int = 4) -> str:
            if "rag_search" not in tools_used:
                tools_used.append("rag_search")
            if not self.vectorstore:
                return json.dumps({"status": "error", "message": "Vector store not ready."})
            docs = self.vectorstore.similarity_search(query, k=k)
            contexts: List[str] = []
            sources: List[str] = []
            for doc in docs:
                contexts.append(doc.page_content)
                source = doc.metadata.get("source", "unknown")
                if source not in sources:
                    sources.append(source)
            session.last_contexts = contexts
            session.last_sources = sources
            payload = {"status": "ok", "contexts": contexts, "sources": sources}
            return json.dumps(payload)

        def email_tool(
            intent: str,
            visitor_name: Optional[str] = None,
            visitor_email: Optional[str] = None,
            message_summary: Optional[str] = None,
            verification_code: Optional[str] = None,
        ) -> str:
            if "email_workflow" not in tools_used:
                tools_used.append("email_workflow")
            # Proceed directly to EmailAgent
            try:
                payload = EmailAgentPayload(
                    intent=intent,
                    visitor_name=visitor_name,
                    visitor_email=visitor_email,
                    message_summary=message_summary,
                    verification_code=verification_code,
                )
            except ValidationError as exc:
                logger.warning("Invalid email tool payload: %s", exc)
                return json.dumps(
                    {
                        "status": "error",
                        "user_message": "I need a valid email address and short note before I can continue.",
                        "error": "Invalid email payload",
                    }
                )
            result: EmailAgentResult = self.email_agent.handle(session.email_flow, payload)
            return result.json()

        rag_structured = StructuredTool.from_function(
            rag_tool,
            name="rag_search",
            description=(
                "Use this to retrieve factual context from Srihari's approved knowledge base. "
                "Provide the exact question the visitor is asking."
            ),
            args_schema=RagToolInput,
        )

        email_structured = StructuredTool.from_function(
            email_tool,
            name="email_workflow",
            description=(
                "Coordinate email introductions. Call this when the visitor wants to contact or collaborate "
                "with Srihari. Supply any contact details or verification codes the visitor has already shared."
            ),
        )

        return [rag_structured, email_structured]

    @staticmethod
    def _extract_sources(steps: Iterable[Tuple[object, str]]) -> Tuple[List[str], List[str]]:
        sources: List[str] = []
        contexts: List[str] = []
        for action, observation in steps:
            tool_name = getattr(action, "tool", None)
            if tool_name != "rag_search":
                continue
            if isinstance(observation, str):
                try:
                    data = json.loads(observation)
                except json.JSONDecodeError:
                    logger.warning("Failed to decode rag_search output: %s", observation)
                    continue
            elif isinstance(observation, dict):
                data = observation
            else:
                continue

            for source in data.get("sources", []) or []:
                if source not in sources:
                    sources.append(source)
            contexts.extend(data.get("contexts", []) or [])
        return sources, contexts


__all__ = ["AgentController", "ControllerResult", "SessionState"]
