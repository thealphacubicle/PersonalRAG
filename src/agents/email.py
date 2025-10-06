"""Email workflow agent for introductions and outreach requests."""

from __future__ import annotations

import logging
import os
import random
import re
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

_CODE_CHARS = "0123456789"
_CODE_EXPIRATION_MINUTES = 15
_MAX_VERIFICATION_ATTEMPTS = 3

EMAIL_REGEX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)


@dataclass
class EmailFlowState:
    """Per-session state machine for contact introductions."""

    status: str = "idle"
    visitor_name: Optional[str] = None
    visitor_email: Optional[str] = None
    message_summary: Optional[str] = None
    verification_code: Optional[str] = None
    code_expires_at: Optional[datetime] = None
    attempts: int = 0
    audit_trail: List[str] = field(default_factory=list)
    last_error: Optional[str] = None

    def reset(self) -> None:
        self.status = "idle"
        self.visitor_name = None
        self.visitor_email = None
        self.message_summary = None
        self.verification_code = None
        self.code_expires_at = None
        self.attempts = 0
        self.audit_trail.clear()
        self.last_error = None


class EmailAgentPayload(BaseModel):
    """Input schema used when the LLM calls into the email workflow tool."""

    intent: str = Field(
        ..., description="Latest user request or clarification about contacting Srihari"
    )
    visitor_name: Optional[str] = Field(
        None,
        description="Name supplied by the visitor, if explicitly provided",
        min_length=1,
    )
    visitor_email: Optional[str] = Field(
        None,
        description="Email address the visitor wants to use for introductions",
    )
    message_summary: Optional[str] = Field(
        None,
        description="Short summary of what the visitor wants to discuss with Srihari",
        min_length=4,
    )
    verification_code: Optional[str] = Field(
        None,
        description="Verification code the visitor received via email",
        min_length=4,
        max_length=10,
    )


class EmailAgentResult(BaseModel):
    """Result returned to the tool caller for relay to the user."""

    status: str
    user_message: str
    completed: bool = False
    error: Optional[str] = None


class EmailService:
    """Lightweight SMTP wrapper so we can swap transports later."""

    def __init__(self) -> None:
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() != "false"
        self.from_address = os.getenv("EMAIL_FROM")
        self.owner_address = os.getenv("OWNER_EMAIL")

    @property
    def is_configured(self) -> bool:
        required = [self.owner_address]
        if not self.smtp_host:
            # Allow dry-run logging even when SMTP is absent
            return False
        required.extend([self.smtp_username, self.smtp_password, self.from_address])
        return all(required)

    def send_email(
        self,
        subject: str,
        body: str,
        to_address: str,
        *,
        cc: Optional[List[str]] = None,
    ) -> bool:
        """Send an email using SMTP; falls back to logging when unavailable."""

        if not self.smtp_host:
            logger.info(
                "[EmailService] SMTP not configured; logging email instead. Subject=%s, To=%s, CC=%s",
                subject,
                to_address,
                cc,
            )
            logger.info(body)
            return False

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.from_address or self.smtp_username
        msg["To"] = to_address
        if cc:
            msg["Cc"] = ",".join(cc)
        msg.set_content(body)

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as smtp:
                if self.smtp_use_tls:
                    smtp.starttls()
                if self.smtp_username and self.smtp_password:
                    smtp.login(self.smtp_username, self.smtp_password)
                smtp.send_message(msg)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to send email via SMTP: %s", exc)
            return False

    def send_verification_code(self, to_address: str, code: str) -> bool:
        subject = "Verify your email to reach Srihari"
        body = (
            "Hello!\n\n"
            "You asked to contact Srihari Raman through his personal assistant. "
            "Please confirm your email by replying in the chat with this verification code:\n\n"
            f"    {code}\n\n"
            "This code expires in 15 minutes.\n"
        )
        return self.send_email(subject, body, to_address)

    def send_intro_email(
        self,
        *,
        visitor_name: Optional[str],
        visitor_email: str,
        summary: str,
        conversation: List[str],
    ) -> bool:
        subject = f"New introduction request from {visitor_name or 'a visitor'}"
        body = (
            "Hi Srihari,\n\n"
            "Someone just reached out through your personal assistant. Here are the details:\n"
            f"- Name: {visitor_name or 'Not provided'}\n"
            f"- Email: {visitor_email}\n"
            f"- Summary: {summary}\n\n"
            "Recent conversation log:\n"
            + "\n".join(f"• {entry}" for entry in conversation[-6:])
            + "\n\nCheers,\nInquira"
        )
        cc = [visitor_email]
        to_address = self.owner_address or ""
        if not to_address:
            logger.warning(
                "OWNER_EMAIL not configured; cannot deliver introduction email"
            )
            return False
        return self.send_email(subject, body, to_address, cc=cc)


class EmailAgent:
    """Handles multi-step email introductions with verification."""

    def __init__(self, email_service: EmailService) -> None:
        self._svc = email_service

    def handle(
        self, state: EmailFlowState, payload: EmailAgentPayload
    ) -> EmailAgentResult:
        state.audit_trail.append(payload.intent)

        if state.status in {"completed", "blocked"}:
            return EmailAgentResult(
                status=state.status,
                user_message="The email workflow is already finished for this session.",
                completed=state.status == "completed",
            )

        # Normalize potential structured fields the LLM already extracted
        if payload.visitor_name:
            state.visitor_name = payload.visitor_name.strip()
        if payload.message_summary:
            state.message_summary = payload.message_summary.strip()
        if payload.visitor_email:
            normalized = _normalize_email(payload.visitor_email)
            if not normalized:
                return EmailAgentResult(
                    status=state.status,
                    user_message="That email address doesn't look right. Could you double-check and share it again?",
                    error="Invalid email format",
                )
            state.visitor_email = normalized
        else:
            inferred = _extract_email(payload.intent)
            if inferred:
                state.visitor_email = inferred

        if state.status == "idle":
            state.status = "awaiting_email"
            return EmailAgentResult(
                status=state.status,
                user_message=(
                    "Happy to make the introduction! Please share your name, the email address "
                    "you'd like us to CC, and a short summary of why you're reaching out."
                ),
            )

        if state.status == "awaiting_email":
            if not state.visitor_email:
                return EmailAgentResult(
                    status=state.status,
                    user_message="I still need the email address you'd like me to use. Could you share it?",
                )
            if not state.message_summary:
                return EmailAgentResult(
                    status=state.status,
                    user_message="Thanks! Could you add a quick summary of what you'd like to discuss with Srihari?",
                )
            if not self._svc.owner_address:
                state.status = "blocked"
                state.last_error = "Missing OWNER_EMAIL configuration"
                return EmailAgentResult(
                    status="blocked",
                    user_message=(
                        "I can't complete the introduction right now because the mailbox isn't configured. "
                        "Please try again later."
                    ),
                    error="Email routing not configured",
                )
            state.verification_code = _generate_code()
            state.code_expires_at = datetime.now(tz=timezone.utc) + timedelta(
                minutes=_CODE_EXPIRATION_MINUTES
            )
            state.attempts = 0
            delivered = self._svc.send_verification_code(
                state.visitor_email, state.verification_code
            )
            if not delivered:
                state.last_error = "Failed to dispatch verification email"
            state.status = "awaiting_code"
            return EmailAgentResult(
                status=state.status,
                user_message=(
                    "I've sent a verification code to your email. Once you receive it, drop the code here "
                    "so I can introduce you to Srihari."
                    if delivered
                    else "I tried to send a verification code, but it may be delayed. Check your inbox and spam "
                    "folder—if nothing arrives, let me know and we can retry."
                ),
            )

        if state.status == "awaiting_code":
            if not payload.verification_code:
                return EmailAgentResult(
                    status=state.status,
                    user_message="Please share the verification code that was emailed to you so I can continue.",
                )
            if (
                state.code_expires_at
                and datetime.now(tz=timezone.utc) > state.code_expires_at
            ):
                state.status = "blocked"
                state.last_error = "Verification code expired"
                return EmailAgentResult(
                    status="blocked",
                    user_message=(
                        "That code expired. Let's restart the process—ask me again when you're ready and "
                        "I'll send a new one."
                    ),
                    error="Verification expired",
                )
            if payload.verification_code.strip() != (state.verification_code or ""):
                state.attempts += 1
                if state.attempts >= _MAX_VERIFICATION_ATTEMPTS:
                    state.status = "blocked"
                    state.last_error = "Verification attempts exceeded"
                    return EmailAgentResult(
                        status="blocked",
                        user_message=(
                            "Verification failed too many times, so I've closed the email workflow. "
                            "Feel free to start over whenever you're ready."
                        ),
                        error="Too many failed verification attempts",
                    )
                return EmailAgentResult(
                    status=state.status,
                    user_message="That code doesn't match. Please try again—I've got it here when you're ready.",
                )

            # Verification success -> send the actual introduction email
            conversation = state.audit_trail[-8:]
            delivered = self._svc.send_intro_email(
                visitor_name=state.visitor_name,
                visitor_email=state.visitor_email or "",
                summary=state.message_summary or payload.intent,
                conversation=conversation,
            )
            state.status = "completed"
            state.last_error = (
                None if delivered else "Failed to dispatch introduction email"
            )
            return EmailAgentResult(
                status="completed",
                user_message=(
                    "All set! I've introduced you to Srihari over email and CC'd you so you can continue "
                    "the conversation there."
                    if delivered
                    else "I verified you, but the email service had an issue. I'll let Srihari know to follow up."
                ),
                completed=True,
                error=None if delivered else "Delivery failure",
            )

        # Fallback guard
        return EmailAgentResult(
            status=state.status,
            user_message="I'm keeping the line open—let me know if you'd like to start again.",
        )


def _generate_code(length: int = 6) -> str:
    import secrets

    return "".join(secrets.choice(_CODE_CHARS) for _ in range(length))


def _extract_email(text: str) -> Optional[str]:
    if not text:
        return None
    match = EMAIL_REGEX.search(text)
    if match:
        return _normalize_email(match.group(0))
    return None


def _normalize_email(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    candidate = value.strip().lower()
    if EMAIL_REGEX.fullmatch(candidate):
        return candidate
    return None


__all__ = [
    "EmailAgent",
    "EmailAgentPayload",
    "EmailAgentResult",
    "EmailFlowState",
    "EmailService",
]
