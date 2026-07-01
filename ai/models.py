"""Data models for AI / LLM inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Standardised message roles for chat completions."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    """A single message in a chat conversation."""

    role: Role
    content: str


@dataclass(frozen=True)
class ModelInfo:
    """Metadata about an available model."""

    name: str
    provider: str
    available: bool = True
    path: str = ""
    size_bytes: int = 0
    description: str = ""


@dataclass
class AIResult:
    """Result of a single inference call (generate or chat).

    ``success`` is ``False`` when the call failed due to connection,
    timeout, or model errors — inspect ``error`` for details.
    Partial text may still be available in ``text`` on failure.
    """

    success: bool
    text: str = ""
    model_used: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    processing_time_seconds: float = 0.0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatHistory:
    """A managed conversation history with a system prompt."""

    system_prompt: str = ""
    messages: list[Message] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.messages.append(Message(role=Role.USER, content=content))

    def add_assistant(self, content: str) -> None:
        self.messages.append(Message(role=Role.ASSISTANT, content=content))

    def to_openai_list(self) -> list[dict[str, str]]:
        """Return messages as OpenAI-style dict list."""
        result: list[dict[str, str]] = []
        if self.system_prompt:
            result.append({"role": Role.SYSTEM.value, "content": self.system_prompt})
        for msg in self.messages:
            result.append({"role": msg.role.value, "content": msg.content})
        return result

    def clear(self) -> None:
        self.messages.clear()
