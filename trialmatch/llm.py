"""Thin wrapper around the Anthropic Messages API for schema-constrained calls."""

from __future__ import annotations

import threading
from typing import Protocol, TypeVar

import anthropic
from pydantic import BaseModel

from .config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    pass


class LLMRefusal(LLMError):
    """The model declined the request (stop_reason == "refusal")."""


class StructuredLLM(Protocol):
    def structured(self, *, system: str, user_blocks: list[str], schema: type[T]) -> T: ...


class AnthropicLLM:
    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None):
        self.settings = settings
        self.client = client or anthropic.Anthropic(max_retries=5)
        self._usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        self._usage_lock = threading.Lock()

    def usage_summary(self) -> dict:
        """Tokens used so far (input includes cache reads and writes)."""
        with self._usage_lock:
            return dict(self._usage)

    def structured(self, *, system: str, user_blocks: list[str], schema: type[T]) -> T:
        """One request whose reply is guaranteed to parse as ``schema``.

        ``user_blocks`` are sent in order; every block except the last is marked
        cacheable, so calls that share a prefix (same note, different criteria)
        can reuse it.
        """
        content = [{"type": "text", "text": block} for block in user_blocks]
        for block in content[:-1]:
            block["cache_control"] = {"type": "ephemeral"}

        # Streamed so a long decomposition cannot hit the HTTP timeout.
        with self.client.messages.stream(
            model=self.settings.model,
            max_tokens=32000,
            thinking={"type": "adaptive"},
            output_config={"effort": self.settings.effort},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
            output_format=schema,
        ) as stream:
            message = stream.get_final_message()

        usage = message.usage
        with self._usage_lock:
            self._usage["requests"] += 1
            self._usage["output_tokens"] += usage.output_tokens
            self._usage["input_tokens"] += (
                usage.input_tokens + (usage.cache_read_input_tokens or 0) + (usage.cache_creation_input_tokens or 0)
            )

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise LLMRefusal(f"Model declined the request ({getattr(details, 'category', None)})")
        if message.stop_reason == "max_tokens":
            raise LLMError("Model output was truncated at max_tokens")

        parsed = getattr(message, "parsed_output", None)
        if parsed is None:
            text = next((b.text for b in message.content if b.type == "text"), None)
            if text is None:
                raise LLMError("Model returned no text block")
            parsed = schema.model_validate_json(text)
        return parsed
