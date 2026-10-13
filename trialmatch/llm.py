"""Thin wrapper around the Anthropic Messages API for schema-constrained calls."""

from __future__ import annotations

import threading
from typing import Protocol, TypeVar

import anthropic
from pydantic import BaseModel

from .config import Settings
from .tools import Tool

T = TypeVar("T", bound=BaseModel)

# Upper bound on model turns in one tool-using request, so a model that keeps
# calling tools cannot loop (and bill) without end.
MAX_TOOL_TURNS = 12


class LLMError(Exception):
    pass


class LLMRefusal(LLMError):
    """The model declined the request (stop_reason == "refusal")."""


class MissingCredentials(LLMError):
    """No Anthropic credentials are configured."""


class StructuredLLM(Protocol):
    def structured(
        self,
        *,
        system: str,
        user_blocks: list[str],
        schema: type[T],
        tools: list[Tool] | None = None,
        trace: list[dict] | None = None,
    ) -> T: ...


def _replay_block(block) -> dict:
    """An assistant content block in the form the API accepts back on the next turn.

    Thinking blocks must be returned exactly as received; text and tool-use
    blocks are rebuilt so SDK-only fields (such as ``parsed_output``) are not sent.
    """
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return block.model_dump(mode="json", exclude_none=True)


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

    def structured(
        self,
        *,
        system: str,
        user_blocks: list[str],
        schema: type[T],
        tools: list[Tool] | None = None,
        trace: list[dict] | None = None,
    ) -> T:
        """A request whose final reply is guaranteed to parse as ``schema``.

        ``user_blocks`` are sent in order; every block except the last is marked
        cacheable, so calls that share a prefix (same note, different criteria)
        can reuse it.

        With ``tools``, the model may call them before answering: each call is
        run here and its result sent back, until the model gives its final
        answer. Every call is appended to ``trace`` (name, input, output) so the
        run keeps a record of what was computed or looked up.
        """
        content = [{"type": "text", "text": block} for block in user_blocks]
        for block in content[:-1]:
            block["cache_control"] = {"type": "ephemeral"}
        messages: list[dict] = [{"role": "user", "content": content}]
        by_name = {tool.name: tool for tool in tools or []}
        definitions = [tool.definition() for tool in tools or []]

        for _ in range(MAX_TOOL_TURNS):
            message = self._send(system, messages, schema, definitions)
            if message.stop_reason != "tool_use":
                return self._parse(message, schema)

            results = []
            for block in message.content:
                if block.type != "tool_use":
                    continue
                tool = by_name.get(block.name)
                output, is_error = tool.run(block.input) if tool else (f"Unknown tool {block.name!r}", True)
                if trace is not None:
                    trace.append({"tool": block.name, "input": block.input, "output": output, "is_error": is_error})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error})
            messages.append({"role": "assistant", "content": [_replay_block(b) for b in message.content]})
            # All results of one turn go back together in a single user message.
            messages.append({"role": "user", "content": results})
        raise LLMError(f"Model was still calling tools after {MAX_TOOL_TURNS} turns")

    def _send(self, system: str, messages: list[dict], schema: type[T], tools: list[dict]):
        try:
            message = self._request(system, messages, schema, tools)
        except TypeError as exc:
            # The SDK reports absent credentials as a TypeError when the request is built.
            if "authentication" not in str(exc):
                raise
            raise MissingCredentials(
                "No Anthropic API key found. Copy .env.example to .env and set ANTHROPIC_API_KEY."
            ) from exc

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
        return message

    @staticmethod
    def _parse(message, schema: type[T]) -> T:
        parsed = getattr(message, "parsed_output", None)
        if parsed is None:
            text = next((b.text for b in message.content if b.type == "text"), None)
            if text is None:
                raise LLMError("Model returned no text block")
            parsed = schema.model_validate_json(text)
        return parsed

    def _request(self, system: str, messages: list[dict], schema: type[T], tools: list[dict]):
        # Streamed so a long decomposition cannot hit the HTTP timeout.
        extra = {"tools": tools} if tools else {}
        with self.client.messages.stream(
            model=self.settings.model,
            max_tokens=32000,
            thinking={"type": "adaptive"},
            output_config={"effort": self.settings.effort},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            output_format=schema,
            **extra,
        ) as stream:
            return stream.get_final_message()
