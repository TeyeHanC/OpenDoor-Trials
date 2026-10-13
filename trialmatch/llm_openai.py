"""Model access through an OpenAI-compatible chat completions endpoint.

For gateways that serve non-Claude models. It offers the same ``structured``
call as ``AnthropicLLM`` (schema-checked final answer, optional tool loop), but
such endpoints differ in what they accept, so it degrades in steps:

    schema    response_format json_schema (strict) + strict tool definitions
    json      response_format json_object, schema described in the prompt
    prompt    no response_format, schema described in the prompt

A 400 from the endpoint moves one step down and the level is remembered. In
every mode the reply is validated against the schema here; an invalid reply is
sent back to the model once with the validation errors.

The prompts were written for Claude. Results from another model say nothing
about how the pipeline performs on Claude, and the reverse.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import TypeVar

import httpx
from anthropic import transform_schema
from pydantic import BaseModel, ValidationError

from .config import Settings
from .llm import MAX_TOOL_TURNS, LLMError, LLMRefusal, MissingCredentials
from .tools import Tool

T = TypeVar("T", bound=BaseModel)

MODES = ("schema", "json", "prompt")
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _schema_instruction(schema: dict) -> str:
    return (
        "\n\nReply with a single JSON object and nothing else: no prose, no code fence. "
        "It must conform to this JSON Schema:\n" + json.dumps(schema, ensure_ascii=False)
    )


def _extract_json(text: str) -> str:
    """The JSON object in a reply that may be fenced or wrapped in a little prose."""
    text = text.strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else text


class OpenAICompatLLM:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL")
        if not key or not base:
            raise MissingCredentials(
                "An OpenAI-compatible model needs a key and an endpoint. Set ANTHROPIC_API_KEY and "
                "ANTHROPIC_BASE_URL (or OPENAI_API_KEY and OPENAI_BASE_URL) in .env."
            )
        base = re.sub(r"/v1/?$", "", base.rstrip("/"))
        self.url = f"{base}/v1/chat/completions"
        self.client = client or httpx.Client(timeout=httpx.Timeout(600, connect=30))
        self.headers = {"Authorization": f"Bearer {key}"}
        self.mode = MODES[0]
        self._usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        self._usage_lock = threading.Lock()

    def usage_summary(self) -> dict:
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
        json_schema = transform_schema(schema)
        by_name = {tool.name: tool for tool in tools or []}
        messages: list[dict] = [{"role": "user", "content": "\n\n".join(user_blocks)}]
        repaired = False

        for _ in range(MAX_TOOL_TURNS):
            choice = self._send(system, messages, schema.__name__, json_schema, tools or [])
            message = choice.get("message") or {}
            calls = message.get("tool_calls") or []
            if calls:
                messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
                for call in calls:
                    function = call.get("function") or {}
                    name = function.get("name", "")
                    try:
                        tool_input = json.loads(function.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        tool_input, output, is_error = {}, "Tool arguments were not valid JSON.", True
                    else:
                        tool = by_name.get(name)
                        output, is_error = tool.run(tool_input) if tool else (f"Unknown tool {name!r}", True)
                    if trace is not None:
                        trace.append({"tool": name, "input": tool_input, "output": output, "is_error": is_error})
                    messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": output})
                continue

            if message.get("refusal") or choice.get("finish_reason") == "content_filter":
                raise LLMRefusal(f"Model declined the request ({message.get('refusal') or 'content_filter'})")
            if choice.get("finish_reason") == "length":
                raise LLMError("Model output was truncated at the endpoint's output limit")
            text = message.get("content") or ""
            try:
                return schema.model_validate_json(_extract_json(text))
            except ValidationError as exc:
                if repaired:
                    raise LLMError(f"Model reply did not match the {schema.__name__} schema after one retry") from exc
                repaired = True
                problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:10])
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": f"That reply did not match the required JSON schema ({problems}). "
                        "Reply again with only the corrected JSON object.",
                    }
                )
        raise LLMError(f"Model was still calling tools after {MAX_TOOL_TURNS} turns")

    def _body(self, mode: str, system: str, messages: list[dict], name: str, json_schema: dict, tools: list[Tool]) -> dict:
        strict = mode == "schema"
        body: dict = {"model": self.settings.model, "stream": False}
        if strict:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": json_schema, "strict": True},
            }
        else:
            system += _schema_instruction(json_schema)
            if mode == "json":
                body["response_format"] = {"type": "json_object"}
        body["messages"] = [{"role": "system", "content": system}, *messages]
        if tools:
            body["tools"] = []
            for tool in tools:
                definition = tool.definition()
                function = {"name": tool.name, "description": tool.description, "parameters": definition["input_schema"]}
                if strict:
                    function["strict"] = True
                body["tools"].append({"type": "function", "function": function})
        return body

    def _send(self, system: str, messages: list[dict], name: str, json_schema: dict, tools: list[Tool]) -> dict:
        while True:
            mode = self.mode
            response = self._post(self._body(mode, system, messages, name, json_schema, tools))
            if response.status_code == 400 and mode != MODES[-1]:
                # The endpoint rejected this request shape; fall back one step and remember it.
                if self.mode == mode:
                    self.mode = MODES[MODES.index(mode) + 1]
                continue
            break
        if response.status_code in (401, 403):
            raise LLMError(f"The endpoint rejected the API key ({response.status_code}): {response.text[:200]}")
        if response.status_code >= 400:
            raise LLMError(f"Endpoint error {response.status_code}: {response.text[:300]}")
        try:
            payload = response.json()
            choice = payload["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected reply from the endpoint: {response.text[:300]}") from exc
        usage = payload.get("usage") or {}
        with self._usage_lock:
            self._usage["requests"] += 1
            self._usage["input_tokens"] += usage.get("prompt_tokens") or 0
            self._usage["output_tokens"] += usage.get("completion_tokens") or 0
        return choice

    def _post(self, body: dict) -> httpx.Response:
        for attempt in range(4):
            try:
                response = self.client.post(self.url, json=body, headers=self.headers)
            except httpx.TransportError as exc:
                if attempt == 3:
                    raise LLMError(f"Could not reach the endpoint: {type(exc).__name__}") from exc
            else:
                if response.status_code not in _RETRY_STATUS or attempt == 3:
                    return response
            time.sleep(2**attempt)
        raise AssertionError("unreachable")
