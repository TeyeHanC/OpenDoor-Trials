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
        self._ask_for_usage = True
        # Some gateways cannot combine a model's reasoning with tool calls on this API and say so
        # in a 400; reasoning is then switched off for requests that offer tools (and only those).
        self._no_reasoning_with_tools = False
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
        body: dict = {"model": self.settings.model}
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
            body = self._body(mode, system, messages, name, json_schema, tools)
            # Streamed: a long answer sent in one piece can exceed a gateway's response timeout.
            body["stream"] = True
            if self._ask_for_usage:
                body["stream_options"] = {"include_usage": True}
            if tools and self._no_reasoning_with_tools:
                body["reasoning_effort"] = "none"
            status, text, choice, usage = self._post(body)
            if status == 400 and tools and not self._no_reasoning_with_tools and "reasoning" in text.lower():
                self._no_reasoning_with_tools = True
                continue
            if status == 400 and self._ask_for_usage:
                # Possibly the usage option itself; drop it before giving up on the request shape.
                self._ask_for_usage = False
                continue
            if status == 400 and mode != MODES[-1]:
                # The endpoint rejected this request shape; fall back one step and remember it.
                if self.mode == mode:
                    self.mode = MODES[MODES.index(mode) + 1]
                continue
            break
        if status in (401, 403):
            raise LLMError(f"The endpoint rejected the API key ({status}): {text[:200]}")
        if status >= 400:
            if text.lstrip().startswith("<"):
                text = "the gateway returned an HTML error page" + (" (response timeout)" if status == 524 else "")
            raise LLMError(f"Endpoint error {status}: {text[:300]}")
        if choice is None:
            raise LLMError(f"Unexpected reply from the endpoint: {text[:300]}")
        with self._usage_lock:
            self._usage["requests"] += 1
            self._usage["input_tokens"] += usage.get("prompt_tokens") or 0
            self._usage["output_tokens"] += usage.get("completion_tokens") or 0
        return choice

    def _post(self, body: dict) -> tuple[int, str, dict | None, dict]:
        """(status, raw text for error messages, the first choice, usage). Retries transient failures."""
        for attempt in range(4):
            try:
                with self.client.stream("POST", self.url, json=body, headers=self.headers) as response:
                    if response.status_code in _RETRY_STATUS and attempt < 3:
                        response.read()
                    elif response.status_code != 200:
                        return response.status_code, response.read().decode("utf-8", "replace"), None, {}
                    elif "text/event-stream" in response.headers.get("content-type", ""):
                        choice, usage = _read_stream(response.iter_lines())
                        return 200, "", choice, usage
                    else:  # the endpoint ignored stream=true and answered in one piece
                        text = response.read().decode("utf-8", "replace")
                        try:
                            payload = json.loads(text)
                            return 200, text, payload["choices"][0], payload.get("usage") or {}
                        except (ValueError, KeyError, IndexError, TypeError):
                            return 200, text, None, {}
            except httpx.TransportError as exc:
                if attempt == 3:
                    raise LLMError(f"Could not reach the endpoint: {type(exc).__name__}") from exc
            time.sleep(2**attempt)
        raise AssertionError("unreachable")


def _read_stream(lines) -> tuple[dict, dict]:
    """Assemble a streamed chat completion into the same shape as a non-streamed choice."""
    content: list[str] = []
    refusal: list[str] = []
    calls: dict[int, dict] = {}
    finish, usage = None, {}
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if chunk.get("error"):
            raise LLMError(f"Endpoint error mid-stream: {json.dumps(chunk['error'], ensure_ascii=False)[:300]}")
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("refusal"):
                refusal.append(delta["refusal"])
            for part in delta.get("tool_calls") or []:
                call = calls.setdefault(part.get("index", 0), {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                call["id"] = part.get("id") or call["id"]
                function = part.get("function") or {}
                call["function"]["name"] += function.get("name") or ""
                call["function"]["arguments"] += function.get("arguments") or ""
    if finish is None and not content and not calls:
        raise LLMError("The endpoint closed the stream without sending an answer")
    message: dict = {"role": "assistant", "content": "".join(content) or None}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    if refusal:
        message["refusal"] = "".join(refusal)
    return {"message": message, "finish_reason": finish}, usage
