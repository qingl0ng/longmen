"""HTTP client for vLLM OpenAI-compatible API with streaming."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

import httpx
import structlog

log = structlog.get_logger(__name__)


@dataclass
class StreamChunk:
    delta_text: str | None = None
    tool_call_deltas: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


class VLLMClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        timeout: float | httpx.Timeout = 3600.0,
        first_token_timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.first_token_timeout = first_token_timeout
        # Always build an explicit httpx.Timeout with read=None for streaming.
        # read=None prevents false ReadTimeouts during long prompt-cache management
        # on the LLM server (can take 90+ seconds before the first token arrives).
        # Note: isinstance check must include int — config values are int, not float.
        if isinstance(timeout, int | float):
            self.timeout = httpx.Timeout(
                connect=5.0,
                read=None,   # no read timeout: LLM may be silent for a long time
                write=None,
                pool=float(timeout),
            )
        else:
            self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def health(self) -> bool:
        """True if the model server is reachable and ready. Never raises.

        Probes /health, which returns 200 only when the model is loaded. Both
        llama.cpp and vLLM expose it with compatible readiness semantics:
        llama.cpp returns 503 while loading and 200 once ready; vLLM's /health
        is unreachable (connection refused) during startup and returns 200 only
        once serving. In both cases status_code == 200 means "ready," and every
        non-200 / unreachable case maps to False — so /health is a safe common
        denominator across the two engines and stays a single hardcoded constant.

        Uses a short read timeout (this probe must be fast and must not inherit
        the streaming read=None), but keeps connect=5.0 to match stream(): a
        probe with a tighter connect budget than the real request could report
        "down" for a server the stream would have reached.
        """
        url = f"{self.base_url}/health"
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=2.0, write=2.0, pool=2.0)
            ) as client:
                resp = await client.get(url, headers=self._headers())
                return resp.status_code == 200
        # NOTE: load-bearing `except Exception` (not BaseException). CancelledError
        # is a BaseException, so an abort that arrives during the probe propagates
        # out of health() to the wait loop's CancelledError path instead of being
        # swallowed here (which would make abort-during-probe hang).
        except Exception:
            return False

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = tools
        return body

    def _parse_chunk(self, raw: dict[str, Any]) -> StreamChunk:
        chunk = StreamChunk()
        # Top-level usage (some vLLM versions put it here)
        if "usage" in raw and raw["usage"]:
            chunk.usage = raw["usage"]

        choices = raw.get("choices", [])
        if not choices:
            return chunk

        choice = choices[0]
        chunk.finish_reason = choice.get("finish_reason")

        # Per-choice usage
        if "usage" in choice and choice["usage"]:
            chunk.usage = choice["usage"]

        delta = choice.get("delta", {})
        if "content" in delta and delta["content"] is not None:
            chunk.delta_text = delta["content"]

        if "tool_calls" in delta and delta["tool_calls"]:
            chunk.tool_call_deltas = delta["tool_calls"]

        return chunk

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chunks, bounding only the time-to-first-token.

        read=None is kept for inter-chunk gaps (long generations are legitimate),
        but the wait for the *first* SSE frame is bounded by first_token_timeout.
        connect / raise_for_status / SSE parsing all happen lazily on the first
        advance of the inner generator, so a ConnectError / HTTPStatusError
        surfaces *through* the wait_for and propagates to agent_loop normally.
        """
        inner = self._stream_impl(messages, tools=tools, **kwargs)
        try:
            # "First chunk" is the first yielded StreamChunk of any kind (incl. a
            # role-only frame): it proves streaming began. We do NOT bound "first
            # chunk with content," which would wrongly kill a model that pauses
            # between its role frame and first token.
            first = await asyncio.wait_for(inner.__anext__(), timeout=self.first_token_timeout)
        except TimeoutError:
            # Land in agent_loop's existing ReadTimeout branch (broadened to cover
            # the warming/stalled case). A dedicated exception/code is not needed.
            await inner.aclose()
            raise httpx.ReadTimeout("no first token within vllm_first_token") from None
        except StopAsyncIteration:
            return
        yield first
        async for chunk in inner:  # subsequent chunks: read=None, unbounded
            yield chunk

    async def _stream_impl(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk]:
        body = self._build_request(messages, tools=tools, **kwargs)
        url = f"{self.base_url}/v1/chat/completions"

        async with (
            httpx.AsyncClient(timeout=self.timeout) as client,
            client.stream("POST", url, json=body, headers=self._headers()) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    try:
                        raw = json.loads(data)
                        yield self._parse_chunk(raw)
                    except json.JSONDecodeError as e:
                        log.warning("vllm_client.parse_error", line=line, error=str(e))
