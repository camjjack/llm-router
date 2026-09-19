"""The two API surfaces this router serves, and everything that differs between them.

Routing, capacity and affinity are identical for both. What differs is the shape
of the request, the shape of an error, where the token counts live, and how deep a
prefix has to be before it identifies a conversation. Keeping those differences
here means the proxy itself never branches on which API it is serving.
"""

from __future__ import annotations

import json
from typing import Any

from starlette.responses import JSONResponse

from .affinity import ANTHROPIC_MIN_AFFINITY_DEPTH, MIN_AFFINITY_DEPTH, session_keys
from .stats import TokenUsage

# Error kinds the router itself raises, mapped per surface to the closest thing
# the client will recognise.
NO_BACKEND = "no_backend"
QUEUE_TIMEOUT = "queue_timeout"
UNREACHABLE = "unreachable"
BAD_REQUEST = "bad_request"
MODEL_NOT_FOUND = "model_not_found"
ALL_FAILED = "all_failed"


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _int_or_none(value: Any) -> int | None:
    """Like _int, but keeps "the backend never said" distinct from "it said zero"."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class Surface:
    """Base class; see OpenAISurface and AnthropicSurface."""

    name: str
    path: str
    # Boundaries shallower than this never pin a session.
    min_affinity_depth: int = MIN_AFFINITY_DEPTH

    def session_keys(self, body: dict) -> list[str]:
        raise NotImplementedError

    def error(self, status: int, message: str, kind: str) -> JSONResponse:
        raise NotImplementedError

    def usage_from_body(self, data: Any) -> TokenUsage | None:
        raise NotImplementedError

    def usage_from_event(self, event: Any) -> dict | None:
        """Pull a raw usage object out of one parsed SSE event, if it has one."""
        raise NotImplementedError

    def usage_from_stream(self, merged: dict) -> TokenUsage | None:
        """Convert the merged usage seen across a stream into token counts."""
        raise NotImplementedError


class OpenAISurface(Surface):
    name = "openai"
    path = "/v1/chat/completions"
    min_affinity_depth = MIN_AFFINITY_DEPTH

    def session_keys(self, body: dict) -> list[str]:
        # messages[0] is the system prompt, so it is already in the hash chain.
        return session_keys(body, root_fields=("model", "tools"))

    def error(self, status: int, message: str, kind: str) -> JSONResponse:
        types = {
            NO_BACKEND: "service_unavailable",
            QUEUE_TIMEOUT: "service_unavailable",
            UNREACHABLE: "service_unavailable",
            ALL_FAILED: "service_unavailable",
            BAD_REQUEST: "invalid_request_error",
            MODEL_NOT_FOUND: "invalid_request_error",
        }
        codes = {
            NO_BACKEND: "no_backend_available",
            QUEUE_TIMEOUT: "queue_timeout",
            UNREACHABLE: "backend_unreachable",
            MODEL_NOT_FOUND: "model_not_found",
        }
        payload: dict[str, Any] = {
            "message": message,
            "type": types.get(kind, "internal_error"),
        }
        if kind in codes:
            payload["code"] = codes[kind]
        return JSONResponse({"error": payload}, status_code=status)

    def usage_from_body(self, data: Any) -> TokenUsage | None:
        if not isinstance(data, dict):
            return None
        usage = data.get("usage")
        return self._counts(usage)

    def usage_from_event(self, event: Any) -> dict | None:
        if isinstance(event, dict) and isinstance(event.get("usage"), dict):
            return event["usage"]
        return None

    def usage_from_stream(self, merged: dict) -> TokenUsage | None:
        return self._counts(merged)

    @staticmethod
    def _counts(usage: Any) -> TokenUsage | None:
        if not isinstance(usage, dict):
            return None
        prompt = usage.get("prompt_tokens")
        if not isinstance(prompt, int) or isinstance(prompt, bool):
            return None
        # Absent entirely on vLLM without --enable-prompt-tokens-details, which
        # means unknown rather than none: see TokenUsage.
        details = usage.get("prompt_tokens_details")
        cached = _int_or_none(details.get("cached_tokens")) if isinstance(details, dict) else None
        output = usage.get("completion_tokens_details")
        reasoning = _int_or_none(output.get("reasoning_tokens")) if isinstance(output, dict) else None
        # OpenAI's prompt_tokens already includes the cached portion.
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=_int(usage.get("completion_tokens")),
            cached_tokens=cached,
            reasoning_tokens=reasoning,
        )


class AnthropicSurface(Surface):
    name = "anthropic"
    path = "/v1/messages"
    # The system prompt is a separate top-level field folded into the root hash,
    # so messages[0] is already the first user message: unique to a conversation,
    # and therefore safe to pin on immediately.
    min_affinity_depth = ANTHROPIC_MIN_AFFINITY_DEPTH

    def session_keys(self, body: dict) -> list[str]:
        return session_keys(body, root_fields=("model", "system", "tools"))

    def error(self, status: int, message: str, kind: str) -> JSONResponse:
        # Anthropic's error taxonomy: overloaded_error carries 529.
        types = {
            NO_BACKEND: "overloaded_error",
            QUEUE_TIMEOUT: "overloaded_error",
            UNREACHABLE: "api_error",
            ALL_FAILED: "api_error",
            BAD_REQUEST: "invalid_request_error",
            MODEL_NOT_FOUND: "not_found_error",
        }
        return JSONResponse(
            {
                "type": "error",
                "error": {"type": types.get(kind, "api_error"), "message": message},
            },
            status_code=status,
        )

    def usage_from_body(self, data: Any) -> TokenUsage | None:
        if not isinstance(data, dict):
            return None
        return self._counts(data.get("usage"))

    def usage_from_event(self, event: Any) -> dict | None:
        """Anthropic spreads usage over two events.

        `message_start` carries the input side (including cache reads and writes),
        `message_delta` carries the running output count. Both are merged before
        being converted.
        """
        if not isinstance(event, dict):
            return None
        message = event.get("message")
        if isinstance(message, dict) and isinstance(message.get("usage"), dict):
            return message["usage"]
        if isinstance(event.get("usage"), dict):
            return event["usage"]
        return None

    def usage_from_stream(self, merged: dict) -> TokenUsage | None:
        return self._counts(merged)

    @staticmethod
    def _counts(usage: Any) -> TokenUsage | None:
        if not isinstance(usage, dict):
            return None
        fresh = usage.get("input_tokens")
        # Neither field present means the backend reports nothing about caching
        # (vLLM's Anthropic surface omits both without --enable-prompt-tokens-details),
        # as opposed to reporting a miss.
        read = _int_or_none(usage.get("cache_read_input_tokens"))
        write = _int_or_none(usage.get("cache_creation_input_tokens"))
        reported = read is not None or write is not None
        cache_read, cache_write = read or 0, write or 0
        if not isinstance(fresh, int) or isinstance(fresh, bool):
            if not reported:
                return None
            fresh = 0
        # Anthropic's input_tokens EXCLUDES cached tokens, unlike OpenAI's
        # prompt_tokens. Sum them so the cache-hit ratio means the same thing on
        # both surfaces.
        return TokenUsage(
            prompt_tokens=fresh + cache_read + cache_write,
            completion_tokens=_int(usage.get("output_tokens")),
            cached_tokens=cache_read if reported else None,
        )


OPENAI = OpenAISurface()
ANTHROPIC = AnthropicSurface()

MAX_SSE_LINE_BYTES = 1 << 20


class StreamTap:
    """Observes a pass-through SSE stream to recover its token usage.

    The bytes are forwarded untouched -- this only watches them go by, so the
    client receives exactly what the backend sent. Usage is *merged* rather than
    replaced, because Anthropic reports the input and output halves in different
    events.
    """

    def __init__(self, surface: Surface) -> None:
        self._surface = surface
        self._buffer = b""
        self._merged: dict = {}
        self.chunks = 0

    @property
    def usage(self) -> TokenUsage | None:
        return self._surface.usage_from_stream(self._merged) if self._merged else None

    def feed(self, chunk: bytes) -> None:
        self.chunks += 1
        if b"usage" not in chunk and b"usage" not in self._buffer:
            # Fast path: keep a short tail in case a line straddles chunks.
            self._buffer = (self._buffer + chunk)[-256:]
            return

        self._buffer += chunk
        if len(self._buffer) > MAX_SSE_LINE_BYTES:
            self._buffer = self._buffer[-MAX_SSE_LINE_BYTES:]

        lines = self._buffer.split(b"\n")
        self._buffer = lines.pop()
        for line in lines:
            line = line.strip()
            if not line.startswith(b"data:") or b'"usage"' not in line:
                continue
            payload = line[5:].strip()
            if payload in (b"", b"[DONE]"):
                continue
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            usage = self._surface.usage_from_event(event)
            if isinstance(usage, dict):
                self._merged.update(usage)
