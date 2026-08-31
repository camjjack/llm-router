"""A stand-in for ninfer/llama.cpp that behaves like the real thing where it matters.

Critically it *enforces* its own concurrency limit and records the high-water mark
of simultaneous requests, so a router that oversubscribes it is caught rather than
merely suspected. It also models node-local prefix reuse, so affinity can be
measured the same way it is in production: via `cached_tokens`.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from dataclasses import dataclass, field

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

TOKENS_PER_MESSAGE = 10


def _prefix_hashes(messages: list) -> list[str]:
    running = hashlib.blake2b(b"root", digest_size=8).hexdigest()
    out = []
    for message in messages:
        blob = json.dumps(message, sort_keys=True, separators=(",", ":")).encode()
        running = hashlib.blake2b(running.encode() + blob, digest_size=8).hexdigest()
        out.append(running)
    return out


@dataclass
class FakeUpstream:
    name: str = "fake"
    max_concurrency: int = 2
    # ninfer's --max-pending-requests. Requests beyond capacity+pending get 429.
    max_pending: int = 0
    model: str = "test-model"
    # Seconds of simulated work per request.
    latency_s: float = 0.05
    # Emit this many SSE chunks for streaming requests.
    chunks: int = 5
    healthy: bool = True
    # Force failures for retry testing.
    fail_with: int | None = None

    # Each engine publishes context in a different place, and two of them publish a
    # plausible-looking wrong number alongside the right one.
    #   ninfer/vllm -> /v1/models max_model_len          (correct)
    #   llamacpp    -> /props default_generation_settings.n_ctx  (per slot)
    #   lmstudio    -> /api/v0/models loaded_context_length      (as allocated)
    kind: str = "ninfer"
    # The served context. None means the backend advertises nothing at all.
    context_length: int | None = 32768
    # llama.cpp only: the model's trained context, deliberately much larger than the
    # served per-slot figure, so a router reading the wrong field is caught.
    n_ctx_train: int = 131072
    # LM Studio only: what the model *could* do, vs context_length as actually loaded.
    max_context_length: int = 131072
    # LM Studio only: "loaded" or "not-loaded".
    state: str = "loaded"
    # vLLM only: whether /load is enabled (--enable-server-load-tracking).
    load_tracking: bool = True
    # Simulate llama.cpp started with --no-props, forcing the fallback path.
    props_available: bool = True

    active: int = 0
    outstanding: int = 0
    max_observed: int = 0
    total_requests: int = 0
    overload_responses: int = 0
    seen_prefixes: set[str] = field(default_factory=set)
    request_log: list[dict] = field(default_factory=list)
    # Headers and bodies exactly as received, so forwarding rules can be asserted.
    header_log: list[dict] = field(default_factory=list)
    body_log: list[dict] = field(default_factory=list)
    query_log: list[str] = field(default_factory=list)
    # Emit an SSE ping mid-stream; Claude Code needs these relayed unfiltered.
    emit_ping: bool = False

    _port: int = 0
    _server: uvicorn.Server | None = None
    _task: asyncio.Task | None = None

    # ------------------------------------------------------------------ server

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def build_app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/health", self._health, methods=["GET"]),
                Route("/props", self._props, methods=["GET"]),
                Route("/slots", self._slots, methods=["GET"]),
                Route("/load", self._load, methods=["GET"]),
                Route("/v1/models", self._models, methods=["GET"]),
                Route("/api/v0/models", self._lmstudio_models, methods=["GET"]),
                Route("/v1/chat/completions", self._chat, methods=["POST"]),
                Route("/v1/messages", self._messages, methods=["POST"]),
                Route("/v1/messages/count_tokens", self._count_tokens, methods=["POST"]),
            ]
        )

    async def start(self) -> None:
        config = uvicorn.Config(
            self.build_app(), host="127.0.0.1", port=0, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        # Wait for the socket to be bound so callers can read the real port.
        for _ in range(200):
            if self._server.started and self._server.servers:
                sockets = self._server.servers[0].sockets
                if sockets:
                    self._port = sockets[0].getsockname()[1]
                    return
            await asyncio.sleep(0.02)
        raise RuntimeError("fake upstream failed to start")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._task, timeout=5)

    # --------------------------------------------------------------- endpoints

    async def _health(self, request: Request) -> JSONResponse:
        if self.kind == "lmstudio":
            # LM Studio genuinely has no /health endpoint.
            return JSONResponse({"error": "not found"}, status_code=404)
        if not self.healthy:
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return JSONResponse({"status": "ok"})

    async def _props(self, request: Request) -> JSONResponse:
        if self.kind != "llamacpp" or not self.props_available:
            return JSONResponse({"error": "not found"}, status_code=404)
        settings = {"id": 0, "is_processing": False}
        if self.context_length is not None:
            settings["n_ctx"] = self.context_length
        return JSONResponse({"default_generation_settings": settings})

    async def _slots(self, request: Request) -> JSONResponse:
        if self.kind != "llamacpp":
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(
            [
                {"id": i, "is_processing": i < self.active}
                for i in range(self.max_concurrency)
            ]
        )

    async def _load(self, request: Request) -> JSONResponse:
        if self.kind != "vllm" or not self.load_tracking:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse({"server_load": self.active})

    async def _lmstudio_models(self, request: Request) -> JSONResponse:
        if self.kind != "lmstudio":
            return JSONResponse({"error": "not found"}, status_code=404)
        if not self.healthy:
            return JSONResponse({"error": "unavailable"}, status_code=503)
        entry: dict = {
            "id": self.model,
            "object": "model",
            "type": "llm",
            "publisher": "test",
            "arch": "qwen",
            "quantization": "Q4_K_M",
            "state": self.state,
            # The model's ceiling, NOT what was allocated. Reading this is the bug.
            "max_context_length": self.max_context_length,
        }
        if self.context_length is not None:
            entry["loaded_context_length"] = self.context_length
        return JSONResponse({"object": "list", "data": [entry]})

    async def _models(self, request: Request) -> JSONResponse:
        entry: dict = {
            "id": self.model,
            "object": "model",
            "created": 0,
            "owned_by": self.kind,
        }
        if self.kind == "llamacpp":
            # Note: n_ctx_train, NOT the served context. Reading this is the bug.
            entry["meta"] = {"n_ctx_train": self.n_ctx_train}
        elif self.kind == "lmstudio":
            # LM Studio's OpenAI surface carries no context information at all.
            pass
        elif self.context_length is not None:
            entry["max_model_len"] = self.context_length
            entry["meta"] = {"n_ctx": self.context_length}
        return JSONResponse({"object": "list", "data": [entry]})

    def _cached_tokens(self, messages: list) -> int:
        """Longest prefix this host has already seen, in tokens."""
        hashes = _prefix_hashes(messages)
        matched = 0
        for index, digest in enumerate(hashes):
            if digest in self.seen_prefixes:
                matched = index + 1
            else:
                break
        self.seen_prefixes.update(hashes)
        return matched * TOKENS_PER_MESSAGE

    async def _messages(self, request: Request):
        """Anthropic Messages API, as ninfer/llama.cpp/vLLM/LM Studio all expose it."""
        body = await request.json()
        self.header_log.append({k.lower(): v for k, v in request.headers.items()})
        self.body_log.append(body)
        self.query_log.append(str(request.url.query))

        if self.fail_with is not None:
            return JSONResponse(
                {"type": "error",
                 "error": {"type": "api_error", "message": "forced failure"}},
                status_code=self.fail_with,
            )

        if self.outstanding >= self.max_concurrency + self.max_pending:
            self.overload_responses += 1
            # Anthropic's overload status.
            return JSONResponse(
                {"type": "error",
                 "error": {"type": "overloaded_error", "message": "overloaded"}},
                status_code=529,
            )

        self.outstanding += 1
        messages = body.get("messages") or []
        cached = self._cached_tokens(messages)
        # Anthropic's input_tokens EXCLUDES the cached portion.
        fresh = max(1, len(messages) * TOKENS_PER_MESSAGE - cached)
        self.total_requests += 1
        self.request_log.append(
            {"messages": len(messages), "cached_tokens": cached,
             "stream": bool(body.get("stream")), "body_keys": sorted(body)}
        )

        if body.get("stream"):
            return StreamingResponse(
                self._stream_anthropic(fresh, cached), media_type="text/event-stream"
            )
        async with self._slot():
            await asyncio.sleep(self.latency_s)
            return JSONResponse({
                "id": "msg_fake",
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": fresh,
                    "cache_read_input_tokens": cached,
                    "cache_creation_input_tokens": 0,
                    "output_tokens": self.chunks,
                },
            })

    async def _stream_anthropic(self, fresh: int, cached: int):
        async with self._slot():
            def sse(event: str, data: dict) -> bytes:
                return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

            # The input side of usage arrives here...
            yield sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_fake", "type": "message", "role": "assistant",
                    "model": self.model, "content": [],
                    "usage": {
                        "input_tokens": fresh,
                        "cache_read_input_tokens": cached,
                        "cache_creation_input_tokens": 0,
                        "output_tokens": 0,
                    },
                },
            })
            yield sse("content_block_start", {
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""},
            })
            for i in range(self.chunks):
                await asyncio.sleep(self.latency_s / max(1, self.chunks))
                if self.emit_ping and i == 1:
                    # Keep-alive traffic during a thinking pause. Claude Code
                    # aborts a stream that goes silent, so this must be relayed.
                    yield b"event: ping\ndata: {\"type\": \"ping\"}\n\n"
                yield sse("content_block_delta", {
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": f"t{i}"},
                })
            yield sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            # ...and the output side only here.
            yield sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": self.chunks},
            })
            yield sse("message_stop", {"type": "message_stop"})

    async def _count_tokens(self, request: Request) -> JSONResponse:
        body = await request.json()
        self.header_log.append({k.lower(): v for k, v in request.headers.items()})
        messages = body.get("messages") or []
        return JSONResponse({"input_tokens": len(messages) * TOKENS_PER_MESSAGE})

    async def _chat(self, request: Request):
        body = await request.json()
        self.header_log.append({k.lower(): v for k, v in request.headers.items()})
        self.body_log.append(body)
        self.query_log.append(str(request.url.query))

        if self.fail_with is not None:
            return JSONResponse(
                {"error": {"message": "forced failure", "type": "server_error"}},
                status_code=self.fail_with,
            )

        # Mirror ninfer's admission control: capacity + bounded pending, then 429.
        if self.outstanding >= self.max_concurrency + self.max_pending:
            self.overload_responses += 1
            return JSONResponse(
                {
                    "error": {
                        "message": "server overloaded",
                        "type": "server_error",
                        "code": "server_overloaded",
                    }
                },
                status_code=429,
            )

        self.outstanding += 1
        messages = body.get("messages") or []
        cached = self._cached_tokens(messages)
        prompt_tokens = max(1, len(messages) * TOKENS_PER_MESSAGE)
        self.total_requests += 1
        self.request_log.append(
            {
                "messages": len(messages),
                "cached_tokens": cached,
                "stream": bool(body.get("stream")),
                "body_keys": sorted(body),
            }
        )

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": self.chunks,
            "total_tokens": prompt_tokens + self.chunks,
            "prompt_tokens_details": {"cached_tokens": cached},
        }

        if body.get("stream"):
            return StreamingResponse(
                self._stream(usage), media_type="text/event-stream"
            )
        return await self._buffered(usage)

    @contextlib.asynccontextmanager
    async def _slot(self):
        """Occupy an execution slot, recording the concurrency high-water mark."""
        self.active += 1
        self.max_observed = max(self.max_observed, self.active)
        try:
            yield
        finally:
            self.active -= 1
            self.outstanding -= 1

    async def _buffered(self, usage: dict) -> JSONResponse:
        async with self._slot():
            await asyncio.sleep(self.latency_s)
            return JSONResponse(
                {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion",
                    "model": self.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                }
            )

    async def _stream(self, usage: dict):
        async with self._slot():
            for i in range(self.chunks):
                await asyncio.sleep(self.latency_s / max(1, self.chunks))
                chunk = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "model": self.model,
                    "choices": [{"index": 0, "delta": {"content": f"t{i}"}}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            final = {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "model": self.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            }
            yield f"data: {json.dumps(final)}\n\n".encode()
            yield b"data: [DONE]\n\n"
