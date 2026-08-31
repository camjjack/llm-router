"""Starlette app: OpenAI-compatible chat completions with sticky, capacity-aware routing."""

from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import surfaces
from .affinity import SessionMap, explicit_session_id
from .backend import BackendClients
from .config import Config
from .scheduler import Lease, NoBackendError, QueueTimeout, Scheduler
from .stats import RouterStats
from .surfaces import ANTHROPIC, OPENAI, StreamTap, Surface

log = logging.getLogger("llm_router.proxy")

# Upstream statuses worth trying on a different backend.
RETRYABLE_STATUSES = {429, 502, 503, 504, 529}

# Hop-by-hop headers that must not be forwarded in either direction.
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "content-encoding",
    "host",
}

# Headers forwarded to the backend verbatim. The Claude Code gateway contract is
# explicit that `anthropic-*` must be treated as an OPEN list: capabilities arrive
# as new headers, and a gateway pinned to today's names breaks the release that
# introduces the next one. The client's own credentials are deliberately not
# forwarded -- each backend authenticates with its own configured key.
FORWARDED_HEADER_PREFIXES = ("anthropic-",)


class Router:
    def __init__(self, config: Config):
        self.config = config
        self.scheduler = Scheduler(config.backends, config.health)
        self.clients = BackendClients(config)
        self.stats = RouterStats()
        self.sessions = SessionMap(
            ttl_s=config.routing.session_ttl_s,
            max_entries=config.routing.max_sessions,
            depth=config.routing.affinity_depth,
        )
        for backend in config.backends:
            self.stats.backend(backend.name)
        # Models we have already complained about having mismatched contexts.
        self._context_warned: set[str] = set()

    async def start(self) -> None:
        await self.scheduler.start()
        await self.clients.start_probing(self.scheduler)

    async def stop(self) -> None:
        await self.scheduler.stop()
        await self.clients.aclose()

    # ------------------------------------------------------------------ routing

    async def chat_completions(self, request: Request) -> Response:
        return await self._handle(request, OPENAI)

    async def messages(self, request: Request) -> Response:
        """Anthropic Messages API -- what Claude Code speaks.

        Note the route matches on path only: Claude Code posts inference requests
        to `/v1/messages?beta=true`, and the query string is forwarded upstream
        rather than matched on.
        """
        return await self._handle(request, ANTHROPIC)

    def resolve_model(self, model: str) -> str:
        """Apply configured aliases, falling back to a catch-all if one is set.

        Claude Code sends whatever model name it was configured with, and also
        issues background requests for its small/fast model. Without an alias for
        those, the background traffic 404s.
        """
        aliases = self.config.model_aliases
        if not aliases:
            return model
        return aliases.get(model) or aliases.get("*") or model

    async def _handle(self, request: Request, surface: Surface) -> Response:
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            return surface.error(400, "request body must be valid JSON", surfaces.BAD_REQUEST)
        if not isinstance(body, dict):
            return surface.error(400, "request body must be a JSON object", surfaces.BAD_REQUEST)

        requested = body.get("model")
        if not isinstance(requested, str) or not requested:
            return surface.error(400, "'model' is required", surfaces.BAD_REQUEST)

        model = self.resolve_model(requested)
        if not self.config.backends_for(model):
            return surface.error(
                404,
                f"model '{requested}' is not served by any configured backend "
                f"(known: {', '.join(self.config.all_models)})",
                surfaces.MODEL_NOT_FOUND,
            )

        streaming = bool(body.get("stream"))
        routing = self.config.routing

        explicit = explicit_session_id(body, request.headers)
        keys = [explicit] if explicit else surface.session_keys(body)
        min_depth = 1 if explicit else surface.min_affinity_depth
        if surface is OPENAI:
            # `session_id` is ours, not part of the OpenAI schema; never forward it.
            # The Anthropic body is passed through untouched, since capability
            # headers pair with body fields and breaking a pair is a hard 400.
            body.pop("session_id", None)

        preferred = self.sessions.lookup(keys, min_depth) if keys else None
        if keys:
            if preferred:
                self.stats.affinity_hits += 1
            else:
                self.stats.affinity_misses += 1

        if (
            surface is OPENAI
            and streaming
            and routing.inject_usage
            and "stream_options" not in body
        ):
            body["stream_options"] = {"include_usage": True}

        excluded: set[str] = set()
        last_error: Response | None = None

        for attempt in range(routing.max_retries + 1):
            try:
                lease = await self.scheduler.acquire(
                    model,
                    preferred=preferred,
                    affinity_wait_s=routing.affinity_wait_ms / 1000.0,
                    timeout_s=routing.queue_timeout_s,
                    exclude=frozenset(excluded),
                    unavailable_grace_s=routing.unavailable_grace_s,
                )
            except NoBackendError:
                if last_error is not None:
                    return last_error
                self.stats.rejected_no_backend += 1
                return surface.error(
                    503,
                    f"no healthy backend available for model '{model}'",
                    surfaces.NO_BACKEND,
                )
            except QueueTimeout:
                self.stats.queue_timeouts += 1
                return surface.error(
                    503,
                    f"timed out waiting {routing.queue_timeout_s:.0f}s for a free slot",
                    surfaces.QUEUE_TIMEOUT,
                )

            self._record_lease(lease, pinned=preferred is not None)
            if attempt:
                self.stats.retries += 1

            outcome, response = await self._dispatch(
                lease, body, streaming, keys, surface, request, min_depth, model
            )
            if outcome == "ok":
                return response
            # Retryable: the lease is already released. Try elsewhere.
            last_error = response
            excluded.add(lease.name)
            preferred = None

        return last_error or surface.error(502, "all backends failed", surfaces.ALL_FAILED)

    async def count_tokens(self, request: Request) -> Response:
        """Anthropic token counting.

        Deliberately does not take a capacity slot: it is a cheap, non-generating
        call, and holding a generation slot for it would let Claude Code's
        bookkeeping block real work. It still prefers the session's pinned host,
        but never creates a pin of its own.
        """
        surface = ANTHROPIC
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            return surface.error(400, "request body must be valid JSON", surfaces.BAD_REQUEST)
        if not isinstance(body, dict):
            return surface.error(400, "request body must be a JSON object", surfaces.BAD_REQUEST)

        requested = body.get("model")
        if not isinstance(requested, str) or not requested:
            return surface.error(400, "'model' is required", surfaces.BAD_REQUEST)
        model = self.resolve_model(requested)

        candidates = [
            b
            for b in self.config.backends_for(model)
            if self.scheduler.backends[b.name].healthy
        ]
        if not candidates:
            return surface.error(
                503,
                f"no healthy backend available for model '{model}'",
                surfaces.NO_BACKEND,
            )

        keys = surface.session_keys(body)
        pinned = self.sessions.lookup(keys, surface.min_affinity_depth) if keys else None
        backend = next((b for b in candidates if b.name == pinned), None) or min(
            candidates, key=lambda b: self.scheduler.backends[b.name].inflight
        )

        payload = dict(body)
        if backend.upstream_model:
            payload["model"] = backend.upstream_model
        else:
            payload["model"] = model

        headers = self._upstream_headers(backend, request, streaming=False)
        try:
            upstream = await self.clients.client(backend.name).post(
                "/v1/messages/count_tokens",
                json=payload,
                headers=headers,
                params=dict(request.query_params),
            )
        except (httpx.HTTPError, OSError) as exc:
            return surface.error(
                502,
                f"backend '{backend.name}' unreachable: {exc}",
                surfaces.UNREACHABLE,
            )
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
            headers={"x-llm-router-backend": backend.name},
        )

    def _record_lease(self, lease: Lease, pinned: bool) -> None:
        stats = self.stats
        bstats = stats.backend(lease.name)
        bstats.requests += 1
        if pinned:
            if lease.affinity_honored:
                stats.affinity_honored += 1
            else:
                stats.affinity_spills += 1
        if lease.spilled:
            bstats.spilled_in += 1
        if lease.queue_wait_s > 0.001:
            stats.queued += 1
            stats.total_queue_wait_s += lease.queue_wait_s

    # ---------------------------------------------------------------- dispatch

    def _upstream_headers(self, backend, request: Request, streaming: bool) -> dict:
        """Headers for the upstream call.

        The client's credentials are consumed, not forwarded: each backend
        authenticates with its own configured key. `anthropic-*` headers are
        forwarded verbatim as an open list, because Claude Code capabilities pair
        a beta header with body fields and dropping either half breaks them.
        """
        headers = self.clients.headers_for(backend)
        headers["content-type"] = "application/json"
        headers["accept"] = "text/event-stream" if streaming else "application/json"
        for name, value in request.headers.items():
            lowered = name.lower()
            if lowered in HOP_BY_HOP:
                continue
            if lowered.startswith(FORWARDED_HEADER_PREFIXES):
                headers[lowered] = value
        return headers

    async def _dispatch(
        self,
        lease: Lease,
        body: dict,
        streaming: bool,
        keys: list[str],
        surface: Surface,
        request: Request,
        min_depth: int,
        model: str,
    ) -> tuple[str, Response]:
        backend = lease.backend.config
        client = self.clients.client(backend.name)

        # The only body change is the model name, which a gateway is expected to
        # rewrite -- an aliased name must not reach the backend. Everything else
        # passes through untouched.
        payload = dict(body)
        payload["model"] = backend.upstream_model or model

        headers = self._upstream_headers(backend, request, streaming)
        # Claude Code posts inference to /v1/messages?beta=true; forward the query
        # string rather than matching on it.
        params = dict(request.query_params)

        started = time.monotonic()
        try:
            if streaming:
                return await self._dispatch_streaming(
                    lease, client, headers, payload, keys, started, surface,
                    params, min_depth,
                )
            return await self._dispatch_buffered(
                lease, client, headers, payload, keys, started, surface,
                params, min_depth,
            )
        except (httpx.HTTPError, OSError) as exc:
            lease.release()
            self.scheduler.note_failure(backend.name)
            self.stats.backend(backend.name).errors += 1
            log.warning("backend %s transport error: %r", backend.name, exc)
            return (
                "retry",
                surface.error(
                    502,
                    f"backend '{backend.name}' unreachable: {exc}",
                    surfaces.UNREACHABLE,
                ),
            )

    async def _dispatch_buffered(
        self,
        lease: Lease,
        client: httpx.AsyncClient,
        headers: dict,
        payload: dict,
        keys: list[str],
        started: float,
        surface: Surface,
        params: dict,
        min_depth: int,
    ) -> tuple[str, Response]:
        name = lease.name
        upstream = await client.post(
            surface.path, json=payload, headers=headers, params=params
        )

        # Non-streaming: the slot is free the moment the body is in hand.
        if upstream.status_code != 200:
            lease.release()
            return self._upstream_error(name, upstream.status_code, upstream.content, surface)

        lease.release()
        self.scheduler.note_success(name)

        bstats = self.stats.backend(name)
        bstats.completed += 1
        elapsed = time.monotonic() - started
        bstats.ttft_s.append(elapsed)

        try:
            data = upstream.json()
        except ValueError:
            data = None
        usage = surface.usage_from_body(data)
        if usage is not None:
            bstats.record_usage(usage)
            if usage.completion_tokens > 0 and elapsed > 0:
                bstats.tokens_per_s.append(usage.completion_tokens / elapsed)

        if keys:
            self.sessions.assign(keys, name, min_depth)

        return ("ok", Response(
            content=upstream.content,
            status_code=200,
            media_type=upstream.headers.get("content-type", "application/json"),
            headers={"x-llm-router-backend": name},
        ))

    async def _dispatch_streaming(
        self,
        lease: Lease,
        client: httpx.AsyncClient,
        headers: dict,
        payload: dict,
        keys: list[str],
        started: float,
        surface: Surface,
        params: dict,
        min_depth: int,
    ) -> tuple[str, Response]:
        name = lease.name
        request = client.build_request(
            "POST", surface.path, json=payload, headers=headers, params=params
        )
        # send(stream=True) returns once headers are in, so we can still fail over
        # to another backend before any bytes reach the client.
        upstream = await client.send(request, stream=True)

        if upstream.status_code != 200:
            content = await upstream.aread()
            await upstream.aclose()
            lease.release()
            return self._upstream_error(name, upstream.status_code, content, surface)

        self.scheduler.note_success(name)
        if keys:
            self.sessions.assign(keys, name, min_depth)

        return ("ok", StreamingResponse(
            self._stream_body(lease, upstream, started, surface),
            status_code=200,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            headers={
                "x-llm-router-backend": name,
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
            },
        ))

    async def _stream_body(
        self, lease: Lease, upstream: httpx.Response, started: float, surface: Surface
    ):
        name = lease.name
        bstats = self.stats.backend(name)
        tap = StreamTap(surface)
        first_byte_at: float | None = None
        failed = False

        try:
            # aiter_raw forwards bytes exactly as they arrive, including SSE ping
            # events and comment lines. Claude Code counts every byte and aborts a
            # stream that goes silent for 300s, so pings must not be filtered.
            async for chunk in upstream.aiter_raw():
                if first_byte_at is None:
                    first_byte_at = time.monotonic()
                    bstats.ttft_s.append(first_byte_at - started)
                tap.feed(chunk)
                yield chunk
        except (httpx.HTTPError, OSError) as exc:
            # Mid-stream failure: the client has bytes already, so we cannot retry.
            failed = True
            log.warning("backend %s stream aborted: %r", name, exc)
            self.scheduler.note_failure(name)
        finally:
            # Runs on normal completion, upstream error, and client disconnect
            # alike. A leaked slot here is precisely the bug this router exists to
            # avoid.
            await upstream.aclose()
            lease.release()

            if failed:
                bstats.errors += 1
            else:
                bstats.completed += 1
            usage = tap.usage
            if usage is not None:
                bstats.record_usage(usage)
                if usage.completion_tokens > 0 and first_byte_at is not None:
                    decode_s = time.monotonic() - first_byte_at
                    if decode_s > 0:
                        bstats.tokens_per_s.append(usage.completion_tokens / decode_s)

    def _upstream_error(
        self, name: str, status: int, content: bytes, surface: Surface
    ) -> tuple[str, Response]:
        bstats = self.stats.backend(name)
        code = None
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                err = parsed.get("error")
                if isinstance(err, dict):
                    # OpenAI puts the machine-readable tag in `code`; Anthropic
                    # puts it in `type` (e.g. overloaded_error).
                    code = err.get("code") or err.get("type")
        except ValueError:
            parsed = None

        retryable = status in RETRYABLE_STATUSES
        if status in (429, 529) or code in ("server_overloaded", "overloaded_error"):
            # We gate on capacity, so the backend should never be full. If it is, our
            # configured capacity is too high or something else is sharing the host.
            bstats.overloaded += 1
            log.error(
                "backend %s returned 429 %s despite our capacity gate -- lower its "
                "configured capacity to match --max-concurrency, or check for another "
                "client using the same host",
                name,
                code or "",
            )
        bstats.errors += 1
        if retryable:
            self.scheduler.note_failure(name)
        else:
            # A 4xx is the client's fault; it says nothing about backend health.
            self.scheduler.note_success(name)

        # The backend's error body is relayed byte-for-byte. Claude Code's
        # capability-retry logic matches on the upstream's own error wording, so
        # wrapping it in our envelope would break its recovery path.
        response = Response(
            content=content,
            status_code=status,
            media_type="application/json",
            headers={"x-llm-router-backend": name},
        )
        return ("retry" if retryable else "ok", response)

    # ----------------------------------------------------------------- endpoints

    def context_for(self, model: str) -> int | None:
        """The context a client can safely assume for a model.

        The *minimum* across every backend serving it, because a request may land on
        any of them -- advertising the largest would invite prompts that fail on the
        smallest. Backends whose context is still unknown are skipped rather than
        guessed at, and unhealthy ones still count: they will come back.
        """
        known = {
            b.name: self.clients.context_length.get(b.name)
            for b in self.config.backends_for(model)
        }
        values = [v for v in known.values() if isinstance(v, int) and v > 0]
        if not values:
            return None

        if len(set(values)) > 1 and model not in self._context_warned:
            self._context_warned.add(model)
            log.warning(
                "backends for model '%s' disagree on context window (%s); "
                "advertising the smallest (%d) so requests fit wherever they land",
                model,
                ", ".join(f"{n}={v}" for n, v in known.items() if v),
                min(values),
            )
        return min(values)

    async def models(self, request: Request) -> JSONResponse:
        now = int(time.time())
        data = []
        for model in self.config.all_models:
            entry: dict[str, Any] = {
                "id": model,
                "object": "model",
                "created": now,
                "owned_by": "llm-router",
                # Claude Code's gateway model discovery reads id and display_name.
                "display_name": model,
            }
            context = self.context_for(model)
            if context is not None:
                # Three spellings of the same number, because clients disagree:
                # `max_model_len` is the vLLM/ninfer convention, `meta.n_ctx` the
                # llama.cpp one, `context_length` the OpenRouter/models.dev one.
                # Omitted entirely when unknown -- better absent than invented.
                entry["context_length"] = context
                entry["max_model_len"] = context
                entry["meta"] = {"n_ctx": context}
            data.append(entry)
        return JSONResponse({"object": "list", "data": data})

    async def hello(self, request: Request) -> Response:
        return Response(status_code=200)

    async def health(self, request: Request) -> JSONResponse:
        healthy = [s.name for s in self.scheduler.backends.values() if s.healthy]
        ok = bool(healthy)
        return JSONResponse(
            {
                "status": "ok" if ok else "unavailable",
                "healthy_backends": healthy,
                "total_backends": len(self.scheduler.backends),
            },
            status_code=200 if ok else 503,
        )

    async def stats_endpoint(self, request: Request) -> JSONResponse:
        return JSONResponse(self.snapshot())

    def snapshot(self) -> dict:
        sched = self.scheduler
        pins = self.sessions.backend_pin_counts()
        backends = []
        for name, state in sched.backends.items():
            entry = self.stats.backend(name).snapshot(
                inflight=state.inflight,
                capacity=state.capacity,
                healthy=state.healthy,
            )
            entry["kind"] = state.config.kind
            entry["url"] = state.config.url
            entry["models"] = list(state.config.models)
            entry["context_length"] = self.clients.context_length.get(name)
            entry["pinned_sessions"] = pins.get(name, 0)
            entry["observed_busy"] = self.clients.observed_busy.get(name)
            entry["cooling_down"] = state.cooldown_until > time.monotonic()
            backends.append(entry)

        return {
            "router": self.stats.snapshot()
            | {
                "queue_depth": sched.queue_depth,
                "waiting_on_affinity": sched.waiting_on_affinity,
                "tracked_sessions": len(self.sessions),
            },
            "backends": backends,
            "models": {
                model: {"context_length": self.context_for(model)}
                for model in self.config.all_models
            },
        }


def create_app(config: Config, router: Router | None = None) -> Starlette:
    router = router or Router(config)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        await router.start()
        try:
            yield
        finally:
            await router.stop()

    app = Starlette(
        routes=[
            Route("/v1/chat/completions", router.chat_completions, methods=["POST"]),
            # Matches on path, so /v1/messages?beta=true lands here too.
            Route("/v1/messages", router.messages, methods=["POST"]),
            Route("/v1/messages/count_tokens", router.count_tokens, methods=["POST"]),
            Route("/v1/models", router.models, methods=["GET"]),
            # Claude Code's connection-warming probe. Answering it keeps the log
            # clean; it is best-effort and safe to reject, but cheap to serve.
            Route("/api/hello", router.hello, methods=["GET", "HEAD"]),
            Route("/health", router.health, methods=["GET"]),
            Route("/stats", router.stats_endpoint, methods=["GET"]),
        ],
        lifespan=lifespan,
    )
    app.state.router = router
    return app
