"""Starlette app: OpenAI-compatible chat completions with sticky, capacity-aware routing."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import dashboard, surfaces
from .affinity import ExplicitSession, SessionMap, explicit_session
from .backend import BackendClients
from .config import Config
from .scheduler import (
    BackendDiff,
    BackendState,
    Lease,
    NoBackendError,
    QueueTimeout,
    Scheduler,
)
from .stats import RouterStats
from .surfaces import ANTHROPIC, OPENAI, StreamTap, Surface
from .tracking import CANCELLED, ERROR, OK, SessionTracker, Tracked

log = logging.getLogger("llm_router.proxy")

# How many distinct requested model names to remember for the stranded-model
# warning on reload. Names are client-supplied, so this has to be bounded.
MAX_REMEMBERED_MODEL_NAMES = 256

# Re-dos caused by a reload landing mid-request, as opposed to a backend failing.
# It would take a fresh reload inside each attempt to use these up.
MAX_RELOAD_REROUTES = 3

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


async def _until_client_leaves(receive) -> None:
    """Return once the client has hung up.

    Only for after the request body has been read: from then on, the one thing
    receive() can still deliver is word that the client has gone.
    """
    try:
        while (await receive())["type"] != "http.disconnect":
            pass
    except Exception:  # noqa: BLE001 -- never let a failure to watch look like a hang-up
        # Can't tell. Carry on as though the client were still there.
        await asyncio.Event().wait()


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
        self.tracker = SessionTracker(config.tracking)
        self.scheduler.on_drained = self._on_drained
        # Models we have already complained about having mismatched contexts.
        self._context_warned: set[str] = set()
        # Requested model name -> when a request for it was last routed. Lets a
        # reload notice it is about to strand a name clients are still using.
        self._requested_seen: OrderedDict[str, float] = OrderedDict()

        # Reload bookkeeping, surfaced in /stats and the dashboard.
        self.config_generation = 1
        self.config_loaded_at = time.time()
        self.config_error: str | None = None
        # How the config file is being watched ("inotify", "polling"), if at all.
        self.config_watch: str | None = None
        self._applying = False

    async def start(self) -> None:
        await self.scheduler.start()
        await self.clients.start_probing(self.scheduler)

    async def stop(self) -> None:
        await self.scheduler.stop()
        await self.clients.aclose()

    # ------------------------------------------------------------------ reload

    def apply_config(self, new: Config) -> BackendDiff:
        """Switch to a new config without disturbing anything already running.

        Deliberately contains no `await`: the scheduler, the clients, the session
        map and `self.config` all move to the new config in one step, so no request
        can see some of it and not the rest. Requests already running finish where
        they are, even on a backend that has just been removed.
        """
        old = self.config
        self._applying = True
        try:
            diff = self.scheduler.reconfigure(new.backends, new.health)
        finally:
            self._applying = False
        self.clients.reconfigure(new, diff)

        routing = new.routing
        self.sessions.reconfigure(
            routing.session_ttl_s, routing.max_sessions, routing.affinity_depth
        )
        dropped = self.sessions.forget_backends({*diff.removed, *diff.replaced})
        self.tracker.reconfigure(new.tracking)

        # A replaced backend is a different machine under the same name; its
        # predecessor's latency and cache figures say nothing about it. Likewise a
        # name that is back after being removed starts from zero.
        for name in (*diff.replaced, *diff.added):
            self.stats.backends.pop(name, None)
        for backend in new.backends:
            self.stats.backend(backend.name)

        self.config = new
        self._context_warned.clear()
        self.config_generation += 1
        self.config_loaded_at = time.time()
        self.config_error = None

        log.info(
            "config reloaded (generation %d): %s",
            self.config_generation,
            self._describe_reload(old, new, diff, dropped),
        )
        self._warn_stranded_models()
        return diff

    def config_failed(self, message: str) -> None:
        """A reload was rejected; the running config stays in force."""
        self.config_error = message

    def _describe_reload(
        self, old: Config, new: Config, diff: BackendDiff, dropped_pins: int
    ) -> str:
        parts: list[str] = []
        if diff.added:
            parts.append(f"added {', '.join(diff.added)}")
        draining = {s.name: s.inflight for s in self.scheduler.draining}
        for label, names in (("removed", diff.removed), ("re-pointed", diff.replaced)):
            if names:
                parts.append(
                    f"{label} "
                    + ", ".join(
                        f"{n} (draining {draining[n]} in flight)" if n in draining else n
                        for n in names
                    )
                )
        for name, changed in diff.updated.items():
            if "capacity" in changed:
                before = next((b.capacity for b in old.backends if b.name == name), "?")
                after = next(b.capacity for b in new.backends if b.name == name)
                changed = [
                    f"capacity {before}->{after}" if c == "capacity" else c
                    for c in changed
                ]
            parts.append(f"updated {name} ({', '.join(changed)})")
        for section in ("model_aliases", "routing", "health", "timeouts", "tracking"):
            if getattr(old, section) != getattr(new, section):
                parts.append(f"{section} changed")
        if dropped_pins:
            parts.append(f"dropped {dropped_pins} session pins")
        return "; ".join(parts) or "no changes"

    def _warn_stranded_models(self) -> None:
        """Say so when a model name clients were recently using stops resolving.

        The usual way a model swap breaks a running client: Claude Code or opencode
        keeps sending the name it was started with, and after the edit that name
        404s. An alias from the old name to the new model keeps it working.
        """
        horizon = time.monotonic() - self.config.routing.session_ttl_s
        for requested, seen in self._requested_seen.items():
            if seen < horizon:
                continue
            if self.config.backends_for(self.resolve_model(requested)):
                continue
            log.warning(
                "clients requested model '%s' within the last %ds and it no longer "
                "resolves to any backend, so their next requests will 404. To keep "
                "them working, add it to model_aliases, e.g.  model_aliases: "
                "{\"%s\": <new model>}",
                requested,
                self.config.routing.session_ttl_s,
                requested,
            )

    def _on_drained(self, state: BackendState) -> None:
        if state.name not in self.scheduler.backends:
            # Removed outright rather than re-pointed: nothing left to report on.
            self.stats.backends.pop(state.name, None)
        if not self._applying:
            # An idle backend drains inside apply_config, whose summary covers it.
            log.info(
                "backend %s (%s) drained: its last request has finished",
                state.name,
                state.config.url,
            )

    def _note_requested(self, requested: str) -> None:
        seen = self._requested_seen
        seen[requested] = time.monotonic()
        seen.move_to_end(requested)
        while len(seen) > MAX_REMEMBERED_MODEL_NAMES:
            seen.popitem(last=False)

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

        # Read once: a reload may swap self.config while this request waits, and
        # one request should not mix settings from two generations.
        config = self.config
        explicit = explicit_session(body, request.headers, config.routing.session_headers)
        keys = [explicit.key] if explicit else surface.session_keys(body)

        client = request.client
        tracked = self.tracker.begin(
            request.headers.raw,
            client.host if client else None,
            body,
            requested,
            explicit,
            keys,
        )
        # Once the body is in, the client can tell us only one more thing: that
        # it has hung up. If it does before its response is ready, this handler
        # is cancelled -- see the except clause.
        handler = asyncio.current_task()
        finished = abandoned = False

        def client_left(watch: asyncio.Future) -> None:
            nonlocal abandoned
            # `finished` is set in the same step as the response is produced, so
            # a hang-up noticed after that can never cancel a response on its
            # way out.
            if not finished and not watch.cancelled():
                abandoned = True
                handler.cancel()

        watching = asyncio.ensure_future(_until_client_leaves(request.receive))
        watching.add_done_callback(client_left)
        response: Response | None = None
        try:
            response = await self._route(
                request, surface, body, requested, config, explicit, keys, tracked
            )
            return response
        except asyncio.CancelledError:
            # Ours, from client_left -- unless something else cancelled this
            # handler as well (the server shutting down), which then wins.
            if not abandoned or handler.uncancel() > 0:
                raise
            # Left queued, it would have gone on to take a slot; left waiting on
            # a backend, kept that backend generating -- both for nobody. The
            # cancellation has dropped it from the queue, or closed the upstream
            # request, which is how a backend learns to stop.
            tracked.note = f"client left while {tracked.state}"
            self.stats.abandoned += 1
            log.debug("client hung up; abandoned its request (%s)", tracked.state)
            # Nobody will read this. 499 is nginx's "client closed request".
            return Response(status_code=499)
        finally:
            finished = True
            watching.cancel()
            # A stream records its own end, once its last byte has gone out.
            if not tracked.stream_owned:
                self.tracker.finish(
                    tracked, response.status_code if response is not None else None
                )

    async def _route(
        self,
        request: Request,
        surface: Surface,
        body: dict,
        requested: str,
        config: Config,
        explicit: ExplicitSession | None,
        keys: list[str],
        tracked: Tracked,
    ) -> Response:
        model = self.resolve_model(requested)
        if not config.backends_for(model):
            tracked.note = "unknown model"
            return surface.error(
                404,
                f"model '{requested}' is not served by any configured backend "
                f"(known: {', '.join(config.all_models)})",
                surfaces.MODEL_NOT_FOUND,
            )
        self._note_requested(requested)

        streaming = bool(body.get("stream"))
        routing = config.routing

        min_depth = 1 if explicit else surface.min_affinity_depth
        if surface is OPENAI:
            # `session_id` is ours, not part of the OpenAI schema; never forward it.
            # The Anthropic body is passed through untouched, since capability
            # headers pair with body fields and breaking a pair is a hard 400.
            body.pop("session_id", None)

        preferred = self.sessions.lookup(keys, min_depth) if keys else None
        if keys:
            # Track where identity came from, so it is possible to confirm that a
            # client's own conversation id is being used rather than guessed at.
            if explicit:
                self.stats.keys_from_header += 1
            else:
                self.stats.keys_from_prefix += 1
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
        failures = 0
        reroutes = MAX_RELOAD_REROUTES

        while failures <= routing.max_retries:
            tracked.queued(preferred, routing.affinity_wait_ms / 1000.0)
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
                if reroutes and self.config is not config:
                    # A reload landed while this request waited, and may have
                    # renamed its model. Follow the new aliases before giving up.
                    reroutes -= 1
                    config = self.config
                    model = self.resolve_model(requested)
                    if config.backends_for(model):
                        continue
                if last_error is not None:
                    return last_error
                self.stats.rejected_no_backend += 1
                tracked.note = "no healthy backend"
                return surface.error(
                    503,
                    f"no healthy backend available for model '{model}'",
                    surfaces.NO_BACKEND,
                )
            except QueueTimeout:
                self.stats.queue_timeouts += 1
                tracked.note = "queue timeout"
                return surface.error(
                    503,
                    f"timed out waiting {routing.queue_timeout_s:.0f}s for a free slot",
                    surfaces.QUEUE_TIMEOUT,
                )

            if self.scheduler.backends.get(lease.name) is not lease.backend:
                # Granted just before a reload removed this backend or pointed its
                # name at another host. Nothing has been sent, so ask again.
                lease.release()
                if reroutes:
                    reroutes -= 1
                    continue
                return surface.error(
                    503, "backend was reconfigured; retry", surfaces.NO_BACKEND
                )

            self._record_lease(lease, pinned=preferred is not None)
            if failures:
                self.stats.retries += 1
            tracked.dispatched(lease.name, lease.queue_wait_s)

            outcome, response = await self._dispatch(
                lease, body, streaming, keys, surface, request, min_depth, model, tracked
            )
            if outcome == "ok":
                return response
            # Retryable: the lease is already released. Try elsewhere.
            last_error = response
            excluded.add(lease.name)
            preferred = None
            failures += 1

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

        states = [self.scheduler.backends.get(b.name) for b in self.config.backends_for(model)]
        candidates = [s for s in states if s is not None and s.healthy]
        if not candidates:
            return surface.error(
                503,
                f"no healthy backend available for model '{model}'",
                surfaces.NO_BACKEND,
            )

        keys = surface.session_keys(body)
        pinned = self.sessions.lookup(keys, surface.min_affinity_depth) if keys else None
        chosen = next((s for s in candidates if s.name == pinned), None) or min(
            candidates, key=lambda s: s.inflight
        )
        backend = chosen.config

        payload = dict(body)
        if backend.upstream_model:
            payload["model"] = backend.upstream_model
        else:
            payload["model"] = model

        headers = self._upstream_headers(backend, request, streaming=False)
        try:
            # Borrowed, so a reload cannot close the client under this call.
            with self.clients.lend(backend.name) as client:
                upstream = await client.post(
                    "/v1/messages/count_tokens",
                    json=payload,
                    headers=headers,
                    params=dict(request.query_params),
                    timeout=self.clients.request_timeout(),
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
        tracked: Tracked,
    ) -> tuple[str, Response]:
        backend = lease.backend.config
        # Borrowed for exactly as long as the slot is held, so a reload that
        # retires this client closes it only once the request is done with it.
        upstream = self.clients.checkout(backend.name)
        lease.on_release(lambda: self.clients.checkin(upstream))
        client = upstream.client

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
                    params, min_depth, tracked,
                )
            return await self._dispatch_buffered(
                lease, client, headers, payload, keys, started, surface,
                params, min_depth, tracked,
            )
        except (httpx.HTTPError, OSError) as exc:
            lease.release()
            self.scheduler.note_failure(lease.backend)
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
        except BaseException:
            # Cancelled because the client hung up (see _handle), or something
            # unexpected. Either way the slot goes back; releasing is idempotent,
            # so a path that already released it is unaffected.
            lease.release()
            raise

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
        tracked: Tracked,
    ) -> tuple[str, Response]:
        name = lease.name
        upstream = await client.post(
            surface.path,
            json=payload,
            headers=headers,
            params=params,
            timeout=self.clients.request_timeout(),
        )

        # Non-streaming: the slot is free the moment the body is in hand.
        if upstream.status_code != 200:
            lease.release()
            return self._upstream_error(
                lease.backend, upstream.status_code, upstream.content, surface
            )

        lease.release()
        self.scheduler.note_success(lease.backend)

        bstats = self.stats.backend(name)
        bstats.completed += 1
        elapsed = time.monotonic() - started
        bstats.ttft_s.append(elapsed)

        try:
            data = upstream.json()
        except ValueError:
            data = None
        usage = surface.usage_from_body(data)
        tracked.usage = usage
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
        tracked: Tracked,
    ) -> tuple[str, Response]:
        name = lease.name
        request = client.build_request(
            "POST",
            surface.path,
            json=payload,
            headers=headers,
            params=params,
            timeout=self.clients.request_timeout(),
        )
        # send(stream=True) returns once headers are in, so we can still fail over
        # to another backend before any bytes reach the client.
        upstream = await client.send(request, stream=True)

        if upstream.status_code != 200:
            content = await upstream.aread()
            await upstream.aclose()
            lease.release()
            return self._upstream_error(lease.backend, upstream.status_code, content, surface)

        self.scheduler.note_success(lease.backend)
        if keys:
            self.sessions.assign(keys, name, min_depth)

        # From here the stream, not the handler, says when this request is done.
        tracked.stream_owned = True
        return ("ok", StreamingResponse(
            self._stream_body(lease, upstream, started, surface, tracked),
            status_code=200,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            headers={
                "x-llm-router-backend": name,
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
            },
        ))

    async def _stream_body(
        self,
        lease: Lease,
        upstream: httpx.Response,
        started: float,
        surface: Surface,
        tracked: Tracked,
    ):
        name = lease.name
        bstats = self.stats.backend(name)
        tap = StreamTap(surface)
        first_byte_at: float | None = None
        failed = False
        finished = False

        try:
            # aiter_raw forwards bytes exactly as they arrive, including SSE ping
            # events and comment lines. Claude Code counts every byte and aborts a
            # stream that goes silent for 300s, so pings must not be filtered.
            async for chunk in upstream.aiter_raw():
                if first_byte_at is None:
                    first_byte_at = time.monotonic()
                    bstats.ttft_s.append(first_byte_at - started)
                tracked.received(len(chunk))
                tap.feed(chunk)
                yield chunk
            finished = True
        except (httpx.HTTPError, OSError) as exc:
            # Mid-stream failure: the client has bytes already, so we cannot retry.
            failed = True
            log.warning("backend %s stream aborted: %r", name, exc)
            self.scheduler.note_failure(lease.backend)
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
            tracked.usage = usage
            # Neither failed nor finished: the client hung up mid-stream.
            self.tracker.finish(
                tracked, 200, ERROR if failed else OK if finished else CANCELLED
            )

    def _upstream_error(
        self, state: BackendState, status: int, content: bytes, surface: Surface
    ) -> tuple[str, Response]:
        name = state.name
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
            self.scheduler.note_failure(state)
        else:
            # A 4xx is the client's fault; it says nothing about backend health.
            self.scheduler.note_success(state)

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
            entry["draining"] = False
            backends.append(entry)

        # Taken out of the config by a reload, still finishing what they were doing.
        for state in sched.draining:
            backends.append({
                "name": state.name,
                "kind": state.config.kind,
                "url": state.config.url,
                "models": list(state.config.models),
                "inflight": state.inflight,
                "capacity": state.capacity,
                "healthy": state.healthy,
                "draining": True,
            })

        return {
            "router": self.stats.snapshot()
            | {
                "queue_depth": sched.queue_depth,
                "waiting_on_affinity": sched.waiting_on_affinity,
                "tracked_sessions": len(self.sessions),
            },
            "config": {
                "generation": self.config_generation,
                "loaded_at": self.config_loaded_at,
                "error": self.config_error,
                "watch": self.config_watch,
            },
            "backends": backends,
            "models": {
                model: {"context_length": self.context_for(model)}
                for model in self.config.all_models
            },
        }

    def tracking_snapshot(self) -> dict:
        """What /sessions serves: the tracker's view, plus just enough about the
        backends and queue to explain why something is waiting."""
        sched = self.scheduler
        now = time.monotonic()
        backends = [
            {
                "name": name,
                "inflight": state.inflight,
                "capacity": state.capacity,
                "healthy": state.healthy,
                "cooling_down": state.cooldown_until > now,
                "draining": False,
            }
            for name, state in sched.backends.items()
        ]
        backends.extend(
            {
                "name": state.name,
                "inflight": state.inflight,
                "capacity": state.capacity,
                "healthy": state.healthy,
                "cooling_down": False,
                "draining": True,
            }
            for state in sched.draining
        )
        return self.tracker.snapshot() | {
            "router": {
                "now": time.time(),
                "uptime_s": now - self.stats.started_at,
                "queue_depth": sched.queue_depth,
                "waiting_on_affinity": sched.waiting_on_affinity,
                "config_generation": self.config_generation,
                "config_error": self.config_error,
            },
            "backends": backends,
        }


def create_app(
    config: Config, router: Router | None = None, reloader: Any = None
) -> Starlette:
    """`reloader`, if given, is a ConfigReloader started and stopped with the app."""
    router = router or Router(config)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        await router.start()
        if reloader is not None:
            await reloader.start()
        try:
            yield
        finally:
            if reloader is not None:
                await reloader.stop()
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
            *dashboard.routes(router),
        ],
        lifespan=lifespan,
    )
    app.state.router = router
    return app
