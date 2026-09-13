"""Upstream HTTP clients and health probing."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Iterator
from typing import Any

import httpx

from .config import BackendConfig, Config
from .scheduler import BackendDiff, Scheduler

log = logging.getLogger("llm_router.backend")

# Settings that can change what a backend's context window is. A change to any of
# them throws away what was discovered, so the next probe looks again.
REDISCOVER_ON = frozenset({"models", "upstream_model", "kind", "context_length"})


class Upstream:
    """One backend's httpx client, and how many requests are using it right now.

    A reload that changes a backend's URL or capacity needs a new client, but a
    stream already running through the old one must not have it closed underneath
    it. So the old client is retired rather than closed, and closes itself when
    its last user checks it back in.
    """

    __slots__ = ("client", "users", "retired", "closed")

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.users = 0
        self.retired = False
        self.closed = False


class BackendClients:
    """One pooled httpx client per backend, plus the health-probe loop."""

    def __init__(self, config: Config):
        self.config = config
        self._upstreams: dict[str, Upstream] = {}
        self._probe_task: asyncio.Task | None = None
        # Set to cut the wait between probe cycles short, e.g. after a reload.
        self._wake = asyncio.Event()
        # Retired clients still in use, and ones being closed (held so the close
        # tasks are not garbage-collected mid-flight).
        self._retired: set[Upstream] = set()
        self._closing: set[asyncio.Task] = set()
        # Real slot occupancy from llama.cpp's /slots, when it is available. Used to
        # spot drift between what we think is running and what actually is.
        self.observed_busy: dict[str, int | None] = {b.name: None for b in config.backends}
        # Usable context per backend, from config or discovered upstream.
        self.context_length: dict[str, int | None] = {
            b.name: b.context_length for b in config.backends
        }
        for backend in config.backends:
            self._upstreams[backend.name] = Upstream(self._make_client(backend))

    def _make_client(self, backend: BackendConfig) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=backend.url,
            timeout=self.request_timeout(),
            # One connection per slot, plus headroom for probes.
            limits=httpx.Limits(
                max_connections=backend.capacity + 4,
                max_keepalive_connections=backend.capacity + 4,
            ),
            follow_redirects=False,
        )

    def request_timeout(self) -> httpx.Timeout:
        """Timeouts for an inference call, from the *current* config.

        Passed on every request rather than fixed into the client, so a reload
        that changes them applies without having to rebuild any clients.
        """
        timeouts = self.config.timeouts
        return httpx.Timeout(
            connect=timeouts.connect_s,
            read=timeouts.first_byte_s,
            write=timeouts.connect_s,
            pool=timeouts.connect_s,
        )

    def client(self, name: str) -> httpx.AsyncClient:
        return self._upstreams[name].client

    # ---------------------------------------------------------------- lending

    def checkout(self, name: str) -> Upstream:
        """Borrow a backend's current client. Every checkout needs a checkin."""
        upstream = self._upstreams[name]
        upstream.users += 1
        return upstream

    def checkin(self, upstream: Upstream) -> None:
        upstream.users -= 1
        if upstream.retired and upstream.users <= 0:
            self._close(upstream)

    @contextlib.contextmanager
    def lend(self, name: str) -> Iterator[httpx.AsyncClient]:
        upstream = self.checkout(name)
        try:
            yield upstream.client
        finally:
            self.checkin(upstream)

    def _retire(self, upstream: Upstream) -> None:
        upstream.retired = True
        if upstream.users <= 0:
            self._close(upstream)
        else:
            self._retired.add(upstream)

    def _close(self, upstream: Upstream) -> None:
        self._retired.discard(upstream)
        if upstream.closed:
            return
        upstream.closed = True
        try:
            task = asyncio.get_running_loop().create_task(upstream.client.aclose())
        except RuntimeError:
            return  # no loop left to close it on; the process is on its way out
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    # ------------------------------------------------------------- reconfigure

    def reconfigure(self, config: Config, diff: BackendDiff) -> None:
        """Adopt a new config, retiring clients whose host or pool size changed.

        Must run in the same synchronous step as Scheduler.reconfigure, so that the
        two never disagree about which host a name refers to.
        """
        self.config = config
        by_name = {b.name: b for b in config.backends}

        for name in diff.removed:
            self._retire(self._upstreams.pop(name))
            self.observed_busy.pop(name, None)
            self.context_length.pop(name, None)

        for name in (*diff.replaced, *diff.added):
            previous = self._upstreams.pop(name, None)
            if previous is not None:
                self._retire(previous)
            backend = by_name[name]
            self._upstreams[name] = Upstream(self._make_client(backend))
            self.observed_busy[name] = None
            self.context_length[name] = backend.context_length

        for name, changed in diff.updated.items():
            backend = by_name[name]
            if "capacity" in changed:
                # The pool is sized to capacity. Left too small after an increase,
                # requests would queue for a connection inside httpx and then fail
                # with a pool timeout -- the capacity gate would let them through
                # only for the transport to refuse them.
                self._retire(self._upstreams[name])
                self._upstreams[name] = Upstream(self._make_client(backend))
            if REDISCOVER_ON.intersection(changed):
                self.context_length[name] = backend.context_length
                self.observed_busy[name] = None

        # Probe now rather than at the next interval: new hosts start out down.
        self._wake.set()

    def headers_for(self, backend: BackendConfig) -> dict[str, str]:
        headers = dict(backend.headers)
        if backend.api_key:
            headers["authorization"] = f"Bearer {backend.api_key}"
        return headers

    async def aclose(self) -> None:
        await self.stop_probing()
        remaining = [*self._upstreams.values(), *self._retired]
        self._retired.clear()
        for upstream in remaining:
            upstream.closed = True
        await asyncio.gather(
            *(u.client.aclose() for u in remaining),
            *self._closing,
            return_exceptions=True,
        )

    # ------------------------------------------------------------- health probe

    async def start_probing(self, scheduler: Scheduler) -> None:
        if self._probe_task is None:
            self._probe_task = asyncio.create_task(self._probe_loop(scheduler))

    async def stop_probing(self) -> None:
        if self._probe_task is not None:
            self._probe_task.cancel()
            try:
                await self._probe_task
            except asyncio.CancelledError:
                pass
            self._probe_task = None

    async def _probe_loop(self, scheduler: Scheduler) -> None:
        # Probe once immediately so startup does not route into a dead host.
        while True:
            # Cleared before the cycle, not after, so a reload landing mid-cycle
            # still triggers another one straight away.
            self._wake.clear()
            # Re-read every cycle: a reload may have changed the list or interval.
            backends = self.config.backends
            results = await asyncio.gather(
                *(self._probe_one(b, scheduler) for b in backends),
                return_exceptions=True,
            )
            # Transport failures are handled inside _probe_one; anything reaching here
            # is a bug in the probe itself, and must not vanish silently.
            for backend, result in zip(backends, results):
                if isinstance(result, BaseException):
                    log.exception(
                        "health probe for %s raised", backend.name, exc_info=result
                    )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), self.config.health.interval_s)

    async def _probe_one(self, backend: BackendConfig, scheduler: Scheduler) -> None:
        state = scheduler.backends.get(backend.name)
        if state is None or backend.name not in self._upstreams:
            return  # removed by a reload since this cycle began
        # The newest settings for it, in case a reload landed since the cycle began.
        backend = state.config
        timeout = self.config.health.timeout_s
        try:
            with self.lend(backend.name) as client:
                response = await client.get(
                    backend.probe_path,
                    headers=self.headers_for(backend),
                    timeout=timeout,
                )
            # llama.cpp returns 503 with "Loading model" until the model is resident;
            # ninfer's /health is a static 200 and tells us only that it is listening.
            healthy = response.status_code == 200
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            healthy = False
            log.debug("health probe failed for %s: %s", backend.name, exc)

        if scheduler.backends.get(backend.name) is not state:
            # Replaced while the probe was out. This result describes a host that
            # is no longer the one behind the name, so it must not be applied.
            return

        was_healthy = state.healthy
        scheduler.set_healthy(state, healthy)
        if was_healthy != healthy:
            log.warning(
                "backend %s is now %s", backend.name, "UP" if healthy else "DOWN"
            )

        if not healthy:
            return

        # Discover on first sight, and again after a restart -- a host that just came
        # back may have been relaunched with a different --max-context or -c/-np.
        if self.context_length.get(backend.name) is None or not was_healthy:
            await self.discover_context(backend, timeout)

        if backend.load_path:
            await self._probe_load(backend, scheduler, timeout)

    async def discover_context(
        self, backend: BackendConfig, timeout: float | None = None
    ) -> int | None:
        """Find a backend's *usable* context window.

        Deliberately different per backend kind, because the obvious field is wrong
        for llama.cpp: its /v1/models reports `meta.n_ctx_train`, the model's
        architectural limit, which has nothing to do with what the server will
        accept. The served figure lives in /props, and is already divided by the
        number of slots (`-c 65536 -np 4` gives each slot 16384).
        """
        if backend.context_length is not None:
            return backend.context_length  # explicit config always wins
        timeout = timeout if timeout is not None else self.config.health.timeout_s

        if backend.kind == "lmstudio":
            value = await self._context_from_lmstudio(backend, timeout)
            if value is None:
                value = await self._context_from_models(backend, timeout)
        elif backend.kind == "llamacpp":
            value = await self._context_from_props(backend, timeout)
            if value is None:
                value = await self._context_from_models(backend, timeout)
                if value is not None:
                    log.warning(
                        "backend %s: /props unavailable, falling back to /v1/models "
                        "n_ctx_train=%d. This is the model's architectural context, "
                        "not the served per-slot context -- set context_length "
                        "explicitly if requests start failing on length",
                        backend.name,
                        value,
                    )
        else:
            value = await self._context_from_models(backend, timeout)

        # Discovery takes a few round trips. If a reload changed this backend in
        # the meantime, the answer describes settings that no longer apply.
        if value is not None and self._current(backend.name) == backend:
            previous = self.context_length.get(backend.name)
            self.context_length[backend.name] = value
            if previous != value:
                log.info("backend %s: context window %d tokens", backend.name, value)
        return value

    def _current(self, name: str) -> BackendConfig | None:
        return next((b for b in self.config.backends if b.name == name), None)

    async def _get(
        self, backend: BackendConfig, path: str, timeout: float
    ) -> httpx.Response | None:
        """GET through the backend's current client; None if a reload removed it."""
        if backend.name not in self._upstreams:
            return None
        with self.lend(backend.name) as client:
            return await client.get(path, headers=self.headers_for(backend), timeout=timeout)

    async def _context_from_props(
        self, backend: BackendConfig, timeout: float
    ) -> int | None:
        """llama.cpp: /props -> default_generation_settings.n_ctx (per slot)."""
        try:
            response = await self._get(backend, "/props", timeout)
            if response is None or response.status_code != 200:
                return None
            settings = (response.json() or {}).get("default_generation_settings") or {}
            value = settings.get("n_ctx")
            return int(value) if isinstance(value, int) and value > 0 else None
        except (httpx.HTTPError, ValueError, TypeError, AttributeError, asyncio.TimeoutError):
            return None

    async def _context_from_lmstudio(
        self, backend: BackendConfig, timeout: float
    ) -> int | None:
        """LM Studio: /api/v0/models -> loaded_context_length.

        Same trap as llama.cpp. `max_context_length` is what the model could do;
        `loaded_context_length` is what was actually allocated when it was loaded,
        and is the only figure a request has to fit inside.
        """
        try:
            response = await self._get(backend, "/api/v0/models", timeout)
            if response is None or response.status_code != 200:
                return None
            data = (response.json() or {}).get("data") or []
            if not isinstance(data, list):
                return None

            target = backend.upstream_model
            loaded = [
                e
                for e in data
                if isinstance(e, dict) and e.get("state") not in ("not-loaded", None)
            ]
            pool = loaded or [e for e in data if isinstance(e, dict)]
            entry = next((e for e in pool if e.get("id") == target), pool[0] if pool else None)
            if not isinstance(entry, dict):
                return None

            if entry.get("state") == "not-loaded":
                log.info(
                    "backend %s: model '%s' is not loaded; LM Studio will JIT-load it "
                    "on the first request, which will be slow",
                    backend.name,
                    entry.get("id"),
                )

            for key in ("loaded_context_length", "max_context_length"):
                value = entry.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    if key == "max_context_length" and "loaded_context_length" in entry:
                        continue
                    return value
            return None
        except (httpx.HTTPError, ValueError, TypeError, AttributeError, asyncio.TimeoutError):
            return None

    async def _context_from_models(
        self, backend: BackendConfig, timeout: float
    ) -> int | None:
        """ninfer/vLLM: /v1/models -> max_model_len, or meta.n_ctx / n_ctx_train."""
        try:
            response = await self._get(backend, "/v1/models", timeout)
            if response is None or response.status_code != 200:
                return None
            data = (response.json() or {}).get("data") or []
            if not isinstance(data, list) or not data:
                return None

            target = backend.upstream_model
            entry = next(
                (e for e in data if isinstance(e, dict) and e.get("id") == target),
                data[0] if isinstance(data[0], dict) else None,
            )
            if not isinstance(entry, dict):
                return None

            meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
            for value in (
                entry.get("max_model_len"),
                meta.get("n_ctx"),
                meta.get("n_ctx_train"),
            ):
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    return value
            return None
        except (httpx.HTTPError, ValueError, TypeError, AttributeError, asyncio.TimeoutError):
            return None

    async def _probe_load(
        self, backend: BackendConfig, scheduler: Scheduler, timeout: float
    ) -> None:
        """Ask a backend what *it* thinks it is running, and compare with our count.

        Only llama.cpp (/slots) and vLLM (/load) publish this. A backend reporting
        more work than we dispatched means something else is sharing the host, which
        silently invalidates our capacity gate -- worth saying out loud.
        """
        path = backend.load_path
        if path is None:
            return
        try:
            response = await self._get(backend, path, timeout)
            if response is None or response.status_code != 200:
                # vLLM's /load needs --enable-server-load-tracking; absent is fine.
                self.observed_busy[backend.name] = None
                return
            payload: Any = response.json()
        except (httpx.HTTPError, ValueError, asyncio.TimeoutError):
            self.observed_busy[backend.name] = None
            return

        busy = self._parse_load(backend, payload)
        self.observed_busy[backend.name] = busy
        if busy is None:
            return

        state = scheduler.backends.get(backend.name)
        if state is None:
            return
        ours = state.inflight
        if busy > ours:
            log.warning(
                "backend %s reports %d running but we dispatched %d -- another client "
                "may be sharing this host, which breaks the capacity gate",
                backend.name,
                busy,
                ours,
            )

    def _parse_load(self, backend: BackendConfig, payload: Any) -> int | None:
        if backend.kind == "llamacpp":
            if not isinstance(payload, list):
                return None
            if len(payload) < backend.capacity:
                log.warning(
                    "backend %s exposes %d slots but is configured with capacity %d; "
                    "lower capacity to match llama.cpp's -np",
                    backend.name,
                    len(payload),
                    backend.capacity,
                )
            return sum(
                1 for s in payload if isinstance(s, dict) and s.get("is_processing")
            )

        if backend.kind == "vllm":
            # {"server_load": N} -- requests currently occupying the GPU.
            if not isinstance(payload, dict):
                return None
            value = payload.get("server_load")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return int(value)

        return None
