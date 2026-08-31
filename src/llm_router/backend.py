"""Upstream HTTP clients and health probing."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from .config import BackendConfig, Config
from .scheduler import Scheduler

log = logging.getLogger("llm_router.backend")


class BackendClients:
    """One pooled httpx client per backend, plus the health-probe loop."""

    def __init__(self, config: Config):
        self.config = config
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._probe_task: asyncio.Task | None = None
        # Real slot occupancy from llama.cpp's /slots, when it is available. Used to
        # spot drift between what we think is running and what actually is.
        self.observed_busy: dict[str, int | None] = {b.name: None for b in config.backends}
        # Usable context per backend, from config or discovered upstream.
        self.context_length: dict[str, int | None] = {
            b.name: b.context_length for b in config.backends
        }

        timeouts = config.timeouts
        for backend in config.backends:
            self._clients[backend.name] = httpx.AsyncClient(
                base_url=backend.url,
                timeout=httpx.Timeout(
                    connect=timeouts.connect_s,
                    read=timeouts.first_byte_s,
                    write=timeouts.connect_s,
                    pool=timeouts.connect_s,
                ),
                # One connection per slot, plus headroom for probes.
                limits=httpx.Limits(
                    max_connections=backend.capacity + 4,
                    max_keepalive_connections=backend.capacity + 4,
                ),
                follow_redirects=False,
            )

    def client(self, name: str) -> httpx.AsyncClient:
        return self._clients[name]

    def headers_for(self, backend: BackendConfig) -> dict[str, str]:
        headers = dict(backend.headers)
        if backend.api_key:
            headers["authorization"] = f"Bearer {backend.api_key}"
        return headers

    async def aclose(self) -> None:
        await self.stop_probing()
        await asyncio.gather(
            *(c.aclose() for c in self._clients.values()), return_exceptions=True
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
        health = self.config.health
        # Probe once immediately so startup does not route into a dead host.
        while True:
            results = await asyncio.gather(
                *(self._probe_one(b, scheduler) for b in self.config.backends),
                return_exceptions=True,
            )
            # Transport failures are handled inside _probe_one; anything reaching here
            # is a bug in the probe itself, and must not vanish silently.
            for backend, result in zip(self.config.backends, results):
                if isinstance(result, BaseException):
                    log.exception(
                        "health probe for %s raised", backend.name, exc_info=result
                    )
            await asyncio.sleep(health.interval_s)

    async def _probe_one(self, backend: BackendConfig, scheduler: Scheduler) -> None:
        client = self._clients[backend.name]
        timeout = self.config.health.timeout_s
        try:
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

        was_healthy = scheduler.backends[backend.name].healthy
        scheduler.set_healthy(backend.name, healthy)
        if was_healthy != healthy:
            log.warning(
                "backend %s is now %s", backend.name, "UP" if healthy else "DOWN"
            )

        if not healthy:
            return

        # Discover on first sight, and again after a restart -- a host that just came
        # back may have been relaunched with a different --max-context or -c/-np.
        if self.context_length[backend.name] is None or not was_healthy:
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

        if value is not None:
            previous = self.context_length.get(backend.name)
            self.context_length[backend.name] = value
            if previous != value:
                log.info("backend %s: context window %d tokens", backend.name, value)
        return value

    async def _context_from_props(
        self, backend: BackendConfig, timeout: float
    ) -> int | None:
        """llama.cpp: /props -> default_generation_settings.n_ctx (per slot)."""
        try:
            response = await self._clients[backend.name].get(
                "/props", headers=self.headers_for(backend), timeout=timeout
            )
            if response.status_code != 200:
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
            response = await self._clients[backend.name].get(
                "/api/v0/models", headers=self.headers_for(backend), timeout=timeout
            )
            if response.status_code != 200:
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
            response = await self._clients[backend.name].get(
                "/v1/models", headers=self.headers_for(backend), timeout=timeout
            )
            if response.status_code != 200:
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
            response = await self._clients[backend.name].get(
                path, headers=self.headers_for(backend), timeout=timeout
            )
            if response.status_code != 200:
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

        ours = scheduler.backends[backend.name].inflight
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
