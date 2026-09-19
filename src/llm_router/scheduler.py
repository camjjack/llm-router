"""Capacity-aware dispatch.

The rule that makes this different from a plain least-busy router: a request is
never handed to a backend that is already at capacity. It waits here, in the
proxy's own FIFO, where we can see it -- rather than being swallowed by the
backend's internal pending queue where it blocks behind other work while other
hosts sit idle.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .config import BackendConfig, HealthConfig

# How often to re-run placement in the absence of any other event. Affinity
# deadlines expire on their own schedule, so something has to look at the clock.
PUMP_INTERVAL_S = 0.05


class NoBackendError(Exception):
    """No backend is configured to serve the requested model."""


class QueueTimeout(Exception):
    """Waited past the queue timeout without a slot coming free."""


# Compared by identity, not value: after a config reload two states can share a
# name (the old host draining, its replacement taking new work), and every lease
# must go back to the exact one it was granted on.
@dataclass(eq=False)
class BackendState:
    config: BackendConfig
    inflight: int = 0
    healthy: bool = True
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    # Monotonic stamp of the last placement, used to break ties between equally
    # loaded backends. See _choose.
    last_assigned: int = 0
    # Removed from the config (or its URL changed): takes no new work, and is
    # forgotten once the requests already running on it finish.
    draining: bool = False
    # Seconds this backend has spent full with requests queued for a model it
    # serves. A request's share of it is what its session cost everyone else.
    contended_s: float = 0.0
    # Whether that is true right now; see Scheduler._account.
    contended: bool = False

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def capacity(self) -> int:
        return self.config.capacity

    @property
    def free(self) -> int:
        return max(0, self.capacity - self.inflight)

    @property
    def load(self) -> float:
        return self.inflight / self.capacity if self.capacity else 1.0

    def available(self, now: float) -> bool:
        return self.healthy and now >= self.cooldown_until

    def serves(self, model: str) -> bool:
        return model in self.config.models


@dataclass
class Lease:
    """A held slot. Releasing is idempotent -- a leaked slot is the exact failure
    mode this router exists to prevent, so double-release must not corrupt the count."""

    backend: BackendState
    affinity_honored: bool
    spilled: bool
    queue_wait_s: float
    _release: Callable[[BackendState], None] | None = field(repr=False, default=None)
    _released: bool = field(repr=False, default=False)
    _on_release: list[Callable[[], None]] = field(repr=False, default_factory=list)

    @property
    def name(self) -> str:
        return self.backend.name

    def on_release(self, callback: Callable[[], None]) -> None:
        """Run `callback` when the slot is given back. Ties anything whose lifetime
        is the request's -- the upstream client it borrowed -- to the one release
        path that is already known to run exactly once."""
        if self._released:
            callback()
        else:
            self._on_release.append(callback)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        # By object rather than by name: a reload may have replaced the backend
        # of that name since this slot was granted.
        if self._release is not None:
            self._release(self.backend)
        for callback in self._on_release:
            callback()
        self._on_release.clear()


@dataclass
class _Waiter:
    seq: int
    model: str
    preferred: str | None
    affinity_deadline: float
    enqueued_at: float
    future: asyncio.Future
    # Backends already tried and failed for this request; never retry onto them.
    exclude: frozenset[str] = frozenset()
    # Give up at this point if no backend for the model is even up.
    unavailable_deadline: float = 0.0


@dataclass
class BackendDiff:
    """What a reconfigure did, by backend name."""

    added: list[str] = field(default_factory=list)
    # Gone from the config; draining whatever was already running on them.
    removed: list[str] = field(default_factory=list)
    # Same name, different URL -- a different host. The old one drains.
    replaced: list[str] = field(default_factory=list)
    # Same host, some other setting changed; name -> the fields that changed.
    updated: dict[str, list[str]] = field(default_factory=dict)


def _changed_fields(old: BackendConfig, new: BackendConfig) -> list[str]:
    return [
        f.name
        for f in dataclasses.fields(BackendConfig)
        if getattr(old, f.name) != getattr(new, f.name)
    ]


class Scheduler:
    def __init__(
        self,
        backends: tuple[BackendConfig, ...],
        health: HealthConfig | None = None,
    ):
        self.backends: dict[str, BackendState] = {
            b.name: BackendState(config=b) for b in backends
        }
        # States taken out of rotation by a reload, still finishing their work.
        self.draining: list[BackendState] = []
        # Called with each draining state once its last request has finished.
        self.on_drained: Callable[[BackendState], None] | None = None
        self._health = health or HealthConfig()
        self._waiters: list[_Waiter] = []
        # Queued requests per model, so contention is a lookup rather than a scan.
        self._waiting: dict[str, int] = {}
        self._accounted_at = time.monotonic()
        self._seq = itertools.count()
        self._placements = itertools.count(1)
        self._pump_task: asyncio.Task | None = None

    # ---------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if self._pump_task is None:
            self._pump_task = asyncio.create_task(self._pump_loop())

    async def stop(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
            self._pump_task = None

    async def _pump_loop(self) -> None:
        while True:
            await asyncio.sleep(PUMP_INTERVAL_S)
            if self._waiters:
                self._pump()

    # ------------------------------------------------------------- reconfigure

    def reconfigure(
        self, backends: tuple[BackendConfig, ...], health: HealthConfig
    ) -> BackendDiff:
        """Swap in a new backend list without disturbing work already running.

        Synchronous on purpose, so no request can observe a half-applied config.
        A backend whose name and URL are unchanged keeps its state -- in-flight
        count, health, cooldown -- and simply takes the new settings. One that is
        gone, or now points at a different host, drains: requests already running
        on it finish where they are, and it is given nothing new.
        """
        self._account()
        diff = BackendDiff()
        old = self.backends
        new: dict[str, BackendState] = {}
        retired: list[BackendState] = []
        for config in backends:
            state = old.get(config.name)
            if state is None:
                diff.added.append(config.name)
                new[config.name] = self._joining(config)
            elif state.config.url != config.url:
                diff.replaced.append(config.name)
                retired.append(state)
                new[config.name] = self._joining(config)
            else:
                changed = _changed_fields(state.config, config)
                if changed:
                    diff.updated[config.name] = changed
                    # A capacity cut below the current in-flight count needs no
                    # special case: `free` stays at zero until enough work drains.
                    state.config = config
                new[config.name] = state
        for name, state in old.items():
            if name not in new:
                diff.removed.append(name)
                retired.append(state)

        self.backends = new
        self._health = health
        # After the swap, so an on_drained callback sees the new backend table.
        for state in retired:
            self._retire(state)

        gone = set(diff.removed) | set(diff.replaced)
        for waiter in list(self._waiters):
            if waiter.preferred in gone:
                # Pinned to a host that is no longer there: nothing to hold out for.
                waiter.preferred = None
            if not any(b.serves(waiter.model) for b in new.values()):
                # The model itself is gone. Fail now rather than after the grace
                # period, so the caller can re-resolve it against the new aliases.
                self._drop(waiter)
                if not waiter.future.done():
                    waiter.future.set_exception(
                        NoBackendError(f"model '{waiter.model}' is no longer configured")
                    )

        # Capacity may have grown, or a backend may now serve a queued model.
        self._pump()
        return diff

    def _joining(self, config: BackendConfig) -> BackendState:
        """State for a backend (re)joining the rotation.

        If a reload took this same host out moments ago -- a backend commented out
        and straight back in -- and it is still finishing that work, pick the old
        state back up. Its in-flight requests are real load on that host; starting
        from zero would let the capacity gate hand it more than it can take.
        """
        for state in self.draining:
            if state.name == config.name and state.config.url == config.url:
                self.draining.remove(state)
                state.draining = False
                state.config = config
                return state
        # Unproven until probed; the caller probes straight away.
        return BackendState(config=config, healthy=False)

    def _retire(self, state: BackendState) -> None:
        state.draining = True
        if state.inflight > 0:
            self.draining.append(state)
        else:
            self._drained(state)

    def _drained(self, state: BackendState) -> None:
        if self.on_drained is not None:
            self.on_drained(state)

    # ----------------------------------------------------------------- acquire

    async def acquire(
        self,
        model: str,
        preferred: str | None = None,
        affinity_wait_s: float = 1.5,
        timeout_s: float = 300.0,
        exclude: frozenset[str] = frozenset(),
        unavailable_grace_s: float = 10.0,
    ) -> Lease:
        """Wait for a slot, honouring the session pin where it is worth doing."""
        if not any(
            b.serves(model) and b.name not in exclude for b in self.backends.values()
        ):
            raise NoBackendError(f"no backend serves model '{model}'")

        # A pin to a backend that no longer exists (config change) is meaningless,
        # and so is a pin to one we just failed over from.
        if preferred is not None and (preferred not in self.backends or preferred in exclude):
            preferred = None

        now = time.monotonic()
        waiter = _Waiter(
            seq=next(self._seq),
            model=model,
            preferred=preferred,
            affinity_deadline=now + affinity_wait_s,
            enqueued_at=now,
            future=asyncio.get_running_loop().create_future(),
            exclude=exclude,
            unavailable_deadline=now + unavailable_grace_s,
        )
        self._waiters.append(waiter)
        self._waiting[model] = self._waiting.get(model, 0) + 1

        # Fast path: placement is synchronous, so an idle pool never touches the queue.
        self._pump()
        if waiter.future.done():
            return waiter.future.result()

        try:
            return await asyncio.wait_for(waiter.future, timeout_s)
        except TimeoutError as exc:
            self._drop(waiter)
            raise QueueTimeout(
                f"no slot for model '{model}' within {timeout_s:.0f}s"
            ) from exc
        except asyncio.CancelledError:
            # Client hung up while queued. If a slot was granted in the same tick we
            # were cancelled, hand it straight back rather than leaking it.
            self._drop(waiter)
            if waiter.future.done() and not waiter.future.cancelled():
                if waiter.future.exception() is None:
                    waiter.future.result().release()
            raise

    def _drop(self, waiter: _Waiter) -> None:
        try:
            self._waiters.remove(waiter)
        except ValueError:
            return
        remaining = self._waiting.get(waiter.model, 1) - 1
        if remaining > 0:
            self._waiting[waiter.model] = remaining
        else:
            self._waiting.pop(waiter.model, None)
        self._account()

    # ------------------------------------------------------- queue contention

    def _account(self) -> None:
        """Advance each backend's contended-time clock, then re-read the state.

        A backend is contended while it is full and something is queued for a
        model it serves: every request holding one of its slots is keeping that
        queue waiting. Called after anything that can change either side of
        that, so the flag set last time covers the whole interval since.
        """
        now = time.monotonic()
        elapsed = now - self._accounted_at
        self._accounted_at = now
        waiting = self._waiting
        for state in self.backends.values():
            if state.contended:
                state.contended_s += elapsed
            state.contended = state.free == 0 and any(
                model in waiting for model in state.config.models
            )

    def contended_seconds(self, state: BackendState) -> float:
        """Its contended time, including the stretch in progress."""
        if not state.contended:
            return state.contended_s
        return state.contended_s + (time.monotonic() - self._accounted_at)

    def waiting_for(self, state: BackendState) -> int:
        """Requests queued for a model this backend serves."""
        return sum(self._waiting.get(model, 0) for model in state.config.models)

    # ------------------------------------------------------------------- pump

    def _pump(self) -> None:
        """Place as many queued waiters as capacity allows, oldest first.

        Contains no `await`, so the capacity check and the matching `inflight += 1`
        cannot be interleaved: slots can never be handed out twice.
        """
        if not self._waiters:
            return
        now = time.monotonic()

        for waiter in list(self._waiters):
            if waiter.future.done():
                self._drop(waiter)
                continue

            candidates = self._candidates(waiter, now)
            if not candidates:
                # Nothing serving this model is up. Waiting for a busy pool is
                # useful; waiting for a dead one just hangs the client.
                if now >= waiter.unavailable_deadline:
                    self._drop(waiter)
                    waiter.future.set_exception(
                        NoBackendError(
                            f"no healthy backend for model '{waiter.model}'"
                        )
                    )
                continue

            choice = self._choose(waiter, now, candidates)
            if choice is None:
                # Either nothing is free, or this waiter is still holding out for its
                # pinned backend. Skipping rather than stopping is what keeps one
                # session's affinity wait from blocking everyone behind it.
                continue

            backend, honored, spilled = choice
            backend.inflight += 1
            backend.last_assigned = next(self._placements)
            self._drop(waiter)
            waiter.future.set_result(
                Lease(
                    backend=backend,
                    affinity_honored=honored,
                    spilled=spilled,
                    queue_wait_s=now - waiter.enqueued_at,
                    _release=self.release,
                )
            )
        self._account()

    def _candidates(self, waiter: _Waiter, now: float) -> list[BackendState]:
        """Backends that are up, serve this model, and have not already failed it."""
        return [
            b
            for b in self.backends.values()
            if b.serves(waiter.model)
            and b.available(now)
            and b.name not in waiter.exclude
        ]

    def _choose(
        self, waiter: _Waiter, now: float, candidates: list[BackendState]
    ) -> tuple[BackendState, bool, bool] | None:
        """Pick a backend for a waiter, or None to leave it queued."""
        pinned = None
        if waiter.preferred is not None:
            pinned = next((b for b in candidates if b.name == waiter.preferred), None)

        if pinned is not None:
            if pinned.free > 0:
                return (pinned, True, False)
            # Pinned host is busy. Hold out for it until the affinity window closes:
            # for a long agentic loop, a short wait usually beats re-prefilling the
            # whole conversation on a cold host.
            if now < waiter.affinity_deadline:
                return None

        # Spill (or first placement): least loaded by fraction of capacity used, so a
        # 4-slot host takes proportionally more work than a 2-slot one.
        #
        # Ties are broken by least-recently-assigned rather than by name. It
        # matters more than it looks: chat traffic is often sequential, so every
        # backend is idle when each new conversation starts. A fixed tie-break
        # would send every one of them to the same host, which then pins them all
        # and holds every conversation's KV while the others stay cold. Rotating
        # spreads new conversations, and keeps each host's retained-checkpoint set
        # small enough to survive (ninfer keeps only 2x max-concurrency of them).
        free = [b for b in candidates if b.free > 0]
        if not free:
            return None
        best = min(free, key=lambda b: (b.load, b.inflight, b.last_assigned, b.name))
        return (best, False, pinned is not None)

    # ----------------------------------------------------------------- release

    def release(self, state: BackendState) -> None:
        if state.inflight > 0:
            state.inflight -= 1
        if state.draining:
            # Frees nothing anyone can use. Just notice when it is empty.
            if state.inflight == 0 and state in self.draining:
                self.draining.remove(state)
                self._drained(state)
            return
        self._pump()
        self._account()

    # ------------------------------------------------------------------ health

    def _state(self, target: BackendState | str) -> BackendState | None:
        """Accept the state itself, which is exact even across a reload, or a name
        (tests, and callers that only ever deal with the live config)."""
        if isinstance(target, BackendState):
            return target
        return self.backends.get(target)

    def note_success(self, target: BackendState | str) -> None:
        state = self._state(target)
        if state is None:
            return
        state.consecutive_failures = 0
        state.cooldown_until = 0.0
        if not state.healthy:
            state.healthy = True
            self._pump()

    def note_failure(self, target: BackendState | str) -> None:
        """Passive failure seen while serving a request: back the backend off."""
        state = self._state(target)
        if state is None:
            return
        state.consecutive_failures += 1
        backoff = min(
            self._health.cooldown_base_s * (2 ** (state.consecutive_failures - 1)),
            self._health.cooldown_max_s,
        )
        state.cooldown_until = time.monotonic() + backoff
        if state.consecutive_failures >= self._health.failure_threshold:
            state.healthy = False

    def set_healthy(self, target: BackendState | str, healthy: bool) -> None:
        """Result of an active probe."""
        state = self._state(target)
        if state is None:
            return
        was = state.healthy
        state.healthy = healthy
        if healthy:
            state.consecutive_failures = 0
            state.cooldown_until = 0.0
            if not was:
                self._pump()

    # -------------------------------------------------------------- inspection

    @property
    def queue_depth(self) -> int:
        return len(self._waiters)

    @property
    def waiting_on_affinity(self) -> int:
        """Queued requests still holding out for a pinned backend."""
        now = time.monotonic()
        return sum(
            1
            for w in self._waiters
            if w.preferred is not None and now < w.affinity_deadline
        )

    def snapshot(self) -> dict:
        return {
            "queue_depth": self.queue_depth,
            "waiting_on_affinity": self.waiting_on_affinity,
            "backends": {
                name: {
                    "inflight": s.inflight,
                    "capacity": s.capacity,
                    "healthy": s.healthy,
                    "cooling_down": s.cooldown_until > time.monotonic(),
                }
                for name, s in self.backends.items()
            },
        }
