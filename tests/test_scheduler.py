"""Scheduler behaviour: the capacity guarantee and the affinity/spill trade-off."""

from __future__ import annotations

import asyncio
import time

import pytest

from llm_router.config import BackendConfig, HealthConfig
from llm_router.scheduler import NoBackendError, QueueTimeout, Scheduler

MODEL = "test-model"


def make_scheduler(*capacities: int) -> Scheduler:
    backends = tuple(
        BackendConfig(
            name=f"b{i}", url=f"http://host{i}", capacity=cap, models=(MODEL,)
        )
        for i, cap in enumerate(capacities)
    )
    return Scheduler(backends, HealthConfig())


async def test_never_exceeds_capacity_under_load():
    """The core guarantee: no backend is ever handed more than `capacity` at once."""
    scheduler = make_scheduler(4, 2, 4)
    await scheduler.start()
    breaches: list[str] = []

    async def worker() -> None:
        lease = await scheduler.acquire(MODEL, timeout_s=30)
        try:
            state = lease.backend
            if state.inflight > state.capacity:
                breaches.append(f"{state.name}: {state.inflight}/{state.capacity}")
            await asyncio.sleep(0.005)
        finally:
            lease.release()

    try:
        await asyncio.gather(*(worker() for _ in range(200)))
    finally:
        await scheduler.stop()

    assert breaches == [], f"capacity exceeded: {breaches}"
    assert all(s.inflight == 0 for s in scheduler.backends.values())


async def test_idle_backend_used_before_queueing():
    """The LiteLLM regression: work must go to an idle host, not queue on a busy one."""
    scheduler = make_scheduler(2, 2)
    await scheduler.start()
    try:
        held = [await scheduler.acquire(MODEL, preferred="b0") for _ in range(2)]
        assert all(lease.name == "b0" for lease in held)
        assert scheduler.backends["b0"].free == 0

        # b0 is full; this must land on b1 immediately rather than waiting.
        started = time.monotonic()
        lease = await scheduler.acquire(MODEL, timeout_s=5)
        assert lease.name == "b1"
        assert time.monotonic() - started < 0.05
        lease.release()
        for lease in held:
            lease.release()
    finally:
        await scheduler.stop()


async def test_affinity_honored_when_pinned_backend_free():
    scheduler = make_scheduler(2, 2)
    await scheduler.start()
    try:
        # b1 is the least loaded, but the pin must win.
        busy = await scheduler.acquire(MODEL, preferred="b0")
        lease = await scheduler.acquire(MODEL, preferred="b0")
        assert lease.name == "b0"
        assert lease.affinity_honored is True
        assert lease.spilled is False
        lease.release()
        busy.release()
    finally:
        await scheduler.stop()


async def test_affinity_spills_after_wait_window():
    """A pinned request holds out briefly, then gives up and takes an idle host."""
    scheduler = make_scheduler(1, 1)
    await scheduler.start()
    try:
        blocker = await scheduler.acquire(MODEL, preferred="b0")
        assert blocker.name == "b0"

        started = time.monotonic()
        lease = await scheduler.acquire(MODEL, preferred="b0", affinity_wait_s=0.2)
        elapsed = time.monotonic() - started

        assert lease.name == "b1", "should have spilled to the idle backend"
        assert lease.spilled is True
        assert lease.affinity_honored is False
        assert elapsed >= 0.2, "spilled before the affinity window closed"
        assert elapsed < 1.0, "waited far longer than the affinity window"
        lease.release()
        blocker.release()
    finally:
        await scheduler.stop()


async def test_affinity_wait_does_not_block_others():
    """A request holding out for a busy pin must not stall the queue behind it."""
    scheduler = make_scheduler(1, 1)
    await scheduler.start()
    try:
        blocker = await scheduler.acquire(MODEL, preferred="b0")

        # Enqueued first, pinned to the full b0, with a long affinity window.
        waiter = asyncio.create_task(
            scheduler.acquire(MODEL, preferred="b0", affinity_wait_s=5.0)
        )
        await asyncio.sleep(0.05)
        assert not waiter.done()

        # Enqueued second, unpinned: must overtake and take the idle b1.
        started = time.monotonic()
        lease = await asyncio.wait_for(scheduler.acquire(MODEL), timeout=1.0)
        assert lease.name == "b1"
        assert time.monotonic() - started < 0.2

        lease.release()
        blocker.release()
        granted = await asyncio.wait_for(waiter, timeout=2.0)
        assert granted.name == "b0", "pinned request should get its host once freed"
        granted.release()
    finally:
        await scheduler.stop()


async def test_pinned_request_gets_its_host_when_slot_frees():
    scheduler = make_scheduler(1, 4)
    await scheduler.start()
    try:
        blocker = await scheduler.acquire(MODEL, preferred="b0")
        waiter = asyncio.create_task(
            scheduler.acquire(MODEL, preferred="b0", affinity_wait_s=5.0)
        )
        await asyncio.sleep(0.05)
        blocker.release()
        lease = await asyncio.wait_for(waiter, timeout=2.0)
        assert lease.name == "b0"
        assert lease.affinity_honored is True
        lease.release()
    finally:
        await scheduler.stop()


async def test_queue_timeout():
    scheduler = make_scheduler(1)
    await scheduler.start()
    try:
        held = await scheduler.acquire(MODEL)
        with pytest.raises(QueueTimeout):
            await scheduler.acquire(MODEL, timeout_s=0.2)
        held.release()
        assert scheduler.queue_depth == 0
    finally:
        await scheduler.stop()


async def test_cancelled_waiter_leaves_no_trace():
    """Client disconnects while queued: no leaked slot, no orphaned waiter."""
    scheduler = make_scheduler(1)
    await scheduler.start()
    try:
        held = await scheduler.acquire(MODEL)
        waiter = asyncio.create_task(scheduler.acquire(MODEL, timeout_s=10))
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await asyncio.sleep(0.1)

        assert scheduler.queue_depth == 0
        held.release()
        assert scheduler.backends["b0"].inflight == 0
    finally:
        await scheduler.stop()


async def test_release_is_idempotent():
    scheduler = make_scheduler(2)
    lease = await scheduler.acquire(MODEL)
    assert scheduler.backends["b0"].inflight == 1
    lease.release()
    lease.release()
    lease.release()
    assert scheduler.backends["b0"].inflight == 0


async def test_unhealthy_backend_is_skipped():
    scheduler = make_scheduler(2, 2)
    await scheduler.start()
    try:
        scheduler.set_healthy("b0", False)
        leases = [await scheduler.acquire(MODEL) for _ in range(2)]
        assert all(lease.name == "b1" for lease in leases)
        for lease in leases:
            lease.release()
    finally:
        await scheduler.stop()


async def test_pin_to_unhealthy_backend_spills_immediately():
    """Do not hold out for a host that is down -- spill without waiting."""
    scheduler = make_scheduler(2, 2)
    await scheduler.start()
    try:
        scheduler.set_healthy("b0", False)
        started = time.monotonic()
        lease = await scheduler.acquire(MODEL, preferred="b0", affinity_wait_s=5.0)
        assert lease.name == "b1"
        assert time.monotonic() - started < 0.2
        lease.release()
    finally:
        await scheduler.stop()


async def test_excluded_backends_are_not_reused():
    scheduler = make_scheduler(2, 2)
    lease = await scheduler.acquire(MODEL, exclude=frozenset({"b0"}))
    assert lease.name == "b1"
    lease.release()

    with pytest.raises(NoBackendError):
        await scheduler.acquire(MODEL, exclude=frozenset({"b0", "b1"}))


async def test_unknown_model_rejected():
    scheduler = make_scheduler(2)
    with pytest.raises(NoBackendError):
        await scheduler.acquire("nope")


async def test_least_loaded_uses_capacity_fraction():
    """A 4-slot host should absorb proportionally more than a 2-slot host."""
    scheduler = make_scheduler(4, 2)
    await scheduler.start()
    try:
        leases = [await scheduler.acquire(MODEL) for _ in range(6)]
        placed = {"b0": 0, "b1": 0}
        for lease in leases:
            placed[lease.name] += 1
        assert placed == {"b0": 4, "b1": 2}
        for lease in leases:
            lease.release()
    finally:
        await scheduler.stop()


async def test_all_backends_down_fails_fast():
    """Queueing is right when hosts are busy, wrong when none are up."""
    scheduler = make_scheduler(2, 2)
    await scheduler.start()
    try:
        scheduler.set_healthy("b0", False)
        scheduler.set_healthy("b1", False)

        started = time.monotonic()
        with pytest.raises(NoBackendError):
            # A long queue timeout must not keep the client hanging.
            await scheduler.acquire(
                MODEL, timeout_s=300, unavailable_grace_s=0.2
            )
        elapsed = time.monotonic() - started
        assert 0.2 <= elapsed < 2.0, f"took {elapsed:.2f}s to give up"
    finally:
        await scheduler.stop()


async def test_backend_recovering_within_grace_is_used():
    """A blip that resolves inside the grace window should ride through."""
    scheduler = make_scheduler(2)
    await scheduler.start()
    try:
        scheduler.set_healthy("b0", False)
        waiter = asyncio.create_task(
            scheduler.acquire(MODEL, timeout_s=10, unavailable_grace_s=5.0)
        )
        await asyncio.sleep(0.2)
        assert not waiter.done()

        scheduler.set_healthy("b0", True)
        lease = await asyncio.wait_for(waiter, timeout=2.0)
        assert lease.name == "b0"
        lease.release()
    finally:
        await scheduler.stop()


async def test_busy_pool_still_queues_for_the_full_timeout():
    """The grace period must not short-circuit ordinary backpressure."""
    scheduler = make_scheduler(1)
    await scheduler.start()
    try:
        held = await scheduler.acquire(MODEL)
        waiter = asyncio.create_task(
            scheduler.acquire(MODEL, timeout_s=10, unavailable_grace_s=0.1)
        )
        # Well past the grace window, but the host is up -- just busy.
        await asyncio.sleep(0.4)
        assert not waiter.done(), "healthy-but-busy pool must keep queueing"

        held.release()
        lease = await asyncio.wait_for(waiter, timeout=2.0)
        assert lease.name == "b0"
        lease.release()
    finally:
        await scheduler.stop()
