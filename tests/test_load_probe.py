"""Cross-checking our in-flight count against what backends report about themselves."""

from __future__ import annotations

import asyncio
import contextlib

from fake_upstream import FakeUpstream

from llm_router.backend import BackendClients
from llm_router.config import BackendConfig, Config, HealthConfig
from llm_router.scheduler import Scheduler


@contextlib.asynccontextmanager
async def probed(upstream: FakeUpstream):
    """Start a backend and run one probe cycle against it."""
    await upstream.start()
    config = Config(
        backends=(
            BackendConfig(
                name=upstream.name,
                url=upstream.url,
                capacity=upstream.max_concurrency,
                models=(upstream.model,),
                kind=upstream.kind,
            ),
        ),
        health=HealthConfig(interval_s=0.2, timeout_s=2),
    )
    scheduler = Scheduler(config.backends, config.health)
    clients = BackendClients(config)
    try:
        await clients.start_probing(scheduler)
        await asyncio.sleep(0.6)  # let a couple of probe cycles run
        yield clients, scheduler
    finally:
        await clients.aclose()
        await upstream.stop()


async def test_llamacpp_slots_are_counted():
    upstream = FakeUpstream(name="l", model="m", kind="llamacpp", max_concurrency=4,
                            context_length=8192)
    upstream.active = 2  # pretend two slots are mid-generation
    async with probed(upstream) as (clients, scheduler):
        assert clients.observed_busy["l"] == 2


async def test_vllm_server_load_is_read():
    upstream = FakeUpstream(name="v", model="m", kind="vllm", max_concurrency=64,
                            context_length=8192)
    upstream.active = 5
    async with probed(upstream) as (clients, scheduler):
        assert clients.observed_busy["v"] == 5


async def test_vllm_without_load_tracking_degrades_quietly():
    """/load needs --enable-server-load-tracking; absent must not be an error."""
    upstream = FakeUpstream(name="v", model="m", kind="vllm", max_concurrency=64,
                            context_length=8192, load_tracking=False)
    async with probed(upstream) as (clients, scheduler):
        assert clients.observed_busy["v"] is None
        assert scheduler.backends["v"].healthy is True


async def test_backends_without_telemetry_report_nothing():
    """ninfer publishes no load information at all -- that is the whole problem."""
    upstream = FakeUpstream(name="n", model="m", kind="ninfer", context_length=8192)
    async with probed(upstream) as (clients, scheduler):
        assert clients.observed_busy["n"] is None
        assert scheduler.backends["n"].healthy is True


async def test_drift_is_reported(caplog):
    """A backend running more than we dispatched means someone else is using it."""
    upstream = FakeUpstream(name="v", model="m", kind="vllm", max_concurrency=64,
                            context_length=8192)
    upstream.active = 7  # we dispatched nothing
    with caplog.at_level("WARNING"):
        async with probed(upstream) as (clients, scheduler):
            assert clients.observed_busy["v"] == 7
    assert any("another client may be sharing" in r.message for r in caplog.records)


async def test_llamacpp_slot_count_mismatch_is_reported(caplog):
    """Configured capacity above llama.cpp's -np would break the gate."""
    upstream = FakeUpstream(name="l", model="m", kind="llamacpp", max_concurrency=2,
                            context_length=8192)
    # The router is told capacity 8; the server only exposes 2 slots.
    await upstream.start()
    config = Config(
        backends=(
            BackendConfig(name="l", url=upstream.url, capacity=8, models=("m",),
                          kind="llamacpp"),
        ),
        health=HealthConfig(interval_s=0.2, timeout_s=2),
    )
    scheduler = Scheduler(config.backends, config.health)
    clients = BackendClients(config)
    try:
        with caplog.at_level("WARNING"):
            await clients.start_probing(scheduler)
            await asyncio.sleep(0.6)
        assert any("lower capacity to match" in r.message for r in caplog.records)
    finally:
        await clients.aclose()
        await upstream.stop()


async def test_tensorfold_requests_running_is_read():
    """/health counts requests holding a stream, the router's and anyone else's."""
    upstream = FakeUpstream(name="t", model="m", kind="tensorfold", max_concurrency=4,
                            context_length=1048576, parallel=4)
    upstream.active = 1
    upstream.foreign_running = 2
    async with probed(upstream) as (clients, scheduler):
        assert clients.observed_busy["t"] == 3
        assert scheduler.backends["t"].healthy is True


async def test_tensorfold_capacity_above_its_parallel_is_warned_about_once(caplog):
    upstream = FakeUpstream(name="t", model="m", kind="tensorfold", max_concurrency=4,
                            context_length=1048576, parallel=2)
    with caplog.at_level("WARNING"):
        async with probed(upstream):
            pass
    warnings = [r for r in caplog.records if "decodes 2 requests at once" in r.getMessage()]
    assert len(warnings) == 1


async def test_sglang_loads_are_summed_over_its_ranks():
    """/v1/loads reports each data-parallel rank; the router sent to all of them."""
    upstream = FakeUpstream(name="s", model="m", kind="sglang", max_concurrency=8,
                            context_length=8192, dp_ranks=2)
    upstream.active = 3
    upstream.foreign_running = 2
    async with probed(upstream) as (clients, scheduler):
        assert clients.observed_busy["s"] == 5
        assert scheduler.backends["s"].healthy is True


async def test_sglang_capacity_above_its_max_running_requests_is_warned_about_once(caplog):
    upstream = FakeUpstream(name="s", model="m", kind="sglang", max_concurrency=8,
                            context_length=8192, dp_ranks=2, max_running=2)
    with caplog.at_level("WARNING"):
        async with probed(upstream):
            pass
    warnings = [r for r in caplog.records if "runs 4 requests at once" in r.getMessage()]
    assert len(warnings) == 1
