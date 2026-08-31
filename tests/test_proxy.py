"""End-to-end: the router in front of fake backends that enforce their own limits."""

from __future__ import annotations

import asyncio
import contextlib

import httpx
from conftest import running_app
from fake_upstream import FakeUpstream

from llm_router.config import BackendConfig, Config, HealthConfig, RoutingConfig
from llm_router.proxy import Router, create_app

MODEL = "test-model"
SYSTEM = {"role": "system", "content": "You are a coding agent."}


def build_config(upstreams: list[FakeUpstream], **routing) -> Config:
    return Config(
        backends=tuple(
            BackendConfig(
                name=u.name,
                url=u.url,
                capacity=u.max_concurrency,
                models=(u.model,),
            )
            for u in upstreams
        ),
        routing=RoutingConfig(**routing),
        # Probe rarely: these tests control health explicitly.
        health=HealthConfig(interval_s=60, timeout_s=2),
    )


@contextlib.asynccontextmanager
async def router_stack(upstreams: list[FakeUpstream], **routing):
    for upstream in upstreams:
        await upstream.start()
    router = Router(build_config(upstreams, **routing))
    app = create_app(router.config, router)
    try:
        async with running_app(app) as base_url:
            async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
                yield client, router
    finally:
        for upstream in upstreams:
            await upstream.stop()


def turn(turns: int, seed: str = "a", stream: bool = False) -> dict:
    messages = [SYSTEM, {"role": "user", "content": f"task {seed}"}]
    for i in range(turns):
        messages.append({"role": "assistant", "content": f"step {i}"})
        messages.append({"role": "user", "content": f"result {i}"})
    body = {"model": MODEL, "messages": messages}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


# ---------------------------------------------------------------- the guarantee


async def test_capacity_is_never_exceeded_end_to_end():
    """The headline guarantee, proven by backends that would 429 if we overshot.

    max_pending=0 means these fakes have no internal queue at all: any request the
    router sends beyond capacity is rejected outright rather than quietly absorbed.
    """
    upstreams = [
        FakeUpstream(name="a", max_concurrency=4, max_pending=0, model=MODEL, latency_s=0.02),
        FakeUpstream(name="b", max_concurrency=2, max_pending=0, model=MODEL, latency_s=0.02),
        FakeUpstream(name="c", max_concurrency=4, max_pending=0, model=MODEL, latency_s=0.02),
    ]
    async with router_stack(upstreams) as (client, router):
        responses = await asyncio.gather(
            *(client.post("/v1/chat/completions", json=turn(1, seed=str(i))) for i in range(80))
        )

    assert all(r.status_code == 200 for r in responses), (
        f"statuses: {sorted({r.status_code for r in responses})}"
    )
    for upstream in upstreams:
        assert upstream.max_observed <= upstream.max_concurrency, (
            f"{upstream.name} saw {upstream.max_observed} concurrent, "
            f"capacity {upstream.max_concurrency}"
        )
        assert upstream.overload_responses == 0, f"{upstream.name} had to reject work"

    # Work actually spread out rather than piling on one host.
    assert all(u.total_requests > 0 for u in upstreams)
    assert sum(u.total_requests for u in upstreams) == 80


async def test_idle_backend_is_used_instead_of_queueing():
    """The LiteLLM regression, end to end.

    One host is saturated; a second request must go to the idle host *immediately*
    rather than joining a queue behind the busy one.
    """
    # Both slow, so whichever host takes the first request is still busy when the
    # second arrives. Capacity 1 each, so the first request saturates its host.
    a = FakeUpstream(name="a", max_concurrency=1, max_pending=0, model=MODEL, latency_s=1.0)
    b = FakeUpstream(name="b", max_concurrency=1, max_pending=0, model=MODEL, latency_s=1.0)

    async with router_stack([a, b]) as (client, router):
        first = asyncio.create_task(client.post("/v1/chat/completions", json=turn(1, seed="x")))
        await asyncio.sleep(0.3)
        busy = [n for n, s in router.scheduler.backends.items() if s.inflight]
        assert len(busy) == 1, f"expected exactly one busy host, got {busy}"

        second = asyncio.create_task(client.post("/v1/chat/completions", json=turn(1, seed="y")))
        await asyncio.sleep(0.3)

        # The decisive assertion: it is running, not queued.
        assert router.scheduler.queue_depth == 0, "request queued while a host sat idle"
        running = [n for n, s in router.scheduler.backends.items() if s.inflight]
        assert len(running) == 2, f"idle host was not used: {running}"

        r1, r2 = await asyncio.gather(first, second)
        assert r1.status_code == r2.status_code == 200
        assert r1.headers["x-llm-router-backend"] != r2.headers["x-llm-router-backend"]


# ------------------------------------------------------------------- affinity


async def test_agentic_loop_stays_on_one_backend():
    upstreams = [
        FakeUpstream(name=n, max_concurrency=4, max_pending=0, model=MODEL, latency_s=0.01)
        for n in ("a", "b", "c")
    ]
    async with router_stack(upstreams) as (client, router):
        backends = []
        for t in range(1, 13):
            response = await client.post("/v1/chat/completions", json=turn(t))
            assert response.status_code == 200
            backends.append(response.headers["x-llm-router-backend"])

    assert len(set(backends)) == 1, f"session hopped between hosts: {backends}"

    # And the host it stayed on actually saw growing prefix reuse.
    chosen = next(u for u in upstreams if u.name == backends[0])
    cached = [entry["cached_tokens"] for entry in chosen.request_log]
    assert cached[0] == 0, "first turn cannot hit a cache"
    assert cached[-1] > cached[1] > 0, f"prefix reuse did not grow: {cached}"


async def test_cache_hit_rate_is_reported():
    upstreams = [
        FakeUpstream(name=n, max_concurrency=4, max_pending=0, model=MODEL, latency_s=0.01)
        for n in ("a", "b")
    ]
    async with router_stack(upstreams) as (client, router):
        for t in range(1, 11):
            await client.post("/v1/chat/completions", json=turn(t))
        snapshot = router.snapshot()

    rates = [b["cache_hit_rate"] for b in snapshot["backends"] if b["requests"]]
    assert rates and max(r for r in rates if r is not None) > 0.5, (
        f"expected high prefix reuse on the pinned host, got {rates}"
    )
    assert snapshot["router"]["affinity_honored"] >= 9


async def test_pinned_session_spills_when_its_host_is_saturated():
    fast = FakeUpstream(name="fast", max_concurrency=1, max_pending=0, model=MODEL, latency_s=0.02)
    other = FakeUpstream(name="other", max_concurrency=1, max_pending=0, model=MODEL, latency_s=0.02)

    async with router_stack([fast, other], affinity_wait_ms=200) as (client, router):
        first = await client.post("/v1/chat/completions", json=turn(1))
        pinned = first.headers["x-llm-router-backend"]

        # Occupy the pinned host with a long unrelated request.
        blocker_upstream = next(u for u in (fast, other) if u.name == pinned)
        blocker_upstream.latency_s = 1.5
        blocker = asyncio.create_task(
            client.post("/v1/chat/completions", json=turn(1, seed="blocker"))
        )
        await asyncio.sleep(0.3)

        # The pinned session should hold out briefly, then spill.
        followup = await asyncio.wait_for(
            client.post("/v1/chat/completions", json=turn(2)), timeout=5.0
        )
        assert followup.status_code == 200
        assert followup.headers["x-llm-router-backend"] != pinned
        assert router.stats.affinity_spills >= 1
        await blocker


# ------------------------------------------------------------------ streaming


async def test_streaming_passes_through_and_records_usage():
    upstream = FakeUpstream(name="a", max_concurrency=2, max_pending=0, model=MODEL, chunks=6)
    async with router_stack([upstream]) as (client, router):
        chunks = []
        async with client.stream("POST", "/v1/chat/completions", json=turn(1, stream=True)) as r:
            assert r.status_code == 200
            assert r.headers["x-llm-router-backend"] == "a"
            async for raw in r.aiter_bytes():
                chunks.append(raw)
        await asyncio.sleep(0.05)

        body = b"".join(chunks).decode()
        assert "[DONE]" in body
        assert body.count("chat.completion.chunk") >= 6
        stats = router.stats.backend("a")
        assert stats.completed == 1
        assert stats.prompt_tokens > 0
        assert router.scheduler.backends["a"].inflight == 0


async def test_client_disconnect_releases_the_slot():
    """A leaked slot degrades into exactly the failure this router prevents."""
    upstream = FakeUpstream(
        name="a", max_concurrency=1, max_pending=0, model=MODEL, chunks=50, latency_s=2.0
    )
    async with router_stack([upstream]) as (client, router):
        async with client.stream("POST", "/v1/chat/completions", json=turn(1, stream=True)) as r:
            assert r.status_code == 200
            async for _ in r.aiter_bytes():
                break  # hang up after the first chunk

        for _ in range(100):
            if router.scheduler.backends["a"].inflight == 0:
                break
            await asyncio.sleep(0.05)
        assert router.scheduler.backends["a"].inflight == 0, "slot leaked on disconnect"

        # And the freed slot is genuinely reusable.
        upstream.latency_s = 0.01
        upstream.chunks = 2
        follow = await asyncio.wait_for(
            client.post("/v1/chat/completions", json=turn(1, seed="after")), timeout=5.0
        )
        assert follow.status_code == 200


# -------------------------------------------------------------------- failover


async def test_failover_to_healthy_backend():
    broken = FakeUpstream(name="broken", max_concurrency=2, model=MODEL, fail_with=503)
    good = FakeUpstream(name="good", max_concurrency=2, model=MODEL)

    async with router_stack([broken, good]) as (client, router):
        results = [
            await client.post("/v1/chat/completions", json=turn(1, seed=str(i)))
            for i in range(6)
        ]

    assert all(r.status_code == 200 for r in results)
    assert all(r.headers["x-llm-router-backend"] == "good" for r in results)
    assert router.stats.retries >= 1


async def test_overload_response_is_counted_and_retried():
    """If a backend 429s despite our gate, that is a capacity misconfiguration."""
    # Equally-loaded backends are chosen by name, so the 429 host is tried first.
    overloaded = FakeUpstream(name="a-overloaded", max_concurrency=2, model=MODEL, fail_with=429)
    good = FakeUpstream(name="b-healthy", max_concurrency=2, model=MODEL)

    async with router_stack([overloaded, good]) as (client, router):
        response = await client.post("/v1/chat/completions", json=turn(1))
        assert response.status_code == 200
        assert response.headers["x-llm-router-backend"] == "b-healthy"
        assert router.stats.backend("a-overloaded").overloaded >= 1


async def test_all_backends_failing_surfaces_the_upstream_error():
    a = FakeUpstream(name="a", max_concurrency=2, model=MODEL, fail_with=503)
    b = FakeUpstream(name="b", max_concurrency=2, model=MODEL, fail_with=503)

    async with router_stack([a, b], max_retries=2) as (client, router):
        response = await client.post("/v1/chat/completions", json=turn(1))
    assert response.status_code == 503


async def test_client_error_is_not_retried():
    """A 400 is the client's fault; failing over would just repeat it."""
    a = FakeUpstream(name="a", max_concurrency=2, model=MODEL, fail_with=400)
    b = FakeUpstream(name="b", max_concurrency=2, model=MODEL)

    async with router_stack([a, b]) as (client, router):
        response = await client.post("/v1/chat/completions", json=turn(1))
        assert response.status_code == 400
        assert router.stats.retries == 0
        assert b.total_requests == 0


# ------------------------------------------------------------------- routing


async def test_requests_route_by_model():
    qwen = FakeUpstream(name="qwen-host", max_concurrency=2, model="qwen")
    llama = FakeUpstream(name="llama-host", max_concurrency=2, model="llama")

    async with router_stack([qwen, llama]) as (client, router):
        r1 = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
        r2 = await client.post(
            "/v1/chat/completions",
            json={"model": "llama", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert r1.headers["x-llm-router-backend"] == "qwen-host"
    assert r2.headers["x-llm-router-backend"] == "llama-host"


async def test_unknown_model_is_rejected():
    upstream = FakeUpstream(name="a", max_concurrency=2, model=MODEL)
    async with router_stack([upstream]) as (client, router):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


async def test_malformed_requests_are_rejected():
    upstream = FakeUpstream(name="a", max_concurrency=2, model=MODEL)
    async with router_stack([upstream]) as (client, router):
        assert (await client.post("/v1/chat/completions", content=b"not json")).status_code == 400
        assert (
            await client.post("/v1/chat/completions", json={"messages": []})
        ).status_code == 400


async def test_models_health_and_stats_endpoints():
    a = FakeUpstream(name="a", max_concurrency=2, model="m1")
    b = FakeUpstream(name="b", max_concurrency=4, model="m2")

    async with router_stack([a, b]) as (client, router):
        models = (await client.get("/v1/models")).json()
        assert {m["id"] for m in models["data"]} == {"m1", "m2"}

        health = await client.get("/health")
        assert health.status_code == 200

        stats = (await client.get("/stats")).json()
        assert {entry["name"] for entry in stats["backends"]} == {"a", "b"}
        assert stats["backends"][0]["capacity"] in (2, 4)
        assert "queue_depth" in stats["router"]


async def test_explicit_session_header_pins():
    upstreams = [
        FakeUpstream(name=n, max_concurrency=4, model=MODEL, latency_s=0.01)
        for n in ("a", "b", "c")
    ]
    async with router_stack(upstreams) as (client, router):
        backends = []
        for i in range(8):
            response = await client.post(
                "/v1/chat/completions",
                # Deliberately unrelated message bodies: only the header ties them together.
                json={"model": MODEL, "messages": [{"role": "user", "content": f"q{i}"}]},
                headers={"x-session-id": "my-session"},
            )
            backends.append(response.headers["x-llm-router-backend"])
    assert len(set(backends)) == 1, f"explicit session hopped: {backends}"


async def test_session_id_is_not_forwarded_upstream():
    """`session_id` is ours, not part of the OpenAI schema -- strict backends 400 on it."""
    upstream = FakeUpstream(name="a", max_concurrency=2, model=MODEL)

    async with router_stack([upstream]) as (client, router):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "session_id": "leak-me",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 200
    assert upstream.request_log[-1]["body_keys"] == ["messages", "model"]
