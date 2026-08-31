"""The Anthropic Messages surface -- what Claude Code speaks.

These assert the gateway contract Claude Code documents: forward anthropic-*
headers as an open list, never modify the request body, relay error bodies
unmodified, and never buffer or filter the stream.
"""

from __future__ import annotations

import asyncio
import contextlib

import httpx
from conftest import running_app
from fake_upstream import FakeUpstream

from llm_router.config import BackendConfig, Config, HealthConfig, RoutingConfig
from llm_router.proxy import Router, create_app

MODEL = "test-model"
SYSTEM = [
    {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}}
]


def build_config(upstreams, aliases=None, **routing) -> Config:
    return Config(
        backends=tuple(
            BackendConfig(name=u.name, url=u.url, capacity=u.max_concurrency,
                          models=(u.model,))
            for u in upstreams
        ),
        routing=RoutingConfig(**routing),
        model_aliases=aliases or {},
        health=HealthConfig(interval_s=60, timeout_s=2),
    )


@contextlib.asynccontextmanager
async def router_stack(upstreams, aliases=None, **routing):
    for u in upstreams:
        await u.start()
    config = build_config(upstreams, aliases, **routing)
    router = Router(config)
    app = create_app(config, router)
    try:
        async with running_app(app) as base_url:
            async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
                yield client, router
    finally:
        for u in upstreams:
            await u.stop()


def turn(turns: int = 0, seed: str = "a", stream: bool = False, model: str = MODEL) -> dict:
    """A Claude Code shaped body: system is top-level, not messages[0]."""
    messages = [{"role": "user", "content": f"task {seed}"}]
    for i in range(turns):
        messages.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": f"t{i}", "name": "bash", "input": {"cmd": "ls"}}]})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": f"out {i}"}]})
    body = {"model": model, "max_tokens": 1024, "system": SYSTEM, "messages": messages}
    if stream:
        body["stream"] = True
    return body


CLAUDE_HEADERS = {
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "context-management-2025-06-27,fine-grained-tool-streaming-2025-05-14",
    "x-api-key": "developer-gateway-credential",
    "authorization": "Bearer developer-gateway-credential",
    "x-claude-code-session-id": "sess-123",
}


# ------------------------------------------------------------------ basics


async def test_messages_roundtrip():
    upstream = FakeUpstream(name="a", model=MODEL, max_concurrency=2)
    async with router_stack([upstream]) as (client, router):
        r = await client.post("/v1/messages", json=turn(), headers=CLAUDE_HEADERS)
    assert r.status_code == 200
    assert r.headers["x-llm-router-backend"] == "a"
    body = r.json()
    assert body["type"] == "message"
    assert body["content"][0]["text"] == "ok"


async def test_beta_query_string_is_forwarded():
    """Claude Code posts inference to /v1/messages?beta=true."""
    upstream = FakeUpstream(name="a", model=MODEL, max_concurrency=2)
    async with router_stack([upstream]) as (client, router):
        r = await client.post("/v1/messages?beta=true", json=turn(), headers=CLAUDE_HEADERS)
    assert r.status_code == 200
    assert "beta=true" in upstream.query_log[-1]


async def test_anthropic_headers_forwarded_as_an_open_list():
    """Allowlisting anthropic-* breaks the next Claude Code release."""
    upstream = FakeUpstream(name="a", model=MODEL, max_concurrency=2)
    headers = dict(CLAUDE_HEADERS, **{"anthropic-not-invented-yet": "future-value"})
    async with router_stack([upstream]) as (client, router):
        await client.post("/v1/messages", json=turn(), headers=headers)

    seen = upstream.header_log[-1]
    assert seen["anthropic-version"] == "2023-06-01"
    assert seen["anthropic-beta"] == CLAUDE_HEADERS["anthropic-beta"]
    assert seen["anthropic-not-invented-yet"] == "future-value"


async def test_client_credentials_are_consumed_not_forwarded():
    """The developer's gateway credential must not reach the backend."""
    upstream = FakeUpstream(name="a", model=MODEL, max_concurrency=2)
    async with router_stack([upstream]) as (client, router):
        await client.post("/v1/messages", json=turn(), headers=CLAUDE_HEADERS)
    seen = upstream.header_log[-1]
    assert "developer-gateway-credential" not in seen.get("authorization", "")
    assert "developer-gateway-credential" not in seen.get("x-api-key", "")


async def test_request_body_is_not_modified():
    """Capability headers pair with body fields; breaking a pair is a hard 400.

    The system array in particular must arrive unchanged and still first, or
    Claude Code's attribution block stops being stripped positionally.
    """
    upstream = FakeUpstream(name="a", model=MODEL, max_concurrency=2)
    sent = turn(turns=2)
    sent["context_management"] = {"edits": [{"type": "clear_tool_uses_20250919"}]}
    sent["thinking"] = {"type": "adaptive"}

    async with router_stack([upstream]) as (client, router):
        await client.post("/v1/messages", json=sent, headers=CLAUDE_HEADERS)

    got = upstream.body_log[-1]
    assert got["system"] == SYSTEM, "system array must pass through untouched"
    assert got["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert got["context_management"] == sent["context_management"]
    assert got["thinking"] == sent["thinking"]
    assert got["messages"] == sent["messages"]
    # The model is the one field a gateway is expected to rewrite.
    assert set(got) == set(sent)


async def test_hello_probe_answered():
    upstream = FakeUpstream(name="a", model=MODEL, max_concurrency=2)
    async with router_stack([upstream]) as (client, router):
        assert (await client.head("/api/hello")).status_code == 200


# ---------------------------------------------------------------- streaming


async def test_streaming_events_and_merged_usage():
    """Anthropic splits usage across message_start and message_delta."""
    upstream = FakeUpstream(name="a", model=MODEL, max_concurrency=2, chunks=4)
    async with router_stack([upstream]) as (client, router):
        chunks = []
        async with client.stream("POST", "/v1/messages", json=turn(stream=True),
                                 headers=CLAUDE_HEADERS) as r:
            assert r.status_code == 200
            async for raw in r.aiter_bytes():
                chunks.append(raw)
        await asyncio.sleep(0.05)

        body = b"".join(chunks).decode()
        for event in ("message_start", "content_block_delta", "message_delta", "message_stop"):
            assert event in body, f"missing {event}"

        stats = router.stats.backend("a")
        assert stats.completed == 1
        # input side from message_start, output side from message_delta.
        assert stats.prompt_tokens > 0
        assert stats.completion_tokens == 4
        assert router.scheduler.backends["a"].inflight == 0


async def test_ping_events_are_relayed():
    """Claude Code aborts a stream that goes silent for 300s; pings are the only
    traffic during a thinking pause, so filtering them breaks long generations."""
    upstream = FakeUpstream(name="a", model=MODEL, max_concurrency=2, chunks=4,
                            emit_ping=True)
    async with router_stack([upstream]) as (client, router):
        received = b""
        async with client.stream("POST", "/v1/messages", json=turn(stream=True),
                                 headers=CLAUDE_HEADERS) as r:
            async for raw in r.aiter_bytes():
                received += raw
    assert b"event: ping" in received


async def test_streaming_disconnect_releases_slot():
    upstream = FakeUpstream(name="a", model=MODEL, max_concurrency=1, chunks=50,
                            latency_s=2.0)
    async with router_stack([upstream]) as (client, router):
        async with client.stream("POST", "/v1/messages", json=turn(stream=True),
                                 headers=CLAUDE_HEADERS) as r:
            async for _ in r.aiter_bytes():
                break
        for _ in range(100):
            if router.scheduler.backends["a"].inflight == 0:
                break
            await asyncio.sleep(0.05)
        assert router.scheduler.backends["a"].inflight == 0


# ------------------------------------------------------- capacity + affinity


async def test_capacity_respected_on_messages():
    ups = [
        FakeUpstream(name="a", model=MODEL, max_concurrency=4, max_pending=0, latency_s=0.02),
        FakeUpstream(name="b", model=MODEL, max_concurrency=2, max_pending=0, latency_s=0.02),
    ]
    async with router_stack(ups) as (client, router):
        rs = await asyncio.gather(*(
            client.post("/v1/messages", json=turn(seed=str(i)), headers=CLAUDE_HEADERS)
            for i in range(60)
        ))
    assert all(r.status_code == 200 for r in rs)
    for u in ups:
        assert u.max_observed <= u.max_concurrency
        assert u.overload_responses == 0


async def test_agentic_session_stays_on_one_backend():
    """A Claude Code loop: system is top-level, so messages[0] already identifies
    the conversation and affinity can pin from the very first turn."""
    ups = [FakeUpstream(name=n, model=MODEL, max_concurrency=4, latency_s=0.01)
           for n in ("a", "b", "c")]
    async with router_stack(ups) as (client, router):
        backends = []
        for t in range(1, 11):
            r = await client.post("/v1/messages", json=turn(turns=t), headers=CLAUDE_HEADERS)
            backends.append(r.headers["x-llm-router-backend"])
    assert len(set(backends)) == 1, f"session hopped: {backends}"

    chosen = next(u for u in ups if u.name == backends[0])
    cached = [e["cached_tokens"] for e in chosen.request_log]
    assert cached[-1] > cached[1] > 0, f"prefix reuse did not grow: {cached}"


async def test_cache_hit_rate_normalised_across_surfaces():
    """Anthropic's input_tokens excludes cached tokens; OpenAI's includes them.
    Both must yield the same meaning of 'fraction of prompt reused'."""
    ups = [FakeUpstream(name="a", model=MODEL, max_concurrency=4, latency_s=0.01)]
    async with router_stack(ups) as (client, router):
        for t in range(1, 9):
            await client.post("/v1/messages", json=turn(turns=t), headers=CLAUDE_HEADERS)
        rate = router.stats.backend("a").cache_hit_rate
    assert rate is not None and 0.0 < rate < 1.0, rate
    assert rate > 0.4, f"expected substantial reuse, got {rate}"


# ------------------------------------------------------------------- errors


async def test_upstream_error_body_relayed_unmodified():
    """Claude Code's capability-retry matches on the upstream's error wording."""
    a = FakeUpstream(name="a", model=MODEL, max_concurrency=2, fail_with=400)
    async with router_stack([a]) as (client, router):
        r = await client.post("/v1/messages", json=turn(), headers=CLAUDE_HEADERS)
    assert r.status_code == 400
    assert r.json() == {"type": "error",
                        "error": {"type": "api_error", "message": "forced failure"}}


async def test_router_errors_use_anthropic_shape():
    upstream = FakeUpstream(name="a", model=MODEL, max_concurrency=2)
    async with router_stack([upstream]) as (client, router):
        r = await client.post(
            "/v1/messages",
            json={"model": "nope", "max_tokens": 10, "messages": [{"role": "user", "content": "x"}]},
            headers=CLAUDE_HEADERS,
        )
    assert r.status_code == 404
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "not_found_error"


async def test_529_overload_is_retried_elsewhere():
    """ninfer maps admission overload onto 529 on the Anthropic surface."""
    over = FakeUpstream(name="a-over", model=MODEL, max_concurrency=2, fail_with=529)
    good = FakeUpstream(name="b-good", model=MODEL, max_concurrency=2)
    async with router_stack([over, good]) as (client, router):
        r = await client.post("/v1/messages", json=turn(), headers=CLAUDE_HEADERS)
    assert r.status_code == 200
    assert r.headers["x-llm-router-backend"] == "b-good"
    assert router.stats.backend("a-over").overloaded >= 1


# ------------------------------------------------------- count_tokens, aliases


async def test_count_tokens_proxied_without_taking_a_slot():
    upstream = FakeUpstream(name="a", model=MODEL, max_concurrency=1, latency_s=1.0)
    async with router_stack([upstream]) as (client, router):
        # Occupy the only generation slot.
        busy = asyncio.create_task(
            client.post("/v1/messages", json=turn(seed="busy"), headers=CLAUDE_HEADERS))
        await asyncio.sleep(0.2)
        assert router.scheduler.backends["a"].inflight == 1

        # Token counting must not queue behind it.
        r = await asyncio.wait_for(
            client.post("/v1/messages/count_tokens", json=turn(), headers=CLAUDE_HEADERS),
            timeout=3.0)
        assert r.status_code == 200
        assert r.json()["input_tokens"] > 0
        await busy


async def test_model_alias_maps_claude_names():
    upstream = FakeUpstream(name="a", model=MODEL, max_concurrency=2)
    aliases = {"claude-sonnet-4-5": MODEL, "claude-3-5-haiku-20241022": MODEL}
    async with router_stack([upstream], aliases=aliases) as (client, router):
        r = await client.post("/v1/messages", json=turn(model="claude-sonnet-4-5"),
                              headers=CLAUDE_HEADERS)
    assert r.status_code == 200
    # The backend must receive its own model name, not the alias.
    assert upstream.body_log[-1]["model"] == MODEL


async def test_catch_all_alias_absorbs_background_traffic():
    """Claude Code's small/fast model requests 404 without a catch-all."""
    upstream = FakeUpstream(name="a", model=MODEL, max_concurrency=2)
    async with router_stack([upstream], aliases={"*": MODEL}) as (client, router):
        r = await client.post("/v1/messages", json=turn(model="claude-haiku-4-5-anything"),
                              headers=CLAUDE_HEADERS)
    assert r.status_code == 200
    assert upstream.body_log[-1]["model"] == MODEL


async def test_openai_surface_still_works_alongside():
    """The two surfaces share routing; neither may break the other."""
    upstream = FakeUpstream(name="a", model=MODEL, max_concurrency=2)
    async with router_stack([upstream]) as (client, router):
        oai = await client.post("/v1/chat/completions", json={
            "model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
        ant = await client.post("/v1/messages", json=turn(), headers=CLAUDE_HEADERS)
    assert oai.status_code == 200 and oai.json()["object"] == "chat.completion"
    assert ant.status_code == 200 and ant.json()["type"] == "message"


async def test_claude_code_session_header_pins_exactly():
    """Claude Code supplies a real session id; prefer it over inferring one."""
    ups = [FakeUpstream(name=n, model=MODEL, max_concurrency=4, latency_s=0.01)
           for n in ("a", "b", "c")]
    async with router_stack(ups) as (client, router):
        backends = []
        for i in range(8):
            # Deliberately unrelated bodies: only the session header ties them together.
            body = {"model": MODEL, "max_tokens": 10, "system": SYSTEM,
                    "messages": [{"role": "user", "content": f"unrelated {i}"}]}
            r = await client.post("/v1/messages", json=body,
                                  headers=dict(CLAUDE_HEADERS, **{"x-claude-code-session-id": "s1"}))
            backends.append(r.headers["x-llm-router-backend"])
    assert len(set(backends)) == 1, f"session hopped despite a stable id: {backends}"


async def test_subagents_get_their_own_pins():
    """Subagents share a session id but hold separate conversations."""
    ups = [FakeUpstream(name=n, model=MODEL, max_concurrency=1, latency_s=0.01)
           for n in ("a", "b")]
    async with router_stack(ups) as (client, router):
        seen = {}
        for agent in ("agent-1", "agent-2"):
            for _ in range(3):
                r = await client.post("/v1/messages", json=turn(seed=agent), headers=dict(
                    CLAUDE_HEADERS,
                    **{"x-claude-code-session-id": "s1", "x-claude-code-agent-id": agent}))
                seen.setdefault(agent, []).append(r.headers["x-llm-router-backend"])
    # Each agent is individually sticky...
    for agent, backends in seen.items():
        assert len(set(backends)) == 1, f"{agent} hopped: {backends}"
    # ...and they were free to land on different hosts rather than piling up.
    assert len(router.sessions) >= 2


async def test_session_id_survives_context_compaction():
    """The hashed prefix changes on compaction; the session id does not."""
    ups = [FakeUpstream(name=n, model=MODEL, max_concurrency=4, latency_s=0.01)
           for n in ("a", "b", "c")]
    headers = dict(CLAUDE_HEADERS, **{"x-claude-code-session-id": "s-compact"})
    async with router_stack(ups) as (client, router):
        first = await client.post("/v1/messages", json=turn(turns=6), headers=headers)
        # Compaction rewrites history entirely -- a prefix hash would miss here.
        compacted = {"model": MODEL, "max_tokens": 10, "system": SYSTEM,
                     "messages": [{"role": "user", "content": "summary of earlier work"}]}
        second = await client.post("/v1/messages", json=compacted, headers=headers)
    assert second.headers["x-llm-router-backend"] == first.headers["x-llm-router-backend"]
