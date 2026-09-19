"""Per-user session tracking and the web dashboard.

The unit tests drive the tracker directly. The end-to-end ones put real HTTP
through the router, into fake backends, and watch each request move through its
states as the dashboard would see it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

import httpx
import pytest
from conftest import running_app
from fake_upstream import FakeUpstream
from starlette.datastructures import Headers

from llm_router.affinity import explicit_session
from llm_router.config import (
    BackendConfig,
    Config,
    ConfigError,
    HealthConfig,
    RoutingConfig,
    TrackingConfig,
    parse_config,
)
from llm_router.dashboard import QuietAccessLog, SnapshotCache
from llm_router.proxy import Router, create_app
from llm_router.surfaces import ANTHROPIC, OPENAI
from llm_router.tracking import (
    CANCELLED,
    ERROR,
    OK,
    SessionTracker,
    inferred_session_id,
)

MODEL = "test-model"
SYSTEM = {"role": "system", "content": "You are a coding agent."}


# ------------------------------------------------------------------ helpers


def raw(headers: dict[str, str]) -> list[tuple[bytes, bytes]]:
    return [(k.lower().encode(), v.encode()) for k, v in headers.items()]


def begin(tracker, body, headers=None, address="10.0.0.5", surface=OPENAI):
    headers = headers or {}
    explicit = explicit_session(body, Headers(headers))
    keys = [explicit.key] if explicit else surface.session_keys(body)
    return tracker.begin(raw(headers), address, body, body.get("model", MODEL), explicit, keys)


def chat(opening: str, turns: int = 0, system: bool = True) -> dict:
    messages = [SYSTEM] if system else []
    messages.append({"role": "user", "content": opening})
    for i in range(turns):
        messages.append({"role": "assistant", "content": f"step {i}"})
        messages.append({"role": "user", "content": f"result {i}"})
    return {"model": MODEL, "messages": messages}


def sessions(tracker) -> list[dict]:
    return tracker.snapshot()["sessions"]


@contextlib.asynccontextmanager
async def router_stack(upstreams, tracking: TrackingConfig | None = None, **routing):
    for upstream in upstreams:
        await upstream.start()
    config = Config(
        backends=tuple(
            BackendConfig(name=u.name, url=u.url, capacity=u.max_concurrency, models=(u.model,))
            for u in upstreams
        ),
        routing=RoutingConfig(**routing),
        health=HealthConfig(interval_s=60, timeout_s=2),
        tracking=tracking or TrackingConfig(),
    )
    router = Router(config)
    app = create_app(config, router)
    try:
        async with running_app(app) as base_url:
            async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
                yield client, router
    finally:
        for upstream in upstreams:
            await upstream.stop()


async def until(predicate, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.02)
    raise AssertionError("condition never became true")


def in_flight(router) -> list[dict]:
    return router.tracker.snapshot()["in_flight"]


# ----------------------------------------------------------------- identity


def test_open_webui_user_is_named_by_its_forwarded_headers():
    tracker = SessionTracker(TrackingConfig())
    begin(tracker, chat("hello"), {
        "x-openwebui-user-email": "jorg@example.com",
        "x-openwebui-user-id": "8d2f",
        # Open WebUI percent-encodes names so non-ASCII survives the header.
        "x-openwebui-user-name": "J%C3%B6rg M%C3%BCller",
        "x-openwebui-chat-id": "chat-1",
    })
    [user] = tracker.snapshot()["users"]
    assert user["name"] == "Jörg Müller"
    assert user["user"] == "u:jorg@example.com"
    assert user["via"] == "open-webui"
    assert user["clients"] == ["Open WebUI"]
    [session] = sessions(tracker)
    assert session["id"] == "chat-1"
    assert session["via"] == "open-webui"


def test_coding_agent_is_named_by_header_or_else_by_address():
    tracker = SessionTracker(TrackingConfig())
    begin(tracker, chat("a"), {"x-llm-router-user": "alice", "user-agent": "opencode/0.9"})
    begin(tracker, chat("b"), {"user-agent": "claude-cli/2.1.0 (external, cli)"}, address="10.0.0.9")
    users = {u["name"]: u for u in tracker.snapshot()["users"]}
    assert users["alice"]["via"] == "header"
    assert users["alice"]["clients"] == ["opencode"]
    assert users["10.0.0.9"]["via"] == "address"
    assert users["10.0.0.9"]["clients"] == ["Claude Code"]


def test_user_headers_are_configurable():
    tracker = SessionTracker(TrackingConfig(user_headers=("x-team-member",)))
    begin(tracker, chat("a"), {"x-team-member": "bob", "x-llm-router-user": "ignored"})
    assert [u["name"] for u in tracker.snapshot()["users"]] == ["bob"]


def test_claude_code_subagents_belong_to_their_session():
    tracker = SessionTracker(TrackingConfig())
    for agent in (None, "agent-1", "agent-2"):
        headers = {"x-claude-code-session-id": "sess-1"}
        if agent:
            headers["x-claude-code-agent-id"] = agent
        tracker.finish(begin(tracker, chat("x"), headers, surface=ANTHROPIC), 200)
    [session] = sessions(tracker)
    assert session["via"] == "claude-code"
    assert session["client"] == "Claude Code"
    assert session["requests"] == 3
    assert session["agents"] == 2


@pytest.mark.parametrize("system", [True, False])
def test_inferred_session_is_stable_across_turns(system):
    """Without a session id, every turn of one conversation is one session --
    whether or not the client sends a system prompt."""
    tracker = SessionTracker(TrackingConfig())
    for turns in range(4):
        tracker.finish(begin(tracker, chat("fix the bug", turns, system)), 200)
    tracker.finish(begin(tracker, chat("another task", 0, system)), 200)
    listed = sessions(tracker)
    assert len(listed) == 2
    assert sorted(s["requests"] for s in listed) == [1, 4]
    assert all(s["via"] == "inferred" for s in listed)


def test_inferred_session_on_the_anthropic_surface():
    body = {"model": MODEL, "system": "be brief", "messages": [{"role": "user", "content": "hi"}]}
    keys = ANTHROPIC.session_keys(body)
    later = dict(body, messages=[*body["messages"], {"role": "assistant", "content": "yo"},
                                 {"role": "user", "content": "more"}])
    assert inferred_session_id(body, keys) == inferred_session_id(later, ANTHROPIC.session_keys(later))


def test_shared_system_prompt_alone_does_not_merge_sessions():
    tracker = SessionTracker(TrackingConfig())
    for opening in ("task one", "task two", "task three"):
        begin(tracker, chat(opening))
    assert len(sessions(tracker)) == 3


def test_same_opening_from_two_users_stays_two_sessions():
    tracker = SessionTracker(TrackingConfig())
    begin(tracker, chat("hi"), {"x-user": "alice"})
    begin(tracker, chat("hi"), {"x-user": "bob"})
    assert sorted(s["user_name"] for s in sessions(tracker)) == ["alice", "bob"]


def test_header_values_are_bounded():
    tracker = SessionTracker(TrackingConfig())
    begin(tracker, chat("a"), {"x-user": "x" * 5000, "x-session-id": "s" * 5000})
    [session] = sessions(tracker)
    assert len(session["user_name"]) <= 128
    assert len(session["id"]) <= 128


# ------------------------------------------------------------ bookkeeping


def test_finish_aggregates_usage_and_is_idempotent():
    from llm_router.stats import TokenUsage

    tracker = SessionTracker(TrackingConfig())
    tracked = begin(tracker, chat("a"))
    tracked.usage = TokenUsage(prompt_tokens=100, completion_tokens=20, cached_tokens=60)
    tracker.finish(tracked, 200)
    tracker.finish(tracked, 500)  # a second finish changes nothing
    [session] = sessions(tracker)
    assert session["requests"] == 1
    assert session["errors"] == 0
    assert session["prompt_tokens"] == 100
    assert session["completion_tokens"] == 20
    assert session["cache_rate"] == pytest.approx(0.6)
    assert session["last_outcome"] == OK


def test_outcome_follows_status():
    tracker = SessionTracker(TrackingConfig())
    for status, expected in ((200, OK), (404, ERROR), (503, ERROR), (None, CANCELLED)):
        tracked = begin(tracker, chat(f"s{status}"))
        tracker.finish(tracked, status)
        assert tracked.outcome == expected


def test_busy_sessions_are_never_evicted():
    tracker = SessionTracker(TrackingConfig(max_sessions=2))
    busy = begin(tracker, chat("long running"))
    for i in range(10):
        tracker.finish(begin(tracker, chat(f"quick {i}")), 200)
    listed = sessions(tracker)
    assert len(listed) <= 3
    assert busy.session.n in {s["n"] for s in listed}
    tracker.finish(busy, 200)
    assert tracker.snapshot()["summary"]["in_flight"] == 0


def test_idle_sessions_are_forgotten_after_retain_s():
    tracker = SessionTracker(TrackingConfig(retain_s=0.05, slow_after_s=0.01, stuck_after_s=0.02))
    tracker.finish(begin(tracker, chat("old")), 200)
    busy = begin(tracker, chat("still running"))
    time.sleep(0.1)
    listed = sessions(tracker)
    assert [s["n"] for s in listed] == [busy.session.n]
    assert tracker.session_detail(busy.session.n) is not None


def test_stuck_and_slow_are_measured_on_silence():
    tracker = SessionTracker(TrackingConfig(slow_after_s=0.05, stuck_after_s=0.15))
    quiet = begin(tracker, chat("quiet"), {"x-user": "alice"})
    talking = begin(tracker, chat("talking"), {"x-user": "alice"})
    talking.dispatched("a", 0.0)
    time.sleep(0.2)
    talking.received(10)  # bytes are flowing: not stuck, however long it has run
    snap = tracker.snapshot()
    assert snap["summary"]["stuck"] == 1
    assert snap["summary"]["slow"] == 0
    states = {r["id"]: r for r in snap["in_flight"]}
    assert states[quiet.id]["silence_s"] >= 0.15
    assert states[talking.id]["state"] == "streaming"
    assert states[talking.id]["waiting_s"] >= 0.2
    assert states[talking.id]["silence_s"] < 0.05
    [user] = snap["users"]
    assert user["stuck"] == 1 and user["in_flight"] == 2


def test_users_with_something_stuck_are_listed_first():
    tracker = SessionTracker(TrackingConfig(slow_after_s=0.05, stuck_after_s=0.1))
    begin(tracker, chat("waiting"), {"x-user": "quiet"})
    time.sleep(0.15)
    for i in range(3):
        begin(tracker, chat(f"busy {i}"), {"x-user": "busy"})
    assert [u["name"] for u in tracker.snapshot()["users"]] == ["quiet", "busy"]


def test_disabling_tracking_forgets_and_records_nothing():
    tracker = SessionTracker(TrackingConfig())
    running = begin(tracker, chat("a"))
    tracker.reconfigure(TrackingConfig(enabled=False))
    assert tracker.snapshot()["enabled"] is False
    tracker.finish(running, 200)  # started before the switch: quietly ignored
    ignored = begin(tracker, chat("b"))
    ignored.received(5)
    tracker.finish(ignored, 200)
    tracker.reconfigure(TrackingConfig())
    snap = tracker.snapshot()
    assert snap["sessions"] == [] and snap["recent"] == []


def test_snapshot_is_built_at_most_once_per_interval():
    builds = []

    def build():
        builds.append(1)
        return {"n": len(builds)}

    cache = SnapshotCache(build, ttl_s=0.2)
    bodies = {cache.get() for _ in range(100)}
    assert len(builds) == 1 and len(bodies) == 1
    time.sleep(0.25)
    assert cache.get() == b'{"n":2}'


def test_dashboard_polling_is_kept_out_of_the_access_log():
    quiet = QuietAccessLog()

    def record(path):
        return logging.LogRecord(
            "uvicorn.access", logging.INFO, __file__, 1,
            '%s - "%s %s HTTP/%s" %d', ("1.2.3.4:5", "GET", path, "1.1", 200), None,
        )

    assert not quiet.filter(record("/sessions"))
    assert not quiet.filter(record("/sessions/12"))
    assert not quiet.filter(record("/dashboard/app.js"))
    assert quiet.filter(record("/v1/chat/completions"))
    assert quiet.filter(record("/stats"))


# ------------------------------------------------------------------- config


def test_tracking_config_is_parsed_and_validated():
    base = "backends:\n  - {name: a, url: 'http://x', capacity: 1, models: [m]}\n"
    config = parse_config(base + (
        "tracking:\n  user_headers: [X-Team-Member]\n  stuck_after_s: 120\n  max_sessions: 50\n"
    ))
    assert config.tracking.user_headers == ("x-team-member",)
    assert config.tracking.stuck_after_s == 120
    assert config.tracking.max_sessions == 50
    assert parse_config(base).tracking == TrackingConfig()

    for bad in (
        "tracking: {stuck_after_s: 0}",
        "tracking: {slow_after_s: 90, stuck_after_s: 60}",
        "tracking: {max_sessions: 0}",
        "tracking: {enabled: sometimes}",
        "tracking: {user_headers: {a: b}}",
        "tracking: {unknown_key: 1}",
    ):
        with pytest.raises(ConfigError):
            parse_config(base + bad + "\n")


# --------------------------------------------------------------- end to end


async def test_a_request_moves_through_its_states():
    """Queued behind a busy host, then processing, then streaming, then done."""
    upstream = FakeUpstream(name="a", max_concurrency=1, model=MODEL, latency_s=1.2, chunks=4)
    async with router_stack([upstream]) as (client, router):
        body = chat("first", 0) | {"stream": True, "stream_options": {"include_usage": True}}
        first = asyncio.create_task(client.post("/v1/chat/completions", json=body,
                                                headers={"x-user": "alice"}))
        # Dispatched, headers back, no body bytes yet: the fake's first chunk
        # comes after latency/chunks.
        request = await until(lambda: next(iter(in_flight(router)), None))
        assert request["state"] == "processing"
        assert request["backend"] == "a"
        assert request["user_name"] == "alice"

        second = asyncio.create_task(client.post("/v1/chat/completions", json=chat("second"),
                                                 headers={"x-user": "bob"}))
        queued = await until(lambda: [r for r in in_flight(router) if r["user_name"] == "bob"])
        assert queued[0]["state"] == "queued"
        assert queued[0]["backend"] is None

        await until(lambda: any(r["state"] == "streaming" for r in in_flight(router)))
        assert (await first).status_code == 200
        assert (await second).status_code == 200

        snap = router.tracker.snapshot()
        assert snap["in_flight"] == []
        done = {r["user_name"]: r for r in snap["recent"]}
        assert done["alice"]["state"] == OK
        assert done["alice"]["ttft_s"] is not None
        assert done["alice"]["prompt_tokens"] > 0
        assert done["bob"]["queue_s"] > 0.5
        assert done["bob"]["ttft_s"] is None  # not streamed: no first-byte time


async def test_a_pinned_session_shows_as_holding_for_its_host():
    upstream = FakeUpstream(name="a", max_concurrency=1, model=MODEL, latency_s=0.01)
    async with router_stack([upstream], affinity_wait_ms=5000) as (client, router):
        pinned = {"x-session-id": "conversation-1"}
        assert (await client.post("/v1/chat/completions", json=chat("x"), headers=pinned)).status_code == 200

        upstream.latency_s = 1.0
        blocker = asyncio.create_task(client.post("/v1/chat/completions", json=chat("blocker")))
        await until(lambda: router.scheduler.backends["a"].inflight)
        waiting = asyncio.create_task(client.post("/v1/chat/completions", json=chat("x", 1), headers=pinned))
        holding = await until(lambda: [r for r in in_flight(router) if r["session_id"] == "conversation-1"])
        assert holding[0]["state"] == "holding"
        assert holding[0]["pinned"] == "a"
        await blocker
        await waiting


async def test_hanging_up_before_the_first_token_is_recorded_as_cancelled():
    """The client leaves during prefill. Its record must end, and its slot free up."""
    upstream = FakeUpstream(name="a", max_concurrency=1, model=MODEL, latency_s=4.0, chunks=2)
    async with router_stack([upstream]) as (client, router):
        body = chat("slow prefill") | {"stream": True}
        async with client.stream("POST", "/v1/chat/completions", json=body) as response:
            assert response.status_code == 200
            await until(lambda: in_flight(router))
            assert in_flight(router)[0]["state"] == "processing"
            # Leave without reading a byte.

        await until(lambda: router.scheduler.backends["a"].inflight == 0)
        await until(lambda: not in_flight(router))
        [ended] = router.tracker.snapshot()["recent"]
        assert ended["state"] == CANCELLED


async def test_router_rejections_are_recorded_with_a_reason():
    upstream = FakeUpstream(name="a", max_concurrency=1, model=MODEL, latency_s=1.0)
    async with router_stack([upstream], queue_timeout_s=0.2) as (client, router):
        missing = await client.post("/v1/chat/completions", json={"model": "nope", "messages": []},
                                    headers={"x-user": "carol"})
        assert missing.status_code == 404

        blocker = asyncio.create_task(client.post("/v1/chat/completions", json=chat("blocker")))
        await until(lambda: router.scheduler.backends["a"].inflight)
        timed_out = await client.post("/v1/chat/completions", json=chat("impatient"))
        assert timed_out.status_code == 503
        await blocker

        notes = {r["note"] for r in router.tracker.snapshot()["recent"] if r["state"] == ERROR}
        assert notes == {"unknown model", "queue timeout"}
        [carol] = [u for u in router.tracker.snapshot()["users"] if u["name"] == "carol"]
        assert carol["errors"] == 1


async def test_dashboard_endpoints():
    upstream = FakeUpstream(name="a", max_concurrency=2, model=MODEL, latency_s=0.01)
    async with router_stack([upstream]) as (client, _router):
        await client.post("/v1/chat/completions", json=chat("hi"), headers={"x-user": "dana"})

        page = await client.get("/dashboard")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert "script-src 'self'" in page.headers["content-security-policy"]
        assert 'src="dashboard/app.js"' in page.text
        for asset, kind in (("app.js", "javascript"), ("app.css", "css")):
            response = await client.get(f"/dashboard/{asset}")
            assert response.status_code == 200 and kind in response.headers["content-type"]
        assert (await client.get("/dashboard/../proxy.py")).status_code == 404
        assert (await client.get("/dashboard/index.html")).status_code == 404

        data = (await client.get("/sessions")).json()
        assert data["enabled"] is True
        assert data["router"]["queue_depth"] == 0
        assert [b["name"] for b in data["backends"]] == ["a"]
        [session] = data["sessions"]
        assert session["user_name"] == "dana"

        detail = (await client.get(f"/sessions/{session['n']}")).json()
        assert [r["state"] for r in detail["recent"]] == [OK]
        assert (await client.get("/sessions/99999")).status_code == 404


async def test_reload_applies_tracking_settings():
    upstream = FakeUpstream(name="a", max_concurrency=1, model=MODEL, latency_s=0.01)
    async with router_stack([upstream]) as (client, router):
        await client.post("/v1/chat/completions", json=chat("hi"))
        new = Config(
            backends=router.config.backends,
            health=router.config.health,
            tracking=TrackingConfig(enabled=False),
        )
        router.apply_config(new)
        assert router.tracking_snapshot()["enabled"] is False
        assert (await client.post("/v1/chat/completions", json=chat("hi"))).status_code == 200
        assert router.tracker.snapshot()["sessions"] == []
