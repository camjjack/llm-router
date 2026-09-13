"""Config reload: new settings take effect without disturbing work in flight."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
from pathlib import Path

import httpx
import pytest
import yaml
from conftest import running_app
from fake_upstream import FakeUpstream

from llm_router.affinity import SessionMap
from llm_router.config import BackendConfig, HealthConfig, parse_config
from llm_router.proxy import Router, create_app
from llm_router.reload import ConfigReloader
from llm_router.scheduler import NoBackendError, Scheduler

MODEL = "test-model"
SYSTEM = {"role": "system", "content": "You are a coding agent."}

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="inotify is Linux-only"
)


def backend_entry(upstream: FakeUpstream, **overrides) -> dict:
    entry = {
        "name": upstream.name,
        "url": upstream.url,
        "capacity": upstream.max_concurrency,
        "models": [upstream.model],
    }
    entry.update(overrides)
    return entry


def config_text(backends: list[dict], aliases: dict | None = None, **sections) -> str:
    raw: dict = {
        "backends": backends,
        # Probe rarely: reloads wake the prober themselves, which is under test.
        "health": {"interval_s": 60, "timeout_s": 2},
    }
    if aliases:
        raw["model_aliases"] = aliases
    raw.update(sections)
    return yaml.safe_dump(raw, sort_keys=False)


def chat(seed: str = "a", model: str = MODEL, stream: bool = False) -> dict:
    body = {
        "model": model,
        "messages": [SYSTEM, {"role": "user", "content": f"task {seed}"}],
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


async def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)


@contextlib.asynccontextmanager
async def stack(upstreams: list[FakeUpstream], text_fn, reloader_kwargs=None, tmp_path=None):
    """Start upstreams, then a router on the config `text_fn(upstreams)` describes.

    With `tmp_path`, the config lives in a file and a ConfigReloader watches it.
    """
    for upstream in upstreams:
        await upstream.start()
    text = text_fn(upstreams)
    config = parse_config(text)
    router = Router(config)
    reloader = None
    if tmp_path is not None:
        path = tmp_path / "config.yaml"
        path.write_text(text)
        reloader = ConfigReloader(path, router, config, **(reloader_kwargs or {}))
    app = create_app(config, router, reloader)
    try:
        async with running_app(app) as base_url:
            async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
                yield client, router, reloader
    finally:
        for upstream in upstreams:
            await upstream.stop()


# ------------------------------------------------------------------- scheduler


def make_backends(*specs: tuple[str, int], url_suffix: str = "") -> tuple[BackendConfig, ...]:
    return tuple(
        BackendConfig(name=name, url=f"http://{name}{url_suffix}", capacity=cap, models=(MODEL,))
        for name, cap in specs
    )


async def test_unchanged_backend_keeps_its_state_and_takes_new_settings():
    scheduler = Scheduler(make_backends(("a", 2)), HealthConfig())
    lease = await scheduler.acquire(MODEL)
    before = scheduler.backends["a"]

    diff = scheduler.reconfigure(make_backends(("a", 4)), HealthConfig())

    assert diff.updated == {"a": ["capacity"]}
    state = scheduler.backends["a"]
    assert state is before, "a capacity change must not reset in-flight accounting"
    assert state.inflight == 1 and state.capacity == 4
    lease.release()
    assert state.inflight == 0


async def test_capacity_cut_below_inflight_blocks_new_work_until_drained():
    scheduler = Scheduler(make_backends(("a", 3)), HealthConfig())
    await scheduler.start()
    try:
        leases = [await scheduler.acquire(MODEL) for _ in range(3)]
        scheduler.reconfigure(make_backends(("a", 1)), HealthConfig())

        waiter = asyncio.create_task(scheduler.acquire(MODEL, timeout_s=5))
        await asyncio.sleep(0.1)
        leases[0].release()
        leases[1].release()
        await asyncio.sleep(0.1)
        assert not waiter.done(), "granted a slot while still over the new capacity"

        leases[2].release()
        granted = await asyncio.wait_for(waiter, 2)
        assert scheduler.backends["a"].inflight == 1
        granted.release()
    finally:
        await scheduler.stop()


async def test_removed_backend_drains_and_takes_no_new_work():
    scheduler = Scheduler(make_backends(("a", 2), ("b", 2)), HealthConfig())
    drained = []
    scheduler.on_drained = lambda state: drained.append(state.name)
    on_a = await scheduler.acquire(MODEL, preferred="a")
    assert on_a.name == "a"

    diff = scheduler.reconfigure(make_backends(("b", 2)), HealthConfig())

    assert diff.removed == ["a"]
    assert "a" not in scheduler.backends
    assert [s.name for s in scheduler.draining] == ["a"]
    # New work, even pinned to the old host, goes to what is left.
    other = await scheduler.acquire(MODEL, preferred="a")
    assert other.name == "b"

    on_a.release()
    assert drained == ["a"] and scheduler.draining == []
    assert on_a.backend.inflight == 0
    other.release()


async def test_backend_put_straight_back_still_counts_its_draining_work():
    """Comment a backend out and straight back in while it is streaming: the
    stream is still load on that host, so capacity must still account for it."""
    scheduler = Scheduler(make_backends(("a", 1), ("b", 1)), HealthConfig())
    streaming = await scheduler.acquire(MODEL, preferred="a")

    scheduler.reconfigure(make_backends(("b", 1)), HealthConfig())
    scheduler.reconfigure(make_backends(("a", 1), ("b", 1)), HealthConfig())

    state = scheduler.backends["a"]
    assert state is streaming.backend and not state.draining
    assert state.free == 0, "handed a full host another slot"
    assert scheduler.draining == []
    streaming.release()
    assert state.free == 1


async def test_repointed_backend_is_a_new_host_under_the_same_name():
    scheduler = Scheduler(make_backends(("a", 1)), HealthConfig())
    old_lease = await scheduler.acquire(MODEL)
    old_state = old_lease.backend

    diff = scheduler.reconfigure(make_backends(("a", 1), url_suffix=":9"), HealthConfig())

    assert diff.replaced == ["a"]
    new_state = scheduler.backends["a"]
    assert new_state is not old_state
    # Unproven until probed.
    assert new_state.healthy is False
    scheduler.set_healthy(new_state, True)

    # The old host's in-flight request does not count against the new host...
    assert new_state.free == 1
    new_lease = await scheduler.acquire(MODEL)
    assert new_lease.backend is new_state
    # ...and releasing it frees the old host's slot, not the new one's.
    old_lease.release()
    assert old_state.inflight == 0 and new_state.inflight == 1
    new_lease.release()


async def test_waiters_for_a_removed_model_fail_immediately():
    backends = (
        BackendConfig(name="a", url="http://a", capacity=1, models=(MODEL,)),
        BackendConfig(name="b", url="http://b", capacity=1, models=("other",)),
    )
    scheduler = Scheduler(backends, HealthConfig())
    await scheduler.start()
    try:
        blocker = await scheduler.acquire(MODEL)
        waiter = asyncio.create_task(
            scheduler.acquire(MODEL, timeout_s=30, unavailable_grace_s=30)
        )
        await asyncio.sleep(0.05)

        scheduler.reconfigure(backends[1:], HealthConfig())

        with pytest.raises(NoBackendError):
            await asyncio.wait_for(waiter, 1)
        blocker.release()
    finally:
        await scheduler.stop()


async def test_capacity_increase_places_queued_work_at_once():
    scheduler = Scheduler(make_backends(("a", 1)), HealthConfig())
    await scheduler.start()
    try:
        held = await scheduler.acquire(MODEL)
        waiter = asyncio.create_task(scheduler.acquire(MODEL, timeout_s=5))
        await asyncio.sleep(0.05)
        assert not waiter.done()

        scheduler.reconfigure(make_backends(("a", 2)), HealthConfig())
        granted = await asyncio.wait_for(waiter, 1)
        assert granted.name == "a"
        held.release()
        granted.release()
    finally:
        await scheduler.stop()


def test_session_map_forgets_pins_to_named_backends_only():
    sessions = SessionMap()
    sessions.assign(["k1"], "a", min_depth=1)
    sessions.assign(["k2"], "b", min_depth=1)
    assert sessions.forget_backends({"a"}) == 1
    assert sessions.lookup(["k1"], min_depth=1) is None
    assert sessions.lookup(["k2"], min_depth=1) == "b"


def test_session_map_reconfigure_applies_new_limits():
    sessions = SessionMap(max_entries=10, depth=3)
    for i in range(10):
        sessions.assign([f"k{i}"], "a", min_depth=1)
    sessions.reconfigure(ttl_s=60, max_entries=4, depth=1)
    assert len(sessions) == 4


# ---------------------------------------------------------------- end to end


async def test_stream_survives_its_backend_being_removed():
    """The headline promise: an edit never cuts off a response already under way."""
    a = FakeUpstream(name="a", model=MODEL, latency_s=1.0, chunks=5)
    b = FakeUpstream(name="b", model=MODEL, latency_s=0.01)

    async with stack([a, b], lambda us: config_text([backend_entry(us[0])])) as (
        client, router, _
    ):
        old_client = router.clients.client("a")

        async def stream_one() -> bytes:
            async with client.stream(
                "POST", "/v1/chat/completions", json=chat("s", stream=True)
            ) as response:
                assert response.headers["x-llm-router-backend"] == "a"
                return b"".join([chunk async for chunk in response.aiter_raw()])

        streaming = asyncio.create_task(stream_one())
        await wait_for(lambda: router.scheduler.backends["a"].inflight == 1)

        router.apply_config(parse_config(config_text([backend_entry(b)])))

        # Old host is draining in the dashboard; new work goes to the new one.
        snapshot = router.snapshot()
        assert [(e["name"], e["draining"]) for e in snapshot["backends"]] == [
            ("b", False),
            ("a", True),
        ]
        await wait_for(lambda: router.scheduler.backends["b"].healthy)
        response = await client.post("/v1/chat/completions", json=chat("n"))
        assert response.status_code == 200
        assert response.headers["x-llm-router-backend"] == "b"
        assert not old_client.is_closed, "closed a client with a stream still on it"

        body = await asyncio.wait_for(streaming, 10)
        assert body.count(b"data: ") == 7 and body.endswith(b"data: [DONE]\n\n")

        await wait_for(lambda: router.scheduler.draining == [])
        await wait_for(lambda: old_client.is_closed)
        assert "a" not in router.stats.backends


async def test_renamed_model_keeps_working_through_an_alias():
    """Pointing at a new model: clients still asking for the old name follow the
    alias, and nothing has to be restarted on either side."""
    old = FakeUpstream(name="old", model="qwen-old", latency_s=0.01)
    new = FakeUpstream(name="new", model="qwen-new", latency_s=0.01)

    async with stack(
        [old, new], lambda us: config_text([backend_entry(us[0])])
    ) as (client, router, _):
        first = await client.post("/v1/chat/completions", json=chat(model="qwen-old"))
        assert first.headers["x-llm-router-backend"] == "old"

        router.apply_config(
            parse_config(
                config_text([backend_entry(new)], aliases={"qwen-old": "qwen-new"})
            )
        )
        await wait_for(lambda: router.scheduler.backends["new"].healthy)

        after = await client.post("/v1/chat/completions", json=chat(model="qwen-old"))
        assert after.status_code == 200
        assert after.headers["x-llm-router-backend"] == "new"
        # The backend was sent the model it actually serves.
        assert new.body_log[-1]["model"] == "qwen-new"
        models = (await client.get("/v1/models")).json()["data"]
        assert [m["id"] for m in models] == ["qwen-new"]


async def test_queued_request_follows_the_new_alias():
    """A request waiting for a slot when its model is swapped out is re-resolved
    against the new aliases instead of failing."""
    old = FakeUpstream(name="old", model="m-old", max_concurrency=1, latency_s=1.0)
    new = FakeUpstream(name="new", model="m-new", latency_s=0.01)

    async with stack(
        [old, new], lambda us: config_text([backend_entry(us[0])])
    ) as (client, router, _):
        busy = asyncio.create_task(
            client.post("/v1/chat/completions", json=chat("1", model="m-old"))
        )
        await wait_for(lambda: router.scheduler.backends["old"].inflight == 1)
        queued = asyncio.create_task(
            client.post("/v1/chat/completions", json=chat("2", model="m-old"))
        )
        await wait_for(lambda: router.scheduler.queue_depth == 1)

        new_config = parse_config(
            config_text([backend_entry(new)], aliases={"*": "m-new"})
        )
        # Healthy before the swap, so the re-resolved request can be placed at once.
        router.apply_config(new_config)
        await wait_for(lambda: router.scheduler.backends["new"].healthy)

        response = await asyncio.wait_for(queued, 5)
        assert response.status_code == 200, response.text
        assert response.headers["x-llm-router-backend"] == "new"
        assert (await busy).headers["x-llm-router-backend"] == "old"


async def test_lease_granted_just_before_a_reload_is_not_sent_to_a_stale_host():
    """The narrow race: a slot is granted, and before the request uses it a reload
    removes that backend. It must be re-placed, never sent through a stale client."""
    a = FakeUpstream(name="a", model=MODEL, latency_s=0.01)
    b = FakeUpstream(name="b", model=MODEL, latency_s=0.01)

    async with stack([a, b], lambda us: config_text([backend_entry(u) for u in us])) as (
        client, router, _
    ):
        real_acquire = router.scheduler.acquire
        swapped = False

        async def acquire_then_reload(*args, **kwargs):
            nonlocal swapped
            lease = await real_acquire(*args, **kwargs)
            if not swapped:
                swapped = True
                others = [backend_entry(u) for u in (a, b) if u.name != lease.name]
                router.apply_config(parse_config(config_text(others)))
            return lease

        router.scheduler.acquire = acquire_then_reload
        response = await client.post("/v1/chat/completions", json=chat())

        assert response.status_code == 200
        survivor = next(iter(router.scheduler.backends))
        assert response.headers["x-llm-router-backend"] == survivor
        removed = a if survivor == "b" else b
        assert removed.total_requests == 0
        assert router.scheduler.draining == []


async def test_pins_survive_a_reload_that_leaves_their_backend_alone():
    a = FakeUpstream(name="a", model=MODEL, latency_s=0.01)
    b = FakeUpstream(name="b", model=MODEL, latency_s=0.01)

    async with stack([a, b], lambda us: config_text([backend_entry(u) for u in us])) as (
        client, router, _
    ):
        headers = {"x-session-id": "conv-1"}
        first = await client.post("/v1/chat/completions", json=chat(), headers=headers)
        pinned = first.headers["x-llm-router-backend"]

        # Unrelated edit: the other backend's capacity.
        edited = [
            backend_entry(u, capacity=u.max_concurrency + (u.name != pinned))
            for u in (a, b)
        ]
        router.apply_config(parse_config(config_text(edited)))

        for _ in range(3):
            again = await client.post("/v1/chat/completions", json=chat(), headers=headers)
            assert again.headers["x-llm-router-backend"] == pinned


async def test_repointing_a_backend_drops_its_pins():
    a = FakeUpstream(name="a", model=MODEL, latency_s=0.01)
    elsewhere = FakeUpstream(name="elsewhere", model=MODEL, latency_s=0.01)

    async with stack([a, elsewhere], lambda us: config_text([backend_entry(us[0])])) as (
        client, router, _
    ):
        await client.post("/v1/chat/completions", json=chat(), headers={"x-session-id": "c"})
        assert router.sessions.backend_pin_counts() == {"a": 1}

        # Same name, different host: the KV the pin pointed at is not there.
        router.apply_config(
            parse_config(config_text([backend_entry(a, url=elsewhere.url)]))
        )
        assert router.sessions.backend_pin_counts() == {}
        await wait_for(lambda: router.scheduler.backends["a"].healthy)
        await client.post("/v1/chat/completions", json=chat(), headers={"x-session-id": "c"})
        assert elsewhere.total_requests == 1


async def test_capacity_increase_gets_a_connection_pool_to_match():
    a = FakeUpstream(name="a", model=MODEL, max_concurrency=8, latency_s=0.01)

    async with stack([a], lambda us: config_text([backend_entry(us[0], capacity=2)])) as (
        client, router, _
    ):
        before = router.clients.client("a")
        router.apply_config(parse_config(config_text([backend_entry(a, capacity=8)])))
        after = router.clients.client("a")
        assert after is not before
        # Not in use, so retired and closed straight away.
        await wait_for(lambda: before.is_closed)
        response = await client.post("/v1/chat/completions", json=chat())
        assert response.status_code == 200


async def test_stranded_model_name_is_called_out(caplog):
    old = FakeUpstream(name="old", model="m-old", latency_s=0.01)
    new = FakeUpstream(name="new", model="m-new", latency_s=0.01)

    async with stack([old, new], lambda us: config_text([backend_entry(us[0])])) as (
        client, router, _
    ):
        await client.post("/v1/chat/completions", json=chat(model="m-old"))
        with caplog.at_level(logging.WARNING, logger="llm_router.proxy"):
            router.apply_config(parse_config(config_text([backend_entry(new)])))
        assert any(
            "'m-old'" in r.message and "model_aliases" in r.message for r in caplog.records
        )


# -------------------------------------------------------------------- reloader


async def test_invalid_edit_is_rejected_and_the_old_config_keeps_running(tmp_path):
    a = FakeUpstream(name="a", model=MODEL, latency_s=0.01)

    async with stack(
        [a], lambda us: config_text([backend_entry(us[0])]), tmp_path=tmp_path,
        reloader_kwargs={"watch": False},
    ) as (client, router, reloader):
        path = tmp_path / "config.yaml"
        good = path.read_text()
        path.write_text(good.replace("capacity: 2", "capacity: 0"))

        assert reloader.check() is False
        assert router.config_generation == 1
        assert "capacity" in router.snapshot()["config"]["error"]
        response = await client.post("/v1/chat/completions", json=chat())
        assert response.status_code == 200

        # Reverting the bad edit clears the error without a pointless reload.
        path.write_text(good)
        assert reloader.check() is False
        assert router.config_error is None
        assert router.config_generation == 1


async def test_unchanged_file_is_a_no_op_unless_forced(tmp_path):
    a = FakeUpstream(name="a", model=MODEL, latency_s=0.01)

    async with stack(
        [a], lambda us: config_text([backend_entry(us[0])]), tmp_path=tmp_path,
        reloader_kwargs={"watch": False},
    ) as (_client, router, reloader):
        (tmp_path / "config.yaml").touch()
        assert reloader.check() is False
        assert reloader.check(force=True) is True
        assert router.config_generation == 2


async def test_listen_changes_are_kept_until_restart(tmp_path, caplog):
    a = FakeUpstream(name="a", model=MODEL, latency_s=0.01)

    async with stack(
        [a], lambda us: config_text([backend_entry(us[0])]), tmp_path=tmp_path,
        reloader_kwargs={"watch": False},
    ) as (_client, router, reloader):
        path = tmp_path / "config.yaml"
        text = config_text([backend_entry(a)], listen={"host": "127.0.0.9", "port": 9999})
        path.write_text(text)
        with caplog.at_level(logging.WARNING, logger="llm_router.reload"):
            assert reloader.check() is True
        assert (router.config.host, router.config.port) == ("0.0.0.0", 8080)
        assert any("after a restart" in r.message for r in caplog.records)


async def _assert_edit_is_picked_up(tmp_path, reloader_kwargs, write) -> None:
    a = FakeUpstream(name="a", model=MODEL, latency_s=0.01)
    async with stack(
        [a], lambda us: config_text([backend_entry(us[0])]), tmp_path=tmp_path,
        reloader_kwargs=reloader_kwargs,
    ) as (_client, router, _reloader):
        write(tmp_path / "config.yaml", config_text([backend_entry(a, capacity=5)]))
        await wait_for(lambda: router.scheduler.backends["a"].capacity == 5)
        assert router.config_generation == 2


async def test_polling_picks_up_an_edit(tmp_path):
    await _assert_edit_is_picked_up(
        tmp_path,
        {"use_inotify": False, "poll_interval_s": 0.05},
        lambda path, text: path.write_text(text),
    )


def _write_in_place(path: Path, text: str) -> None:
    path.write_text(text)


def _write_and_rename(path: Path, text: str) -> None:
    """How vim, most other editors, and config-management tools save."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


@linux_only
@pytest.mark.parametrize("write", [_write_in_place, _write_and_rename])
async def test_inotify_picks_up_an_edit(tmp_path, write):
    # A poll interval far longer than the test, so only inotify can pass it.
    await _assert_edit_is_picked_up(tmp_path, {"poll_interval_s": 3600}, write)


@linux_only
async def test_inotify_follows_a_configmap_style_symlink_swap(tmp_path):
    """Kubernetes updates a mounted ConfigMap by swapping a `..data` symlink to a
    fresh directory. The file's own name never sees an event."""
    a = FakeUpstream(name="a", model=MODEL, latency_s=0.01)
    await a.start()
    try:
        def publish(version: str, text: str) -> None:
            target = tmp_path / f"..{version}"
            target.mkdir()
            (target / "config.yaml").write_text(text)
            tmp_link = tmp_path / "..data_tmp"
            tmp_link.symlink_to(target.name)
            os.replace(tmp_link, tmp_path / "..data")

        publish("v1", config_text([backend_entry(a)]))
        (tmp_path / "config.yaml").symlink_to("..data/config.yaml")

        config = parse_config((tmp_path / "config.yaml").read_text())
        router = Router(config)
        reloader = ConfigReloader(
            tmp_path / "config.yaml", router, config, poll_interval_s=3600
        )
        async with running_app(create_app(config, router, reloader)):
            assert router.config_watch == "inotify"
            publish("v2", config_text([backend_entry(a, capacity=5)]))
            await wait_for(lambda: router.scheduler.backends["a"].capacity == 5)
    finally:
        await a.stop()


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
async def test_sighup_forces_a_reload(tmp_path):
    a = FakeUpstream(name="a", model=MODEL, latency_s=0.01)

    async with stack(
        [a], lambda us: config_text([backend_entry(us[0])]), tmp_path=tmp_path,
        reloader_kwargs={"watch": False, "handle_sighup": True},
    ) as (_client, router, _reloader):
        assert router.config_watch == "SIGHUP only"
        os.kill(os.getpid(), signal.SIGHUP)
        await wait_for(lambda: router.config_generation == 2)
