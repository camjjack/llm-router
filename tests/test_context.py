"""Advertising the context window: discovery, pooling, and the llama.cpp trap."""

from __future__ import annotations

import contextlib

import httpx
from conftest import running_app
from fake_upstream import FakeUpstream

from llm_router.config import BackendConfig, Config, HealthConfig
from llm_router.proxy import Router, create_app


@contextlib.asynccontextmanager
async def stack(upstreams: list[FakeUpstream], overrides: dict[str, int] | None = None):
    overrides = overrides or {}
    for upstream in upstreams:
        await upstream.start()
    config = Config(
        backends=tuple(
            BackendConfig(
                name=u.name,
                url=u.url,
                capacity=u.max_concurrency,
                models=(u.model,),
                kind=u.kind,
                context_length=overrides.get(u.name),
            )
            for u in upstreams
        ),
        # Probe fast so discovery completes promptly in tests.
        health=HealthConfig(interval_s=0.2, timeout_s=2),
    )
    router = Router(config)
    app = create_app(config, router)
    try:
        async with running_app(app) as base_url:
            async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
                await _await_discovery(router, [u.name for u in upstreams])
                yield client, router
    finally:
        for upstream in upstreams:
            await upstream.stop()


async def _await_discovery(router: Router, names: list[str], tries: int = 60) -> None:
    import asyncio

    for _ in range(tries):
        if all(router.clients.context_length.get(n) is not None for n in names):
            return
        await asyncio.sleep(0.1)


async def test_ninfer_context_discovered_from_models():
    upstream = FakeUpstream(name="a", model="m", kind="ninfer", context_length=65536)
    async with stack([upstream]) as (client, router):
        assert router.clients.context_length["a"] == 65536
        entry = (await client.get("/v1/models")).json()["data"][0]
        assert entry["context_length"] == 65536
        assert entry["max_model_len"] == 65536
        assert entry["meta"]["n_ctx"] == 65536


async def test_llamacpp_uses_props_not_n_ctx_train():
    """The trap: /v1/models reports the architectural context, /props the served one."""
    upstream = FakeUpstream(
        name="llama",
        model="m",
        kind="llamacpp",
        context_length=16384,   # served, per slot
        n_ctx_train=131072,     # model's architectural max -- must NOT be used
    )
    async with stack([upstream]) as (client, router):
        assert router.clients.context_length["llama"] == 16384, (
            "must read /props, not meta.n_ctx_train"
        )
        entry = (await client.get("/v1/models")).json()["data"][0]
        assert entry["context_length"] == 16384


async def test_llamacpp_falls_back_to_models_without_props():
    upstream = FakeUpstream(
        name="llama",
        model="m",
        kind="llamacpp",
        context_length=16384,
        n_ctx_train=131072,
        props_available=False,
    )
    async with stack([upstream]) as (client, router):
        # Only the architectural figure is available; better than nothing, and the
        # router logs a warning saying so.
        assert router.clients.context_length["llama"] == 131072


async def test_pool_advertises_the_smallest_context():
    """A request can land on any host, so the safe number is the minimum."""
    big = FakeUpstream(name="big", model="m", context_length=131072)
    small = FakeUpstream(name="small", model="m", context_length=32768)
    async with stack([big, small]) as (client, router):
        assert router.context_for("m") == 32768
        entry = (await client.get("/v1/models")).json()["data"][0]
        assert entry["context_length"] == 32768


async def test_config_override_wins_over_discovery():
    upstream = FakeUpstream(name="a", model="m", context_length=65536)
    async with stack([upstream], overrides={"a": 8192}) as (client, router):
        assert router.clients.context_length["a"] == 8192
        entry = (await client.get("/v1/models")).json()["data"][0]
        assert entry["context_length"] == 8192


async def test_unknown_context_is_omitted_not_invented():
    """Better an absent field than a wrong number."""
    upstream = FakeUpstream(name="a", model="m", kind="ninfer", context_length=None)
    for u in [upstream]:
        await u.start()
    config = Config(
        backends=(BackendConfig(name="a", url=upstream.url, capacity=2, models=("m",)),),
        health=HealthConfig(interval_s=0.2, timeout_s=2),
    )
    router = Router(config)
    app = create_app(config, router)
    try:
        async with running_app(app) as base_url:
            async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
                entry = (await client.get("/v1/models")).json()["data"][0]
                assert "context_length" not in entry
                assert "max_model_len" not in entry
                assert router.context_for("m") is None
    finally:
        await upstream.stop()


async def test_per_model_context_in_stats():
    qwen = FakeUpstream(name="q", model="qwen", context_length=65536)
    llama = FakeUpstream(name="l", model="llama", kind="llamacpp", context_length=16384)
    async with stack([qwen, llama]) as (client, router):
        stats = (await client.get("/stats")).json()
        assert stats["models"]["qwen"]["context_length"] == 65536
        assert stats["models"]["llama"]["context_length"] == 16384
        by_name = {b["name"]: b for b in stats["backends"]}
        assert by_name["q"]["context_length"] == 65536
        assert by_name["l"]["context_length"] == 16384


async def test_mismatched_pool_warns_once(caplog):
    big = FakeUpstream(name="big", model="m", context_length=131072)
    small = FakeUpstream(name="small", model="m", context_length=32768)
    async with stack([big, small]) as (client, router):
        with caplog.at_level("WARNING"):
            router.context_for("m")
            router.context_for("m")
            router.context_for("m")
    mismatch = [r for r in caplog.records if "disagree on context window" in r.message]
    assert len(mismatch) <= 1


# ------------------------------------------------- vLLM and LM Studio backends


async def test_vllm_context_from_max_model_len():
    upstream = FakeUpstream(name="v", model="m", kind="vllm", context_length=131072,
                            max_concurrency=256)
    async with stack([upstream]) as (client, router):
        assert router.clients.context_length["v"] == 131072
        entry = (await client.get("/v1/models")).json()["data"][0]
        assert entry["max_model_len"] == 131072


async def test_lmstudio_uses_loaded_not_max_context_length():
    """LM Studio's trap: max_context_length is the ceiling, not the allocation."""
    upstream = FakeUpstream(
        name="lms",
        model="m",
        kind="lmstudio",
        context_length=8192,        # loaded_context_length -- what was allocated
        max_context_length=131072,  # what the model could do -- must NOT be used
    )
    async with stack([upstream]) as (client, router):
        assert router.clients.context_length["lms"] == 8192, (
            "must read loaded_context_length, not max_context_length"
        )
        entry = (await client.get("/v1/models")).json()["data"][0]
        assert entry["context_length"] == 8192


async def test_lmstudio_falls_back_to_max_context_length():
    """Older builds report only the ceiling; better than nothing."""
    upstream = FakeUpstream(
        name="lms", model="m", kind="lmstudio",
        context_length=None, max_context_length=32768,
    )
    async with stack([upstream]) as (client, router):
        assert router.clients.context_length["lms"] == 32768


async def test_lmstudio_liveness_uses_model_list():
    """LM Studio has no /health, so the model list is the liveness signal."""
    upstream = FakeUpstream(name="lms", model="m", kind="lmstudio", context_length=8192)
    async with stack([upstream]) as (client, router):
        assert router.scheduler.backends["lms"].healthy is True
        assert (await client.get("/health")).status_code == 200


async def test_mixed_engine_pool_advertises_the_smallest():
    """A realistic migration: ninfer, vLLM and LM Studio serving the same model."""
    nin = FakeUpstream(name="ninfer", model="m", kind="ninfer", context_length=65536)
    vllm = FakeUpstream(name="vllm", model="m", kind="vllm", context_length=131072,
                        max_concurrency=256)
    lms = FakeUpstream(name="lms", model="m", kind="lmstudio", context_length=8192,
                       max_context_length=131072)
    async with stack([nin, vllm, lms]) as (client, router):
        assert router.context_for("m") == 8192, "LM Studio's 8k allocation is the binding limit"
        by_name = {b["name"]: b for b in (await client.get("/stats")).json()["backends"]}
        assert by_name["ninfer"]["context_length"] == 65536
        assert by_name["vllm"]["context_length"] == 131072
        assert by_name["lms"]["context_length"] == 8192
