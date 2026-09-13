"""Cache accounting: a backend that says nothing is not a backend that says zero.

vLLM omits cached-token counts unless started with --enable-prompt-tokens-details.
Counting that as a zero hit rate makes a working prefix cache look like broken
affinity, which is the one thing the cache column exists to tell you about.
"""

from __future__ import annotations

import contextlib

import httpx
from conftest import running_app
from fake_upstream import FakeUpstream

from llm_router.config import BackendConfig, Config, HealthConfig
from llm_router.proxy import Router, create_app
from llm_router.stats import BackendStats, TokenUsage
from llm_router.surfaces import ANTHROPIC, OPENAI

MODEL = "test-model"


def rate_for(*usages: TokenUsage | None) -> float | None:
    stats = BackendStats(name="b")
    for usage in usages:
        stats.record_usage(usage)
    return stats.cache_hit_rate


# -------------------------------------------------------------- what surfaces read


def test_openai_usage_without_details_is_unknown_not_zero():
    usage = OPENAI.usage_from_body(
        {"usage": {"prompt_tokens": 100, "completion_tokens": 5}}
    )
    assert usage.prompt_tokens == 100
    assert usage.cached_tokens is None
    assert rate_for(usage) is None


def test_openai_reported_zero_is_a_real_zero():
    usage = OPENAI.usage_from_body(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 0},
            }
        }
    )
    assert usage.cached_tokens == 0
    assert rate_for(usage) == 0.0


def test_openai_cache_hits_still_counted():
    usage = OPENAI.usage_from_body(
        {
            "usage": {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 75},
            }
        }
    )
    assert rate_for(usage) == 0.75


def test_anthropic_usage_without_cache_fields_is_unknown():
    usage = ANTHROPIC.usage_from_body({"usage": {"input_tokens": 80, "output_tokens": 3}})
    # Nothing is claimed as cached, so the whole prompt is fresh.
    assert usage.prompt_tokens == 80
    assert usage.cached_tokens is None
    assert rate_for(usage) is None


def test_anthropic_reported_zero_is_a_real_zero():
    usage = ANTHROPIC.usage_from_body(
        {
            "usage": {
                "input_tokens": 80,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            }
        }
    )
    assert usage.cached_tokens == 0
    assert rate_for(usage) == 0.0


def test_anthropic_cache_hits_still_counted():
    usage = ANTHROPIC.usage_from_body(
        {"usage": {"input_tokens": 20, "cache_read_input_tokens": 60}}
    )
    # input_tokens excludes the cached part, so the prompt was 80 tokens.
    assert usage.prompt_tokens == 80
    assert rate_for(usage) == 0.75


def test_unknown_requests_do_not_dilute_a_known_rate():
    """A backend reporting cache counts only sometimes is averaged over the
    requests that actually said something."""
    known = TokenUsage(prompt_tokens=100, cached_tokens=50)
    silent = TokenUsage(prompt_tokens=100, cached_tokens=None)
    assert rate_for(known, silent, known) == 0.5


def test_prompt_tokens_counted_even_when_cache_is_unreported():
    stats = BackendStats(name="b")
    stats.record_usage(TokenUsage(prompt_tokens=100, completion_tokens=7))
    assert (stats.prompt_tokens, stats.completion_tokens) == (100, 7)
    assert stats.cached_tokens == 0 and stats.cache_hit_rate is None


# ------------------------------------------------------------------- end to end


@contextlib.asynccontextmanager
async def router_for(upstream: FakeUpstream):
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
        health=HealthConfig(interval_s=60, timeout_s=2),
    )
    router = Router(config)
    try:
        async with running_app(create_app(config, router)) as base_url:
            async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
                yield client, router
    finally:
        await upstream.stop()


def turn(seed: str) -> dict:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"task {seed}"},
        ],
    }


async def test_vllm_without_prompt_tokens_details_reports_unknown_not_zero():
    """A vLLM started without --enable-prompt-tokens-details: the dashboard must
    show `--`, not a red 0%, since it has said nothing either way."""
    upstream = FakeUpstream(
        name="v", model=MODEL, kind="vllm", latency_s=0.01, report_cache=False
    )
    async with router_for(upstream) as (client, router):
        for _ in range(3):
            assert (await client.post("/v1/chat/completions", json=turn("a"))).status_code == 200

        entry = router.snapshot()["backends"][0]
        assert entry["requests"] == 3
        assert entry["prompt_tokens"] > 0, "token counting still works"
        assert entry["cache_hit_rate"] is None


async def test_backend_reporting_cache_still_shows_a_rate():
    upstream = FakeUpstream(name="n", model=MODEL, latency_s=0.01)
    async with router_for(upstream) as (client, router):
        for _ in range(3):
            await client.post("/v1/chat/completions", json=turn("a"))

        entry = router.snapshot()["backends"][0]
        assert entry["cache_hit_rate"] > 0
