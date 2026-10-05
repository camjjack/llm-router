"""Cache accounting: a backend that says nothing is not a backend that says zero.

vLLM omits cached-token counts unless started with --enable-prompt-tokens-details.
Counting that as a zero hit rate makes a working prefix cache look like broken
affinity, which is the one thing the cache column exists to tell you about.
"""

from __future__ import annotations

import contextlib
import json

import httpx
from conftest import running_app
from fake_upstream import FakeUpstream

from llm_router.config import BackendConfig, Config, HealthConfig
from llm_router.proxy import Router, create_app
from llm_router.stats import BackendStats, TokenUsage
from llm_router.surfaces import ANTHROPIC, OPENAI, StreamTap

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


# ----------------------------------------------------------- errors in streams


def _sse(*events: dict) -> bytes:
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events) + b"data: [DONE]\n\n"


def test_an_error_event_is_caught():
    """TensorFold, like vLLM, ends a failed stream with an error event and [DONE]."""
    tap = StreamTap(OPENAI)
    tap.feed(_sse({"choices": [{"delta": {"content": "hi"}}]},
                  {"error": {"message": "prompt too long", "type": "invalid_request_error"}}))
    assert tap.error == "prompt too long"


def test_an_anthropic_error_event_is_caught():
    tap = StreamTap(ANTHROPIC)
    tap.feed(b'event: error\ndata: {"type": "error", "error": {"type": "overloaded_error", '
             b'"message": "Overloaded"}}\n\n')
    assert tap.error == "Overloaded"


def test_an_error_split_across_chunks_is_still_caught():
    raw = _sse({"error": {"message": "engine fault"}})
    tap = StreamTap(OPENAI)
    for i in range(0, len(raw), 7):
        tap.feed(raw[i:i + 7])
    assert tap.error == "engine fault"


def test_the_word_error_in_a_reply_is_not_an_error():
    """Inside generated text its quotes are escaped, so only a key matches."""
    tap = StreamTap(OPENAI)
    tap.feed(_sse({"choices": [{"delta": {"content": 'raise "error" here'}}]},
                  {"choices": [{"delta": {"tool_calls": [{"function": {
                      "arguments": '{"error": "not a real one"}'}}]}}]},
                  {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}))
    assert tap.error is None
    assert tap.usage.completion_tokens == 2


# -------------------------------------------- SGLang: a miss said by silence


def chat(system: str, stream: bool = False) -> dict:
    body = {"model": MODEL, "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": "go"}]}
    return body | ({"stream": True, "stream_options": {"include_usage": True}} if stream else {})


async def test_sglang_misses_count_once_it_has_reported_a_hit():
    """SGLang leaves cached_tokens out when nothing was cached. Left unknown,
    every miss would drop out of the rate and only hits would count."""
    upstream = FakeUpstream(name="s", model=MODEL, kind="sglang", latency_s=0.01)
    async with router_for(upstream) as (client, router):
        # A miss before it has ever reported: nothing yet says it reports at all.
        await client.post("/v1/chat/completions", json=chat("one"))
        stats = router.stats.backend("s")
        assert list(stats.cache_ratios) == []
        # The same prompt again: a hit, which shows it reports.
        await client.post("/v1/chat/completions", json=chat("one"))
        # A new system prompt, streamed: a miss, now counted as one.
        async with client.stream("POST", "/v1/chat/completions", json=chat("two", stream=True)) as r:
            async for _ in r.aiter_bytes():
                pass
        assert len(stats.cache_ratios) == 2
        assert stats.cache_ratios[0] > 0 and stats.cache_ratios[1] == 0


async def test_sglang_misses_count_on_the_anthropic_surface_too():
    upstream = FakeUpstream(name="s", model=MODEL, kind="sglang", latency_s=0.01)

    def message(opening: str) -> dict:
        return {"model": MODEL, "max_tokens": 16,
                "messages": [{"role": "user", "content": opening}]}

    async with router_for(upstream) as (client, router):
        for opening in ("one", "one", "two"):
            assert (await client.post("/v1/messages", json=message(opening))).status_code == 200
        ratios = list(router.stats.backend("s").cache_ratios)
        assert len(ratios) == 2 and ratios[0] > 0 and ratios[1] == 0


def test_only_sglang_has_its_silence_read_as_a_miss():
    from types import SimpleNamespace

    from llm_router.proxy import _misses_counted

    silent = TokenUsage(prompt_tokens=10, completion_tokens=2)
    reporting = BackendStats(name="b", reports_cache=True)
    for kind, expected in (("sglang", 0), ("vllm", None), ("ninfer", None)):
        state = SimpleNamespace(config=SimpleNamespace(kind=kind))
        assert _misses_counted(state, reporting, silent).cached_tokens == expected
    # Not before it has shown it reports at all.
    state = SimpleNamespace(config=SimpleNamespace(kind="sglang"))
    assert _misses_counted(state, BackendStats(name="b"), silent).cached_tokens is None
