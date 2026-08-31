"""Run the whole router against fake backends, under simulated agentic load.

    uv run python scripts/demo.py

Starts three fake ninfer-style hosts (capacity 4, 2 and 4) that enforce their own
concurrency limits, drives a handful of concurrent multi-turn agent sessions
through the router, and shows the live dashboard. Nothing here touches a GPU --
it exists so you can see the routing behaviour, and read the dashboard, before
pointing the router at real hardware.

Watch for: the `cache` column climbing as sessions stick to their host, `pins`
spreading across backends, and `load` bars filling without any host ever being
handed more than its capacity.
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402
from fake_upstream import FakeUpstream  # noqa: E402

from llm_router.config import BackendConfig, Config, HealthConfig, RoutingConfig  # noqa: E402
from llm_router.proxy import Router  # noqa: E402
from llm_router.tui import run_dashboard  # noqa: E402

MODEL = "demo-model"
SESSIONS = int(os.environ.get("DEMO_SESSIONS", "6"))
TURNS_PER_SESSION = int(os.environ.get("DEMO_TURNS", "25"))
SYSTEM = {"role": "system", "content": "You are a coding agent with tools. " + "x" * 400}


async def agent_session(client: httpx.AsyncClient, session: int) -> None:
    """One agentic loop: a conversation that grows by two messages per turn."""
    messages = [SYSTEM, {"role": "user", "content": f"Please refactor module {session}."}]
    await asyncio.sleep(random.uniform(0, 2.0))

    for turn in range(TURNS_PER_SESSION):
        try:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": MODEL, "messages": messages},
                timeout=60.0,
            )
            if response.status_code != 200:
                await asyncio.sleep(1.0)
                continue
        except httpx.HTTPError:
            await asyncio.sleep(1.0)
            continue

        messages.append({"role": "assistant", "content": f"calling tool {turn}"})
        messages.append({"role": "user", "content": f"tool output {turn}: " + "y" * 200})
        # Think time between turns, as a real agent would have.
        await asyncio.sleep(random.uniform(0.3, 1.2))


async def main() -> None:
    upstreams = [
        FakeUpstream(name="ninfer-a", max_concurrency=4, max_pending=0, model=MODEL,
                     latency_s=0.9, chunks=20),
        FakeUpstream(name="ninfer-b", max_concurrency=2, max_pending=0, model=MODEL,
                     latency_s=0.9, chunks=20),
        FakeUpstream(name="llama-1", max_concurrency=4, max_pending=0, model=MODEL,
                     latency_s=1.4, chunks=20),
    ]
    for upstream in upstreams:
        await upstream.start()

    config = Config(
        backends=tuple(
            BackendConfig(name=u.name, url=u.url, capacity=u.max_concurrency, models=(MODEL,))
            for u in upstreams
        ),
        routing=RoutingConfig(affinity_wait_ms=1500),
        health=HealthConfig(interval_s=5),
    )
    router = Router(config)
    await router.start()

    import uvicorn

    from llm_router.proxy import create_app

    app = create_app(config, router)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=8099, log_level="error", access_log=False)
    )
    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.5)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8099") as client:
        load = asyncio.gather(*(agent_session(client, i) for i in range(SESSIONS)))
        dashboard = asyncio.create_task(run_dashboard(router.snapshot))
        try:
            await load
            await asyncio.sleep(2)
        finally:
            dashboard.cancel()
            server.should_exit = True
            await asyncio.gather(serve_task, return_exceptions=True)
            await router.stop()
            for upstream in upstreams:
                await upstream.stop()

    # Final report: the numbers that matter.
    print("\nBackend                requests   max concurrent / capacity   429s")
    for upstream in upstreams:
        flag = "  <-- OVERSHOT" if upstream.max_observed > upstream.max_concurrency else ""
        print(
            f"  {upstream.name:<20} {upstream.total_requests:>5}"
            f"          {upstream.max_observed} / {upstream.max_concurrency}"
            f"                {upstream.overload_responses}{flag}"
        )
    stats = router.snapshot()["router"]
    rate = stats["affinity_honor_rate"]
    print(
        f"\nAffinity: {stats['affinity_honored']} honored, {stats['affinity_spills']} spilled"
        f" ({'--' if rate is None else f'{rate * 100:.0f}%'} honored)"
    )
    for entry in router.snapshot()["backends"]:
        hit = entry["cache_hit_rate"]
        print(
            f"  {entry['name']:<20} prefix reuse "
            f"{'--' if hit is None else f'{hit * 100:.0f}%'}"
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
