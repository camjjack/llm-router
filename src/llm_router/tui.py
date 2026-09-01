"""Live terminal dashboard, rendered from the same snapshot that /stats returns."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import httpx
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

REFRESH_HZ = 4


def _bar(
    inflight: int, capacity: int, observed: int | None = None, width: int = 10
) -> Text:
    """Occupancy bar. Amber at capacity is the state worth noticing."""
    if capacity <= 0:
        return Text("-")
    filled = min(width, round(width * inflight / capacity))
    if inflight >= capacity:
        colour = "bold yellow"
    elif inflight:
        colour = "green"
    else:
        colour = "dim"
    bar = Text("█" * filled, style=colour)
    bar.append("░" * (width - filled), style="dim")
    bar.append(f" {inflight}/{capacity}", style=colour)
    # Only llama.cpp and vLLM report their own load. Showing it when it disagrees
    # with ours flags a host something else is also using.
    if observed is not None and observed != inflight:
        bar.append(f" !{observed}", style="bold red")
    return bar


def _context(value: int | None) -> Text:
    """Context window, abbreviated. Dim `?` means we could not discover it."""
    if not value:
        return Text("?", style="dim")
    if value >= 1024 and value % 1024 == 0:
        return Text(f"{value // 1024}k")
    return Text(f"{value / 1024:.1f}k" if value >= 1024 else str(value))


def _pct(value: float | None) -> Text:
    if value is None:
        return Text("--", style="dim")
    style = "green" if value >= 0.5 else "yellow" if value >= 0.2 else "red"
    return Text(f"{value * 100:.0f}%", style=style)


def _num(value: float | None, fmt: str = "{:.2f}", style: str = "") -> Text:
    if value is None:
        return Text("--", style="dim")
    return Text(fmt.format(value), style=style)


def render(snapshot: dict[str, Any]) -> Group:
    router = snapshot.get("router", {})
    backends = snapshot.get("backends", [])

    table = Table(expand=True, header_style="bold", pad_edge=False)
    table.add_column("backend")
    table.add_column("kind", style="dim")
    table.add_column("ctx", justify="right")
    table.add_column("load", min_width=18)
    table.add_column("pins", justify="right")
    table.add_column("reqs", justify="right")
    table.add_column("err", justify="right")
    table.add_column("ttft", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("cache", justify="right")
    table.add_column("spill in", justify="right")

    for entry in backends:
        healthy = entry.get("healthy")
        cooling = entry.get("cooling_down")
        if not healthy:
            name = Text(f"● {entry['name']}", style="bold red")
        elif cooling:
            name = Text(f"● {entry['name']}", style="bold yellow")
        else:
            name = Text(f"● {entry['name']}", style="bold green")

        errors = entry.get("errors", 0)
        overloaded = entry.get("overloaded", 0)
        err_text = Text(str(errors), style="red" if errors else "dim")
        if overloaded:
            # Means our capacity setting is wrong -- call it out loudly.
            err_text.append(f" (!{overloaded})", style="bold red")

        table.add_row(
            name,
            str(entry.get("kind", "")),
            _context(entry.get("context_length")),
            _bar(
                entry.get("inflight", 0),
                entry.get("capacity", 0),
                entry.get("observed_busy"),
            ),
            str(entry.get("pinned_sessions", 0)),
            str(entry.get("requests", 0)),
            err_text,
            _num(entry.get("avg_ttft_s"), "{:.2f}s"),
            _num(entry.get("avg_tokens_per_s"), "{:.0f}"),
            _pct(entry.get("cache_hit_rate")),
            str(entry.get("spilled_in", 0)),
        )

    queue_depth = router.get("queue_depth", 0)
    waiting = router.get("waiting_on_affinity", 0)
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim")
    summary.add_column()
    summary.add_row(
        "queue",
        Text(
            f"{queue_depth} waiting ({waiting} holding for a pinned host)",
            style="yellow" if queue_depth else "dim",
        ),
    )
    summary.add_row(
        "affinity",
        Text.assemble(
            ("honored ", "dim"),
            (str(router.get("affinity_honored", 0)), "green"),
            ("  spilled ", "dim"),
            (str(router.get("affinity_spills", 0)), "yellow"),
            ("  rate ", "dim"),
            _pct(router.get("affinity_honor_rate")),
        ),
    )
    from_header = router.get("keys_from_header", 0)
    from_prefix = router.get("keys_from_prefix", 0)
    summary.add_row(
        "sessions",
        Text(
            f"{router.get('tracked_sessions', 0)} tracked  "
            f"(new {router.get('affinity_misses', 0)} / returning "
            f"{router.get('affinity_hits', 0)})",
            style="dim",
        ),
    )
    summary.add_row(
        "identity",
        Text.assemble(
            (f"{from_header} from header", "green" if from_header else "dim"),
            ("  ", "dim"),
            (f"{from_prefix} inferred from prompt", "dim"),
        ),
    )
    summary.add_row(
        "waits",
        Text.assemble(
            ("avg queue ", "dim"),
            _num(router.get("avg_queue_wait_s"), "{:.2f}s"),
            ("  timeouts ", "dim"),
            (str(router.get("queue_timeouts", 0)), "red" if router.get("queue_timeouts") else "dim"),
            ("  retries ", "dim"),
            (str(router.get("retries", 0)), "dim"),
        ),
    )

    uptime = router.get("uptime_s", 0)
    return Group(
        Panel(
            table,
            title="[bold]llm-router[/bold]",
            subtitle=f"up {uptime / 60:.0f}m",
            border_style="blue",
        ),
        Panel(summary, border_style="dim"),
    )


async def run_dashboard(snapshot_fn: Callable[[], dict], stop: asyncio.Event | None = None) -> None:
    """Drive the dashboard from a local snapshot callable."""
    with Live(render(snapshot_fn()), refresh_per_second=REFRESH_HZ, screen=False) as live:
        while stop is None or not stop.is_set():
            await asyncio.sleep(1 / REFRESH_HZ)
            live.update(render(snapshot_fn()))


async def run_remote_dashboard(url: str) -> None:
    """Drive the dashboard from a running router's /stats endpoint."""
    stats_url = url.rstrip("/") + "/stats"
    async with httpx.AsyncClient(timeout=5.0) as client:
        placeholder = {"router": {}, "backends": []}
        with Live(render(placeholder), refresh_per_second=REFRESH_HZ) as live:
            while True:
                try:
                    response = await client.get(stats_url)
                    live.update(render(response.json()))
                except (httpx.HTTPError, ValueError) as exc:
                    live.update(
                        Panel(
                            Text(f"cannot reach {stats_url}: {exc}", style="red"),
                            border_style="red",
                        )
                    )
                await asyncio.sleep(1 / REFRESH_HZ)
