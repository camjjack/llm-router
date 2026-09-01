"""Counters and rolling windows backing both the TUI and /stats."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


def _mean(values: deque[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


@dataclass(frozen=True)
class TokenUsage:
    """Token counts normalised across API surfaces.

    `prompt_tokens` is the complete prompt including any cached portion, and
    `cached_tokens` is the part served from a prefix cache. OpenAI reports it that
    way already; Anthropic splits the prompt across input/cache-read/cache-write,
    so its surface sums them.
    """

    prompt_tokens: int
    completion_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class BackendStats:
    """Lifetime counters and recent-performance windows for one backend."""

    name: str
    requests: int = 0
    completed: int = 0
    errors: int = 0
    # Upstream said it was full even though we believed it had a slot. Non-zero here
    # means the configured capacity is higher than the backend's real limit.
    overloaded: int = 0
    # Requests handed to this backend because their pinned backend was busy.
    spilled_in: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0

    ttft_s: deque[float] = field(default_factory=lambda: deque(maxlen=64))
    tokens_per_s: deque[float] = field(default_factory=lambda: deque(maxlen=64))
    # Per-request cache hit ratios, so a few huge prompts cannot dominate the average.
    cache_ratios: deque[float] = field(default_factory=lambda: deque(maxlen=64))

    def record_usage(self, usage: "TokenUsage | None") -> None:
        """Absorb one request's token counts, tolerating missing or odd values.

        Takes already-normalised counts: OpenAI and Anthropic disagree about
        whether cached tokens are included in the prompt total, so the surfaces
        reconcile that before it reaches here.
        """
        if usage is None:
            return
        prompt = usage.prompt_tokens
        completion = usage.completion_tokens
        cached = usage.cached_tokens

        if not isinstance(prompt, int) or prompt < 0:
            return
        if not isinstance(cached, int) or cached < 0:
            cached = 0
        # Guard against a backend reporting more cached than prompt tokens.
        cached = min(cached, prompt)

        self.prompt_tokens += prompt
        self.cached_tokens += cached
        if isinstance(completion, int) and completion > 0:
            self.completion_tokens += completion
        if prompt > 0:
            self.cache_ratios.append(cached / prompt)

    @property
    def cache_hit_rate(self) -> float | None:
        """Mean per-request prefix reuse. The readout for whether affinity is paying."""
        return _mean(self.cache_ratios)

    @property
    def error_rate(self) -> float | None:
        total = self.completed + self.errors
        return (self.errors / total) if total else None

    def snapshot(self, inflight: int, capacity: int, healthy: bool) -> dict:
        return {
            "name": self.name,
            "inflight": inflight,
            "capacity": capacity,
            "healthy": healthy,
            "requests": self.requests,
            "completed": self.completed,
            "errors": self.errors,
            "overloaded": self.overloaded,
            "spilled_in": self.spilled_in,
            "error_rate": self.error_rate,
            "avg_ttft_s": _mean(self.ttft_s),
            "avg_tokens_per_s": _mean(self.tokens_per_s),
            "cache_hit_rate": self.cache_hit_rate,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class RouterStats:
    """Router-wide routing-decision counters."""

    started_at: float = field(default_factory=time.monotonic)
    backends: dict[str, BackendStats] = field(default_factory=dict)

    # Conversation identity taken from a client-supplied header (exact).
    keys_from_header: int = 0
    # Conversation identity inferred by hashing the prompt prefix (fallback).
    keys_from_prefix: int = 0
    # A request arrived with a known session pin.
    affinity_hits: int = 0
    # No pin was known (new conversation, or the pin had expired).
    affinity_misses: int = 0
    # Pinned, and the pinned backend took it -- the outcome we want.
    affinity_honored: int = 0
    # Pinned, but the pin was busy past affinity_wait_ms so we re-pinned elsewhere.
    affinity_spills: int = 0
    # Requests that had to wait for any slot at all.
    queued: int = 0
    queue_timeouts: int = 0
    retries: int = 0
    rejected_no_backend: int = 0

    total_queue_wait_s: float = 0.0

    def backend(self, name: str) -> BackendStats:
        stats = self.backends.get(name)
        if stats is None:
            stats = BackendStats(name=name)
            self.backends[name] = stats
        return stats

    @property
    def affinity_honor_rate(self) -> float | None:
        """Of requests that had a pin, the share that actually landed on it."""
        pinned = self.affinity_honored + self.affinity_spills
        return (self.affinity_honored / pinned) if pinned else None

    @property
    def avg_queue_wait_s(self) -> float | None:
        return (self.total_queue_wait_s / self.queued) if self.queued else None

    def snapshot(self) -> dict:
        return {
            "uptime_s": time.monotonic() - self.started_at,
            "keys_from_header": self.keys_from_header,
            "keys_from_prefix": self.keys_from_prefix,
            "affinity_hits": self.affinity_hits,
            "affinity_misses": self.affinity_misses,
            "affinity_honored": self.affinity_honored,
            "affinity_spills": self.affinity_spills,
            "affinity_honor_rate": self.affinity_honor_rate,
            "queued": self.queued,
            "queue_timeouts": self.queue_timeouts,
            "avg_queue_wait_s": self.avg_queue_wait_s,
            "retries": self.retries,
            "rejected_no_backend": self.rejected_no_backend,
        }
