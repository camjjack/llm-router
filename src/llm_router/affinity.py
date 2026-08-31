"""Inferring conversation identity so agentic loops stay on one backend.

The OpenAI chat-completions protocol carries no session identifier, and agentic
clients resend the whole conversation every turn. So we derive stickiness from the
messages themselves: hash each message boundary cumulatively, and look those hashes
up longest-first. Turn N+1 is turn N plus a couple of messages, so a boundary
recorded on turn N is still present -- and still hashes the same -- on turn N+1.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

# Boundaries shallower than this are not used for pinning. Depth 1 is just the
# system prompt, which every conversation from a given client shares -- pinning on
# it would funnel every new session onto whichever backend served the last one.
# Depth 2 includes the first user message, which is conversation-specific.
MIN_AFFINITY_DEPTH = 2

# Very large messages (pasted files, big tool outputs) are hashed head+tail+length
# rather than in full, to bound per-request hashing cost. Collision risk is nil.
MAX_MESSAGE_HASH_BYTES = 8192


def _digest(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _canonical(value: Any) -> Any:
    """Reduce a message to a stable, JSON-serializable shape."""
    if isinstance(value, dict):
        return {k: _canonical(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _message_bytes(message: Any) -> bytes:
    try:
        blob = json.dumps(_canonical(message), separators=(",", ":")).encode()
    except (TypeError, ValueError):
        blob = repr(message).encode()
    if len(blob) > MAX_MESSAGE_HASH_BYTES:
        half = MAX_MESSAGE_HASH_BYTES // 2
        # Length is included so two different large messages sharing head and tail
        # still differ.
        blob = blob[:half] + f"|{len(blob)}|".encode() + blob[-half:]
    return blob


def session_keys(body: dict[str, Any]) -> list[str]:
    """Cumulative prefix hashes for a chat-completions body, shallowest first.

    The root folds in the model and tool definitions, since both participate in the
    rendered prompt prefix that the backend actually caches.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return []

    root = _digest(
        json.dumps(
            {
                "model": body.get("model"),
                "tools": _canonical(body.get("tools")),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )

    keys: list[str] = []
    running = root
    for message in messages:
        running = _digest(running.encode() + b"|" + _message_bytes(message))
        keys.append(running)
    return keys


def explicit_session_id(body: dict[str, Any], headers: Any) -> str | None:
    """An explicit session id, if the client is kind enough to supply one."""
    header = None
    if headers is not None:
        header = headers.get("x-session-id") or headers.get("x-conversation-id")
    value = header or body.get("session_id")
    if value and isinstance(value, str):
        return f"explicit:{value}"
    return None


class SessionMap:
    """TTL'd LRU mapping prefix hashes to backend names."""

    def __init__(self, ttl_s: float = 1800.0, max_entries: int = 20000, depth: int = 3):
        self._entries: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._ttl = ttl_s
        self._max = max_entries
        # How many of the deepest boundaries to record per request.
        self._depth = max(1, depth)

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, keys: list[str]) -> str | None:
        """Longest-prefix match: the deepest known boundary wins."""
        now = time.monotonic()
        for index in range(len(keys) - 1, -1, -1):
            if index + 1 < MIN_AFFINITY_DEPTH:
                break
            entry = self._entries.get(keys[index])
            if entry is None:
                continue
            backend, expires_at = entry
            if expires_at < now:
                self._entries.pop(keys[index], None)
                continue
            self._entries.move_to_end(keys[index])
            return backend
        return None

    def assign(self, keys: list[str], backend: str) -> None:
        """Record the deepest boundaries of this request against a backend."""
        if not keys:
            return
        expires_at = time.monotonic() + self._ttl
        start = max(MIN_AFFINITY_DEPTH - 1, len(keys) - self._depth)
        for index in range(start, len(keys)):
            key = keys[index]
            self._entries[key] = (backend, expires_at)
            self._entries.move_to_end(key)
        self._evict()

    def forget(self, keys: list[str]) -> None:
        for key in keys:
            self._entries.pop(key, None)

    def _evict(self) -> None:
        now = time.monotonic()
        # Oldest entries sit at the front; stop at the first live one.
        while self._entries:
            key = next(iter(self._entries))
            if self._entries[key][1] >= now:
                break
            self._entries.popitem(last=False)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def backend_pin_counts(self) -> dict[str, int]:
        """Live pins per backend, for the dashboard."""
        now = time.monotonic()
        counts: dict[str, int] = {}
        for backend, expires_at in self._entries.values():
            if expires_at >= now:
                counts[backend] = counts.get(backend, 0) + 1
        return counts
