"""Who is using the router, what each of their sessions is doing, and what is stuck.

This sits on the request path, so it is built to cost next to nothing there: a
request is a handful of attribute writes as it moves from state to state, and
nothing is aggregated, sorted or serialised until a dashboard asks for a snapshot.
Memory is bounded by `max_sessions` however many clients turn up.

Nothing from a prompt is kept. A session is the id its client sends or, failing
that, the prefix hash the router already computes for affinity.
"""

from __future__ import annotations

import itertools
import time
from collections import OrderedDict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

from .affinity import CLAUDE_CODE, ExplicitSession
from .stats import TokenUsage

if TYPE_CHECKING:
    from .config import TrackingConfig

# Where an in-flight request is. HOLDING is never stored: it is a queued request
# still inside its window for waiting on its pinned host, worked out when a
# snapshot is taken.
QUEUED = "queued"
HOLDING = "holding"
# Sent upstream, nothing back yet: prefill, or the whole of a non-streamed reply.
PROCESSING = "processing"
STREAMING = "streaming"

# How a request ended.
OK = "ok"
ERROR = "error"
# The client went away before the response was done.
CANCELLED = "cancelled"

# Headers naming the user, tried in order. Open WebUI sends its X-OpenWebUI-User-*
# set when started with ENABLE_FORWARD_USER_INFO_HEADERS=true. A coding agent can
# be given one of the others (Claude Code: ANTHROPIC_CUSTOM_HEADERS).
DEFAULT_USER_HEADERS = (
    "x-openwebui-user-email",
    "x-openwebui-user-id",
    "x-openwebui-user-name",
    "x-llm-router-user",
    "x-user",
)

_OWUI_PREFIX = b"x-openwebui-"
_OWUI_NAME = b"x-openwebui-user-name"
_USER_AGENT = b"user-agent"

# Messages that precede a conversation's own content. OpenAI bodies carry the
# system prompt as messages[0]; newer ones may call it "developer".
_SYSTEM_ROLES = ("system", "developer")

# Header values are the client's to choose, so bound what is kept of them.
MAX_VALUE_CHARS = 128
# Finished requests remembered per session (for its detail view) and overall.
RECENT_PER_SESSION = 10
RECENT_OVERALL = 200
# What one snapshot carries. Busy sessions are always included on top.
SNAPSHOT_SESSIONS = 200
SNAPSHOT_RECENT = 50
MAX_MODELS_PER_SESSION = 4
MAX_AGENTS_PER_SESSION = 64

# User-agent fragments worth naming. Anything else is shown by the first word of
# its user agent.
KNOWN_CLIENTS = (
    ("claude-cli", "Claude Code"),
    ("opencode", "opencode"),
    ("qwencode", "Qwen Code"),
    ("qwen-code", "Qwen Code"),
    ("aider", "aider"),
    ("cline", "Cline"),
    ("zed", "Zed"),
)


def _clean(value: str) -> str:
    """A header value fit to show: percent-decoded, trimmed and bounded.

    Open WebUI percent-encodes user names, so that non-ASCII survives a header.
    """
    if "%" in value:
        value = unquote(value)
    return value.strip()[:MAX_VALUE_CHARS]


def client_label(user_agent: str, open_webui: bool = False) -> str:
    if open_webui:
        return "Open WebUI"
    lowered = user_agent.lower()
    for fragment, label in KNOWN_CLIENTS:
        if fragment in lowered:
            return label
    words = user_agent.split("/", 1)[0].split(None, 1)
    return words[0][:32] if words else "unknown"


def inferred_session_id(body: dict[str, Any], keys: list[str]) -> str:
    """The prefix hash through a conversation's first message of its own.

    Every later turn extends that prefix, so the id is stable for the whole
    conversation. It has to reach past the system prompt, which every
    conversation from a client shares: that is also why affinity never pins on it.
    """
    messages = body.get("messages")
    index = 0
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            if not (isinstance(message, dict) and message.get("role") in _SYSTEM_ROLES):
                break
    return keys[min(index, len(keys) - 1)]


def _r(seconds: float) -> float:
    return round(seconds, 2)


@dataclass(slots=True, eq=False)
class Session:
    key: tuple[str, str]
    # A short handle for the dashboard to refer to this session by.
    n: int
    id: str
    # How the session was identified: claude-code, open-webui, header, inferred, none.
    via: str
    user: str
    user_name: str
    # How the user was identified: open-webui, header, address.
    user_via: str
    address: str
    first_seen: float
    last_active: float
    user_agent: str = ""
    client: str = ""
    requests: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    # Prompt tokens of the requests that reported cache counts at all, so the hit
    # rate is not diluted by ones that said nothing.
    cache_basis: int = 0
    backend: str | None = None
    models: list[str] = field(default_factory=list)
    agents: set[str] = field(default_factory=set)
    active: dict[int, Tracked] = field(default_factory=dict)
    recent: deque[Tracked] = field(default_factory=lambda: deque(maxlen=RECENT_PER_SESSION))


@dataclass(slots=True, eq=False)
class Tracked:
    """One request, from arrival to its last byte."""

    id: int
    session: Session
    model: str
    stream: bool
    agent: str | None
    started_at: float
    state: str = QUEUED
    state_at: float = 0.0
    attempts: int = 0
    pinned: str | None = None
    hold_until: float = 0.0
    backend: str | None = None
    queue_s: float = 0.0
    first_byte_at: float | None = None
    last_byte_at: float | None = None
    bytes: int = 0
    ended_at: float | None = None
    outcome: str | None = None
    status: int | None = None
    # Why the router itself turned the request away, when it did.
    note: str | None = None
    usage: TokenUsage | None = None
    # A streamed response finishes its own record when the stream ends, which is
    # long after the handler has returned.
    stream_owned: bool = False

    def queued(self, pinned: str | None, hold_s: float) -> None:
        now = time.monotonic()
        self.state = QUEUED
        self.state_at = now
        self.pinned = pinned
        self.hold_until = now + hold_s if pinned is not None else 0.0

    def dispatched(self, backend: str, queue_s: float) -> None:
        self.state = PROCESSING
        self.state_at = time.monotonic()
        self.backend = backend
        self.attempts += 1
        self.queue_s += queue_s
        self.session.backend = backend

    def received(self, size: int) -> None:
        now = time.monotonic()
        if self.first_byte_at is None:
            self.first_byte_at = now
            self.state = STREAMING
            self.state_at = now
        self.last_byte_at = now
        self.bytes += size


# Stands in for a session while tracking is off, so callers never branch on it.
_DETACHED = Session(
    key=("", ""), n=0, id="", via="none", user="", user_name="", user_via="",
    address="", first_seen=0.0, last_active=0.0,
)


class SessionTracker:
    def __init__(self, config: TrackingConfig) -> None:
        # Least recently active first, so eviction reads from the front.
        self._sessions: OrderedDict[tuple[str, str], Session] = OrderedDict()
        self._by_n: dict[int, Session] = {}
        self._active: dict[int, Tracked] = {}
        self._recent: deque[Tracked] = deque(maxlen=RECENT_OVERALL)
        self._request_ids = itertools.count(1)
        self._session_numbers = itertools.count(1)
        self.enabled = True
        self.reconfigure(config)

    def reconfigure(self, config: TrackingConfig) -> None:
        self.enabled = config.enabled
        self.slow_after_s = config.slow_after_s
        self.stuck_after_s = config.stuck_after_s
        self._retain_s = config.retain_s
        self._max_sessions = config.max_sessions
        # ASGI hands header names over lowercased, as bytes.
        self._user_headers = tuple(h.encode("latin-1", "replace") for h in config.user_headers)
        self._wanted = frozenset((*self._user_headers, _OWUI_NAME, _USER_AGENT))
        if self.enabled:
            self._evict(time.monotonic())
        else:
            # Nothing is recorded while off; don't keep showing what was.
            self._sessions.clear()
            self._by_n.clear()
            self._active.clear()
            self._recent.clear()

    # ------------------------------------------------------------ request path

    def begin(
        self,
        headers: Iterable[tuple[bytes, bytes]],
        address: str | None,
        body: dict[str, Any],
        model: str,
        explicit: ExplicitSession | None,
        keys: list[str],
    ) -> Tracked:
        now = time.monotonic()
        stream = bool(body.get("stream"))
        if not self.enabled:
            return Tracked(0, _DETACHED, model, stream, None, now)

        # One pass over the raw headers for everything wanted, rather than a
        # lookup -- itself a scan of every header -- per name.
        wanted = self._wanted
        found: dict[bytes, bytes] = {}
        open_webui = False
        for name, value in headers:
            if name in wanted:
                found[name] = value
            if name.startswith(_OWUI_PREFIX):
                open_webui = True

        user = ""
        for name in self._user_headers:
            value = found.get(name)
            if value:
                user = _clean(value.decode("latin-1"))
                if user:
                    break
        if user:
            user_key = "u:" + user
            user_via = "open-webui" if open_webui else "header"
            owui_name = found.get(_OWUI_NAME)
            user_name = (_clean(owui_name.decode("latin-1")) if owui_name else "") or user
        else:
            user_name = address or "unknown"
            user_key = "ip:" + user_name
            user_via = "address"

        agent = None
        if explicit is not None:
            session_id = explicit.id[:MAX_VALUE_CHARS]
            if explicit.source == CLAUDE_CODE:
                via = CLAUDE_CODE
                agent = explicit.agent[:MAX_VALUE_CHARS] if explicit.agent else None
            else:
                via = "open-webui" if open_webui else "header"
        elif keys:
            session_id, via = inferred_session_id(body, keys), "inferred"
        else:
            session_id, via = "", "none"

        key = (user_key, session_id)
        session = self._sessions.get(key)
        if session is None:
            session = Session(
                key=key,
                n=next(self._session_numbers),
                id=session_id,
                via=via,
                user=user_key,
                user_name=user_name,
                user_via=user_via,
                address=address or "",
                first_seen=now,
                last_active=now,
            )
            self._sessions[key] = session
            self._by_n[session.n] = session
            self._evict(now)
        else:
            session.last_active = now
            session.user_name = user_name
            self._sessions.move_to_end(key)

        raw_agent = found.get(_USER_AGENT)
        user_agent = raw_agent.decode("latin-1")[:MAX_VALUE_CHARS] if raw_agent else ""
        if user_agent != session.user_agent or not session.client:
            session.user_agent = user_agent
            session.client = (
                "Claude Code" if via == CLAUDE_CODE else client_label(user_agent, open_webui)
            )

        model = model[:MAX_VALUE_CHARS]
        models = session.models
        if model not in models:
            if len(models) >= MAX_MODELS_PER_SESSION:
                models.pop(0)
            models.append(model)
        if agent is not None and len(session.agents) < MAX_AGENTS_PER_SESSION:
            session.agents.add(agent)
        session.requests += 1

        tracked = Tracked(next(self._request_ids), session, model, stream, agent, now, state_at=now)
        session.active[tracked.id] = tracked
        self._active[tracked.id] = tracked
        return tracked

    def finish(self, tracked: Tracked, status: int | None, outcome: str | None = None) -> None:
        """Record how a request ended. Safe to call more than once."""
        if self._active.pop(tracked.id, None) is None:
            # Already finished, or tracking was switched off while it ran.
            return
        now = time.monotonic()
        if outcome is None:
            outcome = CANCELLED if status is None else OK if status < 400 else ERROR
        tracked.ended_at = now
        tracked.status = status
        tracked.outcome = outcome

        session = tracked.session
        session.active.pop(tracked.id, None)
        session.last_active = now
        if outcome == ERROR:
            session.errors += 1
        usage = tracked.usage
        if usage is not None and usage.prompt_tokens >= 0:
            session.prompt_tokens += usage.prompt_tokens
            session.completion_tokens += max(0, usage.completion_tokens)
            if usage.cached_tokens is not None and usage.cached_tokens >= 0:
                session.cached_tokens += min(usage.cached_tokens, usage.prompt_tokens)
                session.cache_basis += usage.prompt_tokens
        session.recent.append(tracked)
        self._recent.append(tracked)
        if self._sessions.get(session.key) is session:
            self._sessions.move_to_end(session.key)

    def _evict(self, now: float) -> None:
        """Forget sessions idle past `retain_s`, and the oldest beyond `max_sessions`.

        A session with a request in flight is never forgotten, however old.
        """
        sessions = self._sessions
        cutoff = now - self._retain_s
        for _ in range(len(sessions)):
            key, oldest = next(iter(sessions.items()))
            if len(sessions) <= self._max_sessions and oldest.last_active >= cutoff:
                return
            if oldest.active:
                sessions.move_to_end(key)
                continue
            del sessions[key]
            del self._by_n[oldest.n]

    # ---------------------------------------------------------------- snapshot

    def snapshot(self) -> dict[str, Any]:
        """Everything the dashboard shows. O(tracked sessions); never on the request path."""
        now = time.monotonic()
        thresholds = {
            "slow_s": self.slow_after_s,
            "stuck_s": self.stuck_after_s,
            "retain_s": self._retain_s,
        }
        if not self.enabled:
            return {
                "enabled": False, "thresholds": thresholds, "summary": {},
                "in_flight": [], "users": [], "sessions": [], "recent": [],
            }
        self._evict(now)

        in_flight = [self._request_view(t, now) for t in self._active.values()]
        in_flight.sort(key=lambda v: v["waiting_s"], reverse=True)
        by_request = {v["id"]: v for v in in_flight}

        by_state = {QUEUED: 0, HOLDING: 0, PROCESSING: 0, STREAMING: 0}
        slow = stuck = 0
        for view in in_flight:
            by_state[view["state"]] += 1
            if view["silence_s"] >= self.stuck_after_s:
                stuck += 1
            elif view["silence_s"] >= self.slow_after_s:
                slow += 1

        busy: dict[int, Session] = {}
        for tracked in self._active.values():
            busy[tracked.session.n] = tracked.session
        shown = list(busy.values())
        limit = SNAPSHOT_SESSIONS + len(shown)
        for session in reversed(self._sessions.values()):
            if len(shown) >= limit:
                break
            if session.n not in busy:
                shown.append(session)

        return {
            "enabled": True,
            "thresholds": thresholds,
            "summary": {
                "in_flight": len(in_flight),
                "by_state": by_state,
                "slow": slow,
                "stuck": stuck,
                "longest_wait_s": in_flight[0]["waiting_s"] if in_flight else None,
                "sessions": len(self._sessions),
                "busy_sessions": len(busy),
            },
            "in_flight": in_flight,
            "users": self._users(now, in_flight),
            "sessions": [self._session_view(s, now, by_request) for s in shown],
            "recent": [
                self._request_view(t, now)
                for t in itertools.islice(reversed(self._recent), SNAPSHOT_RECENT)
            ],
        }

    def session_detail(self, n: int) -> dict[str, Any] | None:
        """One session with its recent requests, for when it is opened up."""
        session = self._by_n.get(n)
        if session is None:
            return None
        now = time.monotonic()
        active = [self._request_view(t, now) for t in session.active.values()]
        view = self._session_view(session, now, {v["id"]: v for v in active})
        view["active"] = active
        view["recent"] = [self._request_view(t, now) for t in reversed(session.recent)]
        view["agent_ids"] = sorted(session.agents)
        view["user_agent"] = session.user_agent
        return view

    def _users(self, now: float, in_flight: list[dict[str, Any]]) -> list[dict[str, Any]]:
        users: dict[str, dict[str, Any]] = {}
        for s in self._sessions.values():
            user = users.get(s.user)
            if user is None:
                user = users[s.user] = {
                    "user": s.user,
                    "name": s.user_name,
                    "via": s.user_via,
                    "clients": [],
                    "addresses": [],
                    "sessions": 0,
                    "busy_sessions": 0,
                    "in_flight": 0,
                    "slow": 0,
                    "stuck": 0,
                    "longest_wait_s": None,
                    "requests": 0,
                    "errors": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "idle_s": now - s.last_active,
                }
            user["sessions"] += 1
            if s.active:
                user["busy_sessions"] += 1
            user["requests"] += s.requests
            user["errors"] += s.errors
            user["prompt_tokens"] += s.prompt_tokens
            user["completion_tokens"] += s.completion_tokens
            user["idle_s"] = min(user["idle_s"], now - s.last_active)
            if s.client and s.client not in user["clients"] and len(user["clients"]) < 4:
                user["clients"].append(s.client)
            if s.address and s.address not in user["addresses"] and len(user["addresses"]) < 4:
                user["addresses"].append(s.address)

        for view in in_flight:
            user = users.get(view["user"])
            if user is None:
                continue
            user["in_flight"] += 1
            if view["silence_s"] >= self.stuck_after_s:
                user["stuck"] += 1
            elif view["silence_s"] >= self.slow_after_s:
                user["slow"] += 1
            if user["longest_wait_s"] is None or view["waiting_s"] > user["longest_wait_s"]:
                user["longest_wait_s"] = view["waiting_s"]

        # Whoever has something stuck first, then whoever is busiest.
        ordered = sorted(
            users.values(),
            key=lambda u: (-u["stuck"], -u["slow"], -u["in_flight"], u["idle_s"]),
        )
        for user in ordered:
            user["idle_s"] = _r(user["idle_s"])
        return ordered

    def _session_view(
        self, s: Session, now: float, by_request: dict[int, dict[str, Any]]
    ) -> dict[str, Any]:
        last = s.recent[-1] if s.recent else None
        view: dict[str, Any] = {
            "n": s.n,
            "id": s.id,
            "via": s.via,
            "user": s.user,
            "user_name": s.user_name,
            "client": s.client,
            "address": s.address,
            "models": list(s.models),
            "backend": s.backend,
            "agents": len(s.agents),
            "requests": s.requests,
            "errors": s.errors,
            "prompt_tokens": s.prompt_tokens,
            "completion_tokens": s.completion_tokens,
            "cache_rate": (s.cached_tokens / s.cache_basis) if s.cache_basis else None,
            "first_seen_s": _r(now - s.first_seen),
            "idle_s": _r(now - s.last_active),
            "in_flight": len(s.active),
            "last_outcome": last.outcome if last else None,
            "last_status": last.status if last else None,
            "state": "idle",
            "waiting_s": None,
            "silence_s": None,
        }
        # A session is as far along as its most worrying request: the one that
        # has heard nothing for longest.
        worst = None
        for request_id in s.active:
            current = by_request.get(request_id)
            if current is None:
                continue
            if view["waiting_s"] is None or current["waiting_s"] > view["waiting_s"]:
                view["waiting_s"] = current["waiting_s"]
            if worst is None or current["silence_s"] > worst["silence_s"]:
                worst = current
        if worst is not None:
            view["state"] = worst["state"]
            view["silence_s"] = worst["silence_s"]
            view["stream"] = worst["stream"]
            view["pinned"] = worst["pinned"]
        return view

    def _request_view(self, t: Tracked, now: float) -> dict[str, Any]:
        s = t.session
        view: dict[str, Any] = {
            "id": t.id,
            "session": s.n,
            "session_id": s.id,
            "session_via": s.via,
            "user": s.user,
            "user_name": s.user_name,
            "client": s.client,
            "agent": t.agent,
            "model": t.model,
            "stream": t.stream,
            "backend": t.backend,
            "attempts": t.attempts,
            "queue_s": _r(t.queue_s),
            "ttft_s": _r(t.first_byte_at - t.started_at) if t.first_byte_at is not None else None,
            "bytes": t.bytes,
        }
        if t.ended_at is None:
            state = t.state
            if state == QUEUED and t.pinned is not None and now < t.hold_until:
                state = HOLDING
            heard = t.last_byte_at if t.last_byte_at is not None else t.started_at
            view["state"] = state
            view["pinned"] = t.pinned
            view["waiting_s"] = _r(now - t.started_at)
            view["state_s"] = _r(now - t.state_at)
            view["silence_s"] = _r(now - heard)
        else:
            usage = t.usage
            view["state"] = t.outcome
            view["status"] = t.status
            view["note"] = t.note
            view["duration_s"] = _r(t.ended_at - t.started_at)
            view["ended_s"] = _r(now - t.ended_at)
            view["prompt_tokens"] = usage.prompt_tokens if usage else None
            view["completion_tokens"] = usage.completion_tokens if usage else None
            view["cached_tokens"] = usage.cached_tokens if usage else None
        return view
