"""Session inference: deriving conversation identity from the messages themselves."""

from __future__ import annotations

import time

from llm_router.affinity import (
    MIN_AFFINITY_DEPTH,
    SessionMap,
    explicit_session_id,
    session_keys,
)

SYSTEM = {"role": "system", "content": "You are a coding agent with tools."}


def conversation(turns: int, seed: str = "a") -> dict:
    """A conversation shaped like an agentic loop: shared system prompt, then turns."""
    messages = [SYSTEM, {"role": "user", "content": f"task {seed}"}]
    for i in range(turns):
        messages.append({"role": "assistant", "content": f"step {i}"})
        messages.append({"role": "user", "content": f"tool result {i}"})
    return {"model": "m", "messages": messages}


def test_keys_are_a_stable_growing_prefix():
    """Turn N's keys must be a prefix of turn N+1's -- that is what makes pins stick."""
    early = session_keys(conversation(3))
    later = session_keys(conversation(5))
    assert len(later) > len(early)
    assert later[: len(early)] == early


def test_growing_conversation_keeps_its_pin():
    sessions = SessionMap(depth=3)
    sessions.assign(session_keys(conversation(3)), "b0")
    for turns in (4, 5, 6):
        assert sessions.lookup(session_keys(conversation(turns))) == "b0"


def test_distinct_conversations_are_not_confused():
    sessions = SessionMap(depth=3)
    sessions.assign(session_keys(conversation(3, seed="a")), "b0")
    assert sessions.lookup(session_keys(conversation(3, seed="b"))) is None


def test_shared_system_prompt_alone_does_not_pin():
    """Otherwise every new conversation would pile onto one backend."""
    sessions = SessionMap(depth=3)
    sessions.assign(session_keys(conversation(3, seed="a")), "b0")
    # A brand new conversation shares only the system prompt.
    fresh = {"model": "m", "messages": [SYSTEM, {"role": "user", "content": "unrelated"}]}
    assert sessions.lookup(session_keys(fresh)) is None


def test_min_depth_boundaries_are_never_recorded():
    sessions = SessionMap(depth=8)
    keys = session_keys(conversation(4))
    sessions.assign(keys, "b0")
    # The depth-1 boundary (system prompt only) must not be a pin.
    assert sessions.lookup([keys[0]]) is None
    assert MIN_AFFINITY_DEPTH == 2


def test_tool_definitions_participate_in_identity():
    """Different tools render a different prompt prefix, so a different cache."""
    base = conversation(2)
    with_tools = dict(base, tools=[{"type": "function", "function": {"name": "grep"}}])
    assert session_keys(base) != session_keys(with_tools)


def test_model_participates_in_identity():
    base = conversation(2)
    other = dict(base, model="different")
    assert session_keys(base) != session_keys(other)


def test_rewritten_history_breaks_the_pin():
    """Context compaction genuinely changes the prefix; a miss is the correct answer."""
    sessions = SessionMap(depth=3)
    sessions.assign(session_keys(conversation(4)), "b0")
    compacted = {
        "model": "m",
        "messages": [SYSTEM, {"role": "user", "content": "summary of earlier work"}],
    }
    assert sessions.lookup(session_keys(compacted)) is None


def test_expired_pins_are_dropped():
    sessions = SessionMap(ttl_s=0.05, depth=3)
    keys = session_keys(conversation(3))
    sessions.assign(keys, "b0")
    assert sessions.lookup(keys) == "b0"
    time.sleep(0.08)
    assert sessions.lookup(keys) is None


def test_lru_eviction_bounds_memory():
    sessions = SessionMap(max_entries=50, depth=1)
    for i in range(200):
        sessions.assign(session_keys(conversation(2, seed=str(i))), "b0")
    assert len(sessions) <= 50


def test_repin_moves_the_session():
    sessions = SessionMap(depth=3)
    keys = session_keys(conversation(3))
    sessions.assign(keys, "b0")
    sessions.assign(keys, "b1")
    assert sessions.lookup(keys) == "b1"


def test_explicit_session_id_from_header():
    class Headers(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    assert explicit_session_id({}, Headers({"x-session-id": "abc"})) == "explicit:abc"
    assert explicit_session_id({"session_id": "xyz"}, Headers()) == "explicit:xyz"
    assert explicit_session_id({}, Headers()) is None


def test_empty_and_malformed_bodies_are_safe():
    assert session_keys({}) == []
    assert session_keys({"messages": []}) == []
    assert session_keys({"messages": "not a list"}) == []
    # Unserializable content must not raise.
    assert len(session_keys({"messages": [{"role": "user", "content": object()}]})) == 1


def test_huge_messages_are_hashed_cheaply_but_distinctly():
    big_a = {"model": "m", "messages": [SYSTEM, {"role": "user", "content": "x" * 200_000}]}
    big_b = {"model": "m", "messages": [SYSTEM, {"role": "user", "content": "y" * 200_000}]}
    assert session_keys(big_a) != session_keys(big_b)


def test_pin_counts_reported_for_dashboard():
    sessions = SessionMap(depth=1)
    sessions.assign(session_keys(conversation(2, seed="a")), "b0")
    sessions.assign(session_keys(conversation(2, seed="b")), "b1")
    counts = sessions.backend_pin_counts()
    assert counts.get("b0") == 1
    assert counts.get("b1") == 1
