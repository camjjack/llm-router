"""Config validation: catch mistakes at startup, not at 3am under load."""

from __future__ import annotations

import pytest

from llm_router.config import ConfigError, load_config

VALID = """
listen: {host: 127.0.0.1, port: 9000}
routing: {affinity_wait_ms: 500}
backends:
  - {name: a, url: "http://h1:8000/", capacity: 4, models: [m1, m2]}
  - {name: b, url: "http://h2:8080", capacity: 2, models: m1, kind: llamacpp}
"""


def write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_valid_config(tmp_path):
    config = load_config(write(tmp_path, VALID))
    assert config.port == 9000
    assert config.routing.affinity_wait_ms == 500
    # Defaults survive a partial routing section.
    assert config.routing.max_retries == 2
    assert config.backends[0].url == "http://h1:8000", "trailing slash should be stripped"
    assert config.backends[1].models == ("m1",), "a bare string becomes a one-item list"
    assert config.all_models == ("m1", "m2")
    assert {b.name for b in config.backends_for("m2")} == {"a"}
    assert config.backends[1].load_path == "/slots"
    assert config.backends[0].load_path is None


def test_env_var_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret-value")
    config = load_config(write(tmp_path, """
backends:
  - {name: a, url: "http://h1", capacity: 1, models: [m], api_key: "${MY_KEY}"}
"""))
    assert config.backends[0].api_key == "secret-value"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("backends: []", "at least one backend"),
        ("{}", "at least one backend"),
        ("backends:\n  - {name: a, url: h, models: [m]}", "missing: capacity"),
        ("backends:\n  - {name: a, url: h, capacity: 0, models: [m]}", "capacity must be"),
        ("backends:\n  - {name: a, url: h, capacity: 999, models: [m]}", "exceeds"),
        ("backends:\n  - {name: a, url: h, capacity: 1, models: []}", "non-empty list"),
        ("backends:\n  - {name: a, url: h, capacity: 1, models: [m], kind: sglang}", "kind must be"),
        # vLLM gets a higher ceiling, but not an unlimited one.
        ("backends:\n  - {name: a, url: h, capacity: 99999, models: [m], kind: vllm}", "exceeds"),
        (
            "backends:\n  - {name: a, url: h, capacity: 1, models: [m]}\n"
            "  - {name: a, url: h2, capacity: 1, models: [m]}",
            "duplicate backend name",
        ),
        (
            "routing: {affinity_wait_msec: 5}\nbackends:\n"
            "  - {name: a, url: h, capacity: 1, models: [m]}",
            "unknown key",
        ),
    ],
)
def test_invalid_configs_are_rejected(tmp_path, text, expected):
    with pytest.raises(ConfigError, match=expected):
        load_config(write(tmp_path, text))


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_typo_in_routing_key_names_the_valid_options(tmp_path):
    """A silently-ignored typo would be worse than a crash."""
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, """
routing: {affinity_wait: 100}
backends:
  - {name: a, url: h, capacity: 1, models: [m]}
"""))
    assert "affinity_wait_ms" in str(exc.value)


def test_backend_kinds_probe_the_right_endpoints(tmp_path):
    """Each engine publishes health, load and context in a different place."""
    config = load_config(write(tmp_path, """
backends:
  - {name: n, url: "http://h", capacity: 4, models: [m], kind: ninfer}
  - {name: l, url: "http://h", capacity: 4, models: [m], kind: llamacpp}
  - {name: v, url: "http://h", capacity: 256, models: [m], kind: vllm}
  - {name: s, url: "http://h", capacity: 4, models: [m], kind: lmstudio}
"""))
    by_name = {b.name: b for b in config.backends}

    # LM Studio has no /health; its model list is the liveness signal.
    assert by_name["s"].probe_path == "/api/v0/models"
    assert by_name["n"].probe_path == "/health"
    assert by_name["v"].probe_path == "/health"

    # Only llama.cpp and vLLM report their own load.
    assert by_name["l"].load_path == "/slots"
    assert by_name["v"].load_path == "/load"
    assert by_name["n"].load_path is None
    assert by_name["s"].load_path is None

    # vLLM wants saturating; the others want gating.
    assert by_name["v"].batches_continuously is True
    assert by_name["n"].batches_continuously is False


def test_vllm_accepts_max_num_seqs_sized_capacity(tmp_path):
    config = load_config(write(tmp_path, """
backends:
  - {name: v, url: "http://h", capacity: 256, models: [m], kind: vllm}
"""))
    assert config.backends[0].capacity == 256


def test_health_path_override(tmp_path):
    config = load_config(write(tmp_path, """
backends:
  - {name: a, url: "http://h", capacity: 1, models: [m], health_path: "/ping"}
"""))
    assert config.backends[0].probe_path == "/ping"


def test_shipped_example_config_is_valid():
    """The example must actually load -- it is the first thing anyone copies."""
    config = load_config("config.example.yaml")
    assert len(config.backends) == 5
    assert {b.kind for b in config.backends} == {
        "ninfer", "llamacpp", "vllm", "lmstudio"
    }
