"""Client configs generated from the router's own config.

The generated formats were also checked by running the real clients against a
router: opencode 1.18 (both the file and `opencode auth login`), Claude Code
2.1 and Qwen Code, confirming what each one actually sends. These tests pin
that output down.
"""

from __future__ import annotations

import contextlib
import json

import httpx
import pytest
from conftest import running_app
from fake_upstream import FakeUpstream
from starlette.requests import Request

from llm_router import clients
from llm_router.config import ConfigError, parse_config
from llm_router.proxy import Router, create_app

GLM = "GLM-5.3-Flash-EXL3"

# The deployment the hand-written opencode config below was written for.
GLM_CONFIG = f"""
listen: {{port: 8888}}
backends:
  - {{name: sparks, url: 'http://10.0.0.2:8000', kind: vllm, capacity: 4, models: [{GLM}]}}
models:
  {GLM}:
    name: GLM-5.3 Flash EXL3
    tool_call: true
    reasoning: true
    context_length: 200000
    max_output_tokens: 32768
    request_params:
      reasoning_effort: high
      chat_template_kwargs: {{clear_thinking: true}}
clients:
  provider_id: glm53
  provider_name: GLM-5.3 Flash EXL3 (Sparks)
  base_url: http://10.0.0.1:8888
"""


def router_for(text: str, discovered: dict[str, int] | None = None) -> Router:
    router = Router(parse_config(text))
    for backend, context in (discovered or {}).items():
        router.clients.context_length[backend] = context
    return router


def request(query: bytes = b"", host: bytes = b"router.lan:8888") -> Request:
    return Request({
        "type": "http", "method": "GET", "path": "/clients/opencode", "query_string": query,
        "headers": [(b"host", host)], "scheme": "http", "server": ("router.lan", 8888),
        "root_path": "",
    })


def setup(text: str = GLM_CONFIG, query: bytes = b"", discovered=None) -> clients.Setup:
    router = router_for(text, discovered if discovered is not None else {"sparks": 850000})
    return clients.build_setup(router, request(query))


# ------------------------------------------------------------------ opencode


def test_opencode_config_reproduces_the_hand_written_one():
    """The config this deployment used to maintain by hand, now generated.

    Two deliberate differences, each found by running opencode 1.18 against
    the router rather than read from its docs:
    - reasoning_effort is a variant, made the default for the build and plan
      agents. opencode silently drops it from a model's `options`.
    - chunkTimeout, because the router waits up to timeouts.first_byte_s (600s)
      between chunks, and opencode gives up after 300s by default. With vLLM a
      long prefill happens before the first chunk.
    """
    assert clients.opencode(setup()) == {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "glm53": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "GLM-5.3 Flash EXL3 (Sparks)",
                "options": {
                    "baseURL": "http://10.0.0.1:8888/v1",
                    "apiKey": "unused",
                    # queue_timeout_s + first_byte_s: 300s + 600s.
                    "headerTimeout": 900000,
                    "timeout": False,
                    "chunkTimeout": 600000,
                },
                "models": {
                    GLM: {
                        "name": "GLM-5.3 Flash EXL3",
                        "tool_call": True,
                        "reasoning": True,
                        "limit": {"context": 200000, "output": 32768},
                        "options": {"chat_template_kwargs": {"clear_thinking": True}},
                        "variants": {"high": {"reasoningEffort": "high"}},
                    }
                },
            }
        },
        "model": f"glm53/{GLM}",
        "agent": {
            "build": {"model": f"glm53/{GLM}", "variant": "high"},
            "plan": {"model": f"glm53/{GLM}", "variant": "high"},
        },
    }


def test_reasoning_efforts_become_selectable_variants():
    text = GLM_CONFIG.replace("    request_params:", "    reasoning_efforts: [low, high, max]\n    request_params:")
    model = clients.opencode(setup(text))["provider"]["glm53"]["models"][GLM]
    assert list(model["variants"]) == ["low", "high", "max"]
    assert model["variants"]["max"] == {"reasoningEffort": "max"}


def test_wellknown_carries_the_same_config():
    s = setup()
    wellknown = clients.opencode_wellknown(s)
    assert wellknown["config"] == clients.opencode(s)
    assert wellknown["auth"] == {"command": ["echo", "unused"], "env": "GLM53_API_KEY"}
    assert clients.login_url(s) == "http://10.0.0.1:8888"
    assert clients.login_url(setup(query=b"user=Ren%C3%A9e%20D")) == "http://10.0.0.1:8888/u/Ren%C3%A9e%20D"


# ------------------------------------------------------------ other clients


def test_claude_code_maps_every_tier_onto_served_models():
    text = GLM_CONFIG.replace(
        f"models: [{GLM}]}}", f"models: [{GLM}, glm-air, qwen-coder]}}"
    ).replace("  base_url:", "  small_model: glm-air\n  base_url:")
    env = clients.claude_code(setup(text))["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://10.0.0.1:8888"  # no /v1: it adds its own
    assert env["ANTHROPIC_DEFAULT_MODEL"] == GLM
    for tier in ("OPUS", "SONNET", "FABLE"):
        assert env[f"ANTHROPIC_DEFAULT_{tier}_MODEL"] == GLM
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL_NAME"] == "GLM-5.3 Flash EXL3"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "glm-air"
    assert env["ANTHROPIC_CUSTOM_MODEL_OPTION"] == "qwen-coder"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "200000"
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32768"
    assert env["API_TIMEOUT_MS"] == "900000"
    assert env["API_FORCE_IDLE_TIMEOUT"] == "0"
    assert env["CLAUDE_STREAM_IDLE_TIMEOUT_MS"] == "600000"
    assert all(isinstance(v, str) for v in env.values())


def test_claude_code_shell_exports_are_quoted():
    text = GLM_CONFIG.replace("name: GLM-5.3 Flash EXL3\n", "name: \"GLM's $best `model`\"\n")
    shell = clients.claude_code_shell(setup(text))
    assert "export ANTHROPIC_DEFAULT_SONNET_MODEL_NAME='GLM'\"'\"'s $best `model`'\n" in shell


def test_qwen_code_sends_request_params_as_extra_body():
    config = clients.qwen_code(setup())
    [provider] = config["modelProviders"]["openai"]
    assert provider["baseUrl"] == "http://10.0.0.1:8888/v1"
    assert provider["envKey"] == "GLM53_API_KEY"
    assert config["env"] == {"GLM53_API_KEY": "unused"}
    assert provider["generationConfig"] == {
        "timeout": 900000,
        "contextWindowSize": 200000,
        "samplingParams": {"max_tokens": 32768},
        "extra_body": {"reasoning_effort": "high", "chat_template_kwargs": {"clear_thinking": True}},
    }
    assert config["model"] == {"name": GLM}
    assert config["security"] == {"auth": {"selectedType": "openai"}}


def test_zed_settings():
    config = clients.zed(setup())
    provider = config["language_models"]["openai_compatible"]["glm53"]
    assert provider["api_url"] == "http://10.0.0.1:8888/v1"
    [model] = provider["available_models"]
    assert model == {
        "name": GLM,
        "display_name": "GLM-5.3 Flash EXL3",
        "max_tokens": 200000,
        "max_output_tokens": 32768,
        "reasoning_effort": "high",
        "capabilities": {"tools": True, "images": False, "parallel_tool_calls": False,
                         "prompt_cache_key": False},
    }
    assert config["agent"] == {"default_model": {"provider": "glm53", "model": GLM}}


def test_zed_leaves_out_an_effort_it_has_no_name_for():
    text = GLM_CONFIG.replace("reasoning_effort: high", "reasoning_effort: turbo")
    s = setup(text)
    [model] = clients.zed(s)["language_models"]["openai_compatible"]["glm53"]["available_models"]
    assert "reasoning_effort" not in model
    assert any("turbo" in note for note in clients.CLIENTS["zed"].notes(s))


def test_user_name_reaches_every_client_that_can_send_it():
    s = setup(query=b"user=J%C3%B6rg%20M")
    encoded = "J%C3%B6rg M"
    assert clients.opencode(s)["provider"]["glm53"]["options"]["headers"] == {"X-LLM-Router-User": encoded}
    assert clients.claude_code(s)["env"]["ANTHROPIC_CUSTOM_HEADERS"] == f"X-LLM-Router-User: {encoded}"
    [provider] = clients.qwen_code(s)["modelProviders"]["openai"]
    assert provider["generationConfig"]["customHeaders"] == {"X-LLM-Router-User": encoded}
    zed = clients.zed(s)["language_models"]["openai_compatible"]["glm53"]
    assert zed["custom_headers"] == {"X-LLM-Router-User": encoded}


@pytest.mark.parametrize("query", [b"user=a%0Ab", b"model=nope", b"user=" + b"x" * 65])
def test_bad_query_parameters_are_refused(query):
    with pytest.raises(clients.BadRequest):
        setup(query=query)


# ----------------------------------------------------- where values come from


def test_base_url_is_where_the_config_was_fetched_from_unless_configured():
    text = GLM_CONFIG.replace("  base_url: http://10.0.0.1:8888\n", "")
    s = setup(text)
    assert (s.base_url, s.base_url_from) == ("http://router.lan:8888", "request")
    assert setup().base_url_from == "config"


def test_timeouts_follow_the_router_config():
    text = GLM_CONFIG + "routing: {queue_timeout_s: 100}\ntimeouts: {first_byte_s: 200}\n"
    s = setup(text)
    options = clients.opencode(s)["provider"]["glm53"]["options"]
    # 300s is opencode's own default, so it is kept rather than lowered.
    assert options["headerTimeout"] == 300000
    assert "chunkTimeout" not in options
    env = clients.claude_code(s)["env"]
    assert "API_TIMEOUT_MS" not in env and "API_FORCE_IDLE_TIMEOUT" not in env
    assert env["CLAUDE_STREAM_IDLE_TIMEOUT_MS"] == "200000"


@pytest.mark.parametrize(
    ("configured", "discovered", "expected"),
    [
        (None, 32768, (32768, "discovered")),
        (None, None, (None, "unknown")),
        (200000, None, (200000, "configured")),
        (200000, 850000, (200000, "configured")),
        (900000, 850000, (850000, "capped")),
    ],
)
def test_model_context(configured, discovered, expected):
    assert clients.model_context(configured, discovered) == expected


def test_a_window_larger_than_the_backends_serve_is_capped():
    s = setup(discovered={"sparks": 131072})
    assert s.default.context == 131072
    assert any("131072" in w for w in s.warnings)


def test_an_unknown_window_is_left_out_rather_than_guessed():
    text = GLM_CONFIG.replace("    context_length: 200000\n", "")
    s = setup(text, discovered={})
    assert "limit" not in clients.opencode(s)["provider"]["glm53"]["models"][GLM]
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in clients.claude_code(s)["env"]
    assert s.warnings


def test_unconfigured_models_still_get_sensible_entries():
    router = router_for(
        "backends:\n  - {name: a, url: 'http://x', capacity: 1, models: [m1, m2]}\n",
        {"a": 32768},
    )
    s = clients.build_setup(router, request())
    assert s.provider_id == "llm-router" and s.env_key == "LLM_ROUTER_API_KEY"
    assert s.default.id == "m1"
    models = clients.opencode(s)["provider"]["llm-router"]["models"]
    # opencode insists on an output limit; its own cap stands in for one.
    assert models["m2"] == {"name": "m2", "tool_call": True, "reasoning": False,
                            "limit": {"context": 32768, "output": 32000}}


# ------------------------------------------------------------------- config


def test_config_is_validated():
    base = "backends:\n  - {name: a, url: 'http://x', capacity: 1, models: [m]}\n"
    config = parse_config(base + "clients: {base_url: 'https://gw.example.com/llm/v1/', default_model: m}\n")
    assert config.clients.base_url == "https://gw.example.com/llm"
    for bad in (
        "models: {typo: {name: x}}",
        "models: {m: {context_length: 0}}",
        "models: {m: {reasoning: maybe}}",
        "models: {m: {request_params: [a]}}",
        "models: {m: {reasoning_efforts: high}}",
        "models: {m: {unknown: 1}}",
        "clients: {default_model: typo}",
        "clients: {small_model: typo}",
        "clients: {provider_id: 'has spaces'}",
        "clients: {base_url: 'router:8080'}",
    ):
        with pytest.raises(ConfigError):
            parse_config(base + bad + "\n")


# ---------------------------------------------------------------- endpoints


@contextlib.asynccontextmanager
async def serving(text: str):
    upstream = FakeUpstream(name="a", max_concurrency=2, model="m", context_length=65536)
    await upstream.start()
    router = Router(parse_config(text.replace("URL", upstream.url)))
    try:
        async with running_app(create_app(router.config, router)) as base_url:
            async with httpx.AsyncClient(base_url=base_url, timeout=10) as client:
                yield client, router
    finally:
        await upstream.stop()


ENDPOINT_CONFIG = """
backends:
  - {name: a, url: 'URL', capacity: 2, models: [m]}
models:
  m: {name: Model M, context_length: 16384, request_params: {reasoning_effort: low}}
"""


async def test_endpoints():
    async with serving(ENDPOINT_CONFIG) as (client, _router):
        index = (await client.get("/clients", params={"user": "dana"})).json()
        assert [c["id"] for c in index["clients"]] == ["opencode", "claude-code", "qwen-code", "zed"]
        assert index["models"][0]["context"] == 16384
        assert index["base_url"] == str(client.base_url).rstrip("/")
        opencode = next(c for c in index["clients"] if c["id"] == "opencode")
        assert "user=dana" in opencode["url"]
        assert json.loads(opencode["content"])["model"] == "llm-router/m"

        raw = await client.get("/clients/zed")
        assert raw.headers["content-type"].startswith("application/json")
        shell = await client.get("/clients/claude-code", params={"format": "shell"})
        assert shell.text.startswith("export ANTHROPIC_BASE_URL=")
        assert (await client.get("/clients/nope")).status_code == 404
        assert (await client.get("/clients/opencode", params={"model": "nope"})).status_code == 400

        wellknown = (await client.get("/.well-known/opencode")).json()
        options = wellknown["config"]["provider"]["llm-router"]["options"]
        assert options["baseURL"].endswith("/v1") and "headers" not in options
        # The per-user login URL carries the name into the config it serves.
        named = (await client.get("/u/Ren%C3%A9e%20D/.well-known/opencode")).json()
        options = named["config"]["provider"]["llm-router"]["options"]
        assert options["headers"] == {"X-LLM-Router-User": "Ren%C3%A9e D"}

        page = await client.get("/connect")
        assert page.status_code == 200 and "connect.js" in page.text
        assert (await client.get("/dashboard/connect.js")).status_code == 200
        assert (await client.get("/dashboard/connect.html")).status_code == 404


async def test_v1_models_advertises_what_clients_are_told():
    async with serving(ENDPOINT_CONFIG) as (client, router):
        [model] = (await client.get("/v1/models")).json()["data"]
    assert model["display_name"] == "Model M"
    # The backend serves 65536; the config tells clients 16384.
    assert router.context_for("m") == 65536
    assert model["context_length"] == model["max_model_len"] == 16384


async def test_a_reload_changes_what_clients_are_given():
    async with serving(ENDPOINT_CONFIG) as (client, router):
        before = (await client.get("/clients/opencode")).json()
        router.apply_config(parse_config(
            ENDPOINT_CONFIG.replace("URL", router.config.backends[0].url)
            .replace("context_length: 16384", "context_length: 8192")
        ))
        after = (await client.get("/clients/opencode")).json()
    def limit(config):
        return config["provider"]["llm-router"]["models"]["m"]["limit"]["context"]

    assert (limit(before), limit(after)) == (16384, 8192)
