"""Configuration for coding agents, generated from the router's own config.

A client needs more than the URL to work well: the model ids, the context each
takes, how much output to ask for, how long a request may legitimately wait
before its first byte, and any extra request fields a model wants. The router
already knows all of that, from its config or by discovering it, so it hands it
out here rather than have it copied into each client by hand, where it goes stale
the first time the config changes.

The formats follow each client's own documentation, checked September 2026:
opencode (opencode.json, and its /.well-known/opencode remote config), Claude
Code (settings.json `env`), Qwen Code (settings.json `modelProviders`) and Zed
(settings.json `language_models.openai_compatible`).
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

if TYPE_CHECKING:
    from .proxy import Router

# Names the user to the session dashboard. See tracking.DEFAULT_USER_HEADERS.
USER_HEADER = "X-LLM-Router-User"
MAX_USER_CHARS = 64

# Client defaults the generated configs only override when the router needs a
# client to wait longer than it would by itself.
OPENCODE_DEFAULT_TIMEOUT_MS = 300_000  # headerTimeout and chunkTimeout
# opencode never asks for more output than this itself; it stands in for a
# model's output limit when none is configured, since opencode insists on one.
OPENCODE_OUTPUT_CAP = 32_000
CLAUDE_API_TIMEOUT_MS = 600_000
# Claude Code aborts a response silent for five minutes on any provider but
# Anthropic's own, and its stream watchdog fires after two, clamped to 30.
CLAUDE_BODY_IDLE_MS = 300_000
CLAUDE_WATCHDOG_MS = 120_000
CLAUDE_WATCHDOG_MAX_MS = 1_800_000
# The reasoning_effort values Zed's settings accept; anything else is dropped.
ZED_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})


def model_context(configured: int | None, discovered: int | None) -> tuple[int | None, str]:
    """The context window clients should assume, and where it came from.

    A configured window may tell clients less than the backends serve, never
    more: a prompt sized to a window the backends don't have fails wherever it
    lands.
    """
    if configured is None:
        return discovered, "discovered" if discovered else "unknown"
    if discovered is None:
        return configured, "configured"
    if configured > discovered:
        return discovered, "capped"
    return configured, "configured"


@dataclass
class Model:
    id: str
    name: str
    tool_call: bool
    reasoning: bool
    images: bool
    context: int | None
    context_from: str
    max_output: int | None
    request_params: dict[str, Any]
    efforts: tuple[str, ...]

    @property
    def effort(self) -> str | None:
        effort = self.request_params.get("reasoning_effort")
        return effort if isinstance(effort, str) else None


@dataclass
class Setup:
    """Everything the generators need, worked out once per request."""

    base_url: str
    base_url_from: str
    provider_id: str
    provider_name: str
    api_key: str
    models: list[Model]
    default: Model
    small: Model | None
    # How long a request may take before the router sends response headers:
    # queued for a slot, then waiting on its backend.
    header_wait_ms: int
    # The longest silence the router allows from a backend mid-response.
    idle_ms: int
    user: str | None
    warnings: list[str]

    @property
    def env_key(self) -> str:
        """The API key's environment variable. Zed derives it this same way from
        the provider id, so the name is the same in every client."""
        return re.sub(r"[^A-Za-z0-9]", "_", self.provider_id).upper() + "_API_KEY"

    @property
    def openai_url(self) -> str:
        return self.base_url + "/v1"

    @property
    def user_header(self) -> str | None:
        # Percent-encoded so that any name survives a header; the session
        # tracker decodes it.
        return quote(self.user, safe=" @.-_+") if self.user else None


class BadRequest(ValueError):
    pass


def build_setup(router: Router, request: Request, user: str | None = None) -> Setup:
    """`user` overrides the ?user= query parameter, for URLs that carry it in the path."""
    config = router.config
    clients = config.clients
    served = list(config.all_models)
    warnings: list[str] = []

    models: list[Model] = []
    for model_id in served:
        meta = config.models.get(model_id)
        discovered = router.context_for(model_id)
        context, context_from = model_context(
            meta.context_length if meta else None, discovered
        )
        if context_from == "capped":
            warnings.append(
                f"{model_id}: models.context_length is {meta.context_length}, but its "
                f"backends serve {discovered}. Clients are told {discovered}."
            )
        elif context_from == "unknown":
            warnings.append(
                f"{model_id}: its context window couldn't be discovered, so clients "
                f"aren't told one. Set models['{model_id}'].context_length."
            )
        models.append(Model(
            id=model_id,
            name=(meta.name if meta and meta.name else model_id),
            tool_call=meta.tool_call if meta else True,
            reasoning=meta.reasoning if meta else False,
            images=meta.images if meta else False,
            context=context,
            context_from=context_from,
            max_output=meta.max_output_tokens if meta else None,
            request_params=dict(meta.request_params) if meta else {},
            efforts=meta.reasoning_efforts if meta else (),
        ))
    by_id = {m.id: m for m in models}

    default_id = request.query_params.get("model") or clients.default_model or served[0]
    if default_id not in by_id:
        raise BadRequest(f"no model '{default_id}' (known: {', '.join(served)})")
    small = by_id.get(clients.small_model) if clients.small_model else None

    user = (user if user is not None else request.query_params.get("user") or "").strip() or None
    if user is not None and (
        len(user) > MAX_USER_CHARS or any(ord(c) < 32 or ord(c) == 127 for c in user)
    ):
        raise BadRequest(f"user must be at most {MAX_USER_CHARS} printable characters")

    if clients.base_url:
        base_url, base_url_from = clients.base_url, "config"
    else:
        base_url, base_url_from = str(request.base_url).rstrip("/"), "request"

    routing, timeouts = config.routing, config.timeouts
    return Setup(
        base_url=base_url,
        base_url_from=base_url_from,
        provider_id=clients.provider_id,
        provider_name=clients.provider_name,
        api_key=clients.api_key,
        models=models,
        default=by_id[default_id],
        small=small,
        header_wait_ms=round((routing.queue_timeout_s + timeouts.first_byte_s) * 1000),
        # The upstream read timeout is first_byte_s, applied to every read.
        idle_ms=round(timeouts.first_byte_s * 1000),
        user=user,
        warnings=warnings,
    )


# ------------------------------------------------------------------ opencode


def opencode(setup: Setup) -> dict[str, Any]:
    options: dict[str, Any] = {
        "baseURL": setup.openai_url,
        "apiKey": setup.api_key,
        "headerTimeout": max(setup.header_wait_ms, OPENCODE_DEFAULT_TIMEOUT_MS),
        # No limit on the whole request: a long agentic reply can legitimately
        # stream for longer than any fixed timeout.
        "timeout": False,
    }
    # With vLLM and llama.cpp, response headers arrive at once and prefill
    # happens before the first chunk, so a long prefill counts against this.
    if setup.idle_ms > OPENCODE_DEFAULT_TIMEOUT_MS:
        options["chunkTimeout"] = setup.idle_ms
    if setup.user_header:
        options["headers"] = {USER_HEADER: setup.user_header}

    models: dict[str, Any] = {}
    for m in setup.models:
        entry: dict[str, Any] = {"name": m.name, "tool_call": m.tool_call, "reasoning": m.reasoning}
        if m.images:
            entry["attachment"] = True
            entry["modalities"] = {"input": ["text", "image"], "output": ["text"]}
        if m.context is not None:
            # opencode requires both halves once `limit` is given.
            output = m.max_output or min(OPENCODE_OUTPUT_CAP, m.context)
            entry["limit"] = {"context": m.context, "output": output}
        # opencode drops reasoning_effort from `options`: effort is a variant,
        # which it sends as reasoning_effort. Everything else in options is sent
        # as an extra field in the request body.
        extra = {k: v for k, v in m.request_params.items()
                 if not (k == "reasoning_effort" and m.effort)}
        if extra:
            entry["options"] = extra
        efforts = list(m.efforts)
        if m.effort and m.effort not in efforts:
            efforts.append(m.effort)
        if efforts:
            entry["variants"] = {e: {"reasoningEffort": e} for e in efforts}
        models[m.id] = entry

    config: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            setup.provider_id: {
                "npm": "@ai-sdk/openai-compatible",
                "name": setup.provider_name,
                "options": options,
                "models": models,
            }
        },
        "model": f"{setup.provider_id}/{setup.default.id}",
    }
    if setup.small:
        config["small_model"] = f"{setup.provider_id}/{setup.small.id}"
    if setup.default.effort:
        # There's no global default variant; an agent's only applies to the
        # model it is configured with.
        chosen = {"model": config["model"], "variant": setup.default.effort}
        config["agent"] = {"build": dict(chosen), "plan": dict(chosen)}
    return config


def opencode_wellknown(setup: Setup) -> dict[str, Any]:
    """For `opencode auth login <router url>`.

    opencode fetches this at every start and layers its `config` underneath the
    user's own, so the user keeps overriding whatever they like and everything
    else tracks the router's config. Logging in runs `auth.command` for a token;
    the router checks none, so it only needs to print something.
    """
    return {
        "auth": {"command": ["echo", setup.api_key], "env": setup.env_key},
        "config": opencode(setup),
    }


def _opencode_notes(setup: Setup) -> list[str]:
    notes = []
    if any(m.efforts or m.effort for m in setup.models):
        notes.append(
            "Reasoning effort is set as a variant: switch it with --variant (such as "
            "--variant max) or the variant key in the TUI. opencode ignores it anywhere else."
        )
    if any((m.max_output or 0) > OPENCODE_OUTPUT_CAP for m in setup.models):
        notes.append(
            f"opencode asks for at most {OPENCODE_OUTPUT_CAP:,} output tokens itself, "
            "whatever the model's limit."
        )
    missing = [m.id for m in setup.models if m.context is not None and m.max_output is None]
    if missing:
        notes.append(
            f"No max_output_tokens configured for {', '.join(missing)}, so opencode is told "
            f"{OPENCODE_OUTPUT_CAP}, which is the most it asks for by itself."
        )
    return notes


# --------------------------------------------------------------- Claude Code


def claude_code_env(setup: Setup) -> dict[str, str]:
    default = setup.default
    small = setup.small or default
    env = {
        "ANTHROPIC_BASE_URL": setup.base_url,
        "ANTHROPIC_AUTH_TOKEN": setup.api_key,
        # Where new sessions start (v2.1.236+). Older versions start on an alias
        # instead, and every alias below leads here anyway.
        "ANTHROPIC_DEFAULT_MODEL": default.id,
    }
    # Whichever tier Claude Code reaches for, it lands on a model the router
    # serves. The haiku tier is also its background model.
    for tier, model in (("OPUS", default), ("SONNET", default), ("FABLE", default), ("HAIKU", small)):
        env[f"ANTHROPIC_DEFAULT_{tier}_MODEL"] = model.id
        env[f"ANTHROPIC_DEFAULT_{tier}_MODEL_NAME"] = model.name
    others = [m for m in setup.models if m.id not in (default.id, small.id)]
    if others:
        # The picker has room for exactly one model beyond the tiers.
        env["ANTHROPIC_CUSTOM_MODEL_OPTION"] = others[0].id
        env["ANTHROPIC_CUSTOM_MODEL_OPTION_NAME"] = others[0].name
    if default.context is not None:
        # Otherwise it compacts at whatever window it guesses for an unknown id.
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(default.context)
    if default.max_output is not None:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(default.max_output)
    if setup.header_wait_ms > CLAUDE_API_TIMEOUT_MS:
        env["API_TIMEOUT_MS"] = str(setup.header_wait_ms)
    if setup.idle_ms > CLAUDE_BODY_IDLE_MS:
        env["API_FORCE_IDLE_TIMEOUT"] = "0"
    if setup.idle_ms > CLAUDE_WATCHDOG_MS:
        env["CLAUDE_STREAM_IDLE_TIMEOUT_MS"] = str(min(setup.idle_ms, CLAUDE_WATCHDOG_MAX_MS))
    if setup.user_header:
        env["ANTHROPIC_CUSTOM_HEADERS"] = f"{USER_HEADER}: {setup.user_header}"
    return env


def claude_code(setup: Setup) -> dict[str, Any]:
    return {"env": claude_code_env(setup)}


def claude_code_shell(setup: Setup) -> str:
    return "".join(
        f"export {key}={shlex.quote(value)}\n" for key, value in claude_code_env(setup).items()
    )


def _claude_notes(setup: Setup) -> list[str]:
    notes = [
        ("Claude Code speaks the Anthropic Messages API, so each backend serving these "
         "models has to as well. ninfer, llama.cpp, vLLM and LM Studio do."),
        ("It notes an unrecognized_model at startup for any model id that isn't one of "
         "Claude's. That's expected."),
    ]
    if any(m.request_params for m in (setup.default, setup.small) if m):
        notes.append(
            "Claude Code can't add request_params (such as reasoning_effort) to its "
            "requests, so the backend's own defaults apply."
        )
    others = [m.id for m in setup.models
              if m.id not in (setup.default.id, (setup.small or setup.default).id)]
    if len(others) > 1:
        notes.append(
            f"The /model picker has room for one extra model, {others[0]}; "
            f"{', '.join(others[1:])} can still be chosen with --model."
        )
    return notes


# ----------------------------------------------------------------- Qwen Code


def qwen_code(setup: Setup) -> dict[str, Any]:
    providers = []
    for m in setup.models:
        generation: dict[str, Any] = {"timeout": setup.header_wait_ms}
        if m.context is not None:
            generation["contextWindowSize"] = m.context
        if m.max_output is not None:
            generation["samplingParams"] = {"max_tokens": m.max_output}
        if m.request_params:
            generation["extra_body"] = m.request_params
        if setup.user_header:
            generation["customHeaders"] = {USER_HEADER: setup.user_header}
        providers.append({
            "id": m.id,
            "name": m.name,
            "description": setup.provider_name,
            "baseUrl": setup.openai_url,
            "envKey": setup.env_key,
            "generationConfig": generation,
        })
    return {
        "env": {setup.env_key: setup.api_key},
        "modelProviders": {"openai": providers},
        "security": {"auth": {"selectedType": "openai"}},
        "model": {"name": setup.default.id},
    }


# ----------------------------------------------------------------------- Zed


def zed(setup: Setup) -> dict[str, Any]:
    models = []
    for m in setup.models:
        entry: dict[str, Any] = {"name": m.id, "display_name": m.name}
        if m.context is not None:
            entry["max_tokens"] = m.context
        if m.max_output is not None:
            entry["max_output_tokens"] = m.max_output
        if m.effort in ZED_EFFORTS:
            entry["reasoning_effort"] = m.effort
        entry["capabilities"] = {
            "tools": m.tool_call,
            "images": m.images,
            "parallel_tool_calls": False,
            "prompt_cache_key": False,
        }
        models.append(entry)

    def pick(model: Model) -> dict[str, str]:
        return {"provider": setup.provider_id, "model": model.id}

    agent: dict[str, Any] = {"default_model": pick(setup.default)}
    if setup.small:
        agent["thread_summary_model"] = pick(setup.small)
        agent["commit_message_model"] = pick(setup.small)
    provider: dict[str, Any] = {"api_url": setup.openai_url, "available_models": models}
    if setup.user_header:
        provider["custom_headers"] = {USER_HEADER: setup.user_header}
    return {
        "language_models": {"openai_compatible": {setup.provider_id: provider}},
        "agent": agent,
    }


def _zed_notes(setup: Setup) -> list[str]:
    notes = []
    extra = {k for m in setup.models for k in m.request_params if k != "reasoning_effort"}
    if extra:
        notes.append(
            f"Zed sends reasoning_effort, but not {', '.join(sorted(extra))}, "
            "so the backend's defaults apply for those."
        )
    odd = sorted({m.effort for m in setup.models if m.effort and m.effort not in ZED_EFFORTS})
    if odd:
        notes.append(f"Zed has no reasoning_effort {', '.join(odd)}, so it isn't set.")
    if any(m.context is None for m in setup.models):
        notes.append("Zed needs max_tokens for every model; add it where it's missing.")
    return notes


# ------------------------------------------------------------------ registry


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


@dataclass(frozen=True)
class Client:
    id: str
    label: str
    # Where the file goes.
    file: str
    download: str
    render: Callable[[Setup], str]
    steps: Callable[[Setup, str], list[dict[str, str]]]
    notes: Callable[[Setup], list[str]]


def login_url(setup: Setup) -> str:
    """What to give `opencode auth login`. opencode appends /.well-known/opencode
    to it, so a name in the path survives where a query string would not."""
    if setup.user:
        return f"{setup.base_url}/u/{quote(setup.user, safe='')}"
    return setup.base_url


def _opencode_steps(setup: Setup, url: str) -> list[dict[str, str]]:
    return [
        {"text": ("Recommended: log in to the router once. opencode then fetches this "
                  "config from the router every time it starts, under your own settings, "
                  "so it follows every change to the router's config."),
         "command": f"opencode auth login {login_url(setup)}"},
        {"text": "Or keep a copy alongside your own config, which it's merged with:",
         "command": (f"curl -s '{url}' -o ~/.config/opencode/llm-router.json\n"
                     "export OPENCODE_CONFIG=~/.config/opencode/llm-router.json")},
    ]


def _claude_steps(setup: Setup, url: str) -> list[dict[str, str]]:
    shell = url + ("&" if "?" in url else "?") + "format=shell"
    return [
        {"text": "Try it without changing any files:",
         "command": f"claude --settings \"$(curl -s '{url}')\""},
        {"text": "To keep it, merge the env block into ~/.claude/settings.json, or export "
                 "the same settings in your shell:",
         "command": f"eval \"$(curl -s '{shell}')\""},
    ]


def _qwen_steps(setup: Setup, url: str) -> list[dict[str, str]]:
    return [
        {"text": "Save it as ~/.qwen/settings.json if you have none; otherwise merge in its "
                 "env, modelProviders, security and model keys.",
         "command": f"curl -s '{url}' -o ~/.qwen/settings.json"},
    ]


def _zed_steps(setup: Setup, url: str) -> list[dict[str, str]]:
    return [
        {"text": "Merge this into Zed's settings (the zed: open settings command).",
         "command": ""},
        {"text": "Zed keeps API keys out of settings.json: set this before starting Zed, or "
                 "enter any value in the agent panel's provider settings.",
         "command": f"export {setup.env_key}={shlex.quote(setup.api_key)}"},
    ]


CLIENTS: dict[str, Client] = {
    c.id: c
    for c in (
        Client("opencode", "opencode",
               "~/.config/opencode/opencode.json, or wherever OPENCODE_CONFIG points",
               "opencode.json",
               lambda s: _json(opencode(s)), _opencode_steps, _opencode_notes),
        Client("claude-code", "Claude Code", "~/.claude/settings.json", "settings.json",
               lambda s: _json(claude_code(s)), _claude_steps, _claude_notes),
        Client("qwen-code", "Qwen Code", "~/.qwen/settings.json", "settings.json",
               lambda s: _json(qwen_code(s)), _qwen_steps, lambda s: []),
        Client("zed", "Zed", "~/.config/zed/settings.json", "settings.json",
               lambda s: _json(zed(s)), _zed_steps, _zed_notes),
    )
}


# -------------------------------------------------------------------- routes


def routes(router: Router) -> list[Route]:
    def setup_or_error(request: Request) -> Setup | Response:
        try:
            return build_setup(router, request)
        except BadRequest as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    def config_url(request: Request, client: Client) -> str:
        query = {k: v for k, v in request.query_params.items() if k in ("model", "user") and v}
        url = str(request.base_url).rstrip("/") + f"/clients/{client.id}"
        return url + ("?" + urlencode(query) if query else "")

    async def index(request: Request) -> Response:
        setup = setup_or_error(request)
        if isinstance(setup, Response):
            return setup
        return JSONResponse({
            "base_url": setup.base_url,
            "base_url_from": setup.base_url_from,
            "provider": {"id": setup.provider_id, "name": setup.provider_name},
            "api_key_env": setup.env_key,
            "default_model": setup.default.id,
            "small_model": setup.small.id if setup.small else None,
            "user": setup.user,
            "models": [
                {
                    "id": m.id, "name": m.name, "context": m.context,
                    "context_from": m.context_from, "max_output_tokens": m.max_output,
                    "tool_call": m.tool_call, "reasoning": m.reasoning, "images": m.images,
                    "request_params": m.request_params,
                }
                for m in setup.models
            ],
            "clients": [
                {
                    "id": c.id, "label": c.label, "file": c.file, "download": c.download,
                    "url": config_url(request, c),
                    "steps": c.steps(setup, config_url(request, c)),
                    "notes": c.notes(setup),
                    "content": c.render(setup),
                }
                for c in CLIENTS.values()
            ],
            "warnings": setup.warnings,
        }, headers={"cache-control": "no-store"})

    async def one(request: Request) -> Response:
        client = CLIENTS.get(request.path_params["client"])
        if client is None:
            return JSONResponse(
                {"error": f"no client '{request.path_params['client']}' "
                          f"(known: {', '.join(CLIENTS)})"},
                status_code=404,
            )
        setup = setup_or_error(request)
        if isinstance(setup, Response):
            return setup
        if client.id == "claude-code" and request.query_params.get("format") == "shell":
            return Response(claude_code_shell(setup), media_type="text/plain; charset=utf-8",
                            headers={"cache-control": "no-store"})
        return Response(client.render(setup), media_type="application/json",
                        headers={"cache-control": "no-store"})

    async def wellknown(request: Request) -> Response:
        try:
            setup = build_setup(router, request, user=request.path_params.get("user"))
        except BadRequest as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(opencode_wellknown(setup), headers={"cache-control": "no-store"})

    return [
        Route("/clients", index, methods=["GET"]),
        Route("/clients/{client}", one, methods=["GET"]),
        Route("/.well-known/opencode", wellknown, methods=["GET"]),
        # For `opencode auth login <router>/u/<name>`, so the config it keeps
        # fetching names its user to the session dashboard.
        Route("/u/{user}/.well-known/opencode", wellknown, methods=["GET"]),
    ]
