"""Configuration for coding agents, generated from the router's own config.

A client needs more than the URL to work well: the model ids, the context each
takes, how much output to ask for, how long a request may legitimately wait
before its first byte, and any extra request fields a model wants. The router
already knows all of that, from its config or by discovering it, so it hands it
out here rather than have it copied into each client by hand, where it goes stale
the first time the config changes.

The formats follow each client's own documentation, checked September 2026:
opencode (opencode.json, and its /.well-known/opencode remote config), Claude
Code (settings.json `env`), Qwen Code (settings.json `modelProviders`), Zed
(settings.json `language_models.openai_compatible`) and Oh My Pi (models.yml
`providers`, checked against its source as well).
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

import yaml
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .config import ROLES, RoleConfig

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
# Oh My Pi's thinking levels, in its order. It has no "none": thinking off is a
# level of its own, which sends no effort at all.
OMP_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")
# Oh My Pi's stream watchdogs: the first event, then the gap between events.
OMP_DEFAULT_TIMEOUT_MS = 300_000


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


# Roles that fall back to another when unset. The rest stay unset, and each
# client then does whatever it does without one.
ROLE_FALLBACKS = {"thinking": "default", "plan": "thinking"}


@dataclass
class Role:
    model: Model
    # This role's own effort; None leaves the model's default.
    effort: str | None
    # Set in the router's config, rather than filled in from another role.
    configured: bool

    def selector(self, provider_id: str, suffix: Callable[[str], str | None] = lambda e: e) -> str:
        """provider/model, with `:effort` where the client has a name for it."""
        effort = suffix(self.effort) if self.effort else None
        return f"{provider_id}/{self.model.id}" + (f":{effort}" if effort else "")


@dataclass
class Setup:
    """Everything the generators need, worked out once per request."""

    base_url: str
    base_url_from: str
    provider_id: str
    provider_name: str
    api_key: str
    models: list[Model]
    # Every role with a model, after fallbacks. "default" is always here.
    roles: dict[str, Role]
    # How long a request may take before the router sends response headers:
    # queued for a slot, then waiting on its backend.
    header_wait_ms: int
    # The longest silence the router allows from a backend mid-response.
    idle_ms: int
    user: str | None
    warnings: list[str]

    @property
    def default(self) -> Model:
        return self.roles["default"].model

    @property
    def small(self) -> Model | None:
        role = self.roles.get("small")
        return role.model if role else None

    def role(self, name: str) -> Role | None:
        return self.roles.get(name)

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

    configured = dict(clients.roles)
    asked = request.query_params.get("model")
    if asked:
        if asked not in by_id:
            raise BadRequest(f"no model '{asked}' (known: {', '.join(served)})")
        # The configured effort went with the configured model, not this one.
        kept = configured.get("default")
        configured["default"] = kept if kept and kept.model == asked else RoleConfig(asked)
    roles = _resolve_roles(configured, by_id, served[0], warnings)

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
        roles=roles,
        header_wait_ms=round((routing.queue_timeout_s + timeouts.first_byte_s) * 1000),
        # The upstream read timeout is first_byte_s, applied to every read.
        idle_ms=round(timeouts.first_byte_s * 1000),
        user=user,
        warnings=warnings,
    )


def _resolve_roles(
    configured: dict[str, RoleConfig], by_id: dict[str, Model], first: str, warnings: list[str]
) -> dict[str, Role]:
    roles: dict[str, Role] = {}
    for name in ROLES:
        chosen = configured.get(name)
        if chosen is not None:
            roles[name] = Role(by_id[chosen.model], chosen.effort, True)
        elif name == "default":
            roles[name] = Role(by_id[first], None, False)
        elif ROLE_FALLBACKS.get(name) in roles:
            base = roles[ROLE_FALLBACKS[name]]
            roles[name] = Role(base.model, base.effort, False)

    # A role's effort is one its model can be switched to, whether or not
    # reasoning_efforts lists it, so clients that offer a choice offer it.
    for role in roles.values():
        if role.effort and role.effort not in role.model.efforts:
            role.model.efforts = (*role.model.efforts, role.effort)

    vision = configured.get("vision")
    if vision and not by_id[vision.model].images:
        warnings.append(
            f"clients.roles.vision is {vision.model}, which isn't marked as taking images. "
            f"Set models['{vision.model}'].images: true if it does."
        )
    compaction, default = roles.get("compaction"), roles["default"].model
    if compaction and compaction.model.context and default.context and (
        compaction.model.context < default.context
    ):
        warnings.append(
            f"clients.roles.compaction is {compaction.model.id}, with a smaller context "
            f"({compaction.model.context}) than the default model's ({default.context}): "
            "it can't read a conversation long enough to need compacting."
        )
    return roles


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
        # Titles and summaries.
        config["small_model"] = f"{setup.provider_id}/{setup.small.id}"
    agents = _opencode_agents(setup)
    if agents:
        config["agent"] = agents
    return config


def _opencode_agents(setup: Setup) -> dict[str, Any]:
    """Roles, as opencode's built-in agents.

    There's no global default variant, and an agent's only applies to the model
    it is configured with, so an agent that should use an effort names both.
    """
    def pick(role: Role) -> dict[str, str]:
        entry = {"model": f"{setup.provider_id}/{role.model.id}"}
        # Without a variant opencode sends no reasoning_effort at all, so the
        # model's default has to be asked for too.
        effort = role.effort or role.model.effort
        if effort:
            entry["variant"] = effort
        return entry

    agents: dict[str, Any] = {}
    default, plan = setup.roles["default"], setup.roles["plan"]
    if default.effort or default.model.effort:
        agents["build"] = pick(default)
    if plan.model is not default.model or plan.effort or plan.model.effort:
        agents["plan"] = pick(plan)
    subagent = setup.role("subagent")
    if subagent:
        # Its two built-in subagents: general for delegated work, explore for
        # searching the codebase.
        agents["general"] = pick(subagent)
        agents["explore"] = pick(subagent)
    compaction = setup.role("compaction")
    if compaction:
        agents["compaction"] = pick(compaction)
    return agents


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
    thinking = setup.roles["thinking"].model
    env = {
        "ANTHROPIC_BASE_URL": setup.base_url,
        "ANTHROPIC_AUTH_TOKEN": setup.api_key,
        # Where new sessions start (v2.1.236+). Older versions start on an alias
        # instead, and every alias below leads here anyway.
        "ANTHROPIC_DEFAULT_MODEL": default.id,
    }
    # Whichever tier Claude Code reaches for, it lands on a model the router
    # serves. The haiku tier is also its background model; opus is what the
    # opusplan alias plans with, before sonnet carries the plan out.
    tiers = (("OPUS", thinking), ("FABLE", thinking), ("SONNET", default), ("HAIKU", small))
    for tier, model in tiers:
        env[f"ANTHROPIC_DEFAULT_{tier}_MODEL"] = model.id
        env[f"ANTHROPIC_DEFAULT_{tier}_MODEL_NAME"] = model.name
    subagent = setup.role("subagent")
    conversing = [default, thinking]
    if subagent:
        # For subagents that don't name a model themselves.
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = subagent.model.id
        conversing.append(subagent.model)
    others = [m for m in setup.models if m.id not in {model.id for _, model in tiers}]
    if others:
        # The picker has room for exactly one model beyond the tiers.
        env["ANTHROPIC_CUSTOM_MODEL_OPTION"] = others[0].id
        env["ANTHROPIC_CUSTOM_MODEL_OPTION_NAME"] = others[0].name
    # One limit covers every model a conversation can be on, so it has to fit
    # the smallest of them.
    if default.context is not None:
        # Otherwise it compacts at whatever window it guesses for an unknown id.
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(
            min(m.context for m in conversing if m.context is not None)
        )
    if default.max_output is not None:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(
            min(m.max_output for m in conversing if m.max_output is not None)
        )
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
    if any(role.model.request_params for role in setup.roles.values()):
        notes.append(
            "Claude Code can't add request_params (such as reasoning_effort) to its "
            "requests, so the backend's own defaults apply."
        )
    plan, thinking = setup.role("plan"), setup.roles["thinking"]
    if plan and plan.configured and plan.model is not thinking.model:
        notes.append(
            f"Claude Code plans on its opus model, which is the thinking role's "
            f"{thinking.model.id}; it has no separate place for the plan role's {plan.model.id}."
        )
    notes += _unplaced(setup, "Claude Code", ("vision", "compaction"), per_role_effort=False)
    tier_ids = {setup.default.id, (setup.small or setup.default).id, thinking.model.id}
    others = [m.id for m in setup.models if m.id not in tier_ids]
    if len(others) > 1:
        notes.append(
            f"The /model picker has room for one extra model, {others[0]}; "
            f"{', '.join(others[1:])} can still be chosen with --model."
        )
    return notes


def _and(items: list[str]) -> str:
    """A list as it reads in a sentence: a, b and c."""
    return ", ".join(items[:-1]) + (" and " if len(items) > 1 else "") + items[-1]


def _unplaced(
    setup: Setup, client: str, missing: tuple[str, ...], per_role_effort: bool
) -> list[str]:
    """Notes for configured roles, and role efforts, a client has no place for."""
    notes = []
    roles = [name for name in missing if (role := setup.role(name)) and role.configured]
    if roles:
        notes.append(
            f"{client} has no setting for the {' or '.join(roles)} "
            f"role{'s' if len(roles) > 1 else ''}, so "
            f"{'they are' if len(roles) > 1 else 'it is'} left out."
        )
    if not per_role_effort:
        efforts = [name for name, role in setup.roles.items() if role.configured and role.effort]
        if efforts:
            notes.append(
                f"{client} can't set an effort per role, so the effort given for "
                f"{_and(efforts)} isn't used: each model's own default applies."
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
    config: dict[str, Any] = {
        "env": {setup.env_key: setup.api_key},
        "modelProviders": {"openai": providers},
        "security": {"auth": {"selectedType": "openai"}},
        "model": {"name": setup.default.id},
    }
    # Each is left empty by Qwen Code to mean the main model.
    for role, key in (
        ("small", "fastModel"),          # prompt suggestions, speculative execution
        ("vision", "visionModel"),       # describes images for a text-only model
        ("compaction", "compactionModel"),
    ):
        if chosen := setup.role(role):
            config[key] = chosen.model.id
    if subagent := setup.role("subagent"):
        # Its built-in Explore subagent; custom subagents name their own.
        config["agents"] = {"builtin": {"exploreModel": subagent.model.id}}
    return config


def _qwen_notes(setup: Setup) -> list[str]:
    return _unplaced(setup, "Qwen Code", ("thinking", "plan"), per_role_effort=False)


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

    agent: dict[str, Any] = {"default_model": _zed_pick(setup, setup.roles["default"])}
    for role, keys in (
        ("small", ("thread_summary_model", "commit_message_model")),
        ("subagent", ("subagent_model",)),
        ("compaction", ("compaction_model",)),
    ):
        if chosen := setup.role(role):
            for key in keys:
                agent[key] = _zed_pick(setup, chosen)
    provider: dict[str, Any] = {"api_url": setup.openai_url, "available_models": models}
    if setup.user_header:
        provider["custom_headers"] = {USER_HEADER: setup.user_header}
    return {
        "language_models": {"openai_compatible": {setup.provider_id: provider}},
        "agent": agent,
    }


def _zed_effort(role: Role) -> str | None:
    """The role's effort, if Zed can send it: only for a model it knows thinks,
    which is one with a reasoning_effort, and only as one of its own levels."""
    if role.effort in ZED_EFFORTS and role.model.effort in ZED_EFFORTS:
        return role.effort
    return None


def _zed_pick(setup: Setup, role: Role) -> dict[str, Any]:
    selection: dict[str, Any] = {"provider": setup.provider_id, "model": role.model.id}
    if effort := _zed_effort(role):
        selection["enable_thinking"] = effort != "none"
        selection["effort"] = effort
    return selection


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
    unsent = [name for name, role in setup.roles.items()
              if role.configured and role.effort and not _zed_effort(role)]
    if unsent:
        notes.append(
            f"Zed only sends an effort for a model with a reasoning_effort it knows, so "
            f"the effort for {_and(unsent)} isn't used."
        )
    return notes + _unplaced(setup, "Zed", ("thinking", "plan", "vision"), per_role_effort=True)


# ---------------------------------------------------------------- Oh My Pi


def _omp_efforts(m: Model) -> list[str]:
    """The thinking levels Oh My Pi can offer for a model, or none if a
    configured effort would pin it anyway."""
    if m.effort and m.effort not in OMP_EFFORTS:
        return []
    wanted = set(m.efforts) | ({m.effort} if m.effort else set())
    return [e for e in OMP_EFFORTS if e in wanted]


def oh_my_pi(setup: Setup) -> dict[str, Any]:
    compaction = setup.role("compaction")
    models = []
    for m in setup.models:
        efforts = _omp_efforts(m)
        entry: dict[str, Any] = {
            "id": m.id,
            "name": m.name,
            "reasoning": m.reasoning or bool(efforts),
            "input": ["text", "image"] if m.images else ["text"],
        }
        if not m.tool_call:
            entry["supportsTools"] = False
        if m.context is not None:
            entry["contextWindow"] = m.context
        if m.max_output is not None:
            entry["maxTokens"] = m.max_output
        compat: dict[str, Any] = {}
        if efforts:
            # Oh My Pi sends the chosen level as reasoning_effort. The
            # configured effort is where each session starts.
            thinking: dict[str, Any] = {"mode": "effort", "efforts": efforts}
            if m.effort:
                thinking["defaultLevel"] = m.effort
            entry["thinking"] = thinking
            compat["supportsReasoningEffort"] = True
            # For a model omp takes for Qwen 3.8 or later it switches thinking on
            # with enable_thinking, and sends the level as reasoning_effort only
            # when this is set -- which it does itself for LM Studio and vLLM, but
            # not behind a router, so every level would run at the template's
            # default (xhigh). Other models never read it.
            compat["qwenTemplateReasoningEffort"] = True
        # extraBody is laid over the finished request, so a reasoning_effort
        # left in it would override whichever level was picked.
        extra = {k: v for k, v in m.request_params.items()
                 if not (k == "reasoning_effort" and efforts)}
        if extra:
            compat["extraBody"] = extra
        if compat:
            entry["compat"] = compat
        if compaction and compaction.model is not m:
            # Summarises this model's sessions when they outgrow its context.
            entry["compactionModel"] = _omp_selector(setup, compaction)
        models.append(entry)

    provider: dict[str, Any] = {
        "baseUrl": setup.openai_url,
        "api": "openai-completions",
        # Taken as the name of an environment variable if one exists by that
        # name, and otherwise as the key itself.
        "apiKey": setup.api_key,
    }
    if setup.user_header:
        # A value starting with "!" would be run as a shell command; the
        # percent-encoding leaves none.
        provider["headers"] = {USER_HEADER: setup.user_header}
    # Its first-event watchdog is never shorter than its idle one, and covers
    # the wait for response headers, so one setting covers both. A longer idle
    # limit than the router's own costs nothing: the router gives up first.
    wait = max(setup.header_wait_ms, setup.idle_ms)
    if wait > OMP_DEFAULT_TIMEOUT_MS:
        provider["compat"] = {"streamIdleTimeoutMs": wait}
    provider["models"] = models
    return {"providers": {setup.provider_id: provider}}


# The router's roles, as omp's. Its tiny role (titles) and memory role fall
# back to smol by themselves, but commit has its own list of cloud models first.
OMP_ROLES = {
    "default": ("default",),
    "small": ("smol", "tiny", "commit"),
    "thinking": ("slow",),
    "plan": ("plan",),
    "subagent": ("task",),
    "vision": ("vision",),
}


def _omp_selector(setup: Setup, role: Role) -> str:
    """provider/model, with a thinking-level suffix where omp has that level."""
    return role.selector(setup.provider_id, lambda e: e if e in _omp_efforts(role.model) else None)


def oh_my_pi_roles(setup: Setup) -> dict[str, Any]:
    """The ~/.omp/agent/config.yml half: models.yml can't assign roles.

    Every resolved role is written out, fallbacks included. omp's own fallback
    for an unset role is a list of cloud models, which it would reach for first
    whenever one of them is logged in.
    """
    roles: dict[str, str] = {}
    for ours, theirs in OMP_ROLES.items():
        if role := setup.role(ours):
            for name in theirs:
                roles[name] = _omp_selector(setup, role)
    return {"modelRoles": roles}


def _yaml(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=1000)


def _omp_notes(setup: Setup) -> list[str]:
    notes = []
    pinned = sorted({m.effort for m in setup.models if m.effort and m.effort not in OMP_EFFORTS})
    if pinned:
        notes.append(
            f"Oh My Pi has no thinking level {', '.join(pinned)}, so reasoning_effort is "
            "sent as configured with every request, whatever level is picked."
        )
    if any(_omp_efforts(m) for m in setup.models):
        notes.append(
            "Reasoning effort is Oh My Pi's thinking level: cycle it with Shift+Tab, start "
            "with --thinking high, or add a suffix to a model role such as default: "
            f"{setup.provider_id}/{setup.default.id}:high."
        )
    if (setup.small is None and len(setup.models) > 1):
        notes.append(
            "Set clients.roles.small in the router's config to give Oh My Pi a model for "
            "its background work (titles, commit messages)."
        )
    unsent = sorted(name for name, role in setup.roles.items()
                    if role.configured and role.effort and role.effort not in _omp_efforts(role.model))
    if unsent:
        notes.append(
            f"Oh My Pi has no thinking level for the effort given for {_and(unsent)}, "
            "so it starts those at the model's own default."
        )
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
    media_type: str = "application/json"


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
    shell = _with_query(url, "format=shell")
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
                 "top-level keys.",
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


def _omp_steps(setup: Setup, url: str) -> list[dict[str, str]]:
    return [
        {"text": "Save it as ~/.omp/agent/models.yml if you have none; otherwise add its "
                 f"{setup.provider_id} provider to yours. omp models lists what it loaded.",
         "command": f"curl -s --create-dirs '{url}' -o ~/.omp/agent/models.yml"},
        {"text": "Then give its models their roles, in ~/.omp/agent/config.yml. These are "
                 f"also at {_with_query(url, 'format=config')}:",
         "command": _yaml(oh_my_pi_roles(setup)).rstrip("\n")},
    ]


def _with_query(url: str, query: str) -> str:
    return url + ("&" if "?" in url else "?") + query


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
               lambda s: _json(qwen_code(s)), _qwen_steps, _qwen_notes),
        Client("zed", "Zed", "~/.config/zed/settings.json", "settings.json",
               lambda s: _json(zed(s)), _zed_steps, _zed_notes),
        Client("oh-my-pi", "Oh My Pi", "~/.omp/agent/models.yml", "models.yml",
               lambda s: _yaml(oh_my_pi(s)), _omp_steps, _omp_notes,
               media_type="application/yaml; charset=utf-8"),
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
            "roles": {
                name: {"model": role.model.id, "effort": role.effort,
                       "configured": role.configured}
                for name, role in setup.roles.items()
            },
            "user": setup.user,
            # For clients configured from this document rather than a generated
            # file: how long a request may wait for its response to start
            # (queued, then waiting on its backend; for a non-streamed request,
            # the whole response), and how long a response may go quiet.
            "timeouts": {"header_wait_ms": setup.header_wait_ms, "idle_ms": setup.idle_ms},
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
        fmt = request.query_params.get("format")
        if client.id == "claude-code" and fmt == "shell":
            return Response(claude_code_shell(setup), media_type="text/plain; charset=utf-8",
                            headers={"cache-control": "no-store"})
        if client.id == "oh-my-pi" and fmt == "config":
            return Response(_yaml(oh_my_pi_roles(setup)), media_type=client.media_type,
                            headers={"cache-control": "no-store"})
        return Response(client.render(setup), media_type=client.media_type,
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
