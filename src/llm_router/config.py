"""Configuration loading and validation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .affinity import DEFAULT_SESSION_HEADERS
from .tracking import DEFAULT_USER_HEADERS

# Backend engines we know how to probe. They differ in where they publish context
# length and load, and in whether they want to be saturated -- see BackendConfig.
KINDS = ("ninfer", "llamacpp", "vllm", "lmstudio", "openai")

# ninfer caps --max-concurrency at 8, llama.cpp slots are bounded by -np, LM Studio
# by its parallel-requests setting. vLLM's --max-num-seqs defaults to 256, so it is
# allowed a much higher ceiling.
SANE_CAPACITY_LIMIT = 64
VLLM_CAPACITY_LIMIT = 1024


class ConfigError(ValueError):
    """Raised when a config file is malformed or internally inconsistent."""


@dataclass(frozen=True)
class BackendConfig:
    name: str
    url: str
    capacity: int
    models: tuple[str, ...]
    kind: str = "ninfer"  # see KINDS
    # Model id to send upstream, if it differs from what clients ask for. ninfer 404s
    # on an unknown model id; llama.cpp accepts anything.
    upstream_model: str | None = None
    # Override the liveness endpoint when a backend sits behind something unusual.
    health_path: str | None = None
    # Usable context in tokens. Left unset, the router discovers it from the backend
    # (see BackendClients.discover_context); set it to override a wrong or missing
    # value. This is the *served* context, not the model's architectural maximum.
    context_length: int | None = None
    api_key: str | None = None
    # Extra headers merged into every upstream request.
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def probe_path(self) -> str:
        """Liveness endpoint. LM Studio has no /health, so its model list stands in."""
        if self.health_path:
            return self.health_path
        if self.kind == "lmstudio":
            return "/api/v0/models"
        return "/health"

    @property
    def load_path(self) -> str | None:
        """Endpoint reporting the backend's own view of its load, where one exists.

        Used to cross-check our in-flight count -- a mismatch means something else is
        sharing the host. ninfer and LM Studio publish nothing of the sort.
        """
        if self.kind == "llamacpp":
            return "/slots"  # on unless --no-slots
        if self.kind == "vllm":
            return "/load"  # requires --enable-server-load-tracking
        return None

    @property
    def batches_continuously(self) -> bool:
        """True for engines that want to be saturated rather than trickle-fed.

        vLLM schedules a continuous batch and queues internally without head-of-line
        blocking, so gating it to a small capacity wastes throughput.
        """
        return self.kind == "vllm"


@dataclass(frozen=True)
class RoutingConfig:
    # How long a request will hold out for its pinned backend before spilling.
    affinity_wait_ms: int = 1500
    # How long a session -> backend pin survives without use.
    session_ttl_s: int = 1800
    # Max sessions tracked before LRU eviction.
    max_sessions: int = 20000
    # How long a request may sit in the proxy queue before we give up with a 503.
    # This is normal backpressure: hosts are up, just busy.
    queue_timeout_s: float = 300.0
    # How long to wait when *no* backend for the model is up at all. Short, because
    # unlike a busy pool there is nothing to be gained by waiting -- but non-zero, so
    # a restart or a blip that resolves within a probe cycle or two rides through.
    unavailable_grace_s: float = 10.0
    # Retries on a *different* backend, only before the first byte reaches the client.
    max_retries: int = 2
    # Number of message-boundary hashes recorded per request (longest-prefix depth).
    affinity_depth: int = 3
    # Request headers carrying an exact conversation id, tried in order before
    # falling back to hashing the prompt. Override when a client renames its
    # header (Open WebUI's FORWARD_SESSION_INFO_HEADER_CHAT_ID, for instance).
    session_headers: tuple[str, ...] = DEFAULT_SESSION_HEADERS
    # Add stream_options.include_usage when the client omitted it. Off by default:
    # it appends a chunk the client did not ask for. Most agentic clients (anything
    # on the Vercel AI SDK, which opencode uses) already request usage themselves.
    inject_usage: bool = False


@dataclass(frozen=True)
class HealthConfig:
    interval_s: float = 5.0
    timeout_s: float = 3.0
    # Consecutive probe failures before a backend is pulled from rotation.
    failure_threshold: int = 2
    # Backoff bounds applied after a passive (in-request) failure.
    cooldown_base_s: float = 2.0
    cooldown_max_s: float = 60.0


@dataclass(frozen=True)
class TimeoutConfig:
    connect_s: float = 10.0
    # Time budget for the upstream to produce its first byte. Prefill on a long
    # agentic prompt is slow, so this is generous.
    first_byte_s: float = 600.0
    # Max gap between streamed chunks before we consider the upstream stalled.
    stream_idle_s: float = 300.0


@dataclass(frozen=True)
class TrackingConfig:
    """Per-user session tracking, shown at /dashboard."""

    enabled: bool = True
    # Request headers naming the user, tried in order. With none of them present
    # a user is known by the address they connect from.
    user_headers: tuple[str, ...] = DEFAULT_USER_HEADERS
    # An in-flight request that has had nothing back for this long is flagged
    # slow, and at stuck_after_s, stuck. "Nothing back" is time since its last
    # response byte, or since it arrived if none has come yet.
    slow_after_s: float = 15.0
    stuck_after_s: float = 60.0
    # How long an idle session stays listed.
    retain_s: float = 3600.0
    # Most sessions remembered; beyond it the longest-idle go first.
    max_sessions: int = 2000


@dataclass(frozen=True)
class ModelConfig:
    """What coding agents are told about one model. Everything is optional."""

    # Display name. The model's id is what clients send; this is what they show.
    name: str | None = None
    tool_call: bool = True
    reasoning: bool = False
    # Accepts images as input.
    images: bool = False
    # The window clients should assume. Normally the router discovers it from the
    # backends; set this to tell clients less (to keep long sessions quicker, say),
    # or to fill in a window that can't be discovered. Never more than the
    # backends really serve: a larger value is capped, with a warning.
    context_length: int | None = None
    max_output_tokens: int | None = None
    # Extra request-body fields clients should send with every request, such as
    # reasoning_effort or chat_template_kwargs.
    request_params: dict[str, Any] = field(default_factory=dict)
    # Effort levels the model accepts, for clients that let you switch between
    # them (opencode's variants). request_params.reasoning_effort is the default.
    reasoning_efforts: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClientsConfig:
    """How the router describes itself in the client configs it generates."""

    # The URL clients should use, without /v1. Left unset, each config uses the
    # URL it was fetched from, which is right unless it was fetched as localhost.
    base_url: str | None = None
    # The provider's id and name in clients that group models by provider.
    provider_id: str = "llm-router"
    provider_name: str = "llm-router"
    # The router doesn't check it, but most clients insist on having one.
    api_key: str = "unused"
    # The model clients start on. Defaults to the first model in the config.
    default_model: str | None = None
    # For background work: titles, summaries, commit messages. Clients use
    # their own default when this is unset.
    small_model: str | None = None


@dataclass(frozen=True)
class Config:
    backends: tuple[BackendConfig, ...]
    # Maps a requested model name onto a configured one. A "*" key is a catch-all
    # for anything unmatched. Claude Code sends whatever model name it was given
    # and issues background requests for its own small/fast model, so without an
    # alias that background traffic 404s.
    model_aliases: dict[str, str] = field(default_factory=dict)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    # Per-model metadata for client configs, keyed by served model name.
    models: dict[str, ModelConfig] = field(default_factory=dict)
    clients: ClientsConfig = field(default_factory=ClientsConfig)
    host: str = "0.0.0.0"
    port: int = 8080
    log_file: str | None = None

    def backends_for(self, model: str) -> tuple[BackendConfig, ...]:
        return tuple(b for b in self.backends if model in b.models)

    @property
    def all_models(self) -> tuple[str, ...]:
        return all_model_names(self.backends)


def all_model_names(backends: tuple[BackendConfig, ...]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for backend in backends:
        for model in backend.models:
            seen.setdefault(model, None)
    return tuple(seen)


def _expand(value: Any) -> Any:
    """Expand ${ENV_VAR} references in strings so secrets stay out of the file."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _section(raw: dict[str, Any], key: str, cls: type) -> Any:
    """Build a dataclass from a config section, rejecting unknown keys."""
    data = raw.get(key) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"'{key}' must be a mapping, got {type(data).__name__}")
    known = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"unknown key(s) in '{key}': {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )
    data = dict(data)
    for name in ("session_headers", "user_headers"):
        if name not in data:
            continue
        value = data[name]
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ConfigError(f"{key}.{name} must be a list of header names")
        # Header lookups are lowercase; normalise so config casing does not matter.
        data[name] = tuple(str(v).lower() for v in value)
    return cls(**data)


def _parse_backend(raw: Any, index: int) -> BackendConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"backends[{index}] must be a mapping")

    missing = [k for k in ("name", "url", "capacity", "models") if k not in raw]
    if missing:
        raise ConfigError(f"backends[{index}] is missing: {', '.join(missing)}")

    name = str(raw["name"])
    capacity = raw["capacity"]
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
        raise ConfigError(f"backend '{name}': capacity must be an integer >= 1")
    kind_hint = str(raw.get("kind", "ninfer"))
    limit = VLLM_CAPACITY_LIMIT if kind_hint == "vllm" else SANE_CAPACITY_LIMIT
    if capacity > limit:
        raise ConfigError(
            f"backend '{name}': capacity {capacity} exceeds {limit}; this should match "
            "the host's --max-concurrency (ninfer), -np (llama.cpp), --max-num-seqs "
            "(vLLM), or parallel-request setting (LM Studio)"
        )

    models = raw["models"]
    if isinstance(models, str):
        models = [models]
    if not isinstance(models, list) or not models:
        raise ConfigError(f"backend '{name}': models must be a non-empty list")

    context_length = raw.get("context_length")
    if context_length is not None:
        if (
            not isinstance(context_length, int)
            or isinstance(context_length, bool)
            or context_length < 1
        ):
            raise ConfigError(
                f"backend '{name}': context_length must be a positive integer"
            )

    kind = str(raw.get("kind", "ninfer"))
    if kind not in KINDS:
        raise ConfigError(
            f"backend '{name}': kind must be one of {', '.join(KINDS)} (got '{kind}')"
        )

    headers = raw.get("headers") or {}
    if not isinstance(headers, dict):
        raise ConfigError(f"backend '{name}': headers must be a mapping")

    return BackendConfig(
        name=name,
        url=str(raw["url"]).rstrip("/"),
        capacity=capacity,
        models=tuple(str(m) for m in models),
        kind=kind,
        upstream_model=(str(raw["upstream_model"]) if raw.get("upstream_model") else None),
        health_path=(str(raw["health_path"]) if raw.get("health_path") else None),
        context_length=context_length,
        api_key=(str(raw["api_key"]) if raw.get("api_key") else None),
        headers={str(k): str(v) for k, v in headers.items()},
    )


def _parse_tracking(raw: dict[str, Any]) -> TrackingConfig:
    tracking = _section(raw, "tracking", TrackingConfig)
    if not isinstance(tracking.enabled, bool):
        raise ConfigError("tracking.enabled must be true or false")
    for name in ("slow_after_s", "stuck_after_s", "retain_s"):
        value = getattr(tracking, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"tracking.{name} must be a positive number of seconds")
    if tracking.slow_after_s > tracking.stuck_after_s:
        raise ConfigError("tracking.slow_after_s must not exceed tracking.stuck_after_s")
    limit = tracking.max_sessions
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ConfigError("tracking.max_sessions must be an integer >= 1")
    return tracking


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _parse_models(raw: dict[str, Any], served: set[str]) -> dict[str, ModelConfig]:
    section = raw.get("models") or {}
    if not isinstance(section, dict):
        raise ConfigError("'models' must be a mapping of model name -> settings")
    known = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    models: dict[str, ModelConfig] = {}
    for name, entry in section.items():
        name = str(name)
        where = f"models['{name}']"
        if name not in served:
            raise ConfigError(
                f"{where}: no backend serves a model of that name "
                f"(known: {', '.join(sorted(served))})"
            )
        entry = entry or {}
        if not isinstance(entry, dict):
            raise ConfigError(f"{where} must be a mapping")
        unknown = set(entry) - known
        if unknown:
            raise ConfigError(
                f"unknown key(s) in {where}: {', '.join(sorted(unknown))}. "
                f"Valid keys: {', '.join(sorted(known))}"
            )
        if "reasoning_efforts" in entry:
            efforts = entry["reasoning_efforts"]
            if not isinstance(efforts, list) or not all(isinstance(e, str) for e in efforts):
                raise ConfigError(f"{where}.reasoning_efforts must be a list of names")
            entry = {**entry, "reasoning_efforts": tuple(efforts)}
        model = ModelConfig(**entry)
        if model.name is not None and not isinstance(model.name, str):
            raise ConfigError(f"{where}.name must be a string")
        for flag in ("tool_call", "reasoning", "images"):
            if not isinstance(getattr(model, flag), bool):
                raise ConfigError(f"{where}.{flag} must be true or false")
        for limit in ("context_length", "max_output_tokens"):
            value = getattr(model, limit)
            if value is not None and not _positive_int(value):
                raise ConfigError(f"{where}.{limit} must be a positive integer")
        if not isinstance(model.request_params, dict):
            raise ConfigError(f"{where}.request_params must be a mapping")
        models[name] = model
    return models


PROVIDER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _parse_clients(raw: dict[str, Any], served: set[str]) -> ClientsConfig:
    clients = _section(raw, "clients", ClientsConfig)
    if not PROVIDER_ID.match(str(clients.provider_id)):
        # Clients turn it into an environment variable name (GLM53_API_KEY), so
        # it has to survive that.
        raise ConfigError(
            "clients.provider_id may contain only letters, digits, '-' and '_'"
        )
    for key in ("default_model", "small_model"):
        value = getattr(clients, key)
        if value is not None and value not in served:
            raise ConfigError(
                f"clients.{key} '{value}' is not served by any backend "
                f"(known: {', '.join(sorted(served))})"
            )
    base_url = clients.base_url
    if base_url is not None:
        base_url = str(base_url).rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ConfigError("clients.base_url must start with http:// or https://")
        # Clients add /v1 themselves where they need it; accept it written either way.
        base_url = base_url.removesuffix("/v1")
    return ClientsConfig(
        base_url=base_url,
        provider_id=str(clients.provider_id),
        provider_name=str(clients.provider_name),
        api_key=str(clients.api_key),
        default_model=clients.default_model,
        small_model=clients.small_model,
    )


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    return parse_config(path.read_text())


def parse_config(text: str) -> Config:
    """Parse and validate config text.

    Separate from load_config so a reload can hash and parse the very same bytes,
    rather than reading the file twice and racing a writer in between.
    """
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    raw = _expand(raw)

    backends_raw = raw.get("backends")
    if not backends_raw:
        raise ConfigError("config must define at least one backend")
    if not isinstance(backends_raw, list):
        raise ConfigError("'backends' must be a list")

    backends = tuple(_parse_backend(b, i) for i, b in enumerate(backends_raw))

    names = [b.name for b in backends]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ConfigError(f"duplicate backend name(s): {', '.join(sorted(dupes))}")

    listen = raw.get("listen") or {}
    if not isinstance(listen, dict):
        raise ConfigError("'listen' must be a mapping with host/port")

    aliases_raw = raw.get("model_aliases") or {}
    if not isinstance(aliases_raw, dict):
        raise ConfigError("'model_aliases' must be a mapping of name -> model")
    model_aliases = {str(k): str(v) for k, v in aliases_raw.items()}
    known = set(all_model_names(backends))
    for alias, target in model_aliases.items():
        if target not in known:
            raise ConfigError(
                f"model_aliases['{alias}'] points at '{target}', which no backend "
                f"serves (known: {', '.join(sorted(known))})"
            )

    return Config(
        backends=backends,
        model_aliases=model_aliases,
        routing=_section(raw, "routing", RoutingConfig),
        health=_section(raw, "health", HealthConfig),
        timeouts=_section(raw, "timeouts", TimeoutConfig),
        tracking=_parse_tracking(raw),
        models=_parse_models(raw, known),
        clients=_parse_clients(raw, known),
        host=str(listen.get("host", "0.0.0.0")),
        port=int(listen.get("port", 8080)),
        log_file=(str(raw["log_file"]) if raw.get("log_file") else None),
    )
