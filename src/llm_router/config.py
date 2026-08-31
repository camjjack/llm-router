"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
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
        host=str(listen.get("host", "0.0.0.0")),
        port=int(listen.get("port", 8080)),
        log_file=(str(raw["log_file"]) if raw.get("log_file") else None),
    )
