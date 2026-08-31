# llm-router

A load-balancing proxy for local LLM backends that does two things ordinary routers get wrong:

1. **It never dispatches into a full backend.** Requests queue *here*, where the queue is visible,
   instead of being swallowed by a backend's internal pending queue where they block behind other
   work while other hosts sit idle.
2. **It keeps a conversation on one host.** Agentic loops resend the whole conversation every turn,
   so staying put turns a full prefill into a prompt-cache hit.

Speaks both the **OpenAI chat-completions** API and the **Anthropic Messages** API, so opencode and
Claude Code can point at the same router. Routes to **ninfer-windows**, **llama.cpp**, **vLLM** and
**LM Studio** backends, including a mix of engines serving the same model.

| Endpoint | For |
|---|---|
| `POST /v1/chat/completions` | opencode, aider, Cline, Continue, Zed |
| `POST /v1/messages` | Claude Code (also `?beta=true`) |
| `POST /v1/messages/count_tokens` | Claude Code token accounting |
| `GET /v1/models`, `GET /health`, `GET /stats` | discovery, liveness, telemetry |

| `kind` | Liveness | Load telemetry | Context from | Capacity should match |
|---|---|---|---|---|
| `ninfer` | `/health` | none | `/v1/models` `max_model_len` | `--max-concurrency` |
| `llamacpp` | `/health` | `/slots` | `/props` `n_ctx` (per slot) | `-np` / `--parallel` |
| `vllm` | `/health` | `/load` † | `/v1/models` `max_model_len` | `--max-num-seqs` |
| `lmstudio` | `/api/v0/models` ‡ | none | `/api/v0/models` `loaded_context_length` | parallel-requests setting |
| `openai` | `/health` | none | `/v1/models` | whatever the endpoint allows |

† needs `--enable-server-load-tracking`; absent is handled gracefully.
‡ LM Studio has no `/health` endpoint, so its model list stands in.

## Why not least-busy routing

ninfer publishes no load telemetry — `/health` is a hardcoded `{"status":"ok"}`, and there is no
`/slots` equivalent. Nothing outside the proxy can see a host's occupancy.

Worse, an overloaded ninfer doesn't say so. Each host admits `--max-concurrency` active requests
*plus* `--max-pending-requests` queued behind them (**default 16**). A router that dispatches to a
busy host isn't told "full" — the request is accepted into a 16-deep FIFO and sits there. Meanwhile
another host is idle. That is the failure mode people hit with LiteLLM's least-busy mode, and no
amount of tuning the heuristic fixes it, because the router's information is wrong.

So this router owns the accounting. It tracks in-flight requests per backend itself, treats
`capacity` as a hard gate, and holds anything that doesn't fit in its own FIFO queue.

## Why session affinity

Chat completions is stateless: the client resends the entire conversation each turn, so turn N+1's
prompt is turn N's prompt plus a couple of messages. ninfer reuses compatible prefixes, but its
retained-checkpoint capacity is small (`--max-private-continuations` defaults to `2 ×
max-concurrency`). A conversation that hops between hosts re-prefills from scratch every time.

The OpenAI protocol has **no session identifier** — nothing in the request says which conversation it
belongs to. So the router infers it: it hashes the conversation's message boundaries cumulatively and
looks them up longest-first. A boundary recorded on turn N is still present, and still hashes the
same, on turn N+1. No client changes needed.

Pins ignore the first message boundary by design: every conversation from a given client shares its
system prompt, and pinning on that alone would funnel every new session onto one host.

## Install

```bash
uv venv && uv pip install -e .
cp config.example.yaml config.yaml   # then edit
uv run llm-router check -c config.yaml
```

## Run

```bash
uv run llm-router serve -c config.yaml --tui     # with the live dashboard
uv run llm-router serve -c config.yaml           # plain, logs to stderr
uv run llm-router top --url http://127.0.0.1:8080   # dashboard for a running router
```

Point a client at it:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}]}'
```

`opencode` — in `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llm-router",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": { "qwen3.6-27b": { "name": "Qwen via router" } }
    }
  }
}
```

## Context windows

`/v1/models` advertises each model's usable context, discovered from the backends at startup (and
re-checked whenever a host comes back, since a restart may have changed its flags):

```json
{ "id": "qwen3.6-27b", "object": "model", "owned_by": "llm-router",
  "context_length": 32768, "max_model_len": 32768, "meta": { "n_ctx": 32768 } }
```

Three spellings of the same number, because clients disagree: `max_model_len` is the vLLM/ninfer
convention, `meta.n_ctx` the llama.cpp one, `context_length` the OpenRouter/models.dev one. When the
context cannot be discovered the fields are **omitted entirely** — better absent than invented.

Two things it gets deliberately right:

- **The pool advertises its smallest member.** A request can land on any host, so if `ninfer-a` has
  65536 and `ninfer-b` has 32768, the model advertises 32768. Advertising the larger would invite
  prompts that fail whenever they land on the smaller host. Mismatches are logged once at startup.
- **Two engines publish a plausible wrong number.** llama.cpp is read from `/props`, not
  `/v1/models`; llama-server's `/v1/models` reports
  `meta.n_ctx_train` — the model's *architectural* context, unrelated to what it will accept. The
  served figure is `/props` → `default_generation_settings.n_ctx`, already divided by the slot count
  (`-c 65536 -np 4` gives each request 16384). LM Studio has the same shape: `max_context_length` is
  the model's ceiling, `loaded_context_length` is what was actually allocated. Both routers read the
  allocated figure; reading the obvious one would overstate usable context by 8-16x.

Override per backend with `context_length:` when a host lies or publishes nothing:

```yaml
  - {name: llama-1, url: "http://10.0.0.20:8080", kind: llamacpp,
     capacity: 4, models: [gpt-oss-120b], context_length: 16384}
```

### opencode needs telling separately

opencode pulls context limits from models.dev for known providers and from your own config for
custom ones — it does **not** read them from `/v1/models`. So set them explicitly, matching what
`curl http://127.0.0.1:8080/v1/models` reports:

```json
{
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "qwen3.6-27b": { "limit": { "context": 32768, "output": 8192 } }
      }
    }
  }
}
```

## Claude Code

All four backend engines implement the Anthropic Messages API natively, so this is a passthrough —
no translation layer, and none of the fidelity loss one would bring.

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_AUTH_TOKEN=local        # any non-empty value; the router ignores it
export ANTHROPIC_MODEL=qwen3.6-27b
export ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.6-27b   # background traffic, else it 404s
claude
```

Rather than setting the haiku variable, you can add a `"*"` entry to `model_aliases` and let the
router absorb any model name Claude Code asks for.

The router follows Anthropic's published gateway contract:

- **`anthropic-*` headers are forwarded as an open list**, not an allowlist. Capabilities arrive as
  new beta headers each release; a gateway pinned to today's names breaks the release that adds one.
- **The request body is never modified** except the model name, which a gateway is expected to
  rewrite. Capability betas pair a header with a body field, and breaking a pair is a hard `400`.
  The `system` array in particular passes through untouched and still first, so Claude Code's
  attribution block keeps being stripped positionally rather than polluting the prompt cache key.
- **Upstream error bodies are relayed byte-for-byte.** Claude Code recovers from capability
  rejections by matching on the upstream's own error wording, so wrapping errors in a router
  envelope would break that recovery path.
- **Streams are never buffered, and `ping` events are relayed.** Claude Code counts every byte and
  aborts a stream silent for 300s; during a long thinking pause pings are the only traffic.
- `/v1/messages/count_tokens` is served **without taking a capacity slot** — it is a cheap,
  non-generating call, and holding a generation slot for bookkeeping would let it block real work.

Two things worth knowing. Claude Code's system prompt and tool definitions are large, so give it a
model with **at least ~25k context**; check what the router advertises with `curl
localhost:8080/v1/models`. And its optional model discovery (`CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`)
only keeps model ids containing `claude` or `anthropic`, so `qwen3.6-27b` will not appear in the
`/model` picker unless you alias a claude-ish name to it.

### Affinity is exact for Claude Code

Claude Code sends `x-claude-code-session-id` on every request, so its conversations don't need to be
inferred at all — the router pins on that id directly. That is strictly better than hashing the
prompt: it survives **context compaction**, where the prefix legitimately changes but the session has
not, and it costs nothing to compute. When a session runs subagents, `x-claude-code-agent-id`
separates them so each holds its own pin rather than every subagent piling onto one host.

Prompt-prefix hashing remains the fallback for clients that send no such header.

### Affinity on the OpenAI surface

On the OpenAI surface the system prompt is `messages[0]`, shared by every conversation from a
client, so pinning ignores the first message boundary. Anthropic puts the system prompt in a
separate top-level field, which is folded into the root hash instead — so `messages[0]` is already
the first *user* message, unique to the conversation, and a Claude Code session can be pinned from
its very first turn rather than its second.

Cache-hit accounting is normalised across the two: Anthropic's `input_tokens` *excludes* cached
tokens while OpenAI's `prompt_tokens` includes them, so the surfaces reconcile that before it
reaches the dashboard and the `cache` column means the same thing either way.

## Configure your backends to match

Two settings matter as much as the router config:

**ninfer** — `capacity` must equal `--max-concurrency`, and keep the upstream queue shallow:

```
ninfer-serve --max-concurrency 4 --max-pending-requests 1
```

The router is the queue now. Leaving `--max-pending-requests` at its default of 16 re-creates the
exact blocking this is built to prevent.

**llama.cpp** — `capacity` must equal `-np` / `--parallel`. Leave `/slots` enabled (the default) and
the router cross-checks its own in-flight count against the server's real slot state, warning if they
drift (which means something else is sharing that host).

**vLLM — do not gate it low.** This is the one engine where a small `capacity` actively hurts. vLLM
schedules a continuous batch and queues internally *without* the head-of-line blocking that makes
gating necessary for ninfer, so it wants to be saturated. Set `capacity` to `--max-num-seqs` (default
256). Start it with `--enable-server-load-tracking` and the router will cross-check against `/load`.

**LM Studio** — `capacity` must match the parallel-request setting in its server UI (it serialises by
default, in which case use `1`). Newer builds want an auth token; set `api_key: "${LM_API_TOKEN}"`.
Note that LM Studio's context is whatever you allocated when *loading* the model, not the model's
maximum — load a 128k model with an 8k context and 8k is what you get.

## Reading the dashboard

| Column | Meaning |
|---|---|
| `load` | In-flight vs capacity. Amber means full — expected under load, not an error. |
| `pins` | Live sessions pinned to this backend. |
| `cache` | Mean prefix reuse (`cached_tokens ÷ prompt_tokens`). **The number that tells you affinity is working.** Low on first turns, should climb. |
| `err` | Errors. A red `(!n)` counts 429 `server_overloaded` — that means the backend rejected work the router believed it had room for, so its configured `capacity` is too high, or another client is sharing the host. |
| `spill in` | Requests that landed here because their pinned host was busy. |
| `ctx` | Discovered context window. A dim `?` means discovery failed — set `context_length`. |
| `!n` after the load bar | The backend reports `n` running but we dispatched fewer — something else is using that host, which breaks the capacity gate. Only llama.cpp and vLLM can report this. |

The summary panel shows queue depth, how many requests are holding out for a pinned host, and the
affinity honor rate — the share of pinned requests that actually got their host.

If `cache` sits near zero on a long agentic session, affinity isn't sticking: check whether the
client is rewriting earlier messages (context compaction legitimately breaks the prefix), and whether
`affinity_wait_ms` is long enough for your pool.

## How routing decides

For each request: derive the session key → look up the pinned backend → then

1. Pinned host has a free slot → **use it**.
2. Pinned host is busy → wait up to `affinity_wait_ms` for it. A short wait usually beats
   re-prefilling the whole conversation on a cold host.
3. Window expired (or no pin) → **least-loaded** healthy backend by fraction of capacity used, so a
   4-slot host takes proportionally more than a 2-slot one. Re-pin the session there.
4. Nothing free anywhere → stay queued until `queue_timeout_s`, then 503.

Queueing and outage are treated differently. A *busy* pool is normal backpressure, so requests wait
up to `queue_timeout_s` (default 5 minutes). A pool where nothing is **up** is not worth waiting for,
so those fail after `unavailable_grace_s` (default 10s — long enough to ride through a restart or a
probe cycle, short enough that a client isn't left hanging).

A request holding out for a busy pin is *skipped over*, not blocking: requests behind it that can be
placed are placed. That is what keeps one session's affinity wait from stalling the queue.

On connection errors and 429/502/503/504, the request fails over to a different backend — but only
before the first byte has reached the client, so a stream is never silently restarted.

## See it work without a GPU

```bash
uv run python scripts/demo.py
```

Starts three fake ninfer-style hosts (capacity 4, 2 and 4) that enforce their own concurrency limits,
drives concurrent multi-turn agent sessions through the router, and shows the dashboard. It prints a
report at the end:

```
Backend                requests   max concurrent / capacity   429s
  ninfer-a                66          4 / 4                0
  ninfer-b                36          2 / 2                0
  llama-1                 48          4 / 4                0

Affinity: 125 honored, 0 spilled (100% honored)
  ninfer-a             prefix reuse 68%
```

Every host saturated to exactly its capacity and never past it, no host had to reject work, and
sessions stayed put long enough to reuse ~two thirds of each prompt.

## Tests

```bash
uv run pytest -q
```

## Building a wheel

```bash
uv build          # -> dist/llm_router-0.1.0-py3-none-any.whl
```

CI (`.github/workflows/build.yml`) builds on **ubuntu-24.04 using the system Python 3.12** — no
`actions/setup-python`, and the job asserts the interpreter version so a runner-image change fails
loudly rather than silently shifting what the wheel was tested against.

The wheel is **`py3-none-any`**: pure Python, so one build serves every supported Python (>=3.11) and
every platform. There is nothing 3.12- or Linux-specific about the artifact — building on 24.04 just
pins where it is *verified*.

CI installs the built wheel into a clean venv and runs the suite against **that**, not the checkout,
so a module missing from the wheel fails the build instead of reaching whoever installs it. The
`dist/` contents are uploaded as the `llm-router-dist` artifact.

The suite runs against a fake upstream that enforces its own concurrency limit and records the
high-water mark of simultaneous requests, so oversubscription is caught rather than assumed. It also
models node-local prefix reuse, so affinity is measured the same way it is in production.
