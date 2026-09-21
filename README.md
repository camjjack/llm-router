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
| `GET /dashboard`, `GET /sessions` | who is using it, and what is stuck — see [the web dashboard](#sessions-and-users-the-web-dashboard) |
| `GET /connect`, `GET /clients/<name>` | ready-made configs for opencode, Claude Code, Qwen Code and Zed — see [connecting coding agents](#connecting-coding-agents) |

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
uv run llm-router serve -c config.yaml           # logs to stderr
uv run llm-router serve -c config.yaml --tui     # with the terminal dashboard
uv run llm-router top --url http://127.0.0.1:8080   # terminal dashboard for a running router
```

However it is started, it serves two pages of its own, no flags needed:

| Page | |
|---|---|
| [`/connect`](#connecting-coding-agents) | ready-made configs for opencode, Claude Code, Qwen Code and Zed |
| [`/dashboard`](#sessions-and-users-the-web-dashboard) | who is using it, what each of their sessions is doing, and what is stuck |

Point a client at it:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}]}'
```

For a coding agent, open `http://127.0.0.1:8080/connect` and copy the config for it: the router
fills in its models, their context windows and timeouts that suit its queue. opencode needs no
copying at all — log in once, and it fetches that config from the router every time it starts:

```bash
opencode auth login http://127.0.0.1:8080
```

### As a systemd service

`llm-router.service.example` is a working unit. It expects a venv at `/opt/llm-router/venv` and a
config at `/etc/llm-router/config.yaml`; run these from a checkout to build that:

```bash
# the service account. It only ever runs the venv -- it installs nothing.
sudo useradd --system --no-create-home --shell /usr/sbin/nologin llm-router  # RHEL: /sbin/nologin

# build the venv as yourself, then hand it to root
sudo install -d -m 0755 -o "$USER" -g "$USER" /opt/llm-router
uv venv --python /usr/bin/python3 --python-preference only-system /opt/llm-router/venv
uv pip install --link-mode=copy --python /opt/llm-router/venv/bin/python .
sudo chown -R root:root /opt/llm-router

# config: readable by the service account, not by everyone
sudo install -d -m 0755 /etc/llm-router
sudo install -m 0640 -o root -g llm-router config.example.yaml /etc/llm-router/config.yaml
sudo install -m 0640 -o root -g llm-router /dev/null /etc/llm-router/env  # optional ${VAR} secrets
sudoedit /etc/llm-router/config.yaml
```

Two details that bite:

- **Build the venv on `/usr/bin/python3`.** Left to itself, `uv venv` may point it at a uv-managed
  interpreter inside your home directory. The unit sets `ProtectHome=yes`, so the service then fails
  to start with an error that never mentions the interpreter. Confirm with `readlink -f
  /opt/llm-router/venv/bin/python`; if the answer is under `/home` or `/root`, rebuild it.
- **`--link-mode=copy`.** uv hardlinks out of its cache in your home by default, which would leave
  the service's code writable by you after the chown.

Installing as yourself rather than as root also keeps your own package index configuration in play.
If you do install as root, note that root does not read your `~/.config/uv/uv.toml`, and that **uv
ignores `pip.conf` entirely** — so pass `--index-url`, or `sudo env
UV_CONFIG_FILE="$HOME/.config/uv/uv.toml" …`, or put the file at `/etc/uv/uv.toml` (mode `0600` if it
embeds a token).

Now check it exactly as systemd will run it. This one command proves the config is valid, that the
service account can read it, and that it can execute the venv — before systemd is involved at all:

```bash
sudo -u llm-router /opt/llm-router/venv/bin/llm-router check -c /etc/llm-router/config.yaml
```

Then start it:

```bash
sudo cp llm-router.service.example /etc/systemd/system/llm-router.service
sudo systemctl daemon-reload
sudo systemctl enable --now llm-router
journalctl -u llm-router -f
```

Logs reach the journal because the router writes to stderr whenever no `log_file` is set. So leave
`log_file` out of `config.yaml` and don't pass `--log-file` or `--tui` in the unit, or the journal
gets nothing. To watch a service that's already running, use `llm-router top --url
http://127.0.0.1:8080` rather than `--tui`.

`systemctl reload llm-router` sends SIGHUP, which re-reads the config without dropping a single
session — see below. The router picks up saved edits by itself anyway, so reload mainly matters if
you run with `--no-reload`, or the config sits on a network filesystem.

Upgrading is a restart, not a reload, since reload only re-reads the config. The venv belongs to root
by then, so take it back for the install:

```bash
sudo chown -R "$USER" /opt/llm-router
uv pip install --link-mode=copy --upgrade --python /opt/llm-router/venv/bin/python .
sudo chown -R root:root /opt/llm-router
sudo systemctl restart llm-router
```

## Changing the config while it runs

Edit `config.yaml` and save. The router picks the change up within a fraction of a second, with no
restart and without breaking the sessions using it:

- **Requests already running finish where they are**, including long streams, even on a backend
  you just removed. A removed backend shows as `draining` in the dashboard until its last request
  ends, and takes no new work in the meantime.
- **Session pins survive** for every backend whose name and URL you didn't change, so agentic loops
  stay on the host holding their KV. A backend you removed or pointed at a different URL loses its
  pins, and those sessions re-pin on their next turn.
- **A bad edit changes nothing.** The file is validated first; if it fails, the running config stays
  in force, the error is logged, and the dashboard shows it in red until the file is fixed.

### Pointing at a new model

Clients keep sending the model name they were started with, so renaming a model would make them
404. Alias the old name to the new model in the same edit:

```yaml
model_aliases:
  qwen3.6-27b: qwen3.7-32b     # clients still asking for the old name get the new one
backends:
  - name: ninfer-a
    url: http://10.0.0.11:8000
    capacity: 4
    models: [qwen3.7-32b]
```

The router warns in the log if a reload leaves a name that clients used recently resolving to nothing.
A request already queued for the old model when you save follows the new alias rather than failing.

### What reloads and what doesn't

Everything except `listen` and `log_file`, which are fixed for the life of the process. Changing them
logs a warning and takes effect on the next restart. A backend's context window is rediscovered when
its `models`, `upstream_model`, `kind` or `context_length` change.

### How changes are noticed

On Linux the router watches the config's directory with **inotify**. It reacts when a writer *closes*
the file, so it never reads a half-written one, and it handles editors that save by writing a
temporary file and renaming it over the original. Symlinked configs work too, including Kubernetes
ConfigMaps, which update by swapping a `..data` symlink. Elsewhere, and if inotify is unavailable,
it falls back to one `stat()` every 2 seconds.

To reload by hand, send SIGHUP. It re-applies the file even if it looks unchanged:

```bash
kill -HUP $(pgrep -f "llm-router serve")
systemctl reload llm-router     # with ExecReload=/bin/kill -HUP $MAINPID in the unit
```

`--no-reload` turns file watching off, leaving SIGHUP as the only way to reload. Use it if the config
sits on a network filesystem, where inotify doesn't see changes made on other machines. `llm-router
check -c config.yaml` runs the same validation a reload does, so you can test an edit first.

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
custom ones — it does **not** read them from `/v1/models`. The config the router generates for it
carries them, so this is taken care of; see [connecting coding agents](#connecting-coding-agents).

To tell clients a smaller window than the backends serve (a long session is quicker, and many
models do worse near the end of their window), set it per model:

```yaml
models:
  qwen3.6-27b:
    context_length: 32768     # never more than the backends serve: that is capped, with a warning
```

`/v1/models` and every generated client config then advertise 32768.

## Connecting coding agents

Open `http://<router>:8080/connect`. It has a ready-made config for **opencode**, **Claude Code**,
**Qwen Code** and **Zed**, with where it goes and how to install it, generated from the router's own
config. Each client is told the model ids, the context and output each model takes, the request
fields a model wants, and timeouts long enough for the router's queue. Type your name in and each
config also carries the header the [session dashboard](#sessions-and-users-the-web-dashboard) shows
you by.

The same configs are plain files, for scripts:

```bash
curl -s http://router:8080/clients/opencode          # also claude-code, qwen-code, zed
curl -s 'http://router:8080/clients/claude-code?user=alice&model=glm-air'
curl -s 'http://router:8080/clients/claude-code?format=shell'   # export lines instead of JSON
curl -s http://router:8080/clients                   # all of them, with install steps, as JSON
```

**opencode can keep itself up to date.** Log in to the router once:

```bash
opencode auth login http://router:8080          # or http://router:8080/u/alice, to be named
```

and opencode fetches its config from the router's `/.well-known/opencode` every time it starts,
layered underneath your own settings. Change a model in the router's config and every opencode
logged in to it follows. The other clients need their config fetched again.

### Describing the models

Everything is optional. Without it, clients still get every model with its discovered context.

```yaml
models:
  GLM-5.3-Flash-EXL3:
    name: GLM-5.3 Flash EXL3          # what clients display
    tool_call: true                   # default true
    reasoning: true                   # default false
    images: false                     # default false
    context_length: 200000            # tell clients less than the backends serve
    max_output_tokens: 32768
    reasoning_efforts: [low, high, max]   # levels to switch between, where a client can
    request_params:                   # extra request-body fields clients should send
      reasoning_effort: high          # the default effort
      chat_template_kwargs: {clear_thinking: true}

clients:
  provider_id: glm53                  # default llm-router
  provider_name: GLM-5.3 Flash EXL3 (Sparks)
  default_model: GLM-5.3-Flash-EXL3   # default: the first model in the config
  small_model: GLM-5.3-Flash-EXL3     # titles, summaries, commit messages; unset = client default
  # base_url: http://10.0.0.1:8888    # default: the address the config was fetched from
  # api_key: unused                   # the router ignores it, but clients insist on one
```

`base_url` needs setting only if people reach the router at a different address than the one the
config is fetched from. That happens when it's fetched as `localhost` on the router's own host, or
from behind a proxy.

### Where the numbers come from

- **Context** is what the backends report, capped by `context_length`. An unknown window is left
  out, with a warning on the connect page, rather than guessed.
- **How long to wait for a response to start** is `queue_timeout_s + first_byte_s`: the longest a
  request can sit in the router's queue, plus the longest it waits for its backend. With the
  defaults that's 900s, more than several clients wait by themselves: opencode gives up after
  300s, and Claude Code after 600s.
- **How long a response may go quiet** is `first_byte_s`. With vLLM and llama.cpp a response's
  headers arrive at once and prefill happens before its first chunk, so a long prefill counts
  against this. opencode's own limit is 300s, and Claude Code's stream watchdog fires after two
  minutes.

### What each client gets

These were checked by running each client against the router and looking at what arrived, which
turned up more than the documentation did:

- **opencode** ignores `reasoning_effort` in a model's `options`, silently. Effort is a *variant*,
  so each effort level becomes one, the configured effort becomes the default, and `--variant max`
  (or the variant key in the TUI) switches. opencode uses the lowest variant for titles. Other
  `request_params` go in `options` and are sent. It asks for at most 32,000 output tokens, whatever
  the limit says.
- **Claude Code** gets its `opus`, `sonnet` and `fable` aliases mapped to `default_model`, and
  `haiku`, its background model, to `small_model`. One further model fits in the `/model` picker.
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS` tells it when to compact. It can't send `request_params`, and it
  needs backends that speak the Anthropic Messages API.
- **Qwen Code** sends `request_params` with every request, as `extra_body`.
- **Zed** sends `reasoning_effort` but no other `request_params`. It keeps API keys out of
  `settings.json`, so it reads `<PROVIDER_ID>_API_KEY` (for example `GLM53_API_KEY`) from the
  environment.

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

## Open WebUI

Chats are routed **independently** — they are not lumped together. Pinning ignores the first message
boundary, which is the system prompt every chat in the instance shares, so identity starts at the
first user message. Two chats collide only if they open with byte-identical text, and in that case
they genuinely share a KV prefix, so co-locating them is the right answer; they separate as soon as
they diverge.

For exact per-chat routing, turn on Open WebUI's header forwarding:

```bash
ENABLE_FORWARD_USER_INFO_HEADERS=true
```

Open WebUI then sends `X-OpenWebUI-Chat-Id`, which the router pins on directly. That is better than
inference in one specific way: it survives a user **editing or regenerating** an earlier message,
where the prompt prefix legitimately changes but the conversation has not, so the chat stays on the
host that still holds most of its KV.

The same setting sends the user's name and email, which the
[web dashboard](#sessions-and-users-the-web-dashboard) uses to show each person's chats and whether
any of them are stuck.

Check it is working in the dashboard's `identity` row, or:

```bash
curl -s localhost:8080/stats | jq '.router | {keys_from_header, keys_from_prefix}'
```

### Why chat id and not user id

Open WebUI also forwards `X-OpenWebUI-User-Id`, and routing **deliberately ignores it** (the web
dashboard reads it, but only to label things). Pinning
per user would pile all of one person's chats onto a single host: worse for balance, and no better
for cache reuse than pinning each chat separately. Identity is not the unit of KV locality — a
conversation is. There is a test asserting the user headers never form a pin.

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

Add **`--enable-prompt-tokens-details`** too. Without it vLLM leaves cached-token counts out of its
usage entirely, on both the OpenAI and the Anthropic surface, so the dashboard's `cache` column shows
`--` however well its prefix cache is doing. vLLM's own view doesn't depend on the flag — its stats
log line carries `Prefix cache hit rate`, and `curl http://host:8000/metrics | grep prefix_cache` has
the counters — so check there if the column stays empty after adding it. Some vLLM versions have bugs
that keep the field null regardless.

**LM Studio** — `capacity` must match the parallel-request setting in its server UI (it serialises by
default, in which case use `1`). Newer builds want an auth token; set `api_key: "${LM_API_TOKEN}"`.
Note that LM Studio's context is whatever you allocated when *loading* the model, not the model's
maximum — load a 128k model with an 8k context and 8k is what you get.

## Reading the terminal dashboard

`--tui` and `llm-router top` show the pool itself: one row per backend, and a summary underneath.
For who is *using* it, see [the web dashboard](#sessions-and-users-the-web-dashboard).

| Column | Meaning |
|---|---|
| `load` | In-flight vs capacity. Amber means full — expected under load, not an error. |
| `pins` | Live sessions pinned to this backend. |
| `cache` | Mean prefix reuse (`cached_tokens ÷ prompt_tokens`). **The number that tells you affinity is working.** Low on first turns, should climb. A dim `--` means this backend reports no cache counts at all — not the same as reporting none (see vLLM below). |
| `err` | Errors. A red `(!n)` counts 429 `server_overloaded` — that means the backend rejected work the router believed it had room for, so its configured `capacity` is too high, or another client is sharing the host. |
| `spill in` | Requests that landed here because their pinned host was busy. |
| `ctx` | Discovered context window. A dim `?` means discovery failed — set `context_length`. |
| `!n` after the load bar | The backend reports `n` running but we dispatched fewer — something else is using that host, which breaks the capacity gate. Only llama.cpp and vLLM can report this. |

The summary panel shows queue depth, how many requests are holding out for a pinned host, and the
affinity honor rate — the share of pinned requests that actually got their host. Its `config` row
shows which config generation is running. It turns red when a reload has been rejected, meaning the
file on disk is not what's running. A backend marked `◌ … (draining)` was removed by a reload and
is finishing its last requests.

If `cache` sits near zero on a long agentic session, affinity isn't sticking: check whether the
client is rewriting earlier messages (context compaction legitimately breaks the prefix), and whether
`affinity_wait_ms` is long enough for your pool.

## Sessions and users: the web dashboard

Open `http://127.0.0.1:8080/dashboard` in a browser. It shows who is using the router, what each of
their sessions is doing right now, and anything that has stopped getting answers:

- **In flight**: every request the router is holding, in its current state, with how long the
  client has been waiting and how long it has been since anything came back.
- **Users**: each person, their clients, sessions, errors and tokens. Select one to see only their
  sessions.
- **Sessions**: the conversations themselves. Expand one to see its last few requests: which
  subagent sent each, where it ran, how long it queued, time to first token, and tokens used.
- **Recent requests**: the last 50 to finish, which can be filtered to errors only.

Each request also shows **what it asked the model for** — reasoning effort, thinking and its
budget, and the output limit — and **how long it held a backend**, with how much of that other
requests spent queued behind it. That is what finds the session everyone else is waiting on. Sort
the sessions by *Made others wait*, and see the same totals per person in the users table, next to
how long their own requests were queued.

`reasoning_effort`, Anthropic's `thinking` block, `output_config.effort` and the
`chat_template_kwargs` that vLLM and llama.cpp hand to the chat template are all read, so it does
not matter which spelling a client uses. A model's own default applies when a request says nothing,
and the dashboard shows that as `default`. Where a backend reports reasoning tokens separately,
those are counted too.

| State | Meaning |
|---|---|
| `queued` | Waiting for a free slot on any backend. |
| `holding` | Waiting for the backend its session is pinned to, to reuse the prompt cache there. It spills elsewhere once `affinity_wait_ms` runs out. |
| `processing` | Sent to a backend, nothing back yet: prefill, or the whole of a reply that isn't streamed. |
| `streaming` | Tokens are arriving. |
| `idle` (sessions only) | Nothing in flight. The session is waiting on its client, not on the router. |

**Made others wait** is time a request held a slot on a **full** backend while something was queued
for a model that backend serves. Every request holding one of its slots is charged that time, since
each of them is equally in the way. A busy backend with nobody queued costs nobody anything, so it
counts as zero.

**Slow and stuck are measured on silence:** the time since the last byte came back, or since the
request arrived if none has. A long reply that is still producing tokens is never stuck. A request
queued for a minute, a prefill that hasn't produced a first token, or a stream that stopped
mid-reply, is. Slow is 15s and stuck 60s by default, and stuck requests sort to the top.

### How users are identified

In this order:

1. **Open WebUI**, started with `ENABLE_FORWARD_USER_INFO_HEADERS=true`, which sends each user's
   email, id and name. With the same setting its chat id identifies each session exactly.
2. **An `X-LLM-Router-User` header**, for coding agents on people's own machines:

   ```bash
   # Claude Code
   export ANTHROPIC_CUSTOM_HEADERS="X-LLM-Router-User: alice"
   ```

   ```jsonc
   // opencode: in the provider's "options"
   "options": { "baseURL": "http://router:8080/v1", "headers": { "X-LLM-Router-User": "alice" } }
   ```

3. **Otherwise, by the address they connect from.** That already tells apart people on their own
   machines. Behind a reverse proxy, every client would show up as the proxy, unless its address is
   in `FORWARDED_ALLOW_IPS` (default `127.0.0.1`), in which case the router uses the
   `X-Forwarded-For` it sends.

**One person is one user, whichever client they are in front of.** Open WebUI names somebody by
their email, while their coding agent sends whatever they put in the header, so anything two
identities have in common — a name, an email, an email's local part, or an id — makes them the
same person here. Case and spacing don't matter. Set `X-LLM-Router-User` to your Open WebUI name,
or to the local part of your email, and your chats and your agent's sessions arrive as one user.
The link is made whenever it turns up, so an agent that has been running for an hour joins the
person who opens a chat later. The name shown is one a person is called by, in preference to an
email or an id.

Addresses are never linked to a name: they are reassigned and shared, and say nothing about who is
behind them.

A user name is whatever the client says it is. This is for seeing what is going on, not for access
control.

Sessions are Claude Code's own session id (its subagents are grouped under the session that
started them), Open WebUI's chat id, or an `x-session-id` header. Failing all of those, a session is
inferred from the start of the conversation, the same way affinity does it. That means a
conversation whose opening is edited, or compacted away, shows up as a new session. **Nothing from a
prompt is stored**: an inferred session is known by a hash.

### Settings

All optional, and all reloadable:

```yaml
tracking:
  enabled: true          # false stops recording and forgets what was recorded
  slow_after_s: 15       # silence before a request is flagged slow...
  stuck_after_s: 60      # ...and stuck
  retain_s: 3600         # how long an idle session stays listed
  max_sessions: 2000     # most sessions remembered; the longest idle go first
  # Headers naming the user, first match wins. Replace the list to use your own.
  # user_headers: [x-openwebui-user-email, x-openwebui-user-id, x-openwebui-user-name,
  #                x-llm-router-user, x-user]
```

`GET /sessions` is the same data as JSON, and `GET /sessions/<n>` one session in detail.

### What it costs the router

Next to nothing, by design, and measured:

- **Per request, about 3 µs** from arrival to finish, plus about 70 ns per streamed chunk. It
  writes a few fields as a request changes state. Nothing is hashed that affinity has not already
  hashed, and nothing is sorted, aggregated or serialised on the request path.
- **The queue's contention clock is event-driven**, not sampled: it advances when a request joins
  or leaves the queue, or a backend fills or frees up, and each advance is one pass over the
  backends. Nothing scans the queue, and nothing runs when the queue is empty.
- **Watching costs well under 1% of one core.** The page polls every 2 seconds, and only while its
  tab is visible. The JSON is built at most once a second however many people are watching (every
  viewer in between gets the same bytes). Even at its full 2,000 sessions it takes about 2 ms to
  build. Measured against an idle router, one open dashboard added 0.3% of a core, five added
  0.4%, and 25 polling ten times faster than the page does added 3.4%.
- **With traffic, router CPU per request stayed the same**, about 1.2 ms per streamed request,
  whether tracking was off, on, or on with 25 dashboards attached.
- **Memory is bounded** by `max_sessions`, and header values are truncated, so a client inventing
  a new session id on every request cannot make it grow.
- **Dashboard polling stays out of the access log**, so an open dashboard doesn't add a journal
  line every couple of seconds.

### Who can see it

The dashboard shows user names and emails to anyone who can reach the router's port, as `/stats`
and the API itself are already open to them. The router has no authentication. If that matters,
listen on a private interface, put `/dashboard` and `/sessions` behind a proxy that authenticates,
or set `tracking.enabled: false`. The page itself loads nothing from outside the router, so it
works on an isolated network. It renders every client-supplied value as text, under a strict
content security policy.

## How routing decides

For each request: derive the session key → look up the pinned backend → then

1. Pinned host has a free slot → **use it**.
2. Pinned host is busy → wait up to `affinity_wait_ms` for it. A short wait usually beats
   re-prefilling the whole conversation on a cold host.
3. Window expired (or no pin) → **least-loaded** healthy backend by fraction of capacity used, so a
   4-slot host takes proportionally more than a 2-slot one. Ties among equally idle backends rotate
   (least-recently-assigned wins) rather than resolving to a fixed order — chat traffic is often
   sequential, so *every* backend is idle when each new conversation starts, and a fixed tie-break
   would send them all to the same host and pin them there while the rest stayed cold. Re-pin the
   session there.
4. Nothing free anywhere → stay queued until `queue_timeout_s`, then 503.

Queueing and outage are treated differently. A *busy* pool is normal backpressure, so requests wait
up to `queue_timeout_s` (default 5 minutes). A pool where nothing is **up** is not worth waiting for,
so those fail after `unavailable_grace_s` (default 10s — long enough to ride through a restart or a
probe cycle, short enough that a client isn't left hanging).

A request holding out for a busy pin is *skipped over*, not blocking: requests behind it that can be
placed are placed. That is what keeps one session's affinity wait from stalling the queue.

On connection errors and 429/502/503/504, the request fails over to a different backend — but only
before the first byte has reached the client, so a stream is never silently restarted.

**A client that hangs up takes its request with it**, whatever stage it has reached. Queued, it
leaves the queue, and never takes a slot. Waiting on a backend, the upstream request is closed.
That's how a backend learns to stop generating, and the slot is freed for the next request.
Mid-stream, the same thing happens. This matters most for agents: pressing Esc in Claude Code, or
stopping a reply in Open WebUI, would otherwise leave the old request running for nobody, holding
a slot the next turn needs. These count as `abandoned` in `/stats`, and show as `cancelled` in the
web dashboard, with the stage the client left at.

This assumes a backend stops when its connection closes. vLLM does, and so do recent llama.cpp
builds. One that keeps generating anyway stays busier than the router thinks until it finishes.
The router already makes the same assumption when a client leaves mid-stream.

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
uv build          # -> dist/llm_router-0.3.0-py3-none-any.whl
```

CI (`.github/workflows/build.yml`) builds on **ubuntu-24.04 using the system Python 3.12** — no
`actions/setup-python`, and the job asserts the interpreter version so a runner-image change fails
loudly rather than silently shifting what the wheel was tested against.

The wheel is **`py3-none-any`**: pure Python, so one build serves every supported Python (>=3.11) and
every platform. There is nothing 3.12- or Linux-specific about the artifact — building on 24.04 just
pins where it is *verified*.

CI installs the built wheel into a clean venv and runs the suite against **that**, not the checkout,
so a module missing from the wheel fails the build instead of reaching whoever installs it.

### Getting the wheel

Every build uploads `dist/` as the `llm-router-dist` artifact, but **Actions artifacts require a
GitHub login to download** — even on a public repo, and even though the artifact's metadata is
publicly visible. The ZIP endpoint returns `401` to anonymous callers. That is GitHub's behaviour,
not something a workflow can change.

For a link anyone can `curl`, push a version tag. That runs the release job, which attaches the
wheel and sdist to a GitHub Release, and **release assets are anonymous-downloadable**:

```bash
git tag v0.3.0 && git push origin v0.3.0
```

```bash
pip install https://github.com/camjjack/llm-router/releases/download/v0.3.0/llm_router-0.3.0-py3-none-any.whl
```

The release step is idempotent: re-running a tag build repairs a partial release rather than failing
because it already exists.

The suite runs against a fake upstream that enforces its own concurrency limit and records the
high-water mark of simultaneous requests, so oversubscription is caught rather than assumed. It also
models node-local prefix reuse, so affinity is measured the same way it is in production.
