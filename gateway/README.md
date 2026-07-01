# Gateway

The Gateway is a Python async daemon that mediates between clients (terminal CLI, desktop GUI) and a local or remote LLM. It owns projects, conversation history, tool execution, permission enforcement, and request scheduling. Clients connect over WebSocket and are thin — they render output, collect input, and forward approval decisions.

For complex requests the gateway runs a structured **discover → plan → execute** loop: it explores the codebase with read-only tools, creates an explicit plan, then executes each step — with automatic triage and self-correction when builds or tests fail.

## Requirements

- Python 3.13+
- [Poetry](https://python-poetry.org/docs/#installation)
- A running [vLLM](https://docs.vllm.ai/en/latest/) server (or any OpenAI-compatible API)

> **Model support:** the gateway is tailored to **Qwen3** models — its prompts and
> token counting assume the Qwen3 family. It is developed and tested against
> `Qwen3.5-35B-A3B-UD-Q4_K_XL`. Other OpenAI-compatible models may work but are not
> officially supported.

## Install

```bash
cd gateway
poetry install
```

This creates a virtualenv at `gateway/.venv` and installs all dependencies.

## Configure

Copy or edit `gateway.toml`. The defaults assume a local vLLM instance:

```toml
[server]
host = "0.0.0.0"
port = 8420
data_dir = "/var/lib/longmen/gateway"   # where projects, sessions, and permissions are stored

[model]
vllm_base_url = "http://localhost:8000"
model_name = "qwen3-32b"
api_key = ""          # leave empty for local vLLM with no auth
temperature = 0.7
max_tokens = 4096
tokenizer_path = "/path/to/model/tokenizer.json"   # required, local file

[permissions]
workflow_mode = "allow_all"   # only supported mode; "prompt" is under development
```

> **Workflow mode:** only `allow_all` is currently supported — tool calls
> auto-execute within the project sandbox without prompting. The interactive
> `prompt` mode (per-call approval with persistent "remember" rules) is still
> under development; setting it is not yet recommended.

`tokenizer_path` is **required**: it must point to a local `tokenizer.json` on disk. The gateway never downloads tokenizers; it loads `tokenizer.json` from disk for all token counting. Starting with a missing or empty `tokenizer_path` fails fast with a clear error.

The gateway watches `gateway.toml` for changes and hot-reloads most settings without restart. The exceptions are `host`, `port`, and `data_dir`, which require a restart.

> The gateway currently uses a single model and its `[model].tokenizer_path` for all token counting.

### Environment variable & secret overrides

Every scalar setting can be overridden by an environment variable that takes
precedence over `gateway.toml`. Use the `GATEWAY__` prefix and `__` to separate
nested keys:

```bash
GATEWAY__SERVER__PORT=8420
GATEWAY__MODEL__VLLM_BASE_URL=http://vllm:8000
GATEWAY__MODEL__TOKENIZER_PATH=/models/tokenizer.json
```

Secrets (vLLM API key, Brave key) can instead be supplied as docker-secret files
in `/run/secrets/`, named like the env var (`/run/secrets/GATEWAY__MODEL__API_KEY`);
these take precedence over env vars. Dynamic structures — named backends
(`[model.backends.*]`) — stay in the TOML file. **Precedence: TOML < env vars <
docker secrets.**

### Docker

`gateway/Dockerfile` builds the `longmen-gateway` image (minimal toolchain: git +
python). It bakes only generic operational defaults (`gateway.docker.toml`); your
model/vLLM URL, tokenizer, data dir, and secrets are supplied at run time. The
image answers `GET /health` (200 `ok`) for container healthchecks. See `deploy/`
at the repo root for a ready-to-run Compose stack (gateway + RAG + optional vLLM).

## Run

```bash
# From the gateway directory
poetry run gateway

# Or with an explicit config path
GATEWAY_CONFIG=/etc/assistant/gateway.toml poetry run gateway

# Or activate the venv first
source .venv/bin/activate
gateway
```

The server listens on `ws://0.0.0.0:8420/ws` by default and logs to stdout.

### Running in the background

```bash
poetry run gateway &> gateway.log &
```

Or as a systemd service — see the example unit file below.

<details>
<summary>systemd unit example</summary>

```ini
[Unit]
Description=Local LLM Gateway
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/opt/longmen/gateway
ExecStart=/opt/longmen/gateway/.venv/bin/gateway
Environment=GATEWAY_CONFIG=/opt/longmen/gateway/gateway.toml
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

</details>

## Connect an LLM (vLLM quickstart)

vLLM serves any HuggingFace model on an OpenAI-compatible API. The gateway talks to it over HTTP — it does not load the model itself.

```bash
# Install vLLM (GPU recommended; CPU works for small models)
pip install vllm

# Serve Qwen3-32B (requires ~64 GB VRAM; use a smaller model for testing)
vllm serve Qwen/Qwen3-32B --port 8000

# Serve a smaller model for development / CPU testing
vllm serve Qwen/Qwen3-1.7B --port 8000 --dtype float32
```

Once vLLM is running, start the gateway. No further wiring is needed — the gateway reads `vllm_base_url` from `gateway.toml` and opens HTTP connections on demand.

### Verify the connection

```bash
# Check vLLM is reachable
curl http://localhost:8000/v1/models

# Connect to the gateway with websocat (https://github.com/vi/websocat)
websocat ws://localhost:8420/ws

# The gateway should immediately send:
# {"type": "session_start", "payload": {"auth": "none", ...}}
```

## Develop

```bash
# Lint
poetry run ruff check src/

# Type-check
poetry run mypy src/
```

## WebSocket protocol (quick reference)

Full spec: [`docs/GATEWAY_PROTOCOL.md`](../docs/GATEWAY_PROTOCOL.md)

**Connection flow:**
1. Connect to `ws://host:port/ws`
2. Receive `session_start` — contains the default model and capabilities
3. Send `project_list` → receive `project_registry`
4. Send `project_select` → receive `project_context`
5. Send `prompt` → receive stream of `stream_chunk` / `plan_status` messages → `stream_end`

For complex prompts the gateway emits `plan_status` messages during discovery and between steps so clients can show progress. Simple prompts go straight to `stream_chunk` → `stream_end` with no plan status.

**Minimal session example (JSON frames in order):**

```
→ connect
← {"type":"session_start","payload":{"auth":"none","gateway_version":"1.0.0",...}}

→ {"type":"project_list","id":"1","timestamp":0,"payload":{}}
← {"type":"project_registry","payload":{"projects":{}}}

→ {"type":"project_upsert","id":"2","timestamp":0,"payload":{"project_id":"myapp","project":{"root_path":"/home/user/myapp","description":"My app"}}}
← {"type":"project_context","payload":{"project_id":"myapp",...}}

→ {"type":"project_select","id":"3","timestamp":0,"payload":{"project_id":"myapp"}}
← {"type":"project_context","payload":{"project_id":"myapp",...}}

→ {"type":"prompt","id":"4","timestamp":0,"payload":{"session_id":null,"content":[{"type":"text","text":"Hello"}]}}
← {"type":"stream_chunk","payload":{"delta":"Hello","role":"text",...}}
← {"type":"stream_end","payload":{"aborted":false,"usage":{...}}}
```

## Data directory

The gateway persists projects, permission rules, and conversation sessions under `data_dir` (default: `/var/lib/longmen/gateway`):

```
/var/lib/longmen/gateway/
└── projects/
    └── myapp/
        ├── project.toml       # root_path, description, context_file
        ├── permissions.toml   # stored "yes, always" approvals
        └── sessions/
            ├── <session_id>.jsonl       # append-only message log
            └── <session_id>.meta.json   # session metadata
```

You can edit these files directly — the gateway watches them with inotify and hot-reloads changes within 500ms, broadcasting a `config_reloaded` message to all connected clients.
