# Longmen

A self-hosted coding assistant that runs entirely on your own hardware. A
**gateway** daemon brokers between thin clients and a local
[vLLM](https://docs.vllm.ai/) or [Ollama](https://ollama.com/) model server, with an optional **RAG** service for
semantic search over your own documentation. No code, prompts, or documents ever
leave your machine.

> **Repo:** `longmen` · **Product/artifacts:** `longmen` (CLI) and the
> `longmen-gateway` / `longmen-rag` container images.

> **Model support:** Longmen is tailored to **Qwen3** models — its prompts, the
> agentic loop, and token counting assume the Qwen3 family. It is developed and
> tested against `Qwen3.5-35B-A3B-UD-Q4_K_XL`. Other OpenAI-compatible models may
> work but are not officially supported.

---

## What it is

- **Local-first & air-gapped.** The model runs in your own vLLM; tokenizers and
  embedding models are loaded from disk and never downloaded at runtime.
- **Agentic coding loop.** For complex requests the gateway runs a structured
  **discover → plan → execute** loop — it explores the codebase with read-only
  tools, writes an explicit plan, then executes each step with automatic triage
  and self-correction when builds or tests fail.
- **Tool sandboxing & approvals.** Every tool call is confined to a project's
  root and gated by an interactive approval flow (with "remember" rules).
- **Thin, swappable clients.** The gateway owns all state and logic; clients just
  render output and collect input. A terminal client ships today; a desktop
  client is planned.
- **Bring-your-own knowledge.** Point the RAG service at directories of Markdown,
  text, RST, or PDF and the assistant can search them semantically.

## Architecture

```
                         ┌──────────────────────────┐
   terminal client  ───▶ │   gateway (WebSocket)    │ ───▶  vLLM or Ollama  (OpenAI API)
   (longmen, PyPI)       │   projects · sessions    │
                         │   tools · permissions    │ ───▶  RAG   (/search)
                         │   discover→plan→execute  │
                         └──────────────────────────┘
```

| Component | Role | Artifact |
|---|---|---|
| [`gateway/`](gateway/README.md) | Async daemon: projects, sessions, tools, permissions, planning. Single source of truth. | `longmen-gateway` image |
| [`rag/`](rag/README.md) | FastAPI semantic-search service over your docs (embedded Qdrant). | `longmen-rag` image |
| [`clients/terminal/`](clients/terminal/README.md) | Thin terminal UI over WebSocket. | `longmen` on PyPI |

The contract between the gateway and every client is
[`docs/GATEWAY_PROTOCOL.md`](docs/GATEWAY_PROTOCOL.md) (WebSocket + JSON).

You also need an OpenAI-compatible model server running a **Qwen3** model from
[Hugging Face](https://huggingface.co/Qwen) — both the model weights and the
`tokenizer.json` come from there. [vLLM](https://docs.vllm.ai/) is the reference
backend; [Ollama](https://ollama.com/) also works (handy for local GGUF quants
such as the tested `…UD-Q4_K_XL`). The Compose stack can run vLLM for you or
point at an external endpoint.

---

## Setup (Docker Compose)

The [`deploy/`](deploy/) directory runs the gateway and RAG with Docker Compose.
The terminal client is installed separately and connects over WebSocket.

### Prerequisites

- Docker + Docker Compose.
- The `longmen-gateway` and `longmen-rag` images — pull published ones, or build
  locally:
  ```bash
  docker build -t longmen-gateway:latest gateway
  docker build -t longmen-rag:latest     rag
  ```
- A local **`tokenizer.json`** (model-specific, e.g. Qwen3) and a
  **sentence-transformers embedding model** directory. These are never downloaded
  by the services — you provide them. See
  [`rag/README.md`](rag/README.md#pre-download-models-required--service-is-air-gapped-at-runtime)
  for how to fetch them on a connected machine.

### 1. Configure and start

```bash
cd deploy
cp .env.example .env
$EDITOR .env            # model name, mount paths, embedding model dir, …
docker compose up -d
docker compose ps       # gateway + rag should report healthy
```

`.env` is the single place for non-secret configuration. It overrides the
generic defaults baked into the images. Precedence is:

```
image-baked TOML  <  .env (env vars)  <  docker secrets (./secrets/*)
```

Nothing user-specific or secret is baked into the images — model URL, tokenizer,
data dirs, and secrets are all supplied at run time. The minimum you must set in
`.env`:

```ini
GATEWAY__MODEL__MODEL_NAME=qwen3-32b
GATEWAY__MODEL__TOKENIZER_PATH=/models/tokenizer.json
GATEWAY__MODEL__VLLM_BASE_URL=http://vllm:8000     # your model server
RAG__TOKENIZER__PATH=/models/tokenizer.json
RAG__EMBEDDING__MODEL=/models/<embedding-model-dir>
```

> **Model server:** keep `http://vllm:8000` to use the bundled example `vllm`
> service (uncomment it in `compose.yaml`), or point at an external vLLM. See
> [`gateway/README.md`](gateway/README.md#connect-an-llm-vllm-quickstart) for a
> vLLM quickstart.

### 2. Provide the offline model assets

Both services mount `MODELS_DIR` (default `./models`) read-only at `/models`.
Put your assets there:

```
deploy/models/
├── tokenizer.json              # → GATEWAY__MODEL__TOKENIZER_PATH, RAG__TOKENIZER__PATH
└── bge-small-en-v1.5/          # → RAG__EMBEDDING__MODEL=/models/bge-small-en-v1.5
```

### 3. Gateway — workspace (project code) folder

The gateway runs tools against code on its own host, so your projects must be
mounted into the container. `WORKSPACE` (default `./workspace`) is bind-mounted
to `/workspace`, and every project's `root_path` lives under it:

```ini
# .env
WORKSPACE=/home/me/code
```

```
/home/me/code/            (host)        →     /workspace/        (container)
├── myapp/                              →     /workspace/myapp/
└── another-project/                    →     /workspace/another-project/
```

You **register** a project from the terminal client once it is connected
(see [step 6](#6-terminal-install--connect)):

```
/project add myapp /workspace/myapp "My application"
/project myapp
```

Tool sandboxing still confines each project to its own `root_path` subtree, even
though the whole workspace is mounted.

### 4. Gateway — config & data folder

The gateway's configuration comes from the layered model above (baked TOML +
`.env` + secrets) — for most deployments **you do not mount a config file**; you
just set `GATEWAY__…` variables in `.env`. Common ones:

```ini
#GATEWAY__MODEL__CONTEXT_LIMIT=120000
#GATEWAY__MODEL__MAX_TOKENS=32000
#GATEWAY_PORT=8420                  # host loopback port for the client
```

Every scalar gateway setting can be overridden with a `GATEWAY__SECTION__FIELD`
variable (`__` separates nested keys). The full key reference is in
[`gateway/README.md`](gateway/README.md#configure).

Persistent state — registered projects, conversation sessions, stored permission
rules — lives in the named volume **`gateway-data`** (mounted at `/data`), so it
survives `docker compose down`/`up`. Its layout:

```
/data/projects/<id>/
├── project.toml        # root_path, description, context_file
├── permissions.toml    # stored "always allow" rules
└── sessions/           # append-only conversation logs
```

> Secrets (vLLM API key, Brave search key) are **not** put in `.env`. Use docker
> secrets — see [`deploy/secrets/README.md`](deploy/secrets/README.md) and the
> commented `secrets:` blocks in `compose.yaml`.

### 5. RAG — documents folder & collections

RAG indexes documents from `DOCS` (default `./docs`), bind-mounted read-only to
`/docs`:

```ini
# .env
DOCS=/home/me/reference-docs
```

A **collection** is a named directory of documents. Collections are dynamic
structures that live in a `rag.toml` (not in `.env`), so you mount one and point
RAG at it. Create `deploy/rag.toml`:

```toml
[collections.python-docs]
path = "/docs/python"
description = "Python language & stdlib reference"

[collections.project-notes]
path = "/docs/notes"
description = "Internal design notes"
```

Then uncomment the two `rag.toml` lines in `compose.yaml`'s `rag` service:

```yaml
    volumes:
      # ...
      - ${RAG_CONFIG:-./rag.toml}:/config/rag.toml:ro
    environment:
      # ...
      RAG_CONFIG_PATH: /config/rag.toml
```

Collection `path` values point at directories **under `/docs`**. After
`docker compose up -d`, RAG scans each collection, indexes it in the background,
and then watches it — drop a new file in and it is indexed automatically within a
few seconds. Supported formats: `.md`, `.txt`, `.rst`, `.pdf`. Full details
(per-collection chunk sizes, PDF section maps, reindexing) are in
[`rag/README.md`](rag/README.md).

The gateway is wired to RAG out of the box (`GATEWAY__RAG__ENABLED=true`,
`GATEWAY__RAG__BASE_URL=http://rag:8421`), so the assistant can search your
collections as soon as they are indexed.

### 6. Terminal — install & connect

The client is a separate PyPI package:

```bash
pip install longmen
longmen                 # connects to ws://localhost:8420/ws by default
```

The default URL already matches the gateway's published loopback port, so no
configuration is needed for a local Compose deployment.

To point at a **different gateway** (e.g. a remote host), either pass a flag:

```bash
longmen --gateway ws://my-host:8420/ws
```

…or set it permanently in `~/.longmen/terminal/config.toml` (created on first
run):

```toml
[gateway]
url = "ws://my-host:8420/ws"
```

See [`clients/terminal/README.md`](clients/terminal/README.md) for slash
commands, file attachments (`@path`), and themes.

---

## Going further

- **GPU:** CPU by default. Uncomment the `deploy.resources` block on `rag` (and
  `vllm`) in `compose.yaml` and set `RAG__EMBEDDING__DEVICE=cuda`.
- **Remote access:** the gateway is bound to `127.0.0.1` with open auth (safe
  single-user default). Exposing it on the network is possible by changing the
  port mapping to `0.0.0.0`, but do so only on a trusted network — there is no
  client authentication yet.
- **Other languages in the gateway:** the image carries a minimal toolchain
  (git + Python). Extend it with `FROM longmen-gateway` to add more.
- **Building/publishing images & the CLI:** see
  [`RELEASE-PREP.md`](RELEASE-PREP.md).

## License

[MIT](LICENSE)
```
