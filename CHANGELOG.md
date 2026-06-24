# Changelog

All notable changes to Longmen are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Longmen is a monorepo. The three artifacts share a `1.0.0` starting point but
version and publish **independently** under
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) — each has its own
release tag (`gateway-v*`, `rag-v*`) and may advance on its own. Cross-component
compatibility is tracked by the `GATEWAY_PROTOCOL.md` version, not a unified
product version.

| Component | Artifact | Version |
|---|---|---|
| Gateway | `longmen-gateway` image | 1.0.0 |
| RAG service | `longmen-rag` image | 1.0.0 |
| Terminal client | `longmen` (PyPI) | 1.0.0 |
| Protocol | `docs/GATEWAY_PROTOCOL.md` | 1.0.0 |

## [Unreleased]

## 1.0.0 — 2026-06-24

First public release. A self-hosted coding assistant that runs entirely on your
own hardware: a gateway daemon brokers between thin clients and a local vLLM
model server, with an optional RAG service for semantic search over your own
documents. No code, prompts, or documents leave your machine.

### Gateway (1.0.0)

- **Sessions & projects** — WebSocket transport with a `session_start` handshake,
  30s keepalive, and silent auto-reconnect. Sessions persist as append-only JSONL
  + atomic metadata and survive restarts (resume, history replay, incomplete-turn
  detection). Projects are sandboxed to a `root_path`; all tool execution is
  confined to the active project.
- **Prompting & streaming** — multimodal prompts (text, image, file, project file
  refs); token-by-token streaming of text, thinking, and tool-call deltas;
  stream-end metadata (finish reason, tool calls, token/budget usage); abort and
  escape-to-redirect.
- **Tools** — sandboxed registry: `shell`, `read_file`, `list_dir`, `grep`,
  `tree`, `symbols` (tree-sitter), `write_file`, `search_replace`, `delete_tool`,
  the `git_*` family, `detect_project`, `build`, `run_tests`, `run_app`,
  `sql_query`, plus conditionally registered `web_search`, `web_fetch`, and
  `rag_search`.
- **Approvals & permissions** — risk classification (safe/moderate/destructive),
  per-decision approval flow, persistent `yes_always` glob rules, diff previews
  for writes, and a non-spoofable workflow auto-approve mode.
- **Planning** — discover → plan → execute with a discovery token budget, a
  `create_plan` gate, self-correction (triage → fix → verify), and mid-execution
  `revise_plan`.
- **Context management** — budget tracking, pinning, auto-pruning, model-based
  compaction with fact extraction, mid-loop prune/compact, and manual `/compact`.
- **Backends & config** — vLLM SSE client, multiple named backends with per-backend
  sampling/timeouts, hot-reloadable `gateway.toml`, a FIFO request queue across
  clients, structured logging, and a typed error protocol.
- **Operations** — `GET /health` endpoint; layered configuration
  (TOML < env `GATEWAY__…` < docker secrets < init); multi-stage, non-root,
  multi-arch image.

### RAG service (1.0.0)

- Semantic search over named collections (`POST /search`), re-ranked by score with
  full source attribution.
- Extractors for Markdown, plaintext/RST, and PDF (PyMuPDF); Qwen3-tokenizer-aligned
  chunking with section-boundary snapping and overlap.
- sentence-transformers embeddings (CPU/CUDA/auto) into embedded Qdrant with
  deterministic chunk IDs (re-index upserts, no duplicates).
- Incremental indexing (SHA-256 skip), automatic file watching with debounce,
  startup reconciliation, manual `ragsvc index`, and `ragsvc status`.
- Model-mismatch detection, reindex endpoint with single-job back-pressure, and a
  strictly air-gapped model path (`HF_HUB_OFFLINE=1`).
- `GET /status` endpoint; layered configuration (TOML < env `RAG__…` < docker
  secrets); multi-stage, non-root image.

### Gateway ↔ RAG integration

- `rag_search` exposed as a gateway tool when `[rag] enabled`.
- Per-project collection binding, a context-budget guard, and result
  pruning/injection via interceptor.

### Terminal client — `longmen` (1.0.0)

- Thin WebSocket TUI: streamed output (text/thinking/tool calls), approval dialogs,
  queue-position display, and Markdown rendering.
- `@file` attachments with gateway-backed tab completion; multimodal input.
- Slash commands: `/new`, `/compact`, `/refresh`, `/permissions`, `/config`,
  `/project`, `/status`, `/prompt`, `/help`, `/quit` (fuzzy suggestions for unknown
  commands).
- Themes (default/light/minimal), auto-approve for safe tools, `--debug` message
  logging, and crash-log capture.

### Deployment

- Docker Compose stack (`deploy/`) wiring gateway + RAG with an optional,
  commented-out vLLM example service.
- Strict offline asset policy: the user supplies the tokenizer and embedding model
  via a read-only mount; nothing is baked or downloaded.
- Loopback-only gateway with `open` auth by default; documented path to remote
  access via `paired` auth.
