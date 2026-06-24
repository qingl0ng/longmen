# ragsvc — RAG Service

A standalone FastAPI service that indexes reference materials — documentation, books, notes — into a local vector database (embedded Qdrant) and serves semantic search over them. Within the Longmen stack it runs alongside the gateway, which queries it through the `rag_search` tool to ground model responses in your own documents.

## Architecture

```
rag/
├── pyproject.toml
├── rag.toml.example
└── src/ragsvc/
    ├── main.py           # CLI entry point: serve / index / status subcommands
    ├── config.py         # Load and validate rag.toml (tomllib + Pydantic)
    ├── api.py            # FastAPI routes: /search, /collections, /status
    ├── collections.py    # Collection lifecycle and compatibility checks
    ├── indexer.py        # Pipeline orchestrator: extract → chunk → embed → store
    ├── watcher.py        # File watcher: debounced inotify + config hot-reload
    ├── chunker.py        # Token-aware chunking with section-boundary snapping
    ├── embedder.py       # sentence-transformers model wrapper
    ├── store.py          # Qdrant embedded-mode wrapper
    ├── manifest.py       # SQLite: tracks indexed files (hash, chunk count, model)
    ├── tokenizer.py      # Qwen3 tokenizer wrapper for token counting
    └── extractors/
        ├── base.py       # Abstract Extractor, Section, ExtractedDocument
        ├── markdown.py   # Heading-based splitting, fence-aware
        ├── plaintext.py  # Paragraph splits, RST heading detection
        └── pdf.py        # PyMuPDF extraction: ToC / font-size / page fallback
```

### Data flow

```
Directory on disk
      │
      ▼
  Extractor          (.md / .txt / .rst / .pdf)
      │  ExtractedDocument (list of Section objects with heading hierarchy)
      ▼
  Chunker            token-aware splitting with overlap, section-boundary snapping
      │  list[Chunk]  (text, token_count, heading_hierarchy, start_char, end_char)
      ▼
  Embedder           sentence-transformers, normalize_embeddings=True
      │  list[vector]
      ▼
  VectorStore        Qdrant embedded mode (Distance.DOT = cosine on unit vectors)
      │
  Manifest           SQLite — records file hash, chunk count, embedding model used
```

At query time:

```
POST /search  →  embed query  →  Qdrant search  →  scored StoredChunk list
```

### Key design decisions

- **Embedded Qdrant** — no separate Qdrant process; data lives in `data_dir/qdrant/`.
- **Deterministic chunk IDs** — `{document_id}:{chunk_index}` so re-indexing upserts rather than duplicating.
- **Incremental indexing** — SHA-256 hash comparison skips unchanged files.
- **Automatic indexing** — the file watcher monitors all collection directories with inotify/FSEvents. Drop a file in, and it is indexed automatically after a 3-second debounce. No manual `ragsvc index` needed during normal operation.
- **Startup scan** — on startup, the service compares the disk state against the manifest and queues any new, changed, or deleted files for background indexing. Searches are served immediately from the existing index while this runs.
- **Model mismatch detection** — the manifest records which embedding model indexed each document; searching a collection indexed with a different model returns an error rather than silently wrong results.
- **Overlap rules** — overlap is applied between sub-chunks within the same section AND across h3+ section boundaries (the previous section's tail seeds the first chunk of the next h3+ section). h1/h2 boundaries always start clean.
- **Air-gapped** — the **tokenizer** loads from a local `tokenizer.json` given by `[tokenizer].path` and is never downloaded. The **embedder** resolves from the local HuggingFace cache with `HF_HUB_OFFLINE=1` (set at startup). No network calls are made at runtime.

---

## Dependencies

### Runtime

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | HTTP API server |
| `sentence-transformers` | Embedding model loading and inference |
| `qdrant-client` | Embedded vector database |
| `pymupdf` (`fitz`) | PDF text extraction |
| `tokenizers` | Fast Qwen3 tokenizer for token counting |
| `huggingface-hub` | Local model cache access (network disabled at runtime) |
| `pydantic` | Config and API model validation |
| `structlog` | Structured logging |
| `httpx` | HTTP client (CLI → running service) |
| `watchfiles` | Async inotify/FSEvents wrapper for collection directory watching |

### Dev

| Package | Purpose |
|---|---|
| `ruff` | Linter / formatter |
| `mypy` | Static type checking |

### System

- Python 3.12+
- No external Qdrant server required
- GPU optional — set `device = "cuda"` or `"auto"` in config if available

---

## Setup

```bash
cd rag
poetry install
```

### Pre-download models (required — service is air-gapped at runtime)

On a machine with internet access, obtain the tokenizer file and populate the local
HuggingFace cache for the embedder:

```bash
# Tokenizer (Qwen3 — matches the gateway LLM)
poetry run huggingface-cli download Qwen/Qwen3-32B tokenizer.json tokenizer_config.json

# Embedding model (example — use whichever model you configure)
poetry run huggingface-cli download BAAI/bge-small-en-v1.5
```

**Tokenizer:** copy the downloaded `tokenizer.json` to a stable location (e.g.
`~/.longmen/rag/tokenizer.json`) and point `[tokenizer].path` at it — not at the deep,
hash-named cache path. The gateway and RAG can share the same `tokenizer.json`. The
tokenizer is loaded directly from this file via `from_file` and is never downloaded.

**Embedder:** the cache lives at `~/.cache/huggingface/`. Copy it to the target machine
if deploying air-gapped; the embedder resolves the model from there with `HF_HUB_OFFLINE=1`.

### Config

Place `rag.toml` in the directory you run `ragsvc` from (it is found automatically), or copy the example to the user-level location:

```bash
# Option A — project-local (recommended)
cp rag.toml.example rag.toml
$EDITOR rag.toml

# Option B — user-level
mkdir -p ~/.longmen/rag
cp rag.toml.example ~/.longmen/rag/rag.toml
$EDITOR ~/.longmen/rag/rag.toml
```

At minimum, define at least one collection:

```toml
[collections.my-docs]
path = "/absolute/path/to/your/docs"
description = "My reference documentation"
```

---

Config is resolved in this order (first match wins):

1. `--config /path/to/rag.toml` CLI flag
2. `RAG_CONFIG_PATH` environment variable
3. `./rag.toml` in the current working directory
4. `~/.longmen/rag/rag.toml`

If none are found, the searched paths are logged and startup fails (the offline
service refuses to run without a `[tokenizer].path`).

### Environment variable & secret overrides

Scalar settings can be overridden by environment variables, which take precedence
over the TOML file. Use the `RAG__` prefix and `__` to separate nested keys:

```bash
RAG__SERVICE__HOST=0.0.0.0
RAG__EMBEDDING__MODEL=/models/bge-small-en-v1.5   # a local model directory
RAG__EMBEDDING__DEVICE=cuda
RAG__TOKENIZER__PATH=/models/tokenizer.json
```

Secrets can be supplied as docker-secret files in `/run/secrets/`, named like the
env var; they take precedence over env vars. Dynamic `[collections.*]` tables
stay in the TOML file. **Precedence: TOML < env vars < docker secrets.**

### Docker

`rag/Dockerfile` builds the `longmen-rag` image (air-gapped at runtime,
`HF_HUB_OFFLINE=1`). It bakes only generic operational defaults
(`rag.docker.toml`); your tokenizer, embedding-model path, data dir, and
collections are supplied at run time. The image healthchecks `GET /status`. See
`deploy/` at the repo root for a ready-to-run Compose stack.

### Sections

```toml
[service]
host = "127.0.0.1"
port = 8421
log_level = "info"          # debug | info | warning | error

[embedding]
model = "BAAI/bge-small-en-v1.5"   # any sentence-transformers model
device = "cpu"                      # cpu | cuda | auto
batch_size = 64

[chunking]
default_chunk_size = 1024   # tokens
default_overlap = 128       # tokens

[search]
default_top_k = 10
min_score = 0.6             # minimum cosine similarity (0.0–1.0)

[tokenizer]
path = "~/.longmen/rag/tokenizer.json"   # required, local tokenizer.json (offline; never downloaded)

[storage]
data_dir = "~/.longmen/rag"   # Qdrant data + manifest.db

[watcher]
enabled = true              # set to false to disable automatic file watching
debounce_seconds = 3.0      # quiet window after last change before indexing starts
watch_config = true         # watch rag.toml for changes (hot-reload collections)

[collections.my-docs]
path = "/home/user/docs/reference"
description = "Reference documentation"
# chunk_size = 1536         # optional per-collection override
# overlap = 64              # optional per-collection override
```

**Validation rules:**
- All collection `path` values must be absolute
- `device` must be `cpu`, `cuda`, or `auto`
- `min_score` must be in `[0.0, 1.0]`
- `chunk_size` must be greater than `overlap`
- Collection names: alphanumeric and hyphens only (e.g. `godot-docs`, not `godot docs`)

---

## How To

### Start the server

```bash
ragsvc serve
```

The API is available at `http://127.0.0.1:8421`. On startup the service:

1. Scans all collection directories and queues any new, changed, or deleted files for background indexing
2. Starts watching those directories for further changes
3. Begins serving search requests immediately (from the existing index, while background indexing runs)

**You do not need to run `ragsvc index` manually.** The file watcher handles it automatically: drop files into a collection directory and they will be indexed within a few seconds (after the 3-second debounce). When the service restarts it picks up anything that changed while it was offline.

### File watcher behaviour

| Event | Action |
|---|---|
| File added or modified | Queued for indexing; indexing starts 3 s after the last change in that collection |
| Multiple files copied at once | Debounced — one indexing run fires after the copy completes, not once per file |
| File deleted | Chunks removed from the index immediately (no debounce) |
| `rag.toml` changed | New/removed collections are detected; chunk-size changes log a warning that reindexing is needed |

Watched extensions: `.md`, `.markdown`, `.mkd`, `.txt`, `.text`, `.rst`, `.pdf`. Hidden files, temp files (`.tmp`, `.swp`, `~`, `.partial`), and files over 50 MB are ignored.

### Manual indexing

Manual indexing is only needed when the file watcher is disabled (`watcher.enabled = false`) or when you want to force a full reindex (e.g. after changing `chunk_size`).

Trigger incremental indexing via the running server:

```bash
ragsvc index my-docs
ragsvc index --all          # all collections
```

Index without a running server (starts in-process, indexes, exits):

```bash
ragsvc index my-docs --oneshot
```

The `--oneshot` mode is useful for scripting or first-time setup. **Do not use `--oneshot` while the server is running** — both processes would try to open the same embedded Qdrant database and the second will fail with a lock error.

### Check service status

```bash
ragsvc status               # pretty-printed table
ragsvc status --json        # raw JSON
```

Output includes: embedding model, watcher state, collection stats (document count, chunk count), active indexing job, and any model-incompatible collections.

### Search (HTTP)

```bash
curl -s -X POST http://127.0.0.1:8421/search \
  -H "Content-Type: application/json" \
  -d '{"query": "OAuth2 dependency injection", "collections": ["my-docs"], "top_k": 5}' \
  | python3 -m json.tool
```

Response:

```json
{
  "query": "OAuth2 dependency injection",
  "results": [
    {
      "text": "...",
      "score": 0.847,
      "token_count": 312,
      "source": {
        "collection": "my-docs",
        "document": "security.md",
        "path": "/home/user/docs/reference/security.md",
        "size": "24.3 KB",
        "page": null,
        "section": "Chapter 8 > Authentication > OAuth2",
        "chunk_index": 3
      }
    }
  ],
  "total_results": 7,
  "model": "BAAI/bge-small-en-v1.5"
}
```

`size` is a human-readable file size string (`B`, `KB`, `MB`). It lets the caller decide whether to read the full file or rely on the chunk text alone.

`total_results` is the count of matches above `min_score` before `top_k` truncation.

### Reindex a collection (HTTP)

Force a full reindex (deletes existing data and reprocesses all files):

```bash
curl -X POST http://127.0.0.1:8421/collections/my-docs/reindex
```

Returns `202 Accepted` immediately; poll `GET /status` to track progress.

### PDF manual section map

If a PDF has no table of contents and the font-size heuristic doesn't produce useful headings, create a `sections.toml` file in the same directory as the PDF:

```toml
["my-book.pdf"]
"1-15" = "Introduction"
"16-42" = "Chapter 1 > Basics"
"43-89" = "Chapter 2 > Advanced Topics"
```

Page ranges are inclusive. `>` separates heading hierarchy levels.

---

## Supported file formats

| Extension | Extractor | Notes |
|---|---|---|
| `.md`, `.markdown`, `.mkd` | Markdown | Splits on headings, preserves code fences |
| `.txt`, `.text`, `.rst` | Plain text | Splits on double newlines, detects RST headings |
| `.pdf` | PDF (PyMuPDF) | ToC → font-size heuristic → page fallback; binary sanitizer |

Files are scanned recursively. Hidden files and directories (starting with `.`) and files over 50 MB are skipped.

