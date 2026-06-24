# Deploying Longmen

This directory runs the **gateway** and **RAG** services with Docker Compose.
The terminal client (`longmen`) is installed separately from PyPI (`pip install
longmen`) and connects to the gateway over WebSocket.

## Prerequisites

- Docker + Docker Compose.
- The published images, or build them locally:
  ```bash
  docker build -t longmen-gateway:latest ../gateway
  docker build -t longmen-rag:latest     ../rag
  ```
- A local **tokenizer.json** and a **sentence-transformers embedding model**
  directory. These are model-specific and air-gapped — they are never downloaded
  by the services. Put them under your `MODELS_DIR`.

## Quick start

```bash
cp .env.example .env
$EDITOR .env            # set MODEL_NAME, mount paths, embedding model dir, …

# place your assets, e.g.:
#   ./models/tokenizer.json
#   ./models/bge-small-en-v1.5/   (the embedding model directory)

docker compose up -d
docker compose ps       # gateway + rag should be healthy
```

Then connect with the terminal client:

```bash
pip install longmen
longmen                 # defaults to ws://localhost:8420/ws
```

## Layout / behavior

- **Config precedence:** image-baked TOML < env vars (`.env`) < docker secrets
  (`./secrets/*`). Nothing user-specific or secret is baked into the images.
- **Ports:** the gateway is published on `127.0.0.1:8420` (loopback) only. RAG is
  not published — it is reachable only by the gateway on the internal network.
- **Volumes:** `gateway-data` and `rag-data` (named) hold persistent state;
  `/models` is a read-only mount of your tokenizer/embedding model; `/workspace`
  (gateway) is your code; `/docs` (RAG) is documents to index.
- **vLLM:** bring your own (set `GATEWAY__MODEL__VLLM_BASE_URL`) or uncomment the
  example `vllm` service in `compose.yaml`.
- **GPU / collections / secrets / remote access:** see the inline comments in
  `compose.yaml`, `.env.example`, and `secrets/README.md`.
