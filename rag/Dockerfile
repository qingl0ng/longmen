# syntax=docker/dockerfile:1
#
# longmen-rag — multi-stage, non-root, multi-arch (amd64 + arm64).
# Air-gapped at runtime (HF_HUB_OFFLINE=1): the user mounts the embedding model
# and tokenizer under /models and points RAG__EMBEDDING__MODEL / RAG__TOKENIZER__PATH
# at them. Python 3.12 (broadest torch CPU wheel availability on both arches).

# ---- build stage: venv with CPU torch + the application package ----
FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install CPU-only torch first so sentence-transformers does NOT pull the large
# CUDA build. (For GPU, build a CUDA variant separately and set device=cuda.)
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch

# Then the application package (regular install → ragsvc lands in site-packages).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# ---- runtime stage: clean image with just the venv ----
FROM python:3.12-slim AS runtime

# curl: container HEALTHCHECK against /status.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 longmen

WORKDIR /app
COPY --from=build /venv /venv
COPY rag.docker.toml ./rag.docker.toml

ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    RAG_CONFIG_PATH=/app/rag.docker.toml

# /data = persistent state volume (qdrant + manifest), /models = read-only mount.
RUN mkdir -p /data /models && chown -R longmen:longmen /data /app

USER longmen
EXPOSE 8421

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8421/status || exit 1

CMD ["ragsvc", "serve"]
