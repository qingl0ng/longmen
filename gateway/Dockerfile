# syntax=docker/dockerfile:1
#
# longmen-gateway — multi-stage, non-root, multi-arch (amd64 + arm64).
# Toolchain scope is intentionally minimal (git + python + POSIX utils); extend
# with `FROM longmen-gateway` to add language toolchains for your projects.

# ---- build stage: resolve + install deps into an in-project venv ----
FROM python:3.13-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

RUN pip install "poetry>=2.0"

WORKDIR /app

# Dependencies first (cached until pyproject/lock change).
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root

# Then the application package.
COPY src ./src
RUN poetry install --only main

# ---- runtime stage: clean image with just the venv + source ----
FROM python:3.13-slim AS runtime

# git: required by the git/build/test/shell tools. curl: container HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 longmen

WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY src ./src
COPY gateway.docker.toml ./gateway.docker.toml

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    GATEWAY_CONFIG=/app/gateway.docker.toml

# /data = persistent state volume, /models = read-only model/tokenizer mount.
# Created here so the non-root user owns them (named-volume init inherits this).
RUN mkdir -p /data /models && chown -R longmen:longmen /data /app

USER longmen
EXPOSE 8420

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8420/health || exit 1

CMD ["gateway"]
