# Docker secrets

Sensitive values (vLLM API key, Brave Search key) are supplied as **docker
compose secrets**, never via `.env` or a baked config file. Compose mounts each
file at `/run/secrets/<name>` inside the container; the gateway/RAG config layer
reads `/run/secrets/*` and applies them with the **highest** precedence (above
env and TOML).

## How to use

Each secret file is named exactly like the env override it provides
(case-insensitive), and contains the **raw value** with no quotes/newline:

```bash
# from the deploy/ directory
echo -n 'sk-your-vllm-key'   > secrets/gateway__model__api_key
echo -n 'BSA-your-brave-key' > secrets/gateway__web__brave_api_key
```

Then uncomment the matching entries in `compose.yaml`:

- the per-service `secrets:` list under `gateway:`
- the top-level `secrets:` block (the `file:` definitions)

and `docker compose up -d`.

## Naming

`gateway__model__api_key` → `model.api_key`; `gateway__web__brave_api_key` →
`web.brave_api_key`. The prefix (`GATEWAY__` / `RAG__`) and `__` delimiter match
the env-var scheme, so the same names work for both env and secrets.

## Safety

Everything in this directory **except this README and `.gitkeep` is gitignored**
— real secret material is never committed.
