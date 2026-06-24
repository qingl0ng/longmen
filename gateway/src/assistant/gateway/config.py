"""Configuration models for gateway.toml."""

from __future__ import annotations

import tomllib
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

# Parsed gateway.toml dict, threaded into the settings load by ``from_toml``.
# Lives in a ContextVar so the lowest-priority TOML source can read it without
# leaking state across calls (from_toml sets it, then resets it).
_toml_data: ContextVar[dict[str, Any] | None] = ContextVar("gateway_toml_data", default=None)

# Docker/compose secrets mount. Each file is named after the (case-insensitive)
# env var it overrides, e.g. /run/secrets/GATEWAY__MODEL__API_KEY.
_SECRETS_DIR = Path("/run/secrets")


class _TomlDataSource(PydanticBaseSettingsSource):
    """Lowest-priority settings source: the parsed ``gateway.toml`` dict.

    The dict is supplied by ``GatewayConfig.from_toml`` via ``_toml_data``. Env
    vars and docker secrets layer on top of it (see ``settings_customise_sources``).
    """

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        # Unused: we override __call__ to return the whole dict at once.
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(_toml_data.get() or {})


class _DockerSecretsSource(PydanticBaseSettingsSource):
    """Docker/compose secrets from ``/run/secrets``, layered above env vars.

    Each file is named like the env override it represents (case-insensitive),
    e.g. ``GATEWAY__MODEL__API_KEY``, and contains the raw value. File names are
    split on ``__`` into a nested dict, mirroring the env source so the exact
    same naming works for both. (pydantic's built-in ``SecretsSettingsSource``
    treats a nested-model field as a single JSON file instead.)
    """

    _PREFIX = "GATEWAY__"
    _DELIM = "__"

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        # Unused: we override __call__ to return the whole nested dict at once.
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        if not _SECRETS_DIR.is_dir():
            return {}
        result: dict[str, Any] = {}
        for entry in _SECRETS_DIR.iterdir():
            if not entry.is_file():
                continue
            name = entry.name.upper()
            if not name.startswith(self._PREFIX):
                continue
            parts = [p.lower() for p in name[len(self._PREFIX) :].split(self._DELIM) if p]
            if not parts:
                continue
            node = result
            for key in parts[:-1]:
                node = node.setdefault(key, {})
            node[parts[-1]] = entry.read_text().strip()
        return result


class WebConfig(BaseModel):
    """Web-related configuration (Brave Search, fetch settings)."""

    search_enabled: bool = True
    fetch_enabled: bool = True
    brave_api_key: str = ""
    search_count: int = 5
    fetch_timeout: int = 15
    fetch_max_redirects: int = 5
    fetch_blocked_domains: list[str] = Field(default_factory=list)
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )


class BackendConfig(BaseModel):
    vllm_base_url: str
    model_name: str
    api_key: str = ""
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    context_limit: int | None = None
    # Local tokenizer.json for this backend. Backend-specific (tokenizers are
    # model-specific) and never inherited from [model]. Carried for the future
    # per-backend token counting (task B); A only reads [model].tokenizer_path.
    tokenizer_path: str | None = None


class ModelConfig(BaseModel):
    vllm_base_url: str = "http://localhost:8000"
    model_name: str = "qwen3-32b"
    api_key: str = ""
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    context_limit: int = 32000
    # Required local tokenizer.json (validated in from_toml). Optional at the
    # model level so tests can construct ModelConfig directly; from_toml enforces it.
    tokenizer_path: str | None = None
    backends: dict[str, BackendConfig] = {}


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8420
    log_level: str = "info"
    data_dir: str = "/var/lib/longmen/gateway"


class AuthConfig(BaseModel):
    mode: str = "open"
    pairing_code_ttl: int = 300
    token_lifetime_days: int = 90


class PermissionsConfig(BaseModel):
    workflow_mode: str = "allow_all"
    default_safe: list[str] = [
        "read_file",
        "list_dir",
        "grep",
        "tree",
        "symbols",
        "git_status",
        "git_diff",
        "git_log",
        "web_search",
        "web_fetch",
        "rag_search",
    ]
    default_destructive: list[str] = [
        "rm",
        "git_push_force",
        "drop_table",
        "truncate",
        "delete_tool",
    ]


class TimeoutsConfig(BaseModel):
    tool_execution: int = 120
    approval_wait: int = 300
    vllm_request: int = 3600
    vllm_startup_wait: int = 120  # max seconds to wait for the model to come online
    vllm_first_token: int = 300  # max seconds to wait for the first streamed token


class ContextConfig(BaseModel):
    """Context management thresholds and budget sizes. All hot-reloadable."""

    pin_first_tokens: int = 2000
    pin_recent_tokens: int = 3000
    reserved_response_tokens: int = 4096
    compact_threshold: float = 0.85
    prune_threshold: float = 0.75
    warn_threshold: float = 0.95
    compact_target_tokens: int = 1000
    auto_prune: bool = True
    auto_compact: bool = True


class RAGConfig(BaseModel):
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8421"
    timeout: float = 30.0
    top_k: int = 10
    context_budget_threshold: int = 60000


class PlanningConfig(BaseModel):
    """Planning-related configuration."""

    max_revisions: int = Field(default=5, ge=1, le=20)
    """Maximum number of plan revisions allowed per execution."""


class SessionsConfig(BaseModel):
    max_sessions_per_project: int = 50
    max_session_age_days: int = 30


class GatewayConfig(BaseSettings):
    # Layered config: TOML base < env vars < docker secrets < init kwargs.
    # ``protected_namespaces=()`` silences the warning for the ``model`` field
    # (pydantic otherwise reserves the ``model_`` namespace).
    model_config = SettingsConfigDict(
        env_prefix="GATEWAY__",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    server: ServerConfig = ServerConfig()
    auth: AuthConfig = AuthConfig()
    model: ModelConfig = ModelConfig()
    permissions: PermissionsConfig = PermissionsConfig()
    timeouts: TimeoutsConfig = TimeoutsConfig()
    context: ContextConfig = ContextConfig()
    web: WebConfig = WebConfig()
    planning: PlanningConfig = PlanningConfig()
    rag: RAGConfig = RAGConfig()
    sessions: SessionsConfig = SessionsConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority (highest first): init kwargs > docker secrets > env > TOML.
        return (
            init_settings,
            _DockerSecretsSource(settings_cls),
            env_settings,
            _TomlDataSource(settings_cls),
        )

    @classmethod
    def from_toml(cls, path: str) -> GatewayConfig:
        """Load configuration: TOML file as the base, env vars and docker secrets on top."""
        with open(path, "rb") as f:
            data: dict[str, Any] = tomllib.load(f)
        token = _toml_data.set(data)
        try:
            config = cls()
        finally:
            _toml_data.reset(token)
        if not config.model.tokenizer_path:
            raise ValueError(
                "[model].tokenizer_path is required: set it to a local tokenizer.json "
                "(this is an offline gateway and will not download tokenizers)."
            )
        return config

    def resolve_backend(self, backend_name: str | None) -> BackendConfig:
        """Return named backend with model defaults filled in, or default backend."""
        model = self.model
        if backend_name is None or backend_name not in model.backends:
            return BackendConfig(
                vllm_base_url=model.vllm_base_url,
                model_name=model.model_name,
                api_key=model.api_key,
                temperature=model.temperature,
                top_p=model.top_p,
                max_tokens=model.max_tokens,
                context_limit=model.context_limit,
                tokenizer_path=model.tokenizer_path,
            )
        backend = model.backends[backend_name]
        return BackendConfig(
            vllm_base_url=backend.vllm_base_url,
            model_name=backend.model_name,
            api_key=backend.api_key if backend.api_key else model.api_key,
            temperature=(
                backend.temperature if backend.temperature is not None else model.temperature
            ),
            top_p=backend.top_p if backend.top_p is not None else model.top_p,
            max_tokens=(backend.max_tokens if backend.max_tokens is not None else model.max_tokens),
            context_limit=(
                backend.context_limit if backend.context_limit is not None else model.context_limit
            ),
            # Tokenizers are model-specific: use the backend's own path, never
            # inherited from [model]. Lets task B resolve a distinct tokenizer per backend.
            tokenizer_path=backend.tokenizer_path,
        )

    @property
    def data_dir_path(self) -> Path:
        return Path(self.server.data_dir).expanduser()
