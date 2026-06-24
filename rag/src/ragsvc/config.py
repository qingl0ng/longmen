"""Load and validate rag.toml configuration."""

from __future__ import annotations

import os
import re
import tomllib
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

log = structlog.get_logger(__name__)

_CWD_CONFIG_PATH = Path("rag.toml")
_USER_CONFIG_PATH = Path("~/.longmen/rag/rag.toml")

# Parsed rag.toml dict, threaded into the settings load by ``load_config``.
# Lives in a ContextVar so the lowest-priority TOML source can read it without
# leaking state across calls (load_config sets it, then resets it).
_toml_data: ContextVar[dict[str, Any] | None] = ContextVar("rag_toml_data", default=None)

# Docker/compose secrets mount. Each file is named after the (case-insensitive)
# env var it overrides, e.g. /run/secrets/RAG__EMBEDDING__MODEL.
_SECRETS_DIR = Path("/run/secrets")


class _TomlDataSource(PydanticBaseSettingsSource):
    """Lowest-priority settings source: the parsed ``rag.toml`` dict.

    The dict is supplied by ``load_config`` via ``_toml_data``. Env vars and
    docker secrets layer on top of it (see ``settings_customise_sources``).
    """

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        # Unused: we override __call__ to return the whole dict at once.
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(_toml_data.get() or {})


class _DockerSecretsSource(PydanticBaseSettingsSource):
    """Docker/compose secrets from ``/run/secrets``, layered above env vars.

    Each file is named like the env override it represents (case-insensitive),
    e.g. ``RAG__EMBEDDING__MODEL``, and contains the raw value. File names are
    split on ``__`` into a nested dict, mirroring the env source so the exact
    same naming works for both. (pydantic's built-in ``SecretsSettingsSource``
    treats a nested-model field as a single JSON file instead.)
    """

    _PREFIX = "RAG__"
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


class ServiceConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8421
    log_level: str = "info"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"debug", "info", "warning", "error"}
        if v not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v!r}")
        return v


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    batch_size: int = 64

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        valid = {"cpu", "cuda", "auto"}
        if v not in valid:
            raise ValueError(f"device must be one of {valid}, got {v!r}")
        return v


class ChunkingConfig(BaseModel):
    default_chunk_size: int = 1024
    default_overlap: int = 128

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> ChunkingConfig:
        if self.default_chunk_size <= self.default_overlap:
            raise ValueError(
                f"default_chunk_size ({self.default_chunk_size}) must be greater than "
                f"default_overlap ({self.default_overlap})"
            )
        return self


class SearchConfig(BaseModel):
    default_top_k: int = 10
    min_score: float = 0.6

    @field_validator("min_score")
    @classmethod
    def validate_min_score(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"min_score must be between 0.0 and 1.0, got {v}")
        return v


class TokenizerConfig(BaseModel):
    path: str

    @field_validator("path")
    @classmethod
    def validate_path_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "[tokenizer].path is required: set it to a local tokenizer.json "
                "(this is an offline service and will not download tokenizers)."
            )
        return v

    def resolved_path(self) -> Path:
        return Path(self.path).expanduser()


class StorageConfig(BaseModel):
    data_dir: str = "~/.longmen/rag"

    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir).expanduser()


class WatcherConfig(BaseModel):
    enabled: bool = True
    debounce_seconds: float = 3.0
    watch_config: bool = True


class CollectionConfig(BaseModel):
    path: str
    description: str = ""
    chunk_size: int | None = None
    overlap: int | None = None

    def resolved_path(self) -> Path:
        return Path(self.path).expanduser()

    @field_validator("path")
    @classmethod
    def validate_path_is_absolute(cls, v: str) -> str:
        expanded = Path(v).expanduser()
        if not expanded.is_absolute():
            raise ValueError(f"Collection path must be absolute after ~ expansion, got {v!r}")
        return v


_VALID_COLLECTION_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9\-]*$")


class RAGConfig(BaseSettings):
    # Layered config: TOML base < env vars < docker secrets < init kwargs.
    model_config = SettingsConfigDict(
        env_prefix="RAG__",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    service: ServiceConfig = ServiceConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    search: SearchConfig = SearchConfig()
    tokenizer: TokenizerConfig
    storage: StorageConfig = StorageConfig()
    watcher: WatcherConfig = WatcherConfig()
    collections: dict[str, CollectionConfig] = {}

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

    @model_validator(mode="after")
    def validate_collections(self) -> RAGConfig:
        for name, col in self.collections.items():
            if not _VALID_COLLECTION_NAME.match(name):
                raise ValueError(
                    f"Collection name {name!r} must be alphanumeric with hyphens, no spaces"
                )
            # Validate chunk_size > overlap within collection
            effective_chunk = col.chunk_size or self.chunking.default_chunk_size
            effective_overlap = col.overlap or self.chunking.default_overlap
            if effective_chunk <= effective_overlap:
                raise ValueError(
                    f"Collection {name!r}: chunk_size ({effective_chunk}) "
                    f"must be > overlap ({effective_overlap})"
                )
        return self

    def get_effective_chunk_size(self, name: str) -> int:
        col = self.collections.get(name)
        if col and col.chunk_size is not None:
            return col.chunk_size
        return self.chunking.default_chunk_size

    def get_effective_overlap(self, name: str) -> int:
        col = self.collections.get(name)
        if col and col.overlap is not None:
            return col.overlap
        return self.chunking.default_overlap


def _resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve the config file path.

    Search order:
    1. Explicit --config argument
    2. RAG_CONFIG_PATH environment variable
    3. ./rag.toml  (current working directory)
    4. ~/.longmen/rag/rag.toml  (user-level fallback)
    """
    if config_path is not None:
        return Path(config_path).expanduser()
    env_path = os.environ.get("RAG_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser()
    if _CWD_CONFIG_PATH.exists():
        return _CWD_CONFIG_PATH.resolve()
    return _USER_CONFIG_PATH.expanduser()


def load_config(config_path: str | Path | None = None) -> RAGConfig:
    """Load rag.toml as the base, with env vars and docker secrets layered on top.

    Raises if no config file is found. Env vars (``RAG__SECTION__FIELD``) and
    docker secrets (``/run/secrets/RAG__...``) override individual TOML scalars;
    dynamic ``[collections.*]`` tables come from the TOML file only.
    """
    path = _resolve_config_path(config_path)
    if not path.exists():
        searched = [str(_CWD_CONFIG_PATH.resolve()), str(_USER_CONFIG_PATH.expanduser())]
        log.error("config_not_found", searched=searched)
        raise RuntimeError(
            f"no rag.toml found at {searched}; [tokenizer].path is required "
            "— this offline service will not download a tokenizer."
        )

    log.info("loading_config", path=str(path))
    with open(path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    # Feed the parsed TOML (including [collections.*]) as the lowest-priority
    # source; env vars and docker secrets layer over it during construction.
    token = _toml_data.set(raw)
    try:
        # tokenizer (required) is supplied by the TOML/env/secret sources at
        # runtime; mypy's pydantic plugin can't see that and flags it as missing.
        return RAGConfig()  # type: ignore[call-arg]
    finally:
        _toml_data.reset(token)
