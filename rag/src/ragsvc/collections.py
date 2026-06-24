"""Collection management: lifecycle, status, model compatibility."""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from .config import RAGConfig
from .embedder import Embedder
from .manifest import Manifest
from .store import VectorStore

log = structlog.get_logger(__name__)

_VALID_COLLECTION_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9\-]*$")


@dataclass
class CollectionInfo:
    name: str
    description: str
    path: str
    document_count: int
    chunk_count: int
    total_tokens: int
    chunk_size: int
    overlap: int
    indexed_with_model: str | None
    compatible: bool


class CollectionManager:
    """Manages collection lifecycle and status."""

    def __init__(
        self,
        config: RAGConfig,
        store: VectorStore,
        manifest: Manifest,
        embedder: Embedder,
    ) -> None:
        self._config = config
        self._store = store
        self._manifest = manifest
        self._embedder = embedder

    def list_collections(self) -> list[CollectionInfo]:
        """List all collections defined in config with their status."""
        return [
            self._build_info(name)
            for name in self._config.collections
        ]

    def get_collection(self, name: str) -> CollectionInfo | None:
        """Get detailed info for a single collection."""
        if name not in self._config.collections:
            return None
        return self._build_info(name)

    def _build_info(self, name: str) -> CollectionInfo:
        col_config = self._config.collections[name]
        entries = self._manifest.list_collection(name)
        document_count = len(entries)
        total_tokens = sum(e.token_count for e in entries)

        stats = self._store.collection_stats(name)
        chunk_count = stats.get("chunk_count", 0)

        # Determine indexed model
        models_used = {e.embedding_model for e in entries}
        indexed_with_model: str | None = None
        if models_used:
            if len(models_used) > 1:
                log.warning(
                    "collection_has_mixed_embedding_models",
                    collection=name,
                    models=sorted(models_used),
                )
            indexed_with_model = next(iter(models_used))

        compatible = (
            indexed_with_model is None
            or indexed_with_model == self._embedder.model_name
        )

        return CollectionInfo(
            name=name,
            description=col_config.description,
            path=str(col_config.resolved_path()),
            document_count=document_count,
            chunk_count=chunk_count,
            total_tokens=total_tokens,
            chunk_size=self._config.get_effective_chunk_size(name),
            overlap=self._config.get_effective_overlap(name),
            indexed_with_model=indexed_with_model,
            compatible=compatible,
        )

    def validate_collection(self, name: str) -> list[str]:
        """Validate a collection config. Returns list of issues."""
        issues: list[str] = []
        if name not in self._config.collections:
            issues.append(f"Collection {name!r} is not defined in config")
            return issues

        if not _VALID_COLLECTION_NAME.match(name):
            issues.append(
                f"Collection name {name!r} must be alphanumeric with hyphens, no spaces"
            )

        col_config = self._config.collections[name]
        path = col_config.resolved_path()
        if not path.exists():
            issues.append(f"Collection path does not exist: {path}")
        elif not path.is_dir():
            issues.append(f"Collection path is not a directory: {path}")

        return issues

    def get_effective_chunk_size(self, name: str) -> int:
        return self._config.get_effective_chunk_size(name)

    def get_effective_overlap(self, name: str) -> int:
        return self._config.get_effective_overlap(name)

    def delete_collection_data(self, name: str) -> None:
        """Delete all indexed data for a collection (Qdrant + manifest)."""
        self._store.delete_collection(name)
        self._manifest.delete_collection(name)
        log.info("collection_data_deleted", name=name)

    def get_incompatible_collections(self) -> list[str]:
        """Return names of collections indexed with a different model."""
        current_model = self._embedder.model_name
        incompatible: list[str] = []
        for name in self._config.collections:
            entries = self._manifest.list_collection(name)
            for entry in entries:
                if entry.embedding_model != current_model:
                    incompatible.append(name)
                    break
        return incompatible
