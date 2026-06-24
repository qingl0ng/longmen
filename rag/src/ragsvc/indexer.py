"""Indexing pipeline: scan → extract → chunk → embed → store → manifest."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from .chunker import Chunk, Chunker
from .config import ChunkingConfig, CollectionConfig
from .embedder import Embedder
from .extractors import SUPPORTED_EXTENSIONS, get_extractor
from .manifest import Manifest, ManifestEntry, make_document_id
from .store import StoredChunk, VectorStore
from .tokenizer import TokenCounter

log = structlog.get_logger(__name__)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@dataclass
class IndexResult:
    collection: str
    documents_processed: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    documents_deleted: int = 0
    chunks_created: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def _hash_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _scan_directory(root: Path) -> list[Path]:
    """Recursively scan for supported files, skipping hidden and oversized files."""
    files: list[Path] = []
    for item in sorted(root.rglob("*")):
        # Skip hidden files and directories
        if any(part.startswith(".") for part in item.parts[len(root.parts):]):
            continue
        if not item.is_file():
            continue
        if item.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if item.stat().st_size > MAX_FILE_SIZE:
            log.warning("file_too_large_skipped", path=str(item), size=item.stat().st_size)
            continue
        files.append(item)
    return files


def _process_file(
    file_path: Path,
    collection_root: Path,
    collection_name: str,
    chunker: Chunker,
    embedder: Embedder,
    batch_size: int = 64,
) -> tuple[list[StoredChunk], list[list[float]]]:
    """Extract, chunk, and embed a single file. Returns (stored_chunks, vectors)."""
    extractor = get_extractor(file_path)
    document = extractor.extract(file_path)

    chunks: list[Chunk] = chunker.chunk(document)
    if not chunks:
        return [], []

    rel_path = str(file_path.relative_to(collection_root))
    document_id = make_document_id(collection_name, rel_path)

    texts = [c.text for c in chunks]
    vectors = embedder.embed(texts, batch_size=batch_size)

    stored: list[StoredChunk] = []
    for chunk, _vector in zip(chunks, vectors, strict=True):
        chunk_id = f"{document_id}:{chunk.chunk_index}"
        stored.append(
            StoredChunk(
                id=chunk_id,
                text=chunk.text,
                score=0.0,
                token_count=chunk.token_count,
                collection=collection_name,
                document_id=document_id,
                document_name=file_path.name,
                document_path=str(file_path),
                document_size=file_path.stat().st_size,
                chunk_index=chunk.chunk_index,
                heading_hierarchy=chunk.heading_hierarchy,
                page=chunk.page,
            )
        )

    return stored, vectors


class Indexer:
    """Orchestrates the full indexing pipeline."""

    def __init__(
        self,
        store: VectorStore,
        manifest: Manifest,
        embedder: Embedder,
        token_counter: TokenCounter,
        chunking_config: ChunkingConfig,
    ) -> None:
        self._store = store
        self._manifest = manifest
        self._embedder = embedder
        self._token_counter = token_counter
        self._chunking_config = chunking_config

    def _make_chunker(self, config: CollectionConfig) -> Chunker:
        chunk_size = config.chunk_size or self._chunking_config.default_chunk_size
        overlap = config.overlap or self._chunking_config.default_overlap
        return Chunker(self._token_counter, chunk_size=chunk_size, overlap=overlap)

    async def index_collection(
        self,
        name: str,
        config: CollectionConfig,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> IndexResult:
        """Index a single collection incrementally."""
        start_time = time.time()
        result = IndexResult(collection=name)

        root = config.resolved_path()
        if not root.exists():
            result.errors.append(f"Collection path does not exist: {root}")
            result.duration_seconds = time.time() - start_time
            return result
        if not root.is_dir():
            result.errors.append(f"Collection path is not a directory: {root}")
            result.duration_seconds = time.time() - start_time
            return result

        log.info("indexing_collection", collection=name, path=str(root))

        # Ensure Qdrant collection exists
        self._store.ensure_collection(name, self._embedder.dimension)

        # Scan directory
        files = await asyncio.to_thread(_scan_directory, root)
        total = len(files)
        log.info("files_found", collection=name, count=total)

        # Load existing manifest entries
        existing_entries = {e.file_path: e for e in self._manifest.list_collection(name)}
        current_rel_paths: set[str] = set()

        for idx, file_path in enumerate(files):
            rel_path = str(file_path.relative_to(root))
            current_rel_paths.add(rel_path)

            if progress_callback:
                progress_callback(f"Processing {file_path.name}", idx, total)

            try:
                file_hash = await asyncio.to_thread(_hash_file, file_path)

                # Check if unchanged
                existing = existing_entries.get(rel_path)
                if existing and existing.file_hash == file_hash:
                    result.documents_skipped += 1
                    continue

                # Process the file
                chunker = self._make_chunker(config)
                stored_chunks, vectors = await asyncio.to_thread(
                    _process_file,
                    file_path,
                    root,
                    name,
                    chunker,
                    self._embedder,
                )

                if not stored_chunks:
                    log.debug("file_produced_no_chunks", path=str(file_path))
                    result.documents_processed += 1
                    continue

                document_id = stored_chunks[0].document_id

                # Delete old chunks if re-indexing
                if existing:
                    self._store.delete_by_document(name, document_id)

                # Upsert new chunks
                self._store.upsert_chunks(name, stored_chunks, vectors)

                total_tokens = sum(c.token_count for c in stored_chunks)

                # Update manifest
                entry = ManifestEntry(
                    document_id=document_id,
                    collection=name,
                    file_path=rel_path,
                    file_hash=file_hash,
                    indexed_at=time.time(),
                    chunk_count=len(stored_chunks),
                    token_count=total_tokens,
                    embedding_model=self._embedder.model_name,
                )
                self._manifest.upsert(entry)

                result.documents_processed += 1
                result.chunks_created += len(stored_chunks)
                result.total_tokens += total_tokens

                log.debug(
                    "file_indexed",
                    collection=name,
                    path=rel_path,
                    chunks=len(stored_chunks),
                    tokens=total_tokens,
                )

            except Exception as exc:
                error_msg = f"Failed to index {rel_path}: {exc}"
                log.error("file_indexing_failed", collection=name, path=rel_path, error=str(exc))
                result.documents_failed += 1
                result.errors.append(error_msg)

        if progress_callback:
            progress_callback("Checking for deleted files", total, total)

        # Check for deleted files
        for rel_path, entry in existing_entries.items():
            if rel_path not in current_rel_paths:
                log.info("file_deleted_removing", collection=name, path=rel_path)
                self._store.delete_by_document(name, entry.document_id)
                self._manifest.delete(entry.document_id)
                result.documents_deleted += 1

        result.duration_seconds = time.time() - start_time
        log.info(
            "indexing_complete",
            collection=name,
            processed=result.documents_processed,
            skipped=result.documents_skipped,
            failed=result.documents_failed,
            deleted=result.documents_deleted,
            chunks=result.chunks_created,
            duration=f"{result.duration_seconds:.2f}s",
        )
        return result

    async def index_all(
        self,
        collections: dict[str, CollectionConfig],
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> list[IndexResult]:
        """Index all collections sequentially."""
        results: list[IndexResult] = []
        for name, config in collections.items():
            result = await self.index_collection(name, config, progress_callback)
            results.append(result)
        return results

    async def index_files(
        self,
        collection_name: str,
        config: CollectionConfig,
        file_paths: list[Path],
    ) -> IndexResult:
        """Index specific files in a collection."""
        start_time = time.time()
        result = IndexResult(collection=collection_name)
        root = config.resolved_path()

        self._store.ensure_collection(collection_name, self._embedder.dimension)
        chunker = self._make_chunker(config)

        for file_path in file_paths:
            rel_path = str(file_path.relative_to(root))
            try:
                file_hash = await asyncio.to_thread(_hash_file, file_path)
                stored_chunks, vectors = await asyncio.to_thread(
                    _process_file,
                    file_path,
                    root,
                    collection_name,
                    chunker,
                    self._embedder,
                )
                if not stored_chunks:
                    result.documents_processed += 1
                    continue

                document_id = stored_chunks[0].document_id
                self._store.delete_by_document(collection_name, document_id)
                self._store.upsert_chunks(collection_name, stored_chunks, vectors)
                total_tokens = sum(c.token_count for c in stored_chunks)
                entry = ManifestEntry(
                    document_id=document_id,
                    collection=collection_name,
                    file_path=rel_path,
                    file_hash=file_hash,
                    indexed_at=time.time(),
                    chunk_count=len(stored_chunks),
                    token_count=total_tokens,
                    embedding_model=self._embedder.model_name,
                )
                self._manifest.upsert(entry)
                result.documents_processed += 1
                result.chunks_created += len(stored_chunks)
                result.total_tokens += total_tokens
            except Exception as exc:
                log.error("file_indexing_failed", path=rel_path, error=str(exc))
                result.documents_failed += 1
                result.errors.append(f"Failed to index {rel_path}: {exc}")

        result.duration_seconds = time.time() - start_time
        return result


