"""FastAPI routes for search, collection management, and service status."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from .collections import CollectionInfo, CollectionManager
from .config import RAGConfig
from .embedder import Embedder
from .indexer import Indexer, IndexResult
from .store import VectorStore

log = structlog.get_logger(__name__)


# ─── Request / Response models ────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    collections: list[str] | None = None
    top_k: int | None = None


def _fmt_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string (e.g. '15.2 KB', '1.4 MB')."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024 ** 2:.1f} MB"


class SearchSource(BaseModel):
    collection: str
    document: str
    path: str
    size: str
    page: int | None
    section: str | None
    chunk_index: int


class SearchResult(BaseModel):
    text: str
    score: float
    token_count: int
    source: SearchSource


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_results: int
    model: str


class ReindexResponse(BaseModel):
    status: str
    collection: str


class IndexingProgress(BaseModel):
    active: bool
    current_collection: str | None
    progress: tuple[int, int] | None


class WatcherInfo(BaseModel):
    active: bool
    watched_paths: int
    pending_debounce: list[str]


class StatusResponse(BaseModel):
    status: str
    embedding_model: str
    embedding_dimension: int
    collections: dict[str, dict[str, Any]]
    indexing: IndexingProgress
    watcher: WatcherInfo | None = None
    incompatible_collections: list[str]


# ─── Indexing state ───────────────────────────────────────────────────────────

class IndexingState:
    """Tracks background indexing job. Only one at a time."""

    def __init__(self) -> None:
        self.active: bool = False
        self.current_collection: str | None = None
        self.progress: tuple[int, int] | None = None
        self.result: IndexResult | None = None
        self.lock: asyncio.Lock = asyncio.Lock()

    def _update_progress(self, message: str, current: int, total: int) -> None:
        self.progress = (current, total)

    async def wait_complete(self) -> None:
        """Wait until no indexing is active. Used by shutdown."""
        while self.active:
            await asyncio.sleep(0.5)

    async def run_indexing_files(
        self,
        collection_name: str,
        collection_config: Any,
        file_paths: list[Path],
        indexer: Indexer,
    ) -> None:
        """Run index_files for specific files. Uses the same lock as run_indexing."""
        async with self.lock:
            self.active = True
            self.current_collection = collection_name
            self.progress = None
            try:
                self.result = await indexer.index_files(
                    collection_name, collection_config, file_paths
                )
            except Exception as exc:
                log.error("watcher_indexing_failed", collection=collection_name, error=str(exc))
            finally:
                self.active = False
                self.current_collection = None
                self.progress = None

    async def run_indexing(
        self,
        collection_name: str,
        collection_config: Any,
        indexer: Indexer,
        delete_first: bool = False,
        collection_manager: CollectionManager | None = None,
    ) -> None:
        """Run indexing as a background task. Only one at a time."""
        async with self.lock:
            self.active = True
            self.current_collection = collection_name
            self.progress = None
            try:
                if delete_first and collection_manager:
                    collection_manager.delete_collection_data(collection_name)
                self.result = await indexer.index_collection(
                    collection_name,
                    collection_config,
                    progress_callback=self._update_progress,
                )
            except Exception as exc:
                log.error("background_indexing_failed", collection=collection_name, error=str(exc))
            finally:
                self.active = False
                self.current_collection = None
                self.progress = None


# ─── Application factory ──────────────────────────────────────────────────────

def create_app(
    config: RAGConfig,
    embedder: Embedder,
    store: VectorStore,
    indexer: Indexer,
    collection_manager: CollectionManager,
    indexing_state: IndexingState,
    collection_watcher: Any = None,  # CollectionWatcher | None — Any avoids circular import
) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="RAG Service", version="1.0.0")

    @app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest) -> SearchResponse:
        # Determine which collections to search
        target_collections = request.collections or list(config.collections.keys())

        # Check for model mismatches
        for coll_name in target_collections:
            if coll_name not in config.collections:
                raise HTTPException(
                    status_code=404,
                    detail={"error": "collection_not_found", "collection": coll_name},
                )
            info = collection_manager.get_collection(coll_name)
            if info and not info.compatible and info.indexed_with_model:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "model_mismatch",
                        "message": (
                            f"Collection {coll_name!r} was indexed with {info.indexed_with_model} "
                            f"but current model is {embedder.model_name}. "
                            f"Run 'ragsvc index {coll_name}' to reindex."
                        ),
                        "collection": coll_name,
                    },
                )

        top_k = request.top_k or config.search.default_top_k
        min_score = config.search.min_score

        # Embed query
        query_vectors = await asyncio.to_thread(embedder.embed, [request.query])
        query_vector = query_vectors[0]

        # Search (request top_k * 3 to count total before truncation)
        fetch_k = max(top_k * 3, 50)
        raw_results = await asyncio.to_thread(
            store.search_multi,
            target_collections,
            query_vector,
            fetch_k,
            min_score,
        )

        total_results = len(raw_results)
        top_results = raw_results[:top_k]

        results = [
            SearchResult(
                text=r.text,
                score=r.score,
                token_count=r.token_count,
                source=SearchSource(
                    collection=r.collection,
                    document=r.document_name,
                    path=r.document_path,
                    size=_fmt_size(r.document_size),
                    page=r.page,
                    section=" > ".join(r.heading_hierarchy) if r.heading_hierarchy else None,
                    chunk_index=r.chunk_index,
                ),
            )
            for r in top_results
        ]

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=total_results,
            model=embedder.model_name,
        )

    @app.get("/collections", response_model=list[CollectionInfo])
    async def list_collections() -> list[CollectionInfo]:
        return collection_manager.list_collections()

    @app.get("/collections/{name}", response_model=CollectionInfo)
    async def get_collection(name: str) -> CollectionInfo:
        info = collection_manager.get_collection(name)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Collection {name!r} not found")
        return info

    @app.post("/collections/{name}/reindex", response_model=ReindexResponse)
    async def reindex_collection(
        name: str, background_tasks: BackgroundTasks
    ) -> ReindexResponse:
        if name not in config.collections:
            raise HTTPException(status_code=404, detail=f"Collection {name!r} not found in config")

        if indexing_state.active:
            raise HTTPException(
                status_code=409,
                detail={
                    "status": "indexing_in_progress",
                    "collection": indexing_state.current_collection,
                },
            )

        col_config = config.collections[name]
        background_tasks.add_task(
            indexing_state.run_indexing,
            name,
            col_config,
            indexer,
            True,  # delete_first for full reindex
            collection_manager,
        )

        return ReindexResponse(status="indexing_started", collection=name)

    @app.get("/status", response_model=StatusResponse)
    async def status() -> StatusResponse:
        incompatible = collection_manager.get_incompatible_collections()

        collections_info: dict[str, dict[str, Any]] = {}
        for info in collection_manager.list_collections():
            collections_info[info.name] = {
                "documents": info.document_count,
                "chunks": info.chunk_count,
                "compatible": info.compatible,
            }

        progress_data: tuple[int, int] | None = indexing_state.progress

        watcher_info: WatcherInfo | None = None
        if collection_watcher is not None:
            ws = collection_watcher.get_status()
            watcher_info = WatcherInfo(
                active=ws.active,
                watched_paths=ws.watched_paths,
                pending_debounce=ws.pending_debounce,
            )

        return StatusResponse(
            status="healthy",
            embedding_model=embedder.model_name,
            embedding_dimension=embedder.dimension,
            collections=collections_info,
            indexing=IndexingProgress(
                active=indexing_state.active,
                current_collection=indexing_state.current_collection,
                progress=progress_data,
            ),
            watcher=watcher_info,
            incompatible_collections=incompatible,
        )

    return app
