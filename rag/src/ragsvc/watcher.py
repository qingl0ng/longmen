"""File watcher with debounce for automatic collection indexing."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from watchfiles import Change, awatch

from .config import CollectionConfig, RAGConfig
from .manifest import Manifest, make_document_id
from .store import VectorStore

if TYPE_CHECKING:
    from .api import IndexingState
    from .indexer import Indexer

log = structlog.get_logger(__name__)

_SUPPORTED_EXTENSIONS = frozenset({".md", ".markdown", ".mkd", ".txt", ".text", ".rst", ".pdf"})
_IGNORED_SUFFIXES = frozenset({".tmp", ".swp", ".partial"})
_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@dataclass
class WatcherStatus:
    active: bool
    watched_paths: int
    pending_debounce: list[str]  # collection names with active debounce timers


class CollectionWatcher:
    """Watches collection directories for file changes, triggers incremental indexing."""

    def __init__(
        self,
        collections: dict[str, CollectionConfig],
        indexer: Indexer,
        indexing_state: IndexingState,
        store: VectorStore,
        manifest: Manifest,
        debounce_seconds: float = 3.0,
    ) -> None:
        self._collections = collections
        self._indexer = indexer
        self._indexing_state = indexing_state
        self._store = store
        self._manifest = manifest
        self._debounce_seconds = debounce_seconds
        self._pending: dict[str, set[Path]] = {}
        self._debounce_tasks: dict[str, asyncio.Task[None]] = {}
        self._watch_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._active = False

    def get_status(self) -> WatcherStatus:
        valid_paths = sum(
            1 for c in self._collections.values() if c.resolved_path().exists()
        )
        return WatcherStatus(
            active=self._active,
            watched_paths=valid_paths,
            pending_debounce=list(self._debounce_tasks.keys()),
        )

    async def start(self) -> None:
        """Start watching all collection directories."""
        self._stop_event.clear()
        self._active = True
        for name, config in self._collections.items():
            path = config.resolved_path()
            if path.exists():
                log.info("watching_collection", collection=name, path=str(path))
        self._watch_task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Stop all watchers gracefully."""
        self._stop_event.set()
        self._active = False
        for task in list(self._debounce_tasks.values()):
            task.cancel()
        self._debounce_tasks.clear()
        if self._watch_task:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task
            self._watch_task = None

    async def update_collections(self, new_collections: dict[str, CollectionConfig]) -> None:
        """Update watched collections. Restarts the watch loop."""
        self._collections = new_collections
        if self._watch_task and not self._watch_task.done():
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task
        if self._active:
            self._stop_event.clear()
            self._watch_task = asyncio.create_task(self._watch_loop())

    async def _watch_loop(self) -> None:
        paths = [
            str(config.resolved_path())
            for config in self._collections.values()
            if config.resolved_path().exists()
        ]
        if not paths:
            log.info("no_valid_collection_paths_to_watch")
            return

        try:
            async for changes in awatch(*paths, stop_event=self._stop_event):
                for change_type, file_path_str in changes:
                    file_path = Path(file_path_str)
                    collection = self._path_to_collection(file_path)
                    if collection is None:
                        continue
                    if not self._is_supported_file(file_path):
                        continue
                    log.info(
                        "file_changed",
                        collection=collection,
                        path=file_path.name,
                        change=change_type.name.lower(),
                    )
                    match change_type:
                        case Change.added | Change.modified:
                            self._schedule_index(collection, file_path)
                        case Change.deleted:
                            await self._handle_deletion(collection, file_path)
        except asyncio.CancelledError:
            pass

    def _path_to_collection(self, file_path: Path) -> str | None:
        """Return the most specific collection name whose root contains file_path."""
        best: str | None = None
        best_depth = -1
        for name, config in self._collections.items():
            root = config.resolved_path()
            try:
                file_path.relative_to(root)
            except ValueError:
                continue
            depth = len(root.parts)
            if depth > best_depth:
                best_depth = depth
                best = name
        return best

    def _is_supported_file(self, file_path: Path) -> bool:
        """Return True if the file should trigger indexing."""
        name = file_path.name
        if name.startswith("."):
            return False
        if name.endswith("~") or any(name.endswith(s) for s in _IGNORED_SUFFIXES):
            return False
        if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            return False
        # Skip files inside hidden directories
        for part in file_path.parts:
            if part.startswith("."):
                return False
        # Skip oversized files (if they still exist)
        try:
            if file_path.exists() and file_path.stat().st_size > _MAX_FILE_SIZE:
                log.warning("file_too_large_ignored", path=str(file_path))
                return False
        except OSError:
            pass
        return True

    def _schedule_index(self, collection: str, file_path: Path) -> None:
        """Add file to pending set and (re)start the debounce timer for the collection."""
        if collection not in self._pending:
            self._pending[collection] = set()
        self._pending[collection].add(file_path)

        existing = self._debounce_tasks.get(collection)
        if existing and not existing.done():
            existing.cancel()

        self._debounce_tasks[collection] = asyncio.create_task(
            self._debounce_fire(collection)
        )

    async def _debounce_fire(self, collection: str) -> None:
        """Wait for the debounce window, then trigger indexing for accumulated files."""
        try:
            await asyncio.sleep(self._debounce_seconds)
        except asyncio.CancelledError:
            return

        files = list(self._pending.pop(collection, set()))
        self._debounce_tasks.pop(collection, None)
        if not files:
            return

        log.info("debounce_fired", collection=collection, files_pending=len(files))
        col_config = self._collections.get(collection)
        if col_config is None:
            return

        # Respect the shared indexing lock — wait rather than drop work
        await self._indexing_state.run_indexing_files(
            collection_name=collection,
            collection_config=col_config,
            file_paths=files,
            indexer=self._indexer,
        )
        log.info("watcher_indexing_complete", collection=collection)

    async def _handle_deletion(self, collection: str, file_path: Path) -> None:
        """Remove deleted file's chunks from the index immediately (no debounce)."""
        col_config = self._collections.get(collection)
        if col_config is None:
            return
        root = col_config.resolved_path()
        try:
            rel_path = str(file_path.relative_to(root))
        except ValueError:
            return
        document_id = make_document_id(collection, rel_path)
        self._store.delete_by_document(collection, document_id)
        self._manifest.delete(document_id)
        log.info("file_removed_from_index", collection=collection, path=rel_path)


class ConfigWatcher:
    """Watches rag.toml for changes and calls a callback when the config changes."""

    def __init__(
        self,
        config_path: Path,
        current_config: RAGConfig,
        on_config_change: Callable[[RAGConfig, RAGConfig], Awaitable[None]],
        debounce_seconds: float = 1.0,
    ) -> None:
        self._config_path = config_path
        self._current_config = current_config
        self._on_change = on_config_change
        self._debounce_seconds = debounce_seconds
        self._stop_event: asyncio.Event = asyncio.Event()
        self._watch_task: asyncio.Task[None] | None = None
        self._debounce_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._stop_event.clear()
        self._watch_task = asyncio.create_task(self._watch_loop())
        log.info("watching_config", path=str(self._config_path))

    async def stop(self) -> None:
        self._stop_event.set()
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        if self._watch_task:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task

    async def _watch_loop(self) -> None:
        if not self._config_path.exists():
            log.debug("config_path_not_found_skipping_watch", path=str(self._config_path))
            return
        try:
            async for _changes in awatch(str(self._config_path), stop_event=self._stop_event):
                if self._debounce_task and not self._debounce_task.done():
                    self._debounce_task.cancel()
                self._debounce_task = asyncio.create_task(self._debounce_fire())
        except asyncio.CancelledError:
            pass

    async def _debounce_fire(self) -> None:
        try:
            await asyncio.sleep(self._debounce_seconds)
        except asyncio.CancelledError:
            return

        log.info("config_changed_reloading", path=str(self._config_path))
        from .config import load_config

        try:
            new_config = load_config(self._config_path)
        except Exception as exc:
            log.error("config_reload_failed", path=str(self._config_path), error=str(exc))
            return

        old_config = self._current_config
        self._current_config = new_config
        await self._on_change(old_config, new_config)
