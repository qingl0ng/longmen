"""CLI entry point: serve, index, status subcommands."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass

# Disable all HuggingFace Hub network access for the embedder. The tokenizer now
# loads from a local tokenizer.json ([tokenizer].path) and no longer depends on this
# var; it remains to keep sentence-transformers/huggingface_hub (the embedder) offline.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import structlog

log = structlog.get_logger(__name__)


def _setup_logging(log_level: str) -> None:
    import logging

    import structlog

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def _build_components(config_path: str | None = None):
    """Build all service components from config."""
    from .collections import CollectionManager
    from .config import load_config
    from .embedder import Embedder
    from .indexer import Indexer
    from .manifest import Manifest
    from .store import VectorStore
    from .tokenizer import TokenCounter

    config = load_config(config_path)
    data_dir = config.storage.resolved_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    token_counter = TokenCounter(config.tokenizer.resolved_path())
    embedder = Embedder(config.embedding.model, device=config.embedding.device)
    store = VectorStore(data_dir, embedder.dimension)
    manifest = Manifest(data_dir / "manifest.db")
    indexer = Indexer(
        store=store,
        manifest=manifest,
        embedder=embedder,
        token_counter=token_counter,
        chunking_config=config.chunking,
    )
    collection_manager = CollectionManager(
        config=config,
        store=store,
        manifest=manifest,
        embedder=embedder,
    )

    return config, token_counter, embedder, store, manifest, indexer, collection_manager


@dataclass
class AppState:
    """Holds initialized service components for graceful shutdown."""
    watcher: object  # CollectionWatcher | None
    config_watcher: object  # ConfigWatcher | None
    indexing_state: object  # IndexingState
    manifest: object  # Manifest


async def _startup_scan_and_index(
    config,
    manifest,
    indexer,
    indexing_state,
) -> None:
    """Scan all collections at startup, log summary, run incremental indexing for changed ones."""
    from .indexer import _scan_directory

    needs_indexing: list[str] = []

    for name, col_config in config.collections.items():
        root = col_config.resolved_path()
        if not root.exists() or not root.is_dir():
            log.warning("startup_collection_path_invalid", collection=name, path=str(root))
            continue

        disk_files = await asyncio.to_thread(_scan_directory, root)
        disk_rel = {str(f.relative_to(root)) for f in disk_files}
        existing_entries = {e.file_path: e for e in manifest.list_collection(name)}

        new_count = len(disk_rel - existing_entries.keys())
        deleted_count = len(set(existing_entries.keys()) - disk_rel)
        # Can't distinguish changed vs unchanged without hashing — log as "existing"
        existing_count = len(disk_rel & existing_entries.keys())

        log.info(
            "collection_scan",
            collection=name,
            new=new_count,
            deleted=deleted_count,
            existing=existing_count,
        )

        if disk_files or existing_entries:
            needs_indexing.append(name)

    if needs_indexing:
        log.info("startup_indexing_queued", collections=len(needs_indexing))
        for name in needs_indexing:
            col_config = config.collections[name]
            await indexing_state.run_indexing(name, col_config, indexer, delete_first=False)
            result = indexing_state.result
            if result:
                log.info(
                    "startup_indexing_complete",
                    collection=name,
                    documents=result.documents_processed,
                    skipped=result.documents_skipped,
                    deleted=result.documents_deleted,
                    chunks=result.chunks_created,
                    duration=f"{result.duration_seconds:.2f}s",
                )


async def _handle_config_change(
    old_config,
    new_config,
    watcher,
    indexer,
    indexing_state,
) -> None:
    """React to rag.toml changes detected by ConfigWatcher."""
    # Model change: requires restart
    if old_config.embedding.model != new_config.embedding.model:
        log.error(
            "embedding_model_changed_restart_required",
            old=old_config.embedding.model,
            new=new_config.embedding.model,
        )

    # Service host/port change: requires restart
    if (
        old_config.service.host != new_config.service.host
        or old_config.service.port != new_config.service.port
    ):
        log.warning(
            "service_address_changed_restart_required",
            old=f"{old_config.service.host}:{old_config.service.port}",
            new=f"{new_config.service.host}:{new_config.service.port}",
        )

    old_names = set(old_config.collections.keys())
    new_names = set(new_config.collections.keys())

    # Removed collections
    for name in old_names - new_names:
        log.info("collection_removed_from_config", collection=name)

    # Added collections — start watching and trigger startup scan
    added = new_names - old_names
    if added:
        for name in added:
            log.info("collection_added_to_config", collection=name)
        # Update watcher with new collection set
        if watcher is not None:
            await watcher.update_collections(new_config.collections)
        # Index new collections
        for name in added:
            col_config = new_config.collections[name]
            root = col_config.resolved_path()
            if not root.exists():
                log.warning("new_collection_path_missing", collection=name, path=str(root))
                continue
            await indexing_state.run_indexing(name, col_config, indexer, delete_first=False)

    # Chunk size changes
    for name in old_names & new_names:
        old_col = old_config.collections[name]
        new_col = new_config.collections[name]
        old_size = old_col.chunk_size or old_config.chunking.default_chunk_size
        new_size = new_col.chunk_size or new_config.chunking.default_chunk_size
        if old_size != new_size:
            log.warning(
                "collection_chunk_size_changed_reindex_required",
                collection=name,
                old=old_size,
                new=new_size,
            )

    # Update watcher collections if not already done
    if watcher is not None and not added:
        await watcher.update_collections(new_config.collections)


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI server."""
    import contextlib

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_serve_async(args))


async def _serve_async(args: argparse.Namespace) -> None:
    import uvicorn

    from .api import IndexingState, create_app
    from .config import _resolve_config_path
    from .watcher import CollectionWatcher, ConfigWatcher

    config, token_counter, embedder, store, manifest, indexer, collection_manager = (
        _build_components(args.config)
    )
    _setup_logging(config.service.log_level)

    # Check for model mismatches
    incompatible = collection_manager.get_incompatible_collections()
    if incompatible:
        log.warning(
            "incompatible_collections_detected",
            collections=incompatible,
            current_model=embedder.model_name,
        )

    indexing_state = IndexingState()

    # Create file watcher (if enabled)
    collection_watcher: CollectionWatcher | None = None
    config_watcher: ConfigWatcher | None = None

    if config.watcher.enabled:
        collection_watcher = CollectionWatcher(
            collections=config.collections,
            indexer=indexer,
            indexing_state=indexing_state,
            store=store,
            manifest=manifest,
            debounce_seconds=config.watcher.debounce_seconds,
        )

        if config.watcher.watch_config:
            config_path = _resolve_config_path(args.config)

            async def on_config_change(old, new):
                await _handle_config_change(old, new, collection_watcher, indexer, indexing_state)

            config_watcher = ConfigWatcher(
                config_path=config_path,
                current_config=config,
                on_config_change=on_config_change,
                debounce_seconds=1.0,
            )

    app = create_app(
        config=config,
        embedder=embedder,
        store=store,
        indexer=indexer,
        collection_manager=collection_manager,
        indexing_state=indexing_state,
        collection_watcher=collection_watcher,
    )

    uv_config = uvicorn.Config(
        app,
        host=config.service.host,
        port=config.service.port,
        log_level=config.service.log_level,
    )
    server = uvicorn.Server(uv_config)

    # Start startup indexing scan in background — server becomes available immediately
    if config.collections:
        asyncio.create_task(
            _startup_scan_and_index(config, manifest, indexer, indexing_state)
        )

    # Start watchers
    if collection_watcher is not None:
        await collection_watcher.start()
    if config_watcher is not None:
        await config_watcher.start()

    log.info("starting_server", host=config.service.host, port=config.service.port)
    await server.serve()

    # --- Shutdown hook (runs after uvicorn exits on SIGINT/SIGTERM) ---
    log.info("shutdown_initiated")

    if config_watcher is not None:
        await config_watcher.stop()
        log.info("config_watcher_stopped")

    if collection_watcher is not None:
        await collection_watcher.stop()
        log.info("watcher_stopped")

    if indexing_state.active:
        log.info("waiting_for_indexing", collection=indexing_state.current_collection)
        try:
            await asyncio.wait_for(indexing_state.wait_complete(), timeout=30.0)
            log.info("indexing_completed_before_shutdown")
        except TimeoutError:
            log.warning("indexing_timed_out_during_shutdown")

    manifest.close()
    log.info("shutdown_complete")


def cmd_index(args: argparse.Namespace) -> None:
    """Index collections."""
    if args.oneshot:
        _index_oneshot(args)
    else:
        _index_via_http(args)


def _index_oneshot(args: argparse.Namespace) -> None:
    """Run indexer directly in-process and print results."""
    try:
        config, token_counter, embedder, store, manifest, indexer, collection_manager = (
            _build_components(args.config)
        )
    except RuntimeError as exc:
        if "already accessed" in str(exc):
            print(
                "Error: the Qdrant database is locked by a running ragsvc instance.\n"
                "Use 'ragsvc index' (without --oneshot) to trigger indexing"
                " via the running service.",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    _setup_logging(config.service.log_level)

    async def run() -> None:
        if args.all:
            results = await indexer.index_all(config.collections)
        elif args.collection:
            if args.collection not in config.collections:
                print(f"Error: collection {args.collection!r} not found in config", file=sys.stderr)
                sys.exit(1)
            col_config = config.collections[args.collection]
            results = [await indexer.index_collection(args.collection, col_config)]
        else:
            print("Error: specify a collection name or --all", file=sys.stderr)
            sys.exit(1)

        for result in results:
            print(f"\nCollection: {result.collection}")
            print(f"  Processed : {result.documents_processed}")
            print(f"  Skipped   : {result.documents_skipped}")
            print(f"  Failed    : {result.documents_failed}")
            print(f"  Deleted   : {result.documents_deleted}")
            print(f"  Chunks    : {result.chunks_created}")
            print(f"  Tokens    : {result.total_tokens}")
            print(f"  Duration  : {result.duration_seconds:.2f}s")
            if result.errors:
                print(f"  Errors ({len(result.errors)}):")
                for err in result.errors:
                    print(f"    - {err}")

    asyncio.run(run())


def _index_via_http(args: argparse.Namespace) -> None:
    """Trigger reindex via HTTP and poll for completion."""
    import httpx

    from .config import load_config

    config = load_config(args.config)
    base_url = f"http://{config.service.host}:{config.service.port}"

    collections_to_index: list[str] = []
    if args.all:
        collections_to_index = list(config.collections.keys())
    elif args.collection:
        collections_to_index = [args.collection]
    else:
        print("Error: specify a collection name or --all", file=sys.stderr)
        sys.exit(1)

    with httpx.Client(timeout=10.0) as client:
        for coll_name in collections_to_index:
            print(f"Triggering reindex of {coll_name!r}...")
            try:
                resp = client.post(f"{base_url}/collections/{coll_name}/reindex")
                if resp.status_code == 202:
                    print(f"  Indexing started for {coll_name!r}")
                elif resp.status_code == 409:
                    print(f"  Indexing already in progress: {resp.json()}")
                    continue
                else:
                    print(f"  Error {resp.status_code}: {resp.text}", file=sys.stderr)
                    continue
            except httpx.ConnectError:
                print(
                    f"Error: could not connect to {base_url}. Is the service running?",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Poll status until done
            while True:
                time.sleep(2)
                try:
                    status_resp = client.get(f"{base_url}/status")
                    status_data = status_resp.json()
                    indexing = status_data.get("indexing", {})
                    if indexing.get("active"):
                        progress = indexing.get("progress")
                        if progress:
                            print(f"  Progress: {progress[0]}/{progress[1]}", end="\r")
                    else:
                        print(f"\n  Indexing of {coll_name!r} complete.")
                        break
                except Exception as exc:
                    print(f"\nError polling status: {exc}", file=sys.stderr)
                    break


def cmd_status(args: argparse.Namespace) -> None:
    """Query the running service for status and pretty-print."""
    import httpx

    from .config import load_config

    config = load_config(args.config)
    base_url = f"http://{config.service.host}:{config.service.port}"

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/status")
            data = resp.json()
    except httpx.ConnectError:
        print(f"Error: could not connect to {base_url}. Is the service running?", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(data, indent=2))
        return

    _print_status(data)


def _print_status(data: dict) -> None:
    """Pretty-print the /status response."""
    sep = "\u2500" * 45

    status = data.get("status", "unknown")
    model = data.get("embedding_model", "?")
    dim = data.get("embedding_dimension", "?")
    watcher = data.get("watcher")

    print("\nRAG Service Status")
    print(sep)
    print(f"  Status:  {status}")
    print(f"  Model:   {model} ({dim} dim)")

    if watcher is not None:
        watched = watcher.get("watched_paths", 0)
        watcher_status = "active" if watcher.get("active") else "inactive"
        print(f"  Watcher: {watcher_status} ({watched} directories)")
        pending = watcher.get("pending_debounce", [])
        if pending:
            print(f"           debouncing: {', '.join(pending)}")
    else:
        print("  Watcher: disabled")

    indexing = data.get("indexing", {})
    if indexing.get("active"):
        cur = indexing.get("current_collection", "?")
        progress = indexing.get("progress")
        prog_str = f" ({progress[0]}/{progress[1]})" if progress else ""
        print(f"  Indexing: {cur}{prog_str}")

    collections = data.get("collections", {})
    if collections:
        print("\nCollections:")
        for name, info in collections.items():
            docs = info.get("documents", 0)
            chunks = info.get("chunks", 0)
            compatible = info.get("compatible", True)
            doc_label = "docs" if docs != 1 else "doc"
            compat_str = "" if compatible else "  [incompatible — needs reindex]"

            # Show indexing progress if this collection is currently being indexed
            index_note = ""
            if indexing.get("active") and indexing.get("current_collection") == name:
                progress = indexing.get("progress")
                if progress:
                    index_note = f"  [indexing {progress[0]}/{progress[1]}]"
                else:
                    index_note = "  [indexing]"

            suffix = f"{compat_str}{index_note}"
            line = f"  {name:<20} {docs:>4} {doc_label:<4}  {chunks:>5} chunks{suffix}"
            print(line)

    incompatible = data.get("incompatible_collections", [])
    if incompatible:
        print(f"\nIncompatible collections (need reindex): {', '.join(incompatible)}")
    else:
        print("\nNo incompatible collections.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ragsvc",
        description="RAG service for the terminal coding assistant",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to rag.toml config file (default: ~/.longmen/rag/rag.toml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server")
    serve_parser.set_defaults(func=cmd_serve)

    # index
    index_parser = subparsers.add_parser("index", help="Index collections")
    index_parser.add_argument(
        "collection",
        nargs="?",
        default=None,
        help="Collection name to index",
    )
    index_parser.add_argument(
        "--all",
        action="store_true",
        help="Index all collections defined in config",
    )
    index_parser.add_argument(
        "--oneshot",
        action="store_true",
        help="Run indexer in-process (no server required), print results, exit",
    )
    index_parser.set_defaults(func=cmd_index)

    # status
    status_parser = subparsers.add_parser("status", help="Query the running service for status")
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of pretty-printed table",
    )
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
