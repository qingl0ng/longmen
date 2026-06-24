"""File watcher for hot reload — watches gateway.toml and data_dir."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import structlog

log = structlog.get_logger(__name__)

RESTART_REQUIRED_KEYS = {"host", "port", "data_dir"}


class ConfigWatcher:
    def __init__(
        self,
        config_path: str,
        data_dir: str,
        on_reload: Callable[..., Any],
    ) -> None:
        self._config_path = Path(config_path)
        self._data_dir = Path(data_dir)
        self._on_reload = on_reload
        self._debounce_seconds = 0.5
        self._pending_task: asyncio.Task[None] | None = None
        self._changed_files: set[Path] = set()

    async def watch(self) -> None:
        """Watch for file changes, debounce 500ms, call on_reload."""
        try:
            from watchfiles import awatch

            paths_to_watch = [str(self._config_path.parent)]
            if self._data_dir.exists():
                paths_to_watch.append(str(self._data_dir))

            async for changes in awatch(*paths_to_watch):
                for _change_type, path_str in changes:
                    self._changed_files.add(Path(path_str))

                # Cancel existing debounce task
                if self._pending_task and not self._pending_task.done():
                    self._pending_task.cancel()

                self._pending_task = asyncio.create_task(self._debounced_reload())

        except Exception as e:
            log.error("config_watcher.error", error=str(e))

    async def _debounced_reload(self) -> None:
        await asyncio.sleep(self._debounce_seconds)
        changed = set(self._changed_files)
        self._changed_files.clear()

        if not changed:
            return

        # Filter out session files - they should not trigger reloads
        # This must be done here to avoid logging misleading messages
        filtered_changed: set[Path] = set()
        for path in changed:
            # Skip files in sessions directories
            try:
                rel = path.relative_to(self._data_dir / "projects")
                if "sessions" in rel.parts:
                    continue
            except ValueError:
                pass
            filtered_changed.add(path)

        if not filtered_changed:
            return

        log.info("config_watcher.reload_triggered", files=[str(f) for f in filtered_changed])

        for path in filtered_changed:
            await self._process_change(path)

    async def _process_change(self, path: Path) -> None:
        try:
            # Determine scope
            config_path_resolved = self._config_path.resolve()
            if path.resolve() == config_path_resolved:
                await self._reload_gateway_config(path)
            elif self._data_dir in path.parents:
                await self._reload_project_config(path)
        except Exception as e:
            log.error("config_watcher.process_error", path=str(path), error=str(e))

    async def _reload_gateway_config(self, path: Path) -> None:
        log.info("config_watcher.reload_gateway", path=str(path))
        changed: list[str] = []

        # Determine what changed (we just report all hot-reloadable sections)
        changed = ["model", "auth", "permissions", "timeouts", "server.log_level", "web"]

        # Log warning for restart-required settings (we don't know which changed without diffing)
        log.info(
            "config_watcher.gateway_reloaded",
            note=(
                "Settings requiring restart (host, port, data_dir)"
                " will not take effect until restart."
            ),
        )

        await self._on_reload("gateway", None, changed)

    async def _reload_project_config(self, path: Path) -> None:
        try:
            rel = path.relative_to(self._data_dir / "projects")
            project_id = rel.parts[0]
        except ValueError:
            return

        changed: list[str] = []
        name = path.name
        if name == "project.toml":
            changed = ["project_settings"]
        elif name == "agents.toml":
            changed = ["agents"]
        elif name == "permissions.toml":
            changed = ["permissions"]
        elif path.suffix == ".md" and "prompts" in path.parts:
            changed = ["agents"]
        else:
            changed = [name]

        log.info("config_watcher.reload_project", project_id=project_id, changed=changed)
        await self._on_reload("project", project_id, changed)
