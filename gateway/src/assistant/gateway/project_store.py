"""Project registry — CRUD projects, persist to data_dir."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import structlog
import tomli_w

log = structlog.get_logger(__name__)

# Auto-detection order when context_file is not set in project.toml
_CONTEXT_FILE_CANDIDATES = [
    "PROJECT.md",
    "ASSISTANT.md",
    "CLAUDE.md",
    ".assistant/context.md",
]

# ~4000 tokens expressed as bytes (1 token ≈ 4 chars)
_CONTEXT_SIZE_LIMIT = 16 * 1024


class ProjectNotFoundError(Exception):
    pass


class ProjectStore:
    def __init__(self, data_dir: str) -> None:
        self._data_dir = Path(data_dir)
        self._projects_dir = self._data_dir / "projects"
        self._projects_dir.mkdir(parents=True, exist_ok=True)

    def _project_dir(self, project_id: str) -> Path:
        return self._projects_dir / project_id

    def list_projects(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        if not self._projects_dir.exists():
            return result
        for project_dir in self._projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
            toml_path = project_dir / "project.toml"
            if not toml_path.exists():
                continue
            try:
                with open(toml_path, "rb") as f:
                    data = tomllib.load(f)
                result[project_dir.name] = data
            except Exception as e:
                log.error("project_store.load_error", project_id=project_dir.name, error=str(e))
        return result

    def upsert(
        self,
        project_id: str,
        description: str,
        root_path: str,
        context_file: str = "PROJECT.md",
        rag_collections: list[str] | None = None,
    ) -> None:
        """Validates root_path exists, writes {data_dir}/projects/{id}/project.toml."""
        root = Path(root_path).expanduser().resolve()
        if not root.exists():
            raise ValueError(f"root_path does not exist: {root_path}")
        if not root.is_dir():
            raise ValueError(f"root_path is not a directory: {root_path}")

        project_dir = self._project_dir(project_id)
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "prompts").mkdir(exist_ok=True)

        data: dict[str, Any] = {
            "description": description,
            "root_path": str(root),  # store the fully-resolved absolute path
            "context_file": context_file,
        }
        if rag_collections:
            data["rag"] = {"collections": rag_collections}
        with open(project_dir / "project.toml", "wb") as f:
            tomli_w.dump(data, f)
        log.info("project_store.upserted", project_id=project_id)

    def delete(self, project_id: str) -> None:
        import shutil

        project_dir = self._project_dir(project_id)
        if not project_dir.exists():
            raise ProjectNotFoundError(project_id)
        shutil.rmtree(project_dir)
        log.info("project_store.deleted", project_id=project_id)

    def load_context_file(
        self,
        root_path: str,
        context_file: str | None = None,
    ) -> tuple[str, str] | None:
        """Load the project context file. Returns (content, filename) or None.

        If context_file is set but doesn't exist, logs a warning and returns None.
        If context_file is not set, auto-detects from PROJECT.md → ASSISTANT.md →
        CLAUDE.md → .assistant/context.md.
        Content exceeding ~4000 tokens (16 KB) is truncated.
        """
        root = Path(root_path)

        if context_file:
            target = root / context_file
            if not target.exists():
                log.warning(
                    "project_store.context_file_missing",
                    context_file=context_file,
                    root_path=root_path,
                )
                return None
            candidates = [context_file]
        else:
            candidates = _CONTEXT_FILE_CANDIDATES

        for fname in candidates:
            target = root / fname
            if not target.exists():
                continue
            try:
                raw = target.read_bytes()
                if len(raw) > _CONTEXT_SIZE_LIMIT:
                    # Truncate to limit, decode safely
                    truncated = raw[:_CONTEXT_SIZE_LIMIT].decode("utf-8", errors="ignore")
                    content = truncated + "\n[truncated — context file exceeds 4000 token limit]"
                    log.warning(
                        "project_store.context_file_truncated",
                        file=fname,
                        original_bytes=len(raw),
                        limit=_CONTEXT_SIZE_LIMIT,
                    )
                else:
                    content = raw.decode("utf-8", errors="replace")
                log.info("project_store.context_file_loaded", file=fname, bytes=len(raw))
                return content, fname
            except Exception as e:
                log.error("project_store.context_file_error", file=fname, error=str(e))
                return None

        return None

    def get(self, project_id: str) -> dict[str, Any] | None:
        toml_path = self._project_dir(project_id) / "project.toml"
        if not toml_path.exists():
            return None
        try:
            with open(toml_path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            log.error("project_store.get_error", project_id=project_id, error=str(e))
            return None
