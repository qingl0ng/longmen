"""Persistent session storage — append-only JSONL messages + atomic metadata."""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

    from .session import TrackedMessage

log = structlog.get_logger(__name__)


@dataclass
class SessionMeta:
    session_id: str
    project_id: str
    created_at: float       # unix timestamp (seconds)
    last_active: float      # unix timestamp (seconds)
    turn_count: int         # number of user prompts
    total_tokens: int       # sum of all message tokens
    compacted: bool         # True if compaction has been applied
    compaction_count: int   # how many times compacted


class SessionNotFoundError(Exception):
    pass


class SessionStore:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def _sessions_dir(self, project_id: str) -> Path:
        return self._data_dir / "projects" / project_id / "sessions"

    def _jsonl_path(self, project_id: str, session_id: str) -> Path:
        return self._sessions_dir(project_id) / f"{session_id}.jsonl"

    def _meta_path(self, project_id: str, session_id: str) -> Path:
        return self._sessions_dir(project_id) / f"{session_id}.meta.json"

    async def save_message(
        self, project_id: str, session_id: str, message: TrackedMessage
    ) -> None:
        """Append a single message to the session's JSONL file."""
        sessions_dir = self._sessions_dir(project_id)
        sessions_dir.mkdir(parents=True, exist_ok=True)
        path = self._jsonl_path(project_id, session_id)
        line = json.dumps(dataclasses.asdict(message))
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    async def save_meta(
        self, project_id: str, session_id: str, meta: SessionMeta
    ) -> None:
        """Atomically rewrite the session's metadata file."""
        sessions_dir = self._sessions_dir(project_id)
        sessions_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self._meta_path(project_id, session_id)
        tmp_path = meta_path.with_suffix(".tmp")
        data = dataclasses.asdict(meta)
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp_path, meta_path)

    async def load_session(
        self, project_id: str, session_id: str
    ) -> tuple[SessionMeta, list[TrackedMessage], TrackedMessage | None]:
        """Load a complete session from disk. Returns (meta, messages, compacted_summary).
        Raises SessionNotFound if files don't exist."""
        from .session import TrackedMessage

        meta_path = self._meta_path(project_id, session_id)
        jsonl_path = self._jsonl_path(project_id, session_id)

        if not meta_path.exists() or not jsonl_path.exists():
            raise SessionNotFoundError(
                f"Session {session_id} not found for project {project_id}"
            )

        # Load metadata
        raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = SessionMeta(**raw_meta)

        # Load all JSONL lines
        raw_lines: list[TrackedMessage] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_lines.append(TrackedMessage(**json.loads(line)))

        # Find the last compaction marker
        last_marker_idx = -1
        for i, msg in enumerate(raw_lines):
            if msg.role == "_compaction" and msg.message_type == "compaction_marker":
                last_marker_idx = i

        compacted_summary: TrackedMessage | None = None
        messages: list[TrackedMessage] = []

        if last_marker_idx == -1:
            # No compaction — all messages are live
            messages = [m for m in raw_lines if m.role != "_compaction"]
        else:
            marker = raw_lines[last_marker_idx]
            replaces_before: float = marker.metadata.get("replaces_before", 0.0)

            # The summary is the line immediately after the marker
            if last_marker_idx + 1 < len(raw_lines):
                candidate = raw_lines[last_marker_idx + 1]
                if candidate.message_type == "compaction_summary":
                    compacted_summary = candidate

            # Discard messages before the marker that pre-date replaces_before
            for i, msg in enumerate(raw_lines):
                if msg.role == "_compaction":
                    continue
                if msg.message_type == "compaction_summary":
                    continue
                if i < last_marker_idx and msg.timestamp < replaces_before:
                    continue
                messages.append(msg)

        return meta, messages, compacted_summary

    async def get_last_session(self, project_id: str) -> SessionMeta | None:
        """Return the most recently active session for this project, or None."""
        sessions = await self.list_sessions(project_id)
        return sessions[0] if sessions else None

    async def list_sessions(self, project_id: str, limit: int = 20) -> list[SessionMeta]:
        """List sessions sorted by last_active descending. Reads only .meta.json files."""
        sessions_dir = self._sessions_dir(project_id)
        if not sessions_dir.exists():
            return []

        metas: list[SessionMeta] = []
        for path in sessions_dir.glob("*.meta.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                metas.append(SessionMeta(**raw))
            except Exception as exc:
                log.warning("session_store.bad_meta", path=str(path), error=str(exc))

        metas.sort(key=lambda m: m.last_active, reverse=True)
        return metas[:limit] if limit > 0 else metas

    async def delete_session(self, project_id: str, session_id: str) -> bool:
        """Delete a session's files. Returns False if not found."""
        jsonl = self._jsonl_path(project_id, session_id)
        meta = self._meta_path(project_id, session_id)

        if not jsonl.exists() and not meta.exists():
            return False

        for path in (jsonl, meta):
            if path.exists():
                path.unlink()
        return True

    async def prune_all(self, max_age_days: int, max_per_project: int) -> int:
        """Prune old/excess sessions across all projects. Called at gateway startup.
        Returns total count pruned."""
        total = 0
        projects_dir = self._data_dir / "projects"
        if not projects_dir.exists():
            return 0
        for project_dir in projects_dir.iterdir():
            if project_dir.is_dir():
                total += await self._prune_project(
                    project_dir.name, max_age_days, max_per_project
                )
        return total

    async def _prune_project(
        self, project_id: str, max_age_days: int, max_per_project: int
    ) -> int:
        """Prune sessions for a single project. Returns count pruned."""
        metas = await self.list_sessions(project_id, limit=0)  # all, sorted newest first
        cutoff = time.time() - max_age_days * 86400
        pruned = 0

        for i, meta in enumerate(metas):
            should_prune = meta.last_active < cutoff or i >= max_per_project
            if should_prune:
                deleted = await self.delete_session(project_id, meta.session_id)
                if deleted:
                    pruned += 1
                    log.info(
                        "session_store.pruned",
                        project_id=project_id,
                        session_id=meta.session_id,
                        reason="age" if meta.last_active < cutoff else "count",
                    )

        return pruned
