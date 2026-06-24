"""Abstract tool interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    name: str

    @abstractmethod
    async def execute(self, root_path: str, **kwargs: Any) -> dict[str, Any]:
        """Returns dict with stdout, stderr, exit_code or content."""

    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """Return OpenAI-format function schema."""

    def _safe_path(self, root_path: str, path: str) -> str:
        """Resolve path and ensure it is within root_path. Raises ValueError if not."""
        from pathlib import Path

        root_resolved = Path(root_path).expanduser().resolve()
        target = (root_resolved / path).resolve()
        if not target.is_relative_to(root_resolved):
            raise ValueError(f"Path escape attempt: '{path}' resolves outside root '{root_path}'")
        return str(target)
