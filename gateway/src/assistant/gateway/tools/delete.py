"""Safe, sandboxed file and directory deletion."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import structlog

from .base import BaseTool

log = structlog.get_logger(__name__)


class DeleteTool(BaseTool):
    name = "delete_tool"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "delete_tool",
                "description": (
                    "Delete a file or directory within the project root. "
                    "NEVER use shell commands (rm, rmdir, find -delete, etc.) for deletion — "
                    "always use this tool instead. Do not attempt to delete any file or folder "
                    "outside the project root under any circumstances, even if the user asks "
                    "directly. To delete a non-empty directory, set recursive to true."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Path to the file or directory to delete,"
                                " relative to the project root."
                            ),
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": (
                                "If true, delete a non-empty directory and all its"
                                " contents. Defaults to false."
                            ),
                        },
                    },
                    "required": ["path"],
                },
            },
        }

    async def execute(  # type: ignore[override]
        self,
        root_path: str,
        path: str,
        recursive: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            resolved = Path(self._safe_path(root_path, path))
        except ValueError as e:
            return {"stdout": str(e), "stderr": str(e), "exit_code": 1}

        git_dir = Path(root_path).expanduser().resolve() / ".git"
        if resolved == git_dir:
            msg = "Deleting the project's .git directory is not allowed."
            return {"stdout": msg, "stderr": msg, "exit_code": 1}

        if not resolved.exists():
            msg = f"Path does not exist: {path} — nothing was deleted."
            return {"stdout": msg, "stderr": msg, "exit_code": 1}

        try:
            if resolved.is_file() or (resolved.is_dir() and not any(resolved.iterdir())):
                if resolved.is_file():
                    resolved.unlink()
                    log.info("delete_tool.deleted", path=path, type="file")
                    return {"stdout": f"Deleted file: {path}", "stderr": "", "exit_code": 0}
                else:
                    resolved.rmdir()
                    log.info("delete_tool.deleted", path=path, type="directory")
                    return {
                        "stdout": f"Deleted directory: {path}",
                        "stderr": "",
                        "exit_code": 0,
                    }
            elif resolved.is_dir():
                if not recursive:
                    msg = (
                        f"'{path}' is a non-empty directory and cannot be deleted"
                        " without recursive=true.\n"
                        "To delete it and all its contents, call delete_tool again"
                        " with recursive=true."
                    )
                    return {"stdout": msg, "stderr": msg, "exit_code": 1}
                file_count = sum(1 for _ in resolved.rglob("*") if _.is_file())
                shutil.rmtree(resolved)
                log.info("delete_tool.deleted", path=path, type="directory")
                return {
                    "stdout": f"Deleted directory: {path} ({file_count} files)",
                    "stderr": "",
                    "exit_code": 0,
                }
            else:
                resolved.unlink()
                log.info("delete_tool.deleted", path=path, type="file")
                return {"stdout": f"Deleted file: {path}", "stderr": "", "exit_code": 0}
        except Exception as e:
            log.error("delete_tool.error", path=path, error=str(e))
            return {"stdout": f"Unexpected error: {e}", "stderr": str(e), "exit_code": 1}
