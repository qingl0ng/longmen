"""Shell command execution (scoped to project root_path)."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any

import structlog

from .base import BaseTool

# Best-effort pattern to catch conventional shell deletion commands.
# Not adversarially secure — the real safety boundary is delete_tool's _safe_path.
# Covers: rm, rmdir, find -delete, find -exec rm, unlink, shred, trash variants.
_DELETION_PATTERN = re.compile(
    r"""
    (?<![a-zA-Z0-9_])   # not preceded by an identifier char (avoid "fromdir" etc.)
    (?:
        rm\b |
        rmdir\b |
        unlink\b |
        shred\b |
        trash\b |
        trash-put\b |
        gio\s+trash\b |
        find\b[^;]*-delete\b |
        find\b[^;]*-exec\s+rm\b
    )
    """,
    re.VERBOSE,
)

log = structlog.get_logger(__name__)


class ShellTool(BaseTool):
    name = "shell"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Execute a shell command in the project directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds. Defaults to 120.",
                        },
                    },
                    "required": ["command"],
                },
            },
        }

    async def execute(  # type: ignore[override]
        self,
        root_path: str,
        command: str,
        timeout: int = 120,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Runs command in root_path. Captures stdout/stderr/exit_code/duration_ms."""
        if _DELETION_PATTERN.search(command):
            return {
                "stdout": (
                    "Deletion via shell commands is not allowed. "
                    "Use the 'delete_tool' tool instead — it enforces project path "
                    "sandboxing and prevents accidental deletion of files outside the "
                    "project. Pass 'recursive: true' to delete a non-empty directory."
                ),
                "stderr": "",
                "exit_code": 1,
                "duration_ms": 0,
                "truncated": False,
            }

        root_resolved = Path(root_path).expanduser().resolve()
        if not root_resolved.exists():
            return {
                "stdout": "",
                "stderr": f"root_path does not exist: {root_path}",
                "exit_code": 1,
                "duration_ms": 0,
                "truncated": False,
            }

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(root_resolved),
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                duration_ms = int((time.monotonic() - start) * 1000)
                return {
                    "stdout": "",
                    "stderr": f"Command timed out after {timeout}s",
                    "exit_code": -1,
                    "duration_ms": duration_ms,
                    "truncated": False,
                }

            duration_ms = int((time.monotonic() - start) * 1000)
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = proc.returncode if proc.returncode is not None else -1

            # Truncate large outputs
            max_size = 100_000
            truncated = False
            if len(stdout) > max_size:
                stdout = stdout[:max_size] + "\n... [truncated]"
                truncated = True
            if len(stderr) > max_size:
                stderr = stderr[:max_size] + "\n... [truncated]"
                truncated = True

            log.info(
                "shell.executed",
                command=command,
                exit_code=exit_code,
                duration_ms=duration_ms,
            )
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "duration_ms": duration_ms,
                "truncated": truncated,
            }
        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            log.error("shell.error", command=command, error=str(e))
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": 1,
                "duration_ms": duration_ms,
                "truncated": False,
            }
