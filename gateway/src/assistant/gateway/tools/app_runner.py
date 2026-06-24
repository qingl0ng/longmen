"""Application runner — starts a process, captures output, handles long-running servers."""

from __future__ import annotations

import asyncio
import signal
import time
from pathlib import Path
from typing import Any

import structlog

from .base import BaseTool

log = structlog.get_logger(__name__)

_MAX_OUTPUT_LINES = 200
_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 300
_SIGKILL_GRACE = 5  # seconds between SIGTERM and SIGKILL

# Default commands tried when no command is given (in order)
_DEFAULT_COMMANDS = [
    "python main.py",
    "python app.py",
    "npm start",
    "cargo run",
    "go run .",
]


async def _try_command_exists(root: Path, cmd: str) -> bool:
    """Check if the first file referenced in the command exists."""
    # For "python file.py" or "go run .", verify the file/dir
    parts = cmd.split()
    if len(parts) >= 2 and parts[0] in ("python", "python3"):
        return (root / parts[1]).exists()
    if cmd == "go run .":
        return any(root.glob("*.go"))
    if cmd == "cargo run":
        return (root / "Cargo.toml").exists()
    if cmd == "npm start":
        return (root / "package.json").exists()
    return False


class AppRunnerTool(BaseTool):
    name = "run_app"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "run_app",
                "description": (
                    "Run a script or server and capture output. "
                    "Handles long-running processes: kills after `timeout` seconds and returns "
                    "captured output. Use `wait_for` to stop early when a ready message appears "
                    '(e.g. "Listening on port"). Use for running the app, not for builds or tests.'
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "Command to run. If omitted, tries: "
                                "python main.py, python app.py, npm start, cargo run, go run ."
                            ),
                        },
                        "timeout": {
                            "type": "integer",
                            "description": (
                                "Seconds before killing the process. Default: 30, max: 300. "
                                "Process is returned immediately if it exits before timeout."
                            ),
                        },
                        "wait_for": {
                            "type": "string",
                            "description": (
                                "Return immediately when this substring appears in stdout/stderr "
                                'instead of waiting for timeout. Example: "Listening on port", '
                                '"Application startup complete".'
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }

    async def execute(
        self,
        root_path: str,
        command: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        wait_for: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        root = Path(root_path).expanduser().resolve()
        if not root.exists():
            return {
                "stdout": "",
                "stderr": f"root_path does not exist: {root_path}",
                "exit_code": 1,
            }

        # Clamp timeout
        timeout = max(1, min(timeout, _MAX_TIMEOUT))

        # Resolve command
        if not command:
            for candidate in _DEFAULT_COMMANDS:
                if await _try_command_exists(root, candidate):
                    command = candidate
                    break
            if not command:
                return {
                    "stdout": "",
                    "stderr": (
                        "No command provided and no default entry point found "
                        "(tried: python main.py, python app.py, npm start, cargo run, go run .)"
                    ),
                    "exit_code": 1,
                }

        start = time.monotonic()
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        exit_code: int | None = None
        killed = False
        wait_for_hit = False

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(root),
            )
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "exit_code": 1}

        async def _read_stream(
            stream: asyncio.StreamReader,
            lines_list: list[str],
        ) -> None:
            nonlocal wait_for_hit
            while True:
                try:
                    raw = await asyncio.wait_for(stream.readline(), timeout=1.0)
                except TimeoutError:
                    # Check if process is still running
                    if proc.returncode is not None:
                        break
                    continue
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if len(lines_list) < _MAX_OUTPUT_LINES:
                    lines_list.append(line)
                if wait_for and wait_for in line:
                    wait_for_hit = True

        async def _run() -> None:
            nonlocal exit_code, killed, wait_for_hit
            stdout_task = asyncio.create_task(_read_stream(proc.stdout, stdout_lines))  # type: ignore[arg-type]
            stderr_task = asyncio.create_task(_read_stream(proc.stderr, stderr_lines))  # type: ignore[arg-type]

            deadline = asyncio.get_event_loop().time() + timeout
            while True:
                # Check if process exited
                if proc.returncode is not None:
                    exit_code = proc.returncode
                    break
                # Check wait_for hit
                if wait_for_hit:
                    break
                # Check timeout
                if asyncio.get_event_loop().time() >= deadline:
                    break
                await asyncio.sleep(0.1)

            if proc.returncode is None:
                # Kill the process
                killed = not wait_for_hit
                try:
                    proc.send_signal(signal.SIGTERM)
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=_SIGKILL_GRACE)
                    except TimeoutError:
                        proc.kill()
                        await proc.wait()
                except ProcessLookupError:
                    pass
                exit_code = proc.returncode

            # Let readers finish
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            if exit_code is None:
                exit_code = proc.returncode if proc.returncode is not None else -1

        try:
            await _run()
        except Exception as e:
            log.error("app_runner.error", command=command, error=str(e))
            return {"stdout": "", "stderr": str(e), "exit_code": 1}

        duration_s = time.monotonic() - start
        exit_code = exit_code if exit_code is not None else -1

        # Build output string
        parts: list[str] = []
        if wait_for_hit:
            parts.append(
                f'Process still running — wait_for "{wait_for}" matched (after {duration_s:.1f}s)'
            )
        elif killed:
            parts.append(f"Process killed after {duration_s:.1f}s (timeout)")
        else:
            parts.append(f"Process exited with code {exit_code} (ran for {duration_s:.1f}s)")

        stderr_text = "\n".join(stderr_lines)
        total_stdout = len(stdout_lines)

        parts.append(f"\nstdout ({total_stdout} lines):")
        if stdout_lines:
            parts.append("\n".join(f"  {line}" for line in stdout_lines))
        else:
            parts.append("  (empty)")

        parts.append(f"\nstderr: {'(empty)' if not stderr_lines else ''}")
        if stderr_lines:
            parts.append("\n".join(f"  {line}" for line in stderr_lines))

        return {
            "stdout": "\n".join(parts),
            "stderr": stderr_text,
            "exit_code": exit_code if not wait_for_hit else 0,
            "duration_ms": int(duration_s * 1000),
            "killed": killed,
            "wait_for_hit": wait_for_hit,
        }
