"""Build tool — wraps shell with build-specific error parsing."""

from __future__ import annotations

import re
from typing import Any

import structlog

from .base import BaseTool
from .project_detect import get_cached_project_type
from .shell import ShellTool

log = structlog.get_logger(__name__)

_shell = ShellTool()

# ── Error parsers ─────────────────────────────────────────────────────────────


def _parse_python_errors(output: str) -> list[dict[str, Any]]:
    """Extract structured errors from Python/pip/poetry build output."""
    errors: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    # Traceback: File "path", line N  →  next error line
    for m in re.finditer(r'File "([^"]+)", line (\d+)', output):
        file_path, line_str = m.group(1), m.group(2)
        line = int(line_str)
        key = (file_path, line)
        if key in seen:
            continue
        seen.add(key)
        rest = output[m.end() : m.end() + 300]
        next_lines = [ln.strip() for ln in rest.splitlines() if ln.strip()]
        msg = next_lines[0] if next_lines else ""
        errors.append({"file": file_path, "line": line, "message": msg, "severity": "error"})

    # Bare exception lines (SyntaxError etc.) not already captured
    exc_pattern = re.compile(
        r"^(SyntaxError|NameError|ImportError|ModuleNotFoundError|TypeError|"
        r"ValueError|AttributeError|IndentationError|KeyError|RuntimeError): (.+)$",
        re.MULTILINE,
    )
    for m in exc_pattern.finditer(output):
        msg = f"{m.group(1)}: {m.group(2).strip()}"
        errors.append({"file": "", "line": 0, "message": msg, "severity": "error"})

    return errors


def _parse_gcc_errors(output: str) -> list[dict[str, Any]]:
    """Parse GCC/Clang error format: file.cpp:42:10: error: message"""
    errors: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^(.+?):(\d+):(\d+):\s*(error|warning):\s*(.+)$",
        re.MULTILINE,
    )
    for m in pattern.finditer(output):
        errors.append(
            {
                "file": m.group(1).strip(),
                "line": int(m.group(2)),
                "col": int(m.group(3)),
                "message": m.group(5).strip(),
                "severity": m.group(4),
            }
        )
    return errors


def _parse_typescript_errors(output: str) -> list[dict[str, Any]]:
    """Parse tsc errors (file.ts(42,10): error TS2345: msg) and ESLint."""
    errors: list[dict[str, Any]] = []
    # tsc
    tsc = re.compile(
        r"^(.+?)\((\d+),(\d+)\):\s*(error|warning)\s+(TS\d+):\s*(.+)$",
        re.MULTILINE,
    )
    for m in tsc.finditer(output):
        errors.append(
            {
                "file": m.group(1).strip(),
                "line": int(m.group(2)),
                "col": int(m.group(3)),
                "message": f"{m.group(5)}: {m.group(6).strip()}",
                "severity": m.group(4),
            }
        )
    # ESLint: file.ts:42:10 error rule/message
    eslint = re.compile(
        r"^(.+?):(\d+):(\d+)\s+(error|warning)\s+(.+)$",
        re.MULTILINE,
    )
    for m in eslint.finditer(output):
        errors.append(
            {
                "file": m.group(1).strip(),
                "line": int(m.group(2)),
                "col": int(m.group(3)),
                "message": m.group(5).strip(),
                "severity": m.group(4),
            }
        )
    return errors


def _parse_rust_errors(output: str) -> list[dict[str, Any]]:
    """Parse cargo/rustc: error[E0308]: message \\n --> file.rs:42:10"""
    errors: list[dict[str, Any]] = []
    error_re = re.compile(r"^(error|warning)(?:\[([A-Z0-9]+)\])?:\s*(.+)$", re.MULTILINE)
    loc_re = re.compile(r"-->\s*(.+?):(\d+):(\d+)")
    for m in error_re.finditer(output):
        severity = m.group(1)
        code = m.group(2) or ""
        msg = f"[{code}] {m.group(3).strip()}" if code else m.group(3).strip()
        rest = output[m.end() : m.end() + 400]
        loc = loc_re.search(rest)
        if loc:
            errors.append(
                {
                    "file": loc.group(1).strip(),
                    "line": int(loc.group(2)),
                    "message": msg,
                    "severity": severity,
                }
            )
        else:
            errors.append({"file": "", "line": 0, "message": msg, "severity": severity})
    return errors


def _parse_go_errors(output: str) -> list[dict[str, Any]]:
    """Parse Go error format: ./file.go:42:10: message"""
    errors: list[dict[str, Any]] = []
    pattern = re.compile(r"^(\.?[./][^\s:]+\.go):(\d+):(\d+):\s*(.+)$", re.MULTILINE)
    for m in pattern.finditer(output):
        errors.append(
            {
                "file": m.group(1).strip(),
                "line": int(m.group(2)),
                "col": int(m.group(3)),
                "message": m.group(4).strip(),
                "severity": "error",
            }
        )
    return errors


def _parse_errors(
    project_types: list[str], output: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (errors, warnings) parsed from build output."""
    if any(t.startswith("python") for t in project_types):
        parsed = _parse_python_errors(output)
    elif any(t in ("cmake", "cmake-catch2", "make") for t in project_types):
        parsed = _parse_gcc_errors(output)
    elif any(
        t in ("npm-jest", "npm-vitest", "npm-mocha", "npm", "yarn", "pnpm") for t in project_types
    ):
        parsed = _parse_typescript_errors(output)
    elif "cargo" in project_types:
        parsed = _parse_rust_errors(output)
    elif "go" in project_types:
        parsed = _parse_go_errors(output)
    else:
        # Try GCC first (most common), then Python
        parsed = _parse_gcc_errors(output) or _parse_python_errors(output)

    errors = [e for e in parsed if e.get("severity") == "error"]
    warnings = [e for e in parsed if e.get("severity") == "warning"]
    return errors, warnings


def _fmt_entry(e: dict[str, Any]) -> str:
    file_ = e.get("file", "")
    line = e.get("line", 0)
    msg = e.get("message", "")
    if file_ and line:
        return f"  {file_}:{line}: {msg}"
    if file_:
        return f"  {file_}: {msg}"
    return f"  {msg}"


def _format_build_result(
    exit_code: int,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    raw_output: str,
    tail_lines: int = 50,
) -> str:
    status = "SUCCEEDED" if exit_code == 0 else f"FAILED (exit code {exit_code})"
    parts = [f"Build {status}"]

    if errors:
        parts.append(f"\nErrors ({len(errors)}):")
        parts.extend(_fmt_entry(e) for e in errors)

    if warnings:
        parts.append(f"\nWarnings ({len(warnings)}):")
        parts.extend(_fmt_entry(w) for w in warnings)

    lines = raw_output.splitlines()
    total = len(lines)
    tail = lines[-tail_lines:] if total > tail_lines else lines
    suffix = f" (truncated to last {tail_lines} lines)" if total > tail_lines else ""
    parts.append(f"\nFull output: {total} lines{suffix}")
    parts.append("\n".join(tail))
    return "\n".join(parts)


# ── Tool ──────────────────────────────────────────────────────────────────────


class BuildTool(BaseTool):
    name = "build"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "build",
                "description": (
                    "Build the project using the auto-detected build command. "
                    "Parses output and returns errors as structured file:line:message entries — "
                    "use these paths with read_file to inspect failures. "
                    "Prefer this over shell for builds. "
                    "Requires detect_project to have run first."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "Override the build command."
                                " Omit to use the auto-detected build_cmd."
                                ' Example: "cmake --build build --target mylib".'
                            ),
                        },
                        "clean": {
                            "type": "boolean",
                            "description": (
                                "Delete build artifacts before building. Default: false. "
                                "Use when stale artifacts may be causing incorrect errors."
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
        clean: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        pt = get_cached_project_type(root_path)
        project_types: list[str] = pt.types if pt else []

        if command:
            build_cmd = command
        elif pt and pt.build_cmd:
            build_cmd = pt.build_cmd
        else:
            return {
                "stdout": (
                    "No build command available. "
                    "Run detect_project first or provide a command override."
                ),
                "stderr": "",
                "exit_code": 1,
            }

        # Clean build artifacts if requested
        if clean and project_types:
            clean_cmds: dict[str, str] = {
                "cmake": "rm -rf build",
                "cmake-catch2": "rm -rf build",
                "cargo": "cargo clean",
                "go": "go clean ./...",
            }
            for ptype in project_types:
                if ptype in clean_cmds:
                    await _shell.execute(root_path, command=clean_cmds[ptype])
                    break

        result = await _shell.execute(root_path, command=build_cmd)
        combined = "\n".join(filter(None, [result.get("stdout", ""), result.get("stderr", "")]))
        errors, warnings = _parse_errors(project_types, combined)
        formatted = _format_build_result(
            exit_code=result.get("exit_code", 0),
            errors=errors,
            warnings=warnings,
            raw_output=combined,
        )
        return {
            "stdout": formatted,
            "stderr": "",
            "exit_code": result.get("exit_code", 0),
            "errors": errors,
            "warnings": warnings,
            "duration_ms": result.get("duration_ms", 0),
        }
