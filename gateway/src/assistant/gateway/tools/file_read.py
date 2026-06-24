"""File reading tools — read_file, list_dir, grep, tree, symbols."""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import structlog

from .base import BaseTool

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALWAYS_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)
_ALWAYS_EXCLUDE_FILE_PATS: tuple[str, ...] = ("*.pyc", "*.pyo")
_ALWAYS_EXCLUDE_DIR_SUFFIXES: tuple[str, ...] = (".egg-info",)

# Lines threshold above which read_file warns and shows only first 50 lines
_LARGE_FILE_THRESHOLD: int = 500


# ---------------------------------------------------------------------------
# .gitignore helpers
# ---------------------------------------------------------------------------


def _load_gitignore_patterns(root: Path) -> list[str]:
    """Return non-comment patterns from .gitignore at root."""
    gi = root / ".gitignore"
    if not gi.exists():
        return []
    return [
        ln.strip()
        for ln in gi.read_text(errors="replace").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def _is_gitignored(path: Path, root: Path, patterns: list[str]) -> bool:
    """True if path matches any .gitignore pattern (simplified subset)."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False

    rel_str = str(rel)
    name = path.name

    for pattern in patterns:
        # Trailing slash means directory-only; strip it and match both name and rel
        dir_only = pattern.endswith("/")
        pat = pattern.rstrip("/")
        anchored = pat.startswith("/")
        pat = pat.lstrip("/")

        if dir_only and not path.is_dir():
            continue

        if anchored:
            if fnmatch.fnmatch(rel_str, pat) or fnmatch.fnmatch(name, pat):
                return True
        else:
            if fnmatch.fnmatch(name, pat):
                return True
            if fnmatch.fnmatch(rel_str, pat):
                return True
            # Match any path component
            for part in rel.parts[:-1]:
                if fnmatch.fnmatch(part, pat):
                    return True
    return False


def _should_skip(path: Path, root: Path, gitignore: list[str]) -> bool:
    """True if path should be excluded from tree/grep traversal."""
    name = path.name
    if path.is_dir():
        if name in _ALWAYS_EXCLUDE_DIRS:
            return True
        if any(name.endswith(s) for s in _ALWAYS_EXCLUDE_DIR_SUFFIXES):
            return True
    for pat in _ALWAYS_EXCLUDE_FILE_PATS:
        if fnmatch.fnmatch(name, pat):
            return True
    return _is_gitignored(path, root, gitignore)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _human_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.1f} MB"


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def _rg_available() -> bool:
    try:
        subprocess.run(["rg", "--version"], capture_output=True, timeout=2, check=False)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Tree tool
# ---------------------------------------------------------------------------


class TreeTool(BaseTool):
    name = "tree"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "tree",
                "description": (
                    "List project directory structure with file sizes. Respects .gitignore."
                    " Use this first to understand the codebase layout before reading"
                    " specific files."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path relative to project root. Default: '.'",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Max directory depth to display. Default: 3",
                        },
                        "show_hidden": {
                            "type": "boolean",
                            "description": "Include hidden files (dotfiles). Default: false",
                        },
                    },
                    "required": [],
                },
            },
        }

    async def execute(
        self,
        root_path: str,
        path: str = ".",
        depth: int = 3,
        show_hidden: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            safe = self._safe_path(root_path, path)
        except ValueError as e:
            return {"error": str(e)}

        root = Path(safe)
        if not root.is_dir():
            return {"error": f"Not a directory: {path}"}

        gi_root = Path(root_path).resolve()
        gitignore = _load_gitignore_patterns(gi_root)

        lines: list[str] = []
        total_files = 0
        total_dirs = 0

        def _walk(dir_path: Path, prefix: str, current_depth: int) -> None:
            nonlocal total_files, total_dirs

            try:
                raw_entries = list(dir_path.iterdir())
            except PermissionError:
                return

            entries = sorted(
                (
                    e
                    for e in raw_entries
                    if (show_hidden or not e.name.startswith("."))
                    and not _should_skip(e, gi_root, gitignore)
                ),
                key=lambda p: (0 if p.is_dir() else 1, p.name.lower()),
            )

            for i, entry in enumerate(entries):
                is_last = i == len(entries) - 1
                connector = "└── " if is_last else "├── "
                child_prefix = prefix + ("    " if is_last else "│   ")

                if entry.is_dir():
                    total_dirs += 1
                    if current_depth >= depth:
                        # Collapsed: show summary
                        try:
                            children = [
                                c
                                for c in entry.iterdir()
                                if not _should_skip(c, gi_root, gitignore)
                                and (show_hidden or not c.name.startswith("."))
                            ]
                        except PermissionError:
                            children = []
                        file_cnt = sum(1 for c in children if c.is_file())
                        dir_cnt = sum(1 for c in children if c.is_dir())
                        total_files += file_cnt
                        total_size = sum(c.stat().st_size for c in children if c.is_file())
                        parts = [f"{file_cnt} file{'s' if file_cnt != 1 else ''}"]
                        if dir_cnt:
                            parts.append(f"{dir_cnt} dir{'s' if dir_cnt != 1 else ''}")
                        if total_size:
                            parts.append(f"{_human_size(total_size)} total")
                        lines.append(f"{prefix}{connector}{entry.name}/ ({', '.join(parts)})")
                    else:
                        lines.append(f"{prefix}{connector}{entry.name}/")
                        _walk(entry, child_prefix, current_depth + 1)
                else:
                    total_files += 1
                    try:
                        size_str = f" ({_human_size(entry.stat().st_size)})"
                    except OSError:
                        size_str = ""
                    lines.append(f"{prefix}{connector}{entry.name}{size_str}")

        _walk(root, "", 1)

        try:
            display = str(Path(safe).relative_to(gi_root))
            if display == ".":
                display = gi_root.name
        except ValueError:
            display = path

        header = (
            f"{display}/ "
            f"({total_files} file{'s' if total_files != 1 else ''}, "
            f"{total_dirs} dir{'s' if total_dirs != 1 else ''})"
        )
        output = header + "\n" + "\n".join(lines) if lines else header
        return {"tree": output, "path": path}


# ---------------------------------------------------------------------------
# Read file tool (upgraded: line ranges, token count, large-file guard)
# ---------------------------------------------------------------------------


class ReadFileTool(BaseTool):
    name = "read_file"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Read a file's contents. For large files, use start_line/end_line to read"
                    " specific sections. Use 'symbols' first to find the right line range."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root.",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "First line to read (1-based). Default: 1",
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "Last line to read (inclusive). Default: last line",
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
        start_line: int | None = None,
        end_line: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            safe = self._safe_path(root_path, path)
        except ValueError as e:
            return {"error": str(e)}

        try:
            raw = Path(safe).read_text(errors="replace")
        except FileNotFoundError:
            return {"error": f"File not found: {path}"}
        except Exception as e:
            log.error("read_file.error", path=path, error=str(e))
            return {"error": str(e)}

        all_lines = raw.splitlines()
        total_lines = len(all_lines)

        if start_line is None and end_line is None:
            # Full read
            if total_lines > _LARGE_FILE_THRESHOLD:
                preview = all_lines[:50]
                numbered = "\n".join(f"{i + 1:4d}│{ln}" for i, ln in enumerate(preview))
                tok = estimate_tokens(raw)
                return {
                    "content": numbered,
                    "path": path,
                    "total_lines": total_lines,
                    "lines_read": 50,
                    "tokens_estimated": tok,
                    "warning": (
                        f"File has {total_lines} lines (~{tok} tokens). "
                        "Showing first 50 lines."
                        " Use 'symbols' to find the right section, then read_file with"
                        " start_line/end_line."
                    ),
                }
            numbered = "\n".join(f"{i + 1:4d}│{ln}" for i, ln in enumerate(all_lines))
            tok = estimate_tokens(raw)
            return {
                "content": numbered,
                "path": path,
                "total_lines": total_lines,
                "lines_read": total_lines,
                "tokens_estimated": tok,
            }

        # Partial read
        s = max(0, (start_line or 1) - 1)  # convert to 0-indexed
        e_idx = min(total_lines, end_line if end_line is not None else total_lines)
        e_idx = max(s, e_idx)
        selected = all_lines[s:e_idx]
        numbered = "\n".join(f"{s + i + 1:4d}│{ln}" for i, ln in enumerate(selected))
        snippet = "\n".join(selected)
        tok = estimate_tokens(snippet)
        header = f"{path} (lines {s + 1}-{e_idx} of {total_lines}, ~{tok} tokens):"
        return {
            "content": header + "\n\n" + numbered,
            "path": path,
            "total_lines": total_lines,
            "lines_read": len(selected),
            "start_line": s + 1,
            "end_line": e_idx,
            "tokens_estimated": tok,
        }


# ---------------------------------------------------------------------------
# List directory tool (unchanged)
# ---------------------------------------------------------------------------


class ListDirTool(BaseTool):
    name = "list_dir"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List the contents of a directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Directory path relative to the project root. Defaults to '.'."
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
        path: str = ".",
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            safe = self._safe_path(root_path, path)
            entries = []
            for entry in sorted(Path(safe).iterdir()):
                entries.append(
                    {
                        "name": entry.name,
                        "type": "dir" if entry.is_dir() else "file",
                        "size": entry.stat().st_size if entry.is_file() else None,
                    }
                )
            return {"entries": entries, "path": path}
        except ValueError as e:
            return {"error": str(e)}
        except FileNotFoundError:
            return {"error": f"Directory not found: {path}"}
        except Exception as e:
            log.error("list_dir.error", path=path, error=str(e))
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# Grep tool (upgraded: context, max_results, file_type, literal, .gitignore)
# ---------------------------------------------------------------------------


class GrepTool(BaseTool):
    name = "grep"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "grep",
                "description": (
                    "Search for a text pattern across project files. Returns matching lines with"
                    " context. Use this to find where a function is called, where a variable is"
                    " used, or where an error message originates."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (regex by default)",
                        },
                        "path": {
                            "type": "string",
                            "description": (
                                "Directory or file to search. Default: '.' (project root)"
                            ),
                        },
                        "file_type": {
                            "type": "string",
                            "description": (
                                "Filter by file extension, e.g. 'py', 'toml', 'md'. Default: all"
                            ),
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max matches to return. Default: 20",
                        },
                        "literal": {
                            "type": "boolean",
                            "description": (
                                "Treat pattern as literal string, not regex. Default: false"
                            ),
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": (
                                "Lines of context before and after each match. Default: 2"
                            ),
                        },
                    },
                    "required": ["pattern"],
                },
            },
        }

    async def execute(  # type: ignore[override]
        self,
        root_path: str,
        pattern: str,
        path: str = ".",
        file_type: str | None = None,
        max_results: int = 20,
        literal: bool = False,
        context_lines: int = 2,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            safe = self._safe_path(root_path, path)
        except ValueError as e:
            return {"error": str(e)}

        root = Path(root_path).resolve()
        target = Path(safe)
        gitignore = _load_gitignore_patterns(root)

        if _rg_available():
            return self._grep_rg(
                pattern, target, root, file_type, max_results, literal, context_lines
            )
        return self._grep_python(
            pattern, target, root, file_type, max_results, literal, context_lines, gitignore
        )

    def _grep_rg(
        self,
        pattern: str,
        target: Path,
        root: Path,
        file_type: str | None,
        max_results: int,
        literal: bool,
        context_lines: int,
    ) -> dict[str, Any]:
        cmd = [
            "rg",
            "--line-number",
            f"--context={context_lines}",
            "--color=never",
            "--no-heading",
        ]
        if literal:
            cmd.append("--fixed-strings")
        if file_type:
            cmd.extend(["--glob", f"*.{file_type}"])
        cmd.extend(["--", pattern, str(target)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        except subprocess.TimeoutExpired:
            return {"error": "grep timed out", "pattern": pattern}
        except Exception as e:
            return {"error": str(e)}

        return self._format_rg_output(result.stdout, pattern, max_results, root)

    def _format_rg_output(
        self, raw: str, pattern: str, max_results: int, root: Path
    ) -> dict[str, Any]:
        root_prefix = str(root) + "/"
        raw = raw.replace(root_prefix, "")
        lines = raw.splitlines()

        total_matches = 0
        output_lines: list[str] = []
        current_block: list[str] = []
        shown = 0

        for line in lines:
            if line == "--":
                if current_block:
                    output_lines.extend(current_block)
                    output_lines.append("")
                    current_block = []
                continue

            # Match line: file:linenum:content
            m = re.match(r"^(.*?):(\d+):(.*)", line)
            if m:
                total_matches += 1
                if shown < max_results:
                    shown += 1
                    current_block.append(f"{m.group(1)}:{m.group(2)}:{m.group(3)}")
            else:
                # Context line: file-linenum-content
                cm = re.match(r"^(.*?)-(\d+)-(.*)", line)
                if cm and shown > 0 and shown <= max_results:
                    current_block.append(f"  {cm.group(2)}│  {cm.group(3)}")

        if current_block:
            output_lines.extend(current_block)

        header = f"Found {total_matches} match{'es' if total_matches != 1 else ''}"
        if total_matches > max_results:
            header += f" (showing {shown})"
        header += ":"

        return {
            "matches": header + "\n\n" + "\n".join(output_lines),
            "pattern": pattern,
            "total_matches": total_matches,
            "shown": shown,
        }

    def _grep_python(
        self,
        pattern: str,
        target: Path,
        root: Path,
        file_type: str | None,
        max_results: int,
        literal: bool,
        context_lines: int,
        gitignore: list[str],
    ) -> dict[str, Any]:
        if literal:
            compiled = re.compile(re.escape(pattern))
        else:
            try:
                compiled = re.compile(pattern)
            except re.error as e:
                return {"error": f"Invalid regex: {e}"}

        # Collect files to search
        files: list[Path] = []
        if target.is_file():
            files = [target]
        else:
            for dirpath, dirnames, filenames in os.walk(target):
                dp = Path(dirpath)
                dirnames[:] = sorted(
                    d
                    for d in dirnames
                    if not _should_skip(dp / d, root, gitignore) and not d.startswith(".")
                )
                for fname in sorted(filenames):
                    fp = dp / fname
                    if file_type and not fname.endswith(f".{file_type}"):
                        continue
                    if _should_skip(fp, root, gitignore):
                        continue
                    files.append(fp)

        # Find all matches
        all_matches: list[tuple[Path, int, list[str]]] = []  # (file, lineno, file_lines)
        for fp in files:
            try:
                flines = fp.read_text(errors="replace").splitlines()
            except Exception:
                continue
            for lineno, line in enumerate(flines, 1):
                if compiled.search(line):
                    all_matches.append((fp, lineno, flines))

        total = len(all_matches)
        shown_matches = all_matches[:max_results]

        output_parts: list[str] = []
        header = f"Found {total} match{'es' if total != 1 else ''}"
        if total > max_results:
            header += f" (showing {max_results})"
        header += ":"
        output_parts.append(header)
        output_parts.append("")

        for fp, lineno, flines in shown_matches:
            try:
                rel = str(fp.relative_to(root))
            except ValueError:
                rel = str(fp)
            output_parts.append(f"{rel}:{lineno}:{flines[lineno - 1]}")
            ctx_start = max(0, lineno - 1 - context_lines)
            ctx_end = min(len(flines), lineno + context_lines)
            for ci in range(ctx_start, ctx_end):
                cl = ci + 1
                marker = ">>" if cl == lineno else "  "
                output_parts.append(f"  {cl:4d}│{marker}  {flines[ci]}")
            output_parts.append("")

        return {
            "matches": "\n".join(output_parts),
            "pattern": pattern,
            "total_matches": total,
            "shown": len(shown_matches),
        }


# ---------------------------------------------------------------------------
# Symbols tool (AST-based Python, regex fallback for other languages)
# ---------------------------------------------------------------------------


class SymbolsTool(BaseTool):
    name = "symbols"

    # Dispatch table: file extension → dedicated parser function
    # Each parser takes a file path (str) and returns a formatted body string.
    _CPP_EXTS: frozenset[str] = frozenset({".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"})
    _TS_EXTS: frozenset[str] = frozenset({".ts", ".tsx"})

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "symbols",
                "description": (
                    "Extract function and class definitions from a source file with line numbers. "
                    "Uses AST for Python, dedicated parsers for C++/TypeScript/SQL, "
                    "tree-sitter for other supported languages, and regex as a final fallback. "
                    "Use this to understand a file's structure before reading specific sections."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root",
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
        **kwargs: Any,
    ) -> dict[str, Any]:
        from .symbol_parsers import cpp, python, regex_fallback, sql, treesitter, typescript

        try:
            safe = self._safe_path(root_path, path)
        except ValueError as e:
            return {"error": str(e)}

        if not Path(safe).exists():
            return {"error": f"File not found: {path}"}

        try:
            source = Path(safe).read_text(errors="replace")
        except Exception as e:
            return {"error": str(e)}

        total_lines = len(source.splitlines())
        size_bytes = len(source.encode())
        tok = estimate_tokens(source)
        header = f"{path} ({total_lines} lines, ~{_human_size(size_bytes)}, ~{tok} tokens):\n"

        ext = Path(path).suffix.lower()

        if ext == ".py":
            body = python.parse(source)
        elif ext in self._CPP_EXTS:
            body = cpp.parse(safe)
        elif ext in self._TS_EXTS:
            body = typescript.parse(safe)
        elif ext == ".sql":
            body = sql.parse(safe)
        else:
            # Try tree-sitter; fall back to regex if no grammar available
            ts_result = treesitter.try_parse(safe)
            body = ts_result if ts_result is not None else regex_fallback.parse(source)

        return {
            "symbols": header + "\n" + body,
            "path": path,
            "total_lines": total_lines,
            "tokens_estimated": tok,
        }
