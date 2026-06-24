"""File writing tools — write_file (multi-mode), search_replace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from .base import BaseTool

log = structlog.get_logger(__name__)


def _file_stats(path: Path) -> str:
    """Return 'File now: N lines, ~X KB, ~N tokens'."""
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    kb = len(content.encode()) / 1024
    # Avoid importing count_tokens at module level to keep startup fast;
    # use char/4 fallback directly here for display purposes.
    from assistant.gateway.session import count_tokens

    tokens = count_tokens(content)
    return f"File now: {lines} lines, ~{kb:.1f} KB, ~{tokens} tokens"


class WriteFileTool(BaseTool):
    name = "write_file"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": (
                    "Write content to a file within the project. Supports five modes: "
                    "'create' (new file only — errors if the file already exists), "
                    "'overwrite' (replace the entire file; if file doesn't exist, creates it), "
                    "'append' (add content to end of file with a newline separator), "
                    "'insert' (insert content before a specific line; requires line_number), "
                    "'replace' (replace a range of lines; requires start_line and end_line). "
                    "Default mode is 'overwrite'. All paths must be relative to the project root. "
                    "Parent directories are created automatically in 'create' mode. "
                    "Errors: 'file already exists' (create on existing file), "
                    "'file not found' (append/insert/replace on missing file), "
                    "'path outside project root', 'start_line/end_line out of range'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "File path relative to project root."
                                " Must not start with '/' or contain '..'."
                            ),
                        },
                        "content": {
                            "type": "string",
                            "description": "The full text to write.",
                        },
                        "mode": {
                            "type": "string",
                            "description": (
                                "One of: 'create', 'overwrite', 'append', 'insert', 'replace'."
                                " Default: 'overwrite'. Use 'create' when making a new file;"
                                " use 'search_replace' instead of 'replace' when line numbers"
                                " may have shifted."
                            ),
                        },
                        "start_line": {
                            "type": "integer",
                            "description": (
                                "Required for mode='replace'."
                                " First line to replace, 1-based inclusive."
                            ),
                        },
                        "end_line": {
                            "type": "integer",
                            "description": (
                                "Required for mode='replace'."
                                " Last line to replace, 1-based inclusive."
                            ),
                        },
                        "line_number": {
                            "type": "integer",
                            "description": (
                                "Required for mode='insert'. Content is inserted before this line,"
                                " 1-based."
                            ),
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        }

    async def execute(  # type: ignore[override]
        self,
        root_path: str,
        path: str,
        content: str,
        mode: str = "overwrite",
        start_line: int | None = None,
        end_line: int | None = None,
        line_number: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            safe = self._safe_path(root_path, path)
            target = Path(safe)
        except ValueError as e:
            return {"error": str(e)}

        try:
            if mode == "create":
                if target.exists():
                    return {"error": f"file already exists: {path}"}
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                new_lines = content.count("\n") + (
                    1 if content and not content.endswith("\n") else 0
                )
                summary = f"Wrote to {path} (mode: create, {new_lines} new lines)"
                log.info("write_file.created", path=path)

            elif mode == "overwrite":
                if not target.exists():
                    # Auto-create the file if it doesn't exist
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                    new_lines = content.count("\n") + (
                        1 if content and not content.endswith("\n") else 0
                    )
                    summary = f"Wrote to {path} (mode: overwrite, created new file)"
                    log.info("write_file.overwritten", path=path, created=True)
                else:
                    target.write_text(content, encoding="utf-8")
                    summary = f"Wrote to {path} (mode: overwrite)"
                    log.info("write_file.overwritten", path=path)

            elif mode == "append":
                if not target.exists():
                    return {"error": f"file not found: {path}"}
                existing = target.read_text(encoding="utf-8")
                separator = "\n" if existing and not existing.endswith("\n") else ""
                target.write_text(existing + separator + content, encoding="utf-8")
                summary = f"Wrote to {path} (mode: append)"
                log.info("write_file.appended", path=path)

            elif mode == "insert":
                if not target.exists():
                    return {"error": f"file not found: {path}"}
                if line_number is None:
                    return {"error": "line_number is required for mode='insert'"}
                existing_lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
                n = len(existing_lines)
                ln = int(line_number)
                if ln < 1 or ln > n + 1:
                    return {"error": f"line_number {ln} out of range (file has {n} lines)"}
                insert_lines = content.splitlines(keepends=True)
                if insert_lines and not insert_lines[-1].endswith("\n"):
                    insert_lines[-1] += "\n"
                new_lines_list = existing_lines[: ln - 1] + insert_lines + existing_lines[ln - 1 :]
                target.write_text("".join(new_lines_list), encoding="utf-8")
                summary = (
                    f"Wrote to {path} (mode: insert, before line {ln}, +{len(insert_lines)} lines)"
                )
                log.info("write_file.inserted", path=path, line=ln)

            elif mode == "replace":
                if not target.exists():
                    return {"error": f"file not found: {path}"}
                if start_line is None or end_line is None:
                    return {"error": "start_line and end_line are required for mode='replace'"}
                sl, el = int(start_line), int(end_line)
                existing_lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
                n = len(existing_lines)
                if sl < 1 or el < sl or el > n:
                    return {
                        "error": f"start_line={sl}/end_line={el} out of range (file has {n} lines)"
                    }
                replace_lines = content.splitlines(keepends=True)
                if replace_lines and not replace_lines[-1].endswith("\n"):
                    replace_lines[-1] += "\n"
                new_lines_list = existing_lines[: sl - 1] + replace_lines + existing_lines[el:]
                target.write_text("".join(new_lines_list), encoding="utf-8")
                summary = (
                    f"Wrote to {path} (mode: replace, lines {sl}-{el}"
                    f" → {len(replace_lines)} new lines)"
                )
                log.info("write_file.replaced", path=path, start=sl, end=el)

            else:
                return {
                    "error": f"unknown mode: {mode!r}. Use create/overwrite/append/insert/replace"
                }

            stats = _file_stats(target)
            return {"stdout": f"{summary}\n{stats}"}

        except Exception as e:
            log.error("write_file.error", path=path, mode=mode, error=str(e))
            return {"error": str(e)}


class SearchReplaceTool(BaseTool):
    name = "search_replace"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_replace",
                "description": (
                    "Find an exact block of text in a file and replace it with new text."
                    " Prefer this over write_file mode='replace' when line numbers"
                    " may have shifted."
                    " The search text must match the file exactly"
                    " — including all whitespace and newlines. "
                    "Errors: 'search text not found in file', "
                    "'search text found N times, expected {count} — be more specific', "
                    "'path outside project root'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root.",
                        },
                        "search": {
                            "type": "string",
                            "description": (
                                "Exact text to find, including all whitespace and indentation. "
                                "Must match the file character-for-character."
                            ),
                        },
                        "replace": {
                            "type": "string",
                            "description": "Text to substitute in place of the search block.",
                        },
                        "count": {
                            "type": "integer",
                            "description": (
                                "Expected number of times search appears in the file. "
                                "Default: 1. Errors if actual count differs."
                            ),
                        },
                    },
                    "required": ["path", "search", "replace"],
                },
            },
        }

    async def execute(  # type: ignore[override]
        self,
        root_path: str,
        path: str,
        search: str,
        replace: str,
        count: int = 1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            safe = self._safe_path(root_path, path)
            target = Path(safe)
        except ValueError as e:
            return {"error": str(e)}

        if not target.exists():
            return {"error": f"file not found: {path}"}

        try:
            original = target.read_text(encoding="utf-8")
            actual_count = original.count(search)

            if actual_count == 0:
                return {"error": "search text not found in file"}
            if actual_count != int(count):
                return {
                    "error": (
                        f"search text found {actual_count} times, expected {count}. "
                        "Be more specific."
                    )
                }

            # Find line span of first occurrence for reporting
            start_char = original.index(search)
            start_line = original[:start_char].count("\n") + 1
            end_line = start_line + search.count("\n")
            replace_line_count = replace.count("\n") + (
                1 if replace and not replace.endswith("\n") else 0
            )

            updated = original.replace(search, replace, int(count))
            target.write_text(updated, encoding="utf-8")

            stats = _file_stats(target)
            summary = (
                f"Replaced {actual_count} occurrence(s) in {path} "
                f"(lines {start_line}-{end_line} → {replace_line_count} new lines)"
            )
            log.info("search_replace.done", path=path, occurrences=actual_count)
            return {"stdout": f"{summary}\n{stats}"}

        except Exception as e:
            log.error("search_replace.error", path=path, error=str(e))
            return {"error": str(e)}
