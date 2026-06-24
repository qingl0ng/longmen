"""SQL DDL symbol parser — extracts tables, views, functions, indexes, triggers."""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_CREATE_RE = re.compile(
    r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?"
    r"(TABLE|VIEW|FUNCTION|PROCEDURE|INDEX\s+\w+\s+ON|TRIGGER)\s+([\w.\"]+)",
    re.IGNORECASE,
)
_CREATE_INDEX_RE = re.compile(
    r"^\s*CREATE\s+(?:UNIQUE\s+)?INDEX\s+(\w+)\s+ON\s+(\w+)",
    re.IGNORECASE,
)
_ALTER_TABLE_RE = re.compile(
    r"^\s*ALTER\s+TABLE\s+([\w.\"]+)\s+(.+?)(?:\s*;)?\s*$",
    re.IGNORECASE,
)
_CREATE_TABLE_RE = re.compile(
    r"^\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w.\"]+)\s*\(",
    re.IGNORECASE,
)
_CREATE_VIEW_RE = re.compile(
    r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+([\w.\"]+)",
    re.IGNORECASE,
)
_CREATE_FUNC_RE = re.compile(
    r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE)\s+([\w.\"]+)\s*\(([^)]*)\)",
    re.IGNORECASE,
)
_CREATE_TRIGGER_RE = re.compile(
    r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\s+(\w+)",
    re.IGNORECASE,
)

# SQL keywords that start a column constraint (skip these when listing columns)
_COLUMN_SKIP_WORDS = frozenset(
    [
        "PRIMARY",
        "UNIQUE",
        "FOREIGN",
        "CHECK",
        "INDEX",
        "CONSTRAINT",
        "KEY",
        "REFERENCES",
        "NOT",
        "DEFAULT",
        "ON",
        "DELETE",
        "UPDATE",
        ")",
        "(",
        "COMMENT",
    ]
)

_COLUMN_RE = re.compile(r"^\s*(\w+)\s+([\w]+(?:\s*\([^)]*\))?)", re.IGNORECASE)

_RETURNS_RE = re.compile(r"\bRETURNS\b\s+(\S+)", re.IGNORECASE)


def parse(path: str) -> str:
    """Parse SQL file and return formatted DDL symbol listing."""
    try:
        source = Path(path).read_text(errors="replace")
    except OSError as e:
        return f"  [Error reading file: {e}]"

    lines = source.splitlines()
    parts: list[str] = []

    i = 0
    while i < len(lines):
        lineno = i + 1
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines, comments
        if not stripped or stripped.startswith("--") or stripped.startswith("/*"):
            i += 1
            continue

        # CREATE TABLE
        m = _CREATE_TABLE_RE.match(line)
        if m:
            table_name = m.group(1).strip('"')
            parts.append(f"  CREATE TABLE {table_name}  # line {lineno}")
            # Collect columns until closing paren
            i += 1
            while i < len(lines):
                col_line = lines[i].strip()
                if col_line.startswith(")") or col_line.startswith(");"):
                    i += 1
                    break
                cm = _COLUMN_RE.match(col_line)
                if cm:
                    first_word = cm.group(1).upper()
                    if first_word not in _COLUMN_SKIP_WORDS:
                        col_name = cm.group(1)
                        col_type = cm.group(2)
                        parts.append(f"    {col_name} {col_type}")
                i += 1
            parts.append("")
            continue

        # CREATE VIEW
        m = _CREATE_VIEW_RE.match(line)
        if m:
            name = m.group(1).strip('"')
            parts.append(f"  CREATE VIEW {name}  # line {lineno}")
            parts.append("")
            i += 1
            continue

        # CREATE FUNCTION / PROCEDURE
        m = _CREATE_FUNC_RE.match(line)
        if m:
            name = m.group(1).strip('"')
            params = m.group(2).strip()
            # Look ahead for RETURNS clause
            returns = ""
            for j in range(i + 1, min(i + 8, len(lines))):
                rm = _RETURNS_RE.search(lines[j])
                if rm:
                    returns = f"\n    RETURNS {rm.group(1)}"
                    break
                if lines[j].strip().startswith("$"):  # function body start
                    break
            keyword = "FUNCTION" if re.search(r"\bFUNCTION\b", line, re.IGNORECASE) else "PROCEDURE"
            parts.append(f"  CREATE {keyword} {name}({params}){returns}  # line {lineno}")
            parts.append("")
            i += 1
            continue

        # CREATE INDEX
        m = _CREATE_INDEX_RE.match(line)
        if m:
            idx_name = m.group(1)
            table_name = m.group(2)
            parts.append(f"  CREATE INDEX {idx_name} ON {table_name}  # line {lineno}")
            parts.append("")
            i += 1
            continue

        # CREATE TRIGGER
        m = _CREATE_TRIGGER_RE.match(line)
        if m:
            name = m.group(1)
            # Look ahead for ON clause
            on_table = ""
            for j in range(i + 1, min(i + 5, len(lines))):
                on_m = re.search(r"\bON\s+(\w+)", lines[j], re.IGNORECASE)
                if on_m:
                    on_table = f"\n    ON {on_m.group(1)}"
                    break
            parts.append(f"  CREATE TRIGGER {name}{on_table}  # line {lineno}")
            parts.append("")
            i += 1
            continue

        # ALTER TABLE
        m = _ALTER_TABLE_RE.match(line)
        if m:
            table_name = m.group(1).strip('"')
            action = m.group(2).strip().rstrip(";")
            if len(action) > 60:  # noqa: PLR2004
                action = action[:57] + "..."
            parts.append(f"  ALTER TABLE {table_name} {action}  # line {lineno}")
            parts.append("")
            i += 1
            continue

        i += 1

    return "\n".join(parts).rstrip() if parts else "  (no symbols found)"
