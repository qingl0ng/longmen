"""SQL query tool — run queries against a configured database."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog

from .base import BaseTool

log = structlog.get_logger(__name__)

_DEFAULT_LIMIT = 50
_MAX_LIMIT = 500

# SQL statements that modify data
_WRITE_PREFIXES = ("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "TRUNCATE", "REPLACE")
_DESTRUCTIVE_PATTERNS = [
    re.compile(r"^\s*DROP\s+TABLE\b", re.IGNORECASE),
    re.compile(r"^\s*TRUNCATE\b", re.IGNORECASE),
    re.compile(r"^\s*DELETE\s+FROM\s+\w+\s*$", re.IGNORECASE),  # DELETE without WHERE
    re.compile(r"^\s*DELETE\s+FROM\s+\w+\s*;?\s*$", re.IGNORECASE),
]


def _classify_query(query: str) -> str:
    """Return 'read', 'write', or 'destructive'."""
    stripped = query.strip().upper()
    read_prefixes = ("SELECT", "SHOW", "DESCRIBE", "EXPLAIN", "PRAGMA", "WITH")
    for prefix in read_prefixes:
        if stripped.startswith(prefix):
            return "read"
    for pattern in _DESTRUCTIVE_PATTERNS:
        if pattern.match(query):
            return "destructive"
    for prefix in _WRITE_PREFIXES:
        if stripped.startswith(prefix):
            return "write"
    return "write"  # Unknown — treat as write


def _load_db_config(root_path: str) -> dict[str, Any] | None:
    """Load [database] config from project.toml in root_path."""
    import tomllib

    config_path = Path(root_path) / "project.toml"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        return data.get("database")
    except Exception as e:
        log.error("sql.config_load_error", error=str(e))
        return None


def _format_table(columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    """Format query results as an ASCII table."""
    if not rows:
        return "(no rows)"
    col_widths: list[int] = [len(c) for c in columns]
    str_rows: list[list[str]] = []
    for row in rows:
        str_row: list[str] = [str(v) if v is not None else "NULL" for v in row]
        str_rows.append(str_row)
        for i, cell in enumerate(str_row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(r: list[str]) -> str:
        return "  " + "  |  ".join(v.ljust(col_widths[i]) for i, v in enumerate(r))

    sep = "  " + "--+--".join("-" * w for w in col_widths)
    lines: list[str] = [fmt_row(columns), sep]
    for str_row in str_rows:
        lines.append(fmt_row(str_row))
    return "\n".join(lines)


async def _execute_sqlite(
    connection_string: str,
    query: str,
    limit: int,
) -> dict[str, Any]:
    import sqlite3

    # Parse path from "sqlite:///path/to/db.sqlite3" or plain path
    db_path = connection_string
    for prefix in ("sqlite:///", "sqlite://"):
        if db_path.lower().startswith(prefix):
            db_path = db_path[len(prefix) :]
            break

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Get total count for SELECT
        query_class = _classify_query(query)
        total_rows: int | None = None

        if query_class == "read":
            # Count total
            count_q = f"SELECT COUNT(*) FROM ({query})"
            try:
                cur.execute(count_q)
                row = cur.fetchone()
                total_rows = row[0] if row else None
            except sqlite3.Error:
                total_rows = None

            cur.execute(query)
            rows = cur.fetchmany(limit)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            conn.close()

            table = _format_table(columns, [tuple(r) for r in rows])
            actual = len(rows)
            count_info = (
                f"{actual} rows (of {total_rows} total)"
                if total_rows is not None and total_rows > actual
                else f"{actual} rows"
            )
            col_count = len(columns)
            header = f"Query: {query}\nResult: {count_info}, {col_count} columns\n"
            return {
                "stdout": header + "\n" + table,
                "stderr": "",
                "exit_code": 0,
            }
        else:
            cur.execute(query)
            affected = cur.rowcount
            conn.commit()
            conn.close()
            return {
                "stdout": f"Query executed. Rows affected: {affected}",
                "stderr": "",
                "exit_code": 0,
            }
    except sqlite3.Error as e:
        return {"stdout": "", "stderr": str(e), "exit_code": 1}


async def _execute_postgresql(
    connection_string: str,
    query: str,
    limit: int,
) -> dict[str, Any]:
    try:
        import psycopg
    except ImportError:
        return {
            "stdout": "",
            "stderr": "PostgreSQL driver not installed. Run: pip install psycopg[binary]",
            "exit_code": 1,
        }

    query_class = _classify_query(query)
    try:
        async with (
            await psycopg.AsyncConnection.connect(connection_string) as conn,
            conn.cursor() as cur,
        ):
            total_rows: int | None = None
            if query_class == "read":
                try:
                    await cur.execute(f"SELECT COUNT(*) FROM ({query}) AS __count")
                    row = await cur.fetchone()
                    total_rows = row[0] if row else None
                except Exception:
                    total_rows = None

                await cur.execute(query + f" LIMIT {limit}")
                rows = await cur.fetchall()
                columns = [desc[0] for desc in cur.description] if cur.description else []
                table = _format_table(list(columns), [tuple(r) for r in rows])
                actual = len(rows)
                count_info = (
                    f"{actual} rows (of {total_rows} total)"
                    if total_rows is not None and total_rows > actual
                    else f"{actual} rows"
                )
                header = f"Query: {query}\nResult: {count_info}, {len(columns)} columns\n"
                return {"stdout": header + "\n" + table, "stderr": "", "exit_code": 0}
            else:
                await cur.execute(query)
                await conn.commit()
                return {
                    "stdout": f"Query executed. Rows affected: {cur.rowcount}",
                    "stderr": "",
                    "exit_code": 0,
                }
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "exit_code": 1}


async def _execute_mysql(
    connection_string: str,
    query: str,
    limit: int,
) -> dict[str, Any]:
    try:
        import pymysql
        import pymysql.cursors
    except ImportError:
        return {
            "stdout": "",
            "stderr": "MySQL driver not installed. Run: pip install pymysql",
            "exit_code": 1,
        }

    # Parse mysql://user:pass@host:port/dbname
    import urllib.parse

    parsed = urllib.parse.urlparse(connection_string)
    query_class = _classify_query(query)
    try:
        conn = pymysql.connect(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            user=parsed.username or "",
            password=parsed.password or "",
            database=parsed.path.lstrip("/") if parsed.path else "",
            cursorclass=pymysql.cursors.DictCursor,
        )
        with conn, conn.cursor() as cur:
            total_rows: int | None = None
            if query_class == "read":
                try:
                    cur.execute(f"SELECT COUNT(*) as c FROM ({query}) AS __count")
                    row = cur.fetchone()
                    total_rows = row["c"] if row else None
                except Exception:
                    total_rows = None

                cur.execute(query + f" LIMIT {limit}")
                rows = cur.fetchall()
                if rows:
                    columns = list(rows[0].keys())
                    row_tuples = [tuple(r.values()) for r in rows]
                else:
                    columns, row_tuples = [], []
                table = _format_table(columns, row_tuples)
                actual = len(rows)
                count_info = (
                    f"{actual} rows (of {total_rows} total)"
                    if total_rows is not None and total_rows > actual
                    else f"{actual} rows"
                )
                header = f"Query: {query}\nResult: {count_info}, {len(columns)} columns\n"
                return {"stdout": header + "\n" + table, "stderr": "", "exit_code": 0}
            else:
                cur.execute(query)
                conn.commit()
                return {
                    "stdout": f"Query executed. Rows affected: {cur.rowcount}",
                    "stderr": "",
                    "exit_code": 0,
                }
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "exit_code": 1}


class SQLQueryTool(BaseTool):
    name = "sql_query"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "sql_query",
                "description": (
                    "Run a SQL query against the project's configured database and return results "
                    "as a formatted table. SELECT/SHOW/DESCRIBE/EXPLAIN run automatically. "
                    "INSERT/UPDATE/DELETE/CREATE/ALTER/DROP require user approval. "
                    "DROP TABLE, TRUNCATE, and DELETE without WHERE are flagged as destructive."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "SQL statement to execute. No trailing semicolon needed. "
                                'Examples: "SELECT id, email FROM users WHERE active = true", '
                                '"DESCRIBE orders".'
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                f"Max rows to return for SELECT queries. "
                                f"Default: {_DEFAULT_LIMIT}, max: {_MAX_LIMIT}. "
                                "Total row count is always shown even when truncated."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(  # type: ignore[override]
        self,
        root_path: str,
        query: str,
        limit: int = _DEFAULT_LIMIT,
        **kwargs: Any,
    ) -> dict[str, Any]:
        limit = max(1, min(limit, _MAX_LIMIT))

        db_config = _load_db_config(root_path)
        if not db_config:
            return {
                "stdout": "",
                "stderr": (
                    "No database configured. "
                    "Add a [database] section to project.toml in the project root:\n\n"
                    "[database]\n"
                    'type = "sqlite"  # or "postgresql" or "mysql"\n'
                    'connection_string = "sqlite:///path/to/db.sqlite3"\n'
                ),
                "exit_code": 1,
            }

        db_type = db_config.get("type", "").lower()
        connection_string = db_config.get("connection_string", "")
        if not connection_string:
            return {
                "stdout": "",
                "stderr": "database.connection_string is empty in project.toml",
                "exit_code": 1,
            }

        query_class = _classify_query(query)
        log.info("sql_query.executing", db_type=db_type, query_class=query_class)

        if db_type == "sqlite":
            return await _execute_sqlite(connection_string, query, limit)
        elif db_type in ("postgresql", "postgres"):
            return await _execute_postgresql(connection_string, query, limit)
        elif db_type == "mysql":
            return await _execute_mysql(connection_string, query, limit)
        else:
            return {
                "stdout": "",
                "stderr": (
                    f"Unsupported database type: {db_type!r}."
                    " Use 'sqlite', 'postgresql', or 'mysql'."
                ),
                "exit_code": 1,
            }
