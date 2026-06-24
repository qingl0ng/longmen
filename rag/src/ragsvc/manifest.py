"""SQLite manifest tracking indexed files."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    indexed_at REAL NOT NULL,
    chunk_count INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    UNIQUE(collection, file_path)
);

CREATE INDEX IF NOT EXISTS idx_collection ON documents(collection);
"""


def make_document_id(collection: str, file_path: str) -> str:
    """Deterministic document ID from collection name + relative file path."""
    return hashlib.sha256(f"{collection}:{file_path}".encode()).hexdigest()[:16]


@dataclass
class ManifestEntry:
    document_id: str
    collection: str
    file_path: str
    file_hash: str
    indexed_at: float
    chunk_count: int
    token_count: int
    embedding_model: str


class Manifest:
    """SQLite database tracking which files have been indexed."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.executescript(_CREATE_TABLE_SQL)
        self._conn.commit()
        log.info("manifest_opened", path=str(db_path))

    def get(self, collection: str, file_path: str) -> ManifestEntry | None:
        """Look up a document by collection + file path."""
        cur = self._conn.execute(
            "SELECT document_id, collection, file_path, file_hash, indexed_at, "
            "chunk_count, token_count, embedding_model "
            "FROM documents WHERE collection = ? AND file_path = ?",
            (collection, file_path),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return ManifestEntry(*row)

    def upsert(self, entry: ManifestEntry) -> None:
        """Insert or update a manifest entry."""
        self._conn.execute(
            "INSERT INTO documents "
            "(document_id, collection, file_path, file_hash, "
            "indexed_at, chunk_count, token_count, embedding_model) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(document_id) DO UPDATE SET "
            "file_hash=excluded.file_hash, indexed_at=excluded.indexed_at, "
            "chunk_count=excluded.chunk_count, token_count=excluded.token_count, "
            "embedding_model=excluded.embedding_model",
            (
                entry.document_id,
                entry.collection,
                entry.file_path,
                entry.file_hash,
                entry.indexed_at,
                entry.chunk_count,
                entry.token_count,
                entry.embedding_model,
            ),
        )
        self._conn.commit()

    def delete(self, document_id: str) -> None:
        """Remove a manifest entry."""
        self._conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        self._conn.commit()

    def delete_collection(self, collection: str) -> None:
        """Remove all entries for a collection."""
        self._conn.execute("DELETE FROM documents WHERE collection = ?", (collection,))
        self._conn.commit()

    def list_collection(self, collection: str) -> list[ManifestEntry]:
        """List all entries in a collection."""
        cur = self._conn.execute(
            "SELECT document_id, collection, file_path, file_hash, indexed_at, "
            "chunk_count, token_count, embedding_model "
            "FROM documents WHERE collection = ? ORDER BY file_path",
            (collection,),
        )
        return [ManifestEntry(*row) for row in cur.fetchall()]

    def get_embedding_models(self) -> set[str]:
        """Return all distinct embedding models used across the manifest."""
        cur = self._conn.execute("SELECT DISTINCT embedding_model FROM documents")
        return {row[0] for row in cur.fetchall()}

    def close(self) -> None:
        self._conn.close()
