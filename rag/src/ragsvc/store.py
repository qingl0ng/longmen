"""Qdrant vector store wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

log = structlog.get_logger(__name__)


@dataclass
class StoredChunk:
    id: str
    text: str
    score: float
    token_count: int
    collection: str
    document_id: str
    document_name: str
    document_path: str
    document_size: int  # bytes
    chunk_index: int
    heading_hierarchy: list[str] = field(default_factory=list)
    page: int | None = None


class VectorStore:
    """Wraps the Qdrant client for vector storage and search."""

    def __init__(self, data_dir: Path, embedding_dimension: int) -> None:
        qdrant_path = data_dir / "qdrant"
        qdrant_path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(qdrant_path))
        self._dimension = embedding_dimension
        log.info("vector_store_opened", path=str(qdrant_path))

    def ensure_collection(self, name: str, dimension: int) -> None:
        """Create a Qdrant collection if it doesn't exist."""
        if self._client.collection_exists(name):
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=dimension,
                distance=qmodels.Distance.DOT,
            ),
        )
        log.info("collection_created", name=name, dimension=dimension)

    def delete_collection(self, name: str) -> None:
        """Delete a Qdrant collection and all its data."""
        if self._client.collection_exists(name):
            self._client.delete_collection(name)
            log.info("collection_deleted", name=name)

    def upsert_chunks(
        self,
        collection: str,
        chunks: list[StoredChunk],
        vectors: list[list[float]],
    ) -> None:
        """Insert or update chunks with their embedding vectors."""
        if not chunks:
            return
        points = [
            qmodels.PointStruct(
                id=_chunk_id_to_int(chunk.id),
                vector=vector,
                payload={
                    "id": chunk.id,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "collection": chunk.collection,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "document_path": chunk.document_path,
                    "document_size": chunk.document_size,
                    "chunk_index": chunk.chunk_index,
                    "heading_hierarchy": chunk.heading_hierarchy,
                    "page": chunk.page,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=collection, points=points)

    def delete_by_document(self, collection: str, document_id: str) -> None:
        """Delete all chunks belonging to a document."""
        self._client.delete(
            collection_name=collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        min_score: float,
    ) -> list[StoredChunk]:
        """Search for similar chunks. Returns results sorted by score descending."""
        if not self._client.collection_exists(collection):
            return []
        response = self._client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            score_threshold=min_score,
            with_payload=True,
        )
        return [_result_to_stored_chunk(r) for r in response.points]

    def search_multi(
        self,
        collections: list[str],
        query_vector: list[float],
        top_k: int,
        min_score: float,
    ) -> list[StoredChunk]:
        """Search across multiple collections. Merges and re-ranks results by score."""
        all_results: list[StoredChunk] = []
        for coll in collections:
            results = self.search(coll, query_vector, top_k, min_score)
            all_results.extend(results)
        # Re-sort by score descending and truncate
        all_results.sort(key=lambda c: c.score, reverse=True)
        return all_results[:top_k]

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists in Qdrant."""
        return self._client.collection_exists(name)

    def collection_stats(self, name: str) -> dict:
        """Return chunk count and other stats for a collection."""
        if not self._client.collection_exists(name):
            return {"chunk_count": 0}
        info = self._client.get_collection(name)
        return {
            "chunk_count": info.points_count or 0,
        }


def _chunk_id_to_int(chunk_id: str) -> int:
    """Convert a string chunk ID to a stable integer for Qdrant."""
    import hashlib
    h = hashlib.sha256(chunk_id.encode()).digest()
    # Use first 8 bytes as unsigned int, keep within safe range
    val = int.from_bytes(h[:8], "big")
    # Qdrant uses u64; keep as-is (Python int handles it)
    return val


def _result_to_stored_chunk(result: qmodels.ScoredPoint) -> StoredChunk:
    """Convert a Qdrant ScoredPoint to a StoredChunk."""
    payload = result.payload or {}
    return StoredChunk(
        id=payload.get("id", ""),
        text=payload.get("text", ""),
        score=result.score,
        token_count=payload.get("token_count", 0),
        collection=payload.get("collection", ""),
        document_id=payload.get("document_id", ""),
        document_name=payload.get("document_name", ""),
        document_path=payload.get("document_path", ""),
        document_size=payload.get("document_size", 0),
        chunk_index=payload.get("chunk_index", 0),
        heading_hierarchy=payload.get("heading_hierarchy", []),
        page=payload.get("page"),
    )
