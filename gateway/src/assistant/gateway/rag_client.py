"""RAG service client — thin httpx wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import structlog

log = structlog.get_logger(__name__)


@dataclass
class RAGChunk:
    text: str
    score: float
    token_count: int
    collection: str
    document: str
    path: str
    size: str
    page: int | None
    section: str | None
    chunk_index: int


@dataclass
class RAGSearchResult:
    query: str
    results: list[RAGChunk]
    total_results: int
    model: str


@dataclass
class RAGCollectionInfo:
    name: str
    description: str
    document_count: int
    chunk_count: int
    compatible: bool


class RAGClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def search(
        self,
        query: str,
        collections: list[str],
        top_k: int = 10,
    ) -> RAGSearchResult | None:
        """POST /search — returns None on any failure."""
        try:
            resp = await self._client.post(
                "/search",
                json={"query": query, "collections": collections, "top_k": top_k},
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.warning("rag_client.search_failed", error=str(exc))
            return None

        if resp.status_code != 200:
            log.error("rag_client.search_bad_status", status=resp.status_code)
            return None

        try:
            data = resp.json()
            chunks = [
                RAGChunk(
                    text=r["text"],
                    score=r["score"],
                    token_count=r["token_count"],
                    collection=r["source"]["collection"],
                    document=r["source"]["document"],
                    path=r["source"]["path"],
                    size=r["source"]["size"],
                    page=r["source"].get("page"),
                    section=r["source"].get("section"),
                    chunk_index=r["source"]["chunk_index"],
                )
                for r in data.get("results", [])
            ]
            return RAGSearchResult(
                query=query,
                results=chunks,
                total_results=data.get("total_results", len(chunks)),
                model=data.get("model", ""),
            )
        except Exception as exc:
            log.error("rag_client.search_parse_error", error=str(exc))
            return None

    async def list_collections(self) -> list[RAGCollectionInfo] | None:
        """GET /collections — returns None if unavailable."""
        try:
            resp = await self._client.get("/collections")
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.warning("rag_client.list_collections_failed", error=str(exc))
            return None

        if resp.status_code != 200:
            log.error("rag_client.list_collections_bad_status", status=resp.status_code)
            return None

        try:
            data = resp.json()
            return [
                RAGCollectionInfo(
                    name=c["name"],
                    description=c.get("description", ""),
                    document_count=c.get("document_count", 0),
                    chunk_count=c.get("chunk_count", 0),
                    compatible=c.get("compatible", True),
                )
                for c in data
            ]
        except Exception as exc:
            log.error("rag_client.list_collections_parse_error", error=str(exc))
            return None

    async def health(self) -> bool:
        """GET /status — returns True if 200, False otherwise."""
        try:
            resp = await self._client.get("/status")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
