"""RAG search tool — schema and registration only."""

from __future__ import annotations

from typing import Any

from .base import BaseTool


class RAGSearchTool(BaseTool):
    name = "rag_search"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "rag_search",
                "description": (
                    "Search indexed reference materials (documentation, books, notes) "
                    "for information relevant to the current task. Use when you need "
                    "domain knowledge, API references, or pattern examples not in the "
                    "project source code. Formulate specific, targeted queries."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(self, root_path: str, **kwargs: object) -> dict[str, Any]:
        # Should not be called — execution is handled by the interceptor.
        return {
            "stdout": "rag_search interceptor not configured",
            "stderr": "",
            "exit_code": 1,
        }
