"""Web search tool using Brave Search API."""

from __future__ import annotations

from typing import Any

import structlog

from .base import BaseTool
from .web_client import get_web_client

log = structlog.get_logger(__name__)


class WebSearchTool(BaseTool):
    """Web search tool that uses Brave Search API."""

    name = "web_search"

    def __init__(
        self,
        search_count: int = 5,
        brave_api_key: str = "",
        user_agent: str = (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
    ):
        super().__init__()
        self._search_count = search_count
        self._brave_api_key = brave_api_key
        self._user_agent = user_agent

    def schema(self) -> dict[str, Any]:
        """Return OpenAI-format function schema."""
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web using Brave Search. Returns titles, URLs, and snippets "
                    "for relevant pages. Use when you need information beyond your training data "
                    "— library recommendations, API docs, current best practices, "
                    "or researching a topic. Read snippets first; they often answer "
                    "the question without needing to fetch the page. "
                    "This is the safe, preferred method for web searches. "
                    "Do not use curl or external HTTP tools to search search engines directly, "
                    "as they have captcha and bot protection that will block your requests. "
                    "Always use web_search tool instead."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Search query. Be specific — e.g. 'Python PDF parsing library "
                                "2025' rather than 'PDF library'"
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(self, root_path: str, query: str, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """Execute web search and return formatted results.

        Args:
            root_path: Not used for web search (required by BaseTool interface)
            query: Search query string
            **kwargs: Additional arguments (ignored)

        Returns:
            dict with stdout, stderr, exit_code
        """
        try:
            client = get_web_client(
                user_agent=self._user_agent,
                brave_api_key=self._brave_api_key,
            )
            results = await client.brave_search(query=query, count=self._search_count)
            formatted = self._format_results(results, query)
            return {
                "stdout": formatted,
                "stderr": "",
                "exit_code": 0,
            }
        except ValueError as e:
            # API key missing
            return {
                "stdout": (
                    "Web search failed: no API key configured. "
                    "Set BRAVE_API_KEY environment variable "
                    "or brave_api_key in gateway.toml [web] section."
                ),
                "stderr": str(e),
                "exit_code": 1,
            }
        except Exception as e:
            # HTTP errors, network errors, etc.
            error_msg = str(e)
            # Try to extract status code from HTTPStatusError
            if hasattr(e, "response") and hasattr(e.response, "status_code"):
                status_code = e.response.status_code
                if status_code == 429:
                    error_msg = "Brave API returned 429 (rate limited). Try again later."
                elif status_code == 500:
                    error_msg = "Brave API returned 500 after retries."
                else:
                    error_msg = f"Brave API returned {status_code}."
            return {
                "stdout": f"Web search failed: {error_msg}",
                "stderr": str(e),
                "exit_code": 1,
            }

    def _format_results(self, api_response: dict[str, Any], query: str) -> str:
        """Format Brave API response into human-readable output.

        Args:
            api_response: Parsed JSON from Brave API
            query: Original search query

        Returns:
            Formatted string with numbered results
        """
        lines = [f'Web search results for "{query}":\n']

        web_data = api_response.get("web", {})
        results = web_data.get("results", [])

        if not results:
            lines.append("No results found.")
            return "\n".join(lines)

        for i, result in enumerate(results, start=1):
            title = result.get("title", "(no title)")
            url = result.get("url", "(no URL)")
            description = result.get("description", "(no description)")

            lines.append(f"\n[{i}] {title}")
            lines.append(f"    {url}")
            lines.append(f"    {description}")

            # Extra snippets
            extra_snippets = result.get("extra_snippets", [])
            if extra_snippets:
                for snippet in extra_snippets:
                    lines.append(f"    · {snippet}")

        lines.append(f"\n({len(results)} results)")
        return "\n".join(lines)
