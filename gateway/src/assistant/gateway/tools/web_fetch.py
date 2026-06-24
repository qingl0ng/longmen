"""Web fetch tool for extracting content from URLs."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import structlog

from assistant.gateway.session import count_tokens

from .base import BaseTool
from .web_client import (
    BlockedDomainError,
    PrivateNetworkError,
    get_web_client,
)

log = structlog.get_logger(__name__)


class WebFetchTool(BaseTool):
    """Web fetch tool that extracts content from URLs."""

    name = "web_fetch"

    def __init__(
        self,
        fetch_timeout: int = 15,
        fetch_max_redirects: int = 5,
        user_agent: str = (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        fetch_blocked_domains: list[str] | None = None,
    ):
        super().__init__()
        self._fetch_timeout = fetch_timeout
        self._fetch_max_redirects = fetch_max_redirects
        self._user_agent = user_agent
        self._fetch_blocked_domains = fetch_blocked_domains or []

    def schema(self) -> dict[str, Any]:
        """Return OpenAI-format function schema."""
        return {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": (
                    "Fetch and extract content from a URL. Returns page content as clean markdown. "
                    "Use to read documentation, GitHub READMEs, PyPI pages, blog posts, "
                    "or any web page. Reports token count of fetched content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Full URL to fetch (must include https:// or http://)",
                        },
                    },
                    "required": ["url"],
                },
            },
        }

    async def execute(self, root_path: str, url: str, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """Execute web fetch and return extracted content.

        Args:
            root_path: Not used for web fetch (required by BaseTool interface)
            url: URL to fetch
            **kwargs: Additional arguments (ignored)

        Returns:
            dict with stdout, stderr, exit_code
        """
        try:
            client = get_web_client(
                user_agent=self._user_agent,
                fetch_blocked_domains=self._fetch_blocked_domains,
            )
            response = await client.fetch_url(url)

            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            content = response.content

            # Handle different content types
            if content_type == "text/html":
                result = self._extract_html(content, url)
            elif content_type == "application/json":
                result = self._format_json(content, url)
            elif content_type in ("text/plain", "text/markdown", "text/csv"):
                result = self._passthrough_text(content, url)
            elif content_type == "application/pdf":
                return {
                    "stdout": (
                        "PDF fetched but cannot be extracted via web_fetch. "
                        "Download the file and use local tools to read it."
                    ),
                    "stderr": "",
                    "exit_code": 1,
                }
            else:
                return {
                    "stdout": f"Fetch failed: unsupported content type '{content_type}' for {url}.",
                    "stderr": "",
                    "exit_code": 1,
                }

            token_count = count_tokens(result)
            return {
                "stdout": f"Fetched {url} (~{token_count} tokens)\n\n---\n\n{result}",
                "stderr": "",
                "exit_code": 0,
            }

        except PrivateNetworkError as e:
            return {
                "stdout": f"Fetch blocked: {e}",
                "stderr": str(e),
                "exit_code": 1,
            }
        except BlockedDomainError as e:
            return {
                "stdout": f"Fetch blocked: {e}",
                "stderr": str(e),
                "exit_code": 1,
            }
        except httpx.TimeoutException as e:
            return {
                "stdout": f"Fetch failed: timed out after {self._fetch_timeout}s for {url}.",
                "stderr": str(e),
                "exit_code": 1,
            }
        except httpx.TooManyRedirects as e:
            return {
                "stdout": (
                    f"Fetch failed: too many redirects ({self._fetch_max_redirects}) for {url}."
                ),
                "stderr": str(e),
                "exit_code": 1,
            }
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            return {
                "stdout": f"Fetch failed: {url} returned HTTP {status_code}.",
                "stderr": str(e),
                "exit_code": 1,
            }
        except Exception as e:
            return {
                "stdout": f"Fetch failed: {e}",
                "stderr": str(e),
                "exit_code": 1,
            }

    def _extract_html(self, html: str, url: str) -> str:
        """Extract markdown from HTML using trafilatura."""
        try:
            import trafilatura  # type: ignore[import-untyped]

            result = trafilatura.extract(  # type: ignore[no-any-return, unused-ignore]
                html,
                output_format="markdown",
                include_links=True,
                include_formatting=True,
                include_tables=True,
                include_images=False,
                no_fallback=False,
            )

            if result is None or result.strip() == "":
                return (
                    "[No content could be extracted from this page. The page may require "
                    "JavaScript to render, or contains no extractable text.]"
                )

            return cast(str, result)

        except ImportError:
            return (
                "[trafilatura library not installed. Please install it with: "
                "pip install trafilatura]"
            )
        except Exception as e:
            return f"[Error extracting content: {e}]"

    def _format_json(self, content: str, url: str) -> str:
        """Pretty-print JSON content."""
        try:
            data = json.loads(content)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError:
            # If it's not valid JSON, return as-is
            return content

    def _passthrough_text(self, content: str, url: str) -> str:
        """Return text content as-is."""
        return content
