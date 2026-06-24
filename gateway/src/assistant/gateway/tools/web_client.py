"""Shared HTTP client for web search and fetch operations."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Any, cast

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)


@dataclass
class HttpResponse:
    """HTTP response wrapper."""

    status_code: int
    content: str
    headers: dict[str, str]


class PrivateNetworkError(Exception):
    """Raised when a private network address is blocked."""

    pass


class BlockedDomainError(Exception):
    """Raised when a domain is in the blacklist."""

    pass


def _is_5xx_error(exc: BaseException) -> bool:
    """Check if exception is a 5xx HTTP error. Used for tenacity retry condition."""
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


class WebClient:
    """Shared HTTP client for Brave Search and general URL fetching.

    Responsibilities:
    - Create and manage a single httpx.AsyncClient instance (reused for connection pooling)
    - Set configured User-Agent header on all requests
    - For Brave API: set X-Subscription-Token and Accept: application/json headers
    - Retry logic via tenacity
    - Private network blocking (localhost, private IPs, .local domains)
    - Domain blacklist checking (for web_fetch only)
    """

    def __init__(
        self,
        user_agent: str,
        brave_api_key: str = "",
        fetch_blocked_domains: list[str] | None = None,
        fetch_max_redirects: int = 5,
    ):
        self._user_agent = user_agent
        self._brave_api_key = brave_api_key
        self._fetch_blocked_domains = fetch_blocked_domains or []
        self._fetch_max_redirects = fetch_max_redirects
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Return the httpx.AsyncClient, creating it if necessary."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "User-Agent": self._user_agent,
                },
                follow_redirects=True,
                max_redirects=self._fetch_max_redirects,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _is_private_ip(self, ip: str) -> bool:
        """Check if an IP address is private or reserved."""
        try:
            parsed = ipaddress.ip_address(ip)
            # Block private ranges
            if parsed.is_private:
                return True
            if parsed.is_loopback:
                return True
            return bool(parsed.is_link_local)
        except ValueError:
            return False

    def _is_blocked_domain(self, domain: str) -> bool:
        """Check if a domain is in the blacklist."""
        domain_lower = domain.lower()
        for blocked in self._fetch_blocked_domains:
            blocked_lower = blocked.lower()
            # Exact match
            if domain_lower == blocked_lower:
                return True
            # Wildcard prefix: *.example.com blocks sub.example.com and example.com
            if blocked_lower.startswith("*."):
                suffix = blocked_lower[2:]  # ".example.com"
                # Match if domain is exactly the suffix or ends with ".<suffix>"
                if domain_lower == suffix or domain_lower.endswith("." + suffix):
                    return True
        return False

    def _resolve_and_check(self, url: str) -> str:
        """Resolve hostname to IP and check if it's blocked.

        Raises PrivateNetworkError if the resolved IP is private.
        Raises BlockedDomainError if the domain is blacklisted.
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        hostname = parsed.hostname
        if hostname is None:
            raise ValueError(f"Invalid URL: {url}")

        # Check domain blacklist first
        if self._is_blocked_domain(hostname):
            raise BlockedDomainError(f"Domain blocked: {hostname}")

        # Block .local domains
        if hostname.endswith(".local"):
            raise PrivateNetworkError(f".local domains are blocked: {hostname}")

        # Resolve hostname to IP
        try:
            # Get all addresses (might be multiple A/AAAA records)
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_INET)
        except socket.gaierror:
            # If DNS resolution fails, try IPv6
            try:
                addr_info = socket.getaddrinfo(hostname, None, socket.AF_INET6)
            except socket.gaierror as err:
                raise PrivateNetworkError(f"DNS resolution failed: {hostname}") from err

        if not addr_info:
            raise PrivateNetworkError(f"No addresses found for: {hostname}")

        # Check all resolved IPs
        for _family, _socktype, _proto, _canonname, sockaddr in addr_info:
            ip = sockaddr[0]
            if isinstance(ip, str) and self._is_private_ip(ip):
                raise PrivateNetworkError(f"Private/reserved IP blocked: {hostname} -> {ip}")

        return hostname

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    async def brave_search(self, query: str, count: int) -> dict[str, Any]:
        """Call the Brave Search API.

        Args:
            query: Search query string
            count: Number of results to request

        Returns:
            Parsed JSON response from Brave API

        Raises:
            PrivateNetworkError: Should not occur for Brave API
            httpx.HTTPStatusError: If API returns error status (retried per tenacity)
            httpx.NetworkError: If network error (retried per tenacity)
        """
        if not self._brave_api_key:
            raise ValueError("Brave API key not configured")

        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "X-Subscription-Token": self._brave_api_key,
            "Accept": "application/json",
        }
        params = {
            "q": query,
            "count": min(count, 20),  # Brave API max is 20
            "extra_snippets": "true",
        }

        # Convert params to proper format for httpx
        # httpx accepts dict[str, Any] but we need to ensure proper types
        response = await self.client.get(url, headers=headers, params=params)  # type: ignore[arg-type]
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    @retry(
        retry=retry_if_exception_type(httpx.NetworkError) | retry_if_exception(_is_5xx_error),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
    )
    async def fetch_url(self, url: str) -> HttpResponse:
        """Fetch a URL with private network blocking and domain blacklist.

        Args:
            url: URL to fetch

        Returns:
            HttpResponse with status_code, content, and headers

        Raises:
            PrivateNetworkError: If the URL resolves to a private IP
            BlockedDomainError: If the domain is blacklisted
            httpx.HTTPStatusError: If HTTP error (5xx only, retried)
            httpx.NetworkError: If network error (retried)
        """
        # Resolve and check before making request
        self._resolve_and_check(url)

        response = await self.client.get(url, follow_redirects=True)
        response.raise_for_status()

        return HttpResponse(
            status_code=response.status_code,
            content=response.text,
            headers=dict(response.headers),
        )


# Singleton instance - created at gateway startup, closed on shutdown
_web_client_instance: WebClient | None = None


def get_web_client(
    user_agent: str,
    brave_api_key: str = "",
    fetch_blocked_domains: list[str] | None = None,
    fetch_max_redirects: int = 5,
) -> WebClient:
    """Get or create the singleton WebClient instance."""
    global _web_client_instance
    if _web_client_instance is None:
        _web_client_instance = WebClient(
            user_agent=user_agent,
            brave_api_key=brave_api_key,
            fetch_blocked_domains=fetch_blocked_domains,
            fetch_max_redirects=fetch_max_redirects,
        )
    return _web_client_instance


async def close_web_client() -> None:
    """Close the singleton WebClient instance."""
    global _web_client_instance
    if _web_client_instance is not None:
        await _web_client_instance.close()
        _web_client_instance = None
