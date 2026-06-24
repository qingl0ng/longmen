"""Auth module — open mode vs paired mode, token validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from .config import AuthConfig

log = structlog.get_logger(__name__)


class AuthManager:
    def __init__(self, config: AuthConfig) -> None:
        self._config = config
        # In a real implementation, paired tokens would be stored/validated here.
        self._valid_tokens: set[str] = set()

    def check_connection(self, token: str | None) -> bool:
        """In open mode, always True. In paired mode, validate token."""
        if self._config.mode == "open":
            return True
        if self._config.mode == "paired":
            if token is None:
                log.warning("auth.rejected", reason="no_token")
                return False
            if token in self._valid_tokens:
                return True
            log.warning("auth.rejected", reason="invalid_token")
            return False
        log.error("auth.unknown_mode", mode=self._config.mode)
        return False

    def register_token(self, token: str) -> None:
        """Register a valid paired token (called after successful pairing)."""
        self._valid_tokens.add(token)

    def revoke_token(self, token: str) -> None:
        self._valid_tokens.discard(token)

    def reload(self, config: AuthConfig) -> None:
        self._config = config
