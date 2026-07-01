"""WebSocket client — connect, reconnect, send/recv."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import websockets
import websockets.exceptions
from websockets.protocol import State as WsState

from .protocol import parse_message

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)

_BACKOFF_SEQUENCE = [1, 2, 4, 8, 16, 30]
_PING_INTERVAL = 30
_PING_TIMEOUT = 10


class GatewayConnectionError(Exception):
    pass


class Connection:
    def __init__(self, url: str, token: str | None = None) -> None:
        self.url = url
        self.token = token
        self.state: str = "disconnected"
        self.session_start_payload: dict[str, Any] | None = None

        self._ws: ClientConnection | None = None
        self._session_id: str | None = None
        self.on_disconnect: Callable[[], None] | None = None

    @classmethod
    async def connect(cls, url: str, token: str | None = None) -> Connection:
        conn = cls(url, token)
        conn.state = "connecting"
        ws_url = conn._build_url()
        conn._ws = await websockets.connect(
            ws_url,
            ping_interval=_PING_INTERVAL,
            ping_timeout=_PING_TIMEOUT,
        )
        conn.state = "connected"
        # Wait for session_start
        msg = await conn.recv()
        if msg["type"] != "session_start":
            raise GatewayConnectionError(f"Expected session_start, got {msg['type']!r}")
        conn.session_start_payload = msg["payload"]
        return conn

    def _build_url(self) -> str:
        if self.token:
            sep = "&" if "?" in self.url else "?"
            return f"{self.url}{sep}token={self.token}"
        return self.url

    async def send(self, message: dict[str, Any]) -> None:
        if self._ws is None:
            raise GatewayConnectionError("Not connected")
        await self._ws.send(json.dumps(message))

    async def recv(self) -> dict[str, Any]:
        if self._ws is None:
            raise GatewayConnectionError("Not connected")
        raw = await self._ws.recv()
        return parse_message(str(raw))

    def _is_closed(self) -> bool:
        return self._ws is None or self._ws.state in (WsState.CLOSING, WsState.CLOSED)

    async def recv_iter(self) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed messages until disconnect. Reconnects automatically."""
        backoff_idx = 0
        while True:
            try:
                if self._is_closed():
                    await self._do_reconnect()
                    backoff_idx = 0
                    yield {"type": "reconnected", "id": "", "timestamp": 0, "payload": {}}

                assert self._ws is not None
                async for raw in self._ws:
                    try:
                        msg = parse_message(str(raw))
                    except ValueError as e:
                        logger.warning("Skipping malformed message: %s", e)
                        continue
                    # Track session_id from any message that carries it
                    sid = msg.get("payload", {}).get("session_id")
                    if sid:
                        self._session_id = sid
                    yield msg

            except websockets.exceptions.ConnectionClosed:
                self.state = "reconnecting"
                logger.info("Connection closed, will reconnect")
                if self.on_disconnect:
                    self.on_disconnect()
            except OSError as e:
                self.state = "reconnecting"
                logger.warning("Connection error: %s", e)
                if self.on_disconnect:
                    self.on_disconnect()
            except websockets.exceptions.WebSocketException as e:
                # Handshake-level failure while reconnecting — e.g. the Docker
                # port is open but the gateway isn't serving valid HTTP yet
                # ("did not receive a valid HTTP response"). Not an OSError, so
                # it would otherwise escape and crash the listener. Retry it.
                self.state = "reconnecting"
                logger.warning("Handshake failed, will retry: %s", e)
                if self.on_disconnect:
                    self.on_disconnect()

            if self.state == "reconnecting":
                delay = _BACKOFF_SEQUENCE[min(backoff_idx, len(_BACKOFF_SEQUENCE) - 1)]
                backoff_idx += 1
                logger.info("Reconnecting in %ds...", delay)
                await asyncio.sleep(delay)

    async def _do_reconnect(self) -> None:
        self.state = "connecting"
        ws_url = self._build_url()
        self._ws = await websockets.connect(
            ws_url,
            ping_interval=_PING_INTERVAL,
            ping_timeout=_PING_TIMEOUT,
        )
        self.state = "connected"
        # Receive session_start
        msg = await self.recv()
        if msg["type"] == "session_start":
            self.session_start_payload = msg["payload"]

    async def close(self) -> None:
        self.state = "disconnected"
        if self._ws is not None and not self._is_closed():
            await self._ws.close()
        self._ws = None
