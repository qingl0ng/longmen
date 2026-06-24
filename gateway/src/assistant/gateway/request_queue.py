"""FIFO request queue — serializes access to vLLM."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class QueuedRequest:
    session_id: str
    ref_id: str
    ws: Any  # WebSocket-like object
    event: asyncio.Event = field(default_factory=asyncio.Event)


class RequestQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[QueuedRequest] = asyncio.Queue()
        self._active: QueuedRequest | None = None
        self._lock: asyncio.Lock = asyncio.Lock()
        self._all_items: list[QueuedRequest] = []

    async def enqueue(self, session_id: str, ref_id: str, ws: Any) -> None:
        """Add request to queue. If model is idle, run immediately.
        If model is busy, send queue_position to client and wait."""
        from .protocol import make_queue_position

        item = QueuedRequest(session_id=session_id, ref_id=ref_id, ws=ws)

        async with self._lock:
            if self._active is None:
                self._active = item
                self._all_items = [item]
                item.event.set()
            else:
                await self._queue.put(item)
                self._all_items.append(item)
                position = self._queue.qsize()
                active_sid = self._active.session_id if self._active else None
                await ws.send(
                    __import__("json").dumps(
                        make_queue_position(session_id, ref_id, position, active_sid)
                    )
                )

        await item.event.wait()

    async def complete(self) -> None:
        """Called when the active agent loop finishes. Dequeue next."""
        async with self._lock:
            if self._active and self._active in self._all_items:
                self._all_items.remove(self._active)
            self._active = None
            if not self._queue.empty():
                next_item = await self._queue.get()
                self._active = next_item
                next_item.event.set()
                await self._notify_positions()

    async def remove(self, session_id: str) -> None:
        """Remove a queued request (client disconnected or aborted)."""

        async with self._lock:
            # Re-build queue without the removed session
            items = []
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    if item.session_id != session_id:
                        items.append(item)
                except asyncio.QueueEmpty:
                    break

            for item in items:
                await self._queue.put(item)

            self._all_items = [i for i in self._all_items if i.session_id != session_id]
            await self._notify_positions()

    async def _notify_positions(self) -> None:
        """Notify queued clients of their updated positions."""
        import json

        from .protocol import make_queue_position

        active_sid = self._active.session_id if self._active else None
        items = []
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        for pos, item in enumerate(items, start=1):
            with contextlib.suppress(Exception):
                await item.ws.send(
                    json.dumps(make_queue_position(item.session_id, item.ref_id, pos, active_sid))
                )
            await self._queue.put(item)

    @property
    def active_session_id(self) -> str | None:
        return self._active.session_id if self._active else None
