"""In-memory session store — conversation history and token budget per session."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from .session_store import SessionMeta, SessionStore

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

# Tokenizers cached by source key (a local tokenizer.json path). Until the active
# tokenizer is ready, count_tokens() returns a char/4 estimate so it never blocks
# and never raises.
_tokenizers: dict[str, Any] = {}  # source_key -> Tokenizer | None (None = load failed)
_default_source: str | None = None
_tokenizer_lock = threading.Lock()


def _load_tokenizer_blocking(source: str) -> None:
    """Load a local tokenizer.json. Blocking (disk) — run in a thread. Offline only."""
    from tokenizers import Tokenizer  # type: ignore[import-untyped]

    tok: Any = None
    try:
        tok = Tokenizer.from_file(source)  # local file; never touches the network
    except Exception as exc:
        log.warning("session.tokenizer_unavailable", source=source, error=str(exc))
    with _tokenizer_lock:
        _tokenizers[source] = tok
    log.info("session.tokenizer_loaded", source=source, available=tok is not None)


def init_tokenizer(source: str) -> None:
    """Register `source` as the default tokenizer and start loading it in a
    background thread (non-blocking). Call once at gateway startup."""
    global _default_source  # noqa: PLW0603
    with _tokenizer_lock:
        _default_source = source
        already = source in _tokenizers
    if not already:
        asyncio.get_running_loop().run_in_executor(None, _load_tokenizer_blocking, source)


def count_tokens(text: str, source: str | None = None) -> int:
    """Token count via the local tokenizer, with a char/4 fallback.

    Never blocks the event loop and never raises. `source` defaults to the
    tokenizer registered by init_tokenizer(); the parameter exists so a future
    per-backend change (B) can pass a session's tokenizer without touching callers.
    """
    with _tokenizer_lock:
        key = source if source is not None else _default_source
        tok = _tokenizers.get(key) if key is not None else None
    if tok is None:
        return max(1, len(text) // 4)
    return len(tok.encode(text).ids)


def _message_text(msg: dict[str, Any]) -> str:
    """Extract flat text from an OpenAI-format message for token counting."""
    content = msg.get("content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        text = " ".join(parts)
    else:
        text = json.dumps(content)

    # Include tool call JSON if present
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        text += " " + json.dumps(tool_calls)

    return text


# ---------------------------------------------------------------------------
# TrackedMessage
# ---------------------------------------------------------------------------


@dataclass
class TrackedMessage:
    """A message in the session history with token tracking and metadata."""

    role: str  # "user" | "assistant" | "tool"
    content: str | list[dict[str, Any]] | None  # text, multimodal, or None for internal markers
    tokens: int  # token count for this message
    timestamp: float  # when this message was added (time.time())
    message_type: str  # "prompt" | "response" | "tool_call" | "tool_result"
    tool_name: str | None = None  # for tool_result: which tool produced this
    tool_call_id: str | None = None  # for tool messages: the matching call ID
    tool_calls: list[dict[str, Any]] | None = None  # for assistant messages with tool calls
    metadata: dict[str, Any] = field(default_factory=dict)  # extensible metadata
    compacted: bool = False  # True if this message has been compacted (hidden from context)
    pruned: bool = False  # True if content was replaced with a placeholder

    def to_openai_format(self) -> dict[str, Any]:
        """Return the OpenAI-compatible message dict."""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        return msg


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class Session:
    session_id: str
    project_id: str
    context_limit: int = 32000
    messages: list[TrackedMessage] = field(default_factory=list)
    tokens_used: int = 0
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    last_active: int = field(default_factory=lambda: int(time.time() * 1000))
    # Compacted summary: replaces compacted middle messages in the assembled context.
    # The raw messages are kept (compacted=True) for auditing.
    compacted_summary: TrackedMessage | None = None
    last_system_prompt: str | None = None  # stores the effective system prompt for /prompt command
    compaction_count: int = 0
    _store: SessionStore | None = field(default=None, init=False, repr=False, compare=False)

    def _touch(self) -> None:
        self.last_active = int(time.time() * 1000)

    async def _save_meta(self) -> None:
        if self._store is None:
            return
        from .session_store import SessionMeta
        turn_count = sum(1 for m in self.messages if m.message_type == "prompt")
        meta = SessionMeta(
            session_id=self.session_id,
            project_id=self.project_id,
            created_at=self.created_at / 1000.0,
            last_active=time.time(),
            turn_count=turn_count,
            total_tokens=self.tokens_used,
            compacted=self.compacted_summary is not None,
            compaction_count=self.compaction_count,
        )
        await self._store.save_meta(self.project_id, self.session_id, meta)

    async def _append_and_persist(self, tracked: TrackedMessage) -> TrackedMessage:
        self.messages.append(tracked)
        self.tokens_used += tracked.tokens
        self._touch()
        if self._store is not None:
            await self._store.save_message(self.project_id, self.session_id, tracked)
            await self._save_meta()
        return tracked

    async def add_user_message(
        self,
        content: Any,
        metadata: dict[str, Any] | None = None,
    ) -> TrackedMessage:
        """Append a user message. content may be str or list[content_block]."""
        msg_dict: dict[str, Any] = {"role": "user", "content": content}
        return await self._append_and_persist(TrackedMessage(
            role="user",
            content=content,
            tokens=count_tokens(_message_text(msg_dict)),
            timestamp=time.time(),
            message_type="prompt",
            metadata=metadata or {},
        ))

    async def add_assistant_message(
        self,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> TrackedMessage:
        msg_dict: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg_dict["tool_calls"] = tool_calls
        return await self._append_and_persist(TrackedMessage(
            role="assistant",
            content=content,
            tokens=count_tokens(_message_text(msg_dict)),
            timestamp=time.time(),
            message_type="tool_call" if tool_calls else "response",
            tool_calls=tool_calls,
        ))

    async def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
        tool_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TrackedMessage:
        msg_dict: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        return await self._append_and_persist(TrackedMessage(
            role="tool",
            content=content,
            tokens=count_tokens(_message_text(msg_dict)),
            timestamp=time.time(),
            message_type="tool_result",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=metadata or {},
        ))

    async def mark_compacted(self, replaces_before: float, summary: TrackedMessage) -> None:
        """Record compaction. Appends marker + summary to JSONL, updates in-memory state."""
        if self.compacted_summary:
            self.tokens_used = max(0, self.tokens_used - self.compacted_summary.tokens)
        self.compacted_summary = summary
        self.tokens_used += summary.tokens
        self.compaction_count += 1

        if self._store is not None:
            marker = TrackedMessage(
                role="_compaction",
                content=None,
                tokens=0,
                timestamp=time.time(),
                message_type="compaction_marker",
                metadata={"replaces_before": replaces_before, "summary_follows": True},
                compacted=True,
            )
            await self._store.save_message(self.project_id, self.session_id, marker)
            await self._store.save_message(self.project_id, self.session_id, summary)
            await self._save_meta()

    @classmethod
    def from_persisted(
        cls,
        meta: SessionMeta,
        messages: list[TrackedMessage],
        compacted_summary: TrackedMessage | None,
        store: SessionStore,
        context_limit: int = 32000,
    ) -> Session:
        """Reconstruct an in-memory Session from persisted data."""
        session = cls(
            session_id=meta.session_id,
            project_id=meta.project_id,
            context_limit=context_limit,
            messages=messages,
            tokens_used=meta.total_tokens,
            created_at=int(meta.created_at * 1000),
            last_active=int(meta.last_active * 1000),
            compacted_summary=compacted_summary,
            compaction_count=meta.compaction_count,
        )
        session._store = store
        return session

    def messages_after(self, msg: TrackedMessage) -> list[TrackedMessage]:
        """Return all messages that appear after the given message."""
        try:
            idx = self.messages.index(msg)
            return self.messages[idx + 1 :]
        except ValueError:
            return []

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def context_budget(self) -> dict[str, int]:
        return {"used": self.tokens_used, "limit": self.context_limit}

    @property
    def context_pressure(self) -> float:
        """Fraction of context budget used. 0.0–1.0."""
        if self.context_limit <= 0:
            return 0.0
        return min(1.0, self.tokens_used / self.context_limit)


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(
        self,
        project_id: str,
        context_limit: int = 32000,
        store: SessionStore | None = None,
    ) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            project_id=project_id,
            context_limit=context_limit,
        )
        session._store = store
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def list_by_project(self, project_id: str) -> list[Session]:
        return [s for s in self._sessions.values() if s.project_id == project_id]

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
