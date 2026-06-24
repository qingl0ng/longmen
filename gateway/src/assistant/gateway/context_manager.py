"""Context manager: budget tracking, pruning, and context assembly.

The context manager sits between the session (raw history) and the vLLM call.
It decides what the model actually sees — which may be less than the full history.

    Session (raw history)
        ↓
    ContextManager (filter, prune, compact, budget)
        ↓
    Messages sent to vLLM (curated context window)

The session's raw history is never modified by assembly. Compaction marks messages
as compacted=True and stores a summary; pruning replaces content with placeholders
but keeps the message structure intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from .session import Session, TrackedMessage, count_tokens

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ContextOverflowError(Exception):
    """Raised when the assembled context exceeds the token budget."""

    def __init__(self, used: int, limit: int) -> None:
        self.used = used
        self.limit = limit
        super().__init__(f"Context overflow: {used} tokens used, {limit} limit")


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass
class ContextBudget:
    total: int  # context_limit from config

    # Reserved (never compactable)
    system_prompt: int = 0
    reserved_response: int = 4096

    # Dynamic
    pinned_first: int = 0
    compacted_summary: int = 0
    active_context: int = 0
    pinned_recent: int = 0

    @property
    def used(self) -> int:
        """Assembled context token count (pruned/compacted-aware)."""
        return (
            self.system_prompt
            + self.pinned_first
            + self.compacted_summary
            + self.active_context
            + self.pinned_recent
        )

    @property
    def available(self) -> int:
        """Tokens available for active context (conversation + tool results)."""
        return (
            self.total
            - self.system_prompt
            - self.reserved_response
            - self.pinned_first
            - self.pinned_recent
            - self.compacted_summary
        )

    @property
    def utilization(self) -> float:
        """Current utilization as a fraction of usable total (excludes response reserve)."""
        denom = self.total - self.reserved_response
        if denom <= 0:
            return 0.0
        return min(1.0, self.used / denom)

    def to_dict(self) -> dict[str, Any]:
        """Return serializable breakdown for status messages."""
        return {
            "used": self.used,
            "limit": self.total,
            "breakdown": {
                "system_prompt": self.system_prompt,
                "pinned_first": self.pinned_first,
                "compacted_summary": self.compacted_summary,
                "active_context": self.active_context,
                "pinned_recent": self.pinned_recent,
                "response_reserve": self.reserved_response,
                "available": self.available,
            },
        }


# ---------------------------------------------------------------------------
# Staleness detection helpers
# ---------------------------------------------------------------------------

# Tools that indicate a file was modified
_WRITE_TOOLS = frozenset({"write_file", "search_replace", "patch_file"})

# Tools whose output becomes stale after any subsequent code change
_VOLATILE_OUTPUT_TOOLS = frozenset({"shell", "run_tests", "build"})

# Tool names we know produce build/test output (command-prefix heuristic)
_BUILD_PREFIXES = ("make ", "cmake", "cargo build", "go build", "gradle", "mvn")
_TEST_PREFIXES = ("pytest", "jest", "cargo test", "go test", "npm test", "make test")


def _tool_produced_volatile_output(msg: TrackedMessage) -> bool:
    """Return True if this tool result is likely build/test output."""
    if msg.tool_name in _VOLATILE_OUTPUT_TOOLS:
        return True
    # Check command stored in metadata
    cmd = msg.metadata.get("command", "")
    if isinstance(cmd, str):
        cmd_lower = cmd.lower().strip()
        for prefix in _BUILD_PREFIXES + _TEST_PREFIXES:
            if cmd_lower.startswith(prefix):
                return True
    return False


def _is_stale_file_read(msg: TrackedMessage, session: Session) -> bool:
    """A file read is stale if there is a newer read or a write to the same file."""
    if msg.tool_name != "read_file":
        return False
    file_path = msg.metadata.get("path")
    if not file_path:
        return False

    for later in session.messages_after(msg):
        if later.compacted or later.pruned:
            continue
        later_path = later.metadata.get("path")
        if later.tool_name == "read_file" and later_path == file_path:
            return True
        if later.tool_name in _WRITE_TOOLS and later_path == file_path:
            return True
    return False


def _is_stale_volatile_output(msg: TrackedMessage, session: Session) -> bool:
    """Build/test output is stale after any subsequent code change."""
    if not _tool_produced_volatile_output(msg):
        return False
    for later in session.messages_after(msg):
        if later.compacted or later.pruned:
            continue
        if later.tool_name in _WRITE_TOOLS:
            return True
        # Check by metadata
        if later.metadata.get("tool_name") in _WRITE_TOOLS:
            return True
    return False


def _line_count(content: str) -> int:
    return content.count("\n") + 1 if content else 0


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------


class ContextManager:
    """Manages the context window: segments, prunes, and assembles messages."""

    def __init__(
        self,
        pin_first_tokens: int = 2000,
        pin_recent_tokens: int = 3000,
        reserved_response_tokens: int = 4096,
        prune_threshold: float = 0.75,
        compact_threshold: float = 0.85,
        warn_threshold: float = 0.95,
    ) -> None:
        self.pin_first_tokens = pin_first_tokens
        self.pin_recent_tokens = pin_recent_tokens
        self.reserved_response_tokens = reserved_response_tokens
        self.prune_threshold = prune_threshold
        self.compact_threshold = compact_threshold
        self.warn_threshold = warn_threshold

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment_messages(
        self, session: Session
    ) -> tuple[list[TrackedMessage], list[TrackedMessage], list[TrackedMessage]]:
        """Split active (non-compacted) messages into pinned_first, middle, pinned_recent.

        Pinned segments are defined by token budgets, not turn counts.
        They never overlap: pinned_recent starts where pinned_first ends.
        """
        active = [m for m in session.messages if not m.compacted]

        # Pinned first: fill from the beginning up to budget
        pinned_first: list[TrackedMessage] = []
        budget_used = 0
        first_end = 0
        for i, msg in enumerate(active):
            if budget_used + msg.tokens <= self.pin_first_tokens:
                pinned_first.append(msg)
                budget_used += msg.tokens
                first_end = i + 1
            else:
                break

        # Pinned recent: fill from the end up to budget (no overlap with pinned_first)
        pinned_recent: list[TrackedMessage] = []
        budget_used = 0
        recent_start = len(active)
        for i in range(len(active) - 1, first_end - 1, -1):
            msg = active[i]
            if budget_used + msg.tokens <= self.pin_recent_tokens:
                pinned_recent.insert(0, msg)
                budget_used += msg.tokens
                recent_start = i
            else:
                break

        # Middle: everything between the pinned segments
        middle = active[first_end:recent_start]

        return pinned_first, middle, pinned_recent

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def compute_budget(self, session: Session, system_tokens: int = 0) -> ContextBudget:
        """Compute the current context budget breakdown."""
        pinned_first, middle, pinned_recent = self.segment_messages(session)
        summary_tokens = session.compacted_summary.tokens if session.compacted_summary else 0

        return ContextBudget(
            total=session.context_limit,
            system_prompt=system_tokens,
            reserved_response=self.reserved_response_tokens,
            pinned_first=sum(m.tokens for m in pinned_first),
            compacted_summary=summary_tokens,
            active_context=sum(m.tokens for m in middle),
            pinned_recent=sum(m.tokens for m in pinned_recent),
        )

    def should_prune(self, session: Session) -> bool:
        return session.context_pressure >= self.prune_threshold

    def should_compact(self, session: Session) -> bool:
        return session.context_pressure >= self.compact_threshold

    def should_warn(self, session: Session) -> bool:
        return session.context_pressure >= self.warn_threshold

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune(self, session: Session) -> int:
        """Prune stale tool outputs in the compactable middle. Returns tokens freed.

        Pruning replaces stale message content with a short placeholder. The message
        structure is preserved — turn ordering is maintained. Pinned messages are
        never pruned.
        """
        _, middle, _ = self.segment_messages(session)
        tokens_freed = 0

        for msg in middle:
            if msg.pruned or msg.compacted:
                continue
            if msg.message_type != "tool_result":
                continue

            freed = self._try_prune_message(msg, session)
            tokens_freed += freed

        if tokens_freed:
            session.tokens_used = max(0, session.tokens_used - tokens_freed)
            log.info("context_manager.pruned", tokens_freed=tokens_freed)

        return tokens_freed

    def prune_stale(self, session: Session) -> int:
        """Prune stale file reads and volatile outputs in the compactable middle.

        Called mid-loop after every tool-call batch. Only handles the two
        structurally unambiguous cases — stale reads and volatile output.
        Tree, grep, and RAG pruning are post-loop only (see prune()).

        Returns tokens freed.
        """
        _, middle, _ = self.segment_messages(session)
        tokens_freed = 0

        for msg in middle:
            if msg.pruned or msg.compacted:
                continue
            if msg.message_type != "tool_result":
                continue
            if msg.metadata.get("pinned"):
                continue

            old_tokens = msg.tokens
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            freed = 0

            if msg.tool_name == "read_file" and _is_stale_file_read(msg, session):
                path = msg.metadata.get("path", "?")
                lines = _line_count(content)
                placeholder = (
                    f"[Previously read {path} ({lines} lines, {old_tokens} tokens)"
                    " — content pruned, file has since been modified]"
                )
                msg.content = placeholder
                msg.tokens = count_tokens(placeholder)
                msg.pruned = True
                freed = max(0, old_tokens - msg.tokens)
            elif _is_stale_volatile_output(msg, session):
                tool = msg.tool_name or "tool"
                cmd = msg.metadata.get("command", "")
                cmd_str = f" ({cmd})" if cmd else ""
                placeholder = (
                    f"[{tool}{cmd_str} output pruned"
                    " — superseded by subsequent code changes]"
                )
                msg.content = placeholder
                msg.tokens = count_tokens(placeholder)
                msg.pruned = True
                freed = max(0, old_tokens - msg.tokens)

            tokens_freed += freed

        if tokens_freed:
            session.tokens_used = max(0, session.tokens_used - tokens_freed)
            log.info("context_manager.prune_stale", tokens_freed=tokens_freed)

        return tokens_freed

    def _try_prune_message(self, msg: TrackedMessage, session: Session) -> int:
        """Attempt to prune a single message. Returns tokens freed (0 if not pruned)."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        old_tokens = msg.tokens

        # NEVER prune pinned messages — they're critical structural metadata
        if msg.metadata.get("pinned"):
            return 0

        if msg.tool_name == "read_file" and _is_stale_file_read(msg, session):
            path = msg.metadata.get("path", "?")
            lines = _line_count(content)
            placeholder = (
                f"[Previously read {path} ({lines} lines, {old_tokens} tokens)"
                " — content pruned, file has since been modified]"
            )
            msg.content = placeholder
            msg.tokens = count_tokens(placeholder)
            msg.pruned = True
            return max(0, old_tokens - msg.tokens)

        if _is_stale_volatile_output(msg, session):
            tool = msg.tool_name or "tool"
            cmd = msg.metadata.get("command", "")
            cmd_str = f" ({cmd})" if cmd else ""
            placeholder = f"[{tool}{cmd_str} output pruned — superseded by subsequent code changes]"
            msg.content = placeholder
            msg.tokens = count_tokens(placeholder)
            msg.pruned = True
            return max(0, old_tokens - msg.tokens)

        # Tree output: prune if content is large and model has since navigated to files
        navigated = self._model_navigated_after(msg, session)
        if msg.tool_name == "tree" and old_tokens > 200 and navigated:
            placeholder = "[Tree output pruned — project structure was explored]"
            msg.content = placeholder
            msg.tokens = count_tokens(placeholder)
            msg.pruned = True
            return max(0, old_tokens - msg.tokens)

        # Large grep results: prune after model has acted on them
        if msg.tool_name == "grep" and old_tokens > 300 and self._model_acted_after(msg, session):
            pattern = msg.metadata.get("pattern", "?")
            placeholder = f'[Grep for "{pattern}" output pruned — results were acted on]'
            msg.content = placeholder
            msg.tokens = count_tokens(placeholder)
            msg.pruned = True
            return max(0, old_tokens - msg.tokens)

        # RAG results: prune after model has produced a text response using them
        if msg.tool_name == "rag_search":
            stale = any(
                later.role == "assistant" and later.message_type == "response"
                for later in session.messages_after(msg)
                if not later.compacted and not later.pruned
            )
            if stale:
                query = msg.metadata.get("query", "?")
                placeholder = (
                    f'[RAG search for "{query}": results pruned'
                    " — information was used in the subsequent response]"
                )
                msg.content = placeholder
                msg.tokens = count_tokens(placeholder)
                msg.pruned = True
                return max(0, old_tokens - msg.tokens)

        return 0

    def _model_navigated_after(self, msg: TrackedMessage, session: Session) -> bool:
        """Return True if the model used read_file after this message."""
        return any(later.tool_name == "read_file" for later in session.messages_after(msg))

    def _model_acted_after(self, msg: TrackedMessage, session: Session) -> bool:
        """Return True if the model made any tool call after this message."""
        for later in session.messages_after(msg):
            if (
                later.message_type in ("tool_call", "tool_result")
                and later is not msg
                and later.role == "assistant"
                and later.tool_calls
            ):
                return True
        return False

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def assemble_context(
        self,
        session: Session,
        system_message: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build the full message list to send to vLLM.

        Order: system → pinned_first → compacted_summary → active_middle → pinned_recent

        Raises ContextOverflowError if the assembled context exceeds the budget.
        """
        messages: list[dict[str, Any]] = [system_message]

        pinned_first, middle, pinned_recent = self.segment_messages(session)

        # Pinned first turns (user's initial instructions, task definition)
        messages.extend(m.to_openai_format() for m in pinned_first)

        # Compacted summary (if compaction has occurred)
        if session.compacted_summary:
            messages.append(session.compacted_summary.to_openai_format())

        # Active middle (may contain pruned messages with placeholders)
        messages.extend(m.to_openai_format() for m in middle)

        # Pinned recent turns (current working context)
        messages.extend(m.to_openai_format() for m in pinned_recent)

        # Verify total fits within budget
        system_tokens = count_tokens(
            system_message.get("content", "")
            if isinstance(system_message.get("content"), str)
            else str(system_message.get("content", ""))
        )
        pinned_first_tokens = sum(m.tokens for m in pinned_first)
        summary_tokens = session.compacted_summary.tokens if session.compacted_summary else 0
        middle_tokens = sum(m.tokens for m in middle)
        pinned_recent_tokens = sum(m.tokens for m in pinned_recent)

        total_tokens = (
            system_tokens
            + pinned_first_tokens
            + summary_tokens
            + middle_tokens
            + pinned_recent_tokens
        )
        hard_limit = session.context_limit - self.reserved_response_tokens
        if total_tokens > hard_limit:
            raise ContextOverflowError(total_tokens, hard_limit)

        return messages
