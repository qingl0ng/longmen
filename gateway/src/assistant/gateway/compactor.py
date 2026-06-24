"""Model-based compaction engine.

Compaction replaces the "compactable middle" of the conversation (between pinned
first and pinned recent turns) with a model-generated summary plus extracted facts.

The raw session history is never modified — compacted messages are marked with
compacted=True and hidden from the assembled context view.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from .fact_extractor import FactExtractor
from .session import Session, TrackedMessage, count_tokens

if TYPE_CHECKING:
    from .context_manager import ContextManager

log = structlog.get_logger(__name__)

_COMPACT_PROMPT_TEMPLATE = """\
Summarize this conversation segment into a concise recap for your own future reference. \
You will continue this conversation after the summary, so preserve everything you need \
to continue working effectively.

Preserve:
- What the user asked for (the original task and any clarifications)
- Key decisions made and why
- What actions were taken (files read, modified, created — but NOT the file contents)
- Current state: what's done, what's remaining, what's blocked
- Any errors encountered and how they were resolved
- Constraints or preferences the user expressed

Discard:
- Full file contents (you can re-read files if needed)
- Verbose tool output (keep only the conclusion)
- Intermediate reasoning that led to the current approach
- Superseded code versions

Write in past tense, as a factual record. Target: under {target_tokens} tokens.

--- Conversation to summarize ---

{conversation}"""


def _format_messages_for_compact_prompt(
    messages: list[TrackedMessage],
) -> str:
    """Format messages as a readable transcript for the compaction prompt."""
    parts: list[str] = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if msg.role == "user":
            parts.append(f"User: {content}")
        elif msg.role == "assistant":
            if msg.tool_calls:
                calls = ", ".join(tc.get("function", {}).get("name", "?") for tc in msg.tool_calls)
                prefix = f"[Calls: {calls}] " if calls else ""
                parts.append(f"Assistant: {prefix}{content}")
            else:
                parts.append(f"Assistant: {content}")
        elif msg.role == "tool":
            tool = msg.tool_name or "tool"
            # Truncate very large tool outputs in the prompt itself
            if len(content) > 2000:
                content = content[:2000] + "\n[... truncated ...]"
            parts.append(f"Tool ({tool}): {content}")

    return "\n\n".join(parts)


class Compactor:
    """Compacts the middle of the conversation using a model-generated summary."""

    def __init__(
        self,
        context_manager: ContextManager,
        compact_target_tokens: int = 1000,
        compact_min_tokens: int = 500,
    ) -> None:
        self.context_manager = context_manager
        self.compact_target_tokens = compact_target_tokens
        # Minimum tokens of middle content that makes compaction worthwhile.
        # Set to 0 in tests to force compaction regardless of middle size.
        self.compact_min_tokens = compact_min_tokens
        self._fact_extractor = FactExtractor()

    async def run(
        self,
        session: Session,
        vllm_client: Any,
        skip_initial_prune: bool = False,
    ) -> int:
        """Prune stale outputs, then compact if still above threshold.

        skip_initial_prune: when True, skip the internal prune() call (caller has
        already called prune_stale() and wants to go straight to model compaction).

        Returns total tokens freed.
        """
        total_freed = 0

        if not skip_initial_prune:
            # Step 1: Prune first (free, no model call)
            freed = self.context_manager.prune(session)
            total_freed += freed
            log.info("compactor.pruned", freed=freed, tokens_used=session.tokens_used)

        # Step 2: Check if compaction is still needed
        _, middle, _ = self.context_manager.segment_messages(session)
        middle_tokens = sum(m.tokens for m in middle)

        if middle_tokens < self.compact_min_tokens:
            log.info(
                "compactor.skip_compact",
                reason="middle_too_small",
                middle_tokens=middle_tokens,
            )
            return total_freed

        if not self.context_manager.should_compact(session):
            log.info("compactor.skip_compact", reason="below_threshold")
            return total_freed

        # Step 3: Model-based compaction
        freed = await self._compact_middle(session, vllm_client, middle)
        total_freed += freed
        log.info("compactor.compacted", freed=freed, tokens_used=session.tokens_used)

        return total_freed

    async def manual_compact(
        self,
        session: Session,
        vllm_client: Any,
    ) -> int:
        """Force prune + compact regardless of thresholds (for /compact command).

        Returns total tokens freed.
        """
        total_freed = 0

        freed = self.context_manager.prune(session)
        total_freed += freed

        _, middle, _ = self.context_manager.segment_messages(session)
        if not middle:
            return total_freed

        middle_tokens = sum(m.tokens for m in middle)
        if middle_tokens < self.compact_min_tokens:
            return total_freed

        freed = await self._compact_middle(session, vllm_client, middle)
        total_freed += freed

        return total_freed

    async def _compact_middle(
        self,
        session: Session,
        vllm_client: Any,
        middle: list[TrackedMessage],
    ) -> int:
        """Replace middle messages with a model-generated summary. Returns tokens freed."""
        old_middle_tokens = sum(m.tokens for m in middle)

        # Collect commands for fact extraction
        commands: list[str] = []
        for msg in middle:
            if msg.tool_name and msg.message_type == "tool_result":
                cmd = msg.metadata.get("command", "")
                if cmd:
                    commands.append(f"{msg.tool_name}({cmd})")
                else:
                    commands.append(msg.tool_name)

        # Extract facts before compaction
        all_text = "\n".join(
            m.content if isinstance(m.content, str) else str(m.content) for m in middle
        )
        facts = self._fact_extractor.extract(all_text)
        facts.commands_run = commands

        # Build compaction prompt
        transcript = _format_messages_for_compact_prompt(middle)
        prompt = _COMPACT_PROMPT_TEMPLATE.format(
            target_tokens=self.compact_target_tokens,
            conversation=transcript,
        )

        # Call the model for a summary
        try:
            summary_text = await self._call_model(vllm_client, prompt)
        except Exception as exc:
            log.error("compactor.model_call_failed", error=str(exc))
            # Fall back to a minimal summary without model
            summary_text = (
                f"[Compaction failed — {len(middle)} messages"
                f" ({old_middle_tokens} tokens) were in this segment]"
            )

        # Append extracted facts
        facts_block = self._fact_extractor.format_facts(facts)
        if facts_block:
            full_summary = f"[Compacted conversation history]:\n\n{summary_text}\n\n{facts_block}"
        else:
            full_summary = f"[Compacted conversation history]:\n\n{summary_text}"

        summary_tokens = count_tokens(full_summary)

        # Mark all middle messages as compacted (hidden from assembled context)
        for msg in middle:
            session.tokens_used = max(0, session.tokens_used - msg.tokens)
            msg.compacted = True

        replaces_before = max(m.timestamp for m in middle)
        summary_msg = TrackedMessage(
            role="assistant",
            content=full_summary,
            tokens=summary_tokens,
            timestamp=time.time(),
            message_type="compaction_summary",
            compacted=True,
        )
        await session.mark_compacted(replaces_before, summary_msg)

        tokens_freed = max(0, old_middle_tokens - summary_tokens)
        return tokens_freed

    async def _call_model(self, vllm_client: Any, prompt: str) -> str:
        """Send the compaction prompt to the model and collect the response."""
        messages = [
            {"role": "user", "content": prompt},
        ]
        accumulated = ""
        async for chunk in vllm_client.stream(messages, tools=None):
            if chunk.delta_text:
                accumulated += chunk.delta_text
            if chunk.finish_reason == "stop":
                break
        return accumulated.strip()
