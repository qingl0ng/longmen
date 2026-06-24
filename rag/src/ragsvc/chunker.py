"""Text chunker with section-boundary snapping."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from .extractors.base import ExtractedDocument
from .tokenizer import TokenCounter

log = structlog.get_logger(__name__)

# A section/paragraph is kept whole if it fits within this multiple of chunk_size.
# Above this threshold it is split further.
_SPLIT_THRESHOLD = 1.3

_SENTENCE_END_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z\n])")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")


@dataclass
class Chunk:
    text: str
    token_count: int
    chunk_index: int
    heading_hierarchy: list[str] = field(default_factory=list)
    page: int | None = None
    start_char: int = 0
    end_char: int = 0


def _split_paragraphs(text: str) -> list[str]:
    """Split text on double newlines, keeping paragraphs whole."""
    parts = re.split(r"\n\n+", text)
    return [p for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries."""
    parts = _SENTENCE_END_RE.split(text)
    return [p for p in parts if p.strip()]


def _has_code_fence(text: str) -> bool:
    """Check if text contains an open code fence that hasn't been closed."""
    in_fence = False
    fence_char = ""
    for line in text.splitlines():
        m = _FENCE_RE.match(line)
        if m:
            char = m.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_char = char
            elif char == fence_char:
                in_fence = False
                fence_char = ""
    return in_fence


def _group_paragraphs_into_chunks(
    paragraphs: list[str],
    token_counter: TokenCounter,
    chunk_size: int,
    overlap: int,
    heading_hierarchy: list[str],
    page: int | None,
    char_offset: int,
    initial_overlap: str = "",
) -> list[Chunk]:
    """Group paragraphs into chunks targeting chunk_size with overlap between chunks.

    initial_overlap: tail text carried in from the previous section (cross-section overlap).
    It seeds the first chunk without advancing char_offset.
    """
    chunks: list[Chunk] = []
    current_paras: list[str] = [initial_overlap] if initial_overlap else []
    current_tokens = token_counter.count(initial_overlap) if initial_overlap else 0
    current_start = char_offset

    def _flush(paras: list[str], start: int) -> Chunk | None:
        if not paras:
            return None
        text = "\n\n".join(paras)
        tokens = token_counter.count(text)
        end = start + len(text)
        return Chunk(
            text=text,
            token_count=tokens,
            chunk_index=0,
            heading_hierarchy=list(heading_hierarchy),
            page=page,
            start_char=start,
            end_char=end,
        )

    for para in paragraphs:
        para_tokens = token_counter.count(para)

        # Single paragraph too large: split at sentence level
        if para_tokens > chunk_size * _SPLIT_THRESHOLD:
            # Flush current accumulation first
            if current_paras:
                c = _flush(current_paras, current_start)
                if c:
                    chunks.append(c)
                    # Set up overlap
                    overlap_text = _get_last_n_tokens(
                        "\n\n".join(current_paras), overlap, token_counter
                    )
                    current_paras = [overlap_text] if overlap_text else []
                    current_tokens = token_counter.count(overlap_text) if overlap_text else 0
                    current_start = char_offset  # approximate
                else:
                    current_paras = []
                    current_tokens = 0

            sentence_chunks = _split_by_sentences(
                para, token_counter, chunk_size, overlap, heading_hierarchy, page, char_offset
            )
            chunks.extend(sentence_chunks)
            char_offset += len(para) + 2
            current_start = char_offset
            current_paras = []
            current_tokens = 0
            continue

        # Would adding this paragraph exceed chunk_size?
        sep_tokens = 2 if current_paras else 0  # "\n\n" approximately 2 tokens
        if current_paras and (current_tokens + sep_tokens + para_tokens) > chunk_size:
            c = _flush(current_paras, current_start)
            if c:
                chunks.append(c)
                # Overlap: prepend tail of previous chunk
                overlap_text = _get_last_n_tokens(
                    "\n\n".join(current_paras), overlap, token_counter
                )
                current_paras = [overlap_text] if overlap_text else []
                current_tokens = token_counter.count(overlap_text) if overlap_text else 0
                current_start = c.end_char + 2

        current_paras.append(para)
        current_tokens = token_counter.count("\n\n".join(current_paras))
        char_offset += len(para) + 2

    if current_paras:
        c = _flush(current_paras, current_start)
        if c:
            chunks.append(c)

    return chunks


def _split_by_sentences(
    text: str,
    token_counter: TokenCounter,
    chunk_size: int,
    overlap: int,
    heading_hierarchy: list[str],
    page: int | None,
    char_offset: int,
) -> list[Chunk]:
    """Split a large paragraph at sentence boundaries."""
    sentences = _split_sentences(text)
    chunks: list[Chunk] = []
    current_sents: list[str] = []
    current_tokens = 0
    current_start = char_offset

    def _flush(sents: list[str], start: int) -> Chunk | None:
        if not sents:
            return None
        t = " ".join(sents)
        tok = token_counter.count(t)
        return Chunk(
            text=t,
            token_count=tok,
            chunk_index=0,
            heading_hierarchy=list(heading_hierarchy),
            page=page,
            start_char=start,
            end_char=start + len(t),
        )

    for sent in sentences:
        sent_tokens = token_counter.count(sent)

        # Single sentence too large: hard-split at token limit
        if sent_tokens > chunk_size * _SPLIT_THRESHOLD:
            if current_sents:
                c = _flush(current_sents, current_start)
                if c:
                    chunks.append(c)
                    overlap_text = _get_last_n_tokens(
                        " ".join(current_sents), overlap, token_counter
                    )
                    current_sents = [overlap_text] if overlap_text else []
                    current_tokens = token_counter.count(overlap_text) if overlap_text else 0
                    current_start = c.end_char + 1

            hard_chunk_text = token_counter.truncate_to_tokens(sent, chunk_size)
            tok = token_counter.count(hard_chunk_text)
            chunks.append(
                Chunk(
                    text=hard_chunk_text,
                    token_count=tok,
                    chunk_index=0,
                    heading_hierarchy=list(heading_hierarchy),
                    page=page,
                    start_char=current_start,
                    end_char=current_start + len(hard_chunk_text),
                )
            )
            current_start += len(sent) + 1
            continue

        if current_sents and (current_tokens + 1 + sent_tokens) > chunk_size:
            c = _flush(current_sents, current_start)
            if c:
                chunks.append(c)
                overlap_text = _get_last_n_tokens(" ".join(current_sents), overlap, token_counter)
                current_sents = [overlap_text] if overlap_text else []
                current_tokens = token_counter.count(overlap_text) if overlap_text else 0
                current_start = c.end_char + 1

        current_sents.append(sent)
        current_tokens = token_counter.count(" ".join(current_sents))

    if current_sents:
        c = _flush(current_sents, current_start)
        if c:
            chunks.append(c)

    return chunks


def _get_last_n_tokens(text: str, n_tokens: int, token_counter: TokenCounter) -> str:
    """Return the last n tokens of text as a string."""
    if n_tokens <= 0:
        return ""
    total = token_counter.count(text)
    if total <= n_tokens:
        return text
    # Approximate by character ratio then adjust
    ratio = n_tokens / total
    start_char = int(len(text) * (1 - ratio))
    candidate = text[start_char:]
    # Adjust if needed
    while token_counter.count(candidate) > n_tokens and len(candidate) > 0:
        candidate = candidate[1:]
    return candidate


class Chunker:
    """Splits an ExtractedDocument into Chunks with section-boundary snapping."""

    def __init__(
        self,
        token_counter: TokenCounter,
        chunk_size: int = 1024,
        overlap: int = 128,
    ) -> None:
        self._counter = token_counter
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, document: ExtractedDocument) -> list[Chunk]:
        """Chunk an ExtractedDocument into a list of Chunks."""
        all_chunks: list[Chunk] = []
        heading_stack: list[tuple[int, str]] = []  # (level, heading_text)
        chunk_index = 0

        # Track character offsets across the full document
        char_offset = 0

        # Tail of the last chunk from the previous section, for cross-section overlap.
        # Reset to "" at h1/h2 boundaries; carried forward at h3+ boundaries.
        cross_section_tail: str = ""

        for section in document.sections:
            # Skip empty sections
            body = section.text.strip() if section.text else ""
            if not body and section.heading is None:
                continue
            if not body:
                continue

            # Update heading stack
            if section.heading is not None and section.heading_level is not None:
                level = section.heading_level
                # Pop all headings at same or deeper level
                heading_stack = [(lvl, h) for lvl, h in heading_stack if lvl < level]
                heading_stack.append((level, section.heading))

            current_hierarchy = [h for _, h in heading_stack]
            current_level = section.heading_level

            # h1/h2 boundaries start clean — no overlap carried in from the previous section.
            # h3+ boundaries carry the tail of the previous section's last chunk.
            is_top_level_boundary = current_level is not None and current_level <= 2
            section_initial_overlap = "" if is_top_level_boundary else cross_section_tail

            section_tokens = self._counter.count(body)
            chunks_before = len(all_chunks)

            if section_tokens <= self._chunk_size * _SPLIT_THRESHOLD:
                # Keep whole section as one chunk.
                # For h3+ sections, prepend the cross-section overlap tail so the chunk
                # carries context from the previous section.
                if section_initial_overlap:
                    text = section_initial_overlap + "\n\n" + body
                    token_count = self._counter.count(text)
                else:
                    text = body
                    token_count = section_tokens
                chunk = Chunk(
                    text=text,
                    token_count=token_count,
                    chunk_index=chunk_index,
                    heading_hierarchy=list(current_hierarchy),
                    page=section.page,
                    start_char=char_offset,
                    end_char=char_offset + len(body),
                )
                all_chunks.append(chunk)
                chunk_index += 1
            else:
                # Section too large — split.
                # If section contains unclosed code fence, keep as oversized chunk.
                if _has_code_fence(body):
                    chunk = Chunk(
                        text=body,
                        token_count=section_tokens,
                        chunk_index=chunk_index,
                        heading_hierarchy=list(current_hierarchy),
                        page=section.page,
                        start_char=char_offset,
                        end_char=char_offset + len(body),
                    )
                    all_chunks.append(chunk)
                    chunk_index += 1
                else:
                    sub_chunks = self._split_section(
                        body,
                        section.page,
                        current_hierarchy,
                        char_offset,
                        initial_overlap=section_initial_overlap,
                    )
                    for sc in sub_chunks:
                        sc.chunk_index = chunk_index
                        all_chunks.append(sc)
                        chunk_index += 1

            char_offset += len(body) + 1  # +1 for section separator

            # Update the cross-section tail from the last chunk produced in this section.
            if len(all_chunks) > chunks_before:
                cross_section_tail = _get_last_n_tokens(
                    all_chunks[-1].text, self._overlap, self._counter
                )
            else:
                cross_section_tail = ""

        return all_chunks

    def _split_section(
        self,
        text: str,
        page: int | None,
        heading_hierarchy: list[str],
        char_offset: int,
        initial_overlap: str = "",
    ) -> list[Chunk]:
        """Split a large section into chunks at paragraph/sentence/token boundaries.

        Code fence blocks are kept whole even if they exceed chunk_size.
        initial_overlap: tail text from the previous section to seed the first chunk.
        """
        parts = self._split_respecting_fences(text)
        all_chunks: list[Chunk] = []
        current_char = char_offset
        # Only the first non-fence part receives the cross-section overlap seed.
        remaining_initial_overlap = initial_overlap

        for part in parts:
            stripped = part.strip()
            is_fence_block = stripped.startswith("```") or stripped.startswith("~~~")

            if is_fence_block:
                # Keep the code fence block whole — never split it
                tokens = self._counter.count(stripped)
                chunk = Chunk(
                    text=stripped,
                    token_count=tokens,
                    chunk_index=0,
                    heading_hierarchy=list(heading_hierarchy),
                    page=page,
                    start_char=current_char,
                    end_char=current_char + len(stripped),
                )
                all_chunks.append(chunk)
            else:
                paragraphs = _split_paragraphs(part)
                sub = _group_paragraphs_into_chunks(
                    paragraphs,
                    self._counter,
                    self._chunk_size,
                    self._overlap,
                    heading_hierarchy,
                    page,
                    current_char,
                    initial_overlap=remaining_initial_overlap,
                )
                all_chunks.extend(sub)
                remaining_initial_overlap = ""  # Only first non-fence part gets the seed

            current_char += len(part) + 2

        return all_chunks

    def _split_respecting_fences(self, text: str) -> list[str]:
        """Split text into parts, isolating fenced code blocks as standalone parts.

        Non-fence content before/after/between code blocks becomes its own part.
        Fenced code blocks (from opening ``` to closing ```) become standalone parts.
        """
        lines = text.splitlines(keepends=True)
        parts: list[str] = []
        current: list[str] = []
        in_fence = False
        fence_char = ""
        fence_block: list[str] = []

        for line in lines:
            stripped = line.rstrip("\n").rstrip("\r")
            if not in_fence:
                m = _FENCE_RE.match(stripped)
                if m:
                    # Flush non-fence content before this fence
                    if current:
                        parts.append("".join(current))
                        current = []
                    in_fence = True
                    fence_char = m.group(1)[0]
                    fence_block = [line]
                else:
                    current.append(line)
            else:
                fence_block.append(line)
                m = _FENCE_RE.match(stripped)
                if m and m.group(1)[0] == fence_char:
                    in_fence = False
                    fence_char = ""
                    # Flush fence block as its own part
                    parts.append("".join(fence_block))
                    fence_block = []

        # Flush remaining content
        if fence_block:
            # Unclosed fence — treat as regular content
            current.extend(fence_block)
        if current:
            parts.append("".join(current))

        return [p for p in parts if p.strip()]
