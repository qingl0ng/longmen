"""PDF extractor using PyMuPDF (fitz)."""

from __future__ import annotations

import re
import statistics
import tomllib
from pathlib import Path
from typing import Any

import fitz  # pymupdf
import structlog

from .base import ExtractedDocument, Extractor, Section

log = structlog.get_logger(__name__)

# Font size must be this many times larger than the median to be considered a heading.
_HEADING_FONT_RATIO = 1.3
# Sequences of this many base64-like chars are treated as binary data and sanitized out.
_MIN_BASE64_SANITIZE_LEN = 101

# Regex for sanitization
_NULL_BYTES_RE = re.compile(r"\x00+")
_NON_PRINTABLE_RE = re.compile(r"[^\x09\x0a\x0d\x20-\x7e\x80-\xff]{4,}")
_BASE64_LIKE_RE = re.compile(rf"[A-Za-z0-9+/=]{{{_MIN_BASE64_SANITIZE_LEN},}}")


def _sanitize(text: str, source: str = "") -> str:
    """Remove binary junk from extracted PDF text."""
    original_len = len(text)

    text = _NULL_BYTES_RE.sub("", text)
    cleaned = _NON_PRINTABLE_RE.sub("", text)
    cleaned = _BASE64_LIKE_RE.sub("", cleaned)

    removed = original_len - len(cleaned)
    if removed > 0:
        log.warning(
            "pdf_sanitizer_removed_content",
            source=source,
            chars_removed=removed,
        )
    return cleaned


def _parse_sections_toml(toml_path: Path) -> dict[str, dict[tuple[int, int], str]]:
    """Parse a sections.toml file. Returns {pdf_name: {(start_page, end_page): heading}}."""
    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    result: dict[str, dict[tuple[int, int], str]] = {}
    for pdf_name, page_map in raw.items():
        result[pdf_name] = {}
        for range_str, heading in page_map.items():
            parts = str(range_str).split("-")
            if len(parts) == 2:
                try:
                    start = int(parts[0])
                    end = int(parts[1])
                    result[pdf_name][(start, end)] = heading
                except ValueError:
                    log.warning("sections_toml_invalid_range", range=range_str)
    return result


def _heading_level_from_hierarchy(heading: str) -> tuple[list[str], int]:
    """Parse 'Chapter 1 > Basics > Detail' into hierarchy and level."""
    parts = [p.strip() for p in heading.split(">")]
    return parts, len(parts)


class PDFExtractor(Extractor):
    """Extract text and sections from PDF files using PyMuPDF."""

    def supported_extensions(self) -> set[str]:
        return {".pdf"}

    def extract(self, file_path: Path) -> ExtractedDocument:
        doc = fitz.open(str(file_path))
        source = file_path.name

        # Check for manual sections.toml override
        sections_toml_path = file_path.parent / "sections.toml"
        manual_sections: dict[tuple[int, int], str] = {}
        if sections_toml_path.exists():
            try:
                all_sections = _parse_sections_toml(sections_toml_path)
                manual_sections = all_sections.get(file_path.name, {})
                if manual_sections:
                    log.info("pdf_using_manual_sections", source=source, count=len(manual_sections))
            except Exception as e:
                log.warning("pdf_sections_toml_error", source=source, error=str(e))

        if manual_sections:
            return self._extract_with_manual_sections(doc, manual_sections, source)

        # Layer 1: Table of Contents
        toc = doc.get_toc()
        if toc:
            return self._extract_with_toc(doc, toc, source)

        # Layer 2: Font-size heuristic
        font_sections = self._extract_with_font_heuristic(doc, source)
        if font_sections:
            return font_sections

        # Layer 3: Page-level fallback
        return self._extract_page_fallback(doc, source)

    def _get_page_text_blocks(self, page: fitz.Page, source: str) -> list[str]:
        """Extract text blocks from a page, filtering to type==0 (text) only."""
        blocks_data = page.get_text("dict")
        texts: list[str] = []
        for block in blocks_data.get("blocks", []):
            if block.get("type") != 0:
                continue
            block_text_parts: list[str] = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text_parts.append(span.get("text", ""))
            block_text = " ".join(block_text_parts).strip()
            if block_text:
                sanitized = _sanitize(block_text, source=source)
                if sanitized.strip():
                    texts.append(sanitized)
        return texts

    def _extract_with_manual_sections(
        self,
        doc: fitz.Document,
        manual_sections: dict[tuple[int, int], str],
        source: str,
    ) -> ExtractedDocument:
        """Extract using manually specified page ranges."""
        # Build page -> section heading map (1-indexed pages)
        page_to_heading: dict[int, tuple[list[str], int]] = {}
        for (start, end), heading in manual_sections.items():
            hierarchy, level = _heading_level_from_hierarchy(heading)
            for page_num in range(start, end + 1):
                page_to_heading[page_num] = (hierarchy, level)

        sections: list[Section] = []
        current_hierarchy: list[str] = []
        current_level: int | None = None
        current_texts: list[str] = []
        current_page: int | None = None

        def _flush_current(page: int | None) -> None:
            body = "\n".join(current_texts).strip()
            heading_str = current_hierarchy[-1] if current_hierarchy else None
            sections.append(
                Section(
                    text=body,
                    heading=heading_str,
                    heading_level=current_level,
                    page=page,
                )
            )

        def _emit_parent_headings(prev: list[str], new: list[str], page: int | None) -> None:
            """Emit synthetic heading-only sections for parent levels that changed."""
            for depth, heading_text in enumerate(new[:-1], start=1):
                if depth > len(prev) or prev[depth - 1] != heading_text:
                    sections.append(
                        Section(
                            text="",
                            heading=heading_text,
                            heading_level=depth,
                            page=page,
                        )
                    )

        for page_idx in range(len(doc)):
            page_num = page_idx + 1  # 1-indexed
            page = doc[page_idx]
            texts = self._get_page_text_blocks(page, source)

            if page_num in page_to_heading:
                # Flush current section
                if current_texts or current_hierarchy:
                    _flush_current(current_page)

                new_hierarchy, new_level = page_to_heading[page_num]
                # Emit synthetic sections for any new parent headings in the hierarchy
                _emit_parent_headings(current_hierarchy, new_hierarchy, page_num)

                current_hierarchy = new_hierarchy
                current_level = new_level
                current_texts = texts
                current_page = page_num
            else:
                current_texts.extend(texts)

        if current_texts or current_hierarchy:
            _flush_current(current_page)

        title = None
        if sections:
            for s in sections:
                if s.heading:
                    title = s.heading
                    break

        return ExtractedDocument(sections=sections, title=title, metadata={"source": "pdf"})

    def _extract_with_toc(
        self,
        doc: fitz.Document,
        toc: list[list[Any]],
        source: str,
    ) -> ExtractedDocument:
        """Extract using the document's table of contents."""
        # toc entries: [level, title, page_number]
        # Build page ranges from ToC
        entries: list[tuple[int, str, int]] = []
        for entry in toc:
            level, title, page = entry[0], entry[1], entry[2]
            entries.append((level, title, page))

        if not entries:
            return self._extract_page_fallback(doc, source)

        sections: list[Section] = []
        title: str | None = entries[0][1] if entries else None

        for i, (level, heading, start_page) in enumerate(entries):
            end_page = entries[i + 1][2] - 1 if i + 1 < len(entries) else len(doc)
            texts: list[str] = []
            for page_idx in range(start_page - 1, min(end_page, len(doc))):
                page = doc[page_idx]
                texts.extend(self._get_page_text_blocks(page, source))

            body = "\n".join(texts).strip()
            if body:
                sections.append(
                    Section(
                        text=body,
                        heading=heading,
                        heading_level=level,
                        page=start_page,
                    )
                )

        return ExtractedDocument(sections=sections, title=title, metadata={"source": "pdf"})

    def _extract_with_font_heuristic(
        self, doc: fitz.Document, source: str
    ) -> ExtractedDocument | None:
        """Detect headings by font size. Returns None if heuristic can't find headings."""
        # Collect all span font sizes
        all_sizes: list[float] = []
        for page in doc:
            blocks_data = page.get_text("dict")
            for block in blocks_data.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = span.get("size", 0)
                        text = span.get("text", "").strip()
                        if text:
                            all_sizes.append(size)

        if not all_sizes:
            return None

        median_size = statistics.median(all_sizes)
        heading_threshold = median_size * _HEADING_FONT_RATIO

        # Find distinct heading sizes
        heading_sizes = sorted(
            {s for s in all_sizes if s > heading_threshold},
            reverse=True,
        )
        if not heading_sizes:
            return None

        # Map sizes to levels (largest = h1, next = h2, etc.)
        size_to_level: dict[float, int] = {}
        for idx, size in enumerate(heading_sizes[:3]):
            size_to_level[size] = idx + 1

        sections: list[Section] = []
        current_heading: str | None = None
        current_level: int | None = None
        current_texts: list[str] = []
        current_page: int | None = None
        title: str | None = None

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            blocks_data = page.get_text("dict")

            for block in blocks_data.get("blocks", []):
                if block.get("type") != 0:
                    continue

                block_parts: list[str] = []
                block_max_size = 0.0

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = span.get("size", 0)
                        text = span.get("text", "")
                        block_parts.append(text)
                        if size > block_max_size:
                            block_max_size = size

                block_text = " ".join(block_parts).strip()
                if not block_text:
                    continue
                block_text = _sanitize(block_text, source=source)
                if not block_text.strip():
                    continue

                if block_max_size in size_to_level:
                    # This block is a heading
                    if current_texts or current_heading is not None:
                        body = "\n".join(current_texts).strip()
                        sections.append(
                            Section(
                                text=body,
                                heading=current_heading,
                                heading_level=current_level,
                                page=current_page,
                            )
                        )
                    level = size_to_level[block_max_size]
                    current_heading = block_text.strip()
                    current_level = level
                    current_texts = []
                    current_page = page_num
                    if title is None and level == 1:
                        title = current_heading
                else:
                    current_texts.append(block_text)
                    if current_page is None:
                        current_page = page_num

        if current_texts or current_heading is not None:
            sections.append(
                Section(
                    text="\n".join(current_texts).strip(),
                    heading=current_heading,
                    heading_level=current_level,
                    page=current_page,
                )
            )

        if not sections:
            return None

        return ExtractedDocument(sections=sections, title=title, metadata={"source": "pdf"})

    def _extract_page_fallback(self, doc: fitz.Document, source: str) -> ExtractedDocument:
        """Fallback: one section per page."""
        sections: list[Section] = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            texts = self._get_page_text_blocks(page, source)
            body = "\n".join(texts).strip()
            if body:
                sections.append(
                    Section(text=body, heading=None, heading_level=None, page=page_num)
                )
        return ExtractedDocument(sections=sections, title=None, metadata={"source": "pdf"})
