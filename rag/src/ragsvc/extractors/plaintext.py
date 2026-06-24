"""Plain text and reStructuredText extractor."""

from __future__ import annotations

from pathlib import Path

import structlog

from .base import ExtractedDocument, Extractor, Section

log = structlog.get_logger(__name__)


def _detect_rst_heading(lines: list[str], idx: int) -> int | None:
    """Check if lines[idx] is a heading underlined by lines[idx+1].

    Returns the heading level (1 or 2) or None if not a heading.
    RST style: text line followed by === or --- of same or greater length.
    """
    if idx + 1 >= len(lines):
        return None
    line = lines[idx]
    underline = lines[idx + 1]
    text = line.rstrip()
    under = underline.rstrip()
    if not text or not under:
        return None
    if len(under) < len(text):
        return None
    if all(c == "=" for c in under):
        return 1
    if all(c == "-" for c in under):
        return 2
    return None


class PlainTextExtractor(Extractor):
    """Extract sections from plain text and reStructuredText files."""

    def supported_extensions(self) -> set[str]:
        return {".txt", ".text", ".rst"}

    def extract(self, file_path: Path) -> ExtractedDocument:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.split("\n")
        sections: list[Section] = []
        title: str | None = None

        # Split on double newlines (paragraph boundaries)
        # First, handle RST headings by scanning for underlined lines
        paragraphs: list[tuple[str, str | None, int | None]] = []  # (text, heading, level)

        i = 0
        while i < len(lines):
            # Check RST heading
            rst_level = _detect_rst_heading(lines, i)
            if rst_level is not None and lines[i].strip():
                heading_text = lines[i].strip()
                # Skip the heading line and its underline
                i += 2
                # Skip optional blank line after heading
                if i < len(lines) and not lines[i].strip():
                    i += 1
                # Collect body until next heading or double newline run
                body_lines: list[str] = []
                while i < len(lines):
                    rst2 = _detect_rst_heading(lines, i)
                    if rst2 is not None and lines[i].strip():
                        break
                    body_lines.append(lines[i])
                    i += 1
                body = _split_paragraphs("\n".join(body_lines))
                if title is None and rst_level == 1:
                    title = heading_text
                # First paragraph stays with the heading section; rest become their own
                first_para = body[0] if body else ""
                paragraphs.append((first_para, heading_text, rst_level))
                for para in body[1:]:
                    paragraphs.append((para, None, None))
                continue

            # Check all-caps heading (short lines only)
            line = lines[i].strip()
            if (
                line
                and line.isupper()
                and len(line) <= 80
                and len(line.split()) >= 1
                and not any(c.isdigit() for c in line)
            ):
                # Check if followed by content (not a separator)
                paragraphs.append((line, line, 1))
                i += 1
                if title is None:
                    title = line
                continue

            i += 1

        if not paragraphs:
            # Fallback: simple double-newline split
            raw_paragraphs = _split_paragraphs(text)
            for para in raw_paragraphs:
                stripped = para.strip()
                if stripped:
                    sections.append(Section(text=stripped, heading=None, heading_level=None))
        else:
            for para_text, heading, level in paragraphs:
                stripped = para_text.strip()
                if stripped or heading:
                    sections.append(
                        Section(text=stripped, heading=heading, heading_level=level)
                    )

        # If we have no sections from special detection, do simple paragraph split
        if not sections:
            raw_paragraphs = _split_paragraphs(text)
            for para in raw_paragraphs:
                stripped = para.strip()
                if stripped:
                    sections.append(Section(text=stripped, heading=None, heading_level=None))

        return ExtractedDocument(sections=sections, title=title, metadata={"source": "plaintext"})


def _split_paragraphs(text: str) -> list[str]:
    """Split text on double newlines."""
    parts = []
    current: list[str] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            # Check if there's another blank line coming (double newline)
            if current:
                parts.append("\n".join(current))
                current = []
            # Skip consecutive blank lines
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue
        current.append(lines[i])
        i += 1
    if current:
        parts.append("\n".join(current))
    return parts
