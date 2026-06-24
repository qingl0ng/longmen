"""Markdown text extractor."""

from __future__ import annotations

import re
from pathlib import Path

import structlog

from .base import ExtractedDocument, Extractor, Section

log = structlog.get_logger(__name__)

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


class MarkdownExtractor(Extractor):
    """Extract sections from Markdown files, splitting on heading lines."""

    def supported_extensions(self) -> set[str]:
        return {".md", ".markdown", ".mkd"}

    def extract(self, file_path: Path) -> ExtractedDocument:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        # Strip HTML comments
        text = _HTML_COMMENT_RE.sub("", text)

        lines = text.splitlines(keepends=True)
        sections: list[Section] = []
        current_lines: list[str] = []
        current_heading: str | None = None
        current_level: int | None = None
        in_fence = False
        fence_char = ""
        title: str | None = None

        def flush() -> None:
            body = "".join(current_lines).strip()
            if body or current_heading is not None:
                sections.append(
                    Section(
                        text=body,
                        heading=current_heading,
                        heading_level=current_level,
                        page=None,
                    )
                )

        for line in lines:
            stripped = line.rstrip("\n").rstrip("\r")

            # Track code fence state
            if not in_fence:
                fence_match = re.match(r"^(`{3,}|~{3,})", stripped)
                if fence_match:
                    in_fence = True
                    fence_char = fence_match.group(1)[0]
                    current_lines.append(line)
                    continue
            else:
                # Check for closing fence (same char, same or greater length)
                fence_close = re.match(r"^(`{3,}|~{3,})\s*$", stripped)
                if fence_close and fence_close.group(1)[0] == fence_char:
                    in_fence = False
                    fence_char = ""
                current_lines.append(line)
                continue

            # Outside fence: check for heading
            heading_match = _HEADING_RE.match(stripped)
            if heading_match and not in_fence:
                flush()
                current_lines = []
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                current_heading = heading_text
                current_level = level
                if title is None and level == 1:
                    title = heading_text
            else:
                current_lines.append(line)

        flush()

        return ExtractedDocument(sections=sections, title=title, metadata={"source": "markdown"})
