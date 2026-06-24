"""File and image attachment — read, base64 encode, build content blocks."""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Any

_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
_WARN_FILE_SIZE = 1 * 1024 * 1024  # 1 MB

_IMAGE_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

# Match @path: must be preceded by whitespace or start-of-string.
# Does NOT match when @ is preceded by a word char (e.g. email addresses).
_ATTACH_RE = re.compile(r"(?<!\w)@(\S+)")


def parse_attachments(
    text: str,
    cwd: Path | None = None,
    file_index: set[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Scan text for @path references.

    Returns (cleaned_text, content_blocks).
    Raises ValueError if a locally-read file exceeds the size limit.

    Decision tree for each @ref:
    1. ref is in file_index → emit file_ref block, strip from text.
    2. ref not in index but readable locally → emit FileContent block, strip from text.
    3. ref not in index, not readable locally → leave @ref as literal text.
    """
    if file_index is None:
        file_index = set()
    cwd = cwd or Path.cwd()
    blocks: list[dict[str, Any]] = []
    replacements: list[tuple[str, str]] = []  # (original_match, replacement)

    for match in _ATTACH_RE.finditer(text):
        ref = match.group(1)

        # Case 1: in file index — emit file_ref
        if ref in file_index:
            blocks.append({"type": "file_ref", "path": ref})
            replacements.append((match.group(0), ""))
            continue

        # Case 2: readable locally — emit FileContent
        path = Path(ref).expanduser()
        if not path.is_absolute():
            path = cwd / path

        if path.exists():
            size = path.stat().st_size
            if size > _MAX_FILE_SIZE:
                raise ValueError(
                    f"File {ref!r} ({size / 1024 / 1024:.1f} MB) exceeds 10 MB limit"
                )

            data = base64.b64encode(path.read_bytes()).decode()

            image_media_type = _IMAGE_MEDIA_TYPES.get(path.suffix.lower())
            if image_media_type:
                blocks.append({
                    "type": "image",
                    "media_type": image_media_type,
                    "data": data,
                })
            else:
                media_type, _ = mimetypes.guess_type(str(path))
                if media_type is None:
                    media_type = "text/plain"
                blocks.append({
                    "type": "file",
                    "filename": path.name,
                    "media_type": media_type,
                    "data": data,
                })
            replacements.append((match.group(0), ""))
        # Case 3: not in index, not readable — leave as literal text (no replacement)

    cleaned = text
    for original, replacement in replacements:
        cleaned = cleaned.replace(original, replacement, 1)

    # Clean up extra whitespace left by removed @refs
    cleaned = re.sub(r"  +", " ", cleaned).strip()

    return cleaned, blocks
