"""Extractor registry: pick extractor by file extension."""

from __future__ import annotations

from pathlib import Path

from .base import Extractor, UnsupportedFormatError
from .markdown import MarkdownExtractor
from .pdf import PDFExtractor
from .plaintext import PlainTextExtractor

_EXTRACTOR_INSTANCES: list[Extractor] = [
    MarkdownExtractor(),
    PlainTextExtractor(),
    PDFExtractor(),
]

_EXTENSION_MAP: dict[str, Extractor] = {}
for _extractor in _EXTRACTOR_INSTANCES:
    for _ext in _extractor.supported_extensions():
        _EXTENSION_MAP[_ext] = _extractor


def get_extractor(file_path: Path) -> Extractor:
    """Return the appropriate extractor based on file extension.

    Raises UnsupportedFormatError for unknown extensions.
    """
    ext = file_path.suffix.lower()
    extractor = _EXTENSION_MAP.get(ext)
    if extractor is None:
        raise UnsupportedFormatError(
            f"No extractor available for extension {ext!r} (file: {file_path})"
        )
    return extractor


SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_EXTENSION_MAP.keys())

__all__ = [
    "get_extractor",
    "SUPPORTED_EXTENSIONS",
    "Extractor",
    "UnsupportedFormatError",
    "MarkdownExtractor",
    "PlainTextExtractor",
    "PDFExtractor",
]
