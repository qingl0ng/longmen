"""Abstract base interface for text extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Section:
    text: str
    heading: str | None = None
    heading_level: int | None = None
    page: int | None = None


@dataclass
class ExtractedDocument:
    sections: list[Section] = field(default_factory=list)
    title: str | None = None
    metadata: dict = field(default_factory=dict)


class Extractor(ABC):
    @abstractmethod
    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract structured text from a file."""

    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """File extensions this extractor handles (e.g., {'.md', '.markdown'})."""


class UnsupportedFormatError(Exception):
    """Raised when no extractor is available for a given file extension."""
