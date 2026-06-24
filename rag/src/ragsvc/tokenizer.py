"""Qwen3 tokenizer wrapper for token counting."""

from __future__ import annotations

from pathlib import Path

import structlog
from tokenizers import Tokenizer

log = structlog.get_logger(__name__)


class TokenCounter:
    """Thin wrapper around the tokenizers library for token counting."""

    def __init__(self, path: str | Path) -> None:
        """Load tokenizer from a local tokenizer.json. Offline — never downloads.

        The caller passes an already-``~``-expanded path
        (``config.tokenizer.resolved_path()``); this constructor does not expand again.
        """
        log.info("loading_tokenizer", path=str(path))
        try:
            self._tokenizer: Tokenizer = Tokenizer.from_file(str(path))
        except Exception as exc:
            log.error("tokenizer_load_failed", path=str(path), error=str(exc))
            raise RuntimeError(
                f"failed to load tokenizer from {str(path)!r}: {exc}. Provide a "
                "valid local tokenizer.json ([tokenizer].path); this offline service "
                "never downloads tokenizers."
            ) from exc
        # Disable truncation/padding so we can count any length
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        log.info("tokenizer_loaded", path=str(path))

    def count(self, text: str) -> int:
        """Count tokens in a text string."""
        encoding = self._tokenizer.encode(text)
        return len(encoding.ids)

    def count_batch(self, texts: list[str]) -> list[int]:
        """Count tokens for a batch of texts."""
        encodings = self._tokenizer.encode_batch(texts)
        return [len(enc.ids) for enc in encodings]

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens. Cuts at token boundary."""
        encoding = self._tokenizer.encode(text)
        if len(encoding.ids) <= max_tokens:
            return text
        truncated_ids = encoding.ids[:max_tokens]
        return self._tokenizer.decode(truncated_ids)
