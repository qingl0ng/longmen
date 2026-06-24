"""Sentence-transformers embedder wrapper."""

from __future__ import annotations

import structlog
from sentence_transformers import SentenceTransformer

log = structlog.get_logger(__name__)


class Embedder:
    """Loads a sentence-transformers model and embeds text."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """Load the sentence-transformers model."""
        self._model_name = model_name
        log.info("loading_embedding_model", model=model_name, device=device)
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_embedding_dimension()
        log.info("embedding_model_loaded", model=model_name, dimension=self._dimension)

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed a batch of texts. Returns normalized vectors."""
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    @property
    def dimension(self) -> int:
        """Vector dimension of the loaded model."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Full model name string."""
        return self._model_name
