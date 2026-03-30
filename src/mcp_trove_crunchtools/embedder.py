"""Embedding generation using fastembed (ONNX runtime)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import get_config
from .errors import EmbeddingError

if TYPE_CHECKING:
    from fastembed import TextEmbedding

_model: TextEmbedding | None = None


def get_model() -> TextEmbedding:
    """Get or create the singleton embedding model."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        model_name = get_config().embedding_model
        _model = TextEmbedding(model_name=model_name)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts.

    Returns a list of float vectors, one per input text.
    """
    if not texts:
        return []
    try:
        model = get_model()
        embeddings = list(model.embed(texts))
        return [emb.tolist() for emb in embeddings]
    except Exception as exc:
        raise EmbeddingError(str(exc)) from exc


def embed_query(text: str) -> list[float]:
    """Generate an embedding for a single query text.

    Uses query_embed for models that distinguish queries from passages.
    """
    try:
        model = get_model()
        embeddings = list(model.query_embed(text))
        vec: list[float] = embeddings[0].tolist()
    except Exception as exc:
        raise EmbeddingError(str(exc)) from exc
    return vec


def get_vector_dims() -> int:
    """Return the output dimension length of the current embedding model."""
    model = get_model()
    return len(list(model.embed(["probe"]))[0])
