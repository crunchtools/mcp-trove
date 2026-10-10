"""Error hierarchy for mcp-trove-crunchtools."""

from __future__ import annotations


class TroveError(Exception):
    """Base error for all trove operations."""


class FileNotIndexedError(TroveError):
    """Raised when a file path is not in the index."""

    def __init__(self, path: str) -> None:
        super().__init__(f"File not indexed: {path}")


class PathNotFoundError(TroveError):
    """Raised when a filesystem path does not exist."""

    def __init__(self, path: str) -> None:
        super().__init__(f"Path not found: {path}")


class ExtractionError(TroveError):
    """Raised when text extraction from a file fails."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Failed to extract text from {path}: {reason}")


class EmbeddingError(TroveError):
    """Raised when embedding generation fails."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Embedding failed: {reason}")


class UnknownEmbeddingModelError(TroveError):
    """Raised when fastembed has no metadata for the configured model."""

    def __init__(self, model_name: str) -> None:
        super().__init__(f"Unknown embedding model '{model_name}': cannot determine dimensions")


class DimensionMismatchError(TroveError):
    """Raised when the database vector size differs from the configured model."""

    def __init__(self, db_dims: int, model_name: str, model_dims: int) -> None:
        super().__init__(
            f"Database vectors have {db_dims} dimensions but embedding model "
            f"'{model_name}' produces {model_dims}. Reindex into a new database "
            f"(set TROVE_DB to a new path) or restore the previous TROVE_EMBEDDING_MODEL."
        )


class UnsupportedFileTypeError(TroveError):
    """Raised when a file type is not supported for indexing."""

    def __init__(self, path: str, suffix: str) -> None:
        super().__init__(f"Unsupported file type '{suffix}' for: {path}")


class InvalidInputError(TroveError):
    """Raised when tool input fails Pydantic validation (length limits, unknown fields, etc.)."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Invalid input: {reason}")
