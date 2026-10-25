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
            f"'{model_name}' produces {model_dims}. Move the index to the new model with "
            f"`mcp-trove-crunchtools --reembed`, or restore the previous TROVE_EMBEDDING_MODEL."
        )


class EmbeddingModelMismatchError(TroveError):
    """Raised when the database was built with a different embedding model."""

    def __init__(self, db_model: str, model_name: str) -> None:
        super().__init__(
            f"Database vectors were built with embedding model '{db_model}' but "
            f"TROVE_EMBEDDING_MODEL is '{model_name}'. Vectors from two models cannot be "
            f"searched together: restore TROVE_EMBEDDING_MODEL={db_model}, or move the "
            f"index to the new model with `mcp-trove-crunchtools --reembed`."
        )


class IndexAbortedError(TroveError):
    """Raised when a run is stopped by a safety check rather than a bug."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Indexing {path} aborted: {reason}")


class VisionDeferredError(TroveError):
    """Raised when this run can make no more vision calls; the file waits for the next run.

    Either the run's TROVE_VISION_MAX_CALLS_PER_RUN is spent, or the provider
    refused for want of credit (reason says which).
    """

    def __init__(self, path: str, limit: int, reason: str | None = None) -> None:
        why = reason or (
            f"the vision budget of {limit} calls for this run is spent "
            "(TROVE_VISION_MAX_CALLS_PER_RUN)"
        )
        super().__init__(f"{path} is deferred to the next run: {why}")


class UnsupportedFileTypeError(TroveError):
    """Raised when a file type is not supported for indexing."""

    def __init__(self, path: str, suffix: str) -> None:
        super().__init__(f"Unsupported file type '{suffix}' for: {path}")


class InvalidInputError(TroveError):
    """Raised when tool input fails Pydantic validation (length limits, unknown fields, etc.)."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Invalid input: {reason}")


class ReindexAllNotConfirmedError(InvalidInputError):
    """Raised when a whole-index reindex is asked for without force=True."""

    def __init__(self) -> None:
        super().__init__(
            "Reindexing the whole index re-reads every file, and re-captions any image "
            "or video whose cached caption came from another vision model. Pass a path, "
            "or force=true to reindex everything."
        )
