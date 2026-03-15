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


class UnsupportedFileTypeError(TroveError):
    """Raised when a file type is not supported for indexing."""

    def __init__(self, path: str, suffix: str) -> None:
        super().__init__(f"Unsupported file type '{suffix}' for: {path}")
