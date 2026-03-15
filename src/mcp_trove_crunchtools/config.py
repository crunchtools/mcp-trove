"""Configuration for mcp-trove-crunchtools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import SecretStr

_config: Config | None = None

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_INDEX_WORKERS = 2
DEFAULT_INDEX_BATCH = 50
DEFAULT_EXCLUDE_PATTERNS = (
    "*.iso,*.zip,*.tar.gz,*.tar.bz2,*.7z,*.rar,"
    "*.exe,*.dll,*.bin,*.dat"
)

DEFAULT_VISION_PROMPT = (
    "Generate a concise caption for this image suitable for search indexing. "
    "Include: what you see, location if identifiable, time of day, any readable "
    "text or signs."
)


class Config:
    """Trove configuration from environment variables.

    No API credentials required. The _api_token field is typed as
    SecretStr | None for constitution compliance and future extensibility.
    """

    _api_token: SecretStr | None = None

    def __init__(self) -> None:
        default_db = str(
            Path.home() / ".local" / "share" / "mcp-trove" / "trove.db"
        )
        self.db_path: str = os.environ.get("TROVE_DB", default_db)
        self.index_paths: list[str] = _parse_paths(
            os.environ.get("TROVE_PATHS", "")
        )
        self.index_workers: int = int(
            os.environ.get("TROVE_INDEX_WORKERS", str(DEFAULT_INDEX_WORKERS))
        )
        self.index_batch: int = int(
            os.environ.get("TROVE_INDEX_BATCH", str(DEFAULT_INDEX_BATCH))
        )
        self.embedding_model: str = os.environ.get(
            "TROVE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.exclude_patterns: list[str] = _parse_exclude(
            os.environ.get("TROVE_EXCLUDE_PATTERNS", DEFAULT_EXCLUDE_PATTERNS)
        )
        self.chunk_size: int = int(
            os.environ.get("TROVE_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))
        )
        self.chunk_overlap: int = int(
            os.environ.get("TROVE_CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))
        )
        self.vision_backend: str = os.environ.get(
            "TROVE_VISION_BACKEND", "none"
        ).lower()
        self.vision_model: str = os.environ.get(
            "TROVE_VISION_MODEL", self._default_vision_model()
        )
        self.vision_prompt: str = os.environ.get(
            "TROVE_VISION_PROMPT", DEFAULT_VISION_PROMPT
        )

    def _default_vision_model(self) -> str:
        """Return default model name based on vision backend."""
        defaults = {
            "gemini": "gemini-2.5-flash",
            "openai": "gpt-4o-mini",
            "ollama": "llava",
        }
        return defaults.get(self.vision_backend, "")

    def ensure_db_dir(self) -> None:
        """Create the database directory if it does not exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)


def _parse_paths(raw: str) -> list[str]:
    """Parse colon-separated directory paths."""
    if not raw.strip():
        return []
    return [p.strip() for p in raw.split(":") if p.strip()]


def _parse_exclude(raw: str) -> list[str]:
    """Parse comma-separated glob patterns."""
    return [p.strip() for p in raw.split(",") if p.strip()]


def get_config() -> Config:
    """Get or create the singleton configuration."""
    global _config
    if _config is None:
        _config = Config()
    return _config
