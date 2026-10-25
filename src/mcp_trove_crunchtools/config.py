"""Configuration for mcp-trove-crunchtools."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import SecretStr

_config: Config | None = None

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_INDEX_WORKERS = 20
DEFAULT_INDEX_BATCH = 50
DEFAULT_VISION_TIMEOUT = 120  # seconds per vision API call
DEFAULT_PDF_MAX_PAGES = 5  # leading PDF pages indexed; 0 or negative = all
DEFAULT_MAX_TEXT_CHARS = 5_000_000  # extracted text kept per file (~6k chunks)
DEFAULT_MAX_EXTRACT_TIMEOUTS = 5  # timed-out extractions before a run aborts
DEFAULT_VISION_MAX_CALLS = 3000  # vision API calls per run (~$3); 0 = no limit
DEFAULT_BACKUP_KEEP = 3  # nightly database copies kept; 0 disables backups
DEFAULT_EXCLUDE_PATTERNS = "*.iso,*.zip,*.tar.gz,*.tar.bz2,*.7z,*.rar,*.exe,*.dll,*.bin,*.dat"

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
        default_db = str(Path.home() / ".local" / "share" / "mcp-trove" / "trove.db")
        self.db_path: str = os.environ.get("TROVE_DB", default_db)
        self.index_paths: list[str] = _parse_paths(os.environ.get("TROVE_PATHS", ""))
        # Inboxes, such as a phone's upload folder, whose files are filed away
        # elsewhere: emptying out is normal, so the shrink and removal guards
        # that catch a dead mount are not applied to them.
        self.inbox_paths: list[str] = _parse_paths(os.environ.get("TROVE_INBOX_PATHS", ""))
        self.index_workers: int = int(
            os.environ.get("TROVE_INDEX_WORKERS", str(DEFAULT_INDEX_WORKERS))
        )
        self.index_batch: int = int(os.environ.get("TROVE_INDEX_BATCH", str(DEFAULT_INDEX_BATCH)))
        self.embedding_model: str = os.environ.get("TROVE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.exclude_patterns: list[str] = _parse_exclude(
            os.environ.get("TROVE_EXCLUDE_PATTERNS", DEFAULT_EXCLUDE_PATTERNS)
        )
        self.chunk_size: int = int(os.environ.get("TROVE_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
        self.chunk_overlap: int = int(
            os.environ.get("TROVE_CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))
        )
        self.pdf_max_pages: int = int(
            os.environ.get("TROVE_PDF_MAX_PAGES", str(DEFAULT_PDF_MAX_PAGES))
        )
        self.max_text_chars: int = _parse_max_text_chars(
            os.environ.get("TROVE_MAX_TEXT_CHARS", str(DEFAULT_MAX_TEXT_CHARS))
        )
        db_dir = Path(self.db_path).parent
        # A mount point the index paths must sit on; checked before each run so
        # an unmounted or hung source fails loudly instead of indexing nothing.
        self.require_mount: str | None = os.environ.get("TROVE_REQUIRE_MOUNT") or None
        self.max_extract_timeouts: int = int(
            os.environ.get("TROVE_MAX_EXTRACT_TIMEOUTS", str(DEFAULT_MAX_EXTRACT_TIMEOUTS))
        )
        self.backup_dir: str = os.environ.get("TROVE_BACKUP_DIR", str(db_dir / "backups"))
        self.backup_keep: int = int(os.environ.get("TROVE_BACKUP_KEEP", str(DEFAULT_BACKUP_KEEP)))
        self.status_file: str = os.environ.get("TROVE_STATUS_FILE", str(db_dir / "last_run.json"))
        # Directory of per-path file listings written by the host before a run
        # (see manifest.py); unset, the scan walks the tree.
        self.manifest_dir: str = os.environ.get("TROVE_MANIFEST_DIR", "")
        # ONNX threads for embedding. fastembed's default is one spinning thread
        # per core, and the MCP server loads a second model on the same host.
        self.embed_threads: int = int(os.environ.get("TROVE_EMBED_THREADS", "2"))
        # Owner-only unless a monitor running as another user needs it: the
        # summary can carry error text that names files.
        self.status_file_mode: int = int(os.environ.get("TROVE_STATUS_FILE_MODE", "600"), 8)
        self.vision_backend: str = os.environ.get("TROVE_VISION_BACKEND", "none").lower()
        self.vision_model: str = os.environ.get("TROVE_VISION_MODEL", self._default_vision_model())
        self.vision_prompt: str = os.environ.get("TROVE_VISION_PROMPT", DEFAULT_VISION_PROMPT)
        self.vision_timeout: int = int(
            os.environ.get("TROVE_VISION_TIMEOUT", str(DEFAULT_VISION_TIMEOUT))
        )
        # Caps what one run can spend: files past the cap are deferred to the
        # next run, not failed, so a big new folder is captioned over nights.
        self.vision_max_calls: int = int(
            os.environ.get("TROVE_VISION_MAX_CALLS_PER_RUN", str(DEFAULT_VISION_MAX_CALLS))
        )
        # OpenRouter only: a second model tried when the first is rate limited.
        # Unset by default, because the usual fallback is a pricier tier.
        self.vision_fallback_model: str = os.environ.get("TROVE_VISION_FALLBACK_MODEL", "")
        # Off unless asked for: captioning a scanned PDF sends an image of its
        # first page (a bank statement, a tax form) to the vision provider.
        self.vision_pdf_scans: bool = os.environ.get("TROVE_VISION_PDF_SCANS", "").lower() in {
            "1",
            "true",
            "yes",
        }

    def _default_vision_model(self) -> str:
        """Return default model name based on vision backend."""
        defaults = {
            "gemini": "gemini-2.5-flash",
            "openai": "gpt-4o-mini",
            "openrouter": "google/gemini-3.1-flash-lite",
            "ollama": "llava",
        }
        return defaults.get(self.vision_backend, "")

    def ensure_db_dir(self) -> None:
        """Create the database directory if it does not exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)


def _parse_paths(raw: str) -> list[str]:
    """Parse directory paths separated by os.pathsep (":" on POSIX, ";" on Windows)."""
    if not raw.strip():
        return []
    return [p.strip() for p in raw.split(os.pathsep) if p.strip()]


def _parse_max_text_chars(raw: str) -> int:
    """Parse the per-file text cap, which must stay finite.

    The cap is what bounds one file's chunks and embeddings in memory, so a
    non-positive value falls back to the default instead of disabling it.
    """
    max_chars = int(raw)
    if max_chars > 0:
        return max_chars
    print(
        f"[WARNING] TROVE_MAX_TEXT_CHARS={max_chars} is not positive. "
        f"Using the default of {DEFAULT_MAX_TEXT_CHARS}.",
        file=sys.stderr,
    )
    return DEFAULT_MAX_TEXT_CHARS


def _parse_exclude(raw: str) -> list[str]:
    """Parse comma-separated glob patterns."""
    return [p.strip() for p in raw.split(",") if p.strip()]


def get_config() -> Config:
    """Get or create the singleton configuration."""
    global _config
    if _config is None:
        _config = Config()
    return _config
