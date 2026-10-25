"""What gets indexed: content categories, and the extractor for each extension.

TROVE_INDEX_TYPES picks categories (an allowlist). A file in none of them,
or in one that is off, is handled by TROVE_INDEX_OTHER: skipped, or indexed
by its name and folder alone so it can still be found.
"""

from __future__ import annotations

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".gif", ".heic", ".heif", ".webp", ".bmp", ".tiff", ".tif"}
)
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".webm", ".mkv"})

CATEGORIES: dict[str, frozenset[str]] = {
    "images": IMAGE_EXTENSIONS,
    "videos": VIDEO_EXTENSIONS,
    "pdf": frozenset({".pdf"}),
    "text": frozenset({".txt", ".md", ".markdown", ".rst", ".org", ".tex"}),
    "documents": frozenset({".docx", ".doc", ".odt", ".rtf"}),
    "spreadsheets": frozenset({".xlsx", ".xls", ".ods", ".csv", ".tsv"}),
    "presentations": frozenset({".pptx", ".odp"}),
    "web": frozenset({".html", ".htm"}),
    "data": frozenset({".json", ".yaml", ".yml", ".xml", ".toml", ".bib", ".svg"}),
    "code": frozenset(
        {
            ".py",
            ".js",
            ".ts",
            ".go",
            ".rs",
            ".java",
            ".c",
            ".cpp",
            ".h",
            ".sh",
            ".bash",
            ".zsh",
            ".fish",
            ".sql",
            ".r",
            ".rb",
            ".pl",
            ".lua",
            ".ini",
            ".cfg",
            ".conf",
            ".env",
            ".log",
        }
    ),
}

DEFAULT_INDEX_TYPES = "images,videos,pdf,text,documents,spreadsheets,presentations"
INDEX_OTHER_MODES = frozenset({"skip", "name"})

# The extractor for each extension; any other categorised extension is read
# as plain text. "name" is the TROVE_INDEX_OTHER=name entry, never read.
KINDS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".odt": "odf",
    ".odp": "odf",
    ".ods": "odf",
    ".rtf": "rtf",
    ".xlsx": "xlsx",
    ".xls": "xls",
    ".csv": "table",
    ".tsv": "table",
    ".pptx": "pptx",
    ".md": "markdown",
    ".markdown": "markdown",
    **dict.fromkeys(IMAGE_EXTENSIONS, "image"),
    **dict.fromkeys(VIDEO_EXTENSIONS, "video"),
}

VISION_KINDS = frozenset({"image", "video"})
NAME_KIND = "name"


def category_of(suffix: str) -> str | None:
    """The category an extension belongs to, or None."""
    for name, extensions in CATEGORIES.items():
        if suffix in extensions:
            return name
    return None


def is_media(suffix: str) -> bool:
    """Images and videos: their captions cost money, so they are never purged by policy."""
    return suffix in IMAGE_EXTENSIONS or suffix in VIDEO_EXTENSIONS


def parse_index_types(raw: str) -> frozenset[str]:
    """Parse TROVE_INDEX_TYPES, refusing unknown category names."""
    names = frozenset(p.strip().lower() for p in raw.split(",") if p.strip())
    unknown = sorted(names - CATEGORIES.keys())
    if unknown:
        msg = (
            f"TROVE_INDEX_TYPES has unknown categories {', '.join(unknown)}; "
            f"choose from {', '.join(CATEGORIES)}"
        )
        raise ValueError(msg)
    return names


def parse_index_other(raw: str) -> str:
    """Parse TROVE_INDEX_OTHER: skip or name."""
    mode = raw.strip().lower()
    if mode not in INDEX_OTHER_MODES:
        msg = f"TROVE_INDEX_OTHER must be skip or name, not {raw!r}"
        raise ValueError(msg)
    return mode
