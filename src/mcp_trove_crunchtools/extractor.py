"""Text extraction from various file formats."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .errors import ExtractionError, UnsupportedFileTypeError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h",
    ".sh", ".bash", ".zsh", ".fish",
    ".html", ".htm", ".xml", ".svg",
    ".sql", ".r", ".rb", ".pl", ".lua",
    ".tex", ".bib", ".org",
    ".log", ".conf", ".env",
}

PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}
MARKDOWN_EXTENSIONS = {".md", ".markdown"}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".heic", ".heif",
    ".webp", ".bmp", ".tiff", ".tif",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}

# Base extensions always supported
_BASE_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS | DOCX_EXTENSIONS


def _get_supported_extensions() -> set[str]:
    """Build supported extensions set, conditionally including vision types."""
    from .vision import get_backend

    extensions = set(_BASE_EXTENSIONS)
    if get_backend() is not None:
        extensions |= IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    return extensions


# Keep module-level name for backward compatibility, but make it dynamic
SUPPORTED_EXTENSIONS = _BASE_EXTENSIONS


def detect_file_type(path: Path) -> str:
    """Detect file type from extension."""
    suffix = path.suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return "pdf"
    if suffix in DOCX_EXTENSIONS:
        return "docx"
    if suffix in MARKDOWN_EXTENSIONS:
        return "markdown"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    msg = suffix or "(no extension)"
    raise UnsupportedFileTypeError(str(path), msg)


def is_supported(path: Path) -> bool:
    """Check if a file type is supported for extraction."""
    return path.suffix.lower() in _get_supported_extensions()


def extract_text(path: Path) -> str:
    """Extract text content from a file.

    Returns the full text content as a string.
    """
    file_type = detect_file_type(path)

    if file_type == "pdf":
        return _extract_pdf(path)
    if file_type == "docx":
        return _extract_docx(path)
    if file_type == "image":
        return _extract_image(path)
    if file_type == "video":
        return _extract_video(path)
    return _extract_text_file(path)


def _extract_text_file(path: Path) -> str:
    """Extract text from a plain text file."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ExtractionError(str(path), str(exc)) from exc


def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF file using pymupdf4llm."""
    try:
        import pymupdf4llm

        result: str = pymupdf4llm.to_markdown(str(path))
    except ImportError as exc:
        raise ExtractionError(
            str(path), "pymupdf4llm not installed"
        ) from exc
    except Exception as exc:
        raise ExtractionError(str(path), str(exc)) from exc
    return result


def _extract_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        import docx

        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError as exc:
        raise ExtractionError(
            str(path), "python-docx not installed"
        ) from exc
    except Exception as exc:
        raise ExtractionError(str(path), str(exc)) from exc


def _extract_exif(path: Path) -> str:
    """Extract EXIF metadata from an image file.

    Returns a formatted metadata string, or empty string if no EXIF data.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS
    except ImportError:
        return ""

    try:
        with Image.open(path) as img:
            exif_data = img.getexif()
            if not exif_data:
                return ""

            fields = _collect_exif_fields(exif_data, GPSTAGS)
            if not fields:
                return ""
            return "[EXIF] " + " | ".join(fields)

    except Exception:
        logger.debug("Failed to read EXIF from %s", path, exc_info=True)
        return ""


def _collect_exif_fields(
    exif_data: Any, gpstags: dict[int, str]
) -> list[str]:
    """Collect EXIF fields from parsed EXIF data."""
    fields: list[str] = []

    # Date/time (DateTimeOriginal, DateTimeDigitized, DateTime)
    for tag_id in (36867, 36868, 306):
        val = exif_data.get(tag_id)
        if val:
            fields.append(f"Date: {val}")
            break

    # Camera make/model
    make = exif_data.get(271, "")
    model = exif_data.get(272, "")
    if model:
        camera = f"{make} {model}".strip() if make and make not in model else model
        fields.append(f"Camera: {camera}")

    # GPS coordinates
    gps_info = exif_data.get_ifd(0x8825)
    if gps_info:
        gps = _parse_gps(gps_info, gpstags)
        if gps:
            fields.append(f"GPS: {gps}")

    # Image dimensions
    width = exif_data.get(256)
    height = exif_data.get(257)
    if width and height:
        fields.append(f"Size: {width}x{height}")

    return fields


def _parse_gps(gps_info: dict[int, object], gpstags: dict[int, str]) -> str:
    """Parse GPS EXIF data into a readable coordinate string."""
    tagged: dict[str, object] = {}
    for key, val in gps_info.items():
        tag_name = gpstags.get(key, str(key))
        tagged[tag_name] = val

    lat = tagged.get("GPSLatitude")
    lat_ref = tagged.get("GPSLatitudeRef")
    lon = tagged.get("GPSLongitude")
    lon_ref = tagged.get("GPSLongitudeRef")

    if not (lat and lat_ref and lon and lon_ref):
        return ""

    try:
        lat_deg = _dms_to_decimal(lat)  # type: ignore[arg-type]
        lon_deg = _dms_to_decimal(lon)  # type: ignore[arg-type]
    except (TypeError, ValueError, ZeroDivisionError):
        return ""
    else:
        if lat_ref == "S":
            lat_deg = -lat_deg
        if lon_ref == "W":
            lon_deg = -lon_deg
        return f"{lat_deg:.4f}\u00b0N, {lon_deg:.4f}\u00b0E"


def _dms_to_decimal(dms: tuple[float, float, float]) -> float:
    """Convert degrees/minutes/seconds tuple to decimal degrees."""
    return float(dms[0]) + float(dms[1]) / 60.0 + float(dms[2]) / 3600.0


def _extract_image(path: Path) -> str:
    """Extract text from an image via vision backend + EXIF metadata."""
    from .vision import get_backend

    backend = get_backend()
    if backend is None:
        raise ExtractionError(str(path), "No vision backend configured")

    parts: list[str] = []

    # EXIF metadata (free, no API call)
    exif = _extract_exif(path)
    if exif:
        parts.append(exif)

    # Vision caption (API call)
    try:
        caption = backend.caption(path, "image")
        parts.append(f"[Caption] {caption}")
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(str(path), f"Vision captioning failed: {exc}") from exc

    # File context
    parts.insert(0, f"[File] {path.name}")

    return "\n".join(parts)


def _extract_video(path: Path) -> str:
    """Extract text from a video via vision backend."""
    from .vision import get_backend

    backend = get_backend()
    if backend is None:
        raise ExtractionError(str(path), "No vision backend configured")

    parts: list[str] = [f"[File] {path.name}"]

    try:
        caption = backend.caption(path, "video")
        parts.append(f"[Caption] {caption}")
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(str(path), f"Vision captioning failed: {exc}") from exc

    return "\n".join(parts)
