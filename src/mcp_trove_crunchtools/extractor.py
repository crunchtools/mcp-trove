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

_BASE_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS | DOCX_EXTENSIONS


def _get_supported_extensions() -> set[str]:
    """Build supported extensions, including vision types when configured."""
    from .vision import get_backend

    extensions = set(_BASE_EXTENSIONS)
    if get_backend() is not None:
        extensions |= IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    return extensions


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
    """Extract text content from a file."""
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
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ExtractionError(str(path), str(exc)) from exc


def _extract_pdf(path: Path) -> str:
    try:
        import pymupdf4llm

        markdown: str = pymupdf4llm.to_markdown(str(path))
    except ImportError as exc:
        raise ExtractionError(str(path), "pymupdf4llm not installed") from exc
    except Exception as exc:
        raise ExtractionError(str(path), str(exc)) from exc
    return markdown


def _extract_docx(path: Path) -> str:
    try:
        import docx

        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError as exc:
        raise ExtractionError(str(path), "python-docx not installed") from exc
    except Exception as exc:
        raise ExtractionError(str(path), str(exc)) from exc


def _extract_exif(path: Path) -> str:
    """Extract EXIF metadata from an image, or empty string if unavailable."""
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
            return _format_exif(exif_data, GPSTAGS)
    except Exception:
        logger.debug("Failed to read EXIF from %s", path, exc_info=True)
        return ""


def _format_exif(exif_data: Any, gpstags: dict[int, str]) -> str:
    """Format EXIF data into a searchable metadata string."""
    fields: list[str] = []

    for tag_id in (36867, 36868, 306):
        date_val = exif_data.get(tag_id)
        if date_val:
            fields.append(f"Date: {date_val}")
            break

    make = exif_data.get(271, "")
    camera_model = exif_data.get(272, "")
    if camera_model:
        if make and make not in camera_model:
            camera = f"{make} {camera_model}".strip()
        else:
            camera = camera_model
        fields.append(f"Camera: {camera}")

    gps_ifd = exif_data.get_ifd(0x8825)
    if gps_ifd:
        coords = _format_gps(gps_ifd, gpstags)
        if coords:
            fields.append(f"GPS: {coords}")

    img_width = exif_data.get(256)
    img_height = exif_data.get(257)
    if img_width and img_height:
        fields.append(f"Size: {img_width}x{img_height}")

    if not fields:
        return ""
    return "[EXIF] " + " | ".join(fields)


def _format_gps(gps_ifd: dict[int, object], gpstags: dict[int, str]) -> str:
    """Format GPS EXIF IFD into a decimal coordinate string."""
    tagged: dict[str, object] = {}
    for key, val in gps_ifd.items():
        tagged[gpstags.get(key, str(key))] = val

    lat_dms = tagged.get("GPSLatitude")
    lat_ref = tagged.get("GPSLatitudeRef")
    lon_dms = tagged.get("GPSLongitude")
    lon_ref = tagged.get("GPSLongitudeRef")

    if not (lat_dms and lat_ref and lon_dms and lon_ref):
        return ""

    try:
        lat_deg = _dms_to_decimal(cast_dms(lat_dms))
        lon_deg = _dms_to_decimal(cast_dms(lon_dms))
    except (TypeError, ValueError, ZeroDivisionError):
        return ""
    else:
        if lat_ref == "S":
            lat_deg = -lat_deg
        if lon_ref == "W":
            lon_deg = -lon_deg
        return f"{lat_deg:.4f}\u00b0N, {lon_deg:.4f}\u00b0E"


def cast_dms(val: object) -> tuple[float, float, float]:
    """Cast an EXIF DMS value to a typed tuple for decimal conversion."""
    seq: tuple[Any, ...] = tuple(val)  # type: ignore[arg-type]
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _dms_to_decimal(dms: tuple[float, float, float]) -> float:
    """Convert degrees/minutes/seconds to decimal degrees."""
    return dms[0] + dms[1] / 60.0 + dms[2] / 3600.0


def _extract_image(path: Path) -> str:
    """Extract text from an image via vision backend + EXIF metadata."""
    from .vision import get_backend

    backend = get_backend()
    if backend is None:
        raise ExtractionError(str(path), "No vision backend configured")

    parts: list[str] = [f"[File] {path.name}"]

    exif = _extract_exif(path)
    if exif:
        parts.append(exif)

    caption = backend.caption(path, "image")
    parts.append(f"[Caption] {caption}")

    return "\n".join(parts)


def _extract_video(path: Path) -> str:
    """Extract text from a video via vision backend."""
    from .vision import get_backend

    backend = get_backend()
    if backend is None:
        raise ExtractionError(str(path), "No vision backend configured")

    caption = backend.caption(path, "video")
    return f"[File] {path.name}\n[Caption] {caption}"
