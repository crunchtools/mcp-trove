"""Text extraction from various file formats."""

from __future__ import annotations

import gc
import logging
import math
import tempfile
from pathlib import Path
from typing import Any

from .config import get_config
from .errors import ExtractionError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
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
    ".html",
    ".htm",
    ".xml",
    ".svg",
    ".sql",
    ".r",
    ".rb",
    ".pl",
    ".lua",
    ".tex",
    ".bib",
    ".org",
    ".log",
    ".conf",
    ".env",
}

PDF_EXTENSIONS = {".pdf"}
# Below this many characters of text on its leading pages, a PDF is treated
# as a scan and its first page is captioned through the vision backend.
SCANNED_PDF_MIN_CHARS = 20
SCANNED_PDF_CHECK_PAGES = 5  # leading pages looked at to decide
SCANNED_PDF_DPI = 150
# A PDF can declare an enormous page; the render is scaled down to stay
# under this many pixels (about 12 MB as RGB) whatever the page size.
SCANNED_PDF_MAX_PIXELS = 4_000_000
DOCX_EXTENSIONS = {".docx"}
MARKDOWN_EXTENSIONS = {".md", ".markdown"}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".heic",
    ".heif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
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


def extract_text(path: Path, checksum: str | None = None) -> str:
    """Extract text content from a file.

    checksum, the file's SHA-256, lets vision captions be reused across files
    with the same content; without it every caption is a new API call.
    """
    file_type = detect_file_type(path)

    if file_type == "pdf":
        return _extract_pdf(path, checksum)
    if file_type == "docx":
        return _extract_docx(path)
    if file_type == "image":
        return _extract_image(path, checksum)
    if file_type == "video":
        return _extract_video(path, checksum)
    return _extract_text_file(path)


def _extract_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ExtractionError(str(path), str(exc)) from exc


def _extract_pdf(path: Path, checksum: str | None = None) -> str:
    """Convert the first TROVE_PDF_MAX_PAGES pages of a PDF to markdown.

    Converting a whole 20-80 MB service manual OOM-killed a 1.5 GB indexer
    (RT #1542); the opening pages are enough to tell what a document is about.
    A scan has no text layer, so its first page is captioned instead, when a
    vision backend is configured and TROVE_VISION_PDF_SCANS allows it.
    """
    try:
        import pymupdf
        import pymupdf4llm
        from pymupdf4llm.helpers.pymupdf_rag import IdentifyHeaders
    except ImportError as exc:
        raise ExtractionError(str(path), "pymupdf4llm not installed") from exc

    from .vision import get_backend

    max_pages = get_config().pdf_max_pages
    scan_png: bytes | None = None
    try:
        with pymupdf.open(str(path)) as doc:
            page_count = doc.page_count
            if max_pages > 0:
                page_count = min(max_pages, page_count)
            pages = list(range(page_count))
            # to_markdown's default header detection scans every page of the
            # document, not just `pages`, so it is built over the same pages.
            markdown: str = pymupdf4llm.to_markdown(
                doc, pages=pages, hdr_info=IdentifyHeaders(doc, pages=pages)
            )
            if (
                pages
                and get_config().vision_pdf_scans
                and get_backend() is not None
                and _is_scan(doc, pages[:SCANNED_PDF_CHECK_PAGES])
            ):
                scan_png = _render_page(doc[0])
    except Exception as exc:
        raise ExtractionError(str(path), str(exc)) from exc
    finally:
        # MuPDF keeps a process-wide store of fonts and images that closing a
        # document doesn't empty; across thousands of PDFs it grew the indexer
        # past its 1.5 GB limit (RT #1542).
        pymupdf.TOOLS.store_shrink(100)
        gc.collect()
    if scan_png is not None:
        caption = _caption_png(scan_png, checksum)
        markdown = f"[File] {path.name}\n[Scanned page 1] {caption}\n{markdown}".rstrip()
    return markdown


def _is_scan(doc: Any, pages: list[int]) -> bool:
    """True when the pages hold under SCANNED_PDF_MIN_CHARS of text.

    Stops at the first page that reaches it, so a long text PDF is not
    read twice.
    """
    text_chars = 0
    for page in pages:
        text_chars += len(doc[page].get_text().strip())
        if text_chars >= SCANNED_PDF_MIN_CHARS:
            return False
    return True


def _render_page(page: Any) -> bytes:
    """Render a PDF page to PNG at SCANNED_PDF_DPI, within SCANNED_PDF_MAX_PIXELS."""
    import pymupdf

    zoom = SCANNED_PDF_DPI / 72
    pixels = page.rect.width * page.rect.height * zoom * zoom
    if pixels > SCANNED_PDF_MAX_PIXELS:
        zoom *= math.sqrt(SCANNED_PDF_MAX_PIXELS / pixels)
    png: bytes = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
    return png


def _caption_png(png: bytes, checksum: str | None) -> str:
    """Caption a rendered page through the vision backend, via a private temp file."""
    from .vision import caption_cached

    with tempfile.TemporaryDirectory(prefix="trove-scan-") as scan_dir:
        page = Path(scan_dir) / "page1.png"
        page.write_bytes(png)
        return caption_cached(page, "image", checksum)


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


def _extract_image(path: Path, checksum: str | None = None) -> str:
    """Extract text from an image via vision backend + EXIF metadata."""
    from .vision import caption_cached

    parts: list[str] = [f"[File] {path.name}"]

    exif = _extract_exif(path)
    if exif:
        parts.append(exif)

    caption = caption_cached(path, "image", checksum)
    parts.append(f"[Caption] {caption}")

    return "\n".join(parts)


def _extract_video(path: Path, checksum: str | None = None) -> str:
    """Extract text from a video via vision backend."""
    from .vision import caption_cached

    caption = caption_cached(path, "video", checksum)
    return f"[File] {path.name}\n[Caption] {caption}"
