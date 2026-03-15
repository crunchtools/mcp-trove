"""Text extraction from various file formats."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import ExtractionError, UnsupportedFileTypeError

if TYPE_CHECKING:
    from pathlib import Path

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

SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS | DOCX_EXTENSIONS


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
    msg = suffix or "(no extension)"
    raise UnsupportedFileTypeError(str(path), msg)


def is_supported(path: Path) -> bool:
    """Check if a file type is supported for extraction."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def extract_text(path: Path) -> str:
    """Extract text content from a file.

    Returns the full text content as a string.
    """
    file_type = detect_file_type(path)

    if file_type == "pdf":
        return _extract_pdf(path)
    if file_type == "docx":
        return _extract_docx(path)
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
