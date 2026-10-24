"""Tests for text extraction from various file types."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_trove_crunchtools.errors import ExtractionError, UnsupportedFileTypeError
from mcp_trove_crunchtools.extractor import detect_file_type, extract_text, is_supported


class TestDetectFileType:
    def test_text_file(self) -> None:
        assert detect_file_type(Path("test.txt")) == "text"

    def test_markdown_file(self) -> None:
        assert detect_file_type(Path("test.md")) == "markdown"

    def test_python_file(self) -> None:
        assert detect_file_type(Path("test.py")) == "text"

    def test_pdf_file(self) -> None:
        assert detect_file_type(Path("test.pdf")) == "pdf"

    def test_docx_file(self) -> None:
        assert detect_file_type(Path("test.docx")) == "docx"

    def test_image_file(self) -> None:
        assert detect_file_type(Path("photo.jpg")) == "image"
        assert detect_file_type(Path("photo.jpeg")) == "image"
        assert detect_file_type(Path("photo.png")) == "image"
        assert detect_file_type(Path("photo.gif")) == "image"
        assert detect_file_type(Path("photo.webp")) == "image"
        assert detect_file_type(Path("photo.heic")) == "image"

    def test_video_file(self) -> None:
        assert detect_file_type(Path("clip.mp4")) == "video"
        assert detect_file_type(Path("clip.mov")) == "video"
        assert detect_file_type(Path("clip.avi")) == "video"
        assert detect_file_type(Path("clip.webm")) == "video"
        assert detect_file_type(Path("clip.mkv")) == "video"

    def test_unsupported_file(self) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            detect_file_type(Path("test.xyz"))

    def test_no_extension(self) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            detect_file_type(Path("Makefile"))


class TestIsSupported:
    def test_supported_extensions(self) -> None:
        assert is_supported(Path("test.txt"))
        assert is_supported(Path("test.md"))
        assert is_supported(Path("test.py"))
        assert is_supported(Path("test.pdf"))
        assert is_supported(Path("test.docx"))
        assert is_supported(Path("test.json"))
        assert is_supported(Path("test.yaml"))

    def test_unsupported_extensions(self) -> None:
        assert not is_supported(Path("test.iso"))
        assert not is_supported(Path("test.exe"))

    def test_image_unsupported_without_vision(self) -> None:
        """Images are not supported when vision backend is none."""
        assert not is_supported(Path("test.jpg"))
        assert not is_supported(Path("test.png"))

    def test_video_unsupported_without_vision(self) -> None:
        """Videos are not supported when vision backend is none."""
        assert not is_supported(Path("test.mp4"))
        assert not is_supported(Path("test.mov"))

    def test_image_supported_with_vision(self) -> None:
        """Images are supported when a vision backend is configured."""
        mock_backend = MagicMock()
        with patch("mcp_trove_crunchtools.vision.get_backend", return_value=mock_backend):
            assert is_supported(Path("test.jpg"))
            assert is_supported(Path("test.png"))
            assert is_supported(Path("test.heic"))

    def test_video_supported_with_vision(self) -> None:
        """Videos are supported when a vision backend is configured."""
        mock_backend = MagicMock()
        with patch("mcp_trove_crunchtools.vision.get_backend", return_value=mock_backend):
            assert is_supported(Path("test.mp4"))
            assert is_supported(Path("test.mov"))
            assert is_supported(Path("test.webm"))


class TestExtractText:
    def test_extract_plain_text(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, world!")
            path = f.name

        content = extract_text(Path(path))
        assert content == "Hello, world!"
        Path(path).unlink()

    def test_extract_markdown(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Title\n\nParagraph text.")
            path = f.name

        content = extract_text(Path(path))
        assert "Title" in content
        assert "Paragraph" in content
        Path(path).unlink()

    def test_extract_python_source(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('def hello():\n    print("Hello")\n')
            path = f.name

        content = extract_text(Path(path))
        assert "def hello" in content
        Path(path).unlink()

    def test_extract_nonexistent_file(self) -> None:
        from mcp_trove_crunchtools.errors import ExtractionError

        with pytest.raises(ExtractionError):
            extract_text(Path("/nonexistent/file.txt"))


class TestPdfDependency:
    """pymupdf4llm is a hard dependency. Its 2.0 release is an empty shim with
    no module, so the lazy import in extractor.py failed every PDF at runtime
    while the rest of the suite stayed green (RT #1542)."""

    def test_pymupdf4llm_to_markdown_importable(self) -> None:
        from pymupdf4llm import to_markdown

        assert callable(to_markdown)

    def test_extract_pdf(self, tmp_path: Path) -> None:
        import pymupdf

        pdf_path = tmp_path / "sample.pdf"
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "Trove indexes PDFs")
        doc.save(str(pdf_path))
        doc.close()

        assert "Trove indexes PDFs" in extract_text(pdf_path)


PDF_PAGES = tuple(f"Manual page {n} marker" for n in range(1, 8))


def _make_pdf(path: Path, pages: tuple[str, ...]) -> None:
    import pymupdf

    doc = pymupdf.open()
    for page_text in pages:
        doc.new_page().insert_text((72, 72), page_text)
    doc.save(str(path))
    doc.close()


class TestPdfPageCap:
    """Only the first TROVE_PDF_MAX_PAGES pages of a PDF are converted.
    Converting whole 20-80 MB manuals OOM-killed the indexer (RT #1542)."""

    def test_default_cap_is_five_pages(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "manual.pdf"
        _make_pdf(pdf_path, PDF_PAGES)

        text = extract_text(pdf_path)

        for page_text in PDF_PAGES[:5]:
            assert page_text in text
        assert "Manual page 6 marker" not in text
        assert "Manual page 7 marker" not in text

    def test_cap_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TROVE_PDF_MAX_PAGES", "2")
        pdf_path = tmp_path / "manual.pdf"
        _make_pdf(pdf_path, PDF_PAGES)

        text = extract_text(pdf_path)

        assert "Manual page 2 marker" in text
        assert "Manual page 3 marker" not in text

    @pytest.mark.parametrize("cap", ["0", "-1"])
    def test_non_positive_cap_converts_every_page(
        self, cap: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TROVE_PDF_MAX_PAGES", cap)
        pdf_path = tmp_path / "manual.pdf"
        _make_pdf(pdf_path, PDF_PAGES)

        text = extract_text(pdf_path)

        for page_text in PDF_PAGES:
            assert page_text in text

    def test_header_scan_limited_to_capped_pages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """to_markdown's default header detection reads every page of the
        document even when `pages` is given, so the cap must reach it too."""
        from pymupdf4llm.helpers import pymupdf_rag

        monkeypatch.setenv("TROVE_PDF_MAX_PAGES", "3")
        pdf_path = tmp_path / "manual.pdf"
        _make_pdf(pdf_path, PDF_PAGES)

        with patch.object(
            pymupdf_rag, "IdentifyHeaders", wraps=pymupdf_rag.IdentifyHeaders
        ) as identify:
            extract_text(pdf_path)

        identify.assert_called_once()
        assert identify.call_args.kwargs["pages"] == [0, 1, 2]


class TestPdfStoreRelease:
    """MuPDF's global store outlives each document, so it is emptied after
    every PDF; otherwise a long index run grows until it is OOM-killed
    (RT #1542)."""

    def test_store_emptied_after_pdf(self, tmp_path: Path) -> None:
        import pymupdf

        pdf_path = tmp_path / "manual.pdf"
        _make_pdf(pdf_path, PDF_PAGES[:1])

        with patch.object(pymupdf.TOOLS, "store_shrink") as shrink:
            extract_text(pdf_path)

        shrink.assert_called_once_with(100)

    def test_store_emptied_when_extraction_fails(self, tmp_path: Path) -> None:
        import pymupdf

        pdf_path = tmp_path / "broken.pdf"
        pdf_path.write_bytes(b"not a pdf")

        with patch.object(pymupdf.TOOLS, "store_shrink") as shrink, pytest.raises(ExtractionError):
            extract_text(pdf_path)

        shrink.assert_called_once_with(100)
