"""Tests for text extraction from various file types."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_trove_crunchtools.errors import UnsupportedFileTypeError
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Hello, world!")
            path = f.name

        content = extract_text(Path(path))
        assert content == "Hello, world!"
        Path(path).unlink()

    def test_extract_markdown(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write("# Title\n\nParagraph text.")
            path = f.name

        content = extract_text(Path(path))
        assert "Title" in content
        assert "Paragraph" in content
        Path(path).unlink()

    def test_extract_python_source(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write('def hello():\n    print("Hello")\n')
            path = f.name

        content = extract_text(Path(path))
        assert "def hello" in content
        Path(path).unlink()

    def test_extract_nonexistent_file(self) -> None:
        from mcp_trove_crunchtools.errors import ExtractionError

        with pytest.raises(ExtractionError):
            extract_text(Path("/nonexistent/file.txt"))
