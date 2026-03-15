"""Tests for text extraction from various file types."""

from __future__ import annotations

import tempfile
from pathlib import Path

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
        assert not is_supported(Path("test.mp4"))
        assert not is_supported(Path("test.exe"))


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
