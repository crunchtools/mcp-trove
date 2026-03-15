"""Tests for file scanning, checksumming, and chunking."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mcp_trove_crunchtools.indexer import chunk_text, compute_checksum, scan_directory


class TestChecksum:
    def test_consistent_checksum(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Checksum test content")
            path = f.name

        hash1 = compute_checksum(Path(path))
        hash2 = compute_checksum(Path(path))
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

        Path(path).unlink()

    def test_different_content_different_checksum(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Content A")
            path_a = f.name
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Content B")
            path_b = f.name

        assert compute_checksum(Path(path_a)) != compute_checksum(Path(path_b))

        Path(path_a).unlink()
        Path(path_b).unlink()


class TestChunking:
    def test_short_text_single_chunk(self) -> None:
        chunks = chunk_text("Short text", chunk_size=100, overlap=20)
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_long_text_multiple_chunks(self) -> None:
        text = "A" * 250
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) == 3

    def test_overlap_present(self) -> None:
        text = "ABCDEFGHIJ" * 10  # 100 chars
        chunks = chunk_text(text, chunk_size=40, overlap=10)
        assert len(chunks) >= 2
        # Verify overlap: end of chunk 0 should appear at start of chunk 1
        overlap_text = chunks[0][-10:]
        assert chunks[1].startswith(overlap_text)

    def test_empty_text(self) -> None:
        chunks = chunk_text("", chunk_size=100, overlap=20)
        assert chunks == []

    def test_whitespace_only(self) -> None:
        chunks = chunk_text("   \n\t  ", chunk_size=100, overlap=20)
        assert chunks == []


class TestScanDirectory:
    def test_scan_finds_supported_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.txt").write_text("content")
            (Path(tmpdir) / "file2.py").write_text("print('hi')")
            (Path(tmpdir) / "file3.md").write_text("# Title")

            files = scan_directory(Path(tmpdir))
            assert len(files) == 3

    def test_scan_skips_excluded_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "good.txt").write_text("content")
            (Path(tmpdir) / "bad.iso").write_text("binary")
            (Path(tmpdir) / "bad.zip").write_text("archive")

            files = scan_directory(Path(tmpdir))
            assert len(files) == 1
            assert files[0].name == "good.txt"

    def test_scan_skips_hidden_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "visible.txt").write_text("content")
            hidden = Path(tmpdir) / ".hidden"
            hidden.mkdir()
            (hidden / "secret.txt").write_text("hidden content")

            files = scan_directory(Path(tmpdir))
            assert len(files) == 1
            assert files[0].name == "visible.txt"

    @pytest.mark.skipif(True, reason="Would need a 50MB+ file")
    def test_scan_skips_large_files(self) -> None:
        pass

    def test_scan_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (Path(tmpdir) / "root.txt").write_text("root")
            (subdir / "nested.txt").write_text("nested")

            files = scan_directory(Path(tmpdir))
            assert len(files) == 2
