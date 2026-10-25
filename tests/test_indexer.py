"""Tests for file scanning, checksumming, and chunking."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from mcp_trove_crunchtools import database as db
from mcp_trove_crunchtools import indexer as indexer_mod
from mcp_trove_crunchtools.indexer import (
    MAX_FILE_SIZE,
    _extract_one,
    _partition_unchanged,
    _store_one,
    chunk_text,
    compute_checksum,
    scan_directory,
)

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator


class TestChecksum:
    def test_consistent_checksum(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Checksum test content")
            path = f.name

        hash1 = compute_checksum(Path(path))
        hash2 = compute_checksum(Path(path))
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

        Path(path).unlink()

    def test_different_content_different_checksum(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Content A")
            path_a = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
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
            assert sorted(f.name for f in files) == ["file1.txt", "file3.md"]

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

    def test_scan_does_not_descend_into_hidden_dirs(self, tmp_path: Path) -> None:
        """Hidden trees such as .hg are pruned, not walked and filtered."""
        (tmp_path / "visible.txt").write_text("content")
        store = tmp_path / "repo" / ".hg" / "store"
        store.mkdir(parents=True)
        (store / "data.txt").write_text("hidden content")

        walked: list[str] = []
        real_walk = os.walk

        def tracking_walk(top: Path) -> Iterator[tuple[str, list[str], list[str]]]:
            for entry in real_walk(top):
                walked.append(entry[0])
                yield entry

        with patch.object(indexer_mod.os, "walk", tracking_walk):
            files = scan_directory(tmp_path)

        assert files == [tmp_path / "visible.txt"]
        assert str(tmp_path / "repo") in walked
        assert not any(".hg" in d for d in walked)

    def test_scan_skips_hidden_files(self, tmp_path: Path) -> None:
        (tmp_path / "visible.txt").write_text("content")
        (tmp_path / ".hidden.txt").write_text("hidden content")

        assert scan_directory(tmp_path) == [tmp_path / "visible.txt"]

    def test_scan_skips_broken_symlinks(self, tmp_path: Path) -> None:
        (tmp_path / "visible.txt").write_text("content")
        (tmp_path / "dangling.txt").symlink_to(tmp_path / "missing.txt")

        assert scan_directory(tmp_path) == [tmp_path / "visible.txt"]

    def test_scan_skips_large_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            small = Path(tmpdir) / "small.txt"
            small.write_text("content")

            # A sparse file reports st_size over MAX_FILE_SIZE without
            # actually writing that many bytes to disk, so this stays fast
            # and reliable in CI instead of needing a real 50MB+ file.
            oversized = Path(tmpdir) / "oversized.txt"
            with oversized.open("wb") as f:
                f.seek(MAX_FILE_SIZE + 1)
                f.write(b"\0")
            assert oversized.stat().st_size > MAX_FILE_SIZE

            files = scan_directory(Path(tmpdir))
            assert files == [small]

    def test_scan_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (Path(tmpdir) / "root.txt").write_text("root")
            (subdir / "nested.txt").write_text("nested")

            files = scan_directory(Path(tmpdir))
            assert len(files) == 2


class TestLargeFileBounds:
    """A huge document must not be chunked or embedded all at once (RT #1542)."""

    def test_extracted_text_capped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TROVE_MAX_TEXT_CHARS", "2500")
        monkeypatch.setenv("TROVE_CHUNK_SIZE", "1000")
        monkeypatch.setenv("TROVE_CHUNK_OVERLAP", "0")
        big = tmp_path / "big.txt"
        content = "".join(f"{n:05d}" for n in range(2000))
        big.write_text(content)

        extraction = _extract_one(big, "sum", big.stat().st_size, None)

        assert "".join(extraction["chunks"]) == content[:2500]

    def test_failed_embed_batch_removes_partial_file(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        extraction = {
            "path": "/data/manual.pdf",
            "status": "extracted",
            "file_type": "pdf",
            "checksum": "abc",
            "file_size": 1,
            "mtime": 0.0,
            "chunks": [f"chunk {i}" for i in range(5)],
            "existing_id": None,
        }
        embed_results = [[[0.1] * 384] * 2, RuntimeError("embedding backend down")]

        with (
            patch.object(indexer_mod, "EMBED_BATCH_SIZE", 2),
            patch.object(indexer_mod, "embed_texts", side_effect=embed_results),
            pytest.raises(RuntimeError),
        ):
            _store_one(extraction)

        assert db.query("SELECT id FROM files") == []
        assert db.query("SELECT id FROM chunks") == []
        assert db.query("SELECT chunk_id FROM chunks_vec") == []

    def test_store_embeds_in_sub_batches(self, in_memory_db: sqlite3.Connection) -> None:
        chunks = [f"chunk {i}" for i in range(5)]
        extraction = {
            "path": "/data/manual.pdf",
            "status": "extracted",
            "file_type": "pdf",
            "checksum": "abc",
            "file_size": 1,
            "mtime": 0.0,
            "chunks": chunks,
            "existing_id": None,
        }

        with (
            patch.object(indexer_mod, "EMBED_BATCH_SIZE", 2),
            patch.object(
                indexer_mod, "embed_texts", side_effect=lambda texts: [[0.1] * 384 for _ in texts]
            ) as embed,
        ):
            result = _store_one(extraction)

        assert result["chunk_count"] == 5
        batch_sizes = [len(call.args[0]) for call in embed.call_args_list]
        assert batch_sizes == [2, 2, 1]
        rows = db.query("SELECT chunk_index, content FROM chunks ORDER BY chunk_index")
        assert [(r["chunk_index"], r["content"]) for r in rows] == list(enumerate(chunks))


class TestDeferredChecksum:
    """A new file is hashed by the extract worker, not by the serial scan
    before extraction; over a network mount that scan took hours (RT #1542)."""

    def test_new_file_not_hashed_during_partition(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        new_file = tmp_path / "new.txt"
        new_file.write_text("brand new")

        with patch.object(indexer_mod, "compute_checksum") as checksum:
            skipped, to_extract = _partition_unchanged([new_file], force=False)

        checksum.assert_not_called()
        assert skipped == []
        assert [item[1] for item in to_extract] == [""]

    def test_changed_file_still_hashed_during_partition(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        known = tmp_path / "known.txt"
        known.write_text("same content")
        db.insert_file(str(known.resolve()), compute_checksum(known), "text", 1, 0.0)

        skipped, to_extract = _partition_unchanged([known], force=False)

        assert to_extract == []
        assert skipped[0]["reason"] == "unchanged"

    def test_extract_one_computes_deferred_checksum(self, tmp_path: Path) -> None:
        new_file = tmp_path / "new.txt"
        new_file.write_text("brand new")

        extraction = _extract_one(new_file, "", new_file.stat().st_size, None)

        assert extraction["checksum"] == compute_checksum(new_file)

    def test_indexed_new_file_stores_real_checksum(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        new_file = tmp_path / "new.txt"
        new_file.write_text("brand new")

        indexer_mod.index_path(tmp_path)

        row = db.query_one("SELECT checksum FROM files WHERE path = ?", (str(new_file.resolve()),))
        assert row is not None
        assert row["checksum"] == compute_checksum(new_file)
