"""Tests for manifest scans, streaming extraction and extraction quality (RT #1542)."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from mcp_trove_crunchtools import database as db
from mcp_trove_crunchtools import embedder
from mcp_trove_crunchtools import indexer as indexer_mod
from mcp_trove_crunchtools.errors import IndexAbortedError
from mcp_trove_crunchtools.extractor import _html_to_text
from mcp_trove_crunchtools.indexer import chunk_text, index_path_async, scan_source
from mcp_trove_crunchtools.manifest import load_manifest, manifest_file

if TYPE_CHECKING:
    import sqlite3


MTIME_ISO = "2024-01-02T03:04:05Z"
MTIME = 1704164645.0


@pytest.fixture
def manifests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "manifests"
    directory.mkdir()
    monkeypatch.setenv("TROVE_MANIFEST_DIR", str(directory))
    return directory


def _write_manifest(tree: Path, entries: list[dict[str, Any]]) -> Path:
    target = manifest_file(tree)
    assert target is not None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(entries))
    return target


def _entry(path: str, size: int = 10, mtime: str = MTIME_ISO) -> dict[str, Any]:
    return {"Path": path, "Name": path.rsplit("/", 1)[-1], "Size": size, "ModTime": mtime}


class TestManifest:
    def test_fresh_manifest_is_used(self, tmp_path: Path, manifests: Path) -> None:
        _write_manifest(tmp_path, [_entry("a/b.txt", 7), {"Path": "a", "IsDir": True}])

        entries = load_manifest(tmp_path)

        assert entries is not None
        assert list(entries) == [tmp_path / "a" / "b.txt"]
        assert entries[tmp_path / "a" / "b.txt"].size == 7
        assert entries[tmp_path / "a" / "b.txt"].mtime == MTIME

    def test_stale_manifest_falls_back_to_the_walk(self, tmp_path: Path, manifests: Path) -> None:
        source = _write_manifest(tmp_path, [_entry("b.txt")])
        old = time.time() - 7 * 3600
        os.utime(source, (old, old))

        assert load_manifest(tmp_path) is None

    def test_oversized_manifest_falls_back(self, tmp_path: Path, manifests: Path) -> None:
        _write_manifest(tmp_path, [_entry("b.txt")])

        with patch("mcp_trove_crunchtools.manifest.MANIFEST_MAX_BYTES", 10):
            assert load_manifest(tmp_path) is None

    def test_unstattable_manifest_falls_back(self, tmp_path: Path, manifests: Path) -> None:
        from mcp_trove_crunchtools import manifest as manifest_mod

        source = _write_manifest(tmp_path, [_entry("b.txt")])
        real_stat = Path.stat

        def stat(self: Path, **kwargs: Any) -> os.stat_result:
            if self == source:
                raise PermissionError("denied")
            return real_stat(self, **kwargs)

        with patch.object(manifest_mod.Path, "stat", stat):
            assert load_manifest(tmp_path) is None

    def test_missing_manifest_falls_back(self, tmp_path: Path, manifests: Path) -> None:
        assert load_manifest(tmp_path / "elsewhere") is None

    def test_manifest_off_without_a_directory(self, tmp_path: Path) -> None:
        assert load_manifest(tmp_path) is None

    @pytest.mark.parametrize("escape", ["../../etc/passwd", "/etc/passwd", "..\\..\\secret.txt"])
    def test_escaping_path_rejects_the_manifest(
        self, tmp_path: Path, manifests: Path, escape: str
    ) -> None:
        _write_manifest(tmp_path, [_entry(escape)])

        assert load_manifest(tmp_path) is None

    def test_similar_paths_get_their_own_manifests(self, manifests: Path) -> None:
        assert manifest_file(Path("/pcloud/a_b/c")) != manifest_file(Path("/pcloud/a/b_c"))

    @pytest.mark.parametrize("listing", ['{"Path": "a.txt"}', "[null]", '["a.txt"]'])
    def test_wrong_shape_falls_back(self, tmp_path: Path, manifests: Path, listing: str) -> None:
        target = manifest_file(tmp_path)
        assert target is not None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(listing)

        assert load_manifest(tmp_path) is None

    def test_scan_applies_the_walk_filters(self, tmp_path: Path, manifests: Path) -> None:
        _write_manifest(
            tmp_path,
            [
                _entry("keep.txt"),
                _entry(".git/config.txt"),
                _entry("archive.zip"),
                _entry("binary.exe"),
                _entry("huge.txt", size=indexer_mod.MAX_FILE_SIZE + 1),
            ],
        )

        with patch.object(indexer_mod.os, "walk") as walk:
            files, stats = scan_source(tmp_path)

        walk.assert_not_called()
        assert files == [tmp_path / "keep.txt"]
        assert stats is not None


class TestManifestPartition:
    async def test_unchanged_files_skipped_without_touching_them(
        self, tmp_path: Path, manifests: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        """The files are not on disk: only the manifest and the DB are consulted."""
        for name in ("a.txt", "b.txt"):
            db.insert_file(str(tmp_path / name), "sum", "text", 10, MTIME + 0.4)
            db.execute("UPDATE files SET chunk_count = 1 WHERE path = ?", (str(tmp_path / name),))
        _write_manifest(tmp_path, [_entry("a.txt"), _entry("b.txt")])

        results = await index_path_async(tmp_path)

        assert sorted(r["status"] for r in results) == ["skipped", "skipped"]

    async def test_files_gone_from_the_manifest_are_removed(
        self, tmp_path: Path, manifests: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        (tmp_path / "kept.txt").write_text("still here")
        db.insert_file(str(tmp_path / "deleted.txt"), "sum", "text", 10, MTIME)
        _write_manifest(tmp_path, [_entry("kept.txt", size=10)])

        results = await index_path_async(tmp_path)

        assert {r["status"] for r in results} == {"indexed", "removed"}
        assert db.query("SELECT path FROM files") == [{"path": str(tmp_path / "kept.txt")}]

    async def test_small_tree_may_empty_out(
        self, tmp_path: Path, manifests: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        """A staging folder like Autosync/Queued empties on purpose."""
        for n in range(5):
            db.insert_file(str(tmp_path / f"staged{n}.jpg"), "sum", "image", 10, MTIME)
        _write_manifest(tmp_path, [])

        results = await index_path_async(tmp_path)

        assert [r["status"] for r in results] == ["removed"] * 5

    @pytest.mark.parametrize("inbox", [True, False])
    async def test_big_inbox_may_empty_out(
        self,
        tmp_path: Path,
        manifests: Path,
        in_memory_db: sqlite3.Connection,
        inbox: bool,
    ) -> None:
        """Autosync/Queued held 103 photos one night and none the next (RT #1542)."""
        if inbox:
            indexer_mod.get_config().inbox_paths = ["/elsewhere", str(tmp_path)]
        for n in range(103):
            db.insert_file(str(tmp_path / f"staged{n}.jpg"), "sum", "image", 10, MTIME)
        db.finish_run(
            db.start_run(str(tmp_path), 103),
            files_indexed=103,
            files_skipped=0,
            files_errored=0,
            total_chunks=103,
        )
        _write_manifest(tmp_path, [])

        if inbox:
            results = await index_path_async(tmp_path)
            assert [r["status"] for r in results] == ["removed"] * 103
        else:
            with pytest.raises(IndexAbortedError, match="under 50%"):
                await index_path_async(tmp_path)
            assert len(db.query("SELECT id FROM files")) == 103

    @pytest.mark.parametrize("sizes", [(100, 90), (99, 0)])
    async def test_removal_guard_boundaries(
        self,
        tmp_path: Path,
        manifests: Path,
        in_memory_db: sqlite3.Connection,
        sizes: tuple[int, int],
    ) -> None:
        """Exactly 10% gone is allowed; under 100 indexed files is never guarded."""
        indexed, listed = sizes
        for n in range(indexed):
            path = str(tmp_path / f"f{n}.txt")
            db.insert_file(path, "sum", "text", 10, MTIME)
            db.execute("UPDATE files SET chunk_count = 1 WHERE path = ?", (path,))
        _write_manifest(tmp_path, [_entry(f"f{n}.txt") for n in range(listed)])

        results = await index_path_async(tmp_path, force=True)

        assert sum(r["status"] == "removed" for r in results) == indexed - listed

    async def test_sibling_directory_rows_are_left_alone(
        self, tmp_path: Path, manifests: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        tree = tmp_path / "docs"
        tree.mkdir()
        db.insert_file(str(tmp_path / "docs-old" / "x.txt"), "sum", "text", 10, MTIME)
        db.insert_file(str(tmp_path / "docs0" / "y.txt"), "sum", "text", 10, MTIME)
        _write_manifest(tree, [])

        await index_path_async(tree)

        assert len(db.query("SELECT id FROM files")) == 2

    async def test_entry_under_a_symlink_out_of_the_tree_is_not_read(
        self, tmp_path: Path, manifests: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("not for the index")
        tree = tmp_path / "tree"
        tree.mkdir()
        (tree / "link").symlink_to(outside)
        _write_manifest(tree, [_entry("link/secret.txt", size=17)])

        results = await index_path_async(tree)

        assert [r["reason"] for r in results] == ["outside_root"]
        assert db.query("SELECT id FROM files") == []

    def test_root_prefix_finds_root_level_rows(self, in_memory_db: sqlite3.Connection) -> None:
        db.insert_file("/gone.txt", "sum", "text", 10, MTIME)

        removed = indexer_mod._remove_missing(Path("/"), set())

        assert [r["path"] for r in removed] == ["/gone.txt"]

    async def test_file_grown_past_the_limit_is_not_read(
        self, tmp_path: Path, manifests: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        """The manifest says 5 bytes; the file is now over the limit."""
        grown = tmp_path / "grown.txt"
        with grown.open("wb") as out:
            out.seek(indexer_mod.MAX_FILE_SIZE + 1)
            out.write(b"\0")
        _write_manifest(tmp_path, [_entry("grown.txt", size=5)])

        with patch.object(indexer_mod, "extract_text") as extract:
            results = await index_path_async(tmp_path)

        extract.assert_not_called()
        assert [r["status"] for r in results] == ["error"]
        assert "over the" in str(results[0]["reason"])

    async def test_batch_limited_run_removes_nothing(
        self, tmp_path: Path, manifests: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        """A run over the first N files sees a partial list, not deletions."""
        for name in ("a.txt", "b.txt"):
            (tmp_path / name).write_text(name)
        db.insert_file(str(tmp_path / "c.txt"), "sum", "text", 10, MTIME)
        _write_manifest(tmp_path, [_entry("a.txt", 5), _entry("b.txt", 5)])

        results = await index_path_async(tmp_path, batch_limit=1)

        assert "removed" not in {r["status"] for r in results}
        assert db.query_one("SELECT id FROM files WHERE path = ?", (str(tmp_path / "c.txt"),))

    async def test_big_tree_losing_many_files_aborts(
        self, tmp_path: Path, manifests: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        listed = [_entry(f"f{n}.txt") for n in range(85)]
        for n in range(100):
            db.insert_file(str(tmp_path / f"f{n}.txt"), "sum", "text", 10, MTIME)
            db.execute(
                "UPDATE files SET chunk_count = 1 WHERE path = ?", (str(tmp_path / f"f{n}.txt"),)
            )
        _write_manifest(tmp_path, listed)

        with pytest.raises(IndexAbortedError, match="lacks 15 of 100"):
            await index_path_async(tmp_path, force=False)

        assert len(db.query("SELECT id FROM files")) == 100


class TestExclusion:
    """Source code and excluded directories are skipped, and purged once indexed."""

    def test_excluded_directory_is_not_walked(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(indexer_mod.get_config(), "exclude_patterns", ["Old Backup"])
        (tmp_path / "Old Backup" / "docs").mkdir(parents=True)
        (tmp_path / "Old Backup" / "docs" / "readme.txt").write_text("old")
        (tmp_path / "notes.txt").write_text("keep")

        assert scan_source(tmp_path)[0] == [tmp_path / "notes.txt"]

    def test_excluded_directory_is_dropped_from_the_manifest(
        self, tmp_path: Path, manifests: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(indexer_mod.get_config(), "exclude_patterns", ["Old Backup"])
        _write_manifest(tmp_path, [_entry("Old Backup/docs/readme.txt"), _entry("notes.txt")])

        assert scan_source(tmp_path)[0] == [tmp_path / "notes.txt"]

    async def test_indexed_code_and_excluded_trees_are_purged(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The purge does not trip the shrink gate, whose baseline is the index."""
        monkeypatch.setattr(indexer_mod.get_config(), "exclude_patterns", ["Old Backup"])
        for n in range(150):
            db.insert_file(str(tmp_path / "src" / f"f{n}.c"), "sum", "text", 1, MTIME)
        db.insert_file(str(tmp_path / "Old Backup" / "notes.txt"), "sum", "text", 1, MTIME)
        db.insert_file(str(tmp_path / "photo.jpg"), "sum", "image", 1, MTIME)
        (tmp_path / "letter.txt").write_text("a letter")

        results = await index_path_async(tmp_path)

        assert sum(r.get("reason") == "excluded" for r in results) == 151
        remaining = {Path(r["path"]).name for r in db.query("SELECT path FROM files")}
        assert remaining == {"photo.jpg", "letter.txt"}

    async def test_purge_reported_when_nothing_is_left(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        db.insert_file(str(tmp_path / "main.c"), "sum", "text", 1, MTIME)
        db.insert_file(str(tmp_path / ".env"), "sum", "text", 1, MTIME)

        results = await index_path_async(tmp_path)

        assert sorted(Path(str(r["path"])).name for r in results) == [".env", "main.c"]
        assert db.query("SELECT path FROM files") == []

    async def test_code_kept_when_asked_for(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(indexer_mod.get_config(), "index_source_code", True)
        (tmp_path / "tool.py").write_text("print('hi')")

        results = await index_path_async(tmp_path)

        assert [r["status"] for r in results] == ["indexed"]


class TestStreaming:
    async def test_slow_file_does_not_hold_back_the_rest(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TROVE_INDEX_WORKERS", "4")
        for name in ("a_slow.jpg", "b.jpg", "c.jpg", "d.jpg"):
            (tmp_path / name).write_bytes(b"\xff\xd8" + name.encode())
        real_extract = indexer_mod._extract_one

        def extract(path: Path, *args: Any) -> dict[str, Any]:
            if "slow" in path.name:
                time.sleep(0.5)
            return real_extract(path, *args)

        backend = MagicMock()
        backend.caption.return_value = "a photo"
        from mcp_trove_crunchtools import vision

        with (
            patch.object(vision, "get_backend", return_value=backend),
            patch.object(indexer_mod, "_extract_one", extract),
        ):
            results = await index_path_async(tmp_path)

        assert [r["status"] for r in results] == ["indexed"] * 4
        assert "slow" in str(results[-1]["path"])

    async def test_pdfs_extract_one_at_a_time(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TROVE_INDEX_WORKERS", "8")
        for n in range(4):
            (tmp_path / f"doc{n}.pdf").write_bytes(b"%PDF-1.4")
        running = 0
        peak = 0
        lock = threading.Lock()

        def extract(path: Path, *_args: Any) -> dict[str, Any]:
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.05)
            with lock:
                running -= 1
            return {
                "path": str(path),
                "status": "extracted",
                "file_type": "pdf",
                "checksum": path.name,
                "file_size": 1,
                "mtime": 1.0,
                "chunks": ["text"],
                "existing_id": None,
            }

        with patch.object(indexer_mod, "_extract_one", extract):
            await index_path_async(tmp_path)

        assert peak == 1

    async def test_slow_embedding_does_not_time_out_reads_in_flight(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Embedding on the event loop stalled it; in-flight reads then timed out."""
        monkeypatch.setattr(indexer_mod.get_config(), "vision_timeout", 0.1)
        (tmp_path / "a_big.txt").write_text("BIG " * 50)
        for n in range(3):
            (tmp_path / f"small{n}.txt").write_text("small")
        real_extract = indexer_mod._extract_one

        def extract(path: Path, *args: Any) -> dict[str, Any]:
            if "small" in path.name:
                time.sleep(0.1)
            return real_extract(path, *args)

        def embed(texts: list[str]) -> list[list[float]]:
            if any("BIG" in t for t in texts):
                time.sleep(1)
            return [[0.0] * 384 for _ in texts]

        with (
            patch.object(indexer_mod, "_extract_one", extract),
            patch.object(indexer_mod, "embed_texts", side_effect=embed),
        ):
            results = await index_path_async(tmp_path)

        assert [r["status"] for r in results] == ["indexed"] * 4

    async def test_overlapping_runs_embed_one_file_at_a_time(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        """Two index calls in the MCP server must not stack embeddings."""
        for tree in ("one", "two"):
            (tmp_path / tree).mkdir()
            for n in range(3):
                (tmp_path / tree / f"f{n}.txt").write_text(f"{tree} {n}")
        running = 0
        peak = 0
        lock = threading.Lock()

        def embed(texts: list[str]) -> list[list[float]]:
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.05)
            with lock:
                running -= 1
            return [[0.0] * 384 for _ in texts]

        with patch.object(indexer_mod, "embed_texts", side_effect=embed):
            first, second = await asyncio.gather(
                index_path_async(tmp_path / "one"), index_path_async(tmp_path / "two")
            )

        assert [r["status"] for r in first + second] == ["indexed"] * 6
        assert peak == 1


class TestCacheDatabase:
    async def test_cache_uses_the_database_actually_opened(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TROVE_DB points elsewhere; the cache must follow the explicit path."""
        monkeypatch.setenv("TROVE_DB", str(tmp_path / "configured.db"))
        db.get_db(str(tmp_path / "opened.db"))
        (tmp_path / "photos").mkdir()
        (tmp_path / "photos" / "a.jpg").write_bytes(b"\xff\xd8 photo")
        backend = MagicMock()
        backend.caption.return_value = "a photo"
        from mcp_trove_crunchtools import vision

        with patch.object(vision, "get_backend", return_value=backend):
            results = await index_path_async(tmp_path / "photos")

        assert [r["status"] for r in results] == ["indexed"]
        assert len(db.query("SELECT sha256 FROM captions")) == 1
        assert not (tmp_path / "configured.db").exists()


class TestHungReadThreshold:
    @pytest.mark.parametrize(("timeouts", "aborts"), [(4, False), (5, True)])
    async def test_abort_at_the_limit(
        self,
        tmp_path: Path,
        in_memory_db: sqlite3.Connection,
        timeouts: int,
        aborts: bool,
    ) -> None:
        for n in range(8):
            (tmp_path / f"f{n}.txt").write_text("x")
        real_extract = indexer_mod._extract_one
        hanging = {f"f{n}.txt" for n in range(timeouts)}

        def extract(path: Path, *args: Any) -> dict[str, Any]:
            if path.name in hanging:
                raise TimeoutError
            return real_extract(path, *args)

        with patch.object(indexer_mod, "_extract_one", extract):
            if aborts:
                with pytest.raises(IndexAbortedError, match="5 extractions timed out"):
                    await index_path_async(tmp_path)
            else:
                results = await index_path_async(tmp_path)
                assert sum(r["status"] == "error" for r in results) == 4


class TestExtractionQuality:
    def test_chunks_end_at_paragraph_breaks(self) -> None:
        text = ("A" * 70 + "\n\n") * 6

        chunks = chunk_text(text, 100, 10)

        assert all(chunk.endswith("\n\n") for chunk in chunks[:-1])

    def test_unbroken_text_still_splits(self) -> None:
        chunks = chunk_text("x" * 250, 100, 10)

        assert [len(c) for c in chunks] == [100, 100, 70]

    def test_html_becomes_visible_text(self) -> None:
        markup = (
            "<html><head><style>p {color: red}</style><script>var x = 1;</script></head>"
            "<body><h1>Invoice</h1><p>Total &amp; tax: $5</p><table><tr><td>Oil</td></tr>"
            "</table></body></html>"
        )

        assert _html_to_text(markup) == "Invoice\nTotal & tax: $5\nOil"

    def test_embedding_threads_are_set(self) -> None:
        with patch("fastembed.TextEmbedding") as model:
            embedder.get_model()

        assert model.call_args.kwargs["threads"] == 2
