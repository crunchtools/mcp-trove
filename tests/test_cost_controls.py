"""Tests for vision spend and retry controls: caption cache, budget, poison files."""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from mcp_trove_crunchtools import database as db
from mcp_trove_crunchtools import indexer as indexer_mod
from mcp_trove_crunchtools import vision
from mcp_trove_crunchtools.indexer import index_path_async
from mcp_trove_crunchtools.tools.run_index import run_index
from mcp_trove_crunchtools.vision import OpenRouterBackend

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def file_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> sqlite3.Connection:
    """A database on disk: the caption cache is read from worker threads."""
    monkeypatch.setenv("TROVE_DB", str(tmp_path / "trove.db"))
    return db.get_db()


@pytest.fixture(autouse=True)
def _pdf_scans_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TROVE_VISION_PDF_SCANS", "true")


@pytest.fixture
def backend() -> Iterator[MagicMock]:
    fake = MagicMock()
    fake.caption.side_effect = lambda path, _file_type: f"caption of {path.name}"
    vision.reset_budget()
    with patch.object(vision, "get_backend", return_value=fake):
        yield fake


def _images(directory: Path, count: int, *, same_content: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for n in range(count):
        content = b"\xff\xd8 same photo" if same_content else f"\xff\xd8 photo {n}".encode()
        (directory / f"img{n}.jpg").write_bytes(content)


def _statuses(results: list[dict[str, Any]]) -> list[str]:
    return sorted(str(r["status"]) for r in results)


class TestCaptionCache:
    async def test_same_content_is_captioned_once(
        self, tmp_path: Path, file_db: sqlite3.Connection, backend: MagicMock
    ) -> None:
        _images(tmp_path / "photos", 3, same_content=True)

        results = await index_path_async(tmp_path / "photos")

        assert _statuses(results) == ["indexed"] * 3
        assert backend.caption.call_count == 1
        assert vision.calls_made() == 1

    async def test_renamed_file_reuses_its_caption(
        self, tmp_path: Path, file_db: sqlite3.Connection, backend: MagicMock
    ) -> None:
        _images(tmp_path / "old", 1)
        await index_path_async(tmp_path / "old")
        (tmp_path / "new").mkdir()
        (tmp_path / "old" / "img0.jpg").rename(tmp_path / "new" / "renamed.jpg")

        await index_path_async(tmp_path / "new")

        assert backend.caption.call_count == 1
        stored = db.query("SELECT content FROM chunks WHERE content LIKE '%renamed.jpg%'")
        assert "caption of img0.jpg" in stored[0]["content"]


class TestVisionBudget:
    async def test_files_past_the_budget_are_deferred_then_resumed(
        self,
        tmp_path: Path,
        file_db: sqlite3.Connection,
        backend: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TROVE_VISION_MAX_CALLS_PER_RUN", "2")
        from mcp_trove_crunchtools import config as config_mod

        config_mod._config = None
        _images(tmp_path / "photos", 4)

        first = await index_path_async(tmp_path / "photos")

        assert _statuses(first) == ["deferred", "deferred", "indexed", "indexed"]
        assert db.query("SELECT id FROM index_errors") == []

        vision.reset_budget()
        second = await index_path_async(tmp_path / "photos")

        assert _statuses(second) == ["indexed", "indexed", "skipped", "skipped"]
        assert backend.caption.call_count == 4

    async def test_deferred_image_is_not_downloaded(
        self,
        tmp_path: Path,
        file_db: sqlite3.Connection,
        backend: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TROVE_VISION_MAX_CALLS_PER_RUN", "1")
        from mcp_trove_crunchtools import config as config_mod

        config_mod._config = None
        _images(tmp_path / "photos", 1)
        vision._reserve_call("spent")

        with patch.object(indexer_mod, "compute_checksum") as checksum:
            results = await index_path_async(tmp_path / "photos")

        assert _statuses(results) == ["deferred"]
        checksum.assert_not_called()


class TestPoisonFiles:
    async def test_permanent_failure_not_retried_until_the_file_changes(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        broken = tmp_path / "broken.txt"
        broken.write_text("unreadable")
        failing = patch.object(indexer_mod, "extract_text", side_effect=ValueError("corrupt"))

        with failing as extract:
            first = await index_path_async(tmp_path)
            second = await index_path_async(tmp_path)

        assert _statuses(first) == ["error"]
        assert [r["reason"] for r in second] == ["known_failure"]
        assert extract.call_count == 1

        broken.write_text("unreadable, edited")
        os.utime(broken, (1, 1))
        third = await index_path_async(tmp_path)

        assert _statuses(third) == ["indexed"]

    async def test_reindex_retries_a_known_failure(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        (tmp_path / "broken.txt").write_text("unreadable")
        with patch.object(indexer_mod, "extract_text", side_effect=ValueError("corrupt")):
            await index_path_async(tmp_path)

        results = await index_path_async(tmp_path, force=True)

        assert _statuses(results) == ["indexed"]

    def test_transient_failures_stop_after_the_limit(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        version = (10, 1.0)
        for _ in range(db.MAX_TRANSIENT_FAILURES - 1):
            db.insert_error(None, "/flaky.jpg", "read timeout", "transient", version)
        assert db.known_failures() == set()

        db.insert_error(None, "/flaky.jpg", "read timeout", "transient", version)

        assert db.known_failures() == {("/flaky.jpg", 10, 1.0)}


class TestScannedPdf:
    def _blank_pdf(self, path: Path) -> None:
        import pymupdf

        doc = pymupdf.open()
        doc.new_page()
        doc.save(str(path))
        doc.close()

    async def test_scan_is_captioned(
        self, tmp_path: Path, file_db: sqlite3.Connection, backend: MagicMock
    ) -> None:
        self._blank_pdf(tmp_path / "receipt.pdf")

        await index_path_async(tmp_path)

        content = db.query("SELECT content FROM chunks")[0]["content"]
        assert "[Scanned page 1] caption of page1.png" in content
        assert backend.caption.call_count == 1

    async def test_scan_not_sent_unless_enabled(
        self,
        tmp_path: Path,
        file_db: sqlite3.Connection,
        backend: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from mcp_trove_crunchtools import config as config_mod

        monkeypatch.setenv("TROVE_VISION_PDF_SCANS", "false")
        config_mod._config = None
        self._blank_pdf(tmp_path / "statement.pdf")

        await index_path_async(tmp_path)

        backend.caption.assert_not_called()

    async def test_scan_without_vision_is_indexed_empty(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        self._blank_pdf(tmp_path / "receipt.pdf")

        results = await index_path_async(tmp_path)

        assert [r["chunk_count"] for r in results] == [0]


class TestFallbackModel:
    def _models_sent(self, tmp_path: Path) -> list[str]:
        img = tmp_path / "a.png"
        img.write_bytes(b"\x89PNG")
        backend = OpenRouterBackend("google/gemini-3.1-flash-lite", "describe")
        client = MagicMock()
        client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="a cat"))
        ]
        backend._client = client
        backend.caption(img, "image")
        models: list[str] = client.chat.completions.create.call_args.kwargs["extra_body"]["models"]
        return models

    def test_no_fallback_unless_configured(self, tmp_path: Path) -> None:
        assert self._models_sent(tmp_path) == ["google/gemini-3.1-flash-lite"]

    def test_configured_fallback_is_offered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TROVE_VISION_FALLBACK_MODEL", "google/gemini-3.8-flash")

        assert self._models_sent(tmp_path) == [
            "google/gemini-3.1-flash-lite",
            "google/gemini-3.8-flash",
        ]


class TestBudgetScope:
    async def test_overlapping_runs_count_separately(
        self,
        tmp_path: Path,
        file_db: sqlite3.Connection,
        backend: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TROVE_VISION_MAX_CALLS_PER_RUN", "2")
        from mcp_trove_crunchtools import config as config_mod

        config_mod._config = None
        _images(tmp_path / "a", 2)
        (tmp_path / "b").mkdir()
        for n in range(2):
            (tmp_path / "b" / f"other{n}.jpg").write_bytes(f"\xff\xd8 other {n}".encode())

        async def one_run(directory: Path) -> tuple[list[str], int]:
            vision.reset_budget()
            results = await index_path_async(directory)
            return _statuses(results), vision.calls_made()

        runs = await asyncio.gather(one_run(tmp_path / "a"), one_run(tmp_path / "b"))

        assert runs == [(["indexed", "indexed"], 2), (["indexed", "indexed"], 2)]

    def test_cli_run_shares_one_budget_across_paths(
        self,
        tmp_path: Path,
        backend: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _images(tmp_path / "a", 2)
        (tmp_path / "b").mkdir()
        for n in range(2):
            (tmp_path / "b" / f"other{n}.jpg").write_bytes(f"\xff\xd8 other {n}".encode())
        monkeypatch.setenv("TROVE_DB", str(tmp_path / "data" / "trove.db"))
        monkeypatch.setenv("TROVE_PATHS", f"{tmp_path / 'a'}{os.pathsep}{tmp_path / 'b'}")
        monkeypatch.setenv("TROVE_VISION_MAX_CALLS_PER_RUN", "3")

        assert run_index() == 0

        summary = json.loads((tmp_path / "data" / "last_run.json").read_text())
        assert summary["vision_calls"] == 3
        assert sum(p["files_deferred"] for p in summary["paths"]) == 1


class TestCacheModel:
    async def test_new_model_re_captions(
        self,
        tmp_path: Path,
        file_db: sqlite3.Connection,
        backend: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from mcp_trove_crunchtools import config as config_mod

        _images(tmp_path / "photos", 1)
        await index_path_async(tmp_path / "photos")
        await index_path_async(tmp_path / "photos", force=True)
        assert backend.caption.call_count == 1

        monkeypatch.setenv("TROVE_VISION_MODEL", "another/model")
        config_mod._config = None
        await index_path_async(tmp_path / "photos", force=True)

        assert backend.caption.call_count == 2


class TestErrorVersionMigration:
    def test_legacy_error_table_gains_columns_and_keeps_rows(self, tmp_path: Path) -> None:
        path = str(tmp_path / "trove.db")
        conn = db.get_db(path)
        conn.execute("ALTER TABLE index_errors DROP COLUMN file_size")
        conn.execute("ALTER TABLE index_errors DROP COLUMN mtime")
        conn.execute(
            "INSERT INTO index_errors (path, error_message, error_type) "
            "VALUES ('/old.jpg', 'bad', 'permanent')"
        )
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()
        db._db = None

        for _ in range(2):  # a second open must be a no-op
            columns = {r[1] for r in db.get_db(path).execute("PRAGMA table_info(index_errors)")}
            assert {"file_size", "mtime"} <= columns
            assert db.query("SELECT path FROM index_errors") == [{"path": "/old.jpg"}]
            db.get_db().close()
            db._db = None


class TestScanThreshold:
    def _pdf(self, path: Path, text: str) -> None:
        import pymupdf

        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), text)
        doc.save(str(path))
        doc.close()

    async def test_short_text_counts_as_a_scan(
        self, tmp_path: Path, file_db: sqlite3.Connection, backend: MagicMock
    ) -> None:
        self._pdf(tmp_path / "stamp.pdf", "p. 1")

        await index_path_async(tmp_path)

        assert backend.caption.call_count == 1

    async def test_real_text_is_not_captioned(
        self, tmp_path: Path, file_db: sqlite3.Connection, backend: MagicMock
    ) -> None:
        self._pdf(tmp_path / "letter.pdf", "Dear Scott, your invoice is attached.")

        await index_path_async(tmp_path)

        backend.caption.assert_not_called()


class TestCaptionPruning:
    def test_only_old_orphaned_captions_are_pruned(self, in_memory_db: sqlite3.Connection) -> None:
        db.insert_file("/kept.jpg", "in-index", "image", 1, 1.0)
        for sha, age in (("in-index", 400), ("recent-orphan", 10), ("old-orphan", 400)):
            db.execute(
                "INSERT INTO captions (sha256, model, caption, created_at) "
                "VALUES (?, 'm', 'c', datetime('now', ?))",
                (sha, f"-{age} days"),
            )

        assert db.prune_captions() == 1
        remaining = {r["sha256"] for r in db.query("SELECT sha256 FROM captions")}
        assert remaining == {"in-index", "recent-orphan"}


class TestScanRenderBound:
    def test_huge_page_render_stays_bounded(self) -> None:
        import pymupdf

        from mcp_trove_crunchtools.extractor import SCANNED_PDF_MAX_PIXELS, _render_page

        doc = pymupdf.open()
        page = doc.new_page(width=14400, height=14400)  # 200 x 200 inches

        rendered = pymupdf.Pixmap(_render_page(page))

        assert rendered.width * rendered.height <= SCANNED_PDF_MAX_PIXELS
        doc.close()


class TestOutOfCredit:
    async def test_empty_account_defers_the_rest_without_poisoning(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        """A 402 recorded per file would mark every photo a permanent failure."""
        from mcp_trove_crunchtools import vision
        from mcp_trove_crunchtools.errors import ExtractionError

        for n in range(4):
            (tmp_path / f"photo{n}.jpg").write_bytes(b"\xff\xd8" + bytes([n]))
        backend = MagicMock()
        backend.caption.side_effect = ExtractionError(
            "photo", "Error code: 402 - Insufficient credits. Add more using the dashboard"
        )

        with patch.object(vision, "get_backend", return_value=backend):
            results = await index_path_async(tmp_path)

        assert [r["status"] for r in results] == ["deferred"] * 4
        assert backend.caption.call_count == 1
        assert db.query("SELECT id FROM index_errors") == []
        assert db.known_failures() == set()
        assert vision.out_of_credit()

    def test_rate_limits_are_retried(self) -> None:
        assert db.classify_error("Error code: 429 - Rate limit exceeded") == "transient"
