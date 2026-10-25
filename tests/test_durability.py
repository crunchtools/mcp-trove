"""Tests for unattended-operation safety: atomic stores, repair, gates (RT #1542)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import threading
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

import mcp_trove_crunchtools as trove_pkg
from mcp_trove_crunchtools import database as db
from mcp_trove_crunchtools import indexer as indexer_mod
from mcp_trove_crunchtools.errors import EmbeddingModelMismatchError, IndexAbortedError
from mcp_trove_crunchtools.indexer import _store_one, index_path_async
from mcp_trove_crunchtools.tools import run_index as run_index_mod
from mcp_trove_crunchtools.tools.run_index import run_index

if TYPE_CHECKING:
    from pathlib import Path


def _extraction(path: str, chunks: int, existing_id: int | None = None) -> dict[str, Any]:
    return {
        "path": path,
        "status": "extracted",
        "file_type": "text",
        "checksum": "new",
        "file_size": 1,
        "mtime": 1.0,
        "chunks": [f"chunk {i}" for i in range(chunks)],
        "existing_id": existing_id,
    }


def _counts() -> tuple[int, int, int]:
    return (
        len(db.query("SELECT id FROM files")),
        len(db.query("SELECT id FROM chunks")),
        len(db.query("SELECT chunk_id FROM chunks_vec")),
    )


class TestAtomicStore:
    """A file's rows are written in one transaction, all or nothing."""

    def test_transaction_rolls_back_on_error(self, in_memory_db: sqlite3.Connection) -> None:
        def write_then_fail() -> None:
            with db.transaction():
                db.insert_file("/a.txt", "x", "text", 1, 1.0)
                raise RuntimeError

        with pytest.raises(RuntimeError):
            write_then_fail()

        assert db.query("SELECT id FROM files") == []

    def test_nested_transaction_commits_once(self, in_memory_db: sqlite3.Connection) -> None:
        with db.transaction():
            with db.transaction():
                db.insert_file("/a.txt", "x", "text", 1, 1.0)
            assert in_memory_db.in_transaction

        assert not in_memory_db.in_transaction
        assert len(db.query("SELECT id FROM files")) == 1

    def test_interrupted_store_leaves_no_partial_file(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        """KeyboardInterrupt stands in for a kill: it bypasses `except Exception`."""
        calls = 0
        real_insert_vector = db.insert_vector

        def dying_insert_vector(chunk_id: int, embedding: list[float] | bytes) -> None:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise KeyboardInterrupt
            real_insert_vector(chunk_id, embedding)

        with (
            patch.object(db, "insert_vector", dying_insert_vector),
            pytest.raises(KeyboardInterrupt),
        ):
            _store_one(_extraction("/data/manual.txt", 5))

        assert _counts() == (0, 0, 0)

    def test_failed_reindex_keeps_previous_entry(self, in_memory_db: sqlite3.Connection) -> None:
        _store_one(_extraction("/data/notes.txt", 2))
        row = db.query_one("SELECT id FROM files")
        assert row is not None
        old_id = row["id"]

        with (
            patch.object(indexer_mod, "embed_texts", side_effect=RuntimeError("backend down")),
            pytest.raises(RuntimeError),
        ):
            _store_one(_extraction("/data/notes.txt", 3, existing_id=old_id))

        assert _counts() == (1, 2, 2)


class TestRepairMigration:
    """Opening a pre-0.10 database drops rows left by a killed run."""

    def test_incomplete_rows_removed_once(self, tmp_path: Path) -> None:
        path = str(tmp_path / "trove.db")
        db.get_db(path)
        _store_one(_extraction("/good.txt", 2))
        _store_one(_extraction("/partial.txt", 3))
        db.execute("DELETE FROM chunks WHERE chunk_index = 2")
        db.insert_file("/no-chunks.txt", "x", "text", 1, 1.0)
        db.get_db().execute("PRAGMA user_version = 0")
        db.get_db().commit()
        db.get_db().close()
        db._db = None

        db.get_db(path)

        paths = {row["path"] for row in db.query("SELECT path FROM files")}
        assert paths == {"/good.txt"}
        assert db.get_db().execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION


class TestModelGuard:
    def test_different_model_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = str(tmp_path / "trove.db")
        db.get_db(path).close()
        db._db = None
        # Same 384 dimensions, so only the recorded model name can tell them apart.
        monkeypatch.setenv("TROVE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        from mcp_trove_crunchtools import config as config_mod

        config_mod._config = None

        with pytest.raises(EmbeddingModelMismatchError, match="bge-small"):
            db.get_db(path)


class TestStaleRuns:
    def test_old_running_rows_aborted(self, in_memory_db: sqlite3.Connection) -> None:
        old = db.start_run("/old", 1)
        db.execute(
            "UPDATE index_runs SET started_at = datetime('now', '-13 hours') WHERE id = ?", (old,)
        )
        recent = db.start_run("/recent", 1)

        assert db.abort_stale_runs() == 1
        statuses = {r["id"]: r["status"] for r in db.query("SELECT id, status FROM index_runs")}
        assert statuses == {old: "aborted", recent: "running"}


class TestShrinkGate:
    """A tree that suddenly lost most of its files is treated as a broken source."""

    def _previous_run(self, directory: Path, files_found: int) -> None:
        for n in range(files_found):
            db.insert_file(str(directory.resolve() / f"gone{n}.txt"), "sum", "text", 1, 1.0)

    async def test_shrunken_tree_aborts(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        (tmp_path / "left.txt").write_text("the only file left")
        self._previous_run(tmp_path, 200)

        with pytest.raises(IndexAbortedError, match="unmounted"):
            await index_path_async(tmp_path)

        newest = db.query_one("SELECT status FROM index_runs ORDER BY id DESC LIMIT 1")
        assert newest == {"status": "aborted"}

    async def test_force_skips_the_gate(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        (tmp_path / "left.txt").write_text("the only file left")
        self._previous_run(tmp_path, 200)

        results = await index_path_async(tmp_path, force=True)

        assert [r["status"] for r in results] == ["indexed"]


class TestHungReads:
    async def test_repeated_timeouts_abort_the_run(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        for n in range(6):
            (tmp_path / f"f{n}.txt").write_text("x")

        with (
            patch.object(indexer_mod, "_extract_one", side_effect=TimeoutError),
            pytest.raises(IndexAbortedError, match="hung"),
        ):
            await index_path_async(tmp_path)

        newest = db.query_one("SELECT status FROM index_runs ORDER BY id DESC LIMIT 1")
        assert newest == {"status": "aborted"}


class TestRunIndexCli:
    """The --index run exits non-zero and says why whenever it did not succeed."""

    @pytest.fixture
    def env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text("alpha")
        monkeypatch.setenv("TROVE_DB", str(tmp_path / "data" / "trove.db"))
        monkeypatch.setenv("TROVE_PATHS", str(docs))
        return tmp_path

    def _summary(self, env: Path) -> dict[str, Any]:
        return json.loads((env / "data" / "last_run.json").read_text())

    def test_success_writes_summary_and_backup(self, env: Path) -> None:
        assert run_index() == 0

        summary = self._summary(env)
        assert summary["status"] == "ok"
        assert summary["paths"][0]["files_indexed"] == 1
        assert len(list((env / "data" / "backups").glob("trove-*.db"))) == 1

    def test_old_backups_pruned(self, env: Path) -> None:
        backups = env / "data" / "backups"
        backups.mkdir(parents=True)
        for day in ("20200101", "20200102", "20200103"):
            (backups / f"trove-{day}.db").write_text("old")

        assert run_index() == 0

        kept = sorted(p.name for p in backups.glob("trove-*.db"))
        assert len(kept) == 3
        assert "trove-20200101.db" not in kept

    def test_unmounted_source_fails(self, env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TROVE_REQUIRE_MOUNT", str(env / "docs"))

        assert run_index() == 1

        summary = self._summary(env)
        assert summary["status"] == "failed"
        assert "not mounted" in summary["problems"][0]
        assert summary["paths"] == []

    def test_missing_path_fails(self, env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TROVE_PATHS", str(env / "gone"))

        assert run_index() == 1
        assert self._summary(env)["paths"][0]["status"] == "missing"

    def test_high_error_rate_fails(self, env: Path) -> None:
        for n in range(25):
            (env / "docs" / f"f{n}.txt").write_text("x")

        with patch.object(indexer_mod, "_extract_one", side_effect=ValueError("corrupt")):
            assert run_index() == 1

        assert self._summary(env)["paths"][0]["status"] == "errors"


class TestRunIndexFailurePaths:
    """Each preflight failure ends the run non-zero with the reason recorded."""

    @pytest.fixture
    def env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        docs = tmp_path / "docs"
        docs.mkdir()
        monkeypatch.setenv("TROVE_DB", str(tmp_path / "data" / "trove.db"))
        monkeypatch.setenv("TROVE_PATHS", str(docs))
        return tmp_path

    def _summary(self, env: Path) -> dict[str, Any]:
        return json.loads((env / "data" / "last_run.json").read_text())

    def test_failed_quick_check_stops_the_run(self, env: Path) -> None:
        with patch.object(db, "quick_check", return_value="row 5 missing from index"):
            assert run_index() == 1

        summary = self._summary(env)
        assert "quick_check" in summary["problems"][0]
        assert summary["paths"] == []

    def test_failed_backup_fails_the_run_but_still_indexes(self, env: Path) -> None:
        (env / "docs" / "a.txt").write_text("alpha")

        with patch.object(db, "backup_to", side_effect=OSError("disk full")):
            assert run_index() == 1

        summary = self._summary(env)
        assert "disk full" in summary["problems"][0]
        assert summary["paths"][0]["files_indexed"] == 1

    def test_backup_is_private(self, env: Path) -> None:
        assert run_index() == 0

        backups = env / "data" / "backups"
        assert backups.stat().st_mode & 0o777 == 0o700
        assert all(p.stat().st_mode & 0o777 == 0o600 for p in backups.glob("trove-*.db"))

    def test_hung_mount_listing_fails_fast(
        self, env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TROVE_REQUIRE_MOUNT", str(env / "docs"))
        never = threading.Event()
        with (
            patch.object(run_index_mod.os.path, "ismount", return_value=True),
            patch.object(run_index_mod.os, "listdir", side_effect=lambda _: never.wait()),
            patch.object(run_index_mod, "MOUNT_PROBE_SECONDS", 0.1),
        ):
            assert run_index() == 1
        never.set()

        assert "did not answer" in self._summary(env)["problems"][0]

    def test_unlistable_mount_fails(self, env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TROVE_REQUIRE_MOUNT", str(env / "docs"))
        with (
            patch.object(run_index_mod.os.path, "ismount", return_value=True),
            patch.object(run_index_mod.os, "listdir", side_effect=OSError("Transport endpoint")),
        ):
            assert run_index() == 1

        assert "cannot be listed" in self._summary(env)["problems"][0]

    @pytest.mark.parametrize(("failing", "expected"), [(0, 0), (1, 0), (2, 1)])
    def test_error_rate_threshold(self, env: Path, failing: int, expected: int) -> None:
        """10 files: one failure is 10% and passes, two is 20% and fails."""
        for n in range(10):
            (env / "docs" / f"f{n}.txt").write_text("x")
        broken = {f"f{n}.txt" for n in range(failing)}
        real_extract = indexer_mod._extract_one

        def extract(path: Path, *args: Any) -> dict[str, Any]:
            if path.name in broken:
                raise ValueError("corrupt")
            return real_extract(path, *args)

        with patch.object(indexer_mod, "_extract_one", extract):
            assert run_index() == expected

    def test_single_failure_in_small_path_fails(self, env: Path) -> None:
        (env / "docs" / "only.txt").write_text("x")

        with patch.object(indexer_mod, "_extract_one", side_effect=ValueError("corrupt")):
            assert run_index() == 1


class TestRunAsync:
    def test_does_not_wait_for_a_stuck_worker_thread(self) -> None:
        """asyncio.run() would join the stuck thread; _run_async must not."""
        never = threading.Event()

        async def stuck_read() -> dict[str, Any]:
            await asyncio.wait_for(asyncio.to_thread(never.wait), 0.1)
            return {}

        started = time.monotonic()
        with pytest.raises(TimeoutError):
            run_index_mod._run_async(stuck_read())
        elapsed = time.monotonic() - started
        never.set()

        assert elapsed < 2


class TestNestedTransaction:
    def test_caught_inner_failure_is_still_undone(self, in_memory_db: sqlite3.Connection) -> None:
        def inner_write_then_fail() -> None:
            with db.transaction():
                db.insert_file("/inner.txt", "x", "text", 1, 1.0)
                raise RuntimeError

        with db.transaction():
            db.insert_file("/outer.txt", "x", "text", 1, 1.0)
            with pytest.raises(RuntimeError):
                inner_write_then_fail()

        assert [r["path"] for r in db.query("SELECT path FROM files")] == ["/outer.txt"]


class TestRunIndexAlwaysReports:
    @pytest.fixture
    def env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        docs = tmp_path / "docs"
        docs.mkdir()
        monkeypatch.setenv("TROVE_DB", str(tmp_path / "data" / "trove.db"))
        monkeypatch.setenv("TROVE_PATHS", str(docs))
        return tmp_path

    def test_unexpected_preflight_error_still_writes_summary(self, env: Path) -> None:
        with patch.object(db, "abort_stale_runs", side_effect=sqlite3.OperationalError("locked")):
            assert run_index() == 1

        summary = json.loads((env / "data" / "last_run.json").read_text())
        assert summary["status"] == "failed"
        assert "locked" in summary["problems"][0]

    def test_unwritable_summary_fails_the_run(
        self, env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        blocker = env / "not-a-dir"
        blocker.write_text("a file where the status directory should be")
        monkeypatch.setenv("TROVE_STATUS_FILE", str(blocker / "last_run.json"))

        assert run_index() == 1


class TestIndexEntryPoint:
    def test_main_exits_with_the_run_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--index ends in os._exit(code) so a stuck worker thread cannot hold it open."""
        monkeypatch.setattr(sys, "argv", ["mcp-trove-crunchtools", "--index"])
        with (
            patch.object(run_index_mod, "run_index", return_value=1),
            patch.object(trove_pkg.os, "_exit", side_effect=SystemExit) as exit_,
            pytest.raises(SystemExit),
        ):
            trove_pkg.main()

        exit_.assert_called_once_with(1)


class TestSummaryForMonitoring:
    @pytest.fixture
    def env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        (tmp_path / "docs").mkdir()
        monkeypatch.setenv("TROVE_DB", str(tmp_path / "data" / "trove.db"))
        monkeypatch.setenv("TROVE_PATHS", str(tmp_path / "docs"))
        return tmp_path / "data" / "last_run.json"

    def test_summary_names_the_backup_and_budget(self, env: Path) -> None:
        assert run_index() == 0

        summary = json.loads(env.read_text())
        assert summary["latest_backup"]["file"].startswith("trove-")
        assert summary["vision_budget"] == 3000

    def test_summary_is_private_by_default(self, env: Path) -> None:
        assert run_index() == 0

        assert env.stat().st_mode & 0o777 == 0o600

    def test_summary_mode_can_open_it_to_a_monitor(
        self, env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TROVE_STATUS_FILE_MODE", "644")

        assert run_index() == 0

        assert env.stat().st_mode & 0o777 == 0o644

    def test_no_backups_recorded_as_none(self, env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TROVE_BACKUP_KEEP", "0")

        assert run_index() == 0

        assert json.loads(env.read_text())["latest_backup"] is None
