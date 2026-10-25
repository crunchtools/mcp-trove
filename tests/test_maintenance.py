"""Re-embedding under a new model, and search evaluation (0.16.0)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from mcp_trove_crunchtools import database as db
from mcp_trove_crunchtools.config import get_config
from mcp_trove_crunchtools.indexer import index_path_async
from mcp_trove_crunchtools.tools import maintenance

if TYPE_CHECKING:
    from pathlib import Path

NEW_MODEL = "BAAI/bge-base-en-v1.5"  # 768 dimensions, where the tests index at 384


def _embed_768(texts: list[str]) -> list[list[float]]:
    return [[1.0] + [0.0] * 767 for _ in texts]


async def _index(tmp_path: Path, files: int) -> Path:
    database = tmp_path / "trove.db"
    db.get_db(str(database))
    for n in range(files):
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / f"f{n}.txt").write_text(f"note number {n}")
    await index_path_async(tmp_path / "docs")
    db.get_db().close()
    db._db = None
    return database


class TestReembed:
    async def test_switches_model_and_dimensions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        database = await _index(tmp_path, 3)
        monkeypatch.setattr(get_config(), "embedding_model", NEW_MODEL)

        with (
            patch.object(maintenance, "embed_texts", side_effect=_embed_768),
            patch.object(maintenance, "get_vector_dims", return_value=768),
            patch.object(db, "get_vector_dims", return_value=768),
        ):
            result = maintenance.reembed(str(database))
            db.get_db(str(database))  # opens with the new model: no mismatch

        assert result == {
            "model": NEW_MODEL,
            "dimensions": 768,
            "chunks": 3,
            "embedded_this_run": 3,
        }
        stored = db.query_one("SELECT value FROM meta WHERE key = 'embedding_model'")
        assert stored == {"value": NEW_MODEL}

    async def test_a_killed_run_resumes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        database = await _index(tmp_path, 5)
        monkeypatch.setattr(get_config(), "embedding_model", NEW_MODEL)
        monkeypatch.setattr(maintenance, "REEMBED_BATCH", 2)
        calls = 0

        def dies_on_the_second_batch(texts: list[str]) -> list[list[float]]:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return _embed_768(texts)

        with (
            patch.object(maintenance, "get_vector_dims", return_value=768),
            patch.object(maintenance, "embed_texts", side_effect=dies_on_the_second_batch),
            pytest.raises(KeyboardInterrupt),
        ):
            maintenance.reembed(str(database))
        with (
            patch.object(maintenance, "get_vector_dims", return_value=768),
            patch.object(maintenance, "embed_texts", side_effect=_embed_768),
        ):
            result = maintenance.reembed(str(database))

        assert result["chunks"] == 5
        assert result["embedded_this_run"] == 3
        conn = sqlite3.connect(database)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        assert "reembed_vectors" not in tables

    async def test_progress_for_another_model_is_discarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        database = await _index(tmp_path, 4)
        conn = sqlite3.connect(database)
        conn.execute(
            "CREATE TABLE reembed_vectors "
            "(chunk_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL, model TEXT NOT NULL)"
        )
        first = conn.execute("SELECT min(id) FROM chunks").fetchone()[0]
        conn.execute("INSERT INTO reembed_vectors VALUES (?, x'00', 'some/other-model')", (first,))
        conn.commit()
        conn.close()
        monkeypatch.setattr(get_config(), "embedding_model", NEW_MODEL)

        with (
            patch.object(maintenance, "get_vector_dims", return_value=768),
            patch.object(maintenance, "embed_texts", side_effect=_embed_768),
        ):
            result = maintenance.reembed(str(database))

        assert result["embedded_this_run"] == 4


class TestEvaluate:
    def test_scores_queries(self, tmp_path: Path, in_memory_db: sqlite3.Connection) -> None:
        """Synchronous, as the CLI calls it: evaluate() runs its own event loop."""
        (tmp_path / "Jeep").mkdir()
        (tmp_path / "Jeep" / "battery.txt").write_text("Duralast Gold battery receipt")
        (tmp_path / "other.txt").write_text("Lawn mower manual")
        asyncio.run(index_path_async(tmp_path))
        queries = tmp_path / "queries.json"
        queries.write_text(
            json.dumps(
                [
                    {"query": "Duralast battery", "expect": ["jeep/battery"]},
                    {"query": "snowblower", "expect": ["nowhere"]},
                ]
            )
        )

        report = maintenance.evaluate(queries)

        assert report["queries"] == 2
        assert report["recall@5"] == 0.5
        assert report["results"][0]["rank"] == 1
        assert report["results"][1]["rank"] is None
