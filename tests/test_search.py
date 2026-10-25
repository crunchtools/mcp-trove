"""Search ranking: rank fusion, path scope, folder names, one result per file (0.16.0)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mcp_trove_crunchtools import database as db
from mcp_trove_crunchtools.indexer import index_path_async
from mcp_trove_crunchtools.tools.search import trove_search

if TYPE_CHECKING:
    import sqlite3

    import pytest

FILLER = "Quarterly planning notes about meetings, agendas and follow-up items."


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestRanking:
    async def test_exact_keyword_hit_is_not_crowded_out(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        """Vector hits used to fill every slot before any keyword hit was added."""
        for n in range(40):
            _write(tmp_path / f"note{n}.txt", f"{FILLER} Number {n}.")
        _write(tmp_path / "receipt.txt", "AutoZone receipt: Duralast Gold battery, $189.99")
        await index_path_async(tmp_path)

        results = await trove_search("Duralast", limit=5)

        # The test embedder is random, so a vector hit can tie it for first.
        top = {Path(r["path"]).name: r for r in results[:2]}
        assert "fts" in top["receipt.txt"]["sources"]

    async def test_folder_names_rank_a_file_its_text_does_not_describe(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        for n in range(20):
            _write(tmp_path / "Office" / f"memo{n}.txt", f"{FILLER} {n}")
        _write(
            tmp_path / "2008 Jeep Grand Cherokee" / "Maintenance" / "scan-04.txt",
            "Total due 64.50. Thank you for your business.",
        )
        await index_path_async(tmp_path)

        results = await trove_search("jeep maintenance", limit=3)

        assert "2008 Jeep Grand Cherokee" in results[0]["path"]
        assert "path" in results[0]["sources"]

    async def test_one_result_per_file(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        _write(tmp_path / "manual.txt", "\n\n".join(["Valvoline oil change interval. " * 40] * 6))
        _write(tmp_path / "other.txt", "Valvoline coupon")
        await index_path_async(tmp_path)

        results = await trove_search("Valvoline", limit=10)

        paths = [r["path"] for r in results]
        assert len(paths) == len(set(paths))
        manual = next(r for r in results if r["path"].endswith("manual.txt"))
        assert manual["matches"] > 1

    async def test_a_long_file_does_not_leave_too_few_results(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        """Its chunks could fill every candidate slot, leaving one distinct file."""
        _write(
            tmp_path / "manual.txt", "\n\n".join(f"Valvoline section {n}. " * 60 for n in range(40))
        )
        for n in range(4):
            _write(tmp_path / f"coupon{n}.txt", f"Valvoline coupon {n}")
        await index_path_async(tmp_path)

        results = await trove_search("Valvoline", limit=5)

        assert len(results) == 5

    async def test_punctuation_does_not_break_keywords(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        _write(tmp_path / "notes.txt", "We don't ship c++ code")
        await index_path_async(tmp_path)

        results = await trove_search('don\'t "c++" (ship) AND OR NEAR*')

        assert results[0]["path"].endswith("notes.txt")


class TestScope:
    async def test_keyword_hits_respect_the_path(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        _write(tmp_path / "Vehicles" / "battery.txt", "Duralast battery")
        _write(tmp_path / "Residential" / "battery.txt", "Duralast battery for the generator")
        await index_path_async(tmp_path)

        results = await trove_search("Duralast", path=str(tmp_path / "Vehicles"))

        assert results
        assert all("/Vehicles/" in r["path"] for r in results)

    async def test_underscore_in_a_folder_is_not_a_wildcard(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        _write(tmp_path / "tax_2024" / "a.txt", "W-2 wages")
        _write(tmp_path / "taxX2024" / "b.txt", "W-2 wages")
        await index_path_async(tmp_path)

        results = await trove_search("wages", path=str(tmp_path / "tax_2024"))

        assert [Path(r["path"]).name for r in results] == ["a.txt"]

    async def test_scope_outside_the_nearest_pool_is_searched_exactly(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the scope's chunks are not among the pool, every one is measured."""
        monkeypatch.setattr(db, "SCOPED_KNN", 2)
        for n in range(30):
            _write(tmp_path / "Elsewhere" / f"f{n}.txt", f"{FILLER} {n}")
        for n in range(3):
            _write(tmp_path / "Target" / f"t{n}.txt", f"Target note {n}")
        await index_path_async(tmp_path)

        results = await trove_search("anything", path=str(tmp_path / "Target"), limit=3)

        assert sorted(Path(r["path"]).name for r in results) == ["t0.txt", "t1.txt", "t2.txt"]

    async def test_scoped_search_finds_files_beyond_the_global_nearest(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection
    ) -> None:
        """The neighbour search ran over the whole index and was filtered after."""
        for n in range(80):
            _write(tmp_path / "Elsewhere" / f"f{n}.txt", f"{FILLER} {n}")
        _write(tmp_path / "Target" / "only.txt", "A note about something else entirely")
        await index_path_async(tmp_path)

        results = await trove_search("meetings agendas", path=str(tmp_path / "Target"))

        assert [Path(r["path"]).name for r in results] == ["only.txt"]
