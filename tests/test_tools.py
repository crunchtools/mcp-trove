"""Tests for mcp-trove-crunchtools tools."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from mcp_trove_crunchtools import config as config_mod
from mcp_trove_crunchtools import database as db_mod
from mcp_trove_crunchtools import indexer as indexer_mod
from mcp_trove_crunchtools.errors import DimensionMismatchError, InvalidInputError
from mcp_trove_crunchtools.server import mcp
from mcp_trove_crunchtools.tools.index import trove_index, trove_reindex, trove_remove
from mcp_trove_crunchtools.tools.search import trove_search, trove_similar
from mcp_trove_crunchtools.tools.status import (
    trove_get_chunks,
    trove_list,
    trove_log,
    trove_quality,
    trove_status,
)

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import AsyncIterator

EXPECTED_TOOL_COUNT = 10


class TestToolCount:
    @pytest.mark.asyncio
    async def test_tool_count(self) -> None:
        tools = await mcp.list_tools()
        assert len(tools) == EXPECTED_TOOL_COUNT


READ_ONLY = frozenset(
    {
        "trove_search_tool",
        "trove_similar_tool",
        "trove_status_tool",
        "trove_log_tool",
        "trove_list_tool",
        "trove_get_chunks_tool",
        "trove_quality_tool",
    }
)
WRITES = frozenset(
    {
        "trove_index_tool",
        "trove_reindex_tool",
        "trove_remove_tool",
    }
)


class TestReadOnlyAnnotation:
    """Every registered tool is classified, and the reads really only read."""

    @pytest.fixture
    async def indexed_file(self, in_memory_db: sqlite3.Connection) -> AsyncIterator[str]:
        """Index one file, then make the connection refuse every write."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Python is a programming language known for its simplicity.")
            path = str(Path(f.name).resolve())
        await trove_index(path)
        run_id = db_mod.start_run(path, 1)
        db_mod.insert_error(run_id, path, "connection reset by peer", "transient")
        in_memory_db.execute("PRAGMA query_only = ON")
        yield path
        Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_every_tool_is_classified(self) -> None:
        tools = await mcp.list_tools()
        assert READ_ONLY.isdisjoint(WRITES)
        assert {tool.name for tool in tools} == READ_ONLY | WRITES
        annotated = {
            tool.name
            for tool in tools
            if tool.annotations is not None
            and tool.annotations.model_dump(by_alias=True).get("readOnlyHint") is True
        }
        assert annotated == READ_ONLY

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", sorted(READ_ONLY))
    async def test_read_only_tool_writes_nothing(
        self, name: str, indexed_file: str, in_memory_db: sqlite3.Connection
    ) -> None:
        """The backend is SQLite, so a write is a statement that changes the database.

        `PRAGMA query_only` makes SQLite raise on any such statement, and
        `total_changes` counts the rows a connection has inserted, updated or
        deleted. The indexer's embedding and vision entry points must stay idle.
        """
        calls: dict[str, dict[str, object]] = {
            "trove_search_tool": {"query": "programming language"},
            "trove_similar_tool": {"file_path": indexed_file},
            "trove_status_tool": {},
            "trove_log_tool": {},
            "trove_list_tool": {},
            "trove_get_chunks_tool": {"file_path": indexed_file},
            "trove_quality_tool": {"show_resolved": True},
        }
        assert calls.keys() == READ_ONLY
        before = in_memory_db.total_changes
        with (
            patch.object(indexer_mod, "embed_texts") as embed_texts,
            patch.object(indexer_mod, "extract_text") as extract_text,
        ):
            result = await mcp.call_tool(name, calls[name])
        assert result.structured_content
        assert in_memory_db.total_changes == before
        embed_texts.assert_not_called()
        extract_text.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", sorted(WRITES))
    async def test_write_tool_is_caught_by_the_same_guard(
        self, name: str, indexed_file: str
    ) -> None:
        """Control: the guard above refuses each write, so that test can fail."""
        with pytest.raises(Exception, match="readonly database"):
            await mcp.call_tool(name, {"path": indexed_file})


class TestIndexTools:
    @pytest.mark.asyncio
    async def test_index_single_file(self, in_memory_db: sqlite3.Connection) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document about Python programming.")
            path = f.name

        result = await trove_index(path)
        assert result["files_indexed"] == 1
        assert result["files_skipped"] == 0
        assert result["total_chunks"] >= 1

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_index_directory(self, in_memory_db: sqlite3.Connection) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.txt").write_text("First document content.")
            (Path(tmpdir) / "file2.md").write_text("# Second document\n\nMarkdown content.")
            (Path(tmpdir) / "skip.iso").write_text("binary")

            result = await trove_index(tmpdir)
            assert result["files_indexed"] == 2
            assert result["total_chunks"] >= 2

    @pytest.mark.asyncio
    async def test_index_skips_unchanged(self, in_memory_db: sqlite3.Connection) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Unchanged content here.")
            path = f.name

        await trove_index(path)
        result = await trove_index(path)
        assert result["files_skipped"] == 1
        assert result["files_indexed"] == 0

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_index_nonexistent_path(self, in_memory_db: sqlite3.Connection) -> None:
        from mcp_trove_crunchtools.errors import PathNotFoundError

        with pytest.raises(PathNotFoundError):
            await trove_index("/nonexistent/path/file.txt")

    @pytest.mark.asyncio
    async def test_reindex_file(self, in_memory_db: sqlite3.Connection) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Content to be reindexed.")
            path = f.name

        await trove_index(path)
        result = await trove_reindex(path)
        assert result["files_reindexed"] == 1

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_reindex_all(self, in_memory_db: sqlite3.Connection) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Content for reindex all test.")
            path = f.name

        await trove_index(path)
        result = await trove_reindex()
        assert result["files_reindexed"] >= 1

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_remove_file(self, in_memory_db: sqlite3.Connection) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Content to be removed.")
            path = f.name

        await trove_index(path)
        result = await trove_remove(path)
        assert result["files_removed"] == 1

        files = await trove_list()
        assert len(files) == 0

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, in_memory_db: sqlite3.Connection) -> None:
        result = await trove_remove("/nonexistent/file.txt")
        assert result["files_removed"] == 0


class TestSearchTools:
    @pytest.fixture(autouse=True)
    async def _setup_indexed_files(self, in_memory_db: sqlite3.Connection) -> None:
        """Seed the index with test files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "Python is a programming language known for its simplicity. "
                "It is widely used in data science, web development, and automation."
            )
            self._file1 = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "Rust is a systems programming language focused on safety and performance. "
                "It prevents memory errors at compile time."
            )
            self._file2 = f.name

        await trove_index(self._file1)
        await trove_index(self._file2)

    @pytest.mark.asyncio
    async def test_search_returns_results(self) -> None:
        results = await trove_search("programming language")
        assert len(results) >= 1
        assert "content" in results[0]
        assert "path" in results[0]

    @pytest.mark.asyncio
    async def test_search_with_path_filter(self) -> None:
        parent = str(Path(self._file1).parent)
        results = await trove_search("Python", path=parent)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_similar_files(self) -> None:
        resolved = str(Path(self._file1).resolve())
        results = await trove_similar(resolved)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_similar_not_indexed(self) -> None:
        from mcp_trove_crunchtools.errors import FileNotIndexedError

        with pytest.raises(FileNotIndexedError):
            await trove_similar("/nonexistent/file.txt")

    def teardown_method(self) -> None:
        for attr in ("_file1", "_file2"):
            path = getattr(self, attr, None)
            if path:
                Path(path).unlink(missing_ok=True)


class TestStatusTools:
    @pytest.mark.asyncio
    async def test_status_empty(self, in_memory_db: sqlite3.Connection) -> None:
        status = await trove_status()
        assert status["total_files"] == 0
        assert status["total_chunks"] == 0
        assert "embedding_model" in status

    @pytest.mark.asyncio
    async def test_status_with_files(self, in_memory_db: sqlite3.Connection) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Status test content.")
            path = f.name

        await trove_index(path)
        status = await trove_status()
        assert status["total_files"] == 1
        assert status["total_chunks"] >= 1

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_list_empty(self, in_memory_db: sqlite3.Connection) -> None:
        files = await trove_list()
        assert files == []

    @pytest.mark.asyncio
    async def test_list_with_files(self, in_memory_db: sqlite3.Connection) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("List test content.")
            path = f.name

        await trove_index(path)
        files = await trove_list()
        assert len(files) == 1
        assert "path" in files[0]
        assert "file_type" in files[0]
        assert "chunk_count" in files[0]

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_get_chunks(self, in_memory_db: sqlite3.Connection) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Chunk test content for inspection.")
            path = f.name

        await trove_index(path)
        resolved = str(Path(path).resolve())
        chunks = await trove_get_chunks(resolved)
        assert len(chunks) >= 1
        assert "content" in chunks[0]
        assert "chunk_index" in chunks[0]

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_get_chunks_not_indexed(self, in_memory_db: sqlite3.Connection) -> None:
        from mcp_trove_crunchtools.errors import FileNotIndexedError

        with pytest.raises(FileNotIndexedError):
            await trove_get_chunks("/nonexistent/file.txt")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("limit", [0, -1, 501])
    async def test_log_and_quality_reject_bad_limit(
        self, in_memory_db: sqlite3.Connection, limit: int
    ) -> None:
        with pytest.raises(InvalidInputError):
            await trove_log(limit)
        with pytest.raises(InvalidInputError):
            await trove_quality(limit=limit)

    @pytest.mark.asyncio
    async def test_log_empty(self, in_memory_db: sqlite3.Connection) -> None:
        logs = await trove_log()
        assert logs == []

    @pytest.mark.asyncio
    async def test_log_after_index(self, in_memory_db: sqlite3.Connection) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Log test content for indexing.")
            path = f.name

        await trove_index(path)
        logs = await trove_log()
        assert len(logs) >= 1
        latest = logs[0]
        assert latest["status"] == "completed"
        assert latest["files_indexed"] == 1
        assert latest["files_errored"] == 0
        assert latest["error_message"] is None

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_status_includes_last_run(self, in_memory_db: sqlite3.Connection) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Last run test content.")
            path = f.name

        await trove_index(path)
        status = await trove_status()
        assert status["last_run"] is not None
        assert status["last_run"]["status"] == "completed"

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_quality_empty(self, in_memory_db: sqlite3.Connection) -> None:
        result = await trove_quality()
        assert result["total_errors"] == 0
        assert result["unresolved"] == 0
        assert result["resolved"] == 0
        assert result["by_type"] == {}
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_quality_after_error(self, in_memory_db: sqlite3.Connection) -> None:
        from mcp_trove_crunchtools import database as test_db

        run_id = test_db.start_run("/nonexistent/test", 1)
        test_db.insert_error(
            run_id,
            "/nonexistent/test/bad.pdf",
            "connection reset by peer",
            "transient",
        )
        test_db.insert_error(
            run_id,
            "/nonexistent/test/corrupt.pdf",
            "invalid PDF structure",
            "permanent",
        )

        result = await trove_quality()
        assert result["total_errors"] == 2
        assert result["unresolved"] == 2
        assert result["resolved"] == 0
        assert result["by_type"]["transient"] == 1
        assert result["by_type"]["permanent"] == 1
        assert len(result["errors"]) == 2

        test_db.resolve_errors("/nonexistent/test/bad.pdf")
        result = await trove_quality(show_resolved=True)
        assert result["resolved"] == 1
        assert result["unresolved"] == 1

        result = await trove_quality()
        assert len(result["errors"]) == 1
        assert result["errors"][0]["path"] == "/nonexistent/test/corrupt.pdf"


class TestVectorDimensions:
    """The vector table size follows the configured embedding model."""

    def test_dims_from_model_metadata(self) -> None:
        from mcp_trove_crunchtools.embedder import get_vector_dims

        assert get_vector_dims("BAAI/bge-small-en-v1.5") == 384
        assert get_vector_dims("intfloat/multilingual-e5-large") == 1024

    def test_new_database_uses_model_dims(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("TROVE_EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        conn = db_mod.get_db(str(tmp_path / "t.db"))
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'chunks_vec'").fetchone()[0]
        assert "float[1024]" in sql

    def test_mismatch_fails_with_both_dims(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path = str(tmp_path / "t.db")
        db_mod.get_db(path).close()
        db_mod._db = None
        config_mod._config = None
        monkeypatch.setenv("TROVE_EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        with pytest.raises(DimensionMismatchError, match=r"384.*1024"):
            db_mod.get_db(path)
