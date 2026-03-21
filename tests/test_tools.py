"""Tests for mcp-trove-crunchtools tools."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

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

EXPECTED_TOOL_COUNT = 10


class TestToolCount:
    @pytest.mark.asyncio
    async def test_tool_count(self) -> None:
        tools = await mcp.list_tools()
        assert len(tools) == EXPECTED_TOOL_COUNT


class TestIndexTools:
    @pytest.mark.asyncio
    async def test_index_single_file(self, in_memory_db: sqlite3.Connection) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Content to be reindexed.")
            path = f.name

        await trove_index(path)
        result = await trove_reindex(path)
        assert result["files_reindexed"] == 1

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_reindex_all(self, in_memory_db: sqlite3.Connection) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Content for reindex all test.")
            path = f.name

        await trove_index(path)
        result = await trove_reindex()
        assert result["files_reindexed"] >= 1

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_remove_file(self, in_memory_db: sqlite3.Connection) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(
                "Python is a programming language known for its simplicity. "
                "It is widely used in data science, web development, and automation."
            )
            self._file1 = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
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
    async def test_log_empty(self, in_memory_db: sqlite3.Connection) -> None:
        logs = await trove_log()
        assert logs == []

    @pytest.mark.asyncio
    async def test_log_after_index(self, in_memory_db: sqlite3.Connection) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
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
            run_id, "/nonexistent/test/bad.pdf",
            "connection reset by peer", "transient",
        )
        test_db.insert_error(
            run_id, "/nonexistent/test/corrupt.pdf",
            "invalid PDF structure", "permanent",
        )

        result = await trove_quality()
        assert result["total_errors"] == 2
        assert result["unresolved"] == 2
        assert result["resolved"] == 0
        assert result["by_type"]["transient"] == 1
        assert result["by_type"]["permanent"] == 1
        assert len(result["errors"]) == 2

        # Resolve one error and verify
        test_db.resolve_errors("/nonexistent/test/bad.pdf")
        result = await trove_quality(show_resolved=True)
        assert result["resolved"] == 1
        assert result["unresolved"] == 1

        # Default (show_resolved=False) should only return unresolved
        result = await trove_quality()
        assert len(result["errors"]) == 1
        assert result["errors"][0]["path"] == "/nonexistent/test/corrupt.pdf"
