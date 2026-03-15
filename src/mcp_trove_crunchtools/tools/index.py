"""Index management tools for mcp-trove-crunchtools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import PathNotFoundError
from ..indexer import index_path, remove_path


async def trove_index(path: str) -> dict[str, Any]:
    """Index a specific file or directory.

    Skips unchanged files based on checksum comparison.
    For directories, scans recursively and indexes supported files.
    """
    target = Path(path).resolve()
    if not target.exists():
        raise PathNotFoundError(path)

    results = index_path(target, force=False)

    indexed = sum(1 for r in results if r["status"] == "indexed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    total_chunks = sum(int(r.get("chunk_count", 0)) for r in results)

    return {
        "path": str(target),
        "files_indexed": indexed,
        "files_skipped": skipped,
        "total_chunks": total_chunks,
        "details": results,
    }


async def trove_reindex(path: str | None = None) -> dict[str, Any]:
    """Force re-index ignoring checksums.

    If no path given, reindexes all previously indexed files.
    """
    from .. import database as db

    if path:
        target = Path(path).resolve()
        if not target.exists():
            raise PathNotFoundError(path)
        results = index_path(target, force=True)
    else:
        all_files = db.query("SELECT path FROM files")
        results = []
        for row in all_files:
            file_path = Path(row["path"])
            if file_path.exists():
                results.extend(index_path(file_path, force=True))
            else:
                existing = db.query_one(
                    "SELECT id FROM files WHERE path = ?", (row["path"],)
                )
                if existing:
                    db.delete_file_data(existing["id"])
                    results.append({
                        "path": row["path"],
                        "status": "removed",
                        "reason": "file_missing",
                        "chunk_count": 0,
                    })

    indexed = sum(1 for r in results if r["status"] == "indexed")
    removed = sum(1 for r in results if r.get("status") == "removed")
    total_chunks = sum(int(r.get("chunk_count", 0)) for r in results)

    return {
        "files_reindexed": indexed,
        "files_removed": removed,
        "total_chunks": total_chunks,
        "details": results,
    }


async def trove_remove(path: str) -> dict[str, Any]:
    """Remove a file or directory from the index.

    Deletes all associated chunks and vector embeddings.
    """
    target = Path(path).resolve()
    result = remove_path(target)
    return {
        "path": result["path"],
        "files_removed": result["removed"],
    }
