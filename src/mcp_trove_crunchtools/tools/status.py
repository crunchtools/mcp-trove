"""Status and inspection tools for mcp-trove-crunchtools."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .. import database as db
from ..config import get_config
from ..errors import FileNotIndexedError


async def trove_status() -> dict[str, Any]:
    """Index statistics: total files, chunks, disk usage, model info."""
    config = get_config()

    file_stats = db.query_one(
        "SELECT COUNT(*) as total_files, "
        "COALESCE(SUM(file_size), 0) as total_size, "
        "COALESCE(SUM(chunk_count), 0) as total_chunks "
        "FROM files"
    )

    type_counts = db.query(
        "SELECT file_type, COUNT(*) as count FROM files GROUP BY file_type"
    )

    last_indexed = db.query_one(
        "SELECT indexed_at FROM files ORDER BY indexed_at DESC LIMIT 1"
    )

    last_run = db.query_one(
        "SELECT id, started_at, finished_at, path, status, "
        "files_found, files_indexed, files_skipped, files_errored, "
        "total_chunks, error_message "
        "FROM index_runs ORDER BY id DESC LIMIT 1"
    )

    return {
        "total_files": file_stats["total_files"] if file_stats else 0,
        "total_chunks": file_stats["total_chunks"] if file_stats else 0,
        "total_file_size_bytes": file_stats["total_size"] if file_stats else 0,
        "files_by_type": {row["file_type"]: row["count"] for row in type_counts},
        "embedding_model": config.embedding_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "database_path": config.db_path,
        "last_indexed": last_indexed["indexed_at"] if last_indexed else None,
        "last_run": dict(last_run) if last_run else None,
    }


async def trove_list(
    path: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List indexed files with metadata."""
    if path:
        files = db.query(
            "SELECT id, path, file_type, file_size, chunk_count, indexed_at "
            "FROM files WHERE path LIKE ? "
            "ORDER BY indexed_at DESC LIMIT ? OFFSET ?",
            (path + "%", limit, offset),
        )
    else:
        files = db.query(
            "SELECT id, path, file_type, file_size, chunk_count, indexed_at "
            "FROM files ORDER BY indexed_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )

    return [
        {
            "id": row["id"],
            "path": row["path"],
            "file_type": row["file_type"],
            "file_size": row["file_size"],
            "chunk_count": row["chunk_count"],
            "indexed_at": row["indexed_at"],
        }
        for row in files
    ]


async def trove_log(limit: int = 20) -> list[dict[str, Any]]:
    """Return recent index runs from the activity log."""
    rows = db.query(
        "SELECT id, started_at, finished_at, path, status, "
        "files_found, files_indexed, files_skipped, files_errored, "
        "total_chunks, error_message "
        "FROM index_runs ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    return [dict(row) for row in rows]


async def trove_get_chunks(
    file_path: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Show the text chunks for a specific indexed file."""
    existing = db.query_one(
        "SELECT id FROM files WHERE path = ?", (file_path,)
    )
    if not existing:
        raise FileNotIndexedError(file_path)

    chunks = db.query(
        "SELECT id, chunk_index, content, metadata "
        "FROM chunks WHERE file_id = ? "
        "ORDER BY chunk_index LIMIT ?",
        (existing["id"], limit),
    )

    return [
        {
            "chunk_id": row["id"],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
        }
        for row in chunks
    ]


async def trove_quality(
    path: str | None = None,
    show_resolved: bool = False,
    limit: int = 100,
) -> dict[str, Any]:
    """Per-file error summary and details from indexing runs."""
    resolved_filter: bool | None = None if show_resolved else False
    errors = db.query_errors(resolved=resolved_filter, path=path, limit=limit)

    all_errors = db.query_errors(resolved=None, path=path, limit=10_000)
    total = len(all_errors)
    resolved_count = sum(1 for e in all_errors if e["resolved"])
    unresolved_count = total - resolved_count

    by_type: dict[str, int] = dict(Counter(e["error_type"] for e in all_errors))

    return {
        "total_errors": total,
        "unresolved": unresolved_count,
        "resolved": resolved_count,
        "by_type": by_type,
        "errors": errors,
    }
