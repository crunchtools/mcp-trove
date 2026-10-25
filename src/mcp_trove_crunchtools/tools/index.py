"""Index management tools for mcp-trove-crunchtools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .. import database as db
from .. import vision
from ..errors import InvalidInputError, PathNotFoundError, ReindexAllNotConfirmedError
from ..indexer import index_path_async, remove_path
from ..models import IndexParams, ReindexParams, RemoveParams

_DETAIL_CAP = 200


def _cap_details(results: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    """Return full details for small runs, errors-only for large ones."""
    if len(results) <= _DETAIL_CAP:
        return results
    return [r for r in results if r.get("status") == "error"]


async def trove_index(path: str, *, new_vision_budget: bool = True) -> dict[str, Any]:
    """Index a specific file or directory.

    Skips unchanged files based on checksum comparison.
    For directories, scans recursively and indexes supported files.
    Each call gets a fresh vision budget unless new_vision_budget is False,
    as in the --index run, which spreads one budget over all its paths.
    """
    try:
        path = IndexParams(path=path).path
    except ValidationError as exc:
        raise InvalidInputError(str(exc)) from exc

    target = Path(path).resolve()
    if not target.exists():
        raise PathNotFoundError(path)

    if new_vision_budget:
        vision.reset_budget()
    results = await index_path_async(target, force=False)

    indexed = sum(1 for r in results if r["status"] == "indexed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errored = sum(1 for r in results if r["status"] == "error")
    deferred = sum(1 for r in results if r["status"] == "deferred")
    total_chunks = sum(int(r.get("chunk_count", 0)) for r in results)

    response: dict[str, Any] = {
        "path": str(target),
        "files_indexed": indexed,
        "files_skipped": skipped,
        "files_errored": errored,
        "files_deferred": deferred,
        "total_chunks": total_chunks,
        "details": _cap_details(results),
    }
    if len(results) > _DETAIL_CAP:
        response["details_note"] = (
            f"Large run ({len(results)} files) — details limited to errors only"
        )
    return response


async def trove_reindex(path: str | None = None, force: bool = False) -> dict[str, Any]:
    """Force re-index ignoring checksums.

    If no path given, reindexes all previously indexed files. That re-reads
    the whole corpus (and re-captions media whose cached caption came from
    another vision model), so it needs force=True.
    """
    try:
        params = ReindexParams(path=path, force=force)
    except ValidationError as exc:
        raise InvalidInputError(str(exc)) from exc
    path = params.path
    if not path and not params.force:
        raise ReindexAllNotConfirmedError

    vision.reset_budget()
    if path:
        target = Path(path).resolve()
        if not target.exists():
            raise PathNotFoundError(path)
        results = await index_path_async(target, force=True)
    else:
        all_files = db.query("SELECT path FROM files")
        results = []
        for row in all_files:
            file_path = Path(row["path"])
            if file_path.exists():
                results.extend(await index_path_async(file_path, force=True))
            else:
                existing = db.query_one("SELECT id FROM files WHERE path = ?", (row["path"],))
                if existing:
                    db.delete_file_data(existing["id"])
                    results.append(
                        {
                            "path": row["path"],
                            "status": "removed",
                            "reason": "file_missing",
                            "chunk_count": 0,
                        }
                    )

    indexed = sum(1 for r in results if r["status"] == "indexed")
    removed = sum(1 for r in results if r.get("status") == "removed")
    errored = sum(1 for r in results if r.get("status") == "error")
    total_chunks = sum(int(r.get("chunk_count", 0)) for r in results)

    response: dict[str, Any] = {
        "files_reindexed": indexed,
        "files_removed": removed,
        "files_errored": errored,
        "total_chunks": total_chunks,
        "details": _cap_details(results),
    }
    if len(results) > _DETAIL_CAP:
        response["details_note"] = (
            f"Large run ({len(results)} files) — details limited to errors only"
        )
    return response


async def trove_remove(path: str) -> dict[str, Any]:
    """Remove a file or directory from the index.

    Deletes all associated chunks and vector embeddings.
    """
    try:
        path = RemoveParams(path=path).path
    except ValidationError as exc:
        raise InvalidInputError(str(exc)) from exc

    target = Path(path).resolve()
    result = remove_path(target)
    return {
        "path": result["path"],
        "files_removed": result["removed"],
    }
