"""File scanning, checksumming, chunking, and indexing orchestration."""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import logging
from typing import TYPE_CHECKING, Any

from . import database as db

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
from .config import get_config
from .embedder import embed_texts
from .extractor import detect_file_type, extract_text, is_supported

logger = logging.getLogger(__name__)

HASH_BUFFER_SIZE = 65536
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

_T = Any  # generic element type for _batched


def _batched(iterable: list[_T], n: int) -> Iterator[list[_T]]:
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def compute_checksum(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(HASH_BUFFER_SIZE)
            if not block:
                break
            sha256.update(block)
    return sha256.hexdigest()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap

    return chunks


def scan_directory(
    directory: Path,
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    """Scan a directory recursively for supported files."""
    patterns = exclude_patterns or get_config().exclude_patterns
    found: list[Path] = []

    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        if path.stat().st_size > MAX_FILE_SIZE:
            continue
        if any(fnmatch.fnmatch(path.name, pat) for pat in patterns):
            continue
        if not is_supported(path):
            continue
        found.append(path)

    return sorted(found)


def _check_unchanged(path: Path) -> tuple[str, int, dict[str, Any] | None]:
    """Check if a file needs re-extraction (main thread, DB-safe)."""
    resolved = path.resolve()
    checksum = compute_checksum(resolved)
    file_size = resolved.stat().st_size
    existing = db.query_one(
        "SELECT id, checksum FROM files WHERE path = ?", (str(resolved),)
    )
    return checksum, file_size, existing


def _extract_one(
    path: Path, checksum: str, file_size: int, existing_id: int | None
) -> dict[str, Any]:
    """Extract text from a single file (phase 1: IO + vision API).

    Called from worker threads — no DB access here.
    """
    resolved = path.resolve()
    text = extract_text(resolved)
    file_type = detect_file_type(resolved)
    config = get_config()
    chunks = chunk_text(text, config.chunk_size, config.chunk_overlap)

    return {
        "path": str(resolved),
        "status": "extracted",
        "file_type": file_type,
        "checksum": checksum,
        "file_size": file_size,
        "chunks": chunks,
        "existing_id": existing_id,
    }


def _store_one(extraction: dict[str, Any]) -> dict[str, str | int]:
    """Store extracted content in the database (phase 2: embed + SQLite).

    Takes the output of _extract_one and embeds/stores it.
    Must run sequentially (SQLite is single-writer).
    """
    if extraction["status"] == "skipped":
        return {
            "path": extraction["path"],
            "status": "skipped",
            "reason": extraction.get("reason", "unchanged"),
            "chunk_count": 0,
        }

    path_str = str(extraction["path"])
    checksum = str(extraction["checksum"])
    file_type = str(extraction["file_type"])
    file_size = int(extraction["file_size"])
    chunks: list[str] = extraction["chunks"]
    existing_id = extraction.get("existing_id")

    if existing_id is not None:
        db.delete_file_data(int(existing_id))

    file_id = db.insert_file(path_str, checksum, file_type, file_size)

    if not chunks:
        db.update_file(file_id, checksum, file_size, 0)
        return {"path": path_str, "status": "indexed", "chunk_count": 0}

    embeddings = embed_texts(chunks)

    for idx, (chunk_content, embedding) in enumerate(
        zip(chunks, embeddings, strict=True)
    ):
        chunk_id = db.insert_chunk(file_id, idx, chunk_content)
        db.insert_vector(chunk_id, embedding)

    db.update_file(file_id, checksum, file_size, len(chunks))

    return {"path": path_str, "status": "indexed", "chunk_count": len(chunks)}


def index_file(path: Path, force: bool = False) -> dict[str, str | int]:
    """Index a single file: extract text, chunk, embed, store."""
    resolved = path.resolve()
    run_id = db.start_run(str(resolved), 1)
    try:
        checksum, file_size, existing = _check_unchanged(resolved)

        if existing and existing["checksum"] == checksum and not force:
            db.finish_run(
                run_id,
                files_indexed=0, files_skipped=1,
                files_errored=0, total_chunks=0,
            )
            result: dict[str, str | int] = {
                "path": str(resolved),
                "status": "skipped",
                "reason": "unchanged",
                "chunk_count": 0,
            }
        else:
            existing_id = existing["id"] if existing else None
            extraction = _extract_one(resolved, checksum, file_size, existing_id)
            result = _store_one(extraction)
            db.finish_run(
                run_id,
                files_indexed=1, files_skipped=0,
                files_errored=0,
                total_chunks=int(result.get("chunk_count", 0)),
            )
    except Exception as exc:
        db.log_run_error(run_id, str(exc))
        raise
    return result


def _partition_unchanged(
    files: list[Path], force: bool,
) -> tuple[list[dict[str, str | int]], list[tuple[Path, str, int, int | None]]]:
    """Split files into skipped (unchanged) and needing extraction."""
    skipped: list[dict[str, str | int]] = []
    to_extract: list[tuple[Path, str, int, int | None]] = []
    for fp in files:
        checksum, file_size, existing = _check_unchanged(fp)
        if existing and existing["checksum"] == checksum and not force:
            skipped.append({
                "path": str(fp.resolve()),
                "status": "skipped",
                "reason": "unchanged",
                "chunk_count": 0,
            })
        else:
            existing_id = existing["id"] if existing else None
            to_extract.append((fp, checksum, file_size, existing_id))
    return skipped, to_extract


async def _extract_and_store_batched(
    to_extract: list[tuple[Path, str, int, int | None]],
    results: list[dict[str, str | int]],
) -> None:
    """Extract files in batches, storing each batch before starting the next.

    This bounds memory usage: only one batch of extracted text lives in memory
    at a time, instead of accumulating all extractions via a single gather.
    """
    config = get_config()
    workers = config.index_workers
    batch_size = config.index_batch
    semaphore = asyncio.Semaphore(workers)

    async def extract_bounded(
        file_path: Path, cs: str, fs: int, eid: int | None,
    ) -> dict[str, Any]:
        async with semaphore:
            return await asyncio.to_thread(
                _extract_one, file_path, cs, fs, eid,
            )

    logger.info(
        "Extracting %d files with %d workers in batches of %d",
        len(to_extract), workers, batch_size,
    )

    for batch_num, chunk in enumerate(_batched(to_extract, batch_size)):
        logger.info(
            "Batch %d: extracting %d files", batch_num + 1, len(chunk),
        )
        extractions = await asyncio.gather(
            *(
                extract_bounded(fp, cs, fs, eid)
                for fp, cs, fs, eid in chunk
            ),
            return_exceptions=True,
        )

        # Store this batch immediately, then free extraction data
        for idx, raw in enumerate(extractions):
            if isinstance(raw, BaseException):
                logger.warning(
                    "Failed to extract %s: %s", chunk[idx][0], raw,
                )
                results.append({
                    "path": str(chunk[idx][0]),
                    "status": "error",
                    "reason": str(raw),
                    "chunk_count": 0,
                })
                continue
            results.append(_store_one(raw))
        del extractions


async def index_path_async(
    path: Path,
    force: bool = False,
    batch_limit: int | None = None,
) -> list[dict[str, str | int]]:
    """Index a file or directory with concurrent vision extraction.

    Phase 1 (concurrent): checksum, extract text, call vision API
    Phase 2 (sequential): embed chunks, store in SQLite
    """
    resolved = path.resolve()

    if resolved.is_file():
        return [index_file(resolved, force=force)]

    if not resolved.is_dir():
        return []

    files = scan_directory(resolved)
    if batch_limit is not None:
        files = files[:batch_limit]

    if not files:
        return []

    run_id = db.start_run(str(resolved), len(files))
    try:
        results, to_extract = _partition_unchanged(files, force)

        if to_extract:
            await _extract_and_store_batched(to_extract, results)

        indexed = sum(1 for r in results if r["status"] == "indexed")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        errored = sum(1 for r in results if r["status"] == "error")
        total_chunks = sum(int(r.get("chunk_count", 0)) for r in results)
        db.finish_run(
            run_id,
            files_indexed=indexed, files_skipped=skipped,
            files_errored=errored, total_chunks=total_chunks,
        )
    except Exception as exc:
        db.log_run_error(run_id, str(exc))
        raise
    return results


def index_path(
    path: Path,
    force: bool = False,
    batch_limit: int | None = None,
) -> list[dict[str, str | int]]:
    """Synchronous wrapper for index_path_async."""
    resolved = path.resolve()

    if resolved.is_file():
        return [index_file(resolved, force=force)]

    if not resolved.is_dir():
        return []

    files = scan_directory(resolved)
    if batch_limit is not None:
        files = files[:batch_limit]
    return [index_file(fp, force=force) for fp in files]


def remove_path(path: Path) -> dict[str, str | int]:
    """Remove a file or directory from the index."""
    resolved = path.resolve()
    resolved_str = str(resolved)

    if resolved.is_file() or not resolved.exists():
        existing = db.query_one(
            "SELECT id FROM files WHERE path = ?", (resolved_str,)
        )
        if existing:
            db.delete_file_data(existing["id"])
            return {"path": resolved_str, "removed": 1}
        return {"path": resolved_str, "removed": 0}

    files = db.query(
        "SELECT id, path FROM files WHERE path LIKE ?",
        (resolved_str + "/%",),
    )
    for file_row in files:
        db.delete_file_data(file_row["id"])
    return {"path": resolved_str, "removed": len(files)}
