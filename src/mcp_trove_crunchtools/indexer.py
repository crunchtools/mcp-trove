"""File scanning, checksumming, chunking, and indexing orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import fnmatch
import gc
import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Any

import sqlite_vec

from . import database as db
from . import vision
from .config import get_config
from .embedder import embed_texts
from .errors import EmbeddingError, ExtractionError, IndexAbortedError, VisionDeferredError
from .extractor import MAX_FILE_SIZE, classify, detect_file_type, extract_text
from .filetypes import NAME_KIND, SAMPLED_KINDS, VISION_KINDS, is_media
from .manifest import Entry, load_manifest
from .media import fingerprint

logger = logging.getLogger(__name__)

HASH_BUFFER_SIZE = 65536
# A directory whose file count falls below this share of its last completed
# run is treated as a broken source (empty or half-mounted), not mass deletion.
MIN_FOUND_RATIO = 0.5
NAME_CHECKSUM_PREFIX = "name:"  # marks a stand-in checksum: the file was never read
MIN_FOUND_BASELINE = 100  # smaller trees are too noisy to judge
EMBED_BATCH_SIZE = 256  # chunks embedded per call, matching fastembed's default batch

# mtimes from the storage API and from the mount can differ below a second.
MTIME_TOLERANCE = 1.0
# A big tree whose manifest lacks more than this share of its indexed files
# aborts instead of dropping them: a listing cut short looks like deletion.
MAX_REMOVED_RATIO = 0.10
# Extractions run per type: PDFs are the memory-heavy ones, media wait on the
# network and the vision API, so they get the worker count.
# A name-only entry reads nothing, so it gets the most.
TYPE_CONCURRENCY = {
    "pdf": 1,
    "docx": 2,
    "doc": 2,
    "odf": 2,
    "rtf": 2,
    "xlsx": 2,
    "xls": 2,
    "table": 2,
    "pptx": 2,
    "text": 2,
    "markdown": 2,
    # A 4K HEVC decoder holds about 200 MB; with images and a PDF in flight,
    # a second one took the indexer past its 1.5 GB limit (RT #1542).
    "video": 1,
    "audio": 1,
    NAME_KIND: 4,
}
PROGRESS_EVERY = 25  # stored files between progress updates
TRIM_EVERY = 50  # stored files between returning freed memory to the OS


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
    """Split text into overlapping chunks, ending each at a natural break.

    A chunk ends at the last paragraph break, else line break, else sentence
    end, in its final 40%; only text with none is cut mid-word.
    """
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            end = _natural_break(text, start + int(chunk_size * 0.6), end)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(end - overlap, start + 1)

    return chunks


def _natural_break(text: str, earliest: int, end: int) -> int:
    """Index just after the best break in text[earliest:end], or end if none."""
    for separator in ("\n\n", "\n", ". "):
        found = text.rfind(separator, earliest, end)
        if found != -1:
            return found + len(separator)
    return end


def scan_directory(
    directory: Path,
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    """Scan a directory recursively for supported files."""
    patterns = exclude_patterns or get_config().exclude_patterns
    found: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(directory):
        # Prune hidden directories before descending: filtering them after
        # the walk still listed every file under .git and .hg, and over the
        # pCloud mount each listing is a network call (RT #1542).
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and not any(fnmatch.fnmatch(d, pat) for pat in patterns)
        ]
        for name in filenames:
            if name.startswith("."):
                continue
            if any(fnmatch.fnmatch(name, pat) for pat in patterns):
                continue
            path = Path(dirpath, name)
            if _is_own_database(path):
                continue
            try:
                if not path.is_file() or classify(path, path.stat().st_size) is None:
                    continue
            except OSError:
                continue
            found.append(path)

    return sorted(found)


def scan_source(directory: Path) -> tuple[list[Path], dict[Path, Entry] | None]:
    """List the supported files under directory, with their stats when known.

    From the directory's manifest when one is fresh (see manifest.py), which
    costs no network round trips; otherwise a walk of the tree, and None.
    """
    entries = load_manifest(directory)
    if entries is None:
        return scan_directory(directory), None
    patterns = get_config().exclude_patterns
    kept = {
        path: entry
        for path, entry in entries.items()
        if _wanted(path.relative_to(directory), entry.size, patterns) and not _is_own_database(path)
    }
    return sorted(kept), kept


def _wanted(relative: Path, size: int, patterns: list[str]) -> bool:
    """The walk's filters, applied to a manifest entry."""
    if any(part.startswith(".") for part in relative.parts):
        return False
    if _excluded(relative, patterns):
        return False
    return classify(relative, size) is not None


def _is_own_database(path: Path) -> bool:
    """Whether path is the index's own database or one of its sidecar files.

    A database kept inside an indexed tree would otherwise get a name-only
    entry, and its WAL a new one on every write.
    """
    opened = db.opened_path() or get_config().db_path
    return str(path).startswith(opened) and str(path)[len(opened) :] in {
        "",
        "-wal",
        "-shm",
        "-journal",
    }


def _excluded(relative: Path, patterns: list[str]) -> bool:
    """Whether a pattern matches the file's name or a directory above it."""
    return any(fnmatch.fnmatch(part, pat) for part in relative.parts for pat in patterns)


def _remove_excluded(directory: Path) -> list[dict[str, str | int]]:
    """Drop index entries under directory that the configuration now excludes.

    TROVE_EXCLUDE_PATTERNS, and files TROVE_INDEX_TYPES and TROVE_INDEX_OTHER
    no longer index in any form. Images and videos are never dropped here, so
    a vision backend left unset does not throw away paid-for captions. A file
    that moves between a category and its name-only entry is re-extracted by
    the partition, not removed.
    """
    patterns = get_config().exclude_patterns
    gone = []
    for row in _rows_under(directory):
        path = Path(row["path"])
        if _excluded(path.relative_to(directory), patterns) or (
            classify(path) is None and not is_media(path.suffix.lower())
        ):
            gone.append(row)
    for row in gone:
        db.delete_file_data(row["id"])
    return [
        {"path": row["path"], "status": "removed", "reason": "excluded", "chunk_count": 0}
        for row in gone
    ]


def _rows_under(directory: Path) -> list[dict[str, Any]]:
    """The id and path of every index entry under directory."""
    # A range on path, so the files index serves it: every path that starts
    # with the prefix sorts at or after it and before the prefix with its "/"
    # replaced by "0", the next character. The root's prefix is just "/".
    prefix = f"{str(directory).rstrip('/')}/"
    return db.query(
        "SELECT id, path FROM files WHERE path >= ? AND path < ?",
        (prefix, f"{prefix[:-1]}0"),
    )


def _check_unchanged(
    path: Path,
    stat: Entry | None = None,
) -> tuple[str, int, float, dict[str, Any] | None]:
    """Check if a file needs re-extraction (main thread, DB-safe).

    Uses mtime+size as a fast pre-filter: if both match the DB record,
    the file is assumed unchanged and the expensive SHA-256 is skipped.
    A file with no record needs no checksum to decide, so its checksum is
    returned empty and computed by the extract worker instead.
    stat, from a manifest, saves resolving and stat'ing the path over the
    mount. Returns (checksum, file_size, mtime, existing_record).
    """
    if stat is None:
        resolved = path.resolve()
        file_stat = resolved.stat()
        file_size, mtime = file_stat.st_size, file_stat.st_mtime
    else:
        resolved = path
        file_size, mtime = stat.size, stat.mtime

    existing = db.query_one(
        "SELECT id, checksum, file_type, file_size, mtime FROM files WHERE path = ?",
        (str(resolved),),
    )

    # New file: hashing here would read every new file serially before any
    # extraction starts, which over a network mount took hours (RT #1542).
    if existing is None:
        return "", file_size, mtime, None

    # Fast path: mtime+size match → skip checksum
    if (
        existing
        and existing["mtime"] is not None
        and abs(existing["mtime"] - mtime) < MTIME_TOLERANCE
        and existing["file_size"] == file_size
    ):
        return existing["checksum"], file_size, mtime, existing

    # Slow path: compute checksum.
    kind = classify(resolved, file_size)
    return content_checksum(resolved, kind, file_size, mtime), file_size, mtime, existing


def content_checksum(path: Path, kind: str | None, file_size: int, mtime: float) -> str:
    """The content key for a file, read the way its kind is read.

    A name-only entry is never read, so its stand-in comes from the stat;
    a video or audio file is fingerprinted from its ends, since hashing all
    of it would download it; anything else gets its SHA-256.
    """
    if kind == NAME_KIND:
        return name_checksum(file_size, mtime)
    if kind in SAMPLED_KINDS:
        return fingerprint(path)
    return compute_checksum(path)


def _current(existing: dict[str, Any], checksum: str, path: Path, file_size: int) -> bool:
    """Whether the indexed entry is this content, read the way it is read now.

    A file whose category was switched on since it got its name-only entry,
    or switched off since it was read, is re-extracted.
    """
    return bool(
        existing["checksum"] == checksum and existing["file_type"] == classify(path, file_size)
    )


def name_checksum(file_size: int, mtime: float) -> str:
    """The checksum recorded for a name-only entry, whose bytes are never read."""
    return f"{NAME_CHECKSUM_PREFIX}{file_size}:{int(mtime)}"


def _extract_one(
    path: Path,
    checksum: str,
    file_size: int,
    existing_id: int | None,
    mtime: float = 0.0,
) -> dict[str, Any]:
    """Extract text from a single file (phase 1: IO + vision API).

    Called from worker threads — no DB access here.
    """
    resolved = path.resolve()
    # Once the run's vision budget is spent, a new image would be downloaded
    # just to be deferred; defer it before reading a byte. That includes one
    # whose caption is cached: finding out needs its checksum, which is the
    # download. It is indexed on the next run.
    if detect_file_type(resolved) in VISION_KINDS and vision.budget_spent():
        raise VisionDeferredError(str(resolved), get_config().vision_max_calls)
    # The scan's size can come from a manifest hours old; the limit is
    # enforced on the file as it is now, before it is read.
    stat = resolved.stat()
    file_size = stat.st_size
    file_type = classify(resolved, file_size)
    if file_type is None:
        raise ExtractionError(
            str(resolved), f"{file_size} bytes is over the {MAX_FILE_SIZE} byte limit"
        )
    if file_type == NAME_KIND or not checksum or checksum.startswith(NAME_CHECKSUM_PREFIX):
        # A file leaving its name-only entry has never been hashed.
        checksum = content_checksum(resolved, file_type, file_size, stat.st_mtime)
    text = extract_text(resolved, checksum, file_type)
    # Text can hold lone UTF-16 surrogates: striprtf turns an RTF emoji's two
    # \u escapes into a pair of them, and PDFs carry strays. The tokenizer
    # rejects a whole batch over one, and SQLite will not store them. Pairs
    # are rejoined into their character; strays become U+FFFD.
    text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    config = get_config()
    if len(text) > config.max_text_chars:
        logger.warning(
            "Truncating %s from %d to %d characters (TROVE_MAX_TEXT_CHARS)",
            resolved,
            len(text),
            config.max_text_chars,
        )
        text = text[: config.max_text_chars]
    chunks = chunk_text(text, config.chunk_size, config.chunk_overlap)

    return {
        "path": str(resolved),
        "status": "extracted",
        "file_type": file_type,
        "checksum": checksum,
        "file_size": file_size,
        "mtime": mtime,
        "chunks": chunks,
        "existing_id": existing_id,
    }


def _store_one(
    extraction: dict[str, Any], vectors: list[bytes] | None = None
) -> dict[str, str | int]:
    """Store extracted content in the database (phase 2: embed + SQLite).

    Takes the output of _extract_one and embeds/stores it, or stores the
    vectors already made for it by _embed_chunks. Must run sequentially
    (SQLite is single-writer) and on the thread that opened the database.
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
    mtime = float(extraction.get("mtime") or 0.0)
    chunks: list[str] = extraction["chunks"]
    existing_id = extraction.get("existing_id")

    # Embed before writing anything: an embedding failure then leaves the
    # index exactly as it was, including the file's old entry.
    if vectors is None:
        vectors = _embed_chunks(chunks)

    # One transaction per file: a process killed part way (OOM, reboot) used
    # to leave a file row matching its checksum but missing chunks, which was
    # then skipped as unchanged on every later run.
    with db.transaction():
        if existing_id is not None:
            db.delete_file_data(int(existing_id))
        file_id = db.insert_file(path_str, checksum, file_type, file_size, mtime)
        for index, (chunk_content, vector) in enumerate(zip(chunks, vectors, strict=True)):
            chunk_id = db.insert_chunk(file_id, index, chunk_content)
            db.insert_vector(chunk_id, vector)
        db.update_file(file_id, checksum, file_size, len(chunks), mtime)
        db.resolve_errors(path_str)

    return {"path": path_str, "status": "indexed", "chunk_count": len(chunks)}


_EMBED_LOCK = threading.Lock()


def _embed_chunks(chunks: list[str]) -> list[bytes]:
    """Embed chunks in sub-batches, as float32 bytes.

    Sub-batches keep a large file's vectors from ever being held as Python
    float lists (RT #1542); each is kept as bytes, 1.5 KB a chunk. Touches
    no database, so it can run on a worker thread. One file embeds at a time
    per process, even when index calls overlap in the MCP server, so the
    model's threads and memory stay bounded.
    """
    vectors: list[bytes] = []
    with _EMBED_LOCK:
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            vectors.extend(sqlite_vec.serialize_float32(e) for e in embed_texts(batch))
    return vectors


def index_file(path: Path, force: bool = False) -> dict[str, str | int]:
    """Index a single file: extract text, chunk, embed, store."""
    resolved = path.resolve()
    run_id = db.start_run(str(resolved), 1)
    try:
        checksum, file_size, mtime, existing = _check_unchanged(resolved)

        if existing is not None and _current(existing, checksum, resolved, file_size) and not force:
            if existing["mtime"] is None:
                db.update_file_mtime(existing["id"], mtime)
            db.finish_run(
                run_id,
                files_indexed=0,
                files_skipped=1,
                files_errored=0,
                total_chunks=0,
            )
            file_status: dict[str, str | int] = {
                "path": str(resolved),
                "status": "skipped",
                "reason": "unchanged",
                "chunk_count": 0,
            }
        else:
            existing_id = existing["id"] if existing else None
            extraction = _extract_one(
                resolved,
                checksum,
                file_size,
                existing_id,
                mtime,
            )
            file_status = _store_one(extraction)
            db.finish_run(
                run_id,
                files_indexed=1,
                files_skipped=0,
                files_errored=0,
                total_chunks=int(file_status.get("chunk_count", 0)),
            )
    except Exception as exc:
        db.log_run_error(run_id, str(exc))
        raise
    return file_status


def _partition_unchanged(
    files: list[Path],
    force: bool,
    stats: dict[Path, Entry] | None = None,
    root: Path | None = None,
) -> tuple[
    list[dict[str, str | int]],
    list[tuple[Path, str, int, int | None, float]],
]:
    """Split files into skipped (unchanged) and needing extraction."""
    skipped: list[dict[str, str | int]] = []
    to_extract: list[tuple[Path, str, int, int | None, float]] = []
    known = set() if force else db.known_failures()
    for fp in files:
        stat = stats.get(fp) if stats else None
        checksum, file_size, mtime, existing = _check_unchanged(fp, stat)
        shown = str(fp) if stat else str(fp.resolve())
        if existing is not None and _current(existing, checksum, fp, file_size) and not force:
            # Backfill mtime if DB has NULL so fast-skip works next run
            if existing["mtime"] is None:
                db.update_file_mtime(existing["id"], mtime)
            skipped.append(
                {
                    "path": shown,
                    "status": "skipped",
                    "reason": "unchanged",
                    "chunk_count": 0,
                }
            )
        elif (str(fp), file_size, mtime) in known:
            # This exact version keeps failing; retrying it every night costs
            # a download and, for media, a vision call. It is retried when
            # the file changes or on trove_reindex.
            skipped.append(
                {
                    "path": shown,
                    "status": "skipped",
                    "reason": "known_failure",
                    "chunk_count": 0,
                }
            )
        elif stat and root and not fp.resolve().is_relative_to(root):
            # A manifest path is not resolved during the scan (that is the
            # round trip it saves), so one under a symlinked directory is
            # caught here, before anything is read, as the walk would have.
            logger.warning("Skipping %s: it resolves outside %s", fp, root)
            skipped.append(
                {"path": shown, "status": "skipped", "reason": "outside_root", "chunk_count": 0}
            )
        else:
            existing_id = existing["id"] if existing else None
            to_extract.append((fp, checksum, file_size, existing_id, mtime))
    return skipped, to_extract


async def _extract_and_store(
    to_extract: list[tuple[Path, str, int, int | None, float]],
    results: list[dict[str, str | int]],
    run_id: int | None = None,
) -> None:
    """Extract files concurrently and store each as soon as it is ready.

    A bounded window of extractions is in flight; whichever finishes first is
    stored, and the next file starts. Lockstep batches waited on their slowest
    file. Concurrency is capped per file type (TYPE_CONCURRENCY), and the
    window bounds how much extracted text is held at once.
    """
    config = get_config()
    media_workers = max(1, config.index_workers)
    limits = {file_type: asyncio.Semaphore(count) for file_type, count in TYPE_CONCURRENCY.items()}
    media = asyncio.Semaphore(media_workers)
    window = media_workers + sum(TYPE_CONCURRENCY.values())
    # Safety-net timeout: 3x the per-call vision timeout to allow for
    # EXIF extraction + file read + API call, but prevent infinite hangs.
    extract_timeout = config.vision_timeout * 3
    hung = _HungReads(config.max_extract_timeouts)
    stored = 0

    async def extract_bounded(item: tuple[Path, str, int, int | None, float]) -> dict[str, Any]:
        async with limits.get(detect_file_type(item[0]), media), asyncio.timeout(extract_timeout):
            return await asyncio.to_thread(_extract_one, *item)

    logger.info(
        "Extracting %d files, up to %d at once (%d media)",
        len(to_extract),
        window,
        media_workers,
    )
    queue = iter(to_extract)
    in_flight: dict[asyncio.Task[dict[str, Any]], tuple[Path, str, int, int | None, float]] = {}
    try:
        while True:
            while len(in_flight) < window and (item := next(queue, None)) is not None:
                in_flight[asyncio.create_task(extract_bounded(item))] = item
            if not in_flight:
                break
            done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                results.append(await _settle(task, in_flight.pop(task), run_id, hung))
                stored += 1
                if stored % PROGRESS_EVERY == 0:
                    _update_progress(run_id, results)
                if stored % TRIM_EVERY == 0:
                    _release_memory()
    finally:
        for task in in_flight:
            task.cancel()
    _update_progress(run_id, results)


class _HungReads:
    """Counts timed-out extractions and aborts the run past a limit.

    A timed-out extraction's thread cannot be cancelled and keeps its read
    open; past a few of them the source is hung (a wedged FUSE mount), and
    carrying on would pile up stuck threads (RT #1542).
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.count = 0

    def saw_timeout(self, path: Path) -> None:
        self.count += 1
        if self.count >= self.limit:
            raise IndexAbortedError(
                str(path.parent), f"{self.count} extractions timed out; the source looks hung"
            )


async def _settle(
    task: asyncio.Task[dict[str, Any]],
    item: tuple[Path, str, int, int | None, float],
    run_id: int | None,
    hung: _HungReads,
) -> dict[str, str | int]:
    """Store a finished extraction, or record why it failed.

    Embedding a big file takes minutes of CPU, so it runs on a thread. On
    the event loop it stalled the loop, and the timeouts of extractions
    still in flight fired on reads that had long finished, five of which
    aborted a run as a hung source (RT #1542).
    """
    error = task.exception()
    if error is None:
        extraction = task.result()
        vectors = None
        if extraction["status"] != "skipped":
            # One file the model cannot embed fails that file, not its tree.
            try:
                vectors = await asyncio.to_thread(_embed_chunks, extraction["chunks"])
            except EmbeddingError as exc:
                return _record_failure(item, exc, run_id)
        return _store_one(extraction, vectors)
    result = _record_failure(item, error, run_id)
    if isinstance(error, TimeoutError):
        hung.saw_timeout(item[0])
    return result


def _record_failure(
    item: tuple[Path, str, int, int | None, float],
    error: BaseException,
    run_id: int | None,
) -> dict[str, str | int]:
    """The result for a file that could not be extracted, logged unless deferred."""
    path = str(item[0])
    if isinstance(error, VisionDeferredError):
        return {
            "path": path,
            "status": "deferred",
            "reason": "vision budget spent",
            "chunk_count": 0,
        }
    message = str(error)
    logger.warning("Failed to extract %s: %s", path, message)
    db.insert_error(run_id, path, message, db.classify_error(message), (item[2], item[4]))
    return {"path": path, "status": "error", "reason": message, "chunk_count": 0}


def _update_progress(run_id: int | None, results: list[dict[str, str | int]]) -> None:
    if run_id is None:
        return
    db.update_run_progress(
        run_id,
        files_indexed=sum(1 for r in results if r["status"] == "indexed"),
        files_skipped=sum(1 for r in results if r["status"] == "skipped"),
        files_errored=sum(1 for r in results if r["status"] == "error"),
        total_chunks=sum(int(r.get("chunk_count", 0)) for r in results),
    )


def _release_memory() -> None:
    """Return freed heap to the OS between files.

    CPython's pymalloc keeps freed pages; gc.collect() breaks cycles, then
    malloc_trim returns free heap pages to the OS.
    """
    gc.collect()
    with contextlib.suppress(OSError):
        ctypes.CDLL("libc.so.6").malloc_trim(0)


def _remove_missing(
    directory: Path, present: set[Path], guarded: bool = True
) -> list[dict[str, str | int]]:
    """Drop index entries under directory that its manifest no longer lists.

    Only with a manifest: a walk that fails part way cannot be told from
    deletion. A tree of MIN_FOUND_BASELINE or more indexed files that would
    lose over MAX_REMOVED_RATIO of them aborts instead, as a listing cut short,
    unless guarded is false (an inbox, which empties out by design).
    """
    rows = _rows_under(directory)
    gone = [row for row in rows if Path(row["path"]) not in present]
    if guarded and len(rows) >= MIN_FOUND_BASELINE and len(gone) > len(rows) * MAX_REMOVED_RATIO:
        msg = (
            f"the manifest lacks {len(gone)} of {len(rows)} indexed files, over "
            f"{MAX_REMOVED_RATIO:.0%}; it may be incomplete. Remove them with "
            "trove_remove if they really were deleted."
        )
        raise IndexAbortedError(str(directory), msg)
    for row in gone:
        db.delete_file_data(row["id"])
    return [
        {"path": row["path"], "status": "removed", "reason": "deleted", "chunk_count": 0}
        for row in gone
    ]


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

    files, stats = scan_source(resolved)
    inbox = _is_inbox(resolved)
    # Before the shrink gate, which judges the scan against what is indexed.
    removed = _remove_excluded(resolved)
    if batch_limit is not None:
        files = files[:batch_limit]
    elif not force and not inbox:
        _check_not_shrunk(resolved, len(files))

    if not files and stats is None:
        return removed

    run_id = db.start_run(str(resolved), len(files))
    try:
        results, to_extract = _partition_unchanged(files, force, stats, resolved)
        results.extend(removed)
        # Deletions are only trusted from a manifest covering the whole tree.
        if stats is not None and batch_limit is None:
            results.extend(_remove_missing(resolved, set(files), guarded=not inbox))
        # Record skipped count immediately so status queries see progress
        _update_progress(run_id, results)
        await _extract_and_store(to_extract, results, run_id)
        _finish(run_id, results)
    except IndexAbortedError as exc:
        db.abort_run(run_id, str(exc))
        raise
    except Exception as exc:
        db.log_run_error(run_id, str(exc))
        raise
    return results


def _finish(run_id: int, results: list[dict[str, str | int]]) -> None:
    db.finish_run(
        run_id,
        files_indexed=sum(1 for r in results if r["status"] == "indexed"),
        files_skipped=sum(1 for r in results if r["status"] == "skipped"),
        files_errored=sum(1 for r in results if r["status"] == "error"),
        total_chunks=sum(int(r.get("chunk_count", 0)) for r in results),
    )


def _is_inbox(directory: Path) -> bool:
    """Whether directory is one of TROVE_INBOX_PATHS."""
    return any(Path(inbox).resolve() == directory for inbox in get_config().inbox_paths)


def _check_not_shrunk(directory: Path, files_found: int) -> None:
    """Abort when a directory suddenly holds far fewer files than are indexed.

    An unmounted or half-listed source looks exactly like a tree whose files
    were deleted. Refusing keeps a broken mount from passing as a quiet,
    successful run; trove_reindex on the path skips this check. The baseline
    is the index, not the last run's count, so files newly excluded by the
    configuration (removed just before) do not look like a shrink.
    """
    indexed = len(_rows_under(directory))
    if indexed < MIN_FOUND_BASELINE:
        return
    if files_found >= indexed * MIN_FOUND_RATIO:
        return
    reason = (
        f"found {files_found} files, under {MIN_FOUND_RATIO:.0%} of the "
        f"{indexed} indexed there; the source may be unmounted. Run "
        "trove_reindex on the path if the files really were removed."
    )
    db.abort_run(db.start_run(str(directory), files_found), reason)
    raise IndexAbortedError(str(directory), reason)


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
        existing = db.query_one("SELECT id FROM files WHERE path = ?", (resolved_str,))
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
