"""File scanning, checksumming, chunking, and indexing orchestration."""

from __future__ import annotations

import fnmatch
import hashlib
from typing import TYPE_CHECKING

from . import database as db

if TYPE_CHECKING:
    from pathlib import Path
from .config import get_config
from .embedder import embed_texts
from .extractor import detect_file_type, extract_text, is_supported

HASH_BUFFER_SIZE = 65536
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


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
    """Split text into overlapping chunks.

    Returns a list of text chunks with the specified size and overlap.
    """
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
    """Scan a directory recursively for supported files.

    Respects exclude patterns and skips hidden directories.
    """
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


def index_file(path: Path, force: bool = False) -> dict[str, str | int]:
    """Index a single file: extract text, chunk, embed, store.

    If force=False, skips files whose checksum hasn't changed.
    Returns a summary dict with path, status, and chunk_count.
    """
    resolved = path.resolve()
    checksum = compute_checksum(resolved)
    file_size = resolved.stat().st_size

    existing = db.query_one(
        "SELECT id, checksum FROM files WHERE path = ?", (str(resolved),)
    )

    if existing and existing["checksum"] == checksum and not force:
        return {
            "path": str(resolved),
            "status": "skipped",
            "reason": "unchanged",
            "chunk_count": 0,
        }

    text = extract_text(resolved)
    file_type = detect_file_type(resolved)
    config = get_config()
    chunks = chunk_text(text, config.chunk_size, config.chunk_overlap)

    if existing:
        db.delete_file_data(existing["id"])

    file_id = db.insert_file(str(resolved), checksum, file_type, file_size)

    if not chunks:
        db.update_file(file_id, checksum, file_size, 0)
        return {
            "path": str(resolved),
            "status": "indexed",
            "chunk_count": 0,
        }

    embeddings = embed_texts(chunks)

    for idx, (chunk_content, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
        chunk_id = db.insert_chunk(file_id, idx, chunk_content)
        db.insert_vector(chunk_id, embedding)

    db.update_file(file_id, checksum, file_size, len(chunks))

    return {
        "path": str(resolved),
        "status": "indexed",
        "chunk_count": len(chunks),
    }


def index_path(
    path: Path,
    force: bool = False,
    batch_limit: int | None = None,
) -> list[dict[str, str | int]]:
    """Index a file or directory.

    For directories, scans recursively and indexes supported files.
    Respects batch_limit to cap the number of files processed.
    """
    resolved = path.resolve()
    results: list[dict[str, str | int]] = []

    if resolved.is_file():
        results.append(index_file(resolved, force=force))
        return results

    if resolved.is_dir():
        files = scan_directory(resolved)
        if batch_limit is not None:
            files = files[:batch_limit]
        for file_path in files:
            results.append(index_file(file_path, force=force))
        return results

    return results


def remove_path(path: Path) -> dict[str, str | int]:
    """Remove a file or directory from the index.

    For directories, removes all files under that path prefix.
    """
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
