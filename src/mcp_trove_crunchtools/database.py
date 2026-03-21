"""SQLite database management for mcp-trove-crunchtools.

Uses sqlite-vec for vector search and FTS5 for keyword search.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import sqlite_vec

from .config import get_config

_db: sqlite3.Connection | None = None

VECTOR_DIMS = 384

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    checksum TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    UNIQUE(file_id, chunk_index)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content, content=chunks, content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content)
    VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO chunks_fts(rowid, content)
    VALUES (new.id, new.content);
END;

CREATE TABLE IF NOT EXISTS index_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    files_found INTEGER NOT NULL DEFAULT 0,
    files_indexed INTEGER NOT NULL DEFAULT 0,
    files_skipped INTEGER NOT NULL DEFAULT 0,
    files_errored INTEGER NOT NULL DEFAULT 0,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS index_errors (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES index_runs(id),
    path TEXT NOT NULL,
    error_message TEXT NOT NULL,
    error_type TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT,
    resolved INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_errors_path ON index_errors(path);
CREATE INDEX IF NOT EXISTS idx_errors_resolved ON index_errors(resolved);
"""

VEC_TABLE_SQL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding float[{VECTOR_DIMS}]
);
"""


def get_db(db_path: str | None = None) -> sqlite3.Connection:
    """Get or create the singleton database connection."""
    global _db
    if _db is None:
        path = db_path or get_config().db_path
        if path != ":memory:":
            get_config().ensure_db_dir()
        _db = sqlite3.connect(path)
        _db.row_factory = sqlite3.Row
        _db.execute("PRAGMA journal_mode=WAL")
        _db.execute("PRAGMA foreign_keys=ON")
        _db.enable_load_extension(True)
        sqlite_vec.load(_db)
        _db.enable_load_extension(False)
        _db.executescript(SCHEMA)
        _db.execute(VEC_TABLE_SQL)
        _db.commit()
    return _db


def query(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Execute a SELECT query and return results as dicts."""
    db = get_db()
    cursor = db.execute(sql, params)
    return [dict(row) for row in cursor.fetchall()]


def query_one(sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    """Execute a SELECT query and return a single result or None."""
    db = get_db()
    cursor = db.execute(sql, params)
    row = cursor.fetchone()
    return dict(row) if row else None


def execute(sql: str, params: tuple[Any, ...] = ()) -> int:
    """Execute an INSERT/UPDATE/DELETE and return lastrowid."""
    db = get_db()
    cursor = db.execute(sql, params)
    db.commit()
    return cursor.lastrowid or 0


def execute_many(sql: str, params_list: list[tuple[Any, ...]]) -> None:
    """Execute a statement with multiple parameter sets."""
    db = get_db()
    db.executemany(sql, params_list)
    db.commit()


def insert_file(
    path: str,
    checksum: str,
    file_type: str,
    file_size: int,
) -> int:
    """Insert a file record and return its ID."""
    return execute(
        "INSERT INTO files (path, checksum, file_type, file_size) "
        "VALUES (?, ?, ?, ?)",
        (path, checksum, file_type, file_size),
    )


def update_file(
    file_id: int,
    checksum: str,
    file_size: int,
    chunk_count: int,
) -> None:
    """Update a file record after re-indexing."""
    execute(
        "UPDATE files SET checksum = ?, file_size = ?, chunk_count = ?, "
        "indexed_at = datetime('now') WHERE id = ?",
        (checksum, file_size, chunk_count, file_id),
    )


def delete_file_data(file_id: int) -> None:
    """Delete all chunks and vectors for a file, then the file record."""
    db = get_db()
    chunk_ids = [
        row["id"] for row in query(
            "SELECT id FROM chunks WHERE file_id = ?", (file_id,)
        )
    ]
    for chunk_id in chunk_ids:
        db.execute("DELETE FROM chunks_vec WHERE chunk_id = ?", (chunk_id,))
    db.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
    db.execute("DELETE FROM files WHERE id = ?", (file_id,))
    db.commit()


def insert_chunk(
    file_id: int,
    chunk_index: int,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Insert a text chunk and return its ID."""
    meta_json = json.dumps(metadata) if metadata else None
    return execute(
        "INSERT INTO chunks (file_id, chunk_index, content, metadata) "
        "VALUES (?, ?, ?, ?)",
        (file_id, chunk_index, content, meta_json),
    )


def insert_vector(chunk_id: int, embedding: list[float]) -> None:
    """Insert a vector embedding for a chunk."""
    db = get_db()
    db.execute(
        "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, sqlite_vec.serialize_float32(embedding)),
    )
    db.commit()


def search_vectors(
    query_embedding: list[float], limit: int, path_filter: str | None = None
) -> list[dict[str, Any]]:
    """Search for similar chunks using vector similarity."""
    serialized = sqlite_vec.serialize_float32(query_embedding)
    if path_filter:
        return query(
            """
            SELECT
                cv.chunk_id,
                cv.distance,
                c.content,
                c.metadata,
                c.chunk_index,
                f.path,
                f.file_type
            FROM chunks_vec cv
            JOIN chunks c ON c.id = cv.chunk_id
            JOIN files f ON f.id = c.file_id
            WHERE cv.embedding MATCH ?
              AND k = ?
              AND f.path LIKE ?
            ORDER BY cv.distance
            """,
            (serialized, limit * 3, path_filter + "%"),
        )
    return query(
        """
        SELECT
            cv.chunk_id,
            cv.distance,
            c.content,
            c.metadata,
            c.chunk_index,
            f.path,
            f.file_type
        FROM chunks_vec cv
        JOIN chunks c ON c.id = cv.chunk_id
        JOIN files f ON f.id = c.file_id
        WHERE cv.embedding MATCH ?
          AND k = ?
        ORDER BY cv.distance
        """,
        (serialized, limit * 3),
    )


def search_fts(query_text: str, limit: int) -> list[dict[str, Any]]:
    """Search chunks using FTS5 full-text search."""
    return query(
        """
        SELECT
            c.id AS chunk_id,
            rank AS score,
            c.content,
            c.metadata,
            c.chunk_index,
            f.path,
            f.file_type
        FROM chunks_fts fts
        JOIN chunks c ON c.id = fts.rowid
        JOIN files f ON f.id = c.file_id
        WHERE chunks_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (query_text, limit),
    )


def get_file_chunks_avg_embedding(file_id: int) -> list[dict[str, Any]]:
    """Get all vector embeddings for chunks of a file."""
    return query(
        """
        SELECT cv.chunk_id, cv.embedding
        FROM chunks_vec cv
        JOIN chunks c ON c.id = cv.chunk_id
        WHERE c.file_id = ?
        """,
        (file_id,),
    )


def start_run(path: str, files_found: int) -> int:
    """Record the start of an indexing run. Returns the run ID."""
    return execute(
        "INSERT INTO index_runs (path, files_found) VALUES (?, ?)",
        (path, files_found),
    )


def finish_run(
    run_id: int,
    *,
    files_indexed: int,
    files_skipped: int,
    files_errored: int,
    total_chunks: int,
) -> None:
    """Mark an indexing run as completed with final counts."""
    execute(
        "UPDATE index_runs SET finished_at = datetime('now'), status = 'completed', "
        "files_indexed = ?, files_skipped = ?, files_errored = ?, total_chunks = ? "
        "WHERE id = ?",
        (files_indexed, files_skipped, files_errored, total_chunks, run_id),
    )


def log_run_error(run_id: int, error_message: str) -> None:
    """Mark an indexing run as failed with an error message."""
    execute(
        "UPDATE index_runs SET finished_at = datetime('now'), status = 'failed', "
        "error_message = ? WHERE id = ?",
        (error_message, run_id),
    )


# --- Per-file error tracking ---

_TRANSIENT_PATTERNS = (
    "connection reset",
    "dns",
    "503",
    "timeout",
    "temporary failure",
    "name resolution",
    "broken pipe",
    "connection refused",
    "network unreachable",
)


def classify_error(error_message: str) -> str:
    """Classify an error as 'transient' or 'permanent'."""
    lower = error_message.lower()
    for pattern in _TRANSIENT_PATTERNS:
        if pattern in lower:
            return "transient"
    return "permanent"


def insert_error(
    run_id: int | None, path: str, error_message: str, error_type: str,
) -> int:
    """Record a per-file indexing failure."""
    return execute(
        "INSERT INTO index_errors (run_id, path, error_message, error_type) "
        "VALUES (?, ?, ?, ?)",
        (run_id, path, error_message, error_type),
    )


def resolve_errors(path: str) -> int:
    """Mark all unresolved errors for a path as resolved.

    Returns the number of rows updated.
    """
    db = get_db()
    cursor = db.execute(
        "UPDATE index_errors SET resolved = 1, resolved_at = datetime('now') "
        "WHERE path = ? AND resolved = 0",
        (path,),
    )
    db.commit()
    return cursor.rowcount


def query_errors(
    *,
    resolved: bool | None = False,
    path: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List per-file indexing errors with optional filters."""
    clauses: list[str] = []
    params: list[Any] = []

    if resolved is not None:
        clauses.append("resolved = ?")
        params.append(1 if resolved else 0)
    if path:
        clauses.append("path LIKE ?")
        params.append(path + "%")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    return query(
        f"SELECT id, run_id, path, error_message, error_type, "  # noqa: S608
        f"created_at, resolved_at, resolved "
        f"FROM index_errors {where} "
        f"ORDER BY created_at DESC LIMIT ?",
        tuple(params),
    )
