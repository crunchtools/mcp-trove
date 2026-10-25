"""Operator commands: re-embedding the index under a new model, and search evaluation.

Run from the CLI (--reembed, --eval FILE), not as MCP tools: both are
long-running and belong to whoever operates the index.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import sys
from typing import TYPE_CHECKING, Any

import sqlite_vec

from .. import database as db
from ..config import get_config
from ..embedder import embed_texts, get_vector_dims
from .search import trove_search

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

REEMBED_BATCH = 256
REEMBED_LOG_EVERY = 40  # batches between progress lines
BUSY_TIMEOUT_SECONDS = 30  # the MCP server and the indexer write to the same file
EVAL_TOP = 10  # MRR is computed over this many results
EVAL_RECALL_AT = 5


def reembed(db_path: str | None = None) -> dict[str, Any]:
    """Rebuild every vector from the stored chunks under TROVE_EMBEDDING_MODEL.

    Nothing is re-extracted or re-captioned, so a model change costs only
    embedding time. Vectors are built in a side table, so a run that is
    killed resumes where it stopped; chunks indexed meanwhile are embedded
    at the swap, which replaces the vector table and records the model in
    one transaction. Searches keep the old vectors until then.

    db_path overrides TROVE_DB. Returns the model and its vector dimensions,
    the number of chunks now embedded under it, and how many this run
    embedded (fewer than chunks when it resumed a killed run).
    """
    path = db_path or get_config().db_path
    model = get_config().embedding_model
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_SECONDS)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        _prepare_side_table(conn, model)
        embedded = _embed_pending(conn)
        conn.execute("BEGIN IMMEDIATE")
        embedded += _embed_pending(conn, commit=False)
        dims = get_vector_dims(model)
        conn.execute("DROP TABLE IF EXISTS chunks_vec")
        db.create_vec_table(conn, dims)
        conn.execute(
            "INSERT INTO chunks_vec (chunk_id, embedding) "
            "SELECT chunk_id, embedding FROM reembed_vectors"
        )
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('embedding_model', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (model,),
        )
        total = conn.execute("SELECT count(*) FROM reembed_vectors").fetchone()[0]
        conn.execute("DROP TABLE reembed_vectors")
        conn.commit()
    finally:
        conn.close()
    return {"model": model, "dimensions": dims, "chunks": total, "embedded_this_run": embedded}


def _prepare_side_table(conn: sqlite3.Connection, model: str) -> None:
    """The side table for model, keeping progress only if it was for the same model."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS reembed_vectors "
        "(chunk_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL, model TEXT NOT NULL)"
    )
    stale = conn.execute("SELECT 1 FROM reembed_vectors WHERE model != ? LIMIT 1", (model,))
    if stale.fetchone():
        conn.execute("DELETE FROM reembed_vectors")
    # Chunks deleted since a killed run would otherwise come back as vectors.
    conn.execute("DELETE FROM reembed_vectors WHERE chunk_id NOT IN (SELECT id FROM chunks)")
    conn.commit()


def _embed_pending(conn: sqlite3.Connection, *, commit: bool = True) -> int:
    """Embed every chunk without a side-table vector, in batches; returns how many.

    A keyset cursor over chunk ids, so each batch costs the same however far
    the run has got, and a resumed run skips what is done at index speed.
    """
    model = get_config().embedding_model
    done = 0
    last_id = 0
    while True:
        rows = conn.execute(
            "SELECT c.id, c.content FROM chunks c "
            "LEFT JOIN reembed_vectors r ON r.chunk_id = c.id "
            "WHERE c.id > ? AND r.chunk_id IS NULL ORDER BY c.id LIMIT ?",
            (last_id, REEMBED_BATCH),
        ).fetchall()
        if not rows:
            return done
        last_id = rows[-1][0]
        vectors = embed_texts([content for _, content in rows])
        conn.executemany(
            "INSERT INTO reembed_vectors (chunk_id, embedding, model) VALUES (?, ?, ?)",
            [
                (chunk_id, sqlite_vec.serialize_float32(vector), model)
                for (chunk_id, _), vector in zip(rows, vectors, strict=True)
            ],
        )
        if commit:
            conn.commit()
        done += len(rows)
        if done % (REEMBED_BATCH * REEMBED_LOG_EVERY) == 0:
            logger.info("Re-embedded %d chunks", done)


def evaluate(queries_file: Path) -> dict[str, Any]:
    """Score search on a list of queries with the path fragments a right answer has.

    The file is a JSON array of {"query": ..., "expect": [fragment, ...]}. A
    result is right when its path contains any fragment, ignoring case.
    Returns the embedding model searched with, the number of queries,
    recall@5 (the share with a right answer in the top 5), MRR over the top
    10, and per query the rank of its first right answer, or None.
    """
    cases = json.loads(queries_file.read_text())
    rows = []
    for case in cases:
        results = asyncio.run(trove_search(case["query"], limit=EVAL_TOP))
        fragments = [fragment.lower() for fragment in case["expect"]]
        rank = next(
            (
                position
                for position, result in enumerate(results, start=1)
                if any(fragment in result["path"].lower() for fragment in fragments)
            ),
            None,
        )
        rows.append({"query": case["query"], "rank": rank})
    found = [row["rank"] for row in rows]
    count = max(1, len(rows))
    return {
        "model": get_config().embedding_model,
        "queries": len(rows),
        f"recall@{EVAL_RECALL_AT}": round(
            sum(1 for r in found if r and r <= EVAL_RECALL_AT) / count, 3
        ),
        "mrr": round(sum(1 / r for r in found if r) / count, 3),
        "results": rows,
    }


def print_json(result: dict[str, Any]) -> None:
    """Write a command's result to standard output as indented JSON."""
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
