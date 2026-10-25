"""Search tools for mcp-trove-crunchtools."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import struct
from typing import Any

from pydantic import ValidationError

from .. import database as db
from ..embedder import embed_query
from ..errors import FileNotIndexedError, InvalidInputError
from ..models import SearchParams, SimilarParams

logger = logging.getLogger(__name__)


# Reciprocal rank fusion: each ranked list gives a hit 1/(RRF_K + rank). It
# needs no score calibration between bm25, vector distance and path matches,
# and a hit near the top of any one list ranks well.
RRF_K = 60
# A path holding every query term is a precise match, so it counts double.
SOURCE_WEIGHTS = {"vector": 1.0, "fts": 1.0, "path": 2.0}
CANDIDATES_PER_RESULT = 5  # hits fetched from each list per result returned
# When one file's chunks fill the candidates, the pool widens up to this.
MAX_CANDIDATES = 2000
_TERM = re.compile(r"\w{2,}")


async def trove_search(
    query: str,
    path: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Hybrid search: vector similarity, keywords and file paths, fused by rank.

    One result per file, its best-matching chunk, with matches counting how
    many of the file's chunks were hits. A file whose text says little (a
    receipt scan, a short caption) can still rank on its folder and name.
    """
    try:
        params = SearchParams(query=query, path=path, limit=limit)
    except ValidationError as exc:
        raise InvalidInputError(str(exc)) from exc
    query, path, limit = params.query, params.path, params.limit
    terms = list(dict.fromkeys(_TERM.findall(query.lower())))
    embedding = embed_query(query)
    wanted = limit * CANDIDATES_PER_RESULT
    while True:
        ranked = {
            "vector": db.search_vectors(embedding, wanted, path_filter=path),
            "fts": _keyword_hits(terms, wanted, path),
            "path": db.search_paths(terms, wanted, path_filter=path),
        }
        fused = _fuse(ranked)
        # A list that came back full may hold more files past its cut.
        more = any(len(hits) >= wanted for hits in ranked.values())
        if len(fused) >= limit or not more or wanted >= MAX_CANDIDATES:
            return fused[:limit]
        wanted = min(wanted * 4, MAX_CANDIDATES)


def _keyword_hits(terms: list[str], limit: int, path: str | None) -> list[dict[str, Any]]:
    """bm25 hits for any of the terms, each quoted so no character is FTS5 syntax."""
    if not terms:
        return []
    match = " OR ".join(f'"{term}"' for term in terms)
    try:
        return db.search_fts(match, limit, path_filter=path)
    except sqlite3.OperationalError as exc:
        # Quoted terms cannot be malformed, so this is the index itself:
        # carry on with the other lists, but say so.
        logger.warning("FTS5 query failed, searching without keywords: %s", exc)
        return []


def _fuse(ranked: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge ranked hit lists by reciprocal rank, keeping each file's best chunk."""
    chunks: dict[int, dict[str, Any]] = {}
    for source, hits in ranked.items():
        for rank, hit in enumerate(hits, start=1):
            entry = chunks.setdefault(
                hit["chunk_id"],
                {
                    "chunk_id": hit["chunk_id"],
                    "path": hit["path"],
                    "file_type": hit["file_type"],
                    "chunk_index": hit["chunk_index"],
                    "content": hit["content"],
                    "metadata": json.loads(hit["metadata"]) if hit["metadata"] else None,
                    "score": 0.0,
                    "sources": [],
                },
            )
            entry["score"] += SOURCE_WEIGHTS[source] / (RRF_K + rank)
            entry["sources"].append(source)
    files: dict[str, dict[str, Any]] = {}
    for entry in sorted(chunks.values(), key=lambda e: e["score"], reverse=True):
        best = files.get(entry["path"])
        if best is None:
            files[entry["path"]] = {**entry, "matches": 1}
        else:
            best["matches"] += 1
    for entry in files.values():
        entry["score"] = round(entry["score"], 6)
    return list(files.values())


async def trove_similar(
    file_path: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find files similar to a given indexed file.

    Uses the average embedding of all chunks in the file as the query vector.
    """
    try:
        params = SimilarParams(file_path=file_path, limit=limit)
    except ValidationError as exc:
        raise InvalidInputError(str(exc)) from exc
    file_path, limit = params.file_path, params.limit

    existing = db.query_one("SELECT id FROM files WHERE path = ?", (file_path,))
    if not existing:
        raise FileNotIndexedError(file_path)

    file_id = existing["id"]
    chunk_vecs = db.get_file_chunks_avg_embedding(file_id)

    if not chunk_vecs:
        return []

    all_embeddings: list[list[float]] = []
    for row in chunk_vecs:
        raw = row["embedding"]
        num_floats = len(raw) // 4
        floats = list(struct.unpack(f"{num_floats}f", raw))
        all_embeddings.append(floats)

    dims = len(all_embeddings[0])
    avg_embedding = [
        sum(emb[d] for emb in all_embeddings) / len(all_embeddings) for d in range(dims)
    ]

    vec_results = db.search_vectors(avg_embedding, limit + 10)

    seen_paths: set[str] = set()
    seen_paths.add(file_path)
    similar_files: list[dict[str, Any]] = []

    for row in vec_results:
        row_path = row["path"]
        if row_path in seen_paths:
            continue
        seen_paths.add(row_path)
        similar_files.append(
            {
                "path": row_path,
                "file_type": row["file_type"],
                "distance": row["distance"],
                "sample_content": row["content"][:200],
            }
        )
        if len(similar_files) >= limit:
            break

    return similar_files
