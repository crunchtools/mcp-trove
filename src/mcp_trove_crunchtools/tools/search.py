"""Search tools for mcp-trove-crunchtools."""

from __future__ import annotations

import json
from typing import Any

from .. import database as db
from ..embedder import embed_query
from ..errors import FileNotIndexedError


async def trove_search(
    query: str,
    path: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Hybrid semantic + FTS5 search over indexed content.

    Combines vector similarity and keyword matching, deduplicates,
    and returns ranked results with file paths and context.
    """
    query_embedding = embed_query(query)
    vec_results = db.search_vectors(query_embedding, limit, path_filter=path)

    try:
        fts_results = db.search_fts(query, limit)
    except Exception:
        fts_results = []

    seen_chunk_ids: set[int] = set()
    merged: list[dict[str, Any]] = []

    for row in vec_results:
        chunk_id = row["chunk_id"]
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        merged.append({
            "chunk_id": chunk_id,
            "path": row["path"],
            "file_type": row["file_type"],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
            "vector_distance": row["distance"],
            "source": "vector",
        })

    for row in fts_results:
        chunk_id = row["chunk_id"]
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        merged.append({
            "chunk_id": chunk_id,
            "path": row["path"],
            "file_type": row["file_type"],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
            "fts_score": row["score"],
            "source": "fts",
        })

    return merged[:limit]


async def trove_similar(
    file_path: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find files similar to a given indexed file.

    Uses the average embedding of all chunks in the file as the query vector.
    """
    existing = db.query_one(
        "SELECT id FROM files WHERE path = ?", (file_path,)
    )
    if not existing:
        raise FileNotIndexedError(file_path)

    file_id = existing["id"]
    chunk_vecs = db.get_file_chunks_avg_embedding(file_id)

    if not chunk_vecs:
        return []

    import struct

    all_embeddings: list[list[float]] = []
    for row in chunk_vecs:
        raw = row["embedding"]
        num_floats = len(raw) // 4
        floats = list(struct.unpack(f"{num_floats}f", raw))
        all_embeddings.append(floats)

    dims = len(all_embeddings[0])
    avg_embedding = [
        sum(emb[d] for emb in all_embeddings) / len(all_embeddings)
        for d in range(dims)
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
        similar_files.append({
            "path": row_path,
            "file_type": row["file_type"],
            "distance": row["distance"],
            "sample_content": row["content"][:200],
        })
        if len(similar_files) >= limit:
            break

    return similar_files
