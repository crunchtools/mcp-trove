"""MCP server registration for mcp-trove-crunchtools."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from .tools import (
    trove_get_chunks,
    trove_index,
    trove_list,
    trove_reindex,
    trove_remove,
    trove_search,
    trove_similar,
    trove_status,
)

mcp = FastMCP(
    "mcp-trove-crunchtools",
    version="0.2.0",
    instructions=(
        "Self-hosted local file indexing MCP server with semantic search. "
        "Index any local directory and search over contents using hybrid "
        "vector + keyword search. Supports PDF, DOCX, Markdown, and text files. "
        "When a vision backend is configured (TROVE_VISION_BACKEND), also indexes "
        "images and videos by captioning them via AI vision APIs, with EXIF "
        "metadata extraction for searchable dates, GPS coordinates, and camera info."
    ),
)


# --- Search Tools ---


@mcp.tool()
async def trove_search_tool(
    query: str,
    path: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Hybrid semantic + keyword search over indexed files.

    Combines vector similarity and FTS5 keyword matching for best results.
    Returns ranked chunks with file paths, scores, and content.

    Args:
        query: Search query text
        path: Optional path prefix to restrict search scope
        limit: Maximum results to return (1-100, default: 10)
    """
    return await trove_search(query, path, limit)


@mcp.tool()
async def trove_similar_tool(
    file_path: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find files similar to a given indexed file.

    Uses the average embedding of the file's chunks as query vector.

    Args:
        file_path: Full path of an indexed file
        limit: Maximum similar files to return (1-100, default: 10)
    """
    return await trove_similar(file_path, limit)


# --- Index Management Tools ---


@mcp.tool()
async def trove_index_tool(path: str) -> dict[str, Any]:
    """Index a specific file or directory.

    Extracts text, generates embeddings, and stores in the search index.
    Skips unchanged files based on SHA-256 checksum comparison.

    Args:
        path: File or directory path to index
    """
    return await trove_index(path)


@mcp.tool()
async def trove_reindex_tool(path: str | None = None) -> dict[str, Any]:
    """Force re-index ignoring checksums.

    If no path given, reindexes all previously indexed files.
    Removes entries for files that no longer exist on disk.

    Args:
        path: Optional file or directory path (reindexes all if omitted)
    """
    return await trove_reindex(path)


@mcp.tool()
async def trove_remove_tool(path: str) -> dict[str, Any]:
    """Remove a file or directory from the index.

    Deletes all associated text chunks and vector embeddings.
    For directories, removes all files under that path prefix.

    Args:
        path: File or directory path to remove from index
    """
    return await trove_remove(path)


# --- Status Tools ---


@mcp.tool()
async def trove_status_tool() -> dict[str, Any]:
    """Index statistics: total files, chunks, disk usage, model info.

    Returns a summary of the current index state including file counts,
    chunk counts, file types breakdown, and configuration details.
    """
    return await trove_status()


@mcp.tool()
async def trove_list_tool(
    path: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List indexed files with metadata.

    Shows file path, type, size, chunk count, and last indexed time.

    Args:
        path: Optional path prefix to filter results
        limit: Maximum files to return (1-500, default: 50)
        offset: Pagination offset (default: 0)
    """
    return await trove_list(path, limit, offset)


@mcp.tool()
async def trove_get_chunks_tool(
    file_path: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Show the text chunks for a specific indexed file.

    Returns the extracted and chunked text content with chunk indices.

    Args:
        file_path: Full path of an indexed file
        limit: Maximum chunks to return (1-500, default: 50)
    """
    return await trove_get_chunks(file_path, limit)
