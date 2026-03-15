"""mcp-trove-crunchtools: Self-hosted local file indexing with semantic search."""

from __future__ import annotations

import argparse
import sys

__version__ = "0.1.0"


def main() -> None:
    """Entry point for mcp-trove-crunchtools."""
    parser = argparse.ArgumentParser(
        prog="mcp-trove-crunchtools",
        description="Self-hosted local file indexing MCP server with semantic search",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8020,
        help="HTTP port (default: 8020)",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index TROVE_PATHS directories and exit (for systemd timer)",
    )

    args = parser.parse_args()

    if args.index:
        _run_index()
        return

    from .database import get_db
    from .server import mcp

    get_db()

    match args.transport:
        case "stdio":
            mcp.run(transport="stdio")
        case "sse":
            mcp.run(transport="sse", host=args.host, port=args.port)
        case _:
            mcp.run(transport="streamable-http", host=args.host, port=args.port)


def _run_index() -> None:
    """Index configured directories (CLI mode for systemd timer)."""
    import asyncio
    from pathlib import Path

    from .config import get_config
    from .database import get_db
    from .tools.index import trove_index

    get_db()
    config = get_config()

    if not config.index_paths:
        print("No TROVE_PATHS configured. Nothing to index.")
        sys.exit(0)

    for dir_path in config.index_paths:
        target = Path(dir_path)
        if not target.is_dir():
            print(f"Skipping non-directory: {dir_path}")
            continue
        result = asyncio.run(trove_index(dir_path))
        indexed = result.get("files_indexed", 0)
        skipped = result.get("files_skipped", 0)
        chunks = result.get("total_chunks", 0)
        print(f"{dir_path}: {indexed} indexed, {skipped} skipped, {chunks} chunks")

    sys.exit(0)
