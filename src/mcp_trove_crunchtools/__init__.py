"""mcp-trove-crunchtools: Self-hosted local file indexing with semantic search."""

from __future__ import annotations

import argparse
import os
import sys

__version__ = "0.15.0"


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
        from .tools.run_index import run_index

        code = run_index()
        # Exit without waiting on worker threads: a read stuck on a hung mount
        # cannot be cancelled and would keep a finished run alive forever.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)

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
