# mcp-trove-crunchtools

Self-hosted local file indexing MCP server with semantic search.

## Quick Start

```bash
uv sync --all-extras
uv run mcp-trove-crunchtools
```

## Environment Variables

- `TROVE_DB` — SQLite database path (default: ~/.local/share/mcp-trove/trove.db)
- `TROVE_PATHS` — Colon-separated directories to index in background mode
- `TROVE_INDEX_WORKERS` — Concurrent embedding workers (default: 2)
- `TROVE_INDEX_BATCH` — Files per indexing batch (default: 50)
- `TROVE_EMBEDDING_MODEL` — fastembed model name (default: BAAI/bge-small-en-v1.5)
- `TROVE_EXCLUDE_PATTERNS` — Comma-separated glob patterns to skip
- `TROVE_CHUNK_SIZE` — Characters per text chunk (default: 1000)
- `TROVE_CHUNK_OVERLAP` — Overlap between chunks (default: 200)

## Tools (8)

### Search (2)
- trove_search_tool, trove_similar_tool

### Index Management (3)
- trove_index_tool, trove_reindex_tool, trove_remove_tool

### Status (3)
- trove_status_tool, trove_list_tool, trove_get_chunks_tool

## Development

```bash
uv run ruff check src tests    # Lint
uv run mypy src                # Type check
uv run pytest -v               # Test
gourmand --full .              # Slop detection
podman build -f Containerfile . # Container
```

## Architecture

- `database.py` — SQLite with sqlite-vec + FTS5
- `embedder.py` — fastembed ONNX-based embeddings (no PyTorch)
- `extractor.py` — PDF (pymupdf4llm), DOCX, Markdown, text extraction
- `indexer.py` — File scanning, checksumming, chunking, orchestration
- `tools/` — Two-layer: pure functions called by server.py wrappers
- `--index` CLI mode for systemd timer background indexing
