# mcp-trove-crunchtools

Self-hosted local file indexing MCP server with semantic search. Index any local directory (pCloud ~/AutoSync/, rclone mounts, ~/Documents/, anything) and search over the contents using hybrid vector + keyword search.

## Features

- **Hybrid search** — Combines semantic vector similarity with FTS5 keyword matching
- **Multiple file formats** — PDF, DOCX, Markdown, plain text, source code
- **Local-first** — No cloud services, no per-seat fees, your data stays on your machine
- **Lightweight embeddings** — Uses fastembed (ONNX runtime) instead of PyTorch (~22MB vs ~2GB)
- **Incremental indexing** — SHA-256 checksum-based change detection
- **Background mode** — `--index` CLI mode for systemd timer automation

## Install

### uvx (recommended)

```bash
uvx mcp-trove-crunchtools
```

### pip

```bash
pip install mcp-trove-crunchtools
```

### Container

```bash
podman run -v trove-data:/data -v ~/Documents:/docs:ro quay.io/crunchtools/mcp-trove
```

## Claude Code Integration

```bash
claude mcp add mcp-trove-crunchtools -- uvx mcp-trove-crunchtools
```

## Tools (8)

### Search (2)

| Tool | Description |
|------|-------------|
| `trove_search` | Hybrid semantic + FTS5 search. Returns ranked chunks with file paths, scores, and content. |
| `trove_similar` | Find files similar to a given indexed file using its average embedding. |

### Index Management (3)

| Tool | Description |
|------|-------------|
| `trove_index` | Index a specific file or directory. Skips unchanged files (checksum-based). |
| `trove_reindex` | Force re-index ignoring checksums. If no path given, reindexes everything. |
| `trove_remove` | Remove a file or directory from the index. |

### Status (3)

| Tool | Description |
|------|-------------|
| `trove_status` | Index statistics: total files, chunks, disk usage, model info. |
| `trove_list` | List indexed files with metadata (size, type, chunk count). |
| `trove_get_chunks` | Show the text chunks for a specific indexed file. |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TROVE_DB` | `~/.local/share/mcp-trove/trove.db` | SQLite database path |
| `TROVE_PATHS` | (none) | Colon-separated directories to index in background mode |
| `TROVE_INDEX_WORKERS` | `2` | Concurrent embedding workers |
| `TROVE_INDEX_BATCH` | `50` | Files per indexing batch |
| `TROVE_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model name |
| `TROVE_EXCLUDE_PATTERNS` | `*.iso,*.zip,...` | Glob patterns to skip |
| `TROVE_CHUNK_SIZE` | `1000` | Characters per text chunk |
| `TROVE_CHUNK_OVERLAP` | `200` | Overlap between chunks |

## Background Indexing

Set up a systemd timer to keep your index fresh:

```bash
TROVE_PATHS=~/Documents:~/AutoSync mcp-trove-crunchtools --index
```

## License

AGPL-3.0-or-later
