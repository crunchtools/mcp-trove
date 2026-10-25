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

## Tools (10)

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

### Status (5)

| Tool | Description |
|------|-------------|
| `trove_status` | Index statistics: total files, chunks, disk usage, model info. |
| `trove_list` | List indexed files with metadata (size, type, chunk count). |
| `trove_get_chunks` | Show the text chunks for a specific indexed file. |
| `trove_log` | Activity log of indexing runs: start/finish times, file counts, errors. Persisted in SQLite, survives container crashes. |
| `trove_quality` | Per-file error tracking from indexing runs: which files failed, why, and whether a later index resolved it. |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TROVE_DB` | `~/.local/share/mcp-trove/trove.db` | SQLite database path |
| `TROVE_PATHS` | (none) | Directories to index in background mode, separated by `:` (`;` on Windows) |
| `TROVE_INDEX_WORKERS` | `20` | Images and videos extracted at once. PDFs run one at a time and text files two, whatever this is set to |
| `TROVE_INDEX_BATCH` | `50` | Unused since 0.12.0, when files began streaming through extraction instead of in batches; still accepted |
| `TROVE_MANIFEST_DIR` | (none) | Directory of per-path file listings the host writes before a run (see Background Indexing); unset, each run walks its trees |
| `TROVE_EMBED_THREADS` | `2` | ONNX threads for embedding |
| `TROVE_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model name |
| `TROVE_EXCLUDE_PATTERNS` | `*.iso,*.zip,...` | Glob patterns to skip |
| `TROVE_CHUNK_SIZE` | `1000` | Characters per text chunk |
| `TROVE_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TROVE_PDF_MAX_PAGES` | `5` | Pages indexed from the start of each PDF, enough to tell what the document is about while keeping memory flat on large manuals; `0` or negative indexes every page |
| `TROVE_MAX_TEXT_CHARS` | `5000000` | Characters of extracted text kept per file; the rest is dropped with a warning so one huge document cannot flood a batch with chunks; must be positive, other values fall back to the default with a warning |
| `TROVE_VISION_MAX_CALLS_PER_RUN` | `3000` | Vision API calls one index run may make; images and videos past it are deferred to the next run, not failed. `0` means no limit |
| `TROVE_VISION_PDF_SCANS` | `false` | Caption the first page of scanned PDFs through the vision backend. Sends an image of the page to the provider, so it is opt-in |
| `TROVE_VISION_FALLBACK_MODEL` | (none) | OpenRouter only: a second model tried when the first is rate limited. Unset by default, since the usual fallback costs more |
| `TROVE_REQUIRE_MOUNT` | (none) | Mount point the index paths live on. `--index` checks it is mounted and answers a listing within 30 s before indexing, and fails the run otherwise |
| `TROVE_MAX_EXTRACT_TIMEOUTS` | `5` | Timed-out file extractions after which a run aborts as a hung source |
| `TROVE_BACKUP_DIR` | `<db dir>/backups` | Where `--index` keeps a daily `VACUUM INTO` copy of the database |
| `TROVE_BACKUP_KEEP` | `3` | Daily database copies kept; `0` disables backups |
| `TROVE_STATUS_FILE` | `<db dir>/last_run.json` | Summary of the last `--index` run, for monitoring |
| `TROVE_STATUS_FILE_MODE` | `600` | Octal file mode of the summary. Use `644` when a monitor reads it as another user; it can carry error text that names files |
| `TROVE_VISION_BACKEND` | `none` | Vision captioning backend: `gemini`, `openai`, `openrouter`, `ollama`, or `none` to disable |
| `TROVE_VISION_MODEL` | per backend | Vision model. Defaults to `gemini-2.5-flash`, `gpt-4o-mini`, `google/gemini-3.1-flash-lite` (openrouter) or `llava` depending on the backend; empty when disabled |
| `TROVE_VISION_PROMPT` | (see below) | Caption prompt sent with each image |
| `TROVE_VISION_TIMEOUT` | `120` | Seconds allowed per vision API call |

Image and video captioning is off until `TROVE_VISION_BACKEND` is set. The
`openrouter` backend reads `OPENROUTER_API_KEY_FILE` or `OPENROUTER_API_KEY`,
handles video as well as images, and only routes to zero-retention providers. The
default caption prompt is:

> Generate a concise caption for this image suitable for search indexing.
> Include: what you see, location if identifiable, time of day, any readable
> text or signs.

## Background Indexing

Set up a systemd timer to keep your index fresh:

```bash
TROVE_PATHS=~/Documents:~/AutoSync mcp-trove-crunchtools --index
```

On a network-backed source, walking the tree is the slow part: one listing per
directory and one stat per file, each a round trip. Point `TROVE_MANIFEST_DIR`
at a directory holding one rclone `lsjson` listing per index path, at that path
under the directory with `.lsjson` appended, and the scan reads it instead. For
the index path `/pcloud/Documents/Academic`:

```bash
mkdir -p /manifests/pcloud/Documents
rclone lsjson -R --fast-list --files-only --no-mimetype pcloud:Documents/Academic \
  > /manifests/pcloud/Documents/Academic.lsjson
```

A listing over 6 hours old, missing or unreadable falls back to the walk. With a
manifest, files it no longer lists are removed from the index, unless a tree of
100 or more files would lose over 10% at once, which aborts the path instead.

An `--index` run is built to be left alone:

- Each file is stored in one transaction, so a run killed part way (OOM, reboot) leaves no half-indexed file behind.
- Before indexing, it runs `PRAGMA quick_check`, keeps a daily backup, marks runs abandoned for more than 12 hours as `aborted`, and probes `TROVE_REQUIRE_MOUNT`.
- A path aborts instead of indexing when its file count falls below half of its last completed run (an unmounted source looks like deleted files), or when extractions keep timing out (a hung source). `trove_reindex` on the path overrides the file-count check.
- The exit code is `0` only when every path completed with no more than 10% of its files failing. Any other outcome exits `1`, so the systemd unit fails visibly. `TROVE_STATUS_FILE` records what happened, path by path.
- The database records its embedding model and refuses to open under a different one, because vectors from two models cannot be searched together.
- Vision captions are cached by file content (SHA-256) and vision model, so a renamed or copied photo reuses its caption. A cached caption is kept while its content is indexed and for 90 days after. A file that failed permanently, or 5 times transiently, is not retried until it changes; `trove_reindex` on its path retries it.
- With `TROVE_VISION_PDF_SCANS=true`, a PDF whose first 5 pages hold under 20 characters of text (a scan) has its first page captioned through the vision backend, under the same budget and cache. It is off by default because it sends an image of the page to the vision provider.
- `trove_reindex` with no path re-reads every file, so it requires `force=true`. Cached captions are reused unless `TROVE_VISION_MODEL` changed.

## License

AGPL-3.0-or-later
