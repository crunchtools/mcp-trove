# mcp-trove-crunchtools Constitution

> **Version:** 1.1.0
> **Ratified:** 2026-03-14
> **Amended:** 2026-10-02
> **Status:** Active
> **Inherits:** [crunchtools/constitution](https://github.com/crunchtools/constitution) v1.20.0
> **Profile:** MCP Server

This file holds what is specific to mcp-trove. The fleet rules and the MCP
Server profile (five-layer security model, two-layer tools, distribution
channels, transport modes, quality gates, Gourmand) apply at the inherited
version and are checked against this repo's files by `constitution.yml`. They
are not restated here.

## Self-Contained Operation

The server MUST work without any external service account. Indexing and
search need no credentials and read local files only:

- SQLite with sqlite-vec and FTS5, auto-created on first run at `TROVE_DB`
  (default `~/.local/share/mcp-trove/trove.db`)
- fastembed on the ONNX runtime for embeddings, computed and stored locally,
  with no PyTorch dependency
- `pymupdf4llm` for PDF and `python-docx` for DOCX extraction

`tools/*.py` call the database, embedder or indexer rather than an HTTP
client.

## Security Model Specifics

- **Credentials:** only the optional vision backends (`TROVE_VISION_BACKEND`
  = gemini, openai or openrouter) take one: `GEMINI_API_KEY`,
  `OPENAI_API_KEY` or `OPENROUTER_API_KEY`, `_FILE` variant preferred where
  supported. Keys are `SecretStr`, never logged, and sent only to that
  backend's own API.
- **Filesystem hardening:** read-only access to indexed files; paths are
  canonicalized to prevent traversal; file size limits prevent memory
  exhaustion; exclude patterns skip binary and large files; indexing batch
  size is bounded.
- **Resource limits:** search queries are length-limited and pagination is
  bounded; an `asyncio.Semaphore` caps concurrent embedding workers.

## Test Database

Tool tests run against a fresh in-memory (`:memory:`) SQLite database per
test, with the schema applied and the fastembed model mocked to return
deterministic vectors.

## Instance

| Context | Name |
|---------|------|
| GitHub repo | `crunchtools/mcp-trove` |
| PyPI package | `mcp-trove-crunchtools` |
| Python module | `mcp_trove_crunchtools` |
| Container image | `quay.io/crunchtools/mcp-trove` |
| systemd service | `mcp-trove.service` |
| HTTP port | 8020 |

## History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-03-14 | Initial constitution |
| 1.0.1 | 2026-03-16 | Container Conventions section added |
| 1.1.0 | 2026-10-02 | Manifest under constitution v1.18.0: profile restatement removed, mcp-trove specifics kept |
