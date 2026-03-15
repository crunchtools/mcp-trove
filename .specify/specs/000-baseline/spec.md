# Spec 000: Baseline

> **Status:** Accepted
> **Created:** 2026-03-14

## Overview

Initial release of mcp-trove-crunchtools with 8 tools across 3 categories: search (2), index management (3), and status (3).

## Tools

### Search (2)
- `trove_search` — Hybrid semantic + FTS5 search
- `trove_similar` — Find files similar to a given indexed file

### Index Management (3)
- `trove_index` — Index a file or directory
- `trove_reindex` — Force re-index ignoring checksums
- `trove_remove` — Remove a file or directory from the index

### Status (3)
- `trove_status` — Index statistics
- `trove_list` — List indexed files
- `trove_get_chunks` — Show text chunks for a file

## Architecture

- SQLite with sqlite-vec for vector search and FTS5 for keyword search
- fastembed (ONNX runtime) for embeddings — no PyTorch dependency
- Text extraction: PDF (pymupdf4llm), DOCX (python-docx), Markdown, plain text
- Two indexing modes: background (`--index` CLI) and on-demand (tool call)
