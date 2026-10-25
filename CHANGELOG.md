# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and this project adheres to
[Semantic Versioning](https://semver.org/).

Entries prior to 2026-09-19 are back-filled from GitHub Release notes (RT #1484),
except `0.3.1` through `0.5.1`, which had no release to back-fill from and were
reconstructed from their commit ranges in RT #1485.

## [Unreleased]

## [0.13.0] - 2026-10-25

### Added

- `TROVE_INBOX_PATHS`: index paths that empty out by design, such as a
  phone's upload folder whose photos are filed elsewhere. The shrink gate and
  the manifest removal guard are not applied to them, so an emptied inbox has
  its entries removed instead of aborting the run as a dead mount (RT #1542).

### Fixed

- Embedding ran on the event loop, so a file with hundreds of chunks stalled
  it for minutes, and the timeouts of extractions still in flight fired on
  reads that had long finished. Five of them aborted the run as a hung
  source. Embedding now runs on a worker thread; the database writes stay on
  the loop's thread (RT #1542).

## [0.12.0] - 2026-10-25

### Added

- Manifest scans (`TROVE_MANIFEST_DIR`): the scan reads paths, sizes and
  mtimes from an rclone `lsjson` listing the host writes before a run (at the
  index path under that directory, plus `.lsjson`), instead
  of walking and stat'ing the tree over the mount. 5,600 pCloud files list in
  3 s. A missing, unreadable or over-6-hour-old listing falls back to the walk.
  mtimes match within 1 s.
- With a manifest, files it no longer lists are removed from the index;
  `files_removed` in results and `last_run.json`. A tree of 100 or more indexed
  files losing over 10% at once aborts the path instead.
- `TROVE_EMBED_THREADS` (default `2`) for ONNX embedding.

### Changed

- Extraction streams: a window of files is in flight and each is stored when
  it finishes, instead of lockstep batches that waited on their slowest file.
  Concurrency is capped per type: PDFs 1, DOCX/text/Markdown 2, images and
  videos `TROVE_INDEX_WORKERS`. `TROVE_INDEX_BATCH` is unused.
- HTML is indexed as its visible text, without markup, scripts or styles.
- Chunks end at a paragraph break, line break or sentence end where there is
  one in their last 40%.
- The container no longer sets `OMP_NUM_THREADS`, which ONNX Runtime ignores.

### Fixed

- The caption cache used `TROVE_DB` rather than the database actually opened,
  so with an explicit database path it read, and created, the wrong file.

## [0.11.1] - 2026-10-25

### Changed

- `TROVE_STATUS_FILE_MODE` (octal, default `600`) sets the mode of
  `last_run.json`, so a monitor running as another user can be let in (`644`).
  The summary records `vision_budget` and `latest_backup` (name and time),
  since the backup directory is owner-only.

## [0.11.0] - 2026-10-25

### Added

- Caption cache keyed by file SHA-256 (`captions` table): renamed, moved or
  duplicate images and videos reuse their caption instead of a new vision call.
- `TROVE_VISION_MAX_CALLS_PER_RUN` (default `3000`): files past the budget are
  `deferred` to the next run, not failed, and are not downloaded first.
  `last_run.json` records `vision_calls` and per-path `files_deferred`.
- `TROVE_VISION_PDF_SCANS` (default off): scanned PDFs (under 20 characters of
  text on their first 5 pages) have their first page captioned through the
  vision backend, rendered within 4 megapixels whatever the declared page size.
- `--index` prunes cached captions whose content has been out of the index
  for 90 days.
- `trove_reindex(force=...)`: reindexing the whole index requires `force=true`.
  Cached captions are reused unless they came from another vision model.

### Changed

- A file version that failed permanently, or 5 times transiently, is skipped
  (`known_failure`) until its size or mtime changes; `trove_reindex` on the
  path retries it. `index_errors` records `file_size` and `mtime` (schema 2).
- The OpenRouter fallback model is no longer built in; set
  `TROVE_VISION_FALLBACK_MODEL` to enable one.

## [0.10.0] - 2026-10-25

### Fixed

- A file's record, chunks and vectors are written in one transaction, after
  its embeddings are computed. A process killed part way (OOM, reboot) used to
  leave a file row matching its checksum but missing chunks, skipped as
  unchanged on every later run (RT #1542). A failed re-index now keeps the
  file's previous entry.
- Opening an older database removes rows left incomplete that way, once
  (`PRAGMA user_version` 1), so the next run indexes them again.
- Runs left `running` for more than 12 hours by a killed process are marked
  `aborted` at the start of the next `--index` run.

### Added

- `--index` safety checks: `PRAGMA quick_check`, a daily `VACUUM INTO` backup
  (`TROVE_BACKUP_DIR`, `TROVE_BACKUP_KEEP`), a mount probe
  (`TROVE_REQUIRE_MOUNT`), a file-count drop check per path, and an abort after
  `TROVE_MAX_EXTRACT_TIMEOUTS` timed-out extractions.
- `--index` exits non-zero when any path fails, aborts, is missing or has more
  than 10% of its files fail, and writes a per-path summary to
  `TROVE_STATUS_FILE`.
- A `meta` table records the embedding model and chunking settings; a
  database built with a different embedding model is refused.
- `PRAGMA busy_timeout=30000` and `synchronous=NORMAL`.

### Changed

- The container installs the exact, hash-checked versions in `uv.lock`
  instead of resolving `pyproject.toml` ranges at build time.

## [0.9.4] - 2026-10-25

### Changed

- The directory scan prunes hidden directories (`.git`, `.hg`, ...) instead
  of walking them and filtering the results. Over the pCloud mount every
  listing is a network call, and one Mercurial store stalled a run (RT #1542).
- Supported type and exclude patterns are checked before a file is stat'ed.

## [0.9.3] - 2026-10-24

### Changed

- A file with no index record is checksummed by the parallel extract worker,
  not by the serial scan before extraction. Over the pCloud mount that scan
  read every new file one at a time and took hours before the first file was
  stored (RT #1542).

## [0.9.2] - 2026-10-24

### Fixed

- Embeddings are encoded 8 texts per ONNX pass instead of fastembed's
  default 256. Long chunks at 256 per pass OOM-killed the 1.5 GB indexer on
  a single large HTML table (RT #1542).

## [0.9.1] - 2026-10-24

### Fixed

- MuPDF's process-wide font and image store is emptied after each PDF. It
  grew across a long index run until the 1.5 GB indexer was OOM-killed
  (RT #1542).

## [0.9.0] - 2026-10-24

### Added

- `TROVE_PDF_MAX_PAGES` (default `5`; `0` or negative for no cap): only the
  first N pages of each PDF are converted and indexed, enough to tell what a
  document is about.
- `TROVE_MAX_TEXT_CHARS` (default `5000000`): extracted text kept per file of
  any type; the rest is dropped with a warning. A non-positive value falls
  back to the default.

### Changed

- PDFs are no longer indexed in full by default. Set `TROVE_PDF_MAX_PAGES=0`
  to restore whole-document indexing.

### Fixed

- The indexer was OOM-killed in a 1.5 GB container converting 20-80 MB PDF
  service manuals, because pymupdf4llm builds the whole document's markdown in
  memory (RT #1542). The page cap bounds that work, and header detection is
  scoped to the same pages (pymupdf4llm's default scans every page even when a
  page list is given). Chunks are now embedded 256 at a time instead of all of
  a file's chunks at once.
- An embedding failure part way through a file no longer leaves its file record
  behind. The record matched the file's checksum, so the file was skipped as
  unchanged on every later run and never indexed.

## [0.8.1] - 2026-10-24

### Fixed

- PDF indexing failed on every file with "pymupdf4llm not installed". pymupdf4llm
  2.0 is an empty shim (the code moved into pymupdf 2.x), and `>=0.0.17` let the
  container resolve it with no pymupdf at all. Pinned to `<2`, the container build
  now imports `pymupdf` and `pymupdf4llm` so a hollow dependency fails the build,
  and a test extracts a real PDF (RT #1542).
- Removed the Containerfile's `libstdc++` copy; the Hummingbird Python runtime
  ships it.

## [0.8.0] - 2026-10-20

### Changed

- Container base images pinned to `quay.io/hummingbird/python:3.14*` instead of
  `latest`, which moved to Python 3.15.0 on 2026-10-20 (crunchtools/constitution#66).

## [0.7.2] - 2026-10-18

### Fixed
- `TROVE_PATHS` is split on `os.pathsep` instead of a hardcoded `:`, so Windows
  paths with drive letters (`C:\Users\...`) are no longer broken apart. On
  Windows, separate directories with `;`. Thanks to @Ged-fi (#8).

## [0.7.1] - 2026-10-11

### Fixed
- `trove_log` and `trove_quality` now validate `limit` (1-500) and raise
  `InvalidInputError` instead of passing it straight to SQLite.
- The `chunks_vec` vector size now comes from the configured
  `TROVE_EMBEDDING_MODEL` instead of a hardcoded 384. Opening an existing
  database whose vector size differs from the model fails at startup with an
  error naming both sizes; reindex into a new database (`TROVE_DB`). The
  database is never rebuilt automatically. Reported by @Ged-fi, who also
  proposed a fix in #9 (#7).

## [0.7.0] - 2026-10-10

### Added
- The seven tools that only read (`trove_search`, `trove_similar`,
  `trove_status`, `trove_log`, `trove_list`, `trove_get_chunks`,
  `trove_quality`) publish `readOnlyHint: true`. A gateway uses it to decide
  whether an invalid optional argument may be dropped or must refuse the call
  (crunchtools/constitution#35).
- Tests pin every registered tool into `READ_ONLY` or `WRITES`, and run each
  read-only tool against a SQLite connection set to `query_only`, which raises
  on any statement that would change the database.

### Changed
- Inherits constitution v1.22.0; the workflow pins and the pre-commit hook rev
  move with it.
- Constitution is now a v1.18.0 manifest: only repo-specific facts remain;
  fleet and profile rules apply by reference.
- Constitution validation is pinned via `.github/workflows/constitution.yml`.
- Dependabot auto-merges GitHub Actions minor and patch updates.

### Fixed
- `server.json` said 0.1.0 and the Containerfile label 0.2.0 through the 0.6.0
  release; both carry the release version again.

## [0.6.0] - 2026-09-25

### Added
- **`openrouter` vision backend** (RT #1505). Captions images and videos through
  OpenRouter chat completions, defaulting to `google/gemini-3.1-flash-lite` with
  `google/gemini-3.8-flash` as the fallback model. Every request asks for
  zero-retention providers that do not collect data. The key comes from
  `OPENROUTER_API_KEY_FILE` (preferred) or `OPENROUTER_API_KEY`.

## [0.5.1] - 2026-03-21

Three fixes for an 8-hour indexing hang.

### Fixed
- **Skipped files never had their mtime written back to the DB**, so the
  mtime+size fast-skip path added in 0.5.0 was unreachable for files indexed
  before it existed. `_partition_unchanged` and `index_file` now backfill mtime
  on skip.
- **Run counters stayed at 0 during indexing** because `finish_run` only fired at
  the end. `update_run_progress` now runs after partition and after each
  extraction batch, so a long run reports live progress.
- **Vision API calls had no timeout** and hung indefinitely against an
  unresponsive endpoint. Added `TROVE_VISION_TIMEOUT` (default 120s), with a 3x
  safety net on the `extract_bounded` wrapper.

## [0.5.0] - 2026-03-21

### Added
- **mtime+size fast skip.** Indexing 35K files (214GB) took 34+ minutes even when
  every file was already indexed, because `_check_unchanged()` computed SHA-256
  on all of them. `stat()` mtime and size are now checked first; when both match
  the DB record the checksum is skipped entirely.

### Fixed
- Bumped `trivy-action` 0.34.1 → 0.35.0 (the pinned tag did not exist) and
  cleared the gourmand violations blocking CI.

## [0.4.0] - 2026-03-21

### Added
- **Per-file error tracking and the `trove_quality` tool.** Individual file
  failures during indexing are recorded in a new `index_errors` table instead of
  being left to grep out of stderr. Errors are classified transient (retryable) or
  permanent, and are marked resolved automatically when a file re-indexes
  successfully.

## [0.3.1] - 2026-03-19

### Fixed
- **Memory climbed monotonically across batches.** CPython's pymalloc holds freed
  heap pages, so RSS grew past 4G even with batched processing. `gc.collect()` now
  breaks reference cycles and `malloc_trim(0)` returns free pages to the OS
  between batches, keeping RSS plateaued around 1.5-2G.

## [0.3.0] - 2026-03-17

### Changed
- Synced the version with the container images (Quay.io/GHCR).

## [0.2.0] - 2026-03-15

Pluggable vision backends for indexing images and videos via AI captioning APIs.

### Added
- **Vision backends**: Gemini (primary), OpenAI, and Ollama support via the
  `TROVE_VISION_BACKEND` env var, with `TROVE_VISION_MODEL` and
  `TROVE_VISION_PROMPT` overrides.
- **EXIF metadata extraction**: dates, camera model, GPS coordinates, and image
  dimensions extracted from image files via Pillow — searchable without API
  calls.
- **Image support**: JPG, JPEG, PNG, GIF, HEIC, HEIF, WebP, BMP, TIFF.
- **Video support**: MP4, MOV, AVI, WebM, MKV (Gemini only — OpenAI and Ollama
  are image-only).
- Install extras `[vision-gemini]` and `[vision]`.

### Changed
- **Opt-in design**: the default `TROVE_VISION_BACKEND=none` preserves existing
  behavior.
- Container image now includes vision dependencies.
- Removed video extensions from the default exclude patterns.

### Removed
- Python 3.10 support (onnxruntime 1.24.3 requires 3.11+).

## [0.1.0] - 2026-03-14

No GitHub Release; no authored release notes exist to back-fill from.
