# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and this project adheres to
[Semantic Versioning](https://semver.org/).

Entries prior to 2026-09-19 are back-filled from GitHub Release notes (RT #1484),
except `0.3.1` through `0.5.1`, which had no release to back-fill from and were
reconstructed from their commit ranges in RT #1485.

## [Unreleased]

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
