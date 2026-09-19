# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and this project adheres to
[Semantic Versioning](https://semver.org/).

Entries prior to 2026-09-19 are back-filled from GitHub Release notes (RT #1484).
Only `v0.2.0` and `v0.3.0` have GitHub Releases; the other five tags do not, so
there are no authored notes to back-fill for them.

## [Unreleased]

## [0.5.1] - 2026-03-21

No GitHub Release; no authored release notes exist to back-fill from.

## [0.5.0] - 2026-03-21

No GitHub Release; no authored release notes exist to back-fill from.

## [0.4.0] - 2026-03-21

No GitHub Release; no authored release notes exist to back-fill from.

## [0.3.1] - 2026-03-19

No GitHub Release; no authored release notes exist to back-fill from.

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
