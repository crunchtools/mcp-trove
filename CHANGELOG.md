# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and this project adheres to
[Semantic Versioning](https://semver.org/).

Entries prior to 2026-09-19 are back-filled from GitHub Release notes (RT #1484),
except `0.3.1` through `0.5.1`, which had no release to back-fill from and were
reconstructed from their commit ranges in RT #1485.

## [Unreleased]

## [1.2.4] - 2026-11-11

### Changed

- No functional change. First release on the build-once container pipeline (RT#1550): `:latest` now names a digest that passed the release gates.

## [1.2.3] - 2026-11-10

### Fixed

- **The shrink-abort message named a remedy that cannot work.** When a
  directory holds under 50% of the files indexed under it, the run aborts
  and told the operator to "run trove_reindex on the path if the files
  really were removed." A reindex skips the gate but never clears it:
  pruning entries for files that are gone happens in `_remove_missing`,
  which only runs from a manifest, and the MCP server has no
  `TROVE_MANIFEST_DIR` — only the nightly `--index` job does. So
  `trove_reindex` reported `files_removed: 0`, left every stale row in
  place, and the next nightly run aborted on the same baseline. Found on
  gloval after `/pcloud/Autosync/Screenshots` went from 129 files to 7: the
  index check sat CRITICAL for eight hours and the documented fix moved
  nothing. The message now names the remedy that works — `trove_remove` on
  the path, then `trove_index` to rebuild it.

### Fixed

- **The locked requirements install could still re-resolve.** The builder
  installed `requirements.txt` (the frozen, hash-checked export of
  `uv.lock`) with `--require-hashes` but without `--no-deps`, so pip was
  still free to walk each package's own dependency metadata and pull in
  whatever satisfied it at build time, rather than exactly what the lock
  pinned. The project's own `.` install already carried `--no-deps`; the
  lockfile install now does too, so `--require-hashes` verifies the pinned
  set and `--no-deps` is what actually keeps pip from re-resolving past it.
  Same class of drift as the mcp-gemini outage (RT #1564): a build silently
  shipping newer-than-tested dependency versions.

## [1.2.1] - 2026-11-09

### Fixed

- **An openai-SDK error body was still unbounded.** 1.2.0 read through
  `with_streaming_response`, which bounds the success path — but on a 4xx or
  5xx the SDK reads the body to completion inside `create()`
  (`_base_client.py`: `if not err.response.is_closed: err.response.read()`)
  before anything is handed back, so the 4MB ceiling never saw it. An error
  body is the likeliest one to be enormous, or not to be from the API at all.

  Both openai-backed clients now carry the existing response hook on
  `openai.DefaultHttpxClient`, which is the SDK's own export and subclasses
  whichever httpx flavour the SDK was built on — so the hook lands on the
  right one without this module guessing, which was the reason 1.2.0 avoided
  passing a client at all. `with_streaming_response` stays: it is still what
  lets the success path be measured before the copy.

  Three tests, checked against the unhooked client to confirm they fail on it:
  an oversized streamed error body and an oversized declared length are both
  refused with the byte limit named, and a small 429 still surfaces the SDK's
  own status message — the hook bounds, it does not swallow.

### Changed

- The `vision-openai` and `vision` extras require `openai>=1.17`, the first
  release that exports `DefaultHttpxClient` (bisected, not assumed: 1.16.2
  does not have it).

## [1.2.0] - 2026-11-09

### Fixed

- **Three of the four vision backends went out nameless and unbounded.** The
  sweep that landed in 1.1.0 fixed the Ollama path in this module and left the
  OpenAI, OpenRouter and Gemini clients beside it untouched: each was
  constructed with no agent string and read its body inside the vendor SDK,
  where the 4MB ceiling could not reach it. `check_http_client` cannot see any
  of this — the trigger it keys on is an `httpx` import — so the gap passed the
  machine check and was caught by review (RT #1601, item I22).

  All three now send
  `User-Agent: mcp-trove-crunchtools/<version> (+<repo url>)`, with
  `Accept-Encoding: identity` so that the bytes counted are the bytes that
  arrive; a caption response is a few hundred bytes, so the compression was
  worth nothing and an honest limit is worth more.

  The two openai-SDK backends read through `with_streaming_response`, which
  hands the body back unread so the ceiling applies while reading it, and the
  caption is parsed from those bounded bytes rather than from the SDK's typed
  object — `.parse()` would read the body itself, with no limit. The Gemini
  backend passes a response hook through `http_options.client_args`, since
  google-genai builds its own transport; the hook rejects an oversized
  `content-length` early and wraps the stream in a byte counter otherwise.

  The stream wrapper is derived from the HTTP library that built the response,
  not from the `httpx` this module imports. The openai SDK moved from `httpx`
  to `httpx2` between 1.x and 3.x, which is exactly how a wrapper built on the
  wrong flavour fails the library's own `isinstance` check on close.

### Changed

- The vision suite drives a real SDK client over a `MockTransport` rather than
  a `MagicMock` scripting `chat.completions.create`. A mock of the call cannot
  see the agent string the client adds or the ceiling the read enforces, and
  the assertions now read the request off the wire, which is stronger than
  reading back the keyword arguments a mock was handed.

- `httpx2` is a declared dev dependency — it is the transport the openai SDK is
  built on, and taking it transitively is how it would silently change flavour.

## [1.1.0] - 2026-11-09

### Fixed

- **The Ollama vision backend read whatever was listening on port 11434.** It
  called `urllib.request.urlopen(...).read()` with no argument and no size
  limit, so the thing on the other end decided how much memory this process
  used. A caption is a sentence or two; a megabyte answer means something else
  is on that port. The body is now read in bounded pieces under a 4 MB cap,
  measured before each chunk is copied in.

- **An Ollama failure says what came back.** Every outcome arrived as
  `Ollama request failed: <exception>`, so a wrong port, a Grafana answering
  on 11434 and a missing model were one message. They now name the method, the
  URL, the status, the content-type and a short body snippet.

### Added

- **The client says what it is.** The Ollama request carries
  `User-Agent: mcp-trove-crunchtools/<version> (+<repo url>)`, with the version
  read from installed package metadata.

### Changed

- The Ollama backend uses `httpx` rather than `urllib.request`. httpx is the
  HTTP client the MCP Server profile names, it is what makes a bounded read
  expressible, and it is now a declared dependency rather than something
  arriving through fastmcp.

- Inherits constitution v1.36.0, whose MCP Server profile 1.8.0 states the
  outbound HTTP contract (RT #1601). This repo was missing from that ticket's
  survey, because the survey grepped for httpx, aiohttp and requests and this
  call used `urllib.request`.

## [1.0.8] - 2026-11-05

### Fixed

- `trove_remove` on a directory that no longer exists removed nothing, so a
  tree deleted or moved out from under the index could never be purged: the
  subtree delete needed the directory to still be on disk, and the nightly
  run only prunes trees it still indexes. It now clears the path's own entry
  and everything indexed beneath it, by an index range scan rather than LIKE.

## [1.0.7] - 2026-10-31

### Security

- Upgraded locked dependencies within their declared ranges to clear the
  advisories the fleet dependency audit found (anyio, authlib, certifi, cryptography, dnspython, fastmcp, filelock, fsspec, joserfc, lxml, opentelemetry-api, pillow, pyasn1, pydantic-settings, pygments, pyjwt, requests, urllib3).

## [1.0.6] - 2026-10-26

### Fixed

- Old camcorder videos (MJPEG/PCM AVIs and 3GPs) failed with "Operation not
  permitted": they keep no index for their sound, so seeking to the audio
  clip's start fails. The clip is now taken from the start of the file, and
  a refused frame seek keeps the frames already taken. Their recorded
  failures are cleared for a retry.

## [1.0.5] - 2026-10-25

### Fixed

- Every song with embedded cover art failed with "Operation not permitted".
  The art is a one-frame video stream marked as an attached picture, and
  seeking in it for frames fails with EPERM. It is no longer sampled; the
  song is captioned from its sound and tags. This, not the pCloud mount as
  1.0.4 supposed, was the cause of the refused songs; EPERM stays a
  transient failure, retried at most five times.

## [1.0.4] - 2026-10-25

### Fixed

- The pCloud mount refuses reads in bursts with "Operation not permitted",
  for files it serves fine minutes later; 1,413 songs in one run were
  recorded as permanent failures and would never have been retried. EPERM
  is now a transient failure, and migration 5 releases the recorded ones.

## [1.0.3] - 2026-10-25

### Fixed

- One file whose text could not be embedded failed the whole configured
  path it was in: an RTF with an emoji (striprtf writes it as two lone
  UTF-16 surrogates) aborted `/pcloud/Documents/Recreational` after 75
  minutes. Extracted text now has surrogate pairs rejoined and strays
  replaced, and an embedding failure is recorded against that file alone.
- Images over the 20 MB inline limit of the OpenRouter backend were a
  permanent failure. They are now captioned from a JPEG copy 2048 pixels on
  the long side.
- Migration 4 releases both kinds of recorded failure, so they are retried.

## [1.0.2] - 2026-10-25

### Fixed

- A vision provider refusing for want of credit (HTTP 402) was recorded as
  a permanent failure for the file, so every photo after the account ran dry
  was skipped for good. The run now stops captioning, defers the rest to the
  next run without recording errors, and sets `vision_out_of_credit` in the
  run summary. Topping up the account lets the next run carry on.
- Rate limits (HTTP 429) count as transient failures and are retried.

## [1.0.1] - 2026-10-25

### Fixed

- Two videos sampled at once, each a 4K decoder of about 200 MB, took the
  indexer past its 1.5 GB limit with images and a PDF in flight. Video and
  audio are now sampled one file at a time.
- MPEG program-stream (.mpg) and Ogg Theora files failed whole when the
  packets after a seek could not be decoded; those packets are skipped.

## [1.0.0] - 2026-10-25

The interface is now stable under SemVer: MCP tool names, parameters and
result fields, the command-line flags, every `TROVE_*` variable and the
database schema (see Stability in the README).

### Added

- docs/runbook.md: monitoring states, restoring a backup, changing the embedding
  model, measuring search and everyday changes.

### Removed

- `TROVE_INDEX_BATCH`, unused since 0.12.0.

## [0.16.0] - 2026-10-25

### Added

- Video and audio are captioned from samples instead of being sent whole:
  10 frames spread from 5% to 95% of a video, taken from the nearest
  keyframes, and a 20-second mono MP3 clip from a quarter of the way in, all
  in one OpenRouter request (images plus `input_audio`). Any length of file
  costs one call and a few ranged reads. Duration, recording time, phone GPS
  and audio tags are indexed alongside the caption. Videos over 20 MB, which
  always failed before, are retried once (migration 3).
- An `audio` category (mp3, m4a, wav, flac, ogg, aac, wma, opus), opt-in
  because it sends sound clips to the vision provider, and more video
  extensions (m4v, mpg, mpeg, 3gp, wmv).
- `TROVE_INDEX_LOCATION` (default `false`): index the GPS of photos and
  videos. Photo GPS was always indexed before; set this to keep it.
- `--reembed`: rebuilds every vector from the stored chunks under
  `TROVE_EMBEDDING_MODEL`, resumable, with an atomic swap; nothing is
  re-extracted or re-captioned.
- `--eval QUERIES_JSON`: recall@5 and MRR for a list of real queries and the
  path fragments a right answer contains.

### Changed

- `trove_search` fuses three ranked lists by reciprocal rank (k=60): vector
  similarity, bm25 keywords, and file paths that hold every query term
  (weighted double). Before, keyword hits were appended after the vector
  hits and dropped whenever those filled the limit. Results are one per
  file, its best chunk, with `score`, `sources` and `matches`; the
  `source`, `vector_distance` and `fts_score` fields are gone.
- Video and audio files are keyed by a fingerprint of their size and first
  and last megabyte, not a hash of every byte, so checking one does not
  download it.

### Fixed

- A search scoped to a path ignored the scope for keyword hits, could
  return nothing when the scope's chunks were not among the 30 nearest
  overall, and treated `_` and `%` in the path as wildcards.
- Query punctuation no longer drops keyword matching: terms are quoted.

## [0.15.0] - 2026-10-25

### Added

- `TROVE_INDEX_TYPES`: an allowlist of content categories (images, videos,
  pdf, text, documents, spreadsheets, presentations, web, data, code), in
  place of indexing every text-like extension. The default reads the first
  seven; web pages, data files and code are opt-in.
- `TROVE_INDEX_OTHER` (default `name`): every file outside the enabled
  categories, and any over 50 MB, is indexed by its name, folder and
  extension without being read or hashed, so it can still be found. `skip`
  leaves such files out.
- Readers for Word 97-2003 `.doc` (its piece table, read with olefile; no
  converter binary), OpenDocument text, spreadsheets and presentations,
  `.xlsx` and `.xls` (first 200 rows of each sheet), `.pptx` (slide text and
  notes), `.rtf`, and the first 200 lines of `.csv`/`.tsv`.
- A file is re-read when its category is switched on or off, and entries
  for files the configuration no longer indexes in any form are removed;
  images and videos are never removed this way.

### Removed

- `TROVE_INDEX_SOURCE_CODE` (0.14.0): add `code` to `TROVE_INDEX_TYPES`.

### Fixed

- The index's own database is never indexed when it sits inside an indexed
  tree.

## [0.14.1] - 2026-10-25

### Fixed

- A password-protected PDF fails with "PDF is password-protected" instead
  of pymupdf4llm's "'NoneType' object is not subscriptable" (#81).

## [0.14.0] - 2026-10-25

### Changed

- Source code, configuration and log files (`.py`, `.c`, `.h`, `.sh`, `.sql`,
  `.toml`, `.ini`, `.conf`, `.env`, `.log` and the like) are no longer indexed
  unless `TROVE_INDEX_SOURCE_CODE` is set. They are not documents, they are
  slow to embed, and a `.env` or `.conf` can hold secrets. Entries already
  indexed are removed on the next run.
- `TROVE_EXCLUDE_PATTERNS` also matches directory names, so a pattern can
  skip a whole tree; the walk does not descend into it. Indexed entries a
  pattern now matches are removed on the next run.
- The shrink gate's baseline is the number of files indexed under the path,
  not the last run's count, so files removed by a configuration change do
  not look like a dead mount.

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
