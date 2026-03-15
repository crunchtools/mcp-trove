# mcp-trove-crunchtools Constitution

> **Version:** 1.0.0
> **Ratified:** 2026-03-14
> **Status:** Active
> **Inherits:** [crunchtools/constitution](https://github.com/crunchtools/constitution) v1.0.0
> **Profile:** MCP Server

This constitution establishes the core principles, constraints, and workflows that govern all development on mcp-trove-crunchtools.

---

## I. Core Principles

### 1. Five-Layer Security Model

Every change MUST preserve all five security layers. No exceptions.

**Layer 1 — Credential Protection:**
- N/A — This server has no external API credentials
- No tokens, API keys, or passwords are required or stored
- The server indexes local files and stores embeddings locally
- SecretStr pattern available in config.py if credentials are added in future

**Layer 2 — Input Validation:**
- Pydantic models enforce strict data types with `extra="forbid"`
- File paths canonicalized and validated
- Search queries length-limited
- Pagination parameters bounded

**Layer 3 — File System Hardening:**
- Read-only access to indexed files (no writes)
- Path canonicalization to prevent traversal attacks
- File size limits to prevent memory exhaustion
- Exclude patterns for binary and large files
- Batch size limits on indexing operations

**Layer 4 — Dangerous Operation Prevention:**
- No shell execution or code evaluation
- No `eval()`/`exec()` functions
- asyncio.Semaphore for resource limiting
- Concurrent embedding workers capped

**Layer 5 — Supply Chain Security:**
- Weekly automated CVE scanning via GitHub Actions
- Hummingbird container base images (minimal CVE surface)
- Gourmand AI slop detection gating all PRs

### 2. Two-Layer Tool Architecture

Tools follow a strict two-layer pattern:
- `server.py` — `@mcp.tool()` decorated functions that validate args and delegate
- `tools/*.py` — Pure async functions that call database, embedder, or indexer

Never put business logic in `server.py`. Never put MCP registration in `tools/*.py`.

### 3. Self-Contained Operation

The server MUST work without any external service accounts:
- `TROVE_DB` configurable (default: `~/.local/share/mcp-trove/trove.db`)
- SQLite database with sqlite-vec and FTS5, auto-created on first run
- fastembed ONNX runtime for embeddings (no PyTorch dependency)
- No authentication required — reads local files only

### 4. Three Distribution Channels

Every release MUST be available through all three channels simultaneously:

| Channel | Command | Use Case |
|---------|---------|----------|
| uvx | `uvx mcp-trove-crunchtools` | Zero-install, Claude Code |
| pip | `pip install mcp-trove-crunchtools` | Virtual environments |
| Container | `podman run quay.io/crunchtools/mcp-trove` | Isolated, systemd |

### 5. Three Transport Modes

The server MUST support all three MCP transports:
- **stdio** (default) — spawned per-session by Claude Code
- **SSE** — legacy HTTP transport
- **streamable-http** — production HTTP, systemd-managed containers

### 6. Semantic Versioning

Follow [Semantic Versioning 2.0.0](https://semver.org/) strictly.

**MAJOR** (breaking changes — consumers must update):
- Removed or renamed tools
- Changed tool parameter names or types
- Renamed environment variables
- Changed default behavior of existing tools

**MINOR** (new functionality — backwards compatible):
- New tools added
- New optional parameters on existing tools
- New tool groups

**PATCH** (fixes — no functional change):
- Bug fixes in existing tools
- Test additions or improvements
- Security patches (dependency updates)

**No version bump required** (infrastructure, not shipped):
- CI/CD changes (workflows, gourmand config)
- Documentation (README, CLAUDE.md, SECURITY.md)
- Issue templates, pre-commit config
- Governance files (.specify/)

**Version bump happens at release time, not per-commit.** Multiple commits can accumulate between releases. The version in `pyproject.toml` and `server.py` is bumped when cutting a release tag.

### 7. AI Code Quality

All code MUST pass Gourmand checks before merge. Zero violations required.

---

## II. Technology Stack

| Layer | Technology | Version |
|-------|------------|---------|
| Language | Python | 3.10+ |
| MCP Framework | FastMCP | Latest |
| Database | SQLite with sqlite-vec + FTS5 | Built-in |
| Embeddings | fastembed (ONNX runtime) | Latest |
| PDF Extraction | pymupdf4llm | Latest |
| DOCX Extraction | python-docx | Latest |
| Validation | Pydantic | v2 |
| Container Base | Hummingbird | Latest |
| Package Manager | uv | Latest |
| Build System | hatchling | Latest |
| Linter | ruff | Latest |
| Type Checker | mypy (strict) | Latest |
| Tests | pytest + pytest-asyncio | Latest |
| Slop Detector | gourmand | Latest |

---

## III. Testing Standards

### In-Memory SQLite Tests (MANDATORY)

Every tool MUST have a corresponding test using in-memory SQLite. Tests use `:memory:` databases — no disk I/O, no cleanup required, fast CI execution.

**Pattern:**
1. Create an in-memory SQLite database with the schema applied
2. Mock the fastembed model to return deterministic vectors
3. Seed test data (files, chunks) as needed
4. Call the tool function directly (not the `_tool` wrapper)
5. Assert response structure and values

**Required test classes per tool group:**

| Tool Group | Test Class | Minimum Tests |
|------------|-----------|---------------|
| Search tools | `TestSearchTools` | Search, similar |
| Index tools | `TestIndexTools` | Index, reindex, remove |
| Status tools | `TestStatusTools` | Status, list, get_chunks |
| Error cases | `TestErrorHandling` | Missing file, invalid path |

**Database reset:** Each test gets a fresh in-memory database to prevent state leakage between tests.

**Tool count assertion:** `test_tool_count` MUST be updated whenever tools are added or removed. This catches accidental regressions.

### Input Validation Tests

Every Pydantic model in `models.py` MUST have tests in `test_validation.py`:
- Valid minimal input
- Valid full input
- Invalid/rejected inputs (empty strings, too-long values, extra fields)

---

## IV. Gourmand (AI Slop Detection)

All code MUST pass `gourmand --full .` with **zero violations** before merge. Gourmand is a CI gate in GitHub Actions.

### Configuration

- `gourmand.toml` — Check settings, excluded paths
- `gourmand-exceptions.toml` — Documented exceptions with justifications
- `.gourmand-cache/` — Must be in `.gitignore`

### Exception Policy

Exceptions MUST have documented justifications in `gourmand-exceptions.toml`. Acceptable reasons:
- Standard API patterns (HTTP status codes, pagination params)
- Test-specific patterns (intentional invalid input)
- Framework requirements (CLAUDE.md for Claude Code)
- Database/embedding patterns (vector similarity terms, index configuration)

Unacceptable reasons:
- "The code is special"
- "The threshold is too strict"

---

## V. Code Quality Gates

Every code change must pass through these gates in order:

1. **Lint** — `uv run ruff check src tests`
2. **Type Check** — `uv run mypy src`
3. **Tests** — `uv run pytest -v` (all passing, in-memory SQLite)
4. **Gourmand** — `gourmand --full .` (zero violations)
5. **Container Build** — `podman build -f Containerfile .`

---

## VI. Naming Conventions

| Context | Name |
|---------|------|
| GitHub repo | `crunchtools/mcp-trove` |
| PyPI package | `mcp-trove-crunchtools` |
| CLI command | `mcp-trove-crunchtools` |
| Python module | `mcp_trove_crunchtools` |
| Container image | `quay.io/crunchtools/mcp-trove` |
| systemd service | `mcp-trove.service` |
| HTTP port | 8020 |
| License | AGPL-3.0-or-later |

---

## VII. Development Workflow

### Adding a New Tool

1. Add the async function to the appropriate `tools/*.py` file
2. Export it from `tools/__init__.py`
3. Import it in `server.py` and register with `@mcp.tool()`
4. Add an in-memory SQLite test in `tests/test_tools.py`
5. Update the tool count in `test_tool_count`
6. Run all five quality gates
7. Update CLAUDE.md tool listing

---

## VIII. Governance

### Amendment Process

1. Create a PR with proposed changes to this constitution
2. Document rationale in PR description
3. Require maintainer approval
4. Update version number upon merge

### Ratification History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-03-14 | Initial constitution |
