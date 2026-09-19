# mcp-trove-crunchtools — Session Context

## Current State (2026-03-15)

**v0.1.0 built, tested, pushed to GitHub.** Not yet published to PyPI/Quay.

### What's Done
- All source code written and passing quality gates (ruff, mypy, 63 tests)
- GitHub repo created: https://github.com/crunchtools/mcp-trove
- Tag v0.1.0 pushed
- Multi-stage Containerfile working (Fedora 44 builder + Hummingbird runtime)
- Container smoke tested: all 8 MCP tools responding
- Live test: indexed 31 receipts (131 chunks) from Financial/Receipts, semantic search working
- Live test: FOSDEM 2026 photos (all JPGs) — trove correctly skips images (no extractor)

### Vision Backend Research (2026-03-15)
- **Moondream 0.5B (local CPU)**: Tested on FOSDEM photos. 23-112 seconds per image, caption quality poor (hallucinated locations, repetition loops on screenshots). Not viable for trove.
- **Gemini Flash (API)**: Tested same photos. 2-3 seconds per image, excellent quality — correctly identified Grand Place Brussels, read terminal text, read tram signs. Clear winner.
- **Architecture decision**: Pluggable vision backend via `TROVE_VISION_BACKEND=gemini|openai|ollama|moondream|exif|none`. Trove calls APIs directly (not through MCP servers). Image gets captioned at index time, caption stored as chunk text.
- Moondream 0.0.5 installed at `~/.local/lib/python3.12/site-packages/moondream/`, model weights at `~/.cache/moondream/moondream-0_5b-int8.mf` (661 MiB). Can be cleaned up.

### HomeDirectories Backup Module (2026-03-15)
Built and deployed `backup_HomeDirectories` module in PersonalBackups.sh:
- Auto-discovers users from /etc/passwd
- SQLite databases get safe `sqlite3 .backup` before sync
- Uses `rclone copy` (not sync) for SQLite overlay to avoid deleting main backup
- Trailing-slash excludes (`--exclude "path/"`) prevent empty directory stubs
- Skip list covers ~150G of regenerable content, backs up ~62 MB of critical config
- Systemd drop-in overrides at `/etc/systemd/system/PersonalBackups-{Weekly-1,Monthly-1,Monthly-2}.service.d/override.conf`
- First automated run: Friday March 20, 2026 at 6 AM
- Monthly Checklist wiki updated to verify all pCloud backup destinations

### What's Left for v0.1.0
1. **PyPI Trusted Publishing** — add pending publisher at https://pypi.org/manage/account/publishing/
   - PyPI project: `mcp-trove-crunchtools`
   - Owner: `crunchtools`
   - Repository: `mcp-trove`
   - Workflow: `publish.yml`
   - Environment: (leave blank)

2. **Quay.io Repository** — create `crunchtools/mcp-trove` at https://quay.io/new/?namespace=crunchtools

3. **GitHub Release** — create release from v0.1.0 tag to trigger PyPI publish

4. **Deploy** — `/deploy-mcp-server` to breetai/lotor

5. **Fleet watchdog** — add to fleet-watchdog.py, Nagios check + service definition

6. **MCP Registry** — publish to registry.modelcontextprotocol.io

7. **WordPress** — update crunchtools.com/software/mcp-servers/ page

8. **Memory** — store final build details

### What's Left for v0.2.0
1. **Vision backend** — pluggable image captioning (gemini, openai, ollama)
2. **EXIF extraction** — date, GPS, camera metadata for photos (no AI needed)

### Backup Gaps Identified
- **Documents/Downloads/Autosync** rotations on pCloud are 9 months stale (last June 2025)
- **Lotor server dumps** (MediaWiki my_wiki.sql, RT rt4.sql) exist locally at /srv/*/data/backups/ with nightly cron, but nothing syncs them to pCloud
- **ROTV** backup status on production unknown — needs checking

### MCP Server Config (already added to project .claude.json)
```json
{
  "mcpServers": {
    "mcp-trove-crunchtools": {
      "command": "/var/home/fatherlinux/Projects/crunchtools/mcp-trove/.venv/bin/python",
      "args": ["-m", "mcp_trove_crunchtools"]
    }
  }
}
```

### Key Architecture Notes
- Hummingbird Python image has Python 3.14, no package manager, runs as uid 65532 (HOME=/tmp)
- `py-rust-stemmers` (fastembed dep) needs gcc to compile on 3.14 — solved via multi-stage build
- Default container DB path: `/tmp/trove-data/trove.db` (non-root writable)
- Default local DB path: `~/.local/share/mcp-trove/trove.db`
- Trove DB is now backed up to pCloud via HomeDirectories module
