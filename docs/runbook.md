# mcp-trove runbook

How to keep a Trove index healthy, written for whoever is on call: a person,
or an agent acting for one. Paths are the container's (`/data` is the
database volume); on the Crunchtools host that volume is
`/srv/mcp-trove.crunchtools.com/data`.

## Where to look

| What | Where |
|------|-------|
| Last nightly run, path by path | `/data/last_run.json` (`TROVE_STATUS_FILE`) |
| Every run, with counts and errors | `trove_log` tool, or the `index_runs` table |
| Files that failed and why | `trove_quality` tool, or the `index_errors` table |
| Daily database copies | `/data/backups/trove-YYYYMMDD.db` (`TROVE_BACKUP_DIR`, newest 3) |
| The unit's own log | `journalctl -u mcp-trove-index` |

`last_run.json` has `status` (`ok` or `failed`), a `problems` list for
run-wide failures (database check, backup, mount), and per path `status`
(`ok`, `aborted`, `failed`, `errors`, `missing`) with found, indexed,
skipped, errored, deferred and removed counts. `vision_calls` against
`vision_budget` shows what the run spent.

## Monitoring states

The Nagios `Index run` check (`check_trove_run.sh`) reads `last_run.json`.

| State | Meaning | What to do |
|-------|---------|------------|
| CRITICAL: no summary | No `--index` run has ever finished | Start one: `systemctl start mcp-trove-index` |
| CRITICAL: finished over 36 h ago | The timer did not fire, or runs keep dying before the summary | `systemctl list-timers mcp-trove-index.timer`; `journalctl -u mcp-trove-index -n 100`. A reboot during the run leaves no summary; the next night fixes it |
| CRITICAL: status failed, path `aborted` | A safety gate stopped the path: the source looked unmounted (far fewer files than indexed), a manifest was cut short, or reads kept timing out | Check the mount (`mountpoint /var/mnt/pcloud`, `ls` a tree). If the files really were deleted, run `trove_reindex` on the path. An inbox that empties by design belongs in `TROVE_INBOX_PATHS` |
| CRITICAL: path `errors` | Over 10% of a path's files failed | `trove_quality` for the path; the messages say whether it is one bad format, the vision provider (quota, 402, 429) or the source |
| CRITICAL: a `problems` entry | `quick_check` failed, the backup failed, or the mount probe failed | Database: see Restore below. Mount: remount, then rerun |
| WARNING: errors over 5% | Many files failing, under the abort line | As for `errors` |
| WARNING: vision budget spent | The run hit `TROVE_VISION_MAX_CALLS_PER_RUN`; the rest was deferred, not failed | Expected while a new tree is backfilled. Otherwise something is re-captioning: check `vision_calls` over several nights |
| WARNING: backup over 36 h old | No new daily copy | Free space on the volume; `TROVE_BACKUP_KEEP` above 0 |

## Restore the database

1. Stop the server and the index: `systemctl stop mcp-trove.crunchtools.com mcp-trove-index`.
2. Keep the damaged file: `mv /data/trove.db /data/trove.db.damaged` (and its `-wal`, `-shm`).
3. Copy the newest good backup in: `cp /data/backups/trove-YYYYMMDD.db /data/trove.db`.
   An off-host copy is kept on pCloud under `Backups/trove/` if the local ones are gone.
4. Check it: `sqlite3 /data/trove.db 'PRAGMA quick_check'` prints `ok`.
5. Start the server, then run the index: files changed since the backup are
   picked up, captions included (they are cached in the database by content).

## Change the embedding model

Vectors from two models cannot be searched together, so the database records
its model and refuses to open under another.

1. Optionally measure first on a copy: `TROVE_DB=/tmp/copy.db TROVE_EMBEDDING_MODEL=<new> mcp-trove-crunchtools --reembed`, then `--eval queries.json` on both.
2. Set `TROVE_EMBEDDING_MODEL` for the indexer.
3. Run `mcp-trove-crunchtools --reembed`. It rebuilds every vector from the
   stored chunks (nothing is re-read or re-captioned), resumes if killed, and
   swaps the new vectors in with one transaction.
4. Restart the MCP server with the same `TROVE_EMBEDDING_MODEL`.

Do not run `--reembed` and `--index` at the same time.

## Measure search

`mcp-trove-crunchtools --eval queries.json` scores search against real
queries: a JSON list of `{"query": "...", "expect": ["path fragment", ...]}`.
It prints recall@5 and MRR. Keep the queries file next to the deployment,
not in this repository: it names personal files.

## Common changes

| Change | How |
|--------|-----|
| Index a new tree | Add it to `TROVE_PATHS`. The first run indexes it, deferring images past the vision budget to later nights |
| A folder that empties by design (an upload inbox) | Add it to `TROVE_INBOX_PATHS` too |
| Stop indexing a folder | Add its name to `TROVE_EXCLUDE_PATTERNS`; its entries are removed on the next run |
| Read another kind of file | Add its category to `TROVE_INDEX_TYPES`; files with name-only entries are read on the next run |
| A file keeps failing | It is skipped until it changes. After fixing the cause, `trove_reindex` on its path retries it |
| Re-read everything | `trove_reindex` with `force=true`. Captions are reused unless `TROVE_VISION_MODEL` changed |
| Spend less on captions | Lower `TROVE_VISION_MAX_CALLS_PER_RUN`; deferred files wait for later nights |
