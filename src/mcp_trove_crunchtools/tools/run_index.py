"""The unattended --index run: safety checks, per-path indexing, run summary.

Built to be left alone for years (RT #1542): every way a run can fail ends in
a non-zero exit and a summary that says why, never a quiet success.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import __version__, vision
from .. import database as db
from ..config import get_config
from ..errors import IndexAbortedError
from .index import trove_index

if TYPE_CHECKING:
    from collections.abc import Coroutine

MOUNT_PROBE_SECONDS = 30
MAX_ERROR_RATIO = 0.10  # errored share of found files that fails a path


def run_index() -> int:
    """Index configured directories (CLI mode for systemd timer).

    Returns the process exit code: 0 only when every path completed with a
    sane error rate. Anything else is non-zero so the timer's unit fails and
    monitoring sees it, rather than a quiet "0 indexed" success.
    """
    config = get_config()
    summary: dict[str, Any] = {
        "app_version": __version__,
        "started_at": _now(),
        "status": "ok",
        "problems": [],
        "paths": [],
    }
    started = time.monotonic()

    vision.reset_budget()
    try:
        code = _preflight(summary)
        if code is None:
            code = 0 if not summary["problems"] else 1
            for dir_path in config.index_paths:
                entry: dict[str, Any] = {"path": dir_path, "status": "ok"}
                summary["paths"].append(entry)
                if not _index_one(dir_path, entry):
                    code = 1
    except Exception as exc:
        # Whatever broke, the summary still has to say so: monitoring would
        # otherwise keep reading the previous run's result. The traceback
        # goes to the journal for whoever has to diagnose it.
        traceback.print_exc()
        code = _report(summary, f"run failed: {exc}")

    summary["vision_calls"] = vision.calls_made()
    summary["vision_budget"] = config.vision_max_calls
    summary["finished_at"] = _now()
    summary["duration_seconds"] = round(time.monotonic() - started)
    if code:
        summary["status"] = "failed"
    if not _write_summary(config.status_file, summary):
        return 1
    return code


def _report(summary: dict[str, Any], problem: str) -> int:
    """Record a problem that fails the run, and return the failing exit code."""
    summary["problems"].append(problem)
    print(problem, file=sys.stderr)
    return 1


def _preflight(summary: dict[str, Any]) -> int | None:
    """Check the database, back it up and probe the source.

    Returns an exit code to stop with, or None to go on indexing. Problems
    that should fail the run without stopping it are added to the summary.
    """
    config = get_config()
    db.get_db()

    stale = db.abort_stale_runs()
    if stale:
        print(f"Marked {stale} abandoned runs as aborted")

    check = db.quick_check()
    if check != "ok":
        return _report(summary, f"index database failed PRAGMA quick_check: {check}")

    backup_problem = _backup(config.db_path, config.backup_dir, config.backup_keep)
    if backup_problem:
        _report(summary, backup_problem)
    summary["latest_backup"] = _latest_backup(config.backup_dir)
    pruned = db.prune_captions()
    if pruned:
        print(f"Pruned {pruned} cached captions no longer in the index")

    if not config.index_paths:
        print("No TROVE_PATHS configured. Nothing to index.")
        return 1 if backup_problem else 0

    if config.require_mount:
        mount_problem = _probe_mount(config.require_mount)
        if mount_problem:
            return _report(summary, mount_problem)
    return None


def _index_one(dir_path: str, entry: dict[str, Any]) -> bool:
    """Index one configured path, recording the outcome in entry.

    Returns False when the path failed, aborted or had too many errors.
    """
    if not Path(dir_path).is_dir():
        entry["status"] = "missing"
        print(f"Skipping non-directory: {dir_path}", file=sys.stderr)
        return False
    try:
        result = _run_async(trove_index(dir_path, new_vision_budget=False))
    except IndexAbortedError as exc:
        entry.update(status="aborted", message=str(exc))
        print(str(exc), file=sys.stderr)
        return False
    except Exception as exc:
        entry.update(status="failed", message=str(exc))
        print(f"{dir_path}: indexing failed: {exc}", file=sys.stderr)
        return False

    indexed = result.get("files_indexed", 0)
    skipped = result.get("files_skipped", 0)
    errored = result.get("files_errored", 0)
    deferred = result.get("files_deferred", 0)
    removed = result.get("files_removed", 0)
    chunks = result.get("total_chunks", 0)
    found = indexed + skipped + errored + deferred
    entry.update(
        files_found=found,
        files_indexed=indexed,
        files_skipped=skipped,
        files_errored=errored,
        files_deferred=deferred,
        files_removed=removed,
        total_chunks=chunks,
    )
    print(f"{dir_path}: {indexed} indexed, {skipped} skipped, {chunks} chunks")
    if errored > found * MAX_ERROR_RATIO:
        entry["status"] = "errors"
        print(f"{dir_path}: {errored} of {found} files failed", file=sys.stderr)
        return False
    return True


def _run_async(coro: Coroutine[Any, Any, dict[str, Any]]) -> dict[str, Any]:
    """Run coro on a fresh event loop, without waiting on stuck worker threads.

    asyncio.run() joins the default executor on the way out, so one file read
    stuck in the kernel on a hung mount would block the run there for good.
    loop.close() shuts the executor down without waiting; the CLI then exits
    with os._exit() and the stuck thread goes with the process.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _probe_mount(mount: str) -> str | None:
    """Return a problem if mount is not mounted or does not answer a listing.

    The listing runs on a daemon thread: on a wedged FUSE mount it blocks in
    the kernel and cannot be interrupted, so it is abandoned, not joined.
    """
    if not os.path.ismount(mount):
        return f"{mount} is not mounted"
    listed = threading.Event()
    failure: list[str] = []

    def probe() -> None:
        try:
            os.listdir(mount)
        except OSError as exc:
            failure.append(str(exc))
        listed.set()

    threading.Thread(target=probe, daemon=True).start()
    if not listed.wait(MOUNT_PROBE_SECONDS):
        return f"{mount} did not answer a listing within {MOUNT_PROBE_SECONDS}s; it may be hung"
    if failure:
        return f"{mount} cannot be listed: {failure[0]}"
    return None


def _backup(db_path: str, backup_dir: str, keep: int) -> str | None:
    """Keep one consistent copy of the database per day, the newest `keep`.

    Returns a problem description, or None. A failed backup does not stop
    indexing, but it does fail the run so it gets noticed.
    """
    if keep <= 0 or db_path == ":memory:":
        return None
    directory = Path(backup_dir)
    target = directory / f"trove-{datetime.now(UTC):%Y%m%d}.db"
    try:
        # Backups hold every indexed chunk, so only the owner may read them.
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
        if not target.exists():
            partial = target.with_suffix(".db.partial")
            partial.unlink(missing_ok=True)
            db.backup_to(str(partial))
            partial.chmod(0o600)
            partial.replace(target)
        for old in sorted(directory.glob("trove-*.db"))[:-keep]:
            old.unlink()
    except Exception as exc:
        return f"backup to {target} failed: {exc}"
    return None


def _latest_backup(backup_dir: str) -> dict[str, str] | None:
    """Name and time of the newest backup, for monitoring that cannot list it.

    The backup directory is owner-only; the summary is how a monitor without
    access learns whether backups are still being taken.
    """
    backups = sorted(Path(backup_dir).glob("trove-*.db"))
    if not backups:
        return None
    newest = backups[-1]
    taken = datetime.fromtimestamp(newest.stat().st_mtime, UTC)
    return {"file": newest.name, "created_at": taken.isoformat(timespec="seconds")}


def _write_summary(status_file: str, summary: dict[str, Any]) -> bool:
    """Write the run summary that monitoring reads, replacing it atomically.

    Returns False when it cannot be written: monitoring would then judge the
    run by a stale summary, so the caller fails the run instead.
    """
    target = Path(status_file)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # A fresh, exclusively created file: never follows a planted symlink.
        fd, partial = tempfile.mkstemp(dir=target.parent, prefix=".last_run.")
        with os.fdopen(fd, "w") as out:
            out.write(json.dumps(summary, indent=2) + "\n")
        Path(partial).chmod(get_config().status_file_mode)
        Path(partial).replace(target)
    except OSError as exc:
        print(f"Cannot write {target}: {exc}", file=sys.stderr)
        return False
    return True
