"""Source manifests: a listing of an index path taken from the storage API.

Walking a network-backed tree through a FUSE mount costs one listing per
directory and one stat per file, each a network round trip, and every
restart repeats it (RT #1542). The storage API can list a whole tree in one
call: `rclone lsjson -R --fast-list --files-only` lists 5,600 pCloud files in
3 s. The host writes that listing to TROVE_MANIFEST_DIR before a run, and the
scan reads paths, sizes and times from it instead of the mount. Files are
still read through the mount, only when they need indexing.

The manifest is rclone's lsjson format, a JSON array of objects with Path
(relative to the index path), Size and ModTime (RFC 3339). One file per index
path, at that path under TROVE_MANIFEST_DIR with .lsjson appended:
/pcloud/Documents/Academic -> <dir>/pcloud/Documents/Academic.lsjson.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

from .config import get_config

logger = logging.getLogger(__name__)

# A listing older than this no longer describes the tree; the scan walks it.
MANIFEST_MAX_AGE_HOURS = 6
# About 500,000 files at ~200 bytes an entry; a bigger listing is not parsed
# into memory, and the scan walks the tree.
MANIFEST_MAX_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class Entry:
    """What the storage API reports for one file."""

    size: int
    mtime: float


def manifest_file(directory: Path) -> Path | None:
    """Where the manifest for an index path lives, or None if manifests are off.

    The index path is mirrored under the manifest directory, so two index
    paths can never share a listing.
    """
    manifest_dir = get_config().manifest_dir
    if not manifest_dir:
        return None
    return Path(manifest_dir) / f"{str(directory).strip('/')}.lsjson"


def load_manifest(directory: Path) -> dict[Path, Entry] | None:
    """Return {absolute path: Entry} for directory, or None to walk the tree instead.

    None when manifests are off, the file is missing or older than
    MANIFEST_MAX_AGE_HOURS, or it cannot be parsed; each case but the first
    is logged, since the run then falls back to the slow walk.
    """
    source = manifest_file(directory)
    if source is None:
        return None
    if not source.exists():
        logger.warning("No manifest at %s; walking %s", source, directory)
        return None
    source_stat = source.stat()
    if source_stat.st_size > MANIFEST_MAX_BYTES:
        logger.warning(
            "Manifest %s is %d bytes, over %d; walking %s",
            source,
            source_stat.st_size,
            MANIFEST_MAX_BYTES,
            directory,
        )
        return None
    age_hours = (time.time() - source_stat.st_mtime) / 3600
    if age_hours > MANIFEST_MAX_AGE_HOURS:
        logger.warning("Manifest %s is %.1fh old; walking %s", source, age_hours, directory)
        return None
    try:
        return _entries(directory, json.loads(source.read_text()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Manifest %s is unreadable (%s); walking %s", source, exc, directory)
        return None


def _entries(directory: Path, listing: object) -> dict[Path, Entry]:
    """Turn lsjson objects into entries, refusing paths that leave directory."""
    if not isinstance(listing, list):
        msg = "the listing is not a JSON array"
        raise TypeError(msg)
    entries: dict[Path, Entry] = {}
    for item in listing:
        if not isinstance(item, dict):
            msg = "a listing entry is not a JSON object"
            raise TypeError(msg)
        if item.get("IsDir"):
            continue
        raw = str(item["Path"])
        relative = PurePosixPath(raw)
        # A backslash is a separator on some platforms, so a "..\\x" would
        # pass the POSIX check below as a single harmless component.
        if "\\" in raw or relative.is_absolute() or ".." in relative.parts:
            msg = f"path {relative} escapes the index path"
            raise ValueError(msg)
        modified = datetime.fromisoformat(str(item["ModTime"]).replace("Z", "+00:00"))
        entries[directory / relative] = Entry(int(str(item["Size"])), modified.timestamp())
    return entries
