"""The context cache that ``cache_dir`` fills: where it is, what it holds, and
removing entries from it."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from ironicl import _core

SUFFIX = ".irctx"
# An interrupted write leaves a temporary file; one older than this is not
# still being written.
STALE_TEMPORARY_SECONDS = 3600


def _cache_dir(cache_dir) -> Path | None:
    """Resolve ``cache_dir`` HuggingFace-style.

    ``None``/``True`` use ``IRONICL_CACHE_DIR`` or ``<cache root>/ironicl/contexts``;
    ``False`` disables caching; a string or path is used directly.
    """
    if cache_dir is False:
        return None
    if cache_dir is None or cache_dir is True:
        override = os.environ.get("IRONICL_CACHE_DIR")
        if override:
            return Path(override)
        root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        return root / "ironicl" / "contexts"
    return Path(cache_dir)


@dataclass(frozen=True)
class CacheEntry:
    """One cached context. ``last_used``, in local time, is when a fit last built or loaded it."""

    path: Path
    size_bytes: int
    last_used: datetime


@dataclass(frozen=True)
class CacheInfo:
    """Cached contexts in ``directory``, most recently used first."""

    directory: Path
    entries: tuple[CacheEntry, ...]

    @property
    def size_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries)

    def __str__(self) -> str:
        lines = [f"{self.directory}: {len(self.entries)} contexts, {_gib(self.size_bytes)}"]
        lines += [
            f"  {_gib(entry.size_bytes):>9}  last used {entry.last_used:%Y-%m-%d %H:%M}  {entry.path.name}"
            for entry in self.entries
        ]
        return "\n".join(lines)


def cache_info(cache_dir: str | os.PathLike | None = None) -> CacheInfo:
    """The contexts cached in ``cache_dir``, by default where ``cache_dir=None`` caches."""
    directory = _cache_dir(True if cache_dir is None else cache_dir)
    if directory is None:
        raise ValueError("cache_dir=False disables caching; pass None or a directory")
    entries = []
    for path in directory.glob(f"*{SUFFIX}") if directory.is_dir() else ():
        try:
            status = path.stat()
        except OSError:
            continue  # Removed meanwhile.
        entries.append(CacheEntry(path, status.st_size, datetime.fromtimestamp(status.st_mtime)))
    entries.sort(key=lambda entry: entry.last_used, reverse=True)
    return CacheInfo(directory, tuple(entries))


def clear_cache(
    cache_dir: str | os.PathLike | None = None,
    *,
    older_than: timedelta | datetime | None = None,
    max_size: int | str | None = None,
) -> CacheInfo:
    """Remove cached contexts from ``cache_dir`` and return what was removed.

    Without arguments, removes every context. ``older_than``, a duration or a
    time, removes contexts last used before it. ``max_size``, in bytes or as a
    size such as ``"10G"``, then removes the least recently used contexts until
    the rest fit. Temporary files that interrupted writes left behind are
    removed in every case. A context that cannot be removed, such as one in use
    on Windows, is kept.
    """
    info = cache_info(cache_dir)
    keep = list(info.entries)
    if older_than is None and max_size is None:
        keep = []
    if older_than is not None:
        cutoff = datetime.now() - older_than if isinstance(older_than, timedelta) else older_than
        if cutoff.tzinfo is not None:
            cutoff = cutoff.astimezone().replace(tzinfo=None)  # last_used is local time.
        keep = [entry for entry in keep if entry.last_used >= cutoff]
    if max_size is not None:
        limit = _core.parse_size(max_size) if isinstance(max_size, str) else max_size
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError(f"max_size must be a number of bytes or a size such as '10G', not {max_size!r}")
        while keep and sum(entry.size_bytes for entry in keep) > limit:
            keep.pop()  # The least recently used.
    removed = []
    for entry in info.entries:
        if entry not in keep:
            try:
                entry.path.unlink()
                removed.append(entry)
            except OSError:
                pass
    for path in info.directory.glob(".ironicl-*.tmp") if info.directory.is_dir() else ():
        try:
            if time.time() - path.stat().st_mtime > STALE_TEMPORARY_SECONDS:
                path.unlink()
        except OSError:
            pass
    return CacheInfo(info.directory, tuple(removed))


def _gib(size: int) -> str:
    return f"{size / 2**30:.2f} GiB"
