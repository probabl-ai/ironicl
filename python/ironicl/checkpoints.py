"""Released checkpoints in Ironicl's cache: one folder per model, downloaded
once and verified against the pinned SHA-256."""

from __future__ import annotations

import hashlib
import os
import secrets
import time
import urllib.request
import warnings
from pathlib import Path

from ironicl import _core
from ironicl.cache import STALE_TEMPORARY_SECONDS

CHUNK_BYTES = 1 << 20
# Matches MAX_ARCHIVE_BYTES in src/models/checkpoint.rs.
MAX_CHECKPOINT_BYTES = 1 << 30
DOWNLOAD_TIMEOUT_SECONDS = 60


def _checkpoints_dir(model: str | None = None) -> Path:
    """Where downloaded checkpoints go: ``checkpoints`` in Ironicl's cache folder,
    one folder per model, such as ``checkpoints/tabicl``."""
    checkpoints = Path(_core.cache_dir()) / "checkpoints"
    return checkpoints if model is None else checkpoints / model


def fetch(
    path: Path,
    url: str,
    sha256: str | None,
    *,
    allow_download: bool,
    data: bytes | None = None,
    max_bytes: int = MAX_CHECKPOINT_BYTES,
    timeout: float = DOWNLOAD_TIMEOUT_SECONDS,
) -> Path:
    """``path``, downloaded from ``url``, or written from ``data``, when missing.

    The file is written to a temporary file of its own next to ``path`` and
    renamed into place, so concurrent fetches each leave a whole file, with the
    permissions the umask gives a new file. Temporary files that interrupted
    fetches left are removed once stale. Auto-downloaded checkpoints must match
    ``sha256`` and stay within ``max_bytes``; mismatched ``data`` (such as
    custom weights restored from a pickle) is kept with a warning.
    """
    if path.exists():
        return path
    if data is None and not allow_download:
        raise FileNotFoundError(f"Checkpoint {path} not found and allow_auto_download=False")
    path.parent.mkdir(parents=True, exist_ok=True)
    _remove_stale(path)
    digest = hashlib.sha256()
    while True:
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.partial")
        try:
            file = temporary.open("xb")  # The OS applies the umask at creation.
            break
        except FileExistsError:
            continue
    try:
        with file:
            if data is None:
                written = 0
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    while chunk := response.read(CHUNK_BYTES):
                        written += len(chunk)
                        if written > max_bytes:
                            raise ValueError(
                                f"Downloaded checkpoint from {url} exceeds the {max_bytes}-byte limit"
                            )
                        digest.update(chunk)
                        file.write(chunk)
            else:
                digest.update(data)
                file.write(data)
        if data is None and sha256 is not None and digest.hexdigest() != sha256:
            raise ValueError(
                f"Downloaded checkpoint from {url} has SHA-256 {digest.hexdigest()}, "
                f"not {sha256} as Ironicl expects"
            )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if data is not None and sha256 is not None and digest.hexdigest() != sha256:
        warnings.warn(
            f"{path} has SHA-256 {digest.hexdigest()}, not {sha256} as Ironicl expects; "
            "it may be custom or changed upstream",
            UserWarning,
            stacklevel=3,
        )
    return path


def _remove_stale(path: Path) -> None:
    """Remove temporary files of ``path`` that interrupted fetches left."""
    for temporary in path.parent.glob(f".{path.name}.*.partial"):
        try:
            if time.time() - temporary.stat().st_mtime > STALE_TEMPORARY_SECONDS:
                temporary.unlink()
        except OSError:
            pass
