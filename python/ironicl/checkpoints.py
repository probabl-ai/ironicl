"""Released checkpoints in Ironicl's cache: one folder per model, downloaded
once, with a warning when a file is not the one Ironicl pins."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
import urllib.request
import warnings
from pathlib import Path

from ironicl import _core
from ironicl.cache import STALE_TEMPORARY_SECONDS

CHUNK_BYTES = 1 << 20


def _checkpoints_dir(model: str | None = None) -> Path:
    """Where downloaded checkpoints go: ``checkpoints`` in Ironicl's cache folder,
    one folder per model, such as ``checkpoints/tabicl``."""
    checkpoints = Path(_core.cache_dir()) / "checkpoints"
    return checkpoints if model is None else checkpoints / model


def fetch(path: Path, url: str, sha256: str | None, *, allow_download: bool, data: bytes | None = None) -> Path:
    """``path``, downloaded from ``url``, or written from ``data``, when missing.

    The file is written to a temporary file of its own next to ``path`` and
    renamed into place, so concurrent fetches each leave a whole file, with the
    permissions the umask gives a new file. Temporary files that interrupted
    fetches left are removed once stale. When its SHA-256 is not ``sha256``, it
    is kept and a warning says so.
    """
    if path.exists():
        return path
    if data is None and not allow_download:
        raise FileNotFoundError(f"Checkpoint {path} not found and allow_auto_download=False")
    path.parent.mkdir(parents=True, exist_ok=True)
    _remove_stale(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".partial")
    os.chmod(temporary, 0o666 & ~_umask())
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as file:
            if data is None:
                with urllib.request.urlopen(url) as response:
                    while chunk := response.read(CHUNK_BYTES):
                        digest.update(chunk)
                        file.write(chunk)
            else:
                digest.update(data)
                file.write(data)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    if sha256 is not None and digest.hexdigest() != sha256:
        warnings.warn(
            f"{path} has SHA-256 {digest.hexdigest()}, not {sha256} as Ironicl expects; "
            "it may be corrupt or changed upstream",
            UserWarning,
            stacklevel=3,
        )
    return path


def _umask() -> int:
    """The process's umask: setting it returns the old one, which is set back."""
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


def _remove_stale(path: Path) -> None:
    """Remove temporary files of ``path`` that interrupted fetches left."""
    for temporary in path.parent.glob(f".{path.name}.*.partial"):
        try:
            if time.time() - temporary.stat().st_mtime > STALE_TEMPORARY_SECONDS:
                temporary.unlink()
        except OSError:
            pass
