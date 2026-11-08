"""``show_config``: what Ironicl detects on this machine and how it would run."""

from __future__ import annotations

import os
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ironicl import _core
from ironicl.cache import cache_info
from ironicl.memory import _plan

PACKAGES = ("numpy", "scikit-learn", "scipy", "tqdm")
ENVIRONMENT = (
    "IRONICL_OPTIMIZE",
    "IRONICL_THREADS",
    "IRONICL_NO_METAL",
    "IRONICL_MEMORY_LIMIT",
    "IRONICL_DEVICE_MEMORY_LIMIT",
    "IRONICL_MEMORY_OVERFLOW",
    "IRONICL_CACHE_DIR",
)
# The shape the storage lines describe.
EXAMPLE = (8, 1000, 100)


def show_config(file=None) -> None:
    """Print Ironicl's version and dependencies, the GPU and CPU it would use,
    memory, the cache storage of each ``optimize`` level, the checkpoints, the
    context cache, and the ``IRONICL_*`` environment variables that are set."""
    print(_config(), file=file)


def _config() -> str:
    info = _core.system_info()
    lines = [
        f"Ironicl {info['version']}",
        f"Python {platform.python_version()}, " + ", ".join(f"{name} {_version(name)}" for name in PACKAGES),
        "",
        f"GPU: {_gpu(info)}",
        f"CPU: {info['architecture']} {info['operating_system']}, {info['threads']} threads, "
        f"BF16 storage for optimize='fast' on the CPU {'supported' if info['bf16_storage'] else 'not supported'}",
        f"Memory: {_memory(info)}",
        "",
        "ICL cache storage by optimize level, for {} estimators x {} rows x {} features:".format(*EXAMPLE),
    ]
    for level in ("safe", "auto", "fast"):
        plan = _plan("classifier", *EXAMPLE, level, None, None)
        lines.append(f"  {level}: {plan.storage}, built on the {plan.builder.upper()}, {_gib(plan.context_bytes)}")
    checkpoints = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ironicl" / "checkpoints"
    found = sorted(checkpoints.glob("*.ckpt")) if checkpoints.is_dir() else []
    lines += ["", f"Checkpoints: {checkpoints}"]
    lines += [f"  {path.name} ({_gib(path.stat().st_size)})" for path in found] or ["  none downloaded"]
    cache = cache_info()
    lines.append(
        f"Context cache (used with cache_dir): {cache.directory}, "
        f"{len(cache.entries)} contexts, {_gib(cache.size_bytes)}"
    )
    set_variables = [f"  {name}={os.environ[name]}" for name in ENVIRONMENT if name in os.environ]
    lines += ["", "Environment:", *(set_variables or ["  no IRONICL_* variables set"])]
    return "\n".join(lines)


def _gpu(info: dict) -> str:
    if not info["metal"]:
        return "none (this build has no Metal support)"
    if info["gpu"] is None:
        return "none found"
    status = (
        "disabled by IRONICL_NO_METAL"
        if os.environ.get("IRONICL_NO_METAL")
        else f"predicts, and builds contexts of {info['gpu_min_build_rows']} training rows or more"
    )
    return f"{info['gpu']} (Metal), working set {_gib(info['gpu_working_set'])}; {status}"


def _memory(info: dict) -> str:
    if info["memory_total"] is None:
        return "unknown on this platform"
    text = f"{_gib(info['memory_total'])} total, about {_gib(info['memory_available'])} available"
    if info["memory_hard_limit"] is not None:
        text += f", hard limit {_gib(info['memory_hard_limit'])} more"
    return text


def _version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _gib(size: int) -> str:
    return f"{size / 2**30:.1f} GiB" if size >= 2**30 else f"{size / 2**20:.0f} MiB"
