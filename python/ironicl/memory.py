"""Memory planning: limits, what to do when a fit does not fit, and the plan itself."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

from ironicl import _core

OVERFLOW_POLICIES = ("warn", "error", "ignore")


class IroniclMemoryWarning(UserWarning):
    """A fit needs more memory than is available, so the system may page."""


class IroniclMemoryError(MemoryError):
    """A fit needs more memory than allowed, or cannot complete on this machine."""


@dataclass(frozen=True)
class MemoryPlan:
    """How much memory building a context needs, and whether this machine has it.

    ``verdict`` is ``"fits"``, ``"overflow"`` (the peak exceeds available memory or
    ``memory_limit``; the system would page), ``"cannot_complete"`` (the peak
    exceeds a hard limit such as a container's, or the context exceeds physical
    memory), or ``"unknown"``. ``report`` explains the plan and lists changes
    that would fit. Byte counts are integers; ``available_bytes`` is ``None``
    when unknown. ``estimators``, ``rows``, ``features``, and ``optimize`` are
    what was planned for.
    """

    verdict: str
    report: str
    estimators: int
    rows: int
    features: int
    optimize: str
    peak_bytes: int
    available_bytes: int | None
    context_bytes: int
    weight_bytes: int
    build_group: int
    builder: str
    storage: str
    memory_limit: int | None
    device_memory_limit: int | None

    def __str__(self) -> str:
        return self.report


def plan_memory(
    n_samples: int,
    n_features: int,
    *,
    task: str = "classifier",
    n_estimators: int = 8,
    optimize: str | None = None,
    memory_limit: int | str | None = None,
    device_memory_limit: int | str | None = None,
) -> MemoryPlan:
    """Plan the memory to fit ``n_samples x n_features`` on this machine, without data.

    Arguments mean what they mean on ``TabICLClassifier``/``TabICLRegressor``.
    ``n_features`` counts columns after preprocessing; constant columns are
    dropped in ``fit``, so the plan for the raw width is an upper bound.
    """
    optimize = optimize if optimize is not None else os.environ.get("IRONICL_OPTIMIZE", "safe")
    return _plan(task, n_estimators, n_samples, n_features, optimize, memory_limit, device_memory_limit)


def _plan(
    task, estimators, rows, features, optimize, memory_limit, device_memory_limit, load=False
) -> MemoryPlan:
    """Plan building the context, or with ``load``, loading the built context."""
    return MemoryPlan(
        **_core.plan_memory(
            task,
            estimators,
            rows,
            features,
            optimize,
            _limit_text("memory_limit", memory_limit),
            _limit_text("device_memory_limit", device_memory_limit),
            load,
        ),
        optimize=optimize,
    )


def _limit_text(name: str, value: int | str | None) -> str | None:
    """A limit as the size text Rust parses; ``None`` leaves it to the environment."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int number of bytes or a size such as '16G', not {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, not {value}")
    return str(value)


def _overflow_policy(value: str | None) -> str:
    """``memory_overflow``, else ``IRONICL_MEMORY_OVERFLOW``, else ``"warn"``."""
    source = "memory_overflow"
    if value is None:
        value = os.environ.get("IRONICL_MEMORY_OVERFLOW") or "warn"
        source = "IRONICL_MEMORY_OVERFLOW"
    if value not in OVERFLOW_POLICIES:
        raise ValueError(f"{source} must be 'warn', 'error', or 'ignore', not {value!r}")
    return value


def _enforce(plan: MemoryPlan, policy: str, stacklevel: int, action: str = "fit") -> None:
    """Warn or raise for a plan that does not fit, as ``policy`` says. ``action``
    names what the plan is for: ``"fit"`` or ``"load"``."""
    if policy == "ignore" or plan.verdict in ("fits", "unknown"):
        return
    if plan.verdict == "cannot_complete":
        raise IroniclMemoryError(
            f"This {action} cannot complete on this machine.\n{plan.report}\n"
            "Pass memory_overflow='ignore' to try anyway."
        )
    if policy == "error":
        raise IroniclMemoryError(
            f"{plan.report}\nPass memory_overflow='warn' or 'ignore' to proceed anyway."
        )
    warnings.warn(
        f"{plan.report}\nThe {action} continues, and the system may page, which is much slower. "
        "Pass memory_overflow='error' to raise instead, or 'ignore' to silence this.",
        IroniclMemoryWarning,
        stacklevel=stacklevel + 1,
    )
