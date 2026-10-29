"""Shared test configuration.

When ``IRONICL_CHECKPOINT_DIR`` is set the estimator reads the official
checkpoint from there, otherwise it downloads it from Hugging Face on first use
(or reuses the Hugging Face cache). Tests marked ``slow`` need the checkpoint;
set ``IRONICL_SKIP_SLOW=1`` to skip them without it.

The tests import ``ironicl`` as installed (editable through ``maturin develop``
or from a wheel); nothing here puts the repository root on ``sys.path``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CHECKPOINT_NAME = "tabicl-classifier-v2-20260212.ckpt"


def pytest_collection_modifyitems(config, items) -> None:
    if os.environ.get("IRONICL_SKIP_SLOW") != "1":
        return
    skip = pytest.mark.skip(
        reason="IRONICL_SKIP_SLOW=1 skips tests that need the TabICLv2 checkpoint"
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


def _checkpoint_kwargs() -> dict[str, str]:
    directory = os.environ.get("IRONICL_CHECKPOINT_DIR")
    if directory:
        return {"model_path": str(Path(directory) / CHECKPOINT_NAME)}
    return {}


@pytest.fixture
def checkpoint_kwargs() -> dict[str, str]:
    """Estimator kwargs selecting a local checkpoint when one is configured."""
    return _checkpoint_kwargs()


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "ironicl-cache"
