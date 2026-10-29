"""IronICL: cache-aware CPU inference for TabICLv2 with a scikit-learn interface."""

from importlib.metadata import PackageNotFoundError, version

from ironicl._classifier import IronICLClassifier
from ironicl._ironicl import IronICLError, StaticExecutor

try:
    __version__ = version("ironicl")
except PackageNotFoundError:  # pragma: no cover - editable builds without metadata
    from ironicl._ironicl import __version__

__all__ = ["IronICLClassifier", "IronICLError", "StaticExecutor", "__version__"]
