"""Type stubs for the compiled ``ironicl._ironicl`` module (Rust, pyo3)."""

from os import PathLike

import numpy as np
from numpy.typing import NDArray

__version__: str

class IronICLError(RuntimeError):
    """Raised when the IronICL engine fails to load an artifact or run a query."""

class StaticExecutor:
    """A resident cached classifier: one model artifact plus one static context."""

    def __init__(
        self, model_path: str | PathLike[str], context_path: str | PathLike[str]
    ) -> None: ...
    @staticmethod
    def from_bytes(
        model_path: str | PathLike[str], context: bytes
    ) -> StaticExecutor: ...
    @property
    def ensemble_count(self) -> int: ...
    @property
    def training_rows(self) -> int: ...
    @property
    def feature_count(self) -> int: ...
    @property
    def class_count(self) -> int: ...
    def predict_proba(self, variants: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run one ``[B, Q, H]`` request; returns ``[Q, C]`` probabilities."""
    def predict_proba_batch(self, variants: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run ``[R, B, Q, H]`` requests as one microbatch; returns ``[R, Q, C]``."""
    def sparse_attention_statistics(self) -> dict[str, int]: ...
    def reset_sparse_attention_statistics(self) -> None: ...

def prepare_model(
    checkpoint_path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
    rebuild: bool = False,
) -> str:
    """Import a TabICLv2 checkpoint into a content-addressed model artifact."""

def inspect_model(path: str | PathLike[str]) -> dict[str, object]:
    """Return the summary of a verified model artifact."""

def build_context(
    model_path: str | PathLike[str],
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    class_permutations: NDArray[np.uint64],
    feature_permutations: NDArray[np.uint64],
    context_sha256: bytes,
    output_path: str | PathLike[str],
    *,
    kv_dtype: str = "fp32",
    pack_keys: bool = False,
    index_clusters: int | None = None,
) -> dict[str, object]:
    """Build the static context of ``[B, N, H]`` training variants natively.

    ``kv_dtype`` is ``"fp32"``, ``"bf16"`` or ``"bf16-v"``.

    ``y`` holds ``[B, N]`` class indices as float32; the permutations are
    ``[B, C]`` and ``[B, H]`` uint64 arrays. Writes ``output_path`` atomically
    and returns its description (``B``, ``N``, ``H``, ``C``,
    ``static_context_version``, ``kv_dtype``, ``packed_icl_keys``,
    ``index_clusters``, ``context_file_bytes``, ``context_file_sha256``).
    """
