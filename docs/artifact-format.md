# Artifact Format

Weights, fitted preprocessing, and the optional static context use one tensor container,
implemented in [`src/format/container.rs`](../src/format/container.rs). Each file embeds its
own JSON header. The separate `ironicl.json` describes the exported directory; it is not a
sidecar for tensor offsets or shapes. Every binary artifact uses `.ir`; its header identifies
its kind. The directory layout and lookup rules stay the same:

```text
<cache-root>/ironicl/
├── models/v1/<checkpoint-sha256>.ir   # shared weights
└── contexts/<fit-identity>.ir         # optional automatic context cache

my-model/
├── ironicl.json
├── preprocessing.ir
└── context.ir                   # optional; rebuildable from preprocessing and weights
```

This pre-release format is incompatible with the old `IRMODEL`, `IRPREP`, and `IRCNTXT` magic
bytes. Reimport weights from the original checkpoint. Re-export fitted models with the new
version (or refit from training data); the new reader cannot convert old exports or pickles.
Legacy automatic weight and context cache entries are misses and can be rebuilt. Renaming a
legacy file to `.ir` does not convert it. `<cache-root>` defaults to `$XDG_CACHE_HOME` or
`~/.cache`. `--weights` and `cache_dir` can select different weights and context directories.

## Layout

| Region | Encoding |
|---|---|
| Bytes 0–7 | ASCII `IRONICL` followed by a zero byte |
| Bytes 8–15 | JSON header length in bytes, little-endian `u64` |
| Header | Exactly that many UTF-8 bytes, at most 16 MiB |
| Header padding | Zero bytes up to `alignment`; the next offset is `data_start` |
| Tensor data | Named, contiguous, row-major, little-endian payloads in directory order, each padded to `alignment`, including the last |

The JSON object has five fields: `format` (currently `1`), `kind` (`model`, `preprocessing`, or
`context`), `alignment` in bytes, kind-specific `metadata`, and an ordered `tensors` array.
For example, an entry for the key of ICL layer 0 can have `dtype: "f16"` and
`shape: [1, 8, 50000, 64]`, with `bytes: 51200000`. An entry's `name` identifies it; `offset`
is **relative to `data_start`**, not the beginning of the file. The first offset is zero. Each
subsequent offset equals the aligned end of the preceding payload.

The envelope recognizes `u8`, `i32`, `i64`, `f8_e4m3`, `f16`, `bf16`, `f32`, and `f64`.
Each loader accepts only the types its model supports. Writers use 16 KiB alignment for weights
and contexts on disk, and 64 bytes for preprocessing and compact in-memory encodings. Readers
accept power-of-two alignments from 64 bytes through 2 MiB. They check unique names, rank,
shape × element size, offsets, and the exact padded file length with checked arithmetic. The
directory is limited to 4,096 tensors, 16 dimensions per tensor and 4,096 bytes per name.

There is **no payload checksum or signature**. A SHA-256 identity identifies inputs, not the
serialized bytes. A structurally valid change to a finite tensor value can go undetected. For
integrity across storage or transport, verify the complete file against an independently trusted
digest before deployment. Publish immutable files by atomic replacement, never by modifying or
truncating a file that readers may have mapped.

## Model

`kind: "model"` stores logical checkpoint tensors, not kernel-packed weights. `metadata` records
the source checkpoint SHA-256 and the architecture configuration: named TabICLv2 dimensions or
Kumo-Tabular's size and task. Inspection needs only the header. Loading an executable model also
checks its expected manifest, configuration and tensor values. On native little-endian hosts,
FP32 weights can borrow their immutable mapping; kernels may still pack or upload them. The model
file is limited to 1 GiB.

## Fitted Models

`kind: "preprocessing"` records `metadata.model` (`tabicl_v2` or `kumo_tabular`), the checkpoint
SHA-256 and the fitted `state`. The model family selects the loader; unknown families are rejected.
The weights, preprocessing, and optional context stay separate so fitted models can share weights
and discard a large cache without losing the training state.

- **TabICLv2:** The state records the feature mask, typed category dictionaries, normalization
  statistics, member groups and permutations, and the classifier's labels and aggregation or the
  regressor's target scale. Optional training tensors are `group.{i}.x` (`f32`, `[rows, features]`)
  and `y` (`f32`, `[rows]`). They are needed to rebuild a missing context. When a pickle omits
  those rows, `metadata.context_identity` binds its preprocessing to the saved context. Decoding checks fitted
  invariants, finite transformed values and valid class indices.
- **Kumo-Tabular:** The state records members, seed, feature names and typed category dictionaries.
  Loading refits preprocessing from `x` (`f64`, `[rows, features]`, permitting NaN for missing
  numeric values and −1 for missing categories). `y` stores integers, floats or booleans in a
  typed tensor. Text targets use `y.lengths` (`i64` byte counts) and `y.utf8` (`u8` bytes).

## Static Context

`kind: "context"` records architecture, checkpoint SHA-256, fit identity, ensemble count, training
rows, features, classes, permutations and the `optimize` level. The filename does not select the
device or optimization policy. Loading checks compatibility against the model and fitted state.

- **TabICLv2:** `column.{0..2}.{key,value}` (`f32`, `[B,H+4,8,128,16]`) precede
  `icl.{0..11}.{key,value}` (`[B,8,N,64]`). Layers are in numeric order, key then value. The
  dtype depends on the backend and `optimize`. CUDA uses FP32 at `safe`, FP16 at `auto`, and
  E4M3 at `fast` with one positive, finite FP32 scale per estimator, head and channel. Each
  `icl.{block}.{key,value}_scale` has shape `[B,8,64]`. Here `B` counts estimators, `H` retained
  features and `N` training rows. CPU and Metal keep their existing layouts.
- **Kumo-Tabular:** Each `{member}.{task}` has `means`, `categorical`, `columns.key`,
  `columns.value`, `icl.key`, and `icl.value`. Column and ICL tensors group layers along the
  leading dimension. CUDA writes both K/V stages as FP16 for classifiers at `auto` or `fast`
  and regressors at `fast`. Regression `auto` and all `safe` caches remain FP32. CPU-built
  members, means and column kinds are always serialized as FP32. The context identity is the
  SHA-256 of its preprocessing bytes.

TabICLv2's native FP8 context requires a CUDA `fast` predictor. It uploads saved bytes without
re-encoding. All-missing query rows now read that same cache on the CPU rather than a separate
FP16 copy, so their `fast` outputs can differ from older releases. If the requested CUDA level
cannot use the saved dtype (for example, CPU FP32 at `auto`), the loader rejects the cache
rather than selecting the CPU. `ironicl prepare` rebuilds an incompatible directory cache from
fitted preprocessing. To move a Python estimator to a
non-CUDA backend, use `save(..., save_kv_cache=False)` and keep its training data. Existing
unified FP16 CUDA contexts still load. The E4M3 encodings `0x7f` and `0xff` are NaNs and are
invalid cache values.

The change reduces eligible Kumo K/V payloads from four bytes per value to two and TabICLv2
`fast` ICL payloads from two bytes to one plus scales. Neither ratio describes whole-file size
or prediction latency. Prediction still loads the context into host memory.

### Validation and loading

`StaticContextArtifact::write` streams the payloads of an already-built context to a sibling
owner-only temporary file, syncs it, then publishes it by rename. `read_mapped_prefault` hints
that pages will be needed and checks finite values before exposing typed views. A page-aligned,
page-padded mapping is eligible for the existing Metal buffer-wrapping path. `from_bytes` owns
its backing storage, so a pickled model need not retain the caller's byte buffer.

`ContextHeader::open` reads only the prefix and header and checks the directory against the file
length. It does not load or validate tensor values.

## Inspection

`ironicl inspect FILE` recognizes the container by its magic. To inspect a header independently:

```python
import json
from pathlib import Path

with Path("my-model/context.ir").open("rb") as f:
    if f.read(8) != b"IRONICL\0":
        raise ValueError("not an Ironicl container")
    size = int.from_bytes(f.read(8), "little")
    if not 0 < size <= 16 << 20:
        raise ValueError("invalid header size")
    header = json.loads(f.read(size))
print(header["kind"], header["alignment"], len(header["tensors"]))
```
