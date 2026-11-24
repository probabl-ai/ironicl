# Artifact Format

Ironicl writes three binary formats:

| Format | Magic | Holds | Source |
|---|---|---|---|
| model | `IRMODEL\0` v1 | checkpoint weights, TabICLv2 or Kumo-Tabular | `src/format/model.rs` |
| preprocessor | `IRPREP\0\0` v1 | fitted preprocessing | `src/format/preprocessor.rs` |
| static context | `IRCNTXT\0` v1 and v2 | the training rows' attention keys and values | `src/format/static_context.rs` |

Every format is little-endian, uses fixed-width fields, and ends with a SHA-256 of all preceding
bytes. None uses native Rust struct serialization.

## Model

`ironicl import` and the content-addressed checkpoint cache write `IRMODEL` v1. It stores the
validated logical checkpoint tensors in a deterministic, 64-byte-aligned layout, with the source
checkpoint's SHA-256 as the model's identity. The header names the architecture, 1 for TabICLv2 or
2 for [Kumo-Tabular](kumo-tabular.md), and seven of its values: TabICLv2's classes, quantiles,
widths, and block counts, or Kumo-Tabular's classes, quantiles, cell channels, column and row
layers, inducing points, ICL channels, and ICL layers, which identify its size. A file holds at
most 1 GiB, for Kumo-Tabular large. It is portable rather than kernel-packed: kernels pack weights
at load time. A packed format can take a new version without changing the CLI or the
cache.

## Fitted Models

A fitted model is three files, linked by the checkpoint's SHA-256: the weights (`IRMODEL`), the
fitted preprocessing with its training rows (`IRPREP`), and optionally the training context
(`IRCNTXT`). Pickling and `save` store the last two as bytes, not as Python objects. `export`
writes them as a model directory, with the weights in a shared folder
([cli.md](cli.md#model-directories-and-weights)).

`IRPREP` v1 starts with the architecture, as `IRMODEL` names it. For TabICLv2 (1) it holds the
checkpoint's SHA-256 (32 bytes, as `IRCNTXT` stores it), the feature mask,
each input feature's categories (none for a numeric feature),
the normalization groups, each member's group and feature and class order, the class labels
(integers, floats, booleans, or UTF-8 text) with the classifier's `softmax_temperature` and
`average_logits`, or the regressor's target mean and scale, the feature names when the training
data had them, each kept feature's training mean, which replaces its missing values, per group each kept feature's statistics as `f64` (mean, scale, range, outlier
bounds, and for the power group its lambda and standardization), and optionally the preprocessed
training rows a context is built from: each group's `[rows, features]` matrix and each row's
target (a class index or the standardized value), as `f32`. Decoding checks every length against
the bytes left before it allocates, then the preprocessing's invariants, finite training values,
and class indices in range.

Loading refuses an `IRPREP` fitted with another checkpoint. With a context, it checks that the
context matches the preprocessing: checkpoint, `B`, `H`, class count, and each member's
permutations. Without one, it builds the context from the training rows that `IRPREP` holds, as
fitting does, so the same context.

For Kumo-Tabular (2), `IRPREP` holds the checkpoint's SHA-256, the members and the seed, the
feature names, each input feature's categories, and the training data: the encoded `[rows,
features]` matrix as `f64` (NaN for a missing number, -1 for a missing category) and each row's
label, a regressor's as a float. Loading fits again from them, which gives the same model; Kumo-Tabular has no `IRCNTXT`.
Decoding checks every length against the bytes left, and that each category code is -1 or one of
its feature's categories.

Neither `IRPREP` nor `IRCNTXT` is signed yet: the digests detect corruption, not tampering.

## Static Context

Most of the work in repeated TabICLv2 inference depends only on the training rows. A static
context stores that work once: the column stage's keys and values for the three induced-attention
blocks, and the ICL keys and values for all twelve layers. A prediction then runs only the query
rows, attending to the stored keys and values. Softmax rules out compressing them into a small
summary, so a prediction scans the whole ICL cache.

### Build Path

The Rust engine builds every context:

1. `Preprocessed::fit` fits the preprocessing on the raw training rows and produces each
   estimator's transformed rows and permutations.
2. `Executor::build_fitted` passes them to `Executor::build_context`, which runs the training rows
   through the model on the CPU, Metal, or CUDA and keeps the caches.
3. The resulting `StaticExecutor` predicts query rows, which `FittedModel` preprocesses first,
   and undoes each estimator's class order when it aggregates probabilities.

`optimize` selects the cache storage: FP32 at `"safe"`; FP16 ICL keys and values on the GPUs at
`"auto"` and `"fast"`; BF16 ICL values with FP32 keys on a BF16-capable CPU at `"fast"`. Column
caches stay FP32.

With a cache directory, `build_context` loads `<identity>.irctx` when that file holds exactly this
context, and otherwise builds the context and writes it there. The identity is a SHA-256 over the
Ironicl version, the source checkpoint's SHA-256, `optimize`, `B`, `N`, `H`, the transformed
training rows and targets, and both permutations. A missing, invalid, or mismatched entry is
rebuilt; a failed write leaves the context in memory only. `Executor::load_context` loads a
context file directly.

Writes are atomic: the encoder streams into a new owner-only temporary file in the same directory,
syncs it, and renames it into place. A reader never sees a partial file.

Any change to the training data builds a new context. Incremental updates are out of scope: each
layer's training representation depends on all of the previous layer's keys and values, so an
insertion is not local.

### Versions

| Version | Tensor storage | ICL K layout |
|---|---|---|
| 1 | FP32 only | logical row-major |
| 2 | explicit FP32, BF16, or FP16 per tensor | logical row-major |
| 3 | removed; the reader rejects it | source-tile, dimension-major |

The writer uses version 1 when every tensor is FP32 and version 2 otherwise. A context with BF16
ICL keys is rejected when it is loaded. Only the removed Python exporter wrote version 3 and BF16
keys; see [removed-experiments.md](removed-experiments.md#ircntxt-v3-pre-packed-keys) and
[removed-experiments.md](removed-experiments.md#bf16-attention-keys). Contexts hold no model
weights.

### Layout

The fixed header holds the source checkpoint's SHA-256, the context identity, `B`, `N`, `H`, the
class count, and the required TabICLv2 architecture values: 3 column layers, 12 ICL layers,
8 heads, column head dimension 16, and ICL head dimension 64. Estimator-major class `[B,C]` and
feature `[B,H]` permutations follow, then a sorted tensor directory, 64-byte-aligned payloads, and
the trailing SHA-256. A regression context stores class count 0 and no class permutations. The
header's last field records the `optimize` level the context was built at: 1 `safe`, 2 `auto`,
or 3 `fast`.

Each directory entry holds the tensor's UTF-8 name, rank, dimensions, payload offset, byte length,
and, in version 2, its dtype. The executor requires exactly these tensors:

- `column.{0..2}.{key,value}`, FP32, `[B,H+4,8,128,16]`;
- `icl.{0..11}.{key,value}`, `[B,8,N,64]`: FP32 keys and values, FP32 keys with BF16 values, or
  FP16 keys and values.

Tensor dimensions are logical. The CPU attention kernels pack each key tile in dimension-major
order into scratch, so one AVX-512 vector holds source lanes while a query dimension is
broadcast. Values stay source-major, so one sequential value row updates all query accumulators.

Ironicl no longer reads the optional `icl.index.*` sparse-attention tensors; a context that has
them loads with them ignored. See
[removed-experiments.md](removed-experiments.md#sparse-icl-attention-sparse-attention).

### Validation and Loading

Tensor names must be strictly sorted and unique, and every dimension and byte calculation is
checked. Loaders reject non-finite FP32 and BF16 values, invalid permutations, bad offsets,
overlaps, changed padding, unknown versions, malformed headers, and every resource-limit
violation. The executor also checks the context's checkpoint SHA-256 and class count against the
model.

`StaticContextArtifact::read_mapped_prefault` verifies the whole file and keeps read-only,
memory-mapped tensor views. It asks the OS to read the pages ahead, so the first prediction runs
at RAM speed. A mapped file must not change while it is loaded; contexts are written once and
replaced by rename, never edited in place. `from_bytes` decodes an owned copy, which `FittedModel`
uses for pickled models.

## Current Guarantees

- Little-endian, fixed-width scalar metadata.
- An explicit format version and artifact kind.
- No native Rust struct serialization or architecture-dependent padding.
- Checked `u64` offsets and lengths.
- 64-byte tensor alignment.
- One trailing whole-file SHA-256.
- SHA-256 identities for the source checkpoint and static context.
- Checked tensor names, shapes, dtypes, offsets, lengths, and finite values.
- Read-only, memory-mapped context loading after whole-file verification.

## Future Format Requirements

- Explicit minimum-reader versions and a mandatory/optional section policy.
- Optional 4 KiB and 2 MiB section alignment for mapping experiments.
- Independent metadata and tensor-section checksums.
- Explicit arithmetic-policy and quantization metadata.
- Signed artifacts, so a server can reject tampered files.

A single serving artifact could combine sections like these:

```text
fixed header
metadata directory
model configuration
feature schema
preprocessing state
ensemble configuration
portable packed weights
Zen 5 packed weights (optional)
column K/V cache
ICL K/V cache
quantization metadata (optional)
checksum table
```

Every tensor entry then needs a stable semantic ID, logical shape, physical tile shape and strides,
storage dtype, arithmetic policy ID, section offset and byte length, required alignment,
quantization scale granularity and offset, and a checksum. Checkpoint tensor names alone are
useful for diagnostics but are not a compact serving contract.

The context identity does not yet cover original feature names, logical types, or preprocessing
algorithm versions; the preprocessing is checked separately when a fitted model loads. A combined
artifact's identity must cover them too.
