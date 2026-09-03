# Artifact Format

## Static Context Versions

Static-context artifacts use the separate `IRONICL\0` format. Three versions
are supported:

| Version | Tensor storage | ICL K layout |
|---|---|---|
| 1 | FP32 only | logical row-major |
| 2 | explicit FP32 or BF16 per tensor | logical row-major |
| 3 | explicit FP32 or BF16 per tensor | source-tile, dimension-major |

Versions 2 and 3 keep column caches in FP32. The exporter can store ICL K/V in
BF16 or retain FP32 K while storing V in BF16. Version 3 physically packs each
ICL K tile for the source-lane score kernel. Tensor dimensions remain logical.
Static-context artifacts do not contain model weights.

The fixed little-endian header contains the source-checkpoint SHA-256 identity
stored inside the model artifact, the context export identity, `B`, `N`, `H`, class count, and the required
TabICLv2 architecture values: 3 column layers, 12 ICL layers, 8 heads, column
head dimension 16, and ICL head dimension 64. It is followed by estimator-major
class `[B,C]` and feature `[B,H]` permutations, a sorted named tensor directory,
64-byte-aligned payloads, and a trailing SHA-256 digest of all preceding bytes.

Each tensor directory entry contains its UTF-8 name, rank, dimensions, payload
offset, and byte length. Versions 2 and 3 also use the existing directory
reserved field for an explicit FP32 or BF16 dtype. Tensor names are strictly sorted and unique. All
dimensions and byte calculations are checked. Loaders reject non-finite FP32
and BF16 values, invalid permutations, bad offsets, overlap, padding changes,
unknown versions, malformed headers, and all resource-limit violations.
`StaticContextArtifact::read` verifies the complete file and retains read-only
mmap-backed tensor views. The owned `from_bytes` test path accepts version 1 up
to 2 GiB.

### Python Exporters

Use `StaticContextArtifact::to_bytes()` as the version 1 test oracle. The Python
exporter writes all versions as a stream into a temporary file, flushes and
syncs it, then atomically replaces the destination. It uses little-endian
fixed-width fields, flattened estimator-major permutations as `u64`, tensor
names in bytewise lexical order, zero alignment padding, and a trailing
`hashlib.sha256(all_prior_bytes).digest()`. The exporter rejects NaN and
infinity. The full-KV exporter writes exactly 30 sorted
tensors: `column.{0..2}.key`, `column.{0..2}.value`, `icl.{0..11}.key`, and
`icl.{0..11}.value`. It preserves official dimensions exactly. Current
classifier caches use column `[B,H+4,8,128,16]` and ICL `[B,8,N,64]` tensors.

### IRQUERY Version 1

`tools/export_static_context.py` writes preprocessed query variants separately.
IRQUERY v1 has a 112-byte little-endian `<8sII32s32sQQQQ>` header containing
`IRQUERY\0`, version, kind, model SHA-256, context SHA-256, `B`, `Q`, `H`, and
element count; contiguous row-major FP32 `[B,Q,H]`; then a SHA-256 trailer. The
model-source and context hashes must match the static-context artifact. The
export metadata records the whole model-artifact file hash separately.

When `--raw-logits-output` is set, the exporter also writes an `IRLOGITS` v1
oracle with a 48-byte `<8sIIQQQQ>` header, FP32 `[B,Q,C]` estimator logits, and
a SHA-256 trailer. `tools/compare_cached_logits.py` compares this oracle with
the output of `ironicl run --context ...`.

Portable model imports use a separate `IRMODEL\0` format. Version 1 stores the
validated logical checkpoint tensors in a deterministic, little-endian,
64-byte-aligned layout with a whole-file SHA-256 checksum. It is intentionally
portable rather than kernel-packed. Automatic checkpoint caching uses this same
format. A future packed format can use a new version without changing the cache
CLI.

## Current Guarantees

- Little-endian, fixed-width scalar metadata.
- An explicit format version and artifact kind.
- No native Rust struct serialization or architecture-dependent padding.
- Checked `u64` offsets and lengths.
- 64-byte tensor alignment.
- One trailing whole-file SHA-256.
- SHA-256 identities for the source checkpoint and static context.
- Checked tensor names, shapes, dtypes, offsets, lengths, and finite values.
- Read-only mmap-backed loading after whole-file verification.

## Future Format Requirements

- Explicit minimum-reader versions and mandatory/optional section policy.
- Optional 4 KiB and 2 MiB section alignment for mapping experiments.
- Independent metadata and tensor-section checksums.
- Explicit arithmetic-policy and quantization metadata.

## Proposed Sections

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
ICL K/V or row-representation cache
quantization metadata (optional)
checksum table
```

## Tensor Metadata

Every tensor entry needs:

- stable semantic tensor ID;
- logical shape;
- physical tile shape and strides;
- storage dtype;
- arithmetic policy ID;
- section offset and byte length;
- required alignment;
- quantization scale granularity and scale offset;
- checksum.

Do not use checkpoint tensor names as the only stable IDs. Names are useful for
diagnostics but are not a compact serving contract.

## Layout Strategy

Version 3 packs each ICL K source tile in dimension-major order. This lets one
AVX-512 vector hold source lanes while a query dimension is broadcast. V remains
source-major so one sequential V row updates all query accumulators. Model
weights remain portable and unpacked.

An optional sparse-attention index adds four FP32 tensors without changing the
version-3 header: `icl.index.centroids`, `icl.index.radii`,
`icl.index.offsets`, and `icl.index.members`. They are estimator-, layer-, and
head-major. The index is all-or-nothing: a context with no index uses dense
attention, while a malformed or incomplete index is rejected. Candidate
retrieval must retain a dense fallback and cannot change the query bundle.
Set `IRONICL_SPARSE_ATTENTION=1` to enable candidate retrieval for an indexed
context. The first implementation selects candidates only when its omitted-mass
certificate is at most `1e-5`; otherwise it uses the dense kernel.

A future combined artifact can organize complete K/V sections so one sequential
tile read supplies all query rows in a microbatch:

```text
[ensemble][layer][training-row tile][head][K or V][head dimension]
```

The final order must be selected by benchmark. Separate K and V streams may be
better because score computation consumes all K before value accumulation.

## Loading Modes

- **Read into anonymous RAM**: baseline and easiest to prefault.
- **Read-only mmap**: current static-context loading mode.
- **Direct I/O**: out of scope while the artifact fits in RAM.

An mmap is not a promise that data is resident. Serving must expose an optional
prefault/warmup step and report major page faults during benchmarks.
