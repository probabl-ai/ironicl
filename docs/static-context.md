# Static Training Context

## Why Static Context Matters

The expensive part of repeated TabICLv2 inference is training-dependent. A
static context converts repeated prediction into test-query projections plus
attention against precomputed training K/V.

## Target Precomputed State

A complete production context can precompute and persist:

- fitted categorical encoders and imputers;
- unique-feature mask and fixed serving schema;
- standardization and normalization state;
- outlier-transform parameters;
- all ensemble feature and class permutations;
- transformed training matrices for each normalization variant;
- training target embeddings;
- column-stage K/V for all three induced-attention blocks;
- training row representations when representation mode is selected;
- ICL K/V for all twelve layers when full-KV mode is selected;
- packed linear weights for every enabled backend;
- constant QASSMax base values that depend only on source length;
- RoPE tables that depend only on fixed feature sequence lengths;
- quantization scales and zero-point metadata;
- section hashes and whole-artifact checksums.

The current executable artifact is narrower. It stores FP32 column caches and
configurable FP32 or BF16 ICL K/V, estimator permutations, model/context
identities, a checked tensor directory, and one trailing whole-file SHA-256.
Version 3 can store ICL K in source-tile, dimension-major order. Representation caches,
fitted preprocessing state, packed weights, RoPE tables, quantization metadata,
and per-section checksums are not implemented.

## Mathematical Work Per Query

Exact inference must still compute:

- input and query projections;
- query-dependent QASSMax modulation;
- query-by-K attention scores;
- stable softmax;
- probability-by-V accumulation;
- row-transformer work for the test row;
- residuals, LayerNorm, GELU, and decoder work.

The current Python exporter performs query preprocessing and ensemble
permutations before it writes `IRQUERY`. The Rust cached executor starts from
these preprocessed variants. Class-permutation reversal and estimator
aggregation are not yet part of cached Rust execution.

Softmax prevents exact compression of arbitrary training K/V into one fixed,
small summary. Full-KV inference must scan the relevant training cache.

## Current Serving Contract

The current static executor consumes:

- column K/V tensors named `column.{0..2}.{key,value}` with shape
  `[B,H+4,8,128,16]`;
- ICL K/V tensors named `icl.{0..11}.{key,value}` with logical shape
  `[B,8,N,64]` and a supported FP32/BF16 policy;
- one checked `IRQUERY` tensor with shape `[B,Q,H]` and matching model/context
  identities.

It returns per-estimator logits with shape `[B,Q,C]`. It does not apply class
permutations or aggregate estimators. The CLI executes this path when both
`--context` and an `IRQUERY` input are supplied.

Only full-K/V mode is executable. Representation mode currently exists only in
the memory planner and design.

## Current Identity

The exporter computes the context identity from canonical training feature and
target arrays plus a serialized cache configuration. That configuration includes
the source-checkpoint hash, complete model-artifact file hash, ensemble count,
batch size, seed, cache mode, normalization methods, shuffle policies, and
outlier threshold. The `model_sha256` field itself is the source-checkpoint
SHA-256 stored in `IRMODEL`, not the complete model-artifact file hash.

The identity does not independently encode original feature names, logical
types, fitted preprocessing state, or preprocessing algorithm versions.

## Production Invalidation Key

A future production identity must also include hashes or canonical encodings of:

- checkpoint bytes and parsed model configuration;
- every retained training feature value;
- every training target;
- original feature names, order, and logical types;
- categorical vocabulary and unknown/missing-value policy;
- preprocessing algorithm versions and fitted values;
- normalization methods;
- outlier threshold;
- feature and class permutations;
- ensemble count and random seed;
- cache mode and storage precision;
- tensor layout version;
- kernel-independent quantization policy.

Any change to these values must invalidate the artifact. Backend-specific packed
weight sections can be regenerated without rebuilding mathematical K/V only if
their source tensors and cache arithmetic policy are unchanged.

## Update Policy

Initial policy: rebuild the context in full after any training-data change.

Do not implement compatibility code for incremental updates. Exact insertion is
not local because each layer's training representation depends on all training
K/V from the previous layer.

## Build Atomicity

The current exporter writes a temporary file, computes the whole-file checksum
while streaming, syncs the file, and atomically replaces the destination. It
removes its temporary file after an error. Per-section checksums, reopen
validation, and directory sync are still pending.

1. Build into a new temporary file.
2. Record section lengths and checksums as sections complete.
3. Flush data and metadata.
4. Reopen and validate the complete artifact.
5. Atomically rename it into the serving location.
6. Keep the old artifact until no executor references it.
