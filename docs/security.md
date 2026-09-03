# Security and Robustness

## Trust Boundaries

Released model checkpoints, generated Ironicl artifacts, training data, and
request rows are distinct inputs. Treat artifacts and checkpoints as untrusted
unless their source and hash are verified.

## Artifact Parser Requirements

- Validate magic and supported version before reading section metadata.
- Use checked arithmetic for every offset, length, element count, and alignment.
- Verify that sections remain inside the file and do not overlap illegally.
- Bound tensor rank and dimension counts.
- Verify payload length against dtype and physical layout.
- Verify checksums before exposing tensor data.
- Reject duplicate mandatory tensor IDs.
- Reject NaN or invalid quantization scales.
- Do not execute embedded code or deserialize Python pickle objects in Rust.
- Avoid writable mappings for serving artifacts.

## Checkpoint Conversion

PyTorch checkpoint conversion is a pure-Rust, offline operation. The importer
uses Candle's pinned pickle object reader after a strict opcode and resource
preflight. Ironicl enforces archive, pickle, opcode, memo, object-depth,
tensor-count, tensor-size, rank, and contiguous-layout limits. It decodes tensor
reductions with checked pattern matching, represents globals as data, and does
not invoke Python callables. Inspection reads every tensor payload so the ZIP
reader verifies its length and CRC.

The importer must verify the expected top-level keys, model configuration,
tensor names, shapes, dtypes, byte counts, and checkpoint hash. Released model
profiles require the pinned official hashes. The official Python loader remains
an independent comparison oracle only.

Portable model writes use a sibling temporary file, `fsync`, complete reader
verification, atomic rename, and parent-directory `fsync`. Cache entries are
addressed by checkpoint SHA-256 and artifact format version. Cache hits verify
the complete model checksum and embedded source hash before reuse.

The serving process reads only the Ironicl artifact format.

## Data Handling

Static artifacts encode information derived from the complete training set and
targets. They are sensitive data even if raw rows are omitted.

- Apply the same access controls as the training data.
- Do not log K/V payloads or target embeddings.
- Zero temporary plaintext buffers when the deployment threat model requires
  it.
- Document whether crash dumps can include model or context memory.
- Use atomic replacement so a failed build cannot corrupt the serving context.

## Denial of Service

- Enforce configured limits on rows, features, estimators, and microbatch size.
- Calculate memory before allocation.
- Reject non-finite request data after the defined missing-value transform.
- Bound concurrent executors and scratch allocations.
- Apply request timeouts outside the kernel layer.
