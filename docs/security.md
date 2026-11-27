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
- The container has no internal checksum. Verify an independently trusted file digest when integrity is required.
- Reject duplicate mandatory tensor IDs.
- Reject NaN or invalid quantization scales.
- Do not execute embedded code or deserialize Python pickle objects in Rust.
- Avoid writable mappings for serving artifacts.

## Checkpoint Conversion

PyTorch checkpoint conversion is a pure-Rust, offline operation. The importer
uses Candle's pinned pickle object reader after a strict opcode and resource
preflight. Ironicl enforces archive, pickle, opcode, memo, object-depth,
tensor-count, tensor-size, and rank limits, and keeps every strided view, such as a transposed
weight, inside its storage before copying it into row-major order. It decodes tensor
reductions with checked pattern matching, represents globals as data, and does
not invoke Python callables. Inspection reads every tensor payload so the ZIP
reader verifies its length and CRC.

The importer must verify the expected top-level keys, model configuration,
tensor names, shapes, dtypes, byte counts, and checkpoint hash. Released model
profiles require the pinned official hashes. The official Python loader remains
an independent comparison oracle only.

Portable model writes use a sibling temporary file, `fsync`, header validation,
atomic rename, and parent-directory `fsync` on Unix. Cache entries are
addressed by checkpoint SHA-256 and artifact format version. Cache hits check
the directory, embedded source identity, and profile; these checks do not authenticate the payload.

The serving process reads only the Ironicl artifact format.

## Preprocessing and Fitted Models

Preprocessing runs in Rust on request data, and the server loads fitted
preprocessing from outside, so `src/preprocess/` forbids `unsafe`.
The loader treats preprocessing bytes as untrusted input; see
[artifact-format.md](artifact-format.md#fitted-models). Python pickles hold the
fitted model in these formats rather than as Python objects, but unpickling an
estimator still runs Python's pickle: load pickles only from trusted sources.

## Data Handling

Static artifacts encode information derived from the complete training set and
targets. They are sensitive data even if raw rows are omitted. Preprocessing with
training tensors holds the preprocessed training rows and targets themselves;
`export` and the context writer create their files readable only by their owner
on Unix.

- Apply the same access controls as the training data.
- Do not log K/V payloads or target embeddings.
- Zero temporary plaintext buffers when the deployment threat model requires
  it.
- Document whether crash dumps can include model or context memory.
- Use atomic replacement so a failed build cannot corrupt the serving context.

## Denial of Service

- Enforce configured limits on rows, features, estimators, and microbatch size.
  The server bounds each request's body, rows, and quantile levels, and refuses
  a request that would wait longer than `--max-queue`
  ([server.md](server.md#batching)).
- Calculate memory before allocation.
- Reject non-finite request data after the defined missing-value transform.
- Bound concurrent executors and scratch allocations.
- A real deployment requires a proper reverse proxy in front of the server.
  The server bounds only body size, rows, quantile levels, and queue wait; TLS,
  timeouts for slow clients, and per-client limits belong to the proxy.
