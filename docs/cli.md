# Command-Line Interface

## Command Shape

Ironicl uses one binary and explicit named inputs. Model commands do not accept
positional file paths.

```text
ironicl platform
ironicl plan [options]
ironicl inspect (--checkpoint FILE | --model FILE)
ironicl import --checkpoint FILE --output FILE
ironicl run (--checkpoint FILE | --model FILE) --input FILE [cache options]
```

Use `ironicl help COMMAND` or `ironicl COMMAND --help` for command-specific
help.

## Source Flags

- `--checkpoint FILE` selects a PyTorch ZIP checkpoint such as `.ckpt`, `.pt`,
  or `.pth`.
- `--model FILE` selects a validated Ironicl model artifact.
- Commands that accept both require exactly one source flag.
- `--output FILE` names a newly generated artifact. It is not a model source.

This convention keeps development execution and optimized execution visibly
different without changing command names.

## Inspect

```bash
ironicl inspect --checkpoint tabicl-classifier-v2-20260212.ckpt
ironicl inspect --model classifier.ironicl
```

Checkpoint inspection is implemented. It parses without executing pickle code
and reports:

- source SHA-256;
- recognized model profile, bound to the official source SHA-256;
- tensor and parameter counts;
- tensor payload bytes and dtypes;
- selected TabICLv2 configuration values.

Portable model artifact inspection is implemented and applies full structural
and checksum validation.

## Import

```bash
ironicl import \
  --checkpoint tabicl-classifier-v2-20260212.ckpt \
  --output classifier.ironicl
```

Import validates the checkpoint, preserves source identity, copies tensors into
a deterministic 64-byte-aligned portable layout, writes atomically, and verifies
the complete artifact before returning. This portable model does not include
preprocessing state or dataset-specific K/V caches.

## Run

`--input FILE` selects a validated classifier input bundle. It contains the
preprocessed feature tensor and training labels for one execution request.

Development path:

```bash
ironicl run --checkpoint tabicl-classifier-v2-20260212.ckpt --input request.irinput
```

Optimized path:

```bash
ironicl run --model classifier.ironicl --input request.irinput
```

Cached query-only path:

```bash
ironicl run \
  --model classifier.ironicl \
  --context context.ironiclctx \
  --input request.irquery
```

When `--context` is present, `--input` must be an `IRQUERY` v1 bundle. Its model
and context identities, ensemble count, and feature count must match the static
context. Cached execution returns per-estimator `[B,Q,C]` logits. It does not
apply class permutations or ensemble aggregation.

Direct checkpoint execution may convert weights in memory and is not the
deployment performance path. By default, `run --checkpoint` imports to a
content-addressed cache and reuses the verified model on later runs. Cache hits
hash the checkpoint but do not parse pickle again.

Checkpoint cache options:

- `--no-cache` validates a checkpoint without persistent import. `run` rejects
  this option because execution requires an imported artifact.
- `--rebuild-cache` replaces an existing cached model.
- `--cache-dir DIR` overrides `$XDG_CACHE_HOME/ironicl` or
  `$HOME/.cache/ironicl`.

`run` loads the input bundle, executes classifier logits, and writes a stable
text result. The first line is `shape: B test_rows C`. Each remaining line is
`logits[ensemble,test_row]:` followed by one FP32 logit per class in scientific
notation.

## Stability

The command names and source-flag convention are intended to remain stable.
Additional preprocessing, precision, and output options will be added as named
flags after their data contracts are implemented.
