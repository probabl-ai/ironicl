# Command-Line Interface

## Command Shape

Ironicl ships one binary with four commands. Model commands take named inputs, not positional file
paths.

```text
ironicl help [COMMAND]
ironicl platform
ironicl inspect (--checkpoint FILE | --model FILE)
ironicl import --checkpoint FILE --output FILE
```

Use `ironicl help COMMAND` or `ironicl COMMAND --help` for command-specific help.

The CLI does not fit or predict yet. Use the [Python estimators](python.md) for that. A CLI that
takes raw rows, as the estimators do, is planned; see the [roadmap](roadmap.md).

## Source Flags

- `--checkpoint FILE` selects a PyTorch ZIP checkpoint such as `.ckpt`, `.pt`, or `.pth`.
- `--model FILE` selects a validated Ironicl model artifact (`IRMODEL`).
- `inspect` requires exactly one source flag.
- `--output FILE` names a newly written artifact. It is not a model source.

## Platform

```bash
ironicl platform
```

`platform` prints the host architecture, operating system, the CPU features Ironicl dispatches on
(NEON, AVX2, AVX-512 subsets), and the instruction set the kernels select.

## Inspect

```bash
ironicl inspect --checkpoint tabicl-classifier-v2-20260212.ckpt
ironicl inspect --model classifier.irmodel
```

Checkpoint inspection parses the checkpoint without executing pickle code and reports:

- the source SHA-256;
- the recognized model profile, bound to the official source SHA-256;
- tensor and parameter counts;
- tensor payload bytes and dtypes;
- selected TabICLv2 configuration values.

Model inspection applies full structural and checksum validation before it reports.

## Import

```bash
ironicl import \
  --checkpoint tabicl-classifier-v2-20260212.ckpt \
  --output classifier.irmodel
```

`import` validates the checkpoint, keeps its source identity, copies the tensors into a
deterministic, 64-byte-aligned portable layout, writes atomically, and verifies the complete
artifact before it returns. The model holds no preprocessing state and no dataset-specific
caches; see [artifact-format.md](artifact-format.md).

The Python estimators import checkpoints themselves, through a content-addressed cache, so a
manual import is only needed to inspect or ship a model artifact.

## Stability

The command names and the source-flag convention are intended to stay stable. New commands and
options arrive as named flags once their data contracts exist.
