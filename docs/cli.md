# Command-Line Interface

## Command Shape

Ironicl ships one binary. Each command takes its file or directory as the first argument.

```text
ironicl help [COMMAND]
ironicl platform
ironicl inspect PATH [--model tabicl|kumo-tabular] [--weights DIR]
ironicl import CHECKPOINT [--model tabicl|kumo-tabular] [-o FILE]
ironicl prepare DIR [--weights DIR] [--optimize safe|auto|fast]
ironicl serve DIR [--weights DIR] [--optimize safe|auto|fast] [--bind ADDR]
              [--max-request-rows N] [--max-body SIZE] [--max-batch-rows N] [--max-queue MS]
```

Use `ironicl help COMMAND` or `ironicl COMMAND --help` for command-specific help, and
`ironicl --version` for the version.

The CLI does not fit yet. Fit with the [Python estimators](python.md) and write a model directory
with `export`.

## Model Directories and Weights

A model directory holds one fitted model: its metadata (`ironicl.json`), its preprocessing with
the training rows (`preprocessing.irprep`), and optionally its KV cache (`context.ircntxt`); see
[python.md](python.md#exporting-for-the-server). It holds no weights. The weights stay in a shared
folder as `<checkpoint SHA-256>.irmodel` files, so many models share one copy.

`--weights DIR` names that folder. By default it is the folder that loading a checkpoint fills:
`~/.cache/ironicl/models/v1`, or `$XDG_CACHE_HOME/ironicl/models/v1` when `XDG_CACHE_HOME` is set.
`export` and `import` write there. On another machine, import the checkpoint once, or copy the
folder and pass `--weights`.

## Platform

```bash
ironicl platform
```

`platform` prints the host architecture, operating system, the CPU features Ironicl dispatches on
(NEON, AVX2, AVX-512 subsets), and the instruction set the kernels select.

## Inspect

```bash
ironicl inspect my-model
ironicl inspect tabicl-classifier-v2-20260212.ckpt
ironicl inspect large-classifier.pt
ironicl inspect large-classifier.pt --model kumo-tabular
ironicl inspect ~/.cache/ironicl/models/v1/<sha256>.irmodel
```

`inspect` tells what `PATH` is:

- a model directory: its task, features and their names, classes, the weights it needs and
  whether they are in the weights folder, and the KV cache's level and size;
- Ironicl weights (`IRMODEL`): full structural and checksum validation, then the source SHA-256,
  profile, tensor and parameter counts, dtypes, and configuration;
- a PyTorch ZIP checkpoint (`.ckpt`, `.pt`, `.pth`), parsed without executing pickle code: the
  same, with the source SHA-256 bound to the recognized official model for TabICLv2. A checkpoint
  with a config is TabICLv2's; one without is [Kumo-Tabular](kumo-tabular.md)'s, whose size and
  task are those of its tensors. `--model` makes `inspect` check that the checkpoint is that
  model's instead.

## Import

```bash
ironicl import tabicl-classifier-v2-20260212.ckpt
ironicl import tabicl-classifier-v2-20260212.ckpt --model tabicl -o classifier.irmodel
ironicl import large-classifier.pt --model kumo-tabular
```

`import` defaults to `tabicl`. For Kumo-Tabular, pass `--model kumo-tabular`; the checkpoint's
tensors must match one release and determine its size and task.

`import` validates the checkpoint, keeps its source identity, copies the tensors into a
deterministic, 64-byte-aligned portable layout, writes atomically, and verifies the complete
weights before it returns. Without `-o`, it writes into the shared weights folder, where `prepare`
and `serve` look; an import that is already there is kept. The weights hold no preprocessing state
and no dataset-specific caches; see [artifact-format.md](artifact-format.md).

## Prepare

```bash
ironicl prepare my-model --optimize auto
```

`prepare` makes sure that `my-model/context.ircntxt` is the KV cache this model builds at the
level: it reuses a saved cache that is exactly that one, and otherwise builds it from the training
rows on this machine's device and replaces the file atomically. A saved cache that cannot be read
is rebuilt too, with a warning. The level defaults to the saved cache's, else `safe`. Running it
twice does nothing the second time. Use it on the serving machine,
for example in a container build, so that `serve` starts at once. If the cache cannot be saved,
`prepare` exits with an error. `serve` instead logs a warning and serves the in-memory model.

`IRONICL_MEMORY_LIMIT` and `IRONICL_DEVICE_MEMORY_LIMIT` set TabICLv2's build budgets. For a
Kumo-Tabular directory, `prepare` and `serve` reject either limit: Kumo-Tabular does not yet
support memory budgets.

## Serve

```bash
ironicl serve my-model
```

`serve` prepares the model as `prepare` does, then serves it over HTTP; it needs the `server`
feature. It predicts concurrent requests together, up to `--max-batch-rows` rows at once, and
refuses a request that would wait longer than `--max-queue` milliseconds. See
[server.md](server.md).

## Stability

The command names are intended to stay stable. The CLI is parsed with `clap`: an unknown option or
an invalid value exits with status 2 and a message naming the option.
