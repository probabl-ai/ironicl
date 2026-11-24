# Kumo-Tabular

[Kumo-Tabular](https://huggingface.co/nvidia/Kumo-Tabular) is NVIDIA's tabular foundation model.
At size `large` it is second on [TabArena](https://huggingface.co/spaces/TabArena/leaderboard),
and first among models whose weights allow commercial use. Ironicl is adding it as its second
model, next to TabICLv2.

Status: Ironicl imports and inspects all six checkpoints. Fitting and predicting come next.

## Checkpoints

NVIDIA releases a classifier and a regressor in three sizes, as FP32 PyTorch files. Ironicl pins
revision `c5893d418e162b9f841ae179dc767afefff73831` of `nvidia/Kumo-Tabular` and recognizes each
file by its SHA-256 (`src/models/kumo_tabular/mod.rs`). The architecture values come from
[`model.py`](https://github.com/NVIDIA/structured-data-models/blob/3fdf108a1f1b719b1b983373b71bbba79fcab7e6/sdm/models/kumo/tabular/model.py#L24-L64),
not from the checkpoint.

| Size | Cell channels | Column and row layers | Inducing points | ICL channels | ICL layers | ICL heads | Classifier | Regressor |
|---|---|---|---|---|---|---|---|---|
| small | 128 | 4 | 128 | 512 | 12 | 8 | 110 MB | 114 MB |
| medium | 256 | 6 | 256 | 512 | 24 | 8 | 246 MB | 250 MB |
| large | 256 | 6 | 256 | 1024 | 24 | 16 | 855 MB | 863 MB |

The classifier outputs 10 logits and the regressor 999 quantiles. In medium and large, test rows
attend to the training rows through 2 of the ICL key and value heads.

```bash
ironicl inspect large/classifier.pt
ironicl import large/classifier.pt
```

Import checks that a checkpoint holds exactly its release's tensors, by name and shape, and stores
them as `IRMODEL` with architecture 2 ([artifact-format.md](artifact-format.md#model)). The label
embeddings are saved transposed; import copies them into row-major order.

## License

The weights are under OpenMDW-1.1, which allows commercial use, modification, conversion, and
redistribution. Distributing the weights, converted or not, requires the license text and NVIDIA's
notices. Ironicl downloads the weights from Hugging Face and never ships them. NVIDIA's code,
`structured-data-models`, is Apache-2.0; Ironicl reimplements it and refers to it by pinned links.

## Plan

1. Import the weights. Done.
2. The classifier on the CPU: preprocessing, the forward pass, and fitting and predicting from a
   cached context, as `KumoTabularClassifier`, with exports, pickling, and the server.
3. The regressor on the CPU, as `KumoTabularRegressor`.
4. Metal.
5. CUDA.

Parity: fixtures from NVIDIA's code, pinned to commit `3fdf108`, run on the CPU in FP32, record
each member's random draws, preprocessed inputs, stage outputs, and predictions. Ironicl makes its
own draws, with the Mersenne Twister and its own samplers, so a `random_state` gives the same
ensemble on every device and in every release; the tests feed it NVIDIA's draws instead, to check
the rest exactly.
