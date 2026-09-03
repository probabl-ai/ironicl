# TabICLv2 Model Reference

## Released Models

| Model | Parameters | FP32 tensor bytes |
|---|---:|---:|
| Classifier v2 | 27,552,258 | 110,209,032 |
| Regressor v2 | 28,544,991 | 114,179,964 |

The classifier checkpoint container is 110,368,038 bytes. The regressor
checkpoint container is 114,324,594 bytes. Container size is not the in-memory
payload contract.

## Architecture Constants

- Input feature group size: 3.
- Column embedding width: 128.
- Column induced-attention blocks: 3.
- Column attention heads: 8.
- Inducing points per column block: 128.
- Reserved row CLS tokens: 4.
- Row-transformer blocks: 3.
- Row-transformer heads: 8.
- Flattened row representation: `4 * 128 = 512`.
- ICL-transformer blocks: 12.
- ICL-transformer heads: 8.
- ICL feed-forward width: 1,024.
- Native classifier outputs: 10.
- Regressor quantile outputs: 999.
- Activation: GELU.
- Transformer normalization: pre-norm.
- Scalable softmax: query-aware, elementwise QASSMax.
- Row positional encoding: non-interleaved RoPE with base 100,000.

The checkpoint configuration, not hard-coded defaults, remains authoritative.
An importer must reject unsupported architecture changes.

## Inference Stages

1. Preprocess and create ensemble variants.
2. Project grouped scalar features into width-128 tokens.
3. Run three induced column-attention blocks across training rows.
4. Run three row-transformer blocks across feature tokens within each row.
5. Flatten four CLS outputs into a width-512 row representation.
6. Add target embeddings to training representations.
7. Run twelve ICL blocks where all queries attend only to training K/V.
8. Decode test representations and aggregate ensemble outputs.

## Exact Test-Row Streaming

With `embed_with_test=false`, test rows do not attend to other test rows:

- Column stage: inducing points attend only to training rows. Each test row then
  attends to the training-derived inducing state.
- Row stage: attention is across features inside one row. The row dimension is a
  batch dimension.
- ICL stage: each test query attends only to training K/V.
- LayerNorm and model nonlinearities are token-local.

Therefore, processing cached test rows individually is mathematically equivalent
to processing them in a batch, apart from floating-point reduction order and
external batch-wide preprocessing behavior.

## Boundaries and Exceptions

- The current cached classifier path supports at most 10 classes.
- Batch-wide all-NaN feature masking in the sklearn wrapper can change behavior.
  Ironicl requires an explicit, fixed feature schema for serving.
- A change to a training feature or label invalidates later-layer cache state.
- Adding a training row is not an exact append-only operation because training
  tokens attend to the full training context at every ICL layer.
- Representation-cache mode is exact but recomputes the ICL stack for every
  request. It is a memory option, not a low-latency substitute for full K/V.
