# Precision Policy

## Reference Policy

The correctness reference uses FP32 model arithmetic and FP32 K/V. Preprocessing
uses the same effective precision as the exported sklearn pipeline, including
FP64 statistics where required for parity.

The released checkpoints contain FP32 weights. Reduced formats are derived
artifacts, not original model representations.

## Recommended First Optimized Policy

```text
large linear weights: BF16 storage
K/V cache:            BF16 storage
dot accumulation:     FP32
residual stream:       FP32
sensitive operations: FP32
final outputs:         FP32
```

`AVX512_BF16` provides BF16 pair dot products accumulated into FP32. Runtime
CPUID detection is required. Code must not infer feature subsets only from a CPU
model name.

## Operations Kept in FP32

- preprocessing transforms and reductions;
- LayerNorm mean, variance, reciprocal square root, affine scale, and bias;
- QASSMax `log(n)`, MLPs, `tanh`, and query scaling;
- RoPE frequency, sine/cosine, and rotation;
- Q-by-K score accumulation and scaling;
- masks and stable softmax;
- probability-by-V accumulation;
- GELU;
- residual additions;
- decoder output;
- classifier temperature and softmax;
- regression quantile postprocessing;
- ensemble reduction.

This mixed policy is a requirement until validation proves a narrower FP32
boundary is safe. It describes the `safe` level and the x86 path. On Apple
Silicon under `optimize=auto` or `fast`, ICL attention (Q-by-K scores, stable
softmax, and probability-by-V accumulation) runs in FP16 on the GPU, under the
FP16 gate below.

## FP32 Variant

Advantages:

- closest to current PyTorch CPU behavior;
- no cache quantization error;
- simplest parity debugging.

Costs:

- twice the BF16 weight and cache traffic;
- a 50K-row, 100-feature, eight-estimator full cache is about 18.62 GiB;
- lower single-query throughput when memory-bound.

## BF16 Variant

Advantages:

- half of FP32 storage and memory traffic;
- native BF16 dot-product support on the target Zen 5 CPU;
- enough exponent range for model values.

Risks:

- only seven explicit mantissa bits;
- K rounding changes logits before softmax;
- V rounding changes the weighted result;
- request output will not match FP32 at the existing `1e-4` cache tolerance in
  every case without evidence.

BF16 K/V must be validated separately from BF16 weights. Do not describe it as
lossless or exact relative to FP32.

## INT8 Weights

Candidate policy:

- symmetric per-output-channel weight scales;
- dynamic per-token activation scales where VNNI kernels require INT8 inputs;
- INT32 accumulation followed by FP32 dequantization and bias;
- FP32 residual stream.

INT8 weights reduce classifier weights from about 52.55 MiB in BF16 to 26.28
MiB. Since one BF16 ICL layer already fits in L3, this may improve compute but
will not solve K/V bandwidth.

## INT8 K/V

INT8 K/V is an experiment:

- K error changes attention ranking and is amplified by softmax.
- V error is approximately linear after probabilities are known.
- Per-tensor scales are unlikely to be sufficient.
- Per-row or small-tile scales add metadata traffic and conversion work.
- VNNI does not directly solve FP32-softmax by INT8-V accumulation.

Required experiment order:

1. BF16 K and BF16 V.
2. BF16 K and INT8 V.
3. INT8 K and BF16 V.
4. INT8 K and INT8 V.

No INT8 cache policy becomes a default without end-to-end benchmark accuracy
results across classification and regression datasets.

## FP16 and the optimize levels

On the x86 Zen 5 path FP16 is not used: BF16 offers more exponent range and
matches the intended dot-product path.

On Apple Silicon, FP16 is a shipped GPU storage policy, selected by the
`optimize` level:

- `safe` keeps FP32 arithmetic and the FP32/BF16 storage above;
- `auto` and `fast` store the ICL keys and values in FP16 and run ICL attention
  in FP16 on the Metal GPU, converting only the queries.

The level is set through the Python API or the `IRONICL_OPTIMIZE` environment
variable. FP16 overflows above 65504, so the reduced-precision levels carry their
own accuracy gate against the `safe` output, not the FP32 `1e-4` tolerance. The
gate and its measurements are in
[experiments.md](experiments.md#fp16-gate-for-the-optimize-levels).
