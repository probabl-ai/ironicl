# Native Preprocessing

Preprocessing today is Python: `ironicl.preprocessing.Preprocessor`
([python.md](python.md#preprocessing)) fits scaling, the power transform,
outlier bounds, and the feature and class ensembles, then hands the Rust engine
grouped FP32 matrices plus `uint64` permutation arrays. Native preprocessing
moves that step into the Rust crate so a fitted classifier needs no scientific
Python stack at runtime. It is the open [roadmap Phase 3](roadmap.md#phase-3-static-full-kv-context)
item "fixed native preprocessing schema and implementation."

**Decision: Candidate, gated on dropping the runtime NumPy/scikit-learn/SciPy
dependency.** The forward pass dominates fitting; preprocessing is not the
bottleneck, so a native port buys no measurable speed. Its one payoff is a lean,
pure-Rust wheel. Until that is a goal, Python preprocessing stays — it gets
CPython's RNG and scikit-learn's power transform for free, which is exactly what
the port must otherwise reproduce.

## Parity contract to preserve

The live parity test fits the real TabICL at the same `random_state` and requires
`predict_proba` within `1e-4` and identical labels
([validation.md](validation.md#python-preprocessing-parity)). Meeting that bound
means reproducing TabICL's exact ensemble members, which rests on two
deterministic Python behaviors. A native port must reproduce both, or the
parity contract has to be relaxed to functional equivalence (same preprocessing
math and ensemble size, statistically matching predictions) first — a separate
product decision, not an implementation detail.

## Reproduction surface 1: the permutations

Feature Latin-square shuffles and cyclic class permutations come from CPython's
`random.Random(seed)` through `choice`, `shuffle`, and `sample`
(`preprocessing.py`, `_feature_shuffles`). Only `random()` — the raw MT19937
stream — is guaranteed stable across CPython versions and platforms. The
selection methods are not promised, though they have been stable since Python
3.2 because they route through `_randbelow`.

This surface is largely solved by existing work, not a from-scratch port:

- The [`mt19937` crate](https://docs.rs/mt19937/) reproduces `random.random()`
  after seeding, and is the generator [RustPython](https://github.com/RustPython/mt19937)
  uses to reimplement the whole `random` module, including `shuffle`, `sample`,
  and `choice`.
- Remaining work: adopt the crate, lift or verify the `_randbelow` consumption
  layer, and confirm integer seeding (`init_by_array`, the one detail the crate
  flags), then freeze golden permutation vectors generated from the pinned
  TabICL commit as Rust fixtures.

Because parity is against a pinned upstream commit
(`bd030d47ff7d9f987f0f33431d743a3003bfe0d3`) on a pinned Python, the oracle is
frozen, not moving. Golden vectors encode it; a future CPython change to the
selection methods surfaces as a fixture mismatch, never a silent divergence.

## Reproduction surface 2: the power transform

`_Normalization` wraps scikit-learn's `PowerTransformer` (Yeo-Johnson): a
per-feature lambda fitted by maximum likelihood, plus the out-of-domain clip
fallback Ironicl preserves. There is no clean existing Rust crate for this, so it
is the real lift of a native port. Within the `1e-4` exact-seed bound the lambda
search must track scikit-learn closely; within functional parity a correct
Yeo-Johnson suffices, since it is a smooth deterministic transform. This piece,
not the RNG, decides the cost.

## The mechanical remainder

Input validation, constant-feature masking, scale-and-clip, the two-pass outlier
clamp, label encoding, and FP32 stacking are straightforward arithmetic and carry
no parity risk. They are not the reason the port is deferred.

## Rejected: a partial port

Generating permutations in Rust while leaving the power transform in Python is
rejected. It splits the parity surface across the FFI boundary, keeps two places
to maintain, and still ships scikit-learn — the worst of both. The port is
all-or-nothing: either both surfaces move and the scientific-Python runtime
dependency goes away, or preprocessing stays in Python.

## References

- [python.md](python.md) — the current Python preprocessing package.
- [roadmap.md](roadmap.md#phase-3-static-full-kv-context) — Phase 3, where this
  lands.
- [`mt19937` crate](https://docs.rs/mt19937/) and
  [RustPython/mt19937](https://github.com/RustPython/mt19937) — CPython-compatible
  RNG prior art.
- CPython `random` "Notes on Reproducibility" — the `random()`-only stability
  guarantee.
