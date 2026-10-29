"""Live preprocessing parity against a pinned, unmodified TabICL checkout.

Run explicitly with IRONICL_TABICL_SOURCE=/path/to/tabicl/src and both projects'
dependencies installed. This directory intentionally has no __init__.py: normal
unittest discovery must remain independent of TabICL and PyTorch.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import pickle
import platform
import subprocess
import sys
import unittest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ironicl.preprocessing import ClassifierPreprocessor, _feature_shuffles

UPSTREAM_COMMIT = "bd030d47ff7d9f987f0f33431d743a3003bfe0d3"
SETUP_HELP = (
    "Set IRONICL_TABICL_SOURCE=/path/to/tabicl/src for a clean checkout at "
    f"{UPSTREAM_COMMIT}. Install Ironicl and that checkout in a separate parity "
    "environment; see docs/validation.md."
)


def load_upstream():
    configured = os.environ.get("IRONICL_TABICL_SOURCE")
    if not configured:
        raise RuntimeError(SETUP_HELP)
    source = Path(configured).resolve()
    expected_file = source / "tabicl" / "_sklearn" / "preprocessing.py"
    if not expected_file.is_file():
        raise RuntimeError(f"Missing upstream preprocessing module: {expected_file}. {SETUP_HELP}")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True, stderr=subprocess.PIPE
        ).strip()
        changes = subprocess.check_output(
            ["git", "-C", str(source.parent), "status", "--porcelain", "--", "src/tabicl"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot verify the upstream Git checkout. {SETUP_HELP}") from error
    if revision != UPSTREAM_COMMIT or changes:
        raise RuntimeError(
            f"Expected unmodified TabICL {UPSTREAM_COMMIT}; found {revision}, "
            f"package changes={changes!r}. {SETUP_HELP}"
        )
    sys.path.insert(0, str(source))
    try:
        upstream = importlib.import_module("tabicl._sklearn.preprocessing")
        upstream_version = importlib.import_module("tabicl.__about__").__version__
    except ImportError as error:
        raise RuntimeError(f"Cannot import upstream dependencies: {error}. {SETUP_HELP}") from error
    finally:
        sys.path.remove(str(source))
    if Path(upstream.__file__).resolve() != expected_file.resolve():
        raise RuntimeError(f"Imported TabICL from {upstream.__file__}, not {expected_file}. {SETUP_HELP}")
    print(
        "Parity environment: "
        + json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "upstream_file": str(expected_file),
                "upstream_commit": revision,
                "tabicl": upstream_version,
                **{name: version(name) for name in ("numpy", "scipy", "scikit-learn", "torch")},
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return upstream


def synthetic(dtype=np.float64, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(120, 7))
    q = rng.normal(size=(17, 7))
    x[:, 1] = 7  # Constant in training, but not in queries.
    x[-1, 2] = 1000
    q[-1, 2] = -10000
    if np.issubdtype(dtype, np.integer):
        x, q = x * 10, q * 10
    y = (np.arange(len(x)) % classes) * 7 - 11
    return x.astype(dtype), q.astype(dtype), y


class TabICLPreprocessingParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = load_upstream()
        cls.fit_cases = 0
        cls.shuffle_cases = 0
        cls.comparisons = 0
        cls.max_absolute_error = 0.0
        cls.max_relative_error = 0.0

    @classmethod
    def tearDownClass(cls):
        print(
            "Parity summary: "
            + json.dumps(
                {
                    "completed_fit_cases": cls.fit_cases,
                    "completed_shuffle_cases": cls.shuffle_cases,
                    "array_comparisons": cls.comparisons,
                    "max_absolute_error": cls.max_absolute_error,
                    "max_relative_error": cls.max_relative_error,
                    "comparison": "exact equality after upstream features are converted to FP32",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def assert_equal(self, actual, expected, stage):
        """Report numerical differences without relaxing exact comparisons."""
        actual, expected = np.asarray(actual), np.asarray(expected)
        context = f"{self.case}; stage={stage}; Ironicl.shape={actual.shape}, TabICL.shape={expected.shape}"
        self.assertEqual(actual.shape, expected.shape, f"{context}: shapes differ")
        cls = type(self)
        cls.comparisons += 1
        metrics = ""
        if actual.dtype.kind in "biuf" and expected.dtype.kind in "biuf":
            self.assertTrue(np.isfinite(actual).all(), f"{context}: Ironicl contains non-finite values")
            self.assertTrue(np.isfinite(expected).all(), f"{context}: TabICL contains non-finite values")
            difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            denominator = np.abs(expected.astype(np.float64))
            relative = np.divide(
                difference, denominator, out=np.zeros_like(difference), where=denominator != 0
            )
            relative[(denominator == 0) & (difference != 0)] = np.inf
            absolute_max = float(difference.max(initial=0))
            relative_max = float(relative.max(initial=0))
            cls.max_absolute_error = max(cls.max_absolute_error, absolute_max)
            cls.max_relative_error = max(cls.max_relative_error, relative_max)
            metrics = f", max_abs={absolute_max:.9g}, max_rel={relative_max:.9g}"
        mismatches = np.flatnonzero(actual != expected)
        if mismatches.size:
            index = np.unravel_index(mismatches[0], actual.shape)
            self.fail(
                f"{context}: first differing index={index}, Ironicl={actual[index]!r}, "
                f"TabICL={expected[index]!r}{metrics}"
            )

    def compare_prepared(self, actual, groups, expected, stage, labels=None):
        self.assertEqual(set(actual.group_names_), set(expected), f"{self.case}; stage={stage}: groups")
        self.assertEqual(len(groups), len(actual.group_names_))
        for group, method in enumerate(actual.group_names_):
            self.assertEqual(groups[group].dtype, np.float32)
            self.assertTrue(groups[group].flags.c_contiguous)
            members = np.flatnonzero(actual.ensemble_groups_ == group)
            self.assertEqual(len(members), len(expected[method][0]))
            for position, ensemble in enumerate(members):
                context = f"{stage}, group={method}, ensemble={ensemble}"
                expanded = groups[group][:, actual.feature_permutations_[ensemble]]
                self.assert_equal(expanded, expected[method][0][position].astype(np.float32), context)
                if labels is not None:
                    self.assert_equal(
                        actual.class_permutations_[ensemble][labels],
                        expected[method][1][position],
                        context + ", permuted labels",
                    )

    def run_case(self, name, x, q, y, **parameters):
        options = {
            "n_estimators": 8,
            "random_state": 0,
            "norm_methods": ("none", "power"),
            "outlier_threshold": 4.0,
            **parameters,
        }
        self.case = f"case={name}, dtype={x.dtype}, options={options}"

        # The oracle starts from raw data and its own label encoder. It never
        # uses Ironicl's fitted statistics, labels, masks, or permutations.
        encoder = LabelEncoder().fit(y)
        upstream_labels = encoder.transform(y)
        numerical = self.upstream.TransformToNumerical()
        upstream_x = numerical.fit_transform(x)
        expected = self.upstream.EnsembleGenerator(classification=True, **options).fit(
            upstream_x, upstream_labels
        )
        actual = ClassifierPreprocessor(**options)
        training = actual.fit_transform(x, y)
        labels = actual.encode_labels(y)
        self.assert_equal(actual.classes_, encoder.classes_, "classes")
        self.assert_equal(labels, upstream_labels, "encoded labels")
        self.assert_equal(
            actual.decode_labels(labels), encoder.inverse_transform(upstream_labels), "decoded labels"
        )
        self.assert_equal(actual.feature_mask_, expected.unique_filter_.features_to_keep_, "feature mask")
        self.assertEqual(actual.n_features_out_, expected.n_features_in_)
        canonical_names = tuple(
            method for method in options["norm_methods"] if method in expected.ensemble_configs_
        )
        self.assertEqual(actual.group_names_, canonical_names)
        counts = [len(expected.feature_shuffles_[method]) for method in canonical_names]
        self.assertEqual(actual.n_estimators_, sum(counts))
        self.assert_equal(
            actual.ensemble_groups_, np.repeat(np.arange(len(counts)), counts), "ensemble groups"
        )
        for group, method in enumerate(canonical_names):
            members = np.flatnonzero(actual.ensemble_groups_ == group)
            self.assert_equal(
                actual.feature_permutations_[members],
                expected.feature_shuffles_[method],
                f"group={method}, feature permutations",
            )
            self.assert_equal(
                actual.class_permutations_[members],
                expected.class_shuffles_[method],
                f"group={method}, class permutations",
            )
        expected_training = expected.transform(X=None, mode="train")
        self.compare_prepared(actual, training, expected_training, "training", labels)
        self.compare_prepared(
            actual, actual.transform(x), expected_training, "training transformed again", labels
        )
        expected_queries = expected.transform(numerical.transform(q), mode="test")
        self.compare_prepared(actual, actual.transform(q), expected_queries, "queries")
        type(self).fit_cases += 1
        return actual, encoder, numerical, expected

    def test_synthetic_dtypes_classes_and_ensembles(self):
        for dtype in (np.float32, np.float64, np.int64):
            for seed, estimators, classes in ((0, 1, 2), (42, 8, 3), (123, 32, 10)):
                with self.subTest(dtype=dtype, seed=seed, estimators=estimators, classes=classes):
                    x, q, y = synthetic(dtype, classes, seed)
                    if classes == 3:
                        y = np.array([f"class-{label}" for label in y])
                    self.run_case("synthetic", x, q, y, n_estimators=estimators, random_state=seed)

    def test_normalization_selection_and_order(self):
        for methods in (("none",), ("power",), ("none", "power"), ("power", "none")):
            for seed, estimators, threshold in ((0, 1, 4.0), (91, 5, 1.5)):
                with self.subTest(methods=methods, seed=seed, estimators=estimators, threshold=threshold):
                    x, q, y = synthetic(seed=seed)
                    self.run_case(
                        "normalization",
                        x,
                        q,
                        y,
                        norm_methods=methods,
                        random_state=seed,
                        n_estimators=estimators,
                        outlier_threshold=threshold,
                    )

    def test_fewer_actual_ensembles_and_tiny_training_set(self):
        x = np.arange(24, dtype=np.float64).reshape(-1, 1)
        y = np.where(np.arange(24) % 2, "positive", "negative")
        actual, _, _, _ = self.run_case("one feature", x, x[:3], y, n_estimators=32)
        self.assertEqual(actual.n_estimators_, 4)
        self.run_case("two training rows", np.array([[1.0], [2.0]]), np.array([[1.5]]), [10, 20])

    def test_near_constant_features(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                x, q, y = synthetic(dtype)
                x[:, 0] = 1 + x[:, 0] * 1e-5
                q[:, 0] = 1 + q[:, 0] * 1e-5
                self.run_case("near constant", x, q, y)

    def test_bundled_real_datasets(self):
        for loader in (load_iris, load_breast_cancer, load_digits):
            data = loader()
            x, q, y, _ = train_test_split(
                data.data, data.target, train_size=100, test_size=20, stratify=data.target, random_state=42
            )
            for dtype in (np.float32, np.float64):
                with self.subTest(dataset=loader.__name__, dtype=dtype):
                    self.run_case(loader.__name__, x.astype(dtype), q.astype(dtype), y, random_state=42)

    def test_query_boundaries_and_noncontiguous_arrays(self):
        x, q, y = synthetic(seed=21)
        self.assertFalse(x[::2].flags.c_contiguous)
        actual, _, numerical, expected = self.run_case("noncontiguous", x[::2], q[::2], y[::2])
        batch = actual.transform(q)
        rows = []
        for row in q:
            prepared = actual.transform(row[None, :])
            oracle = expected.transform(numerical.transform(row[None, :]), mode="test")
            self.compare_prepared(actual, prepared, oracle, "one query row")
            rows.append(prepared)
        for group, method in enumerate(actual.group_names_):
            self.assert_equal(
                np.concatenate([row[group] for row in rows]), batch[group], f"group={method}, row vs batch"
            )
        for group, (view, contiguous) in enumerate(
            zip(actual.transform(q[::2]), actual.transform(q[::2].copy()), strict=True)
        ):
            self.assert_equal(view, contiguous, f"group={group}, view vs contiguous")

    def test_serialization_round_trip(self):
        x, q, y = synthetic(seed=17)
        actual, encoder, numerical, expected = self.run_case("serialization", x, q, y, random_state=17)
        restored = pickle.loads(pickle.dumps(actual))
        encoder, numerical, expected = pickle.loads(pickle.dumps((encoder, numerical, expected)))
        self.assert_equal(restored.classes_, encoder.classes_, "restored classes")
        self.assert_equal(restored.encode_labels(y), encoder.transform(y), "restored labels")
        self.compare_prepared(
            restored,
            restored.transform(x),
            expected.transform(X=None, mode="train"),
            "restored training",
            restored.encode_labels(y),
        )
        self.compare_prepared(
            restored,
            restored.transform(q),
            expected.transform(numerical.transform(q), mode="test"),
            "restored queries",
        )
        for group, (before, after) in enumerate(zip(actual.transform(q), restored.transform(q), strict=True)):
            self.assert_equal(after, before, f"group={group}, restored vs original")

    def test_power_domain_fallback_is_batch_wide(self):
        x, q, y = synthetic(seed=13)
        actual, _, numerical, expected = self.run_case("power fallback", x, q, y, norm_methods=("power",))
        ours = actual.normalizers_[0]
        theirs = expected.preprocessors_["power"]
        # Induce a domain failure deterministically, rather than depend on a
        # particular SciPy overflow. The second row is outside training bounds
        # but inside the injected domain, exposing batch-wide clipping.
        q = np.tile(np.mean(x, axis=0), (2, 1))
        mean = np.mean(x[:, 0])
        scale = np.std(x[:, 0]) + 1e-6
        maximum = np.max((x[:, 0] - mean) / scale)
        q[:, 0] = mean + scale * (maximum + np.array([20.0, 1.0]))

        def bounded_transform(transform, maximum):
            def apply(values):
                if np.any(values[:, 0] > maximum + 5):
                    raise ValueError("test-only power-transform domain failure")
                return transform(values)

            return apply

        with (
            patch.object(
                ours.power, "transform", side_effect=bounded_transform(ours.power.transform, ours.maximum[0])
            ) as our_transform,
            patch.object(
                theirs.normalizer_,
                "transform",
                side_effect=bounded_transform(theirs.normalizer_.transform, theirs.X_max_[0, 0]),
            ) as their_transform,
        ):
            batch = actual.transform(q)
            oracle = expected.transform(numerical.transform(q), mode="test")
            self.compare_prepared(actual, batch, oracle, "domain fallback")
            self.assertEqual(our_transform.call_count, 2)
            self.assertEqual(their_transform.call_count, 2)
            single = actual.transform(q[1:])
            self.compare_prepared(
                actual,
                single,
                expected.transform(numerical.transform(q[1:]), mode="test"),
                "no fallback for second row alone",
            )
            self.assertFalse(
                np.array_equal(batch[0][1], single[0][0]), "Fallback should clip the entire batch"
            )

    def test_latin_and_large_feature_shuffles(self):
        self.case = "feature shuffles"
        for features in (1, 2, 5, 17, 4001):
            for estimators in (1, 8):
                for seed in (0, 42):
                    with self.subTest(features=features, estimators=estimators, seed=seed):
                        self.assert_equal(
                            _feature_shuffles(features, estimators, seed),
                            self.upstream.Shuffler(features, random_state=seed).shuffle(estimators),
                            f"H={features}, estimators={estimators}, seed={seed}",
                        )
                        type(self).shuffle_cases += 1

    def test_hash_seed_probe(self):
        x, q, y = synthetic(seed=42)
        actual, _, numerical, expected = self.run_case("hash seed", x, q, y, random_state=42)
        prepared = actual.transform(q)
        oracle = expected.transform(numerical.transform(q), mode="test")
        ours, theirs = [], []
        for group, method in enumerate(("none", "power")):
            members = np.flatnonzero(actual.ensemble_groups_ == group)
            ours.append(
                [
                    method,
                    actual.feature_permutations_[members].tolist(),
                    actual.class_permutations_[members].tolist(),
                    [prepared[group][:, actual.feature_permutations_[member]].tolist() for member in members],
                ]
            )
            theirs.append(
                [
                    method,
                    np.asarray(expected.feature_shuffles_[method]).tolist(),
                    np.asarray(expected.class_shuffles_[method]).tolist(),
                    oracle[method][0].astype(np.float32).tolist(),
                ]
            )

        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

        self.assertEqual(digest(ours), digest(theirs))
        print(
            "Hash-seed signature: " + json.dumps({"ironicl": digest(ours), "tabicl": digest(theirs)}),
            flush=True,
        )

    def test_python_hash_seed_independence(self):
        signatures = []
        for seed in ("0", "1", "42"):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "TabICLPreprocessingParityTest.test_hash_seed_probe",
                    "-v",
                ],
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(
                completed.returncode, 0, f"PYTHONHASHSEED={seed}\n{completed.stdout}\n{completed.stderr}"
            )
            lines = [
                line.removeprefix("Hash-seed signature: ")
                for line in completed.stdout.splitlines()
                if line.startswith("Hash-seed signature: ")
            ]
            self.assertEqual(len(lines), 1, completed.stdout)
            signatures.append(json.loads(lines[0]))
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[1], signatures[2])


if __name__ == "__main__":
    unittest.main()
