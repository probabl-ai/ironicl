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
import platform
import subprocess
import sys
import unittest
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ironicl import _core
from ironicl.base import _array, _features, _label_values, _labels

# See docs/native-preprocessing.md#parity-contract for this bound.
RTOL = ATOL = 2e-6
DEFAULTS = {"n_estimators": 8, "random_state": 0, "norm_methods": ("none", "power"), "outlier_threshold": 4.0}


def preprocess(x, y, classification=True, **options):
    """Fit Ironicl's Rust preprocessing as the estimators' ``fit`` does."""
    options = {**options, "norm_methods": list(options["norm_methods"])}
    x = _features(x)[0]
    if classification:
        return _core.preprocess(x, labels=_label_values(_labels(y)), **options)
    return _core.preprocess(x, values=np.ascontiguousarray(y, dtype=np.float32), **options)


def upcast(x):
    """What TabICL is fed: Ironicl computes in float64, so float32 input matches TabICL on its upcast."""
    return x.astype(np.float64) if x.dtype == np.float32 else x


TABICL_COMMIT = "bd030d47ff7d9f987f0f33431d743a3003bfe0d3"
SETUP_HELP = (
    "Set IRONICL_TABICL_SOURCE=/path/to/tabicl/src for a clean checkout at "
    f"{TABICL_COMMIT}. Install Ironicl and that checkout in a separate parity "
    "environment; see docs/validation.md."
)


def load_tabicl():
    configured = os.environ.get("IRONICL_TABICL_SOURCE")
    if not configured:
        raise RuntimeError(SETUP_HELP)
    source = Path(configured).resolve()
    expected_file = source / "tabicl" / "_sklearn" / "preprocessing.py"
    if not expected_file.is_file():
        raise RuntimeError(f"Missing TabICL preprocessing module: {expected_file}. {SETUP_HELP}")
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
        raise RuntimeError(f"Cannot verify the TabICL Git checkout. {SETUP_HELP}") from error
    if revision != TABICL_COMMIT or changes:
        raise RuntimeError(
            f"Expected unmodified TabICL {TABICL_COMMIT}; found {revision}, "
            f"package changes={changes!r}. {SETUP_HELP}"
        )
    sys.path.insert(0, str(source))
    try:
        tabicl = importlib.import_module("tabicl._sklearn.preprocessing")
        tabicl_version = importlib.import_module("tabicl.__about__").__version__
    except ImportError as error:
        raise RuntimeError(f"Cannot import TabICL's dependencies: {error}. {SETUP_HELP}") from error
    finally:
        sys.path.remove(str(source))
    if Path(tabicl.__file__).resolve() != expected_file.resolve():
        raise RuntimeError(f"Imported TabICL from {tabicl.__file__}, not {expected_file}. {SETUP_HELP}")
    print(
        "Parity environment: "
        + json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "tabicl_file": str(expected_file),
                "tabicl_commit": revision,
                "tabicl": tabicl_version,
                **{name: version(name) for name in ("numpy", "scipy", "scikit-learn", "torch")},
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return tabicl


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


def skewed(dtype=np.float64):
    """Heavy-tailed columns spanning many orders of magnitude, where TabICL's
    float32 path loses precision before Yeo-Johnson."""
    x, y = make_classification(1200, 30, n_informative=12, n_classes=5, random_state=0)
    rng = np.random.default_rng(0)
    x = np.exp(x * rng.uniform(0.2, 1.5, x.shape[1])) * rng.uniform(0.1, 1e4, x.shape[1])
    return x[:1000].astype(dtype), x[1000:].astype(dtype), y[:1000]


class TabICLPreprocessingParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tabicl = load_tabicl()
        cls.fit_cases = 0
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
                    "array_comparisons": cls.comparisons,
                    "max_absolute_error": cls.max_absolute_error,
                    "max_relative_error": cls.max_relative_error,
                    "comparison": f"exact ensembles and labels; values within rtol={RTOL}, atol={ATOL}",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def record(self, actual, expected, context):
        """Check shapes and finiteness, and count the comparison and its errors."""
        context = f"{self.case}; stage={context}; Ironicl.shape={actual.shape}, TabICL.shape={expected.shape}"
        self.assertEqual(actual.shape, expected.shape, f"{context}: shapes differ")
        cls = type(self)
        cls.comparisons += 1
        if actual.dtype.kind not in "biuf" or expected.dtype.kind not in "biuf":
            return context, ""
        self.assertTrue(np.isfinite(actual).all(), f"{context}: Ironicl contains non-finite values")
        self.assertTrue(np.isfinite(expected).all(), f"{context}: TabICL contains non-finite values")
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        denominator = np.abs(expected.astype(np.float64))
        relative = np.divide(difference, denominator, out=np.zeros_like(difference), where=denominator != 0)
        relative[(denominator == 0) & (difference != 0)] = np.inf
        absolute_max, relative_max = float(difference.max(initial=0)), float(relative.max(initial=0))
        cls.max_absolute_error = max(cls.max_absolute_error, absolute_max)
        cls.max_relative_error = max(cls.max_relative_error, relative_max)
        return context, f", max_abs={absolute_max:.9g}, max_rel={relative_max:.9g}"

    def assert_equal(self, actual, expected, stage):
        """Exact equality, for labels, masks, and permutations."""
        actual, expected = np.asarray(actual), np.asarray(expected)
        context, metrics = self.record(actual, expected, stage)
        mismatches = np.flatnonzero(actual != expected)
        if mismatches.size:
            index = np.unravel_index(mismatches[0], actual.shape)
            self.fail(f"{context}: first differing index={index}, Ironicl={actual[index]!r}, TabICL={expected[index]!r}{metrics}")

    def assert_close(self, actual, expected, stage):
        """Preprocessed values: within RTOL and ATOL, as the frozen fixture checks them."""
        actual, expected = np.asarray(actual), np.asarray(expected, dtype=np.float32)
        context, metrics = self.record(actual, expected, stage)
        outside = np.abs(actual.astype(np.float64) - expected) > ATOL + RTOL * np.abs(expected.astype(np.float64))
        if outside.any():
            index = np.unravel_index(np.flatnonzero(outside)[0], actual.shape)
            self.fail(f"{context}: first index beyond tolerance={index}, Ironicl={actual[index]!r}, TabICL={expected[index]!r}{metrics}")

    def compare_rows(self, preprocessed, rows, expected, stage, targets=None):
        """Each member's ``[rows, H]`` values, and its targets, against TabICL's by group name."""
        self.assertEqual(set(preprocessed.group_names), set(expected), f"{self.case}; stage={stage}: groups")
        groups = np.asarray(preprocessed.member_groups)
        for group, method in enumerate(preprocessed.group_names):
            members = np.flatnonzero(groups == group)
            self.assertEqual(len(members), len(expected[method][0]))
            self.assert_close(rows[members], expected[method][0], f"{stage}, group={method}")
            if targets is not None:
                self.assert_equal(targets[members], expected[method][1], f"{stage}, group={method}, permuted labels")

    def run_case(self, name, x, q, y, **parameters):
        options = {**DEFAULTS, **parameters}
        self.case = f"case={name}, dtype={x.dtype}, options={options}"

        # The oracle starts from raw data and its own label encoder. It never
        # uses Ironicl's fitted statistics, labels, masks, or permutations.
        encoder = LabelEncoder().fit(y)
        tabicl_labels = encoder.transform(y)
        numerical = self.tabicl.TransformToNumerical()
        tabicl_x = numerical.fit_transform(upcast(x))
        expected = self.tabicl.EnsembleGenerator(classification=True, **options).fit(tabicl_x, tabicl_labels)
        preprocessed = preprocess(x, y, **options)
        # The encoded labels are checked through the permuted training targets below.
        self.assert_equal(np.unique(_labels(y)), encoder.classes_, "classes")
        # TabICL 2.1.1's imputer drops a feature missing in every row before its
        # filter sees it; Ironicl's mask drops it.
        empty = np.isnan(upcast(x)).all(axis=0)
        mask = np.zeros(len(empty), dtype=bool)
        mask[~empty] = expected.unique_filter_.features_to_keep_
        self.assert_equal(preprocessed.feature_mask, mask, "feature mask")
        members, training_rows, features = preprocessed.shape
        self.assertEqual(features, expected.n_features_in_)
        canonical_names = tuple(method for method in options["norm_methods"] if method in expected.ensemble_configs_)
        self.assertEqual(tuple(preprocessed.group_names), canonical_names)
        counts = [len(expected.feature_shuffles_[method]) for method in canonical_names]
        self.assertEqual(members, sum(counts))
        groups = np.asarray(preprocessed.member_groups)
        self.assert_equal(groups, np.repeat(np.arange(len(counts)), counts), "member groups")
        feature_orders = _array(*preprocessed.feature_permutations, np.uint64)
        class_orders = _array(*preprocessed.class_permutations, np.uint64)
        for group, method in enumerate(canonical_names):
            selected = np.flatnonzero(groups == group)
            self.assert_equal(feature_orders[selected], expected.feature_shuffles_[method], f"group={method}, feature permutations")
            self.assert_equal(class_orders[selected], expected.class_shuffles_[method], f"group={method}, class permutations")
        rows, targets, _, _ = preprocessed.training_inputs()
        expected_training = expected.transform(X=None, mode="train")
        self.compare_rows(preprocessed, _array(*rows), expected_training, "training", _array(*targets))
        self.compare_rows(preprocessed, _array(*preprocessed.query_rows(_features(x)[0])), expected_training, "training transformed again")
        expected_queries = expected.transform(numerical.transform(upcast(q)), mode="test")
        self.compare_rows(preprocessed, _array(*preprocessed.query_rows(_features(q)[0])), expected_queries, "queries")
        type(self).fit_cases += 1
        return preprocessed, encoder, numerical, expected

    def test_synthetic_dtypes_classes_and_ensembles(self):
        for dtype in (np.float32, np.float64, np.int64):
            for seed, estimators, classes in ((0, 1, 2), (42, 8, 3), (123, 32, 10)):
                with self.subTest(dtype=dtype, seed=seed, estimators=estimators, classes=classes):
                    x, q, y = synthetic(dtype, classes, seed)
                    if classes == 3:
                        y = np.array([f"class-{label}" for label in y])
                    self.run_case("synthetic", x, q, y, n_estimators=estimators, random_state=seed)

    def test_regression_has_no_class_shuffles(self):
        for seed, estimators in ((0, 1), (42, 8), (123, 32)):
            with self.subTest(seed=seed, estimators=estimators):
                x, q, _ = synthetic(seed=seed)
                y = np.random.default_rng(seed).normal(size=len(x))
                options = {**DEFAULTS, "n_estimators": estimators, "random_state": seed}
                self.case = f"case=regression, options={options}"
                numerical = self.tabicl.TransformToNumerical()
                expected = self.tabicl.EnsembleGenerator(classification=False, **options).fit(numerical.fit_transform(x), y)
                preprocessed = preprocess(x, y, classification=False, **options)
                members = preprocessed.shape[0]
                self.assertEqual(_array(*preprocessed.class_permutations, np.uint64).shape, (members, 0))
                groups = np.asarray(preprocessed.member_groups)
                feature_orders = _array(*preprocessed.feature_permutations, np.uint64)
                for group, method in enumerate(preprocessed.group_names):
                    self.assert_equal(
                        feature_orders[groups == group], expected.feature_shuffles_[method], f"group={method}, feature permutations"
                    )
                rows = _array(*preprocessed.training_inputs()[0])
                self.compare_rows(preprocessed, rows, expected.transform(X=None, mode="train"), "training")
                expected_queries = expected.transform(numerical.transform(q), mode="test")
                self.compare_rows(preprocessed, _array(*preprocessed.query_rows(_features(q)[0])), expected_queries, "queries")

    def test_normalization_selection_and_order(self):
        for methods in (("none",), ("power",), ("none", "power"), ("power", "none")):
            for seed, estimators, threshold in ((0, 1, 4.0), (91, 5, 1.5)):
                with self.subTest(methods=methods, seed=seed, estimators=estimators, threshold=threshold):
                    x, q, y = synthetic(seed=seed)
                    self.run_case(
                        "normalization", x, q, y,
                        norm_methods=methods, random_state=seed, n_estimators=estimators, outlier_threshold=threshold,
                    )  # fmt: skip

    def test_fewer_actual_ensembles_and_tiny_training_set(self):
        x = np.arange(24, dtype=np.float64).reshape(-1, 1)
        y = np.where(np.arange(24) % 2, "positive", "negative")
        preprocessed, _, _, _ = self.run_case("one feature", x, x[:3], y, n_estimators=32)
        self.assertEqual(preprocessed.shape[0], 4)
        self.run_case("two training rows", np.array([[1.0], [2.0]]), np.array([[1.5]]), [10, 20])

    def test_near_constant_features(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                x, q, y = synthetic(dtype)
                x[:, 0] = 1 + x[:, 0] * 1e-5
                q[:, 0] = 1 + q[:, 0] * 1e-5
                self.run_case("near constant", x, q, y)

    def test_missing_values(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                x, q, y = synthetic(dtype, seed=5)
                rng = np.random.default_rng(5)
                x[rng.random(x.shape) < 0.15] = np.nan
                q[rng.random(q.shape) < 0.15] = np.nan
                x[:, 3] = np.nan  # Missing in every training row.
                q[:, 5] = np.nan  # Missing in every query row.
                self.run_case("missing", x, q, y, random_state=5)

    def test_skewed_heavy_tails(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                self.run_case("skewed", *skewed(dtype), random_state=42)

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
        preprocessed, _, numerical, expected = self.run_case("noncontiguous", x[::2], q[::2], y[::2])
        batch = _array(*preprocessed.query_rows(_features(q)[0]))
        for index, row in enumerate(q):
            alone = _array(*preprocessed.query_rows(_features(row[None, :])[0]))
            oracle = expected.transform(numerical.transform(upcast(row[None, :])), mode="test")
            self.compare_rows(preprocessed, alone, oracle, "one query row")
            self.assert_close(alone[:, 0], batch[:, index], "row vs batch")
        view, contiguous = (_array(*preprocessed.query_rows(_features(rows)[0])) for rows in (q[::2], q[::2].copy()))
        self.assert_equal(view, contiguous, "view vs contiguous")

    def test_hash_seed_probe(self):
        x, q, y = synthetic(seed=42)
        preprocessed, _, numerical, expected = self.run_case("hash seed", x, q, y, random_state=42)
        groups = np.asarray(preprocessed.member_groups)
        feature_orders = _array(*preprocessed.feature_permutations, np.uint64)
        class_orders = _array(*preprocessed.class_permutations, np.uint64)
        queries = _array(*preprocessed.query_rows(_features(q)[0]))
        ours, theirs = [], []
        for group, method in enumerate(("none", "power")):
            members = np.flatnonzero(groups == group)
            ours.append([method, feature_orders[members].tolist(), class_orders[members].tolist(), queries[members].tobytes().hex()])
            theirs.append([method, np.asarray(expected.feature_shuffles_[method]).tolist(), np.asarray(expected.class_shuffles_[method]).tolist()])

        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

        # The ensembles match TabICL's exactly; the values match within RTOL and
        # ATOL (in run_case) and must not depend on the hash seed.
        self.assertEqual([part[:3] for part in ours], theirs)
        print("Hash-seed signature: " + json.dumps({"ironicl": digest(ours), "tabicl": digest(theirs)}), flush=True)

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
