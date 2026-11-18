"""verbose progress, Ctrl-C during prediction, and show_config."""

import contextlib
import io
import signal
import unittest
from unittest import mock

import numpy as np

import ironicl.progress
from ironicl import TabICLClassifier, show_config
from tests.test_pickle import checkpoint, data


class ProgressTest(unittest.TestCase):
    def test_without_tqdm_progress_prints_a_line_per_tenth(self):
        stderr = io.StringIO()
        with mock.patch.object(ironicl.progress, "tqdm", None), contextlib.redirect_stderr(stderr):
            with ironicl.progress.Progress("Predicting", "row", 100) as progress:
                for done in range(0, 101, 5):
                    progress.update(done)
        lines = stderr.getvalue().splitlines()
        self.assertEqual(len(lines), 11)
        self.assertEqual(lines[-1], "ironicl: Predicting: 100/100 rows")

    def test_verbose_shows_the_build_and_the_prediction(self):
        checkpoint(TabICLClassifier)
        x, y = data()
        stderr = io.StringIO()
        with mock.patch.object(ironicl.progress, "tqdm", None), contextlib.redirect_stderr(stderr):
            model = TabICLClassifier(verbose=True, cache_dir=False).fit(x, (y > 0).astype(int))
            model.predict_proba(np.tile(x, (40, 1)))
        output = stderr.getvalue()
        self.assertIn("ironicl: Building context: 8/8 estimators", output)
        self.assertIn("ironicl: Predicting: 12000/12000 rows", output)

    def test_a_progress_exception_stops_the_prediction(self):
        checkpoint(TabICLClassifier)
        x, y = data()
        model = TabICLClassifier(cache_dir=False).fit(x, (y > 0).astype(int))
        rows = next(model._query_chunks(np.tile(x, (20, 1))))[0]

        def stop(*_):
            raise ValueError("stop")

        with self.assertRaisesRegex(ValueError, "stop"):
            model.context_.predict_proba(rows, 0.9, False, stop)

    def test_ctrl_c_stops_a_prediction_between_pieces(self):
        checkpoint(TabICLClassifier)
        x, y = data()
        model = TabICLClassifier(cache_dir=False).fit(x, (y > 0).astype(int))
        # One Python chunk, so only Rust's pieces can notice the interrupt.
        rows = next(model._query_chunks(np.tile(x, (27, 1))[:8000]))[0]
        pieces = []
        model.context_.predict_proba(rows, 0.9, False, lambda *report: pieces.append(report))
        if len(pieces) < 3:
            self.skipTest(f"this device predicts the rows in {len(pieces)} piece(s)")

        # Ctrl-C arrives as the first piece finishes. Python handles it in this callback or at
        # Rust's next check between pieces, and either way the prediction must stop before its
        # last piece.
        reports = []

        def progress(*report):
            reports.append(report)
            if len(reports) == 1:
                signal.raise_signal(signal.SIGINT)

        # A process started in the background inherits SIGINT ignored, and Python then keeps it
        # ignored: install the handler that raises KeyboardInterrupt for this test.
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            with self.assertRaises(KeyboardInterrupt):
                model.context_.predict_proba(rows, 0.9, False, progress)
        finally:
            signal.signal(signal.SIGINT, previous)
        self.assertLess(len(reports), len(pieces))


class ShowConfigTest(unittest.TestCase):
    def test_prints_the_backend_storage_and_cache(self):
        output = io.StringIO()
        show_config(file=output)
        text = output.getvalue()
        for expected in ("Ironicl ", "GPU: ", "CPU: ", "Memory: ", "  safe: FP32", "Context cache", "Environment:"):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
