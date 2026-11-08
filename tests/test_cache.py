"""The context cache API: cache_info and clear_cache."""

import os
import shutil
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path

import numpy as np

from ironicl import TabICLClassifier, cache_info, clear_cache
from tests.test_pickle import checkpoint, data

DAY = 24 * 3600


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        now = time.time()
        # Oldest first: 3, 2, and 1 day since last use.
        for name, size, age in (("a", 10, 3), ("b", 20, 2), ("c", 30, 1)):
            path = self.directory / f"{name}.irctx"
            path.write_bytes(b"x" * size)
            os.utime(path, (now - age * DAY, now - age * DAY))
        self.stale = self.directory / ".ironicl-1-1.tmp"
        self.stale.write_bytes(b"x")
        os.utime(self.stale, (now - 7200, now - 7200))
        self.writing = self.directory / ".ironicl-1-2.tmp"
        self.writing.write_bytes(b"x")

    def names(self, info):
        return [entry.path.stem for entry in info.entries]

    def test_info_lists_contexts_most_recently_used_first(self):
        info = cache_info(self.directory)
        self.assertEqual(self.names(info), ["c", "b", "a"])
        self.assertEqual(info.size_bytes, 60)
        self.assertIn("3 contexts", str(info))
        self.assertEqual(cache_info(self.directory / "missing").entries, ())
        with self.assertRaises(ValueError):
            cache_info(False)

    def test_older_than_removes_contexts_last_used_before_it(self):
        removed = clear_cache(self.directory, older_than=timedelta(days=1.5))
        self.assertEqual(self.names(removed), ["b", "a"])
        self.assertEqual(self.names(cache_info(self.directory)), ["c"])

    def test_max_size_removes_the_least_recently_used(self):
        self.assertEqual(clear_cache(self.directory, max_size="1K").entries, ())
        removed = clear_cache(self.directory, max_size=35)
        self.assertEqual(self.names(removed), ["b", "a"])
        with self.assertRaises(ValueError):
            clear_cache(self.directory, max_size=-1)

    def test_clear_removes_everything_and_stale_temporary_files(self):
        self.assertEqual(clear_cache(self.directory).size_bytes, 60)
        self.assertEqual(cache_info(self.directory).entries, ())
        self.assertFalse(self.stale.exists())
        self.assertTrue(self.writing.exists())  # May still be written.

    def test_a_cache_hit_marks_the_entry_used(self):
        checkpoint(TabICLClassifier)
        x, y = data()
        model = TabICLClassifier(cache_dir=self.directory / "fit", allow_auto_download=False)
        model.fit(x, (y > 0).astype(int))
        (entry,) = cache_info(self.directory / "fit").entries
        os.utime(entry.path, (time.time() - DAY, time.time() - DAY))
        expected = model.predict_proba(x[:10])
        model.fit(x, (y > 0).astype(int))
        (hit,) = cache_info(self.directory / "fit").entries
        self.assertGreater(hit.last_used, entry.last_used)
        np.testing.assert_array_equal(model.predict_proba(x[:10]), expected)


if __name__ == "__main__":
    unittest.main()
