"""Fetching checkpoints into the cache, from file URLs: no network."""

import hashlib
import os
import stat
import tempfile
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ironicl.checkpoints import fetch

DATA = bytes(range(256)) * 4096


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.source = self.directory / "source.ckpt"
        self.source.write_bytes(DATA)
        self.url = self.source.as_uri()
        self.sha256 = hashlib.sha256(DATA).hexdigest()
        self.path = self.directory / "cache" / "tabicl" / "model.ckpt"

    def leftovers(self):
        return [path.name for path in self.path.parent.iterdir() if path != self.path]

    def test_downloads_once_without_a_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(fetch(self.path, self.url, self.sha256, allow_download=True), self.path)
        self.assertEqual(self.path.read_bytes(), DATA)
        self.source.unlink()
        self.assertEqual(fetch(self.path, self.url, self.sha256, allow_download=True), self.path)
        self.assertEqual(self.leftovers(), [])

    def test_warns_when_the_file_is_not_the_expected_one(self):
        with self.assertWarnsRegex(UserWarning, "not 0+ as Ironicl expects"):
            fetch(self.path, self.url, "0" * 64, allow_download=True)
        self.assertEqual(self.path.read_bytes(), DATA)

    def test_writes_given_bytes_without_downloading(self):
        fetch(self.path, "file:///missing", self.sha256, allow_download=False, data=DATA)
        self.assertEqual(self.path.read_bytes(), DATA)

    def test_respects_allow_auto_download(self):
        with self.assertRaises(FileNotFoundError):
            fetch(self.path, self.url, self.sha256, allow_download=False)
        self.assertFalse(self.path.exists())

    def test_a_failed_download_leaves_nothing(self):
        with self.assertRaises(OSError):
            fetch(self.path, (self.directory / "missing").as_uri(), None, allow_download=True)
        self.assertFalse(self.path.exists())
        self.assertEqual(self.leftovers(), [])

    def test_downloads_follow_the_umask(self):
        previous = os.umask(0o022)
        try:
            fetch(self.path, self.url, self.sha256, allow_download=True)
        finally:
            os.umask(previous)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o644)

    def test_removes_stale_temporary_files(self):
        self.path.parent.mkdir(parents=True)
        stale, fresh = (self.path.parent / f".{self.path.name}.{name}.partial" for name in ("old", "new"))
        stale.write_bytes(b"x")
        fresh.write_bytes(b"x")
        os.utime(stale, (0, 0))
        fetch(self.path, self.url, self.sha256, allow_download=True)
        self.assertFalse(stale.exists())
        self.assertTrue(fresh.exists())

    def test_concurrent_fetches_leave_one_whole_file(self):
        with ThreadPoolExecutor(8) as pool:
            paths = list(pool.map(lambda _: fetch(self.path, self.url, self.sha256, allow_download=True), range(8)))
        self.assertEqual(set(paths), {self.path})
        self.assertEqual(self.path.read_bytes(), DATA)
        self.assertEqual(self.leftovers(), [])


if __name__ == "__main__":
    unittest.main()
