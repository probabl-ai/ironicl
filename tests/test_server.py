"""The server against pandas: a model fit on a DataFrame, exported, served,
and sent the DataFrame as pandas writes it. Runs the binary that
``IRONICL_BIN`` names, built with the ``server`` feature, and needs pandas and
the checkpoints in the default cache; it is skipped without any of them."""

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from ironicl import TabICLClassifier, TabICLRegressor
from test_pickle import checkpoint, data

try:
    import pandas as pd
except ImportError:
    pd = None


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@unittest.skipIf(pd is None, "needs pandas")
@unittest.skipUnless(os.environ.get("IRONICL_BIN"), "needs IRONICL_BIN, an ironicl built with the server feature")
class ServerTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)

    def serve(self, model) -> str:
        """Export `model`, serve it, and return its predict URL."""
        path = self.directory / type(model).__name__
        model.export(path)
        port = free_port()
        command = [os.environ["IRONICL_BIN"], "serve", str(path), "--bind", f"127.0.0.1:{port}"]
        server = subprocess.Popen(command, stdout=subprocess.DEVNULL)
        self.addCleanup(server.wait)
        self.addCleanup(server.terminate)
        deadline = time.monotonic() + 300
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health")
                return f"http://127.0.0.1:{port}/v1/predict"
            except OSError:
                if server.poll() is not None or time.monotonic() > deadline:
                    self.fail("the server did not start")
                time.sleep(0.1)

    def post(self, url: str, body: str) -> dict:
        request = urllib.request.Request(url, body.encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            return json.loads(error.read())

    def frame(self):
        x, y = data()
        return pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])]), y

    def test_a_classifier_takes_a_dataframe_in_any_column_order(self):
        checkpoint(TabICLClassifier)
        x, y = self.frame()
        model = TabICLClassifier(allow_auto_download=False, cache_dir=False)
        model.fit(x, np.where(y > 0, "high", "low"))
        url = self.serve(model)
        test = x.iloc[:20]
        answer = self.post(url, test.to_json(orient="split", index=False))
        self.assertEqual(answer["labels"], list(model.predict(test)))
        np.testing.assert_allclose(answer["probabilities"], model.predict_proba(test), atol=1e-4)
        reordered = test[test.columns[::-1]].to_json(orient="split", index=False)
        self.assertEqual(self.post(url, reordered), answer)
        missing = test.iloc[:2].copy()
        missing.iloc[1, 0] = np.nan
        error = self.post(url, missing.to_json(orient="split", index=False))["error"]["message"]
        self.assertIn("row 1 has a missing value", error)

    def test_a_regressor_takes_options_next_to_the_dataframe(self):
        checkpoint(TabICLRegressor)
        x, y = self.frame()
        model = TabICLRegressor(allow_auto_download=False, cache_dir=False).fit(x, y)
        url = self.serve(model)
        test = x.iloc[:20]
        options = {"output_type": ["mean", "quantiles"], "alphas": [0.1, 0.9]}
        answer = self.post(url, json.dumps({**test.to_dict(orient="split", index=False), **options}))
        np.testing.assert_allclose(answer["mean"], model.predict(test), atol=1e-3)
        quantiles = model.predict(test, output_type="quantiles", alphas=[0.1, 0.9])
        np.testing.assert_allclose(answer["quantiles"], quantiles, atol=1e-3)
        self.assertEqual(answer["alphas"], [0.1, 0.9])


if __name__ == "__main__":
    unittest.main()
