import os
import tempfile
import unittest
from unittest.mock import patch

import joblib

from ml.config import SCHEMA_VERSION
from ml.data import file_fingerprint
from ml.model import artifact_status


class CacheTests(unittest.TestCase):
    def test_artifact_becomes_stale_when_csv_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "prices.csv")
            model_path = os.path.join(tmp, "model.joblib")
            with open(csv_path, "w", encoding="utf-8") as fh:
                fh.write("Date,Close\n2024-01-01,100\n")
            joblib.dump({
                "schema_version": SCHEMA_VERSION,
                "fingerprint": file_fingerprint(csv_path),
                "trained_at": "2024-01-01T00:00:00",
            }, model_path)

            with patch("ml.model.model_path", return_value=model_path), \
                 patch("ml.model.market_path", return_value=csv_path):
                self.assertEqual(artifact_status("test")["state"], "current")
                with open(csv_path, "a", encoding="utf-8") as fh:
                    fh.write("2024-01-02,101\n")
                self.assertEqual(artifact_status("test")["state"], "stale-data")


if __name__ == "__main__":
    unittest.main()
