from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "distill"))

from generate_configs import (
    MODEL_FILES,
    UNIVERSAL_FILES,
    canonical_config,
    write_projections,
)


class GenerateConfigsTests(unittest.TestCase):
    def test_canonical_config_is_a_universal_scrypted_manifest(self):
        labels = {str(index): f"species {index}" for index in range(2498)}
        config = canonical_config(labels)
        self.assertEqual(config["files"], UNIVERSAL_FILES)
        binary_files = [name for name in config["files"] if name.endswith(".bin")]
        self.assertEqual(binary_files[0], "models/ncnn/speciesnet-v4.0.3a.ncnn.bin")

    def test_backend_projections_use_relative_artifact_paths(self):
        labels = {str(index): f"species {index}" for index in range(2498)}
        config = canonical_config(labels)
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary)
            write_projections(config, models, verify_files=False)
            for backend, expected in MODEL_FILES.items():
                projection = json.loads((models / backend / "config.json").read_text())
                self.assertEqual(projection["files"], expected)
                self.assertEqual(projection["labels"], labels)
                self.assertNotIn("backends", projection)


if __name__ == "__main__":
    unittest.main()
