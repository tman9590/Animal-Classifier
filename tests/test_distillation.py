from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "distill"))

from catalog import Species, read_catalog
from label_images import yolo_box
from smoothing import TemporalSmoother, iou


class DistillationTests(unittest.TestCase):
    def test_species_labels_are_granular(self):
        item = Species(906, "Meleagris gallopavo", "Turkey", "Aves", 100)
        self.assertEqual(item.label, "Turkey")

    def test_north_carolina_catalog_is_unique_and_includes_poultry(self):
        metadata, species = read_catalog(ROOT / "species" / "north-carolina.json")
        scientific_names = {item.scientific_name for item in species}
        self.assertEqual(len(species), len(scientific_names))
        self.assertEqual(metadata["region"], "North Carolina, USA")
        self.assertEqual(metadata["labels"][-1], "unknown")
        self.assertGreaterEqual(len(species), 500)
        self.assertEqual(metadata["labels"][:-1], [item.common_name for item in species])
        self.assertNotIn("(", "".join(metadata["labels"]))
        self.assertTrue(
            {
                "Gallus gallus domesticus",
                "Meleagris gallopavo",
                "Anas platyrhynchos domesticus",
                "Cairina moschata",
                "Numida meleagris",
                "Coturnix japonica",
                "Pavo cristatus",
            }.issubset(scientific_names)
        )

    def test_scrypted_backend_configs_match_catalog(self):
        metadata, _ = read_catalog(ROOT / "species" / "north-carolina.json")
        expected_files = {
            "coreml": [
                "north-carolina-wildlife.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                "north-carolina-wildlife.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
                "north-carolina-wildlife.mlpackage/Manifest.json",
            ],
            "openvino": ["north-carolina-wildlife.xml", "north-carolina-wildlife.bin"],
        }
        for backend, files in expected_files.items():
            config = json.loads((ROOT / "models" / backend / "config.json").read_text())
            self.assertEqual(config["input_shape"], [1, 3, 640, 640])
            self.assertEqual(config["model"], "yolov9")
            self.assertEqual(config["files"], files)
            self.assertEqual(list(config["labels"].values()), metadata["labels"])

    def test_temporal_smoothing_and_unknown_threshold(self):
        smoother = TemporalSmoother(alpha=0.5, match_iou=0.3, ttl_frames=10, unknown_threshold=0.7)
        first = smoother.update(1, [((0, 0, 100, 100), {"fox": 0.6, "coyote": 0.4})])
        second = smoother.update(2, [((2, 1, 102, 101), {"fox": 1.0, "coyote": 0.0})])
        self.assertEqual(first[0][1], "unknown")
        self.assertEqual(second[0][0], first[0][0])
        self.assertEqual(second[0][1], "fox")
        self.assertAlmostEqual(second[0][2], 0.8)

    def test_iou_and_yolo_conversion(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
        self.assertEqual(yolo_box((10, 20, 30, 60), 100, 100), (0.2, 0.4, 0.2, 0.4))


if __name__ == "__main__":
    unittest.main()
