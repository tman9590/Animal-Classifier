from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / 'distill'))
from catalog import Species
from download_inaturalist import download_species, write_manifest
from teacher import MegaDetector
from label_images import observation_split, prune_exact_duplicates
from label_scrypted_frames import frame_order
from validate_exports import validate_output


class ResumeTests(unittest.TestCase):
    def test_existing_photo_recovers_attribution_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / '000-1'
            folder.mkdir()
            (folder / '42-7.jpg').write_bytes(b'existing image')
            payload = {'results': [{'id': 42, 'photos': [
                {'id': 7, 'url': 'https://example.org/square.jpg',
                 'license_code': 'cc-by', 'attribution': 'Photographer'}]}]}
            with patch('download_inaturalist.api_get', return_value=payload), patch('download_inaturalist.download') as download:
                rows = download_species({'inaturalist_place_id': 30}, Species(1, 'Test species', 'Test Animal', 'Aves', 20), 0, root, 20)
                download.assert_not_called()
            write_manifest(root, rows)
            with (root / 'attribution.csv').open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(saved[0]['attribution'], 'Photographer')
            self.assertEqual(saved[0]['license'], 'cc-by')
            self.assertFalse(list(root.glob('*.tmp')))

    def test_observation_photos_cannot_cross_splits(self):
        for observation in range(100):
            self.assertEqual(observation_split(f'{observation}-11'), observation_split(f'{observation}-22'))
        self.assertEqual({observation_split(f'{i}-11') for i in range(100)}, {'train', 'val'})

    def test_exact_duplicate_filter_prefers_validation_and_removes_matching_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'val'):
                (root / 'images' / split).mkdir(parents=True)
                (root / 'labels' / split).mkdir(parents=True)
            (root / 'images' / 'train' / 'public-a.jpg').write_bytes(b'same')
            (root / 'labels' / 'train' / 'public-a.txt').write_text('1 0.5 0.5 1 1\n')
            (root / 'images' / 'val' / 'public-b.jpg').write_bytes(b'same')
            (root / 'labels' / 'val' / 'public-b.txt').write_text('1 0.5 0.5 1 1\n')
            removed = prune_exact_duplicates(root, 'public-*.jpg')
            self.assertEqual(len(removed), 1)
            self.assertFalse((root / 'images' / 'train' / 'public-a.jpg').exists())
            self.assertFalse((root / 'labels' / 'train' / 'public-a.txt').exists())
            self.assertTrue((root / 'images' / 'val' / 'public-b.jpg').exists())

    def test_track_frame_numbers_sort_numerically(self):
        paths = [Path(name) for name in ['abcd-100.jpg', 'abcd-2.jpg', 'abcd-11.jpg']]
        self.assertEqual([p.name for p in sorted(paths, key=frame_order)], ['abcd-2.jpg', 'abcd-11.jpg', 'abcd-100.jpg'])

    def test_export_contract_rejects_nms_output_and_nonfinite_scores(self):
        valid = np.zeros((1, 534, 8400), dtype=np.float32)
        validate_output(valid, 530, 640)
        with self.assertRaises(ValueError):
            validate_output(np.zeros((1, 300, 6)), 530, 640)
        valid[0, 4, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_output(valid, 530, 640)

    def test_detector_converts_pil_rgb_to_ultralytics_bgr(self):
        detector = MegaDetector.__new__(MegaDetector)
        detector.device = 'cpu'
        detector.model = Mock()
        detector.model.predict.return_value = []
        detector.detect_many([Image.new('RGB', (2, 2), (255, 10, 30))], .25)
        arrays = detector.model.predict.call_args.args[0]
        np.testing.assert_array_equal(arrays[0][0, 0], [30, 10, 255])


if __name__ == '__main__':
    unittest.main()
