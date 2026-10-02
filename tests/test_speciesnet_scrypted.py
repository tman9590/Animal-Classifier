from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "distill"))

from generate_configs import MODEL_FILES, canonical_config
from speciesnet_scrypted import IMAGE_SIZE, parse_speciesnet_labels, select_regional_indices, should_geofence


class SpeciesNetScryptedTests(unittest.TestCase):
    def test_label_parser_falls_back_and_disambiguates(self):
        labels = parse_speciesnet_labels(
            [
                "1;mammalia;carnivora;felidae;felis;catus;domestic cat",
                "2;mammalia;carnivora;felidae;other;catus;domestic cat",
                "3;aves;passeriformes;family;genus;species;",
                "4;;;;;;vehicle",
                "55555555-a;;;family;bird;one;same bird",
                "66666666-b;;;family;bird;one;same bird",
            ]
        )
        self.assertEqual(labels[0], "domestic cat (felis catus)")
        self.assertEqual(labels[1], "domestic cat (other catus)")
        self.assertEqual(labels[2], "genus species")
        self.assertEqual(labels[3], "vehicle")
        self.assertEqual(labels[4], "same bird (bird one) [55555555]")
        self.assertEqual(labels[5], "same bird (bird one) [66666666]")

    def test_scrypted_config_is_classifier_contract(self):
        labels = {str(index): f"species {index}" for index in range(12)}
        config = canonical_config(labels)
        self.assertEqual(config["model"], "resnet")
        self.assertEqual(config["input_shape"], [1, 3, IMAGE_SIZE, IMAGE_SIZE])
        self.assertEqual(config["mean"], [0.0, 0.0, 0.0])
        self.assertEqual(config["std"], [1.0, 1.0, 1.0])
        self.assertEqual(config["labels"], labels)
        self.assertEqual(set(config["backends"]), set(MODEL_FILES))

    def test_bad_taxonomy_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 7"):
            parse_speciesnet_labels(["too;few;fields"])

    def test_geofence_allow_and_block_rules(self):
        self.assertFalse(should_geofence({"allow": {"USA": []}}, "USA", "NC"))
        self.assertFalse(should_geofence({"allow": {"USA": ["NC"]}}, "USA", "NC"))
        self.assertTrue(should_geofence({"allow": {"USA": ["VA"]}}, "USA", "NC"))
        self.assertTrue(should_geofence({"block": {"USA": []}}, "USA", "NC"))
        self.assertTrue(should_geofence({"block": {"USA": ["NC"]}}, "USA", "NC"))

    def test_regional_selection_keeps_species_ancestors_and_safety_labels(self):
        rows = [
            "1;mammalia;;;;;mammal",
            "2;mammalia;carnivora;;;;carnivore",
            "3;mammalia;carnivora;ursidae;ursus;americanus;American black bear",
            "4;mammalia;carnivora;felidae;panthera;leo;lion",
            "5;;;;;;blank",
            "6;reptilia;squamata;helodermatidae;heloderma;suspectum;Gila monster",
            "7;mammalia;artiodactyla;camelidae;lama;glama;llama",
        ]
        geofence = {
            "mammalia;carnivora;ursidae;ursus;americanus": {"allow": {"USA": ["NC"]}},
            "mammalia;carnivora;felidae;panthera;leo": {"allow": {"ZAF": []}},
        }
        self.assertEqual(select_regional_indices(rows, geofence, "USA", "NC"), [0, 1, 2, 4, 6])


if __name__ == "__main__":
    unittest.main()
