"""Focused tests for the pilot model training workflow."""

import unittest
from pathlib import Path

import numpy as np

from train_evaluate_pilot_models import choose_threshold, load_dataset


class PilotModelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.frame, cls.features = load_dataset(
            cls.root / "data/processed/pilot_ml_dataset.csv"
        )

    def test_fixed_splits_are_disjoint_and_complete(self):
        split_ids = {
            split: set(self.frame.loc[self.frame["pilot_split"] == split, "query_id"])
            for split in ("train", "validation", "test")
        }
        self.assertEqual(sum(map(len, split_ids.values())), 34)
        self.assertFalse(split_ids["train"] & split_ids["validation"])
        self.assertFalse(split_ids["train"] & split_ids["test"])
        self.assertFalse(split_ids["validation"] & split_ids["test"])
        self.assertNotIn("hallucination_label", self.features)
        self.assertNotIn("query_id", self.features)

    def test_threshold_selection_is_deterministic(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 1])
        probabilities = np.array([0.1, 0.3, 0.6, 0.4, 0.55, 0.8, 0.9])
        first = choose_threshold(labels, probabilities)
        second = choose_threshold(labels, probabilities)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0.1)
        self.assertLessEqual(first, 0.9)


if __name__ == "__main__":
    unittest.main()
