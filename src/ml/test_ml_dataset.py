import unittest

from build_pilot_ml_dataset import stratified_splits


class PilotSplitTests(unittest.TestCase):
    def test_stratified_splits_are_deterministic_and_disjoint(self):
        rows = [
            {"query_id": f"q{label}-{index}", "hallucination_label": label}
            for label in (0, 1) for index in range(15)
        ]
        first = stratified_splits(rows)
        second = stratified_splits(list(reversed(rows)))
        self.assertEqual(first, second)
        for split in ("train", "validation", "test"):
            labels = {
                row["hallucination_label"] for row in rows
                if first[row["query_id"]] == split
            }
            self.assertEqual(labels, {0, 1})


if __name__ == "__main__":
    unittest.main()
