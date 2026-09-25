import unittest
from hallucination_labeling import (
    decide_grounding, parse_bool, resolve_label_indices, split_context_blocks,
)


class FakeConfig:
    id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}


class FakeModel:
    model = type("Inner", (), {"config": FakeConfig()})()


class HallucinationLabellingTests(unittest.TestCase):
    def test_boolean_parser_is_strict(self):
        self.assertTrue(parse_bool("true"))
        self.assertFalse(parse_bool("False"))
        with self.assertRaises(ValueError):
            parse_bool("unknown")

    def test_model_labels_are_discovered(self):
        self.assertEqual(
            resolve_label_indices(FakeModel()),
            {"contradiction": 0, "entailment": 1, "neutral": 2},
        )

    def test_context_blocks_keep_combined_evidence(self):
        blocks = split_context_blocks("[Context 1]\nA\n\n[Context 2]\nB")
        self.assertEqual(len(blocks), 3)
        self.assertIn("Context 2", blocks[0])

    def test_uncertain_scores_require_review(self):
        decision, label, _ = decide_grounding(
            {"entailment": 0.44, "neutral": 0.42, "contradiction": 0.14}
        )
        self.assertEqual(decision, "manual_review")
        self.assertIsNone(label)

    def test_clear_entailment_is_supported(self):
        decision, label, _ = decide_grounding(
            {"entailment": 0.80, "neutral": 0.15, "contradiction": 0.05}
        )
        self.assertEqual((decision, label), ("supported", 0))


if __name__ == "__main__":
    unittest.main()
