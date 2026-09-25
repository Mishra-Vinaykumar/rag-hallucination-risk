"""Evaluate automated hallucination labels without counting abstentions as wins."""

import argparse
from pathlib import Path

import pandas as pd

from hallucination_labeling import parse_bool


def binary_metrics(truth, predicted):
    pairs = list(zip(truth, predicted))
    tn = sum(actual == 0 and guess == 0 for actual, guess in pairs)
    fp = sum(actual == 0 and guess == 1 for actual, guess in pairs)
    fn = sum(actual == 1 and guess == 0 for actual, guess in pairs)
    tp = sum(actual == 1 and guess == 1 for actual, guess in pairs)
    total = len(pairs)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    actual_positive = (tp + fn) / total
    predicted_positive = (tp + fp) / total
    expected_agreement = (
        actual_positive * predicted_positive
        + (1 - actual_positive) * (1 - predicted_positive)
    )
    kappa = (
        (accuracy - expected_agreement) / (1 - expected_agreement)
        if expected_agreement < 1 else 0.0
    )
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "kappa": kappa,
    }


def evaluate(input_file, prediction_column, review_column=None):
    df = pd.read_csv(input_file)
    required = {"abstained", "hallucination_label", prediction_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["abstained"] = df["abstained"].map(parse_bool)
    answered = df.loc[~df["abstained"]].copy()
    predictions = pd.to_numeric(answered[prediction_column], errors="coerce")
    decided = answered.loc[predictions.notna()].copy()
    decided[prediction_column] = predictions.loc[predictions.notna()].astype(int)
    truth = decided["hallucination_label"].astype(int)
    predicted = decided[prediction_column]

    if len(answered) == 0 or len(decided) == 0:
        raise ValueError("No answered and automatically decided rows to evaluate")
    if not predicted.isin({0, 1}).all():
        raise ValueError("Predictions must be 0, 1, or blank for manual review")

    coverage = len(decided) / len(answered)
    metrics = binary_metrics(truth.tolist(), predicted.tolist())
    print("AUTOMATED HALLUCINATION-LABELLING VALIDATION")
    print(f"Total responses: {len(df)}")
    print(f"Answered responses: {len(answered)}")
    print(f"Automatically decided: {len(decided)}")
    print(f"Automation coverage: {coverage:.3f}")
    print(f"Accuracy on decided answers: {metrics['accuracy']:.3f}")
    print(f"Hallucination precision: {metrics['precision']:.3f}")
    print(f"Hallucination recall: {metrics['recall']:.3f}")
    print(f"Hallucination F1: {metrics['f1']:.3f}")
    print(f"Cohen's kappa on decided answers: {metrics['kappa']:.3f}")
    if review_column and review_column in answered:
        review_count = answered[review_column].map(parse_bool).sum()
        print(f"Manual-review queue: {review_count}")
    print("\nConfusion matrix (rows=human, columns=automatic; labels 0,1):")
    print(f"[[{metrics['tn']}, {metrics['fp']}],")
    print(f" [{metrics['fn']}, {metrics['tp']}]]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--prediction-column", required=True)
    parser.add_argument("--review-column")
    args = parser.parse_args()
    evaluate(args.input_file, args.prediction_column, args.review_column)


if __name__ == "__main__":
    main()
