"""Validate the manual annotation file and report non-inflated statistics."""

from pathlib import Path
import pandas as pd
from hallucination_labeling import parse_bool

INPUT_FILE = Path("data/processed/rag_50_hallucination_annotation.csv")
ALLOWED_SUPPORT = {"supported", "partially_supported", "unsupported", "abstention"}
EXPECTED_LABEL = {
    "supported": 0, "partially_supported": 1,
    "unsupported": 1, "abstention": 0,
}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}


def main():
    df = pd.read_csv(INPUT_FILE)
    required = {
        "query_id", "question", "retrieved_context", "generated_answer",
        "abstained", "context_support", "hallucination_label",
        "annotation_reason", "annotator_confidence",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["query_id"].duplicated().any():
        raise ValueError("Duplicate query IDs found")

    text_columns = ["context_support", "annotation_reason", "annotator_confidence"]
    blank = df[text_columns].isna() | df[text_columns].astype(str).apply(
        lambda column: column.str.strip().eq("")
    )
    if blank.any(axis=1).any():
        bad_ids = df.loc[blank.any(axis=1), "query_id"].tolist()
        raise ValueError(f"Incomplete annotations for query IDs: {bad_ids}")

    df["abstained"] = df["abstained"].map(parse_bool)
    df["context_support"] = df["context_support"].str.strip().str.lower()
    df["annotator_confidence"] = df["annotator_confidence"].str.strip().str.lower()
    df["hallucination_label"] = pd.to_numeric(
        df["hallucination_label"], errors="raise"
    ).astype(int)

    if not df["context_support"].isin(ALLOWED_SUPPORT).all():
        raise ValueError("Invalid context_support value")
    if not df["annotator_confidence"].isin(ALLOWED_CONFIDENCE).all():
        raise ValueError("Invalid annotator_confidence value")
    if not df["hallucination_label"].isin({0, 1}).all():
        raise ValueError("hallucination_label must be 0 or 1")
    expected = df["context_support"].map(EXPECTED_LABEL)
    if not expected.equals(df["hallucination_label"]):
        bad_ids = df.loc[expected != df["hallucination_label"], "query_id"].tolist()
        raise ValueError(f"Support/label mapping errors for query IDs: {bad_ids}")
    abstention_support = df["context_support"].eq("abstention")
    if not abstention_support.equals(df["abstained"]):
        bad_ids = df.loc[abstention_support != df["abstained"], "query_id"].tolist()
        raise ValueError(f"Abstention consistency errors for query IDs: {bad_ids}")

    answered = df.loc[~df["abstained"]]
    print("HALLUCINATION ANNOTATION VALIDATION")
    print(f"Total rows: {len(df)}")
    print(f"Answered responses: {len(answered)}")
    print(f"Abstentions: {df['abstained'].sum()}")
    print(f"Answer coverage: {len(answered) / len(df):.3f}")
    print(
        "Hallucination rate among answered responses: "
        f"{answered['hallucination_label'].mean():.3f}"
    )
    print("\nSupport distribution:")
    print(df["context_support"].value_counts())
    print("\nConfidence distribution:")
    print(df["annotator_confidence"].value_counts())
    print("\nAll annotation integrity checks passed.")


if __name__ == "__main__":
    main()
