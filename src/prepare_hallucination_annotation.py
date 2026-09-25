import os
import pandas as pd


# ==========================================
# Configuration
# ==========================================

INPUT_FILE = (
    "data/processed/"
    "rag_10_query_pilot_evaluated.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "rag_10_hallucination_annotation.csv"
)


# ==========================================
# Load evaluated pilot
# ==========================================

print(
    "Loading evaluated RAG pilot..."
)

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Rows loaded:",
    len(df)
)


# ==========================================
# Check required columns
# ==========================================

required_columns = [
    "query_id",
    "question",
    "expected_answer",
    "retrieved_context",
    "generated_answer",
    "abstained",
    "exact_match",
    "answer_f1",
    "correctness_status"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(
            missing_columns
        )
    )


# ==========================================
# Create annotation table
# ==========================================

annotation_df = df[
    [
        "query_id",
        "question",
        "expected_answer",
        "retrieved_context",
        "generated_answer",
        "abstained",
        "exact_match",
        "answer_f1",
        "correctness_status"
    ]
].copy()


# ==========================================
# Add human annotation columns
# ==========================================

annotation_df[
    "context_support"
] = ""


annotation_df[
    "hallucination_label"
] = ""


annotation_df[
    "annotation_reason"
] = ""


annotation_df[
    "annotator_confidence"
] = ""


# ==========================================
# Add annotation instructions
# ==========================================

annotation_df[
    "annotation_instruction"
] = (
    "context_support: supported / partially_supported / unsupported / abstention; "
    "hallucination_label: 0=no hallucination, 1=hallucination; "
    "judge only against retrieved context, not outside knowledge."
)


# ==========================================
# Save
# ==========================================

annotation_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# Summary
# ==========================================

print("\n")
print("=" * 70)
print("HALLUCINATION ANNOTATION FILE CREATED")
print("=" * 70)


print(
    "\nRows:"
)

print(
    len(annotation_df)
)


print(
    "\nColumns:"
)

for column in annotation_df.columns:

    print(
        "-",
        column
    )


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)


print(
    "\nIMPORTANT:"
)

print(
    "Do not classify hallucination using "
    "expected-answer mismatch alone."
)

print(
    "Judge whether the generated claim is "
    "supported by the retrieved context."
)