import pandas as pd


# ==========================================
# Configuration
# ==========================================

INPUT_FILE = (
    "data/processed/"
    "rag_50_groundedness_validation_evaluated.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "rag_50_hallucination_annotation.csv"
)


# ==========================================
# Load evaluated validation data
# ==========================================

print(
    "Loading evaluated 50-question validation..."
)

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Rows loaded:",
    len(df)
)


# ==========================================
# Validate required columns
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
        + ", ".join(missing_columns)
    )


# ==========================================
# Normalize abstained
# ==========================================

df["abstained"] = (
    df["abstained"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(
        {
            "true": True,
            "false": False
        }
    )
)


# ==========================================
# Create human annotation columns
# ==========================================

df["context_support"] = ""

df["hallucination_label"] = pd.NA

df["annotation_reason"] = ""

df["annotator_confidence"] = ""


df["annotation_instruction"] = (
    "context_support: supported / "
    "partially_supported / unsupported / "
    "abstention; "
    "hallucination_label: "
    "0=no hallucination, 1=hallucination; "
    "judge only against retrieved context, "
    "not expected-answer mismatch or "
    "outside knowledge."
)


# ==========================================
# Prefill deterministic abstentions
# ==========================================

abstention_mask = (
    df["abstained"] == True
)


df.loc[
    abstention_mask,
    "context_support"
] = "abstention"


df.loc[
    abstention_mask,
    "hallucination_label"
] = 0


df.loc[
    abstention_mask,
    "annotation_reason"
] = (
    "Model returned INSUFFICIENT; "
    "under the frozen rubric, abstention "
    "does not make a factual claim and "
    "is therefore not hallucination."
)


df.loc[
    abstention_mask,
    "annotator_confidence"
] = "high"


# ==========================================
# Keep useful columns
# ==========================================

output_columns = [
    "query_id",
    "question",
    "expected_answer",
    "retrieved_context",
    "generated_answer",
    "abstained",
    "exact_match",
    "answer_f1",
    "correctness_status",
    "context_support",
    "hallucination_label",
    "annotation_reason",
    "annotator_confidence",
    "annotation_instruction"
]


annotation_df = df[
    output_columns
].copy()


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

total_rows = len(
    annotation_df
)


prefilled_abstentions = int(
    annotation_df[
        "abstained"
    ].sum()
)


manual_review_rows = (
    total_rows
    -
    prefilled_abstentions
)


print("\n")
print("=" * 70)

print(
    "50-QUESTION HALLUCINATION "
    "ANNOTATION FILE CREATED"
)

print("=" * 70)


print(
    "\nTotal rows:"
)

print(
    total_rows
)


print(
    "\nAbstentions prefilled:"
)

print(
    prefilled_abstentions
)


print(
    "\nRows requiring human review:"
)

print(
    manual_review_rows
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
    "Do not run NLI V2 on this validation "
    "set until the human annotations are "
    "completed."
)