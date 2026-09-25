import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score
)


# ==========================================
# Configuration
# ==========================================

INPUT_FILE = (
    "data/processed/"
    "rag_50_nli_groundedness_validation.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "rag_50_nli_groundedness_disagreements.csv"
)


# ==========================================
# Load
# ==========================================

print(
    "Loading NLI validation results..."
)

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Rows loaded:",
    len(df)
)


# ==========================================
# Convert abstained safely
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
# Overall agreement
# ==========================================

overall_accuracy = accuracy_score(
    df["hallucination_label"],
    df["v2_hallucination_label"]
)


print("\n")
print("=" * 70)
print("OVERALL VALIDATION")
print("=" * 70)


print(
    "\nOverall agreement:"
)

print(
    overall_accuracy
)


# ==========================================
# Non-abstention only
# ==========================================

answered_df = df[
    df["abstained"] == False
].copy()


y_true = (
    answered_df[
        "hallucination_label"
    ]
)


y_pred = (
    answered_df[
        "v2_hallucination_label"
    ]
)


print("\n")
print("=" * 70)
print("NON-ABSTENTION VALIDATION")
print("=" * 70)


print(
    "\nExamples:"
)

print(
    len(answered_df)
)


# ==========================================
# Metrics
# ==========================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)


recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


kappa = cohen_kappa_score(
    y_true,
    y_pred
)


print(
    "\nAccuracy:"
)

print(
    accuracy
)


print(
    "\nHallucination Precision:"
)

print(
    precision
)


print(
    "\nHallucination Recall:"
)

print(
    recall
)


print(
    "\nHallucination F1:"
)

print(
    f1
)


print(
    "\nCohen's Kappa:"
)

print(
    kappa
)


# ==========================================
# Confusion matrix
# ==========================================

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)


print(
    "\nConfusion Matrix:"
)

print(
    matrix
)


print(
    "\nRows = Human"
)

print(
    "Columns = NLI V2"
)


# ==========================================
# Classification report
# ==========================================

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "No Hallucination",
            "Hallucination"
        ],
        zero_division=0
    )
)


# ==========================================
# Disagreements
# ==========================================

disagreements = answered_df[
    answered_df[
        "hallucination_label"
    ]
    !=
    answered_df[
        "v2_hallucination_label"
    ]
].copy()


print("\n")
print("=" * 70)
print("DISAGREEMENT ANALYSIS")
print("=" * 70)


print(
    "\nTotal disagreements:"
)

print(
    len(disagreements)
)


for number, (_, row) in enumerate(
    disagreements.iterrows(),
    start=1
):

    print(
        f"\n--- Disagreement {number} ---"
    )

    print(
        "Question:"
    )

    print(
        row["question"]
    )


    print(
        "\nGenerated:"
    )

    print(
        row["generated_answer"]
    )


    print(
        "\nHuman support:"
    )

    print(
        row["context_support"]
    )


    print(
        "Human hallucination:"
    )

    print(
        row["hallucination_label"]
    )


    print(
        "NLI label:"
    )

    print(
        row["v2_nli_label"]
    )


    print(
        "NLI hallucination:"
    )

    print(
        row["v2_hallucination_label"]
    )


    print(
        "Entailment probability:"
    )

    print(
        row["v2_nli_entailment"]
    )


# ==========================================
# Save disagreements
# ==========================================

disagreement_columns = [
    "query_id",
    "question",
    "retrieved_context",
    "generated_answer",
    "context_support",
    "hallucination_label",
    "annotation_reason",
    "v2_hypothesis",
    "v2_nli_contradiction",
    "v2_nli_entailment",
    "v2_nli_neutral",
    "v2_nli_label",
    "v2_hallucination_label"
]


disagreements[
    disagreement_columns
].to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nDisagreements saved to:"
)

print(
    OUTPUT_FILE
)


print(
    "\nNLI VALIDATION EVALUATION COMPLETE"
)