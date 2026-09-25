import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# Configuration
# ==========================================

INPUT_FILE = (
    "data/processed/"
    "rag_10_nli_groundedness.csv"
)


# ==========================================
# Load results
# ==========================================

print(
    "Loading NLI groundedness results..."
)

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Rows loaded:",
    len(df)
)


# ==========================================
# Overall agreement
# ==========================================

overall_accuracy = accuracy_score(
    df["hallucination_label"],
    df["automatic_hallucination_label"]
)


print("\n")
print("=" * 70)
print("OVERALL NLI EVALUATION")
print("=" * 70)


print(
    "\nOverall agreement:"
)

print(
    overall_accuracy
)


# ==========================================
# Evaluate only actual answers
# ==========================================

answered_df = df[
    df["abstained"] == False
].copy()


print("\n")
print("=" * 70)
print("NON-ABSTENTION NLI EVALUATION")
print("=" * 70)


print(
    "\nAnswered examples:"
)

print(
    len(answered_df)
)


y_true = (
    answered_df[
        "hallucination_label"
    ]
)


y_pred = (
    answered_df[
        "automatic_hallucination_label"
    ]
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
    "Columns = Automatic"
)


# ==========================================
# Detailed report
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
    y_true != y_pred
]


print("\n")
print("=" * 70)
print("DISAGREEMENTS")
print("=" * 70)


print(
    "\nNumber of disagreements:"
)

print(
    len(disagreements)
)


for _, row in disagreements.iterrows():

    print("\nQuestion:")

    print(
        row["question"]
    )


    print(
        "Generated:"
    )

    print(
        row["generated_answer"]
    )


    print(
        "Human label:"
    )

    print(
        row["hallucination_label"]
    )


    print(
        "NLI label:"
    )

    print(
        row[
            "automatic_hallucination_label"
        ]
    )


    print(
        "NLI entailment:"
    )

    print(
        row[
            "nli_entailment"
        ]
    )


print("\n")
print(
    "NLI PILOT EVALUATION COMPLETE"
)