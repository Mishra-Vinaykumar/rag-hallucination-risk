import re
import string
from collections import Counter

import pandas as pd


# ==========================================
# Configuration
# ==========================================

INPUT_FILE = (
    "data/processed/"
    "rag_10_query_pilot.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "rag_10_query_pilot_evaluated.csv"
)


# ==========================================
# Normalize answer
# ==========================================

def normalize_answer(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()


    # --------------------------------------
    # Remove punctuation
    # --------------------------------------

    text = "".join(
        char
        for char in text
        if char not in string.punctuation
    )


    # --------------------------------------
    # Remove English articles
    # --------------------------------------

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text
    )


    # --------------------------------------
    # Normalize whitespace
    # --------------------------------------

    text = " ".join(
        text.split()
    )


    return text


# ==========================================
# Exact Match
# ==========================================

def exact_match_score(
    prediction,
    reference
):

    normalized_prediction = (
        normalize_answer(prediction)
    )

    normalized_reference = (
        normalize_answer(reference)
    )


    return int(
        normalized_prediction
        ==
        normalized_reference
    )


# ==========================================
# Token-level F1
# ==========================================

def token_f1_score(
    prediction,
    reference
):

    prediction_tokens = (
        normalize_answer(
            prediction
        ).split()
    )

    reference_tokens = (
        normalize_answer(
            reference
        ).split()
    )


    # --------------------------------------
    # Handle empty answers
    # --------------------------------------

    if (
        len(prediction_tokens) == 0
        and
        len(reference_tokens) == 0
    ):
        return 1.0


    if (
        len(prediction_tokens) == 0
        or
        len(reference_tokens) == 0
    ):
        return 0.0


    # --------------------------------------
    # Token overlap
    # --------------------------------------

    prediction_counter = Counter(
        prediction_tokens
    )

    reference_counter = Counter(
        reference_tokens
    )


    common = (
        prediction_counter
        &
        reference_counter
    )


    overlap = sum(
        common.values()
    )


    if overlap == 0:
        return 0.0


    precision = (
        overlap
        /
        len(prediction_tokens)
    )


    recall = (
        overlap
        /
        len(reference_tokens)
    )


    f1 = (
        2
        * precision
        * recall
        /
        (precision + recall)
    )


    return f1


# ==========================================
# Load pilot results
# ==========================================

print(
    "Loading RAG pilot results..."
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
    "generated_answer",
    "abstained"
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
# Convert abstained column safely
# ==========================================

df["abstained"] = (
    df["abstained"]
    .astype(str)
    .str.lower()
    .map(
        {
            "true": True,
            "false": False
        }
    )
)


# ==========================================
# Evaluate each answer
# ==========================================

exact_matches = []

f1_scores = []

correctness_statuses = []


for _, row in df.iterrows():

    expected = row[
        "expected_answer"
    ]

    generated = row[
        "generated_answer"
    ]

    abstained = row[
        "abstained"
    ]


    # --------------------------------------
    # Exact Match
    # --------------------------------------

    exact_match = (
        exact_match_score(
            generated,
            expected
        )
    )


    # --------------------------------------
    # Token F1
    # --------------------------------------

    f1 = token_f1_score(
        generated,
        expected
    )


    # --------------------------------------
    # Correctness category
    # --------------------------------------

    if abstained is True:

        correctness_status = (
            "abstention"
        )

    elif exact_match == 1:

        correctness_status = (
            "correct"
        )

    else:

        correctness_status = (
            "incorrect"
        )


    exact_matches.append(
        exact_match
    )

    f1_scores.append(
        f1
    )

    correctness_statuses.append(
        correctness_status
    )


# ==========================================
# Add evaluation columns
# ==========================================

df["exact_match"] = (
    exact_matches
)

df["answer_f1"] = (
    f1_scores
)

df["correctness_status"] = (
    correctness_statuses
)


# ==========================================
# Save evaluated results
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# Print individual results
# ==========================================

print("\n")
print(
    "=" * 70
)

print(
    "INDIVIDUAL ANSWER EVALUATION"
)

print(
    "=" * 70
)


for number, row in df.iterrows():

    print(
        f"\nQuestion {number + 1}"
    )

    print(
        "Question:"
    )

    print(
        row["question"]
    )


    print(
        "\nExpected:"
    )

    print(
        row["expected_answer"]
    )


    print(
        "\nGenerated:"
    )

    print(
        row["generated_answer"]
    )


    print(
        "\nAbstained:"
    )

    print(
        row["abstained"]
    )


    print(
        "Exact Match:"
    )

    print(
        row["exact_match"]
    )


    print(
        "Answer F1:"
    )

    print(
        round(
            row["answer_f1"],
            4
        )
    )


    print(
        "Correctness Status:"
    )

    print(
        row[
            "correctness_status"
        ]
    )


# ==========================================
# Summary
# ==========================================

print("\n")
print(
    "=" * 70
)

print(
    "RAG CORRECTNESS EVALUATION"
)

print(
    "=" * 70
)


print(
    "\nTotal answers:"
)

print(
    len(df)
)


print(
    "\nExact matches:"
)

print(
    int(
        df[
            "exact_match"
        ].sum()
    )
)


print(
    "\nExact Match Accuracy:"
)

print(
    df[
        "exact_match"
    ].mean()
)


print(
    "\nAverage Answer F1:"
)

print(
    df[
        "answer_f1"
    ].mean()
)


print(
    "\nCorrectness categories:"
)

print(
    df[
        "correctness_status"
    ].value_counts()
)


# ==========================================
# Non-abstention accuracy
# ==========================================

answered_df = df[
    df[
        "correctness_status"
    ]
    !=
    "abstention"
]


print(
    "\nNon-abstained answers:"
)

print(
    len(
        answered_df
    )
)


if len(answered_df) > 0:

    print(
        "\nExact Match Accuracy "
        "on non-abstained answers:"
    )

    print(
        answered_df[
            "exact_match"
        ].mean()
    )


print(
    "\nEvaluated results saved to:"
)

print(
    OUTPUT_FILE
)