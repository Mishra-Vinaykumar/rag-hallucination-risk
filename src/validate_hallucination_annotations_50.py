import pandas as pd


# ==========================================
# Configuration
# ==========================================

INPUT_FILE = (
    "data/processed/"
    "rag_50_hallucination_annotation.csv"
)


# ==========================================
# Allowed values
# ==========================================

ALLOWED_SUPPORT = {
    "supported",
    "partially_supported",
    "unsupported",
    "abstention"
}


EXPECTED_LABEL_MAPPING = {
    "supported": 0,
    "abstention": 0,
    "partially_supported": 1,
    "unsupported": 1
}


ALLOWED_CONFIDENCE = {
    "high",
    "medium",
    "low"
}


# ==========================================
# Load annotations
# ==========================================

print(
    "Loading hallucination annotations..."
)

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Rows loaded:",
    len(df)
)


# ==========================================
# Required columns
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
    "correctness_status",
    "context_support",
    "hallucination_label",
    "annotation_reason",
    "annotator_confidence"
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
# Check missing annotations
# ==========================================

annotation_columns = [
    "context_support",
    "hallucination_label",
    "annotation_reason",
    "annotator_confidence"
]


missing_annotations = (
    df[annotation_columns]
    .isna()
    .any(axis=1)
)


if missing_annotations.any():

    print(
        "\nERROR:"
    )

    print(
        "Some rows have missing annotations."
    )

    print(
        df.loc[
            missing_annotations,
            [
                "query_id",
                "question"
            ]
        ]
    )

    raise ValueError(
        "Incomplete manual annotations."
    )


# ==========================================
# Normalize annotation text
# ==========================================

df["context_support"] = (
    df["context_support"]
    .astype(str)
    .str.strip()
    .str.lower()
)


df["annotator_confidence"] = (
    df["annotator_confidence"]
    .astype(str)
    .str.strip()
    .str.lower()
)


df["hallucination_label"] = (
    pd.to_numeric(
        df["hallucination_label"],
        errors="raise"
    )
    .astype(int)
)


# ==========================================
# Validate support values
# ==========================================

invalid_support = ~df[
    "context_support"
].isin(
    ALLOWED_SUPPORT
)


if invalid_support.any():

    print(
        "\nInvalid context_support values:"
    )

    print(
        df.loc[
            invalid_support,
            [
                "query_id",
                "context_support"
            ]
        ]
    )

    raise ValueError(
        "Invalid context_support value."
    )


# ==========================================
# Validate hallucination labels
# ==========================================

invalid_labels = ~df[
    "hallucination_label"
].isin(
    [0, 1]
)


if invalid_labels.any():

    raise ValueError(
        "hallucination_label must be 0 or 1."
    )


# ==========================================
# Validate confidence
# ==========================================

invalid_confidence = ~df[
    "annotator_confidence"
].isin(
    ALLOWED_CONFIDENCE
)


if invalid_confidence.any():

    print(
        "\nInvalid confidence values:"
    )

    print(
        df.loc[
            invalid_confidence,
            [
                "query_id",
                "annotator_confidence"
            ]
        ]
    )

    raise ValueError(
        "Invalid annotator confidence."
    )


# ==========================================
# Validate support -> label mapping
# ==========================================

mapping_errors = []


for index, row in df.iterrows():

    expected_label = (
        EXPECTED_LABEL_MAPPING[
            row["context_support"]
        ]
    )

    actual_label = (
        row["hallucination_label"]
    )


    if actual_label != expected_label:

        mapping_errors.append(
            index
        )


if mapping_errors:

    print(
        "\nSupport/label mapping errors:"
    )

    print(
        df.loc[
            mapping_errors,
            [
                "query_id",
                "context_support",
                "hallucination_label"
            ]
        ]
    )

    raise ValueError(
        "Hallucination label does not "
        "match annotation rubric."
    )


# ==========================================
# Check abstention consistency
# ==========================================

abstention_errors = df[
    (df["abstained"] == True)
    &
    (
        df["context_support"]
        !=
        "abstention"
    )
]


if len(abstention_errors) > 0:

    print(
        "\nAbstention consistency errors:"
    )

    print(
        abstention_errors[
            [
                "query_id",
                "generated_answer",
                "context_support"
            ]
        ]
    )

    raise ValueError(
        "Abstained answers should use "
        "context_support='abstention'."
    )


# ==========================================
# Duplicate query check
# ==========================================

duplicate_ids = df[
    "query_id"
].duplicated()


if duplicate_ids.any():

    raise ValueError(
        "Duplicate query IDs found."
    )


# ==========================================
# Summary
# ==========================================

print("\n")
print("=" * 70)

print(
    "HALLUCINATION ANNOTATION VALIDATION"
)

print("=" * 70)


print(
    "\nTotal annotations:"
)

print(
    len(df)
)


print(
    "\nContext-support distribution:"
)

print(
    df[
        "context_support"
    ].value_counts()
)


print(
    "\nHallucination labels:"
)

print(
    df[
        "hallucination_label"
    ].value_counts()
)


print(
    "\nHallucination rate:"
)

print(
    df[
        "hallucination_label"
    ].mean()
)


print(
    "\nCorrectness vs hallucination:"
)

print(
    pd.crosstab(
        df[
            "correctness_status"
        ],
        df[
            "hallucination_label"
        ],
        margins=True
    )
)


print(
    "\nAnnotator confidence:"
)

print(
    df[
        "annotator_confidence"
    ].value_counts()
)


print("\n")

print(
    "ALL ANNOTATIONS PASSED VALIDATION!"
)