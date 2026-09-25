import numpy as np
import pandas as pd

from sentence_transformers import CrossEncoder


# ==========================================
# Configuration
# ==========================================

INPUT_FILE = (
    "data/processed/"
    "rag_10_hallucination_annotation.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "rag_10_nli_groundedness.csv"
)

NLI_MODEL = (
    "cross-encoder/"
    "nli-deberta-v3-base"
)


# ==========================================
# Helper
# ==========================================

def to_bool(value):

    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .lower()
        ==
        "true"
    )


# ==========================================
# Build NLI hypothesis
# ==========================================

def build_hypothesis(
    question,
    answer
):

    return (
        "For the question "
        f"'{question}', "
        "the answer is "
        f"'{answer}'."
    )


# ==========================================
# Softmax
# ==========================================

def softmax(logits):

    logits = np.asarray(
        logits,
        dtype=float
    )

    shifted = (
        logits
        -
        np.max(logits)
    )

    exponentials = np.exp(
        shifted
    )

    return (
        exponentials
        /
        exponentials.sum()
    )


# ==========================================
# Load manual annotations
# ==========================================

print(
    "Loading human annotations..."
)

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Rows loaded:",
    len(df)
)


# ==========================================
# Prepare columns
# ==========================================

df["abstained"] = (
    df["abstained"]
    .apply(to_bool)
)


df["hallucination_label"] = (
    pd.to_numeric(
        df["hallucination_label"],
        errors="raise"
    )
    .astype(int)
)


# ==========================================
# Load NLI model
# ==========================================

print(
    "\nLoading NLI model..."
)

nli_model = CrossEncoder(
    NLI_MODEL,
    max_length=512,
    device="cpu"
)

print(
    "NLI model loaded!"
)


# ==========================================
# NLI label mapping
# ==========================================
#
# Model output order:
#
# 0 = contradiction
# 1 = entailment
# 2 = neutral
#
# ==========================================

LABELS = [
    "contradiction",
    "entailment",
    "neutral"
]


# ==========================================
# Result storage
# ==========================================

contradiction_scores = []

entailment_scores = []

neutral_scores = []

nli_labels = []

automatic_support = []

automatic_hallucination = []


# ==========================================
# Evaluate each example
# ==========================================

for index, row in df.iterrows():

    print(
        f"\nProcessing {index + 1}/"
        f"{len(df)}"
    )


    # --------------------------------------
    # Abstention
    # --------------------------------------

    if row["abstained"]:

        contradiction_scores.append(
            np.nan
        )

        entailment_scores.append(
            np.nan
        )

        neutral_scores.append(
            np.nan
        )

        nli_labels.append(
            "abstention"
        )

        automatic_support.append(
            "abstention"
        )

        automatic_hallucination.append(
            0
        )


        print(
            "Generated answer: "
            "INSUFFICIENT"
        )

        print(
            "Automatic result: "
            "abstention"
        )

        continue


    # --------------------------------------
    # Premise
    # --------------------------------------

    premise = str(
        row[
            "retrieved_context"
        ]
    )


    # --------------------------------------
    # Hypothesis
    # --------------------------------------

    hypothesis = (
        build_hypothesis(
            question=row[
                "question"
            ],
            answer=row[
                "generated_answer"
            ]
        )
    )


    # --------------------------------------
    # NLI prediction
    # --------------------------------------

    logits = nli_model.predict(
        [
            (
                premise,
                hypothesis
            )
        ]
    )[0]


    probabilities = softmax(
        logits
    )


    contradiction_probability = float(
        probabilities[0]
    )

    entailment_probability = float(
        probabilities[1]
    )

    neutral_probability = float(
        probabilities[2]
    )


    predicted_index = int(
        np.argmax(
            probabilities
        )
    )


    predicted_nli_label = (
        LABELS[
            predicted_index
        ]
    )


    # --------------------------------------
    # Convert NLI to groundedness
    # --------------------------------------

    if (
        predicted_nli_label
        ==
        "entailment"
    ):

        predicted_support = (
            "supported"
        )

        predicted_hallucination = 0

    else:

        predicted_support = (
            "unsupported"
        )

        predicted_hallucination = 1


    # --------------------------------------
    # Store
    # --------------------------------------

    contradiction_scores.append(
        contradiction_probability
    )

    entailment_scores.append(
        entailment_probability
    )

    neutral_scores.append(
        neutral_probability
    )

    nli_labels.append(
        predicted_nli_label
    )

    automatic_support.append(
        predicted_support
    )

    automatic_hallucination.append(
        predicted_hallucination
    )


    # --------------------------------------
    # Print
    # --------------------------------------

    print(
        "Question:"
    )

    print(
        row[
            "question"
        ]
    )


    print(
        "Generated answer:"
    )

    print(
        row[
            "generated_answer"
        ]
    )


    print(
        "NLI label:"
    )

    print(
        predicted_nli_label
    )


    print(
        "Entailment probability:"
    )

    print(
        round(
            entailment_probability,
            4
        )
    )


    print(
        "Human hallucination:"
    )

    print(
        row[
            "hallucination_label"
        ]
    )


    print(
        "Automatic hallucination:"
    )

    print(
        predicted_hallucination
    )


# ==========================================
# Save NLI outputs
# ==========================================

df[
    "nli_contradiction"
] = contradiction_scores


df[
    "nli_entailment"
] = entailment_scores


df[
    "nli_neutral"
] = neutral_scores


df[
    "nli_label"
] = nli_labels


df[
    "automatic_context_support"
] = automatic_support


df[
    "automatic_hallucination_label"
] = automatic_hallucination


# ==========================================
# Compare against humans
# ==========================================

df[
    "automatic_correct"
] = (
    df[
        "automatic_hallucination_label"
    ]
    ==
    df[
        "hallucination_label"
    ]
)


agreement = (
    df[
        "automatic_correct"
    ]
    .mean()
)


# ==========================================
# Save
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# Summary
# ==========================================

print("\n")
print("=" * 70)

print(
    "NLI GROUNDEDNESS PILOT"
)

print("=" * 70)


print(
    "\nTotal examples:"
)

print(
    len(df)
)


print(
    "\nHuman labels:"
)

print(
    df[
        "hallucination_label"
    ].value_counts()
)


print(
    "\nAutomatic labels:"
)

print(
    df[
        "automatic_hallucination_label"
    ].value_counts()
)


print(
    "\nHuman vs automatic:"
)

print(
    pd.crosstab(
        df[
            "hallucination_label"
        ],
        df[
            "automatic_hallucination_label"
        ],
        rownames=[
            "Human"
        ],
        colnames=[
            "Automatic"
        ],
        margins=True
    )
)


print(
    "\nAgreement:"
)

print(
    agreement
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)