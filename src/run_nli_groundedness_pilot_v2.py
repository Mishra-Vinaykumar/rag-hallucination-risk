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
    "rag_10_nli_groundedness_v2.csv"
)

NLI_MODEL = (
    "cross-encoder/"
    "nli-deberta-v3-base"
)


# ==========================================
# Helpers
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


def softmax(logits):

    logits = np.asarray(
        logits,
        dtype=float
    )

    logits = logits - np.max(logits)

    exp_values = np.exp(
        logits
    )

    return (
        exp_values
        /
        exp_values.sum()
    )


# ==========================================
# Detect short answer
# ==========================================

def is_short_answer(answer):

    text = str(
        answer
    ).strip()

    words = text.split()

    # Short entity / phrase answers
    if len(words) <= 4:

        return True

    return False


# ==========================================
# Build grounding hypothesis
# ==========================================

def build_grounding_hypothesis(
    question,
    answer
):

    answer = str(
        answer
    ).strip()


    # --------------------------------------
    # Short QA-style response
    # --------------------------------------

    if is_short_answer(
        answer
    ):

        return (
            f"For the question "
            f"'{question}', "
            f"the answer is "
            f"'{answer}'."
        )


    # --------------------------------------
    # Full generated statement
    #
    # Judge the factual statement itself,
    # not whether it answers the question.
    # --------------------------------------

    return answer


# ==========================================
# Load human annotations
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


LABELS = [
    "contradiction",
    "entailment",
    "neutral"
]


# ==========================================
# Output storage
# ==========================================

contradiction_scores = []

entailment_scores = []

neutral_scores = []

nli_labels = []

hypothesis_types = []

hypotheses = []

automatic_labels = []


# ==========================================
# Evaluate
# ==========================================

for index, row in df.iterrows():

    print(
        f"\nProcessing {index + 1}/"
        f"{len(df)}"
    )


    # ======================================
    # Abstention
    # ======================================

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

        hypothesis_types.append(
            "abstention"
        )

        hypotheses.append(
            ""
        )

        automatic_labels.append(
            0
        )

        print(
            "Generated answer: INSUFFICIENT"
        )

        print(
            "Automatic result: abstention"
        )

        continue


    # ======================================
    # Premise
    # ======================================

    premise = str(
        row["retrieved_context"]
    )


    # ======================================
    # Hypothesis
    # ======================================

    generated_answer = str(
        row["generated_answer"]
    )


    hypothesis = (
        build_grounding_hypothesis(
            question=row["question"],
            answer=generated_answer
        )
    )


    if is_short_answer(
        generated_answer
    ):

        hypothesis_type = (
            "question_answer_claim"
        )

    else:

        hypothesis_type = (
            "generated_claim"
        )


    # ======================================
    # NLI
    # ======================================

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


    contradiction = float(
        probabilities[0]
    )

    entailment = float(
        probabilities[1]
    )

    neutral = float(
        probabilities[2]
    )


    predicted_index = int(
        np.argmax(
            probabilities
        )
    )


    predicted_label = (
        LABELS[
            predicted_index
        ]
    )


    # ======================================
    # Binary hallucination
    # ======================================

    if predicted_label == "entailment":

        hallucination = 0

    else:

        hallucination = 1


    # ======================================
    # Store
    # ======================================

    contradiction_scores.append(
        contradiction
    )

    entailment_scores.append(
        entailment
    )

    neutral_scores.append(
        neutral
    )

    nli_labels.append(
        predicted_label
    )

    hypothesis_types.append(
        hypothesis_type
    )

    hypotheses.append(
        hypothesis
    )

    automatic_labels.append(
        hallucination
    )


    # ======================================
    # Print
    # ======================================

    print(
        "Generated:"
    )

    print(
        generated_answer
    )


    print(
        "Hypothesis type:"
    )

    print(
        hypothesis_type
    )


    print(
        "Hypothesis:"
    )

    print(
        hypothesis
    )


    print(
        "NLI label:"
    )

    print(
        predicted_label
    )


    print(
        "Entailment probability:"
    )

    print(
        round(
            entailment,
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
        hallucination
    )


# ==========================================
# Save
# ==========================================

df[
    "v2_hypothesis_type"
] = hypothesis_types


df[
    "v2_hypothesis"
] = hypotheses


df[
    "v2_nli_contradiction"
] = contradiction_scores


df[
    "v2_nli_entailment"
] = entailment_scores


df[
    "v2_nli_neutral"
] = neutral_scores


df[
    "v2_nli_label"
] = nli_labels


df[
    "v2_hallucination_label"
] = automatic_labels


df[
    "v2_correct"
] = (
    df[
        "v2_hallucination_label"
    ]
    ==
    df[
        "hallucination_label"
    ]
)


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
    "NLI GROUNDEDNESS PILOT V2"
)

print("=" * 70)


print(
    "\nOverall agreement:"
)

print(
    df[
        "v2_correct"
    ].mean()
)


answered_df = df[
    df[
        "abstained"
    ]
    ==
    False
]


print(
    "\nNon-abstention examples:"
)

print(
    len(
        answered_df
    )
)


print(
    "\nNon-abstention agreement:"
)

print(
    answered_df[
        "v2_correct"
    ].mean()
)


print(
    "\nHuman vs V2 automatic:"
)

print(
    pd.crosstab(
        answered_df[
            "hallucination_label"
        ],
        answered_df[
            "v2_hallucination_label"
        ],
        rownames=[
            "Human"
        ],
        colnames=[
            "V2"
        ],
        margins=True
    )
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)