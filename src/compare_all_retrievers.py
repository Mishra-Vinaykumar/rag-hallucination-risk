import pandas as pd


# ==================================
# Load experiment summaries
# ==================================

bm25_df = pd.read_csv(
    "data/processed/bm25_100_query_summary.csv"
)

dense_df = pd.read_csv(
    "data/processed/dense_100_query_summary.csv"
)

hybrid_df = pd.read_csv(
    "data/processed/hybrid_100_query_summary.csv"
)


print("==============================")
print("ALL RETRIEVAL METHODS")
print("==============================")


# ==================================
# Keep useful columns
# ==================================

bm25 = bm25_df[
    [
        "query_id",
        "question",
        "question_type",
        "difficulty",
        "precision_at_5",
        "recall_at_5",
        "hit_at_5",
        "first_relevant_rank"
    ]
].copy()

dense = dense_df[
    [
        "query_id",
        "precision_at_5",
        "recall_at_5",
        "hit_at_5",
        "first_relevant_rank"
    ]
].copy()

hybrid = hybrid_df[
    [
        "query_id",
        "precision_at_5",
        "recall_at_5",
        "hit_at_5",
        "first_relevant_rank"
    ]
].copy()


# ==================================
# Rename metric columns
# ==================================

bm25 = bm25.rename(
    columns={
        "precision_at_5": "precision_bm25",
        "recall_at_5": "recall_bm25",
        "hit_at_5": "hit_bm25",
        "first_relevant_rank": "first_rank_bm25"
    }
)

dense = dense.rename(
    columns={
        "precision_at_5": "precision_dense",
        "recall_at_5": "recall_dense",
        "hit_at_5": "hit_dense",
        "first_relevant_rank": "first_rank_dense"
    }
)

hybrid = hybrid.rename(
    columns={
        "precision_at_5": "precision_hybrid",
        "recall_at_5": "recall_hybrid",
        "hit_at_5": "hit_hybrid",
        "first_relevant_rank": "first_rank_hybrid"
    }
)


# ==================================
# Merge all three
# ==================================

comparison = bm25.merge(
    dense,
    on="query_id"
)

comparison = comparison.merge(
    hybrid,
    on="query_id"
)


print("\nMatched queries:")
print(len(comparison))


# ==================================
# Average metrics
# ==================================

print("\n==============================")
print("AVERAGE PERFORMANCE")
print("==============================")


methods = [
    "bm25",
    "dense",
    "hybrid"
]


for method in methods:

    print(f"\n--- {method.upper()} ---")

    print("Precision@5:")
    print(
        comparison[
            f"precision_{method}"
        ].mean()
    )

    print("Recall@5:")
    print(
        comparison[
            f"recall_{method}"
        ].mean()
    )

    print("Hit@5:")
    print(
        comparison[
            f"hit_{method}"
        ].mean()
    )

    print("Average First Relevant Rank:")
    print(
        comparison[
            f"first_rank_{method}"
        ].mean()
    )


# ==================================
# Recall winner per question
# ==================================

def recall_winner(row):

    scores = {
        "BM25": row["recall_bm25"],
        "Dense": row["recall_dense"],
        "Hybrid": row["recall_hybrid"]
    }

    maximum = max(scores.values())

    winners = [
        method
        for method, score in scores.items()
        if score == maximum
    ]

    if len(winners) == 1:
        return winners[0]

    return "Tie"


comparison["recall_winner"] = comparison.apply(
    recall_winner,
    axis=1
)


print("\n==============================")
print("RECALL WINNERS")
print("==============================")


print(
    comparison[
        "recall_winner"
    ].value_counts()
)


# ==================================
# Full Recall counts
# ==================================

print("\n==============================")
print("FULL RECALL COUNTS")
print("==============================")


for method in methods:

    count = (
        comparison[
            f"recall_{method}"
        ] == 1.0
    ).sum()

    print(
        f"{method.upper()}: {count}"
    )


# ==================================
# Complete failures
# ==================================

print("\n==============================")
print("COMPLETE FAILURE COUNTS")
print("==============================")


for method in methods:

    count = (
        comparison[
            f"hit_{method}"
        ] == False
    ).sum()

    print(
        f"{method.upper()}: {count}"
    )


# ==================================
# Questions where Hybrid beats BOTH
# ==================================

hybrid_better = comparison[
    (
        comparison["recall_hybrid"]
        >
        comparison["recall_bm25"]
    )
    &
    (
        comparison["recall_hybrid"]
        >
        comparison["recall_dense"]
    )
]


print("\n==============================")
print("HYBRID BEATS BOTH")
print("==============================")


print("\nCount:")
print(len(hybrid_better))


for _, row in hybrid_better.iterrows():

    print("\n------------------------------")

    print("Question:")
    print(row["question"])

    print("BM25 Recall:")
    print(row["recall_bm25"])

    print("Dense Recall:")
    print(row["recall_dense"])

    print("Hybrid Recall:")
    print(row["recall_hybrid"])


# ==================================
# Hybrid failures
# ==================================

hybrid_failures = comparison[
    comparison["hit_hybrid"] == False
]


print("\n==============================")
print("HYBRID COMPLETE FAILURES")
print("==============================")


print("\nCount:")
print(len(hybrid_failures))


for _, row in hybrid_failures.iterrows():

    print("\n------------------------------")

    print("Question:")
    print(row["question"])

    print("BM25 Recall:")
    print(row["recall_bm25"])

    print("Dense Recall:")
    print(row["recall_dense"])

    print("Hybrid Recall:")
    print(row["recall_hybrid"])


# ==================================
# Save comparison
# ==================================

comparison.to_csv(
    "data/processed/all_retrieval_comparison.csv",
    index=False
)


print("\n==============================")
print("COMPARISON SAVED")
print("==============================")


print(
    "\ndata/processed/"
    "all_retrieval_comparison.csv"
)