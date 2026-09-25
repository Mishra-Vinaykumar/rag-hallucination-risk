import pandas as pd


# ==================================
# Load both experiment summaries
# ==================================

bm25_df = pd.read_csv(
    "data/processed/bm25_100_query_summary.csv"
)

dense_df = pd.read_csv(
    "data/processed/dense_100_query_summary.csv"
)


print("==============================")
print("BM25 VS DENSE COMPARISON")
print("==============================")


# ==================================
# Merge using Query ID
# ==================================

comparison_df = bm25_df.merge(
    dense_df,
    on="query_id",
    suffixes=("_bm25", "_dense")
)


print("\nTotal matched queries:")
print(len(comparison_df))


# ==================================
# Overall metrics
# ==================================

print("\n==============================")
print("AVERAGE METRICS")
print("==============================")


print("\nBM25 Precision@5:")
print(
    comparison_df[
        "precision_at_5_bm25"
    ].mean()
)

print("\nDense Precision@5:")
print(
    comparison_df[
        "precision_at_5_dense"
    ].mean()
)


print("\nBM25 Recall@5:")
print(
    comparison_df[
        "recall_at_5_bm25"
    ].mean()
)

print("\nDense Recall@5:")
print(
    comparison_df[
        "recall_at_5_dense"
    ].mean()
)


print("\nBM25 Hit@5:")
print(
    comparison_df[
        "hit_at_5_bm25"
    ].mean()
)

print("\nDense Hit@5:")
print(
    comparison_df[
        "hit_at_5_dense"
    ].mean()
)


print("\nBM25 Average First Relevant Rank:")
print(
    comparison_df[
        "first_relevant_rank_bm25"
    ].mean()
)

print("\nDense Average First Relevant Rank:")
print(
    comparison_df[
        "first_relevant_rank_dense"
    ].mean()
)


# ==================================
# Query-by-query Recall comparison
# ==================================

bm25_better = (
    comparison_df["recall_at_5_bm25"]
    >
    comparison_df["recall_at_5_dense"]
).sum()


dense_better = (
    comparison_df["recall_at_5_dense"]
    >
    comparison_df["recall_at_5_bm25"]
).sum()


same_recall = (
    comparison_df["recall_at_5_bm25"]
    ==
    comparison_df["recall_at_5_dense"]
).sum()


print("\n==============================")
print("QUERY-BY-QUERY RECALL")
print("==============================")


print("\nBM25 had higher Recall@5:")
print(bm25_better)

print("\nDense had higher Recall@5:")
print(dense_better)

print("\nSame Recall@5:")
print(same_recall)


# ==================================
# Retrieval failures
# ==================================

bm25_failures = comparison_df[
    comparison_df["hit_at_5_bm25"] == False
]

dense_failures = comparison_df[
    comparison_df["hit_at_5_dense"] == False
]


print("\n==============================")
print("FAILURE COUNTS")
print("==============================")


print("\nBM25 complete failures:")
print(len(bm25_failures))

print("\nDense complete failures:")
print(len(dense_failures))


# ==================================
# Find queries Dense rescued
# ==================================

dense_rescued = comparison_df[
    (comparison_df["hit_at_5_bm25"] == False)
    &
    (comparison_df["hit_at_5_dense"] == True)
]


print("\n==============================")
print("DENSE RESCUED BM25 FAILURES")
print("==============================")


for _, row in dense_rescued.iterrows():

    print("\n------------------------------")

    print("Question:")
    print(row["question_bm25"])

    print("BM25 Recall:")
    print(row["recall_at_5_bm25"])

    print("Dense Recall:")
    print(row["recall_at_5_dense"])


# ==================================
# Find queries BM25 rescued
# ==================================

bm25_rescued = comparison_df[
    (comparison_df["hit_at_5_dense"] == False)
    &
    (comparison_df["hit_at_5_bm25"] == True)
]


print("\n==============================")
print("BM25 RESCUED DENSE FAILURES")
print("==============================")


for _, row in bm25_rescued.iterrows():

    print("\n------------------------------")

    print("Question:")
    print(row["question_bm25"])

    print("BM25 Recall:")
    print(row["recall_at_5_bm25"])

    print("Dense Recall:")
    print(row["recall_at_5_dense"])


# ==================================
# Save comparison
# ==================================

comparison_df.to_csv(
    "data/processed/bm25_vs_dense_comparison.csv",
    index=False
)


print("\n==============================")
print("COMPARISON SAVED")
print("==============================")


print(
    "\ndata/processed/"
    "bm25_vs_dense_comparison.csv"
)