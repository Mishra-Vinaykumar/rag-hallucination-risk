import pandas as pd


# Load the 100-query summary
summary_df = pd.read_csv(
    "data/processed/bm25_100_query_summary.csv"
)


print("==============================")
print("BM25 RESULT ANALYSIS")
print("==============================")


# Total queries
total_queries = len(summary_df)

print("\nTotal Queries:")
print(total_queries)


# ------------------------------
# Hit@5 analysis
# ------------------------------

hit_count = summary_df["hit_at_5"].sum()

miss_count = total_queries - hit_count

print("\nQueries with at least one gold chunk in Top 5:")
print(hit_count)

print("\nQueries with NO gold chunk in Top 5:")
print(miss_count)


# ------------------------------
# Recall analysis
# ------------------------------

full_recall = (
    summary_df["recall_at_5"] == 1.0
).sum()

zero_recall = (
    summary_df["recall_at_5"] == 0.0
).sum()

partial_recall = (
    (summary_df["recall_at_5"] > 0.0)
    &
    (summary_df["recall_at_5"] < 1.0)
).sum()


print("\nQueries with FULL Recall@5:")
print(full_recall)

print("\nQueries with PARTIAL Recall@5:")
print(partial_recall)

print("\nQueries with ZERO Recall@5:")
print(zero_recall)


# ------------------------------
# First relevant rank
# ------------------------------

average_first_rank = (
    summary_df["first_relevant_rank"].mean()
)

print("\nAverage First Relevant Rank:")
print(average_first_rank)


# ------------------------------
# Average metrics
# ------------------------------

print("\nAverage Precision@5:")
print(
    summary_df["precision_at_5"].mean()
)

print("\nAverage Recall@5:")
print(
    summary_df["recall_at_5"].mean()
)

print("\nHit@5 Rate:")
print(
    summary_df["hit_at_5"].mean()
)


# ------------------------------
# Show failures
# ------------------------------

failed_queries = summary_df[
    summary_df["hit_at_5"] == False
]


print("\n==============================")
print("FAILED QUERIES")
print("==============================")


if len(failed_queries) == 0:

    print("\nNo failed queries.")

else:

    for _, row in failed_queries.iterrows():

        print("\n------------------------------")

        print("Query ID:")
        print(row["query_id"])

        print("Question:")
        print(row["question"])

        print("Question Type:")
        print(row["question_type"])

        print("Difficulty:")
        print(row["difficulty"])

        print("Recall@5:")
        print(row["recall_at_5"])