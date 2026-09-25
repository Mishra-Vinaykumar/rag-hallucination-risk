from datasets import load_dataset
import pandas as pd

from preprocessing import create_chunks
from bm25_retriever import retrieve_bm25
from retrieval_evaluation import evaluate_retrieval


print("Loading HotpotQA...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor"
)

train_data = dataset["train"]

# Development experiment:
# use first 100 questions only
small_data = train_data.select(range(100))

print("Dataset loaded.")
print("Questions to process:", len(small_data))


all_trace_rows = []
all_summary_rows = []


for query_number, record in enumerate(
    small_data,
    start=1
):

    # -----------------------------
    # 1. Create chunks
    # -----------------------------

    chunks = create_chunks(record)


    # -----------------------------
    # 2. Run BM25
    # -----------------------------

    bm25_results = retrieve_bm25(
        query=record["question"],
        chunks=chunks,
        top_k=5
    )


    # -----------------------------
    # 3. Evaluate retrieval
    # -----------------------------

    evaluation = evaluate_retrieval(
        results=bm25_results,
        all_chunks=chunks
    )


    # -----------------------------
    # 4. Save Top-5 trace rows
    # -----------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        trace_row = {
            "query_id": record["id"],
            "question": record["question"],
            "expected_answer": record["answer"],
            "question_type": record["type"],
            "difficulty": record["level"],
            "retrieval_method": "BM25",
            "rank": rank,
            "document_id": result["document_id"],
            "chunk_id": result["chunk_id"],
            "title": result["title"],
            "text": result["text"],
            "retrieval_score": result["bm25_score"],
            "is_supporting": result["is_supporting"]
        }

        all_trace_rows.append(trace_row)


    # -----------------------------
    # 5. Save one summary row
    # -----------------------------

    summary_row = {
        "query_id": record["id"],
        "question": record["question"],
        "question_type": record["type"],
        "difficulty": record["level"],
        "retrieval_method": "BM25",
        "top_k": evaluation["k"],
        "total_gold_chunks": evaluation["total_relevant"],
        "gold_chunks_retrieved": evaluation["retrieved_relevant"],
        "precision_at_5": evaluation["precision_at_k"],
        "recall_at_5": evaluation["recall_at_k"],
        "hit_at_5": evaluation["hit_at_k"],
        "first_relevant_rank": evaluation["first_relevant_rank"]
    }

    all_summary_rows.append(summary_row)


    # Progress message
    print(
        f"Processed {query_number}/100"
    )


# ==================================
# Create DataFrames
# ==================================

trace_df = pd.DataFrame(
    all_trace_rows
)

summary_df = pd.DataFrame(
    all_summary_rows
)


# ==================================
# Save files
# ==================================

trace_df.to_csv(
    "data/processed/bm25_100_query_trace.csv",
    index=False
)

summary_df.to_csv(
    "data/processed/bm25_100_query_summary.csv",
    index=False
)


print("\n==============================")
print("BM25 100-QUERY EXPERIMENT DONE")
print("==============================")

print("\nQuestions processed:")
print(len(summary_df))

print("\nRetrieval trace rows:")
print(len(trace_df))

print("\nAverage Precision@5:")
print(summary_df["precision_at_5"].mean())

print("\nAverage Recall@5:")
print(summary_df["recall_at_5"].mean())

print("\nHit@5 Rate:")
print(summary_df["hit_at_5"].mean())

print("\nTrace saved to:")
print(
    "data/processed/bm25_100_query_trace.csv"
)

print("\nSummary saved to:")
print(
    "data/processed/bm25_100_query_summary.csv"
)