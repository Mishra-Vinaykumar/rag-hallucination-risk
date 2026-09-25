from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd

from preprocessing import create_chunks
from dense_retriever import retrieve_dense
from retrieval_evaluation import evaluate_retrieval


print("Loading HotpotQA...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor"
)

train_data = dataset["train"]

# Use exactly the same 100 questions
small_data = train_data.select(range(100))

print("Dataset loaded.")
print("Questions to process:", len(small_data))


# ==================================
# Load embedding model ONCE
# ==================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


all_trace_rows = []
all_summary_rows = []


# ==================================
# Process 100 questions
# ==================================

for query_number, record in enumerate(
    small_data,
    start=1
):

    # ------------------------------
    # 1. Create chunks
    # ------------------------------

    chunks = create_chunks(record)


    # ------------------------------
    # 2. Dense retrieval
    # ------------------------------

    dense_results = retrieve_dense(
        query=record["question"],
        chunks=chunks,
        model=model,
        top_k=5
    )


    # ------------------------------
    # 3. Evaluate retrieval
    # ------------------------------

    evaluation = evaluate_retrieval(
        results=dense_results,
        all_chunks=chunks
    )


    # ------------------------------
    # 4. Save detailed trace
    # ------------------------------

    for rank, result in enumerate(
        dense_results,
        start=1
    ):

        trace_row = {
            "query_id": record["id"],
            "question": record["question"],
            "expected_answer": record["answer"],
            "question_type": record["type"],
            "difficulty": record["level"],
            "retrieval_method": "Dense",
            "rank": rank,
            "document_id": result["document_id"],
            "chunk_id": result["chunk_id"],
            "title": result["title"],
            "text": result["text"],
            "retrieval_score": result["dense_score"],
            "is_supporting": result["is_supporting"]
        }

        all_trace_rows.append(trace_row)


    # ------------------------------
    # 5. Save summary
    # ------------------------------

    summary_row = {
        "query_id": record["id"],
        "question": record["question"],
        "question_type": record["type"],
        "difficulty": record["level"],
        "retrieval_method": "Dense",
        "top_k": evaluation["k"],
        "total_gold_chunks": evaluation["total_relevant"],
        "gold_chunks_retrieved": evaluation["retrieved_relevant"],
        "precision_at_5": evaluation["precision_at_k"],
        "recall_at_5": evaluation["recall_at_k"],
        "hit_at_5": evaluation["hit_at_k"],
        "first_relevant_rank": evaluation["first_relevant_rank"]
    }

    all_summary_rows.append(summary_row)


    print(
        f"Processed {query_number}/100"
    )


# ==================================
# Convert to DataFrames
# ==================================

trace_df = pd.DataFrame(
    all_trace_rows
)

summary_df = pd.DataFrame(
    all_summary_rows
)


# ==================================
# Save CSV files
# ==================================

trace_df.to_csv(
    "data/processed/dense_100_query_trace.csv",
    index=False
)

summary_df.to_csv(
    "data/processed/dense_100_query_summary.csv",
    index=False
)


# ==================================
# Final results
# ==================================

print("\n==============================")
print("DENSE 100-QUERY EXPERIMENT DONE")
print("==============================")

print("\nQuestions processed:")
print(len(summary_df))

print("\nRetrieval trace rows:")
print(len(trace_df))

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

print("\nAverage First Relevant Rank:")
print(
    summary_df["first_relevant_rank"].mean()
)

print("\nTrace saved to:")
print(
    "data/processed/dense_100_query_trace.csv"
)

print("\nSummary saved to:")
print(
    "data/processed/dense_100_query_summary.csv"
)