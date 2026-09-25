from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd

from preprocessing import create_chunks
from bm25_retriever import retrieve_bm25
from dense_retriever import retrieve_dense
from hybrid_retriever import retrieve_hybrid
from retrieval_evaluation import evaluate_retrieval


# ==========================================
# Experiment configuration
# ==========================================

TOP_K_VALUES = [3, 5, 10]

NUMBER_OF_QUESTIONS = 100

HYBRID_ALPHA = 0.5


# ==========================================
# Load dataset
# ==========================================

print("Loading HotpotQA...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor"
)

train_data = dataset["train"]

small_data = train_data.select(
    range(NUMBER_OF_QUESTIONS)
)

print("Dataset loaded.")

print(
    "Questions:",
    len(small_data)
)


# ==========================================
# Load embedding model once
# ==========================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ==========================================
# Store experiment results
# ==========================================

summary_rows = []


# ==========================================
# Process questions
# ==========================================

for query_number, record in enumerate(
    small_data,
    start=1
):

    chunks = create_chunks(record)


    # --------------------------------------
    # Retrieve maximum K once
    # --------------------------------------

    max_k = max(TOP_K_VALUES)


    bm25_full = retrieve_bm25(
        query=record["question"],
        chunks=chunks,
        top_k=max_k
    )


    dense_full = retrieve_dense(
        query=record["question"],
        chunks=chunks,
        model=model,
        top_k=max_k
    )


    hybrid_full = retrieve_hybrid(
        query=record["question"],
        chunks=chunks,
        model=model,
        top_k=max_k,
        alpha=HYBRID_ALPHA
    )


    retrieval_outputs = {
        "BM25": bm25_full,
        "Dense": dense_full,
        "Hybrid": hybrid_full
    }


    # --------------------------------------
    # Evaluate each Top-K
    # --------------------------------------

    for method_name, full_results in (
        retrieval_outputs.items()
    ):

        for top_k in TOP_K_VALUES:

            results = full_results[:top_k]

            evaluation = evaluate_retrieval(
                results=results,
                all_chunks=chunks
            )


            row = {
                "query_id": record["id"],

                "question":
                    record["question"],

                "question_type":
                    record["type"],

                "difficulty":
                    record["level"],

                "retrieval_method":
                    method_name,

                "top_k":
                    top_k,

                "hybrid_alpha":
                    HYBRID_ALPHA
                    if method_name == "Hybrid"
                    else None,

                "total_gold_chunks":
                    evaluation[
                        "total_relevant"
                    ],

                "gold_chunks_retrieved":
                    evaluation[
                        "retrieved_relevant"
                    ],

                "precision":
                    evaluation[
                        "precision_at_k"
                    ],

                "recall":
                    evaluation[
                        "recall_at_k"
                    ],

                "hit":
                    evaluation[
                        "hit_at_k"
                    ],

                "first_relevant_rank":
                    evaluation[
                        "first_relevant_rank"
                    ]
            }

            summary_rows.append(row)


    print(
        f"Processed "
        f"{query_number}/"
        f"{NUMBER_OF_QUESTIONS}"
    )


# ==========================================
# Create DataFrame
# ==========================================

summary_df = pd.DataFrame(
    summary_rows
)


# ==========================================
# Save raw results
# ==========================================

output_path = (
    "data/processed/"
    "retrieval_topk_experiment.csv"
)

summary_df.to_csv(
    output_path,
    index=False
)


# ==========================================
# Aggregate results
# ==========================================

aggregate = (
    summary_df
    .groupby(
        [
            "retrieval_method",
            "top_k"
        ]
    )
    .agg(
        average_precision=(
            "precision",
            "mean"
        ),

        average_recall=(
            "recall",
            "mean"
        ),

        hit_rate=(
            "hit",
            "mean"
        ),

        average_first_rank=(
            "first_relevant_rank",
            "mean"
        )
    )
    .reset_index()
)


print(
    "\n=============================="
)

print(
    "TOP-K RETRIEVAL EXPERIMENT"
)

print(
    "=============================="
)


print(
    aggregate.to_string(
        index=False
    )
)


# ==========================================
# Save aggregate table
# ==========================================

aggregate_path = (
    "data/processed/"
    "retrieval_topk_aggregate.csv"
)

aggregate.to_csv(
    aggregate_path,
    index=False
)


print(
    "\nDetailed results saved to:"
)

print(
    output_path
)


print(
    "\nAggregate results saved to:"
)

print(
    aggregate_path
)