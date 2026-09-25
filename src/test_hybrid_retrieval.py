from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from preprocessing import create_chunks
from hybrid_retriever import retrieve_hybrid
from retrieval_evaluation import evaluate_retrieval


print("Loading HotpotQA...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor"
)

train_data = dataset["train"]

record = train_data[0]


print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Model loaded successfully!")


chunks = create_chunks(record)


results = retrieve_hybrid(
    query=record["question"],
    chunks=chunks,
    model=model,
    top_k=5,
    alpha=0.5
)


print("\nQuestion:")
print(record["question"])


print("\n==============================")
print("HYBRID TOP 5 RESULTS")
print("==============================")


for rank, result in enumerate(
    results,
    start=1
):

    print("\n------------------------------")

    print("Rank:")
    print(rank)

    print("Chunk ID:")
    print(result["chunk_id"])

    print("Title:")
    print(result["title"])

    print("Text:")
    print(result["text"])

    print("BM25 Score:")
    print(result["bm25_score"])

    print("Dense Score:")
    print(result["dense_score"])

    print("Hybrid Score:")
    print(result["hybrid_score"])

    print("Gold Supporting:")
    print(result["is_supporting"])


# --------------------------------
# Evaluate
# --------------------------------

evaluation = evaluate_retrieval(
    results=results,
    all_chunks=chunks
)


print("\n==============================")
print("HYBRID RETRIEVAL EVALUATION")
print("==============================")


print("\nPrecision@5:")
print(evaluation["precision_at_k"])

print("\nRecall@5:")
print(evaluation["recall_at_k"])

print("\nHit@5:")
print(evaluation["hit_at_k"])

print("\nFirst Relevant Rank:")
print(evaluation["first_relevant_rank"])