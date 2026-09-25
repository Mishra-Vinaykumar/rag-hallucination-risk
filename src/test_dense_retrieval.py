from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from preprocessing import create_chunks
from dense_retriever import retrieve_dense
from retrieval_evaluation import evaluate_retrieval


print("Loading HotpotQA...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor"
)

train_data = dataset["train"]


# Use first question only
record = train_data[0]


print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Model loaded successfully!")


# Create chunks for first question
chunks = create_chunks(record)


print("\nQuestion:")
print(record["question"])

print("\nTotal chunks:")
print(len(chunks))


# Run Dense Retrieval
results = retrieve_dense(
    query=record["question"],
    chunks=chunks,
    model=model,
    top_k=5
)


print("\n==============================")
print("DENSE TOP 5 RESULTS")
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

    print("Dense Similarity:")
    print(result["dense_score"])

    print("Gold Supporting Chunk:")
    print(result["is_supporting"])

print("\n==============================")
print("DENSE RETRIEVAL EVALUATION")
print("==============================")

evaluation = evaluate_retrieval(
    results=results,
    all_chunks=chunks
)

print("\nTop K:")
print(evaluation["k"])

print("\nTotal Gold Supporting Chunks:")
print(evaluation["total_relevant"])

print("\nGold Chunks Retrieved:")
print(evaluation["retrieved_relevant"])

print("\nPrecision@5:")
print(evaluation["precision_at_k"])

print("\nRecall@5:")
print(evaluation["recall_at_k"])

print("\nHit@5:")
print(evaluation["hit_at_k"])

print("\nFirst Relevant Rank:")
print(evaluation["first_relevant_rank"])