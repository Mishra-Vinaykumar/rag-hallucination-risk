from datasets import load_dataset
from preprocessing import extract_documents, create_chunks
from bm25_retriever import retrieve_bm25
from retrieval_evaluation import evaluate_retrieval
from retrieval_trace import save_retrieval_trace

print("Loading HotpotQA dataset...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor"
)

print("\nDataset loaded successfully!")


# Get training dataset
train_data = dataset["train"]

print("\nTotal training records:")
print(len(train_data))


# Select only first 100 records for development
small_data = train_data.select(range(100))

print("\nDevelopment dataset created!")

print("Number of development records:")
print(len(small_data))


# Print first question
print("\nFirst development question:")
print(small_data[0]["question"])

print("\nCorrect answer:")
print(small_data[0]["answer"])


# Get first development record
first_record = small_data[0]

# Get context
context = first_record["context"]

titles = context["title"]
sentences = context["sentences"]


print("\n==============================")
print("CONTEXT DOCUMENTS")
print("==============================")


for i in range(len(titles)):

    title = titles[i]

    document_sentences = sentences[i]

    document_text = " ".join(document_sentences)

    print(f"\nDocument {i + 1}")
    print(f"Title: {title}")
    print(f"Text: {document_text}")

print("\n==============================")
print("STRUCTURED DOCUMENT RECORDS")
print("==============================")

structured_documents = extract_documents(first_record)

for document in structured_documents:

    print("\n------------------------------")

    print("Query ID:")
    print(document["query_id"])

    print("Document ID:")
    print(document["document_id"])

    print("Title:")
    print(document["title"])

    print("Supporting document:")
    print(document["is_supporting"])

print("\n==============================")
print("CHUNKS")
print("==============================")

chunks = create_chunks(first_record)

print("\nTotal chunks:")
print(len(chunks))

for chunk in chunks:

    print("\n------------------------------")

    print("Chunk ID:")
    print(chunk["chunk_id"])

    print("Document ID:")
    print(chunk["document_id"])

    print("Title:")
    print(chunk["title"])

    print("Text:")
    print(chunk["text"])

    print("Supporting:")
    print(chunk["is_supporting"])

print("\n==============================")
print("BM25 TOP 5 RESULTS")
print("==============================")

query = first_record["question"]

bm25_results = retrieve_bm25(
    query=query,
    chunks=chunks,
    top_k=5
)

print("\nQuestion:")
print(query)

for rank, result in enumerate(
    bm25_results,
    start=1
):

    print("\n------------------------------")

    print(f"Rank: {rank}")

    print("Chunk ID:")
    print(result["chunk_id"])

    print("Title:")
    print(result["title"])

    print("Text:")
    print(result["text"])

    print("BM25 Score:")
    print(result["bm25_score"])

    print("Gold Supporting Chunk:")
    print(result["is_supporting"])

print("\n==============================")
print("BM25 RETRIEVAL EVALUATION")
print("==============================")

evaluation = evaluate_retrieval(
    results=bm25_results,
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

print("\n==============================")
print("SAVING RETRIEVAL TRACE")
print("==============================")

trace_dataframe = save_retrieval_trace(
    record=first_record,
    results=bm25_results,
    output_path="data/processed/bm25_first_query_trace.csv"
)

print("\nRetrieval trace saved successfully!")

print("\nSaved rows:")
print(len(trace_dataframe))

print("\nSaved file:")
print("data/processed/bm25_first_query_trace.csv")