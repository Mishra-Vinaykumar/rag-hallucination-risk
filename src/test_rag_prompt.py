from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from preprocessing import create_chunks
from hybrid_retriever import retrieve_hybrid


# ==========================================
# Function to build RAG prompt
# ==========================================

def build_rag_prompt(
    question,
    retrieved_results
):

    context_blocks = []

    for rank, result in enumerate(
        retrieved_results,
        start=1
    ):

        context_block = (
            f"[Context {rank}]\n"
            f"Title: {result['title']}\n"
            f"Text: {result['text']}"
        )

        context_blocks.append(
            context_block
        )


    combined_context = "\n\n".join(
        context_blocks
    )


    prompt = f"""
You are a question-answering assistant.

Answer the question using only the information
provided in the retrieved context below.

Do not add facts that are not supported by the
retrieved context.

If the retrieved context does not contain enough
information to answer the question, state that
the context is insufficient.

RETRIEVED CONTEXT:

{combined_context}

QUESTION:

{question}

ANSWER:
""".strip()


    return prompt


# ==========================================
# Load HotpotQA
# ==========================================

print("Loading HotpotQA...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor"
)

record = dataset["train"][0]


# ==========================================
# Load embedding model
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Model loaded!")


# ==========================================
# Create chunks
# ==========================================

chunks = create_chunks(
    record
)


# ==========================================
# Retrieve Hybrid Top 5
# ==========================================

results = retrieve_hybrid(
    query=record["question"],
    chunks=chunks,
    model=model,
    top_k=5,
    alpha=0.5
)


# ==========================================
# Build prompt
# ==========================================

prompt = build_rag_prompt(
    question=record["question"],
    retrieved_results=results
)


# ==========================================
# Print prompt
# ==========================================

print("\n")
print("=" * 70)
print("RAG PROMPT")
print("=" * 70)

print(prompt)

print("\n")
print("=" * 70)
print("PROMPT CHECK")
print("=" * 70)

print(
    "Question:",
    record["question"]
)

print(
    "\nExpected answer for researcher only:"
)

print(
    record["answer"]
)

print(
    "\nIMPORTANT:"
)

print(
    "The expected answer above is NOT "
    "included inside the RAG prompt."
)