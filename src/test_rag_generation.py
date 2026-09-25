import torch

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from preprocessing import create_chunks
from hybrid_retriever import retrieve_hybrid


# ==========================================
# Configuration
# ==========================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)


# ==========================================
# Build RAG prompt
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
Answer the question using only the information
provided in the retrieved context below.

You may combine information from multiple retrieved
contexts and make straightforward logical inferences
from the information provided.

For example, if the question asks which event happened
first and the retrieved context provides the relevant
dates, compare those dates to determine the answer.

Do not use outside knowledge.

Do not invent facts that are not supported by the
retrieved context.

If the answer cannot be determined from the retrieved
context, say exactly:

"The retrieved context is insufficient."

Give only the concise final answer.

RETRIEVED CONTEXT:

{combined_context}

QUESTION:

{question}
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

record = dataset["train"][1]


print("\nQuestion:")
print(record["question"])


# ==========================================
# Load embedding model
# ==========================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded!")


# ==========================================
# Create chunks
# ==========================================

chunks = create_chunks(
    record
)


# ==========================================
# Hybrid retrieval
# ==========================================

print("\nRunning Hybrid retrieval...")

retrieved_results = retrieve_hybrid(
    query=record["question"],
    chunks=chunks,
    model=embedding_model,
    top_k=5,
    alpha=0.5
)

print("Retrieval complete!")


# ==========================================
# Build RAG prompt
# ==========================================

rag_prompt = build_rag_prompt(
    question=record["question"],
    retrieved_results=retrieved_results
)


# ==========================================
# Load Qwen tokenizer
# ==========================================

print("\nLoading LLM tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL
)

print("Tokenizer loaded!")


# ==========================================
# Load Qwen model
# ==========================================

print("\nLoading LLM...")

llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype="auto",
    device_map="auto"
)

print("LLM loaded!")

print("\nLLM device:")
print(llm.device)


# ==========================================
# Prepare Qwen chat format
# ==========================================

messages = [
    {
        "role": "system",
        "content": (
            "You are a question-answering "
            "assistant. Follow the user's "
            "instructions carefully."
        )
    },
    {
        "role": "user",
        "content": rag_prompt
    }
]


formatted_prompt = (
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
)


# ==========================================
# Tokenize
# ==========================================

inputs = tokenizer(
    formatted_prompt,
    return_tensors="pt"
)

inputs = inputs.to(
    llm.device
)


# ==========================================
# Generate answer
# ==========================================

print("\nGenerating answer...")


with torch.no_grad():

    output_ids = llm.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )


# ==========================================
# Extract ONLY newly generated tokens
# ==========================================

generated_ids = output_ids[
    :,
    inputs["input_ids"].shape[1]:
]


generated_answer = tokenizer.decode(
    generated_ids[0],
    skip_special_tokens=True
).strip()


# ==========================================
# Show result
# ==========================================

print("\n")
print("=" * 70)
print("RAG GENERATION RESULT")
print("=" * 70)


print("\nQuestion:")
print(
    record["question"]
)


print("\nRetrieved contexts:")

for rank, result in enumerate(
    retrieved_results,
    start=1
):

    print(
        f"\nContext {rank}:"
    )

    print(
        result["title"]
    )

    print(
        result["text"]
    )


print("\n------------------------------")

print("GENERATED ANSWER:")

print(
    generated_answer
)


print("\n------------------------------")

print(
    "EXPECTED ANSWER "
    "(researcher only):"
)

print(
    record["answer"]
)


print("\n==============================")
print("END-TO-END RAG TEST COMPLETE")
print("==============================")