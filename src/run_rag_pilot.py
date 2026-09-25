import torch
import pandas as pd

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from preprocessing import create_chunks
from hybrid_retriever import retrieve_hybrid
from retrieval_evaluation import evaluate_retrieval


# ==========================================
# Configuration
# ==========================================

NUMBER_OF_QUESTIONS = 10

TOP_K = 5

HYBRID_ALPHA = 0.5

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

        block = (
            f"[Context {rank}]\n"
            f"Title: {result['title']}\n"
            f"Text: {result['text']}"
        )

        context_blocks.append(block)


    combined_context = "\n\n".join(
        context_blocks
    )


    prompt = f"""
Use only the retrieved context below to answer the question.

You may combine information from multiple contexts and
make straightforward logical inferences from the supplied
information.

Do not use outside knowledge.

Return EXACTLY ONE LINE.

If the answer can be determined from the retrieved context,
return:

ANSWER: <short final answer>

If the answer cannot be determined from the retrieved context,
return exactly:

ANSWER: INSUFFICIENT

Do not explain your reasoning.
Do not provide bullet points.
Do not provide additional text.

RETRIEVED CONTEXT:

{combined_context}

QUESTION:

{question}
""".strip()


    return prompt, combined_context


# ==========================================
# Parse model generation
# ==========================================

def parse_generation(raw_generation):

    text = str(
        raw_generation
    ).strip()


    # --------------------------------------
    # Standard format:
    # ANSWER: <answer>
    # --------------------------------------

    if text.upper().startswith("ANSWER:"):

        parsed_answer = (
            text.split(
                ":",
                1
            )[1]
            .strip()
        )

        return parsed_answer


    # --------------------------------------
    # Model sometimes returns
    # INSUFFICIENT without ANSWER:
    # --------------------------------------

    if text.upper() == "INSUFFICIENT":

        return "INSUFFICIENT"


    # --------------------------------------
    # Fallback for format non-compliance
    # --------------------------------------

    return text


# ==========================================
# Load dataset
# ==========================================

print("Loading HotpotQA...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor"
)

data = dataset["train"].select(
    range(NUMBER_OF_QUESTIONS)
)

print(
    "Questions:",
    len(data)
)


# ==========================================
# Load embedding model ONCE
# ==========================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded!")


# ==========================================
# Load LLM ONCE
# ==========================================

print("\nLoading LLM tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL
)

print("Tokenizer loaded!")


print("\nLoading LLM...")

llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype="auto",
    device_map="auto"
)

print("LLM loaded!")

print(
    "LLM device:",
    llm.device
)


# ==========================================
# Run pilot
# ==========================================

rows = []


for number, record in enumerate(
    data,
    start=1
):

    print(
        f"\nProcessing {number}/"
        f"{NUMBER_OF_QUESTIONS}"
    )


    # --------------------------------------
    # Chunk
    # --------------------------------------

    chunks = create_chunks(
        record
    )


    # --------------------------------------
    # Retrieve Hybrid Top-5
    # --------------------------------------

    retrieved = retrieve_hybrid(
        query=record["question"],
        chunks=chunks,
        model=embedding_model,
        top_k=TOP_K,
        alpha=HYBRID_ALPHA
    )


    # --------------------------------------
    # Retrieval evaluation
    # --------------------------------------

    retrieval_eval = evaluate_retrieval(
        results=retrieved,
        all_chunks=chunks
    )


    # --------------------------------------
    # Build prompt
    # --------------------------------------

    rag_prompt, context_text = (
        build_rag_prompt(
            question=record["question"],
            retrieved_results=retrieved
        )
    )


    # --------------------------------------
    # Chat template
    # --------------------------------------

    messages = [
        {
            "role": "system",
            "content": (
                "You are a retrieval-grounded "
                "question-answering system. "
                "Return only the requested answer "
                "format. Never provide reasoning "
                "or explanation."
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


    # --------------------------------------
    # Tokenize
    # --------------------------------------

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    )

    inputs = inputs.to(
        llm.device
    )


    # --------------------------------------
    # Generate
    # --------------------------------------

    with torch.no_grad():

        output_ids = llm.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=(
                tokenizer.eos_token_id
            )
        )


    generated_ids = output_ids[
        :,
        inputs["input_ids"].shape[1]:
    ]


    raw_generation = (
        tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        .strip()
    )


    parsed_answer = parse_generation(
        raw_generation
    )


    abstained = (
        parsed_answer.upper()
        ==
        "INSUFFICIENT"
    )


    # --------------------------------------
    # Save
    # --------------------------------------

    row = {

        "query_id":
            record["id"],

        "question":
            record["question"],

        "expected_answer":
            record["answer"],

        "question_type":
            record["type"],

        "difficulty":
            record["level"],

        "retrieval_method":
            "Hybrid",

        "top_k":
            TOP_K,

        "hybrid_alpha":
            HYBRID_ALPHA,

        "retrieval_precision":
            retrieval_eval[
                "precision_at_k"
            ],

        "retrieval_recall":
            retrieval_eval[
                "recall_at_k"
            ],

        "retrieval_hit":
            retrieval_eval[
                "hit_at_k"
            ],

        "first_relevant_rank":
            retrieval_eval[
                "first_relevant_rank"
            ],

        "retrieved_context":
            context_text,

        "raw_generation":
            raw_generation,

        "generated_answer":
            parsed_answer,

        "abstained":
            abstained
    }


    rows.append(
        row
    )


    # --------------------------------------
    # Print result
    # --------------------------------------

    print("Question:")
    print(
        record["question"]
    )

    print("Raw generation:")
    print(
        raw_generation
    )

    print("Parsed answer:")
    print(
        parsed_answer
    )

    print("Expected:")
    print(
        record["answer"]
    )

    print("Abstained:")
    print(
        abstained
    )


# ==========================================
# Save pilot
# ==========================================

pilot_df = pd.DataFrame(
    rows
)


output_path = (
    "data/processed/"
    "rag_10_query_pilot.csv"
)


pilot_df.to_csv(
    output_path,
    index=False
)


# ==========================================
# Summary
# ==========================================

print("\n")

print(
    "=" * 70
)

print(
    "10-QUESTION RAG PILOT COMPLETE"
)

print(
    "=" * 70
)


print("\nTotal questions:")

print(
    len(pilot_df)
)


print("\nAbstentions:")

print(
    pilot_df[
        "abstained"
    ].sum()
)


print("\nNon-abstentions:")

print(
    (
        ~pilot_df[
            "abstained"
        ]
    ).sum()
)


print("\nAverage retrieval recall:")

print(
    pilot_df[
        "retrieval_recall"
    ].mean()
)


print("\nSaved to:")

print(
    output_path
)