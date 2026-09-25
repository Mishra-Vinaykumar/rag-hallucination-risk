import torch
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# ==========================================
# Configuration
# ==========================================

INPUT_FILE = (
    "data/processed/"
    "rag_50_nli_groundedness_disagreements.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "rag_50_nli_v3_claim_pilot.csv"
)

LLM_MODEL = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)


# ==========================================
# Build claim-rewriting prompt
# ==========================================

def build_claim_prompt(
    question,
    answer
):

    prompt = f"""
Convert the question and proposed answer into ONE factual
declarative claim that can be checked against evidence.

IMPORTANT RULES:

1. Do NOT decide whether the answer is correct.
2. Do NOT use outside knowledge.
3. Use only the meaning contained in the question and answer.
4. Preserve all important relationships from the question.
5. Preserve comparisons such as before, after, older, larger,
   more, fewer, first, or last.
6. For yes/no questions, express the complete proposition
   with the meaning of the proposed answer.
7. Do not explain anything.
8. Return exactly one line in this format:

CLAIM: <declarative factual claim>

QUESTION:
{question}

PROPOSED ANSWER:
{answer}
""".strip()

    return prompt


# ==========================================
# Parse claim
# ==========================================

def parse_claim(
    raw_generation
):

    text = str(
        raw_generation
    ).strip()

    if text.upper().startswith(
        "CLAIM:"
    ):

        return (
            text.split(
                ":",
                1
            )[1]
            .strip()
        )

    return text


# ==========================================
# Load disagreements
# ==========================================

print(
    "Loading NLI disagreements..."
)

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Rows loaded:",
    len(df)
)


# ==========================================
# Load tokenizer
# ==========================================

print(
    "\nLoading tokenizer..."
)

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        LLM_MODEL
    )
)

print(
    "Tokenizer loaded!"
)


# ==========================================
# Load model
# ==========================================

print(
    "\nLoading claim-rewriting LLM..."
)

llm = (
    AutoModelForCausalLM
    .from_pretrained(
        LLM_MODEL,
        torch_dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True
    )
)

print(
    "LLM loaded!"
)

print(
    "Device:",
    llm.device
)


# ==========================================
# Generate claims
# ==========================================

claims = []

raw_claim_generations = []


for number, row in df.iterrows():

    print(
        f"\nProcessing "
        f"{number + 1}/"
        f"{len(df)}"
    )


    question = str(
        row["question"]
    )

    answer = str(
        row["generated_answer"]
    )


    # --------------------------------------
    # Build prompt
    # --------------------------------------

    prompt = build_claim_prompt(
        question=question,
        answer=answer
    )


    messages = [
        {
            "role": "system",
            "content": (
                "You convert question-answer pairs "
                "into factual declarative claims. "
                "You do not judge whether the "
                "answer is correct."
            )
        },
        {
            "role": "user",
            "content": prompt
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
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=(
                tokenizer.eos_token_id
            )
        )


    generated_ids = output_ids[
        :,
        inputs[
            "input_ids"
        ].shape[1]:
    ]


    raw_generation = (
        tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        .strip()
    )


    claim = parse_claim(
        raw_generation
    )


    raw_claim_generations.append(
        raw_generation
    )

    claims.append(
        claim
    )


    # --------------------------------------
    # Print
    # --------------------------------------

    print(
        "\nQuestion:"
    )

    print(
        question
    )


    print(
        "\nGenerated answer:"
    )

    print(
        answer
    )


    print(
        "\nV3 claim:"
    )

    print(
        claim
    )


# ==========================================
# Save
# ==========================================

df[
    "v3_raw_claim_generation"
] = raw_claim_generations


df[
    "v3_claim"
] = claims


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# Summary
# ==========================================

print("\n")
print("=" * 70)

print(
    "NLI V3 CLAIM PILOT COMPLETE"
)

print("=" * 70)


print(
    "\nClaims generated:"
)

print(
    len(df)
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)


print(
    "\nIMPORTANT:"
)

print(
    "Inspect the claims before using "
    "them with the NLI model."
)