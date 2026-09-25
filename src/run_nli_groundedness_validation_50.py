"""Run evidence-grounded NLI labelling on the 50-response validation set."""

from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from hallucination_labeling import (
    decide_grounding, parse_bool, resolve_label_indices, score_grounding,
)

INPUT_FILE = Path("data/processed/rag_50_hallucination_annotation.csv")
OUTPUT_FILE = Path("data/processed/rag_50_nli_groundedness_validation_v3.csv")
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
SUPPORT_THRESHOLD = 0.50
REVIEW_MARGIN = 0.10


def main():
    df = pd.read_csv(INPUT_FILE)
    required = {
        "query_id", "question", "retrieved_context", "generated_answer",
        "abstained", "hallucination_label",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["abstained"] = df["abstained"].map(parse_bool)
    df["hallucination_label"] = pd.to_numeric(
        df["hallucination_label"], errors="raise"
    ).astype(int)

    print(f"Loading {NLI_MODEL}...")
    model = CrossEncoder(NLI_MODEL, max_length=512, device="cpu")
    label_indices = resolve_label_indices(model)
    print(f"Resolved model labels: {label_indices}")

    results = []
    for number, row in df.iterrows():
        if row["abstained"]:
            result = {
                "v3_claim": "", "v3_evidence_scope": "abstention",
                "v3_nli_contradiction": np.nan, "v3_nli_entailment": np.nan,
                "v3_nli_neutral": np.nan, "v3_decision": "abstention",
                "v3_hallucination_label": np.nan,
                "v3_decision_margin": np.nan,
                "v3_requires_manual_review": False,
            }
        else:
            scores = score_grounding(
                model, row["question"], row["generated_answer"],
                row["retrieved_context"], label_indices,
            )
            decision, label, margin = decide_grounding(
                scores, SUPPORT_THRESHOLD, REVIEW_MARGIN
            )
            result = {
                "v3_claim": scores["claim"],
                "v3_evidence_scope": scores["evidence_scope"],
                "v3_nli_contradiction": scores["contradiction"],
                "v3_nli_entailment": scores["entailment"],
                "v3_nli_neutral": scores["neutral"],
                "v3_decision": decision,
                "v3_hallucination_label": label,
                "v3_decision_margin": margin,
                "v3_requires_manual_review": decision == "manual_review",
            }
        results.append(result)
        print(f"Processed {number + 1}/{len(df)}: {result['v3_decision']}")

    result_df = pd.concat([df, pd.DataFrame(results)], axis=1)
    result_df.to_csv(OUTPUT_FILE, index=False)
    answered = result_df.loc[~result_df["abstained"]]
    decided = answered.loc[answered["v3_hallucination_label"].notna()].copy()
    decided["v3_hallucination_label"] = decided["v3_hallucination_label"].astype(int)
    agreement = (
        decided["v3_hallucination_label"] == decided["hallucination_label"]
    ).mean() if len(decided) else float("nan")
    print("\nNLI V3 validation complete")
    print(f"Answered responses: {len(answered)}")
    print(f"Automatically decided: {len(decided)}")
    print(f"Automation coverage: {len(decided) / len(answered):.3f}")
    print(f"Agreement on decided responses: {agreement:.3f}")
    print(f"Manual review required: {answered['v3_requires_manual_review'].sum()}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
