"""Reusable utilities for evidence-grounded hallucination labelling."""

import re
import numpy as np


def parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def softmax(logits):
    values = np.asarray(logits, dtype=float)
    values = values - np.max(values)
    exponentials = np.exp(values)
    return exponentials / exponentials.sum()


def resolve_label_indices(model):
    """Read the model configuration rather than assuming a label order."""
    config = getattr(getattr(model, "model", None), "config", None)
    raw_mapping = getattr(config, "id2label", None) or {}
    mapping = {int(index): str(label).lower() for index, label in raw_mapping.items()}
    aliases = {
        "contradiction": {"contradiction", "contradictory"},
        "entailment": {"entailment", "entails"},
        "neutral": {"neutral"},
    }
    resolved = {}
    for canonical, names in aliases.items():
        matches = [index for index, label in mapping.items() if label in names]
        if len(matches) != 1:
            raise ValueError(
                f"Could not resolve {canonical!r} from model id2label={raw_mapping!r}"
            )
        resolved[canonical] = matches[0]
    return resolved


def build_grounding_claim(question, answer):
    return (
        f"For the question '{str(question).strip()}', "
        f"the answer is '{str(answer).strip()}'."
    )


def split_context_blocks(context):
    """Return complete context plus individual retrieved context blocks."""
    text = str(context).strip()
    if not text:
        return []
    parts = [
        part.strip()
        for part in re.split(r"(?=\[Context\s+\d+\])", text, flags=re.IGNORECASE)
        if part.strip()
    ]
    return [text, *parts] if len(parts) > 1 else [text]


def score_grounding(model, question, answer, context, label_indices):
    claim = build_grounding_claim(question, answer)
    premises = split_context_blocks(context)
    if not premises:
        return {
            "claim": claim,
            "contradiction": 0.0,
            "entailment": 0.0,
            "neutral": 1.0,
            "evidence_scope": "missing_context",
        }
    logits = np.atleast_2d(model.predict([(premise, claim) for premise in premises]))
    probabilities = np.vstack([softmax(row) for row in logits])
    best_index = int(np.argmax(probabilities[:, label_indices["entailment"]]))
    best = probabilities[best_index]
    return {
        "claim": claim,
        "contradiction": float(best[label_indices["contradiction"]]),
        "entailment": float(best[label_indices["entailment"]]),
        "neutral": float(best[label_indices["neutral"]]),
        "evidence_scope": "combined" if best_index == 0 else f"context_{best_index}",
    }


def decide_grounding(scores, support_threshold=0.50, review_margin=0.10):
    """Return a decision without silently converting uncertainty into a label."""
    entailment = scores["entailment"]
    strongest_unsupported = max(scores["contradiction"], scores["neutral"])
    margin = abs(entailment - strongest_unsupported)
    if margin < review_margin:
        return "manual_review", None, margin
    if entailment >= support_threshold and entailment > strongest_unsupported:
        return "supported", 0, margin
    return "unsupported", 1, margin
