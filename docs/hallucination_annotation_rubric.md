# Frozen hallucination annotation rubric v1.0

## Unit of assessment

Assess the generated answer only against the retrieved context supplied to the
model. Expected-answer mismatch and outside knowledge may inform answer
correctness, but they do not determine contextual hallucination.

## Categories

| Context support | Hallucination label | Rule |
|---|---:|---|
| supported | 0 | Every material factual claim is entailed by the retrieved context. |
| partially_supported | 1 | At least one material claim is supported and at least one is unsupported. |
| unsupported | 1 | The answer is not established by the context or contradicts it. |
| abstention | 0 | The model explicitly returns `INSUFFICIENT` and makes no factual claim. |

## Review rules

1. Preserve comparisons, dates, negation and entity relationships from the question.
2. Treat a plausible answer as unsupported when the retrieved evidence does not establish it.
3. Treat contradictory evidence as hallucination even when the answer matches a reference answer.
4. Record a concrete evidence-based reason and confidence for every human label.
5. Automated NLI output is a review aid. It never overwrites a completed human annotation.
6. Report answer coverage and hallucination rate among answered responses separately.

## Frozen version

Version `rubric_v1.0` is frozen for the 60-response pilot. Future changes require
a new version and must preserve the original annotations and audit trail.
