# Experimental database ERD

```mermaid
erDiagram
    QUESTIONS ||--o{ DOCUMENTS : contains
    DOCUMENTS ||--o{ DOCUMENT_CHUNKS : contains
    QUESTIONS ||--o{ RETRIEVAL_EVENTS : queried
    DOCUMENT_CHUNKS ||--o{ RETRIEVAL_EVENTS : retrieved
    EXPERIMENTS ||--o{ RETRIEVAL_EVENTS : records
    EXPERIMENTS ||--o{ RETRIEVAL_METRICS : measures
    QUESTIONS ||--o{ RETRIEVAL_METRICS : evaluated_for
    EXPERIMENTS ||--o{ RESPONSES : generates
    QUESTIONS ||--o{ RESPONSES : answered_by
    RESPONSES ||--o{ HUMAN_ANNOTATIONS : reviewed_by
    RESPONSES ||--o{ AUTOMATED_LABELS : evaluated_by
    EXPERIMENTS ||--o{ RETRIEVAL_FEATURES : produces
    QUESTIONS ||--o{ RETRIEVAL_FEATURES : described_by
```

The schema separates experimental observations from their configuration and
keeps human ground truth independent from automated evaluator outputs. This
allows evaluator versions to be compared without overwriting annotations.
