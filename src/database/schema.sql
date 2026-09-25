PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS questions (
    query_id TEXT PRIMARY KEY,
    question_text TEXT NOT NULL,
    expected_answer TEXT,
    question_type TEXT,
    difficulty TEXT,
    dataset_source TEXT NOT NULL DEFAULT 'HotpotQA',
    experiment_split TEXT,
    source_split TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS documents (
    query_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    title TEXT NOT NULL,
    source TEXT,
    document_text TEXT,
    PRIMARY KEY (query_id, document_id),
    FOREIGN KEY (query_id) REFERENCES questions(query_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS document_chunks (
    query_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    token_count INTEGER,
    is_supporting INTEGER CHECK (is_supporting IN (0, 1)),
    PRIMARY KEY (query_id, chunk_id),
    FOREIGN KEY (query_id, document_id)
        REFERENCES documents(query_id, document_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_key TEXT NOT NULL UNIQUE,
    stage TEXT NOT NULL CHECK (stage IN ('retrieval', 'generation', 'labelling', 'prediction')),
    retrieval_method TEXT,
    top_k INTEGER CHECK (top_k IS NULL OR top_k > 0),
    hybrid_alpha REAL CHECK (hybrid_alpha IS NULL OR hybrid_alpha BETWEEN 0 AND 1),
    embedding_model TEXT,
    llm_model TEXT,
    prompt_version TEXT,
    source_file TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS retrieval_events (
    retrieval_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    query_id TEXT NOT NULL,
    rank INTEGER NOT NULL CHECK (rank > 0),
    document_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    retrieval_score REAL,
    bm25_score REAL,
    dense_score REAL,
    normalized_bm25 REAL,
    normalized_dense REAL,
    hybrid_score REAL,
    UNIQUE (experiment_id, query_id, rank),
    UNIQUE (experiment_id, query_id, chunk_id),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY (query_id, chunk_id)
        REFERENCES document_chunks(query_id, chunk_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS retrieval_metrics (
    experiment_id INTEGER NOT NULL,
    query_id TEXT NOT NULL,
    total_gold_chunks INTEGER,
    gold_chunks_retrieved INTEGER,
    precision_at_k REAL,
    recall_at_k REAL,
    hit_at_k INTEGER CHECK (hit_at_k IN (0, 1)),
    first_relevant_rank INTEGER,
    PRIMARY KEY (experiment_id, query_id),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY (query_id) REFERENCES questions(query_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS responses (
    response_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    query_id TEXT NOT NULL,
    generated_answer TEXT NOT NULL,
    raw_generation TEXT,
    retrieved_context TEXT,
    abstained INTEGER NOT NULL CHECK (abstained IN (0, 1)),
    exact_match INTEGER CHECK (exact_match IN (0, 1)),
    answer_f1 REAL,
    correctness_status TEXT,
    source_file TEXT NOT NULL,
    UNIQUE (experiment_id, query_id),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY (query_id) REFERENCES questions(query_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS human_annotations (
    human_annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id INTEGER NOT NULL,
    rubric_version TEXT NOT NULL,
    context_support TEXT NOT NULL CHECK (
        context_support IN ('supported', 'partially_supported', 'unsupported', 'abstention')
    ),
    hallucination_label INTEGER NOT NULL CHECK (hallucination_label IN (0, 1)),
    annotation_reason TEXT NOT NULL,
    annotator_confidence TEXT NOT NULL CHECK (
        annotator_confidence IN ('high', 'medium', 'low')
    ),
    annotator_id TEXT,
    source_file TEXT NOT NULL,
    UNIQUE (response_id, rubric_version),
    FOREIGN KEY (response_id) REFERENCES responses(response_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS automated_labels (
    automated_label_id INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id INTEGER NOT NULL,
    evaluator_version TEXT NOT NULL,
    evaluator_model TEXT,
    hypothesis TEXT,
    contradiction_probability REAL,
    entailment_probability REAL,
    neutral_probability REAL,
    nli_label TEXT,
    hallucination_label INTEGER CHECK (hallucination_label IN (0, 1)),
    decision_margin REAL,
    requires_manual_review INTEGER NOT NULL DEFAULT 0 CHECK (requires_manual_review IN (0, 1)),
    source_file TEXT NOT NULL,
    UNIQUE (response_id, evaluator_version),
    FOREIGN KEY (response_id) REFERENCES responses(response_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS retrieval_features (
    experiment_id INTEGER NOT NULL,
    query_id TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    score_max REAL,
    score_min REAL,
    score_mean REAL,
    score_median REAL,
    score_stddev REAL,
    score_range REAL,
    score_coefficient_variation REAL,
    rank1_rank2_gap REAL,
    rank1_rankk_decay REAL,
    score_entropy REAL,
    normalized_score_entropy REAL,
    near_top_passage_count INTEGER,
    unique_document_count INTEGER,
    source_diversity_ratio REAL,
    duplicate_context_ratio REAL,
    context_character_count INTEGER,
    context_token_count INTEGER,
    query_character_count INTEGER,
    query_token_count INTEGER,
    lexical_overlap REAL,
    mean_pairwise_context_diversity REAL,
    bm25_dense_score_correlation REAL,
    PRIMARY KEY (experiment_id, query_id, feature_version),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY (query_id) REFERENCES questions(query_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS final_label_audits (
    response_id INTEGER PRIMARY KEY,
    human_hallucination_label INTEGER NOT NULL CHECK (human_hallucination_label IN (0, 1)),
    automatic_hallucination_label INTEGER CHECK (automatic_hallucination_label IN (0, 1)),
    final_hallucination_label INTEGER NOT NULL CHECK (final_hallucination_label IN (0, 1)),
    review_status TEXT NOT NULL CHECK (
        review_status IN ('agreement_confirmed', 'human_label_confirmed', 'abstention_confirmed')
    ),
    audit_reason TEXT NOT NULL,
    rubric_version TEXT NOT NULL,
    source_file TEXT NOT NULL,
    audited_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (response_id) REFERENCES responses(response_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pilot_ml_dataset (
    response_id INTEGER PRIMARY KEY,
    query_id TEXT NOT NULL,
    retrieval_experiment_id INTEGER NOT NULL,
    feature_version TEXT NOT NULL,
    pilot_split TEXT NOT NULL CHECK (pilot_split IN ('train', 'validation', 'test')),
    retrieval_method TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    score_max REAL NOT NULL,
    score_min REAL NOT NULL,
    score_mean REAL NOT NULL,
    score_median REAL NOT NULL,
    score_stddev REAL NOT NULL,
    score_range REAL NOT NULL,
    score_coefficient_variation REAL,
    rank1_rank2_gap REAL,
    rank1_rankk_decay REAL,
    score_entropy REAL NOT NULL,
    normalized_score_entropy REAL NOT NULL,
    near_top_passage_count INTEGER NOT NULL,
    unique_document_count INTEGER NOT NULL,
    source_diversity_ratio REAL NOT NULL,
    duplicate_context_ratio REAL NOT NULL,
    context_character_count INTEGER NOT NULL,
    context_token_count INTEGER NOT NULL,
    query_character_count INTEGER NOT NULL,
    query_token_count INTEGER NOT NULL,
    lexical_overlap REAL NOT NULL,
    mean_pairwise_context_diversity REAL NOT NULL,
    bm25_dense_score_correlation REAL,
    hallucination_label INTEGER NOT NULL CHECK (hallucination_label IN (0, 1)),
    FOREIGN KEY (response_id) REFERENCES responses(response_id) ON DELETE CASCADE,
    FOREIGN KEY (query_id) REFERENCES questions(query_id) ON DELETE CASCADE,
    FOREIGN KEY (retrieval_experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS import_runs (
    import_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    database_version TEXT NOT NULL,
    source_directory TEXT NOT NULL,
    details TEXT
);

CREATE INDEX IF NOT EXISTS idx_retrieval_events_query
    ON retrieval_events(query_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_events_experiment_score
    ON retrieval_events(experiment_id, retrieval_score DESC);
CREATE INDEX IF NOT EXISTS idx_responses_query ON responses(query_id);
CREATE INDEX IF NOT EXISTS idx_human_labels_label
    ON human_annotations(hallucination_label);
CREATE INDEX IF NOT EXISTS idx_automated_labels_label
    ON automated_labels(hallucination_label);

CREATE VIEW IF NOT EXISTS v_response_labels AS
SELECT
    r.response_id,
    r.query_id,
    q.question_text,
    e.experiment_key,
    e.retrieval_method,
    e.top_k,
    r.generated_answer,
    r.abstained,
    h.context_support,
    h.hallucination_label AS human_hallucination_label,
    a.hallucination_label AS automatic_hallucination_label,
    a.evaluator_version,
    a.requires_manual_review
FROM responses r
JOIN questions q ON q.query_id = r.query_id
JOIN experiments e ON e.experiment_id = r.experiment_id
LEFT JOIN human_annotations h ON h.response_id = r.response_id
LEFT JOIN automated_labels a ON a.response_id = r.response_id;

CREATE VIEW IF NOT EXISTS v_retrieval_analysis AS
SELECT
    e.experiment_key,
    e.retrieval_method,
    e.top_k,
    e.hybrid_alpha,
    m.query_id,
    q.question_type,
    q.difficulty,
    m.precision_at_k,
    m.recall_at_k,
    m.hit_at_k,
    m.first_relevant_rank
FROM retrieval_metrics m
JOIN experiments e ON e.experiment_id = m.experiment_id
JOIN questions q ON q.query_id = m.query_id;
