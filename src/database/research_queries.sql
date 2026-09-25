-- Retrieval quality by method and Top-K.
SELECT retrieval_method, top_k,
       ROUND(AVG(precision_at_k), 4) AS mean_precision,
       ROUND(AVG(recall_at_k), 4) AS mean_recall,
       ROUND(AVG(hit_at_k), 4) AS hit_rate
FROM v_retrieval_analysis
WHERE experiment_key LIKE 'retrieval_topk_%'
GROUP BY retrieval_method, top_k
ORDER BY retrieval_method, top_k;

-- Human hallucination rate among answered responses.
SELECT e.retrieval_method,
       COUNT(*) AS answered_responses,
       ROUND(AVG(h.hallucination_label), 4) AS hallucination_rate
FROM responses r
JOIN experiments e ON e.experiment_id = r.experiment_id
JOIN human_annotations h ON h.response_id = r.response_id
WHERE r.abstained = 0
GROUP BY e.retrieval_method;

-- Human/automatic disagreements requiring inspection.
SELECT query_id, question_text, generated_answer,
       context_support, human_hallucination_label,
       automatic_hallucination_label, evaluator_version
FROM v_response_labels
WHERE abstained = 0
  AND human_hallucination_label <> automatic_hallucination_label;

-- Retrieval features joined to gold retrieval metrics for research analysis.
SELECT e.retrieval_method, f.*, m.precision_at_k, m.recall_at_k, m.hit_at_k
FROM retrieval_features f
JOIN experiments e ON e.experiment_id = f.experiment_id
LEFT JOIN retrieval_metrics m
  ON m.experiment_id = f.experiment_id AND m.query_id = f.query_id
ORDER BY e.retrieval_method, f.query_id;

-- Low-confidence retrieval candidates based only on deployable trace features.
SELECT e.retrieval_method, q.question_text, f.score_max,
       f.rank1_rank2_gap, f.score_entropy, f.lexical_overlap
FROM retrieval_features f
JOIN experiments e ON e.experiment_id = f.experiment_id
JOIN questions q ON q.query_id = f.query_id
ORDER BY f.score_max ASC, f.score_entropy DESC
LIMIT 25;
