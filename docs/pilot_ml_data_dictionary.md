# Pilot ML dataset data dictionary

The target is `hallucination_label` (`0` = supported/faithful, `1` = partially
supported or unsupported). Abstentions are excluded because they make no factual
claim and would otherwise inflate the non-hallucination class.

| Column | Role | Description |
|---|---|---|
| query_id | Identifier | Stable HotpotQA query identifier; never a model feature. |
| retrieval_method | Configuration | Retrieval method; currently Hybrid only. |
| top_k | Configuration | Number of retrieved chunks; currently 5. |
| pilot_split | Split | Deterministic, label-stratified train/validation/test assignment. |
| score_max/min/mean/median/stddev/range | Feature | Distribution of retrieval scores. |
| score_coefficient_variation | Feature | Score standard deviation divided by absolute mean. |
| rank1_rank2_gap | Feature | Difference between the first two ranked scores. |
| rank1_rankk_decay | Feature | Difference between first and last Top-K scores. |
| score_entropy | Feature | Entropy after softmax-normalising the Top-K scores. |
| normalized_score_entropy | Feature | Entropy divided by the maximum entropy for K. |
| near_top_passage_count | Feature | Chunks within 10% of the observed score range from Rank 1. |
| unique_document_count | Feature | Distinct source documents in Top-K. |
| source_diversity_ratio | Feature | Unique document count divided by K. |
| duplicate_context_ratio | Feature | Fraction of normalised duplicate chunk texts. |
| context_character/token_count | Feature | Size of the retrieved context. |
| query_character/token_count | Feature | Surface complexity of the question. |
| lexical_overlap | Feature | Fraction of unique query tokens present in the context. |
| mean_pairwise_context_diversity | Feature | Mean pairwise Jaccard distance between chunks. |
| bm25_dense_score_correlation | Feature | Pearson correlation of BM25 and dense scores in hybrid results. |
| hallucination_label | Target | Final audited human label. |

Gold supporting-fact indicators, expected answers, correctness metrics and NLI
probabilities are deliberately excluded from the exported modelling columns.
