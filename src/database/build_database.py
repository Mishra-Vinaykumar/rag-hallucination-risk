"""Build the dissertation SQLite database from the existing processed CSV files."""

import argparse
import csv
import json
import math
import re
import sqlite3
import statistics
from pathlib import Path

DATABASE_VERSION = "1.0"
FEATURE_VERSION = "retrieval_trace_v2"


def text(value):
    return None if value is None or str(value).strip() == "" else str(value).strip()


def number(value, cast=float):
    value = text(value)
    if value is None or value.lower() == "nan":
        return None
    return int(float(value)) if cast is int else cast(value)


def boolean(value):
    value = text(value)
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return 1
    if lowered in {"false", "0", "no"}:
        return 0
    raise ValueError(f"Invalid boolean: {value!r}")


def rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def upsert_question(connection, row, split=None, source_split=None):
    query_id = text(row.get("query_id") or row.get("id"))
    question = text(row.get("question") or row.get("question_text"))
    if not query_id or not question:
        raise ValueError("Every question requires query_id and question text")
    connection.execute(
        """
        INSERT INTO questions (
            query_id, question_text, expected_answer, question_type, difficulty,
            experiment_split, source_split
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(query_id) DO UPDATE SET
            question_text=excluded.question_text,
            expected_answer=COALESCE(questions.expected_answer, excluded.expected_answer),
            question_type=COALESCE(questions.question_type, excluded.question_type),
            difficulty=COALESCE(questions.difficulty, excluded.difficulty),
            experiment_split=COALESCE(questions.experiment_split, excluded.experiment_split),
            source_split=COALESCE(questions.source_split, excluded.source_split)
        """,
        (
            query_id, question, text(row.get("answer") or row.get("expected_answer")),
            text(row.get("question_type")), text(row.get("difficulty")),
            split or text(row.get("experiment_split")),
            source_split or text(row.get("source_split")),
        ),
    )
    return query_id


def experiment(connection, key, stage, method, top_k, alpha, source_file,
               embedding_model=None, llm_model=None, prompt_version=None):
    connection.execute(
        """
        INSERT INTO experiments (
            experiment_key, stage, retrieval_method, top_k, hybrid_alpha,
            embedding_model, llm_model, prompt_version, source_file
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(experiment_key) DO NOTHING
        """,
        (key, stage, method, top_k, alpha, embedding_model, llm_model,
         prompt_version, source_file),
    )
    return connection.execute(
        "SELECT experiment_id FROM experiments WHERE experiment_key=?", (key,)
    ).fetchone()[0]


def import_splits(connection, processed):
    split_dir = processed / "experiment_splits"
    for name in ("development", "training", "validation", "final_test"):
        path = split_dir / f"{name}_queries.csv"
        for row in rows(path):
            upsert_question(connection, row, name, row.get("source_split"))


def import_trace(connection, path, method):
    key = f"retrieval_trace_{method.lower()}_k5"
    experiment_id = experiment(
        connection, key, "retrieval", method, 5,
        0.5 if method == "Hybrid" else None, path.name,
        "sentence-transformers/all-MiniLM-L6-v2" if method != "BM25" else None,
    )
    for row in rows(path):
        query_id = upsert_question(connection, row)
        document_id = text(row["document_id"])
        chunk_id = text(row["chunk_id"])
        connection.execute(
            """INSERT INTO documents(query_id, document_id, title)
               VALUES (?, ?, ?) ON CONFLICT(query_id, document_id)
               DO UPDATE SET title=excluded.title""",
            (query_id, document_id, text(row["title"]) or "Untitled"),
        )
        connection.execute(
            """INSERT INTO document_chunks(
                   query_id, chunk_id, document_id, chunk_text, is_supporting
               ) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(query_id, chunk_id) DO UPDATE SET
                   chunk_text=excluded.chunk_text,
                   is_supporting=excluded.is_supporting""",
            (query_id, chunk_id, document_id, text(row["text"]) or "",
             boolean(row.get("is_supporting"))),
        )
        score = number(row.get("retrieval_score") or row.get("hybrid_score"))
        connection.execute(
            """INSERT INTO retrieval_events(
                   experiment_id, query_id, rank, document_id, chunk_id,
                   retrieval_score, bm25_score, dense_score, normalized_bm25,
                   normalized_dense, hybrid_score
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(experiment_id, query_id, rank) DO UPDATE SET
                   document_id=excluded.document_id, chunk_id=excluded.chunk_id,
                   retrieval_score=excluded.retrieval_score,
                   bm25_score=excluded.bm25_score, dense_score=excluded.dense_score,
                   normalized_bm25=excluded.normalized_bm25,
                   normalized_dense=excluded.normalized_dense,
                   hybrid_score=excluded.hybrid_score""",
            (
                experiment_id, query_id, number(row["rank"], int), document_id,
                chunk_id, score, number(row.get("bm25_score")),
                number(row.get("dense_score")), number(row.get("normalized_bm25")),
                number(row.get("normalized_dense")), number(row.get("hybrid_score")),
            ),
        )
    return experiment_id


def import_summary(connection, path, experiment_id):
    for row in rows(path):
        query_id = upsert_question(connection, row)
        connection.execute(
            """INSERT INTO retrieval_metrics(
                   experiment_id, query_id, total_gold_chunks,
                   gold_chunks_retrieved, precision_at_k, recall_at_k,
                   hit_at_k, first_relevant_rank
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(experiment_id, query_id) DO UPDATE SET
                   total_gold_chunks=excluded.total_gold_chunks,
                   gold_chunks_retrieved=excluded.gold_chunks_retrieved,
                   precision_at_k=excluded.precision_at_k,
                   recall_at_k=excluded.recall_at_k,
                   hit_at_k=excluded.hit_at_k,
                   first_relevant_rank=excluded.first_relevant_rank""",
            (
                experiment_id, query_id, number(row.get("total_gold_chunks"), int),
                number(row.get("gold_chunks_retrieved"), int),
                number(row.get("precision_at_5") or row.get("precision")),
                number(row.get("recall_at_5") or row.get("recall")),
                boolean(row.get("hit_at_5") or row.get("hit")),
                number(row.get("first_relevant_rank"), int),
            ),
        )


def import_topk_metrics(connection, path):
    experiment_ids = {}
    for row in rows(path):
        method = text(row["retrieval_method"])
        top_k = number(row["top_k"], int)
        key = f"retrieval_topk_{method.lower()}_k{top_k}"
        experiment_ids.setdefault(
            key,
            experiment(
                connection, key, "retrieval", method, top_k,
                number(row.get("hybrid_alpha")), path.name,
                "sentence-transformers/all-MiniLM-L6-v2" if method != "BM25" else None,
            ),
        )
        query_id = upsert_question(connection, row)
        connection.execute(
            """INSERT INTO retrieval_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(experiment_id, query_id) DO UPDATE SET
                   total_gold_chunks=excluded.total_gold_chunks,
                   gold_chunks_retrieved=excluded.gold_chunks_retrieved,
                   precision_at_k=excluded.precision_at_k,
                   recall_at_k=excluded.recall_at_k,
                   hit_at_k=excluded.hit_at_k,
                   first_relevant_rank=excluded.first_relevant_rank""",
            (
                experiment_ids[key], query_id,
                number(row.get("total_gold_chunks"), int),
                number(row.get("gold_chunks_retrieved"), int),
                number(row.get("precision")), number(row.get("recall")),
                boolean(row.get("hit")), number(row.get("first_relevant_rank"), int),
            ),
        )


def import_responses(connection, path, key):
    experiment_id = experiment(
        connection, key, "generation", "Hybrid", 5, 0.5, path.name,
        "sentence-transformers/all-MiniLM-L6-v2", "Qwen/Qwen2.5-1.5B-Instruct",
        "grounded_answer_v1",
    )
    for row in rows(path):
        query_id = upsert_question(connection, row)
        connection.execute(
            """INSERT INTO responses(
                   experiment_id, query_id, generated_answer, raw_generation,
                   retrieved_context, abstained, exact_match, answer_f1,
                   correctness_status, source_file
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(experiment_id, query_id) DO UPDATE SET
                   generated_answer=excluded.generated_answer,
                   raw_generation=excluded.raw_generation,
                   retrieved_context=excluded.retrieved_context,
                   abstained=excluded.abstained, exact_match=excluded.exact_match,
                   answer_f1=excluded.answer_f1,
                   correctness_status=excluded.correctness_status""",
            (
                experiment_id, query_id, text(row["generated_answer"]) or "",
                text(row.get("raw_generation")), text(row.get("retrieved_context")),
                boolean(row["abstained"]), boolean(row.get("exact_match")),
                number(row.get("answer_f1")), text(row.get("correctness_status")),
                path.name,
            ),
        )
    return experiment_id


def response_id(connection, experiment_id, query_id):
    result = connection.execute(
        "SELECT response_id FROM responses WHERE experiment_id=? AND query_id=?",
        (experiment_id, query_id),
    ).fetchone()
    if not result:
        raise ValueError(f"No response for {query_id}")
    return result[0]


def import_human_annotations(connection, path, experiment_id):
    for row in rows(path):
        rid = response_id(connection, experiment_id, row["query_id"])
        connection.execute(
            """INSERT INTO human_annotations(
                   response_id, rubric_version, context_support,
                   hallucination_label, annotation_reason, annotator_confidence,
                   source_file
               ) VALUES (?, 'rubric_v1', ?, ?, ?, ?, ?)
               ON CONFLICT(response_id, rubric_version) DO UPDATE SET
                   context_support=excluded.context_support,
                   hallucination_label=excluded.hallucination_label,
                   annotation_reason=excluded.annotation_reason,
                   annotator_confidence=excluded.annotator_confidence,
                   source_file=excluded.source_file""",
            (
                rid, text(row["context_support"]).lower(),
                number(row["hallucination_label"], int),
                text(row["annotation_reason"]) or "No reason recorded",
                text(row["annotator_confidence"]).lower(), path.name,
            ),
        )


def import_automated_labels(connection, path, experiment_id):
    for row in rows(path):
        rid = response_id(connection, experiment_id, row["query_id"])
        abstained = boolean(row["abstained"])
        label = None if abstained else number(row.get("v2_hallucination_label"), int)
        connection.execute(
            """INSERT INTO automated_labels(
                   response_id, evaluator_version, evaluator_model, hypothesis,
                   contradiction_probability, entailment_probability,
                   neutral_probability, nli_label, hallucination_label,
                   requires_manual_review, source_file
               ) VALUES (?, 'nli_v2', ?, ?, ?, ?, ?, ?, ?, 0, ?)
               ON CONFLICT(response_id, evaluator_version) DO UPDATE SET
                   hypothesis=excluded.hypothesis,
                   contradiction_probability=excluded.contradiction_probability,
                   entailment_probability=excluded.entailment_probability,
                   neutral_probability=excluded.neutral_probability,
                   nli_label=excluded.nli_label,
                   hallucination_label=excluded.hallucination_label,
                   source_file=excluded.source_file""",
            (
                rid, "cross-encoder/nli-deberta-v3-base",
                text(row.get("v2_hypothesis")),
                number(row.get("v2_nli_contradiction")),
                number(row.get("v2_nli_entailment")),
                number(row.get("v2_nli_neutral")), text(row.get("v2_nli_label")),
                label, path.name,
            ),
        )


def tokens(value):
    return re.findall(r"\b\w+\b", value.lower())


def correlation(left, right):
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    denominator = left_scale * right_scale
    return numerator / denominator if denominator else None


def mean_pairwise_diversity(chunk_texts):
    token_sets = [set(tokens(value)) for value in chunk_texts]
    distances = []
    for index, left in enumerate(token_sets):
        for right in token_sets[index + 1:]:
            union = left | right
            similarity = len(left & right) / len(union) if union else 1.0
            distances.append(1.0 - similarity)
    return statistics.fmean(distances) if distances else 0.0


def engineer_features(connection):
    groups = connection.execute(
        """SELECT experiment_id, query_id FROM retrieval_events
           GROUP BY experiment_id, query_id"""
    ).fetchall()
    for experiment_id, query_id in groups:
        event_rows = connection.execute(
            """SELECT r.retrieval_score, r.document_id, c.chunk_text,
                      r.bm25_score, r.dense_score
               FROM retrieval_events r
               JOIN document_chunks c
                 ON c.query_id=r.query_id AND c.chunk_id=r.chunk_id
               WHERE r.experiment_id=? AND r.query_id=? ORDER BY r.rank""",
            (experiment_id, query_id),
        ).fetchall()
        scores = [row[0] for row in event_rows if row[0] is not None]
        if not scores:
            continue
        shifted = [math.exp(score - max(scores)) for score in scores]
        probabilities = [value / sum(shifted) for value in shifted]
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        question_text = connection.execute(
            "SELECT question_text FROM questions WHERE query_id=?", (query_id,)
        ).fetchone()[0]
        question_token_list = tokens(question_text)
        context_token_list = tokens(" ".join(row[2] for row in event_rows))
        question_tokens = set(question_token_list)
        context_tokens = set(context_token_list)
        overlap = (
            len(question_tokens & context_tokens) / len(question_tokens)
            if question_tokens else 0.0
        )
        score_range = max(scores) - min(scores)
        score_mean = statistics.fmean(scores)
        score_stddev = statistics.pstdev(scores)
        near_top_threshold = max(scores) - (0.10 * score_range)
        unique_documents = len({row[1] for row in event_rows})
        chunk_texts = [row[2] for row in event_rows]
        normalized_chunks = [" ".join(tokens(value)) for value in chunk_texts]
        duplicate_ratio = 1.0 - (len(set(normalized_chunks)) / len(normalized_chunks))
        bm25_scores = [row[3] for row in event_rows]
        dense_scores = [row[4] for row in event_rows]
        hybrid_correlation = (
            correlation(bm25_scores, dense_scores)
            if all(value is not None for value in bm25_scores + dense_scores)
            else None
        )
        connection.execute(
            """INSERT INTO retrieval_features VALUES (
                   ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
               )
               ON CONFLICT(experiment_id, query_id, feature_version) DO UPDATE SET
                   score_max=excluded.score_max, score_min=excluded.score_min,
                   score_mean=excluded.score_mean, score_median=excluded.score_median,
                   score_stddev=excluded.score_stddev,
                   score_range=excluded.score_range,
                   score_coefficient_variation=excluded.score_coefficient_variation,
                   rank1_rank2_gap=excluded.rank1_rank2_gap,
                   rank1_rankk_decay=excluded.rank1_rankk_decay,
                   score_entropy=excluded.score_entropy,
                   normalized_score_entropy=excluded.normalized_score_entropy,
                   near_top_passage_count=excluded.near_top_passage_count,
                   unique_document_count=excluded.unique_document_count,
                   source_diversity_ratio=excluded.source_diversity_ratio,
                   duplicate_context_ratio=excluded.duplicate_context_ratio,
                   context_character_count=excluded.context_character_count,
                   context_token_count=excluded.context_token_count,
                   query_character_count=excluded.query_character_count,
                   query_token_count=excluded.query_token_count,
                   lexical_overlap=excluded.lexical_overlap,
                   mean_pairwise_context_diversity=excluded.mean_pairwise_context_diversity,
                   bm25_dense_score_correlation=excluded.bm25_dense_score_correlation""",
            (
                experiment_id, query_id, FEATURE_VERSION, max(scores), min(scores),
                score_mean, statistics.median(scores), score_stddev, score_range,
                score_stddev / abs(score_mean) if score_mean else None,
                scores[0] - scores[1] if len(scores) > 1 else None,
                scores[0] - scores[-1] if len(scores) > 1 else None,
                entropy, entropy / math.log(len(scores)) if len(scores) > 1 else 0.0,
                sum(score >= near_top_threshold for score in scores),
                unique_documents, unique_documents / len(event_rows), duplicate_ratio,
                sum(len(value) for value in chunk_texts), len(context_token_list),
                len(question_text), len(question_token_list), overlap,
                mean_pairwise_diversity(chunk_texts), hybrid_correlation,
            ),
        )


def build_database(project_root, database_path):
    processed = project_root / "data" / "processed"
    schema = project_root / "src" / "database" / "schema.sql"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        connection.executescript(schema.read_text(encoding="utf-8"))
        cursor = connection.execute(
            "INSERT INTO import_runs(status, database_version, source_directory) "
            "VALUES ('running', ?, ?)",
            (DATABASE_VERSION, str(processed)),
        )
        run_id = cursor.lastrowid
        import_splits(connection, processed)
        trace_ids = {}
        for method, prefix in (("BM25", "bm25"), ("Dense", "dense"), ("Hybrid", "hybrid")):
            trace_ids[method] = import_trace(
                connection, processed / f"{prefix}_100_query_trace.csv", method
            )
            import_summary(
                connection, processed / f"{prefix}_100_query_summary.csv",
                trace_ids[method],
            )
        import_topk_metrics(connection, processed / "retrieval_topk_experiment.csv")
        rag10 = import_responses(
            connection, processed / "rag_10_query_pilot_evaluated.csv",
            "rag_hybrid_k5_pilot10",
        )
        rag50 = import_responses(
            connection, processed / "rag_50_groundedness_validation_evaluated.csv",
            "rag_hybrid_k5_validation50",
        )
        import_human_annotations(
            connection, processed / "rag_10_hallucination_annotation.csv", rag10
        )
        import_human_annotations(
            connection, processed / "rag_50_hallucination_annotation.csv", rag50
        )
        import_automated_labels(
            connection, processed / "rag_10_nli_groundedness_v2.csv", rag10
        )
        import_automated_labels(
            connection, processed / "rag_50_nli_groundedness_validation.csv", rag50
        )
        engineer_features(connection)
        connection.execute(
            "UPDATE import_runs SET status='completed', completed_at=CURRENT_TIMESTAMP, "
            "details=? WHERE import_run_id=?",
            (json.dumps({"feature_version": FEATURE_VERSION}), run_id),
        )
        connection.commit()
    except Exception as error:
        connection.rollback()
        raise RuntimeError(f"Database build failed: {error}") from error
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    database = args.database or args.project_root / "data" / "database" / "experiments.sqlite3"
    build_database(args.project_root.resolve(), database.resolve())
    print(f"Database created: {database.resolve()}")


if __name__ == "__main__":
    main()
