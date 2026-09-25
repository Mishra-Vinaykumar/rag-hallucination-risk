"""Create the leakage-controlled pilot ML dataset from SQLite and final labels."""

import argparse
import csv
import hashlib
import sqlite3
from collections import Counter
from pathlib import Path


FEATURE_COLUMNS = [
    "score_max", "score_min", "score_mean", "score_median", "score_stddev",
    "score_range", "score_coefficient_variation", "rank1_rank2_gap",
    "rank1_rankk_decay", "score_entropy", "normalized_score_entropy",
    "near_top_passage_count", "unique_document_count", "source_diversity_ratio",
    "duplicate_context_ratio", "context_character_count", "context_token_count",
    "query_character_count", "query_token_count", "lexical_overlap",
    "mean_pairwise_context_diversity", "bm25_dense_score_correlation",
]
IDENTIFIER_COLUMNS = ["query_id", "retrieval_method", "top_k", "pilot_split"]
TARGET_COLUMN = "hallucination_label"
BANNED_LEAKAGE_NAMES = {
    "expected_answer", "is_supporting", "gold_chunks_retrieved",
    "precision_at_k", "recall_at_k", "hit_at_k", "answer_f1", "exact_match",
    "nli_entailment", "nli_neutral", "nli_contradiction",
    "automatic_hallucination_label",
}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def stratified_splits(rows, seed="rag-hallucination-pilot-v1"):
    assignments = {}
    labels = sorted({int(row[TARGET_COLUMN]) for row in rows})
    for label in labels:
        members = [row for row in rows if int(row[TARGET_COLUMN]) == label]
        members.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['query_id']}".encode("utf-8")
            ).hexdigest()
        )
        total = len(members)
        train_end = max(1, round(total * 0.60))
        validation_end = min(total - 1, train_end + max(1, round(total * 0.20)))
        for index, row in enumerate(members):
            split = "train" if index < train_end else (
                "validation" if index < validation_end else "test"
            )
            assignments[row["query_id"]] = split
    return assignments


def load_final_audits(connection, audit_path):
    audits = read_csv(audit_path)
    required = {
        "query_id", "human_hallucination_label", "automatic_hallucination_label",
        "final_hallucination_label", "review_status", "audit_reason", "rubric_version",
    }
    if not audits or required - set(audits[0]):
        raise ValueError(f"Audit file is missing columns: {sorted(required - set(audits[0] if audits else []))}")
    for row in audits:
        response = connection.execute(
            "SELECT response_id FROM responses WHERE query_id=?", (row["query_id"],)
        ).fetchall()
        if len(response) != 1:
            raise ValueError(f"Expected one response for {row['query_id']}, found {len(response)}")
        automatic = row["automatic_hallucination_label"].strip()
        connection.execute(
            """INSERT INTO final_label_audits(
                   response_id, human_hallucination_label,
                   automatic_hallucination_label, final_hallucination_label,
                   review_status, audit_reason, rubric_version, source_file
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(response_id) DO UPDATE SET
                   human_hallucination_label=excluded.human_hallucination_label,
                   automatic_hallucination_label=excluded.automatic_hallucination_label,
                   final_hallucination_label=excluded.final_hallucination_label,
                   review_status=excluded.review_status,
                   audit_reason=excluded.audit_reason,
                   rubric_version=excluded.rubric_version,
                   source_file=excluded.source_file""",
            (
                response[0][0], int(row["human_hallucination_label"]),
                int(automatic) if automatic else None,
                int(row["final_hallucination_label"]), row["review_status"],
                row["audit_reason"], row["rubric_version"], audit_path.name,
            ),
        )
    return audits


def eligible_rows(connection):
    columns = ", ".join(f"f.{column}" for column in FEATURE_COLUMNS)
    query = f"""
        SELECT r.response_id, r.query_id, e.experiment_id, e.retrieval_method,
               e.top_k, f.feature_version, {columns},
               a.final_hallucination_label AS hallucination_label
        FROM responses r
        JOIN final_label_audits a ON a.response_id=r.response_id
        JOIN experiments e ON e.experiment_key='retrieval_trace_hybrid_k5'
        JOIN retrieval_features f
          ON f.experiment_id=e.experiment_id AND f.query_id=r.query_id
        WHERE r.abstained=0 AND f.feature_version='retrieval_trace_v2'
        ORDER BY r.query_id
    """
    names = [description[0] for description in connection.execute(query).description]
    return [dict(zip(names, row)) for row in connection.execute(query).fetchall()]


def validate_rows(rows, assignments):
    if not rows:
        raise ValueError("No eligible answered responses matched retrieval features")
    ids = [row["query_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate query IDs in pilot dataset")
    for row in rows:
        missing = [name for name in FEATURE_COLUMNS if row[name] is None]
        allowed_missing = {"score_coefficient_variation", "bm25_dense_score_correlation"}
        if set(missing) - allowed_missing:
            raise ValueError(f"Missing features for {row['query_id']}: {missing}")
    exported = set(IDENTIFIER_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN])
    leakage = exported & BANNED_LEAKAGE_NAMES
    if leakage:
        raise ValueError(f"Leakage columns present: {sorted(leakage)}")
    split_sets = {
        name: {query_id for query_id, split in assignments.items() if split == name}
        for name in ("train", "validation", "test")
    }
    if any(split_sets[left] & split_sets[right]
           for left, right in (("train", "validation"), ("train", "test"),
                               ("validation", "test"))):
        raise ValueError("Query leakage detected between pilot splits")
    for split, query_ids in split_sets.items():
        labels = {int(row[TARGET_COLUMN]) for row in rows if row["query_id"] in query_ids}
        if labels != {0, 1}:
            raise ValueError(f"Split {split} does not contain both classes")


def populate_table(connection, rows, assignments):
    connection.execute("DELETE FROM pilot_ml_dataset")
    columns = [
        "response_id", "query_id", "retrieval_experiment_id", "feature_version",
        "pilot_split", "retrieval_method", "top_k", *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]
    placeholders = ",".join("?" for _ in columns)
    sql = f"INSERT INTO pilot_ml_dataset ({','.join(columns)}) VALUES ({placeholders})"
    for row in rows:
        values = [
            row["response_id"], row["query_id"], row["experiment_id"],
            row["feature_version"], assignments[row["query_id"]],
            row["retrieval_method"], row["top_k"],
            *[row[name] for name in FEATURE_COLUMNS], int(row[TARGET_COLUMN]),
        ]
        connection.execute(sql, values)


def export_csv(rows, assignments, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = IDENTIFIER_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            output = {name: row[name] for name in columns if name != "pilot_split"}
            output["pilot_split"] = assignments[row["query_id"]]
            writer.writerow(output)


def write_report(rows, assignments, output_path):
    split_counts = Counter(assignments.values())
    label_counts = Counter(int(row[TARGET_COLUMN]) for row in rows)
    lines = [
        "# Pilot ML dataset validation", "", "Validation status: **Passed**", "",
        f"- Eligible answered responses: {len(rows)}",
        f"- Faithful class (0): {label_counts[0]}",
        f"- Hallucination class (1): {label_counts[1]}",
        f"- Train rows: {split_counts['train']}",
        f"- Validation rows: {split_counts['validation']}",
        f"- Test rows: {split_counts['test']}",
        "- Duplicate query IDs: 0", "- Missing required features: 0",
        "- Query overlap between splits: 0", "- Prohibited leakage columns: 0", "",
        "This dataset is suitable for validating the ML pipeline only. Its 34 rows",
        "come from Hybrid Top-5 pilot responses and are not sufficient for final",
        "dissertation claims or comparison of retrieval methods.", "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build(database, audit_path, csv_output, report_output):
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        load_final_audits(connection, audit_path)
        rows = eligible_rows(connection)
        assignments = stratified_splits(rows)
        validate_rows(rows, assignments)
        populate_table(connection, rows, assignments)
        export_csv(rows, assignments, csv_output)
        write_report(rows, assignments, report_output)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return rows, assignments


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path,
                        default=root / "data/database/experiments.sqlite3")
    parser.add_argument("--audit", type=Path,
                        default=root / "data/processed/rag_60_final_label_audit.csv")
    parser.add_argument("--output", type=Path,
                        default=root / "data/processed/pilot_ml_dataset.csv")
    parser.add_argument("--report", type=Path,
                        default=root / "docs/pilot_ml_dataset_validation.md")
    args = parser.parse_args()
    rows, assignments = build(args.database, args.audit, args.output, args.report)
    print(f"Pilot ML dataset created: {args.output}")
    print(f"Rows: {len(rows)}")
    print(f"Splits: {dict(Counter(assignments.values()))}")


if __name__ == "__main__":
    main()
