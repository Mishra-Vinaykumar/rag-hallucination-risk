"""Validate referential integrity, expected coverage and analytical readiness."""

import argparse
import sqlite3
from pathlib import Path


MINIMUM_COUNTS = {
    "questions": 3100,
    "experiments": 14,
    "retrieval_events": 1500,
    "retrieval_metrics": 1200,
    "responses": 60,
    "human_annotations": 60,
    "automated_labels": 60,
    "retrieval_features": 300,
}


def validate(database_path):
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if quick_check != "ok":
            raise AssertionError(f"SQLite quick_check failed: {quick_check}")
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            raise AssertionError(f"Foreign-key violations: {foreign_keys[:10]}")

        counts = {}
        for table, minimum in MINIMUM_COUNTS.items():
            count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            counts[table] = count
            if count < minimum:
                raise AssertionError(f"{table}: expected at least {minimum}, found {count}")

        duplicate_ranks = connection.execute(
            """SELECT COUNT(*) FROM (
                   SELECT experiment_id, query_id, rank, COUNT(*) AS n
                   FROM retrieval_events GROUP BY experiment_id, query_id, rank
                   HAVING n > 1
               )"""
        ).fetchone()[0]
        if duplicate_ranks:
            raise AssertionError(f"Duplicate retrieval ranks: {duplicate_ranks}")

        missing_human = connection.execute(
            """SELECT COUNT(*) FROM responses r
               LEFT JOIN human_annotations h ON h.response_id=r.response_id
               WHERE h.response_id IS NULL"""
        ).fetchone()[0]
        if missing_human:
            raise AssertionError(f"Responses without human annotations: {missing_human}")

        answered = connection.execute(
            "SELECT COUNT(*) FROM responses WHERE abstained=0"
        ).fetchone()[0]
        hallucinations = connection.execute(
            """SELECT COUNT(*) FROM human_annotations h
               JOIN responses r ON r.response_id=h.response_id
               WHERE r.abstained=0 AND h.hallucination_label=1"""
        ).fetchone()[0]

        print("DATABASE VALIDATION PASSED")
        for table, count in counts.items():
            print(f"{table}: {count}")
        print(f"answered_responses: {answered}")
        print(f"human_hallucinations_among_answers: {hallucinations}")
    finally:
        connection.close()
    return counts


def main():
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database", type=Path,
        default=default_root / "data" / "database" / "experiments.sqlite3",
    )
    args = parser.parse_args()
    validate(args.database.resolve())


if __name__ == "__main__":
    main()
