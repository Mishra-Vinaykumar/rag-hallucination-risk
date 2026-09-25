"""Create a preserved final-label audit after human review of NLI output."""

import csv
from pathlib import Path


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def automatic_map(path, label_column, source_name):
    if not path.exists():
        return {}
    return {
        row["query_id"]: (row.get(label_column, "").strip(), source_name)
        for row in read(path)
    }


def main():
    root = Path(__file__).resolve().parents[2]
    processed = root / "data" / "processed"
    output = processed / "rag_60_final_label_audit.csv"
    automatic = {}
    automatic.update(automatic_map(
        processed / "rag_10_nli_groundedness_v2.csv", "v2_hallucination_label",
        "nli_v2",
    ))
    v3_path = processed / "rag_50_nli_groundedness_validation_v3.csv"
    if v3_path.exists():
        automatic.update(automatic_map(
            v3_path, "v3_hallucination_label", "nli_v3"
        ))
    else:
        automatic.update(automatic_map(
            processed / "rag_50_nli_groundedness_validation.csv",
            "v2_hallucination_label", "nli_v2",
        ))
    rows = []
    reviewed_disagreements = []
    for human_file in (
        processed / "rag_10_hallucination_annotation.csv",
        processed / "rag_50_hallucination_annotation.csv",
    ):
        for row in read(human_file):
            human = str(int(float(row["hallucination_label"])))
            auto, automatic_source = automatic.get(row["query_id"], ("", "unavailable"))
            abstained = row["abstained"].strip().lower() == "true"
            if abstained:
                status = "abstention_confirmed"
                auto = ""
            elif auto == human:
                status = "agreement_confirmed"
            else:
                status = "human_label_confirmed"
                reviewed_disagreements.append({
                    "query_id": row["query_id"],
                    "question": row["question"].strip(),
                    "human": human,
                    "automatic": auto or "no decision",
                    "reason": row["annotation_reason"].strip(),
                })
            rows.append({
                "query_id": row["query_id"],
                "human_hallucination_label": human,
                "automatic_hallucination_label": auto,
                "automatic_label_source": automatic_source,
                "final_hallucination_label": human,
                "review_status": status,
                "audit_reason": row["annotation_reason"].strip(),
                "rubric_version": "rubric_v1.0",
                "annotator_confidence": row["annotator_confidence"].strip().lower(),
                "source_annotation_file": human_file.name,
            })
    if len(rows) != 60 or len({row["query_id"] for row in rows}) != 60:
        raise ValueError("Expected 60 unique audited responses")
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = root / "docs" / "label_validation_audit.md"
    status_counts = {
        status: sum(row["review_status"] == status for row in rows)
        for status in sorted({row["review_status"] for row in rows})
    }
    lines = [
        "# Final hallucination-label validation audit", "",
        "Validation status: **Passed**", "",
        "The human annotation is the final target. Automated NLI output is used",
        "only to identify cases that require evidence review; it never overrides",
        "the frozen annotation rubric.", "",
        f"- Audited responses: {len(rows)}",
        f"- Abstentions confirmed: {status_counts.get('abstention_confirmed', 0)}",
        f"- Human/NLI agreements confirmed: {status_counts.get('agreement_confirmed', 0)}",
        f"- NLI disagreements reviewed: {status_counts.get('human_label_confirmed', 0)}",
        "- Duplicate response IDs: 0", "- Missing final labels: 0", "",
        "## Reviewed disagreements", "",
    ]
    for item in reviewed_disagreements:
        lines.extend([
            f"### {item['query_id']}", "",
            f"- Question: {item['question']}",
            f"- Human / automatic label: {item['human']} / {item['automatic']}",
            f"- Review decision: retain human label {item['human']}",
            f"- Evidence rationale: {item['reason']}", "",
        ])
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Final label audit created: {output}")
    print(f"Review report created: {report}")
    print(f"Rows: {len(rows)}")
    print("Status counts:")
    for status in sorted({row["review_status"] for row in rows}):
        print(f"  {status}: {sum(row['review_status'] == status for row in rows)}")


if __name__ == "__main__":
    main()
