# Experimental database

This directory contains the reproducible SQLite layer for the dissertation.
The source CSV files remain the immutable inputs; the database can be rebuilt
from them at any time.

## Build and validate

Run these commands from the project root:

```powershell
python src/database/build_database.py
python src/database/validate_database.py
python src/database/test_database.py -v
```

The build creates `data/database/experiments.sqlite3`. It deletes and rebuilds
only that generated database file. It never edits the processed CSV files.

## Design decisions

- HotpotQA document and chunk identifiers are scoped by `query_id` because the
  current experiments construct a separate distractor collection per question.
- Experiment keys preserve the retrieval method, Top-K and pipeline stage.
- Human and automated labels are separate versioned tables. Abstentions remain
  responses but do not receive an automatic hallucination label.
- Retrieval features contain only signals available from the question and
  retrieval trace. Gold supporting-fact flags and answer-evaluation fields are
  deliberately excluded from the feature table to prevent target leakage.
- Imports use unique keys and upserts, while the build itself starts from a new
  generated database for deterministic, auditable results.

Use `research_queries.sql` for example dissertation analyses. The ERD is in
`docs/database_erd.md`.
