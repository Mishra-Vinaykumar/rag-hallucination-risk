# Database build report

Build status: **Passed**

The SQLite database was rebuilt from the processed CSV sources and validated
with `PRAGMA quick_check`, `PRAGMA foreign_key_check`, record-count assertions,
duplicate-rank checks and integration tests.

## Imported records

| Entity | Records |
|---|---:|
| Questions | 3,100 |
| Experiment configurations | 14 |
| Retrieval events | 1,500 |
| Retrieval metric rows | 1,200 |
| Generated responses | 60 |
| Human annotations | 60 |
| Automated labels | 60 |
| Retrieval-feature rows | 300 |

Of the 60 generated responses, 34 contain answers and 26 are abstentions.
Fifteen of the answered responses have a human hallucination label of 1.

## Validation outcome

- SQLite quick check: passed
- Foreign-key check: passed with zero violations
- Duplicate experiment/query/rank records: zero
- Responses missing human annotations: zero
- Database integration tests: 2 passed
- Analysis views: verified
- Example research queries: parsed successfully

The database is generated at `data/database/experiments.sqlite3`. It can be
recreated from the CSV sources by running `src/database/build_database.py`.
