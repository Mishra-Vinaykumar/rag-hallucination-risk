# rag-hallucination-risk

Research framework for predicting hallucination risk in retrieval-augmented generation using retrieval-trace features, comparing BM25, dense and hybrid retrieval with explainable machine learning.

## Current status

Research pilot: implemented retrieval methods, retrieval comparisons, hybrid RAG generation, annotation workflows and NLI groundedness evaluation. Predictive risk modelling and full-scale evaluation remain future work.

## Project structure

- `src/`: Python retrieval, generation, annotation and evaluation scripts.
- `data/processed/`: Saved pilot results, annotations and experiment split manifests.
- `requirements.txt`: Python dependency versions from the development environment.

Run scripts from the project root. Model and dataset scripts may download Hugging Face resources and require substantial memory. The saved annotations and automated labels are research pilot outputs and require further validation.

The Python virtual environment and downloaded model caches are excluded from version control.
