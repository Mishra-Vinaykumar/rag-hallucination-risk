# rag-hallucination-risk

Research framework for predicting hallucination risk in retrieval-augmented generation using retrieval-trace features, comparing BM25, dense and hybrid retrieval with explainable machine learning.

## Current status

Research pilot: implemented retrieval methods, retrieval comparisons, hybrid RAG generation, audited hallucination labels, retrieval-risk features and an end-to-end pilot predictive-modelling workflow. Expansion to a larger, multi-configuration labelled dataset remains future work.

## Project structure

- `src/`: Python retrieval, generation, annotation and evaluation scripts.
- `data/processed/`: Saved pilot results, annotations and experiment split manifests.
- `requirements.txt`: Python dependency versions from the development environment.

Run scripts from the project root. Model and dataset scripts may download Hugging Face resources and require substantial memory. The saved annotations and automated labels are research pilot outputs and require further validation.

The Python virtual environment and downloaded model caches are excluded from version control.

## Pilot ML evaluation

The pilot modelling script trains a dummy baseline, logistic regression, decision
tree, random forest and histogram gradient boosting model. It selects
hyperparameters using repeated cross-validation on the training split, selects
decision thresholds and the champion model on the validation split, and evaluates
the untouched seven-row test split once.

```powershell
python src/ml/train_evaluate_pilot_models.py
python src/ml/test_pilot_model_training.py -v
```

Outputs include model-comparison and prediction CSVs under `data/processed/`,
figures under `docs/figures/pilot_ml/`, a saved selected-model bundle under
`artifacts/pilot_ml/`, and `docs/pilot_ml_training_evaluation.md`. These are pilot
pipeline results and are not sufficient for confirmatory dissertation claims.
