# Pilot ML training and evaluation

Experiment status: **Passed**

## Experimental design

The pilot contains 34 answered responses: 19 faithful and 15 hallucinated. It uses 22 retrieval-trace features. The fixed split contains 20 training, 7 validation and 7 test rows.

Hyperparameters were selected using repeated stratified cross-validation on the training split. Decision thresholds and the champion model were selected using the validation split. The test split was used once for the final pilot comparison.

## Model comparison

| Model | Training CV balanced accuracy | Threshold | Validation balanced accuracy | Test balanced accuracy | Test hallucination F1 | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| decision tree | 0.481 ± 0.189 | 0.50 | 0.875 | 0.750 | 0.750 | 0.500 | 0.600 |
| hist gradient boosting | 0.492 ± 0.148 | 0.50 | 0.875 | 0.500 | 0.600 | 0.667 | 0.589 |
| random forest | 0.431 ± 0.169 | 0.45 | 0.875 | 0.625 | 0.667 | 0.667 | 0.700 |
| logistic regression | 0.628 ± 0.161 | 0.80 | 0.667 | 0.292 | 0.286 | 0.417 | 0.610 |
| dummy prior | 0.500 ± 0.000 | 0.45 | 0.500 | 0.500 | 0.600 | 0.500 | 0.429 |

## Selected pilot model

The validation-selected model is **decision tree** with configuration `max_depth=None` and threshold **0.50**. On the seven-row test split it achieved balanced accuracy **0.750** (bootstrap 95% interval 0.500–1.000) and hallucination-class F1 **0.750** (bootstrap 95% interval 0.400–1.000).

## Exploratory feature interpretation

Permutation importance was calculated on the seven validation responses, so rankings are unstable and should be treated as hypotheses for the expanded experiment.

| Feature | Mean importance | Standard deviation |
|---|---:|---:|
| score_median | 0.362 | 0.160 |
| normalized_score_entropy | 0.029 | 0.106 |
| score_max | 0.014 | 0.122 |
| score_min | 0.000 | 0.000 |
| score_mean | 0.000 | 0.000 |
| score_stddev | 0.000 | 0.000 |
| score_range | 0.000 | 0.000 |
| score_coefficient_variation | 0.000 | 0.000 |
| rank1_rank2_gap | 0.000 | 0.000 |
| rank1_rankk_decay | 0.000 | 0.000 |

## Feature-family ablation

Ablation uses repeated training-only cross-validation with logistic regression.

| Variant | Features | CV balanced accuracy | Change from all features |
|---|---:|---:|---:|
| all_features | 22 | 0.583 | +0.000 |
| without_score_distribution | 11 | 0.425 | -0.158 |
| without_retrieval_diversity | 17 | 0.519 | -0.064 |
| without_query_context | 16 | 0.644 | +0.061 |

## Validity limits

This is an end-to-end pipeline validation, not a final effectiveness result. The test set has only seven responses, confidence intervals are consequently wide, and all rows use Hybrid Top-5 retrieval. The model comparison, calibration curve and feature ranking must not be used as confirmatory dissertation evidence until the labelled dataset is expanded across retrieval methods and Top-K settings with question-grouped evaluation.

The dummy model can rank examples through its constant probability only at chance level. Threshold optimisation on seven validation rows is highly discrete and may not generalise. The frozen test split must remain unchanged when the pilot code is rerun.

## Reproduction

Run from the repository root:

```powershell
python src/ml/train_evaluate_pilot_models.py
```

The script validates split integrity, uses random seed 42, writes all tabular results and figures, and saves the selected estimator with its threshold and feature list.
