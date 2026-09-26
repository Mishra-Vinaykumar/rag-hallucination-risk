"""Train and evaluate exploratory hallucination-risk models on fixed pilot splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rag-pilot-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
IDENTIFIERS = ["query_id", "retrieval_method", "top_k", "pilot_split"]
TARGET = "hallucination_label"
SPLITS = ("train", "validation", "test")
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)


@dataclass(frozen=True)
class Candidate:
    model_name: str
    configuration: str
    estimator: object


def load_dataset(path: Path):
    frame = pd.read_csv(path)
    required = set(IDENTIFIERS + [TARGET])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if frame["query_id"].duplicated().any():
        raise ValueError("Duplicate query IDs are not allowed")
    if set(frame["pilot_split"]) != set(SPLITS):
        raise ValueError("Expected train, validation and test pilot splits")
    if set(frame[TARGET]) != {0, 1}:
        raise ValueError("Expected binary hallucination labels 0 and 1")
    feature_columns = [
        column for column in frame.columns
        if column not in IDENTIFIERS + [TARGET]
    ]
    if not feature_columns:
        raise ValueError("No model features were found")
    for split in SPLITS:
        if set(frame.loc[frame["pilot_split"] == split, TARGET]) != {0, 1}:
            raise ValueError(f"Split {split} must contain both target classes")
    return frame, feature_columns


def scaled_pipeline(classifier):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])


def tree_pipeline(classifier):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("classifier", classifier),
    ])


def candidates():
    items = [Candidate(
        "dummy_prior", "strategy=prior",
        tree_pipeline(DummyClassifier(strategy="prior")),
    )]
    for c in (0.1, 1.0, 10.0):
        items.append(Candidate(
            "logistic_regression", f"C={c:g}",
            scaled_pipeline(LogisticRegression(
                C=c, class_weight="balanced", solver="liblinear",
                random_state=RANDOM_STATE, max_iter=2000,
            )),
        ))
    for depth in (1, 2, 3, None):
        items.append(Candidate(
            "decision_tree", f"max_depth={depth}",
            tree_pipeline(DecisionTreeClassifier(
                max_depth=depth, min_samples_leaf=2, class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ))
    for depth in (2, 3, None):
        items.append(Candidate(
            "random_forest", f"max_depth={depth}",
            tree_pipeline(RandomForestClassifier(
                n_estimators=200, max_depth=depth, min_samples_leaf=2,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1,
            )),
        ))
    for learning_rate in (0.03, 0.10):
        items.append(Candidate(
            "hist_gradient_boosting", f"learning_rate={learning_rate:g}",
            tree_pipeline(HistGradientBoostingClassifier(
                learning_rate=learning_rate, max_iter=150, max_leaf_nodes=7,
                min_samples_leaf=3, l2_regularization=1.0,
                class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ))
    return items


def positive_probability(estimator, features):
    probabilities = estimator.predict_proba(features)
    class_index = int(np.where(estimator.classes_ == 1)[0][0])
    return probabilities[:, class_index]


def safe_auc(metric, labels, probabilities):
    return float(metric(labels, probabilities)) if len(np.unique(labels)) == 2 else math.nan


def metrics(labels, probabilities, threshold):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": safe_auc(roc_auc_score, labels, probabilities),
        "pr_auc": safe_auc(average_precision_score, labels, probabilities),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "tn": int(confusion_matrix(labels, predictions, labels=[0, 1])[0, 0]),
        "fp": int(confusion_matrix(labels, predictions, labels=[0, 1])[0, 1]),
        "fn": int(confusion_matrix(labels, predictions, labels=[0, 1])[1, 0]),
        "tp": int(confusion_matrix(labels, predictions, labels=[0, 1])[1, 1]),
    }


def choose_threshold(labels, probabilities):
    scored = []
    for threshold in THRESHOLDS:
        result = metrics(labels, probabilities, float(threshold))
        scored.append((
            result["balanced_accuracy"], result["f1"], result["recall"],
            -abs(float(threshold) - 0.5), float(threshold),
        ))
    return max(scored)[-1]


def training_cv_score(estimator, features, labels):
    cv = RepeatedStratifiedKFold(
        n_splits=3, n_repeats=5, random_state=RANDOM_STATE
    )
    scores = cross_validate(
        estimator, features, labels, cv=cv,
        scoring={"balanced_accuracy": "balanced_accuracy", "f1": "f1"},
        n_jobs=1, error_score="raise",
    )
    return {
        "cv_balanced_accuracy_mean": float(scores["test_balanced_accuracy"].mean()),
        "cv_balanced_accuracy_std": float(scores["test_balanced_accuracy"].std()),
        "cv_f1_mean": float(scores["test_f1"].mean()),
        "cv_f1_std": float(scores["test_f1"].std()),
    }


def select_configurations(all_candidates, train_x, train_y):
    records = []
    for candidate in all_candidates:
        cv_scores = training_cv_score(candidate.estimator, train_x, train_y)
        records.append({"candidate": candidate, **cv_scores})
    selected = {}
    for model_name in sorted({record["candidate"].model_name for record in records}):
        model_records = [r for r in records if r["candidate"].model_name == model_name]
        selected[model_name] = max(
            model_records,
            key=lambda r: (
                r["cv_balanced_accuracy_mean"], r["cv_f1_mean"],
                -r["cv_balanced_accuracy_std"], r["candidate"].configuration,
            ),
        )
    return selected, records


def bootstrap_interval(labels, probabilities, threshold, metric_name, repeats=1000):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    rng = np.random.default_rng(RANDOM_STATE)
    values = []
    for _ in range(repeats):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        values.append(metrics(sampled_labels, probabilities[indices], threshold)[metric_name])
    if not values:
        return math.nan, math.nan
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def evaluate_models(selected, train_x, train_y, validation_x, validation_y,
                    test_x, test_y):
    fitted, comparison, threshold_rows = {}, [], []
    for model_name, selection in selected.items():
        estimator = clone(selection["candidate"].estimator).fit(train_x, train_y)
        validation_probabilities = positive_probability(estimator, validation_x)
        threshold = choose_threshold(validation_y, validation_probabilities)
        validation_metrics = metrics(validation_y, validation_probabilities, threshold)
        test_probabilities = positive_probability(estimator, test_x)
        test_metrics = metrics(test_y, test_probabilities, threshold)
        bal_low, bal_high = bootstrap_interval(
            test_y, test_probabilities, threshold, "balanced_accuracy"
        )
        f1_low, f1_high = bootstrap_interval(
            test_y, test_probabilities, threshold, "f1"
        )
        fitted[model_name] = {
            "estimator": estimator, "threshold": threshold,
            "validation_probabilities": validation_probabilities,
            "test_probabilities": test_probabilities,
        }
        comparison.append({
            "model": model_name,
            "configuration": selection["candidate"].configuration,
            **{key: value for key, value in selection.items() if key != "candidate"},
            "selected_threshold": threshold,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
            "test_balanced_accuracy_ci_low": bal_low,
            "test_balanced_accuracy_ci_high": bal_high,
            "test_f1_ci_low": f1_low,
            "test_f1_ci_high": f1_high,
        })
        for candidate_threshold in THRESHOLDS:
            row = metrics(validation_y, validation_probabilities, float(candidate_threshold))
            threshold_rows.append({
                "model": model_name, "threshold": float(candidate_threshold), **row,
                "selected": float(candidate_threshold) == threshold,
            })
    champion = max(
        comparison,
        key=lambda row: (
            row["validation_balanced_accuracy"], row["validation_f1"],
            row["validation_pr_auc"], row["cv_balanced_accuracy_mean"],
        ),
    )["model"]
    return fitted, comparison, threshold_rows, champion


def prediction_rows(frame, feature_columns, fitted):
    rows = []
    for model_name, details in fitted.items():
        probabilities = positive_probability(details["estimator"], frame[feature_columns])
        predictions = (probabilities >= details["threshold"]).astype(int)
        for index, row in frame.iterrows():
            rows.append({
                "query_id": row["query_id"], "pilot_split": row["pilot_split"],
                "true_label": int(row[TARGET]), "model": model_name,
                "selected_threshold": details["threshold"],
                "hallucination_probability": float(probabilities[index]),
                "predicted_label": int(predictions[index]),
                "correct": int(predictions[index] == int(row[TARGET])),
            })
    return rows


def feature_importance_rows(champion, fitted, feature_columns,
                            validation_x, validation_y):
    estimator = fitted[champion]["estimator"]
    result = permutation_importance(
        estimator, validation_x, validation_y, scoring="balanced_accuracy",
        n_repeats=50, random_state=RANDOM_STATE, n_jobs=1,
    )
    rows = [{
        "model": champion, "feature": feature,
        "permutation_importance_mean": float(result.importances_mean[index]),
        "permutation_importance_std": float(result.importances_std[index]),
    } for index, feature in enumerate(feature_columns)]
    return sorted(rows, key=lambda row: row["permutation_importance_mean"], reverse=True)


FEATURE_FAMILIES = {
    "score_distribution": [
        "score_max", "score_min", "score_mean", "score_median", "score_stddev",
        "score_range", "score_coefficient_variation", "rank1_rank2_gap",
        "rank1_rankk_decay", "score_entropy", "normalized_score_entropy",
    ],
    "retrieval_diversity": [
        "near_top_passage_count", "unique_document_count", "source_diversity_ratio",
        "duplicate_context_ratio", "mean_pairwise_context_diversity",
    ],
    "query_context": [
        "context_character_count", "context_token_count", "query_character_count",
        "query_token_count", "lexical_overlap", "bm25_dense_score_correlation",
    ],
}


def ablation_rows(train_frame, feature_columns):
    estimator = scaled_pipeline(LogisticRegression(
        C=1.0, class_weight="balanced", solver="liblinear",
        random_state=RANDOM_STATE, max_iter=2000,
    ))
    rows = []
    variants = {"all_features": feature_columns}
    for family, members in FEATURE_FAMILIES.items():
        variants[f"without_{family}"] = [name for name in feature_columns if name not in members]
    for variant, columns in variants.items():
        score = training_cv_score(estimator, train_frame[columns], train_frame[TARGET])
        rows.append({"variant": variant, "feature_count": len(columns), **score})
    baseline = next(row["cv_balanced_accuracy_mean"] for row in rows
                    if row["variant"] == "all_features")
    for row in rows:
        row["balanced_accuracy_change_vs_all"] = (
            row["cv_balanced_accuracy_mean"] - baseline
        )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_model_comparison(comparison, output):
    ordered = sorted(comparison, key=lambda row: row["test_balanced_accuracy"], reverse=True)
    names = [row["model"].replace("_", "\n") for row in ordered]
    validation = [row["validation_balanced_accuracy"] for row in ordered]
    test = [row["test_balanced_accuracy"] for row in ordered]
    positions = np.arange(len(names))
    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(positions - 0.18, validation, 0.36, label="Validation")
    axis.bar(positions + 0.18, test, 0.36, label="Test")
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
    axis.set_xticks(positions, names)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("Pilot hallucination-risk model comparison")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_confusion(labels, probabilities, threshold, model_name, output):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row, column], ha="center", va="center", fontsize=14)
    axis.set_xticks([0, 1], ["Faithful", "Hallucination"])
    axis.set_yticks([0, 1], ["Faithful", "Hallucination"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(f"Test confusion matrix: {model_name.replace('_', ' ')}")
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_roc_pr(test_y, fitted, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for model_name, details in sorted(fitted.items()):
        probabilities = details["test_probabilities"]
        fpr, tpr, _ = roc_curve(test_y, probabilities)
        precision, recall, _ = precision_recall_curve(test_y, probabilities)
        axes[0].plot(fpr, tpr, marker="o", label=f"{model_name} ({roc_auc_score(test_y, probabilities):.2f})")
        axes[1].plot(recall, precision, marker="o", label=f"{model_name} ({average_precision_score(test_y, probabilities):.2f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Test ROC curves")
    axes[1].axhline(float(np.mean(test_y)), color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Test precision-recall curves")
    for axis in axes:
        axis.set_xlim(0, 1); axis.set_ylim(0, 1.05); axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_importance(rows, output):
    top = list(reversed(rows[:12]))
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        [row["feature"] for row in top],
        [row["permutation_importance_mean"] for row in top],
        xerr=[row["permutation_importance_std"] for row in top],
    )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_xlabel("Validation permutation importance (balanced accuracy)")
    axis.set_title("Exploratory feature importance for selected pilot model")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_calibration(test_y, probabilities, model_name, output):
    table = pd.DataFrame({"label": np.asarray(test_y), "probability": probabilities})
    table["bin"] = pd.qcut(table["probability"], q=min(3, len(table)), duplicates="drop")
    grouped = table.groupby("bin", observed=True).agg(
        predicted=("probability", "mean"), observed=("label", "mean"), count=("label", "size")
    )
    fig, axis = plt.subplots(figsize=(5.5, 5))
    axis.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axis.plot(grouped["predicted"], grouped["observed"], "o-", label=model_name.replace("_", " "))
    for _, row in grouped.iterrows():
        axis.annotate(f"n={int(row['count'])}", (row["predicted"], row["observed"]), xytext=(5, 5), textcoords="offset points")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed hallucination rate", title="Exploratory test calibration")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def format_metric(value):
    return "NA" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def write_report(path, frame, feature_columns, comparison, champion,
                 importance, ablations):
    best = next(row for row in comparison if row["model"] == champion)
    class_counts = Counter(frame[TARGET])
    split_counts = Counter(frame["pilot_split"])
    lines = [
        "# Pilot ML training and evaluation", "", "Experiment status: **Passed**", "",
        "## Experimental design", "",
        f"The pilot contains {len(frame)} answered responses: {class_counts[0]} faithful and "
        f"{class_counts[1]} hallucinated. It uses {len(feature_columns)} retrieval-trace features. "
        f"The fixed split contains {split_counts['train']} training, {split_counts['validation']} "
        f"validation and {split_counts['test']} test rows.", "",
        "Hyperparameters were selected using repeated stratified cross-validation on the training "
        "split. Decision thresholds and the champion model were selected using the validation "
        "split. The test split was used once for the final pilot comparison.", "",
        "## Model comparison", "",
        "| Model | Training CV balanced accuracy | Threshold | Validation balanced accuracy | Test balanced accuracy | Test hallucination F1 | Test ROC-AUC | Test PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(comparison, key=lambda item: item["validation_balanced_accuracy"], reverse=True):
        lines.append(
            f"| {row['model'].replace('_', ' ')} | "
            f"{row['cv_balanced_accuracy_mean']:.3f} ± {row['cv_balanced_accuracy_std']:.3f} | "
            f"{row['selected_threshold']:.2f} | {row['validation_balanced_accuracy']:.3f} | "
            f"{row['test_balanced_accuracy']:.3f} | {row['test_f1']:.3f} | "
            f"{row['test_roc_auc']:.3f} | {row['test_pr_auc']:.3f} |"
        )
    lines.extend([
        "", "## Selected pilot model", "",
        f"The validation-selected model is **{champion.replace('_', ' ')}** with configuration "
        f"`{best['configuration']}` and threshold **{best['selected_threshold']:.2f}**. On the "
        f"seven-row test split it achieved balanced accuracy **{best['test_balanced_accuracy']:.3f}** "
        f"(bootstrap 95% interval {format_metric(best['test_balanced_accuracy_ci_low'])}–"
        f"{format_metric(best['test_balanced_accuracy_ci_high'])}) and hallucination-class F1 "
        f"**{best['test_f1']:.3f}** (bootstrap 95% interval {format_metric(best['test_f1_ci_low'])}–"
        f"{format_metric(best['test_f1_ci_high'])}).", "",
        "## Exploratory feature interpretation", "",
        "Permutation importance was calculated on the seven validation responses, so rankings are "
        "unstable and should be treated as hypotheses for the expanded experiment.", "",
        "| Feature | Mean importance | Standard deviation |", "|---|---:|---:|",
    ])
    for row in importance[:10]:
        lines.append(f"| {row['feature']} | {row['permutation_importance_mean']:.3f} | {row['permutation_importance_std']:.3f} |")
    lines.extend([
        "", "## Feature-family ablation", "",
        "Ablation uses repeated training-only cross-validation with logistic regression.", "",
        "| Variant | Features | CV balanced accuracy | Change from all features |",
        "|---|---:|---:|---:|",
    ])
    for row in ablations:
        lines.append(
            f"| {row['variant']} | {row['feature_count']} | "
            f"{row['cv_balanced_accuracy_mean']:.3f} | "
            f"{row['balanced_accuracy_change_vs_all']:+.3f} |"
        )
    lines.extend([
        "", "## Validity limits", "",
        "This is an end-to-end pipeline validation, not a final effectiveness result. The test set "
        "has only seven responses, confidence intervals are consequently wide, and all rows use "
        "Hybrid Top-5 retrieval. The model comparison, calibration curve and feature ranking must "
        "not be used as confirmatory dissertation evidence until the labelled dataset is expanded "
        "across retrieval methods and Top-K settings with question-grouped evaluation.", "",
        "The dummy model can rank examples through its constant probability only at chance level. "
        "Threshold optimisation on seven validation rows is highly discrete and may not generalise. "
        "The frozen test split must remain unchanged when the pilot code is rerun.", "",
        "## Reproduction", "",
        "Run from the repository root:", "",
        "```powershell", "python src/ml/train_evaluate_pilot_models.py", "```", "",
        "The script validates split integrity, uses random seed 42, writes all tabular results and "
        "figures, and saves the selected estimator with its threshold and feature list.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(input_path: Path, output_dir: Path, figure_dir: Path,
        artifact_dir: Path, report_path: Path):
    frame, feature_columns = load_dataset(input_path)
    subsets = {name: frame.loc[frame["pilot_split"] == name].reset_index(drop=True)
               for name in SPLITS}
    selected, candidate_scores = select_configurations(
        candidates(), subsets["train"][feature_columns], subsets["train"][TARGET]
    )
    fitted, comparison, threshold_rows, champion = evaluate_models(
        selected,
        subsets["train"][feature_columns], subsets["train"][TARGET],
        subsets["validation"][feature_columns], subsets["validation"][TARGET],
        subsets["test"][feature_columns], subsets["test"][TARGET],
    )
    predictions = prediction_rows(frame, feature_columns, fitted)
    importance = feature_importance_rows(
        champion, fitted, feature_columns,
        subsets["validation"][feature_columns], subsets["validation"][TARGET],
    )
    ablations = ablation_rows(subsets["train"], feature_columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pilot_ml_model_comparison.csv", comparison)
    write_csv(output_dir / "pilot_ml_predictions.csv", predictions)
    write_csv(output_dir / "pilot_ml_threshold_analysis.csv", threshold_rows)
    write_csv(output_dir / "pilot_ml_feature_importance.csv", importance)
    write_csv(output_dir / "pilot_ml_feature_ablation.csv", ablations)
    write_csv(output_dir / "pilot_ml_candidate_cv_results.csv", [{
        "model": row["candidate"].model_name,
        "configuration": row["candidate"].configuration,
        **{key: value for key, value in row.items() if key != "candidate"},
        "selected_configuration": selected[row["candidate"].model_name]["candidate"].configuration == row["candidate"].configuration,
    } for row in candidate_scores])

    plot_model_comparison(comparison, figure_dir / "pilot_model_comparison.png")
    plot_confusion(
        subsets["test"][TARGET], fitted[champion]["test_probabilities"],
        fitted[champion]["threshold"], champion,
        figure_dir / "pilot_selected_model_confusion_matrix.png",
    )
    plot_roc_pr(subsets["test"][TARGET], fitted, figure_dir / "pilot_test_roc_pr_curves.png")
    plot_importance(importance, figure_dir / "pilot_feature_importance.png")
    plot_calibration(
        subsets["test"][TARGET], fitted[champion]["test_probabilities"], champion,
        figure_dir / "pilot_selected_model_calibration.png",
    )

    bundle = {
        "estimator": fitted[champion]["estimator"],
        "threshold": fitted[champion]["threshold"],
        "feature_columns": feature_columns,
        "model_name": champion,
        "random_state": RANDOM_STATE,
        "training_query_ids": subsets["train"]["query_id"].tolist(),
        "validation_query_ids": subsets["validation"]["query_id"].tolist(),
    }
    joblib.dump(bundle, artifact_dir / "pilot_selected_model.joblib")
    (artifact_dir / "pilot_selected_model_metadata.json").write_text(
        json.dumps({key: value for key, value in bundle.items() if key != "estimator"}, indent=2),
        encoding="utf-8",
    )
    write_report(report_path, frame, feature_columns, comparison, champion, importance, ablations)
    return {"champion": champion, "comparison": comparison, "rows": len(frame)}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=root / "data/processed/pilot_ml_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "data/processed")
    parser.add_argument("--figure-dir", type=Path, default=root / "docs/figures/pilot_ml")
    parser.add_argument("--artifact-dir", type=Path, default=root / "artifacts/pilot_ml")
    parser.add_argument("--report", type=Path, default=root / "docs/pilot_ml_training_evaluation.md")
    args = parser.parse_args()
    result = run(args.input, args.output_dir, args.figure_dir, args.artifact_dir, args.report)
    champion = next(row for row in result["comparison"] if row["model"] == result["champion"])
    print("Pilot ML training and evaluation complete")
    print(f"Rows: {result['rows']}")
    print(f"Selected model: {result['champion']}")
    print(f"Selected threshold: {champion['selected_threshold']:.2f}")
    print(f"Test balanced accuracy: {champion['test_balanced_accuracy']:.3f}")
    print(f"Test hallucination F1: {champion['test_f1']:.3f}")


if __name__ == "__main__":
    main()
