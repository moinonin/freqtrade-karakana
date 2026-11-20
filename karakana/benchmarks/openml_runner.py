"""OpenML benchmark runner for karakana."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openml
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import build_benchmark_record, summarize_benchmark_records, write_benchmark_outputs

_PAIRWISE_METRICS: tuple[tuple[str, bool], ...] = (
    ("success_rate", True),
    ("expectancy", True),
    ("ranking_score", False),
    ("alpha_s_sil", False),
    ("normalized_mse_cofactors", False),
    ("distributional_structural_term", False),
)

_PREFERRED_STRUCTURAL_METRICS: tuple[str, ...] = (
    "ranking_score",
    "alpha_s_sil",
    "normalized_mse_cofactors",
)


@dataclass
class BenchmarkModel:
    name: str
    estimator: Any


def _parse_models(models_value: str | None, fallback_model: str) -> list[str]:
    raw = str(models_value or "").strip()
    if not raw:
        raw = str(fallback_model or "").strip()
    models = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not models:
        raise ValueError("At least one model must be provided")
    deduped: list[str] = []
    for model in models:
        if model not in deduped:
            deduped.append(model)
    return deduped


def _build_model(model_name: str) -> BenchmarkModel:
    model_name = model_name.lower().strip()
    if model_name == "dummy":
        return BenchmarkModel(
            name="dummy",
            estimator=DummyClassifier(strategy="most_frequent"),
        )
    if model_name == "logreg":
        return BenchmarkModel(
            name="logreg",
            estimator=LogisticRegression(max_iter=1000, solver="lbfgs"),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _preprocess_and_fit(X, y, model: BenchmarkModel, seed: int):
    categorical_cols = list(X.select_dtypes(include=["object", "category", "bool"]).columns)
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model.estimator),
        ]
    )

    y_series = y.iloc[:, 0] if hasattr(y, "iloc") and len(getattr(y, "columns", [])) == 1 else y
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_series,
        test_size=0.25,
        random_state=seed,
        stratify=y_series,
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(X_test)
        except Exception:
            probs = None
    return X_train, X_test, y_train, y_test, preds, probs


def _inverse_frequency_weights(labels) -> dict[Any, float]:
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    if len(values) == 0:
        return {}
    total = float(np.sum(counts))
    raw = {value: total / float(count) for value, count in zip(values, counts)}
    mean_weight = float(np.mean(list(raw.values()))) if raw else 1.0
    return {value: float(weight / max(mean_weight, 1e-12)) for value, weight in raw.items()}


def _prediction_confidences(y_pred, probs: np.ndarray | None) -> np.ndarray:
    count = len(y_pred)
    if probs is None:
        return np.ones(count, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        return np.clip(probs, 0.0, 1.0)
    return np.clip(np.max(probs, axis=1), 0.0, 1.0)


def _classification_outcomes(
    y_true,
    y_pred,
    probs: np.ndarray | None,
    positive_gain: float,
    failure_cost: float,
    cost_mode: str,
    confidence_mode: str,
    confidence_power: float,
    confidence_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    positives: list[float] = []
    negatives: list[float] = []
    cost_weights = _inverse_frequency_weights(y_true) if cost_mode == "balanced" else {}
    confidences = _prediction_confidences(y_pred, probs)
    for actual, predicted, confidence in zip(y_true, y_pred, confidences):
        class_weight = float(cost_weights.get(actual, 1.0))
        confidence_weight = 1.0
        if confidence_mode == "predicted_max":
            confidence_weight = float(max(confidence_floor, confidence) ** confidence_power)
        if actual == predicted:
            positives.append(float(positive_gain) * class_weight * confidence_weight)
        else:
            negatives.append(-float(failure_cost) * class_weight * confidence_weight)
    metadata = {
        "cost_mode": cost_mode,
        "confidence_mode": confidence_mode,
        "confidence_power": float(confidence_power),
        "confidence_floor": float(confidence_floor),
        "class_cost_weights": {
            str(key): float(value) for key, value in sorted(cost_weights.items(), key=lambda item: str(item[0]))
        },
        "mean_confidence": float(np.mean(confidences)) if len(confidences) else 1.0,
    }
    return np.asarray(positives, dtype=float), np.asarray(negatives, dtype=float), metadata


def _run_task(
    task_id: int,
    model_name: str,
    positive_gain: float,
    failure_cost: float,
    cost_mode: str,
    confidence_mode: str,
    confidence_power: float,
    confidence_floor: float,
    seed: int,
    ranking_profile: str = "conservative",
) -> dict[str, Any]:
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y(dataset_format="dataframe")
    dataset = task.get_dataset()
    model = _build_model(model_name)
    _, X_test, y_train, y_test, preds, probs = _preprocess_and_fit(X, y, model, seed=seed)
    positives, negatives, payoff_metadata = _classification_outcomes(
        y_test,
        preds,
        probs,
        positive_gain,
        failure_cost,
        cost_mode,
        confidence_mode,
        confidence_power,
        confidence_floor,
    )
    class_labels = np.unique(np.asarray(y_test))
    accuracy = float(accuracy_score(y_test, preds))
    balanced_accuracy = float(balanced_accuracy_score(y_test, preds))
    return build_benchmark_record(
        benchmark="openml",
        benchmark_id=f"{task_id}:{model.name}",
        subject=f"{dataset.name}:{model.name}",
        positives=positives,
        negatives=negatives,
        metadata={
            "task_id": task_id,
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.name,
            "model": model.name,
            "rows_test": int(len(X_test)),
            "rows_train": int(len(y_train)),
            "class_count": int(len(class_labels)),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "positive_gain": positive_gain,
            "failure_cost": failure_cost,
            "seed": seed,
            **payoff_metadata,
        },
        ranking_profile=ranking_profile,
    )


def _build_pairwise_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        metadata = record.get("metadata") or {}
        dataset_name = str(metadata.get("dataset_name") or "")
        model_name = str(metadata.get("model") or "")
        if not dataset_name or not model_name:
            continue
        grouped.setdefault(dataset_name, {})[model_name] = record

    model_names = sorted(
        {
            str((record.get("metadata") or {}).get("model") or "")
            for record in records
            if str((record.get("metadata") or {}).get("model") or "")
        }
    )
    comparisons: list[dict[str, Any]] = []
    for idx, model_a in enumerate(model_names):
        for model_b in model_names[idx + 1 :]:
            metric_summary = {
                metric_name: {
                    "higher_is_better": higher_is_better,
                    "wins": {model_a: 0, model_b: 0},
                    "ties": 0,
                    "total": 0,
                }
                for metric_name, higher_is_better in _PAIRWISE_METRICS
            }
            details: list[dict[str, Any]] = []
            for dataset_name, dataset_records in sorted(grouped.items()):
                if model_a not in dataset_records or model_b not in dataset_records:
                    continue
                record_a = dataset_records[model_a]
                record_b = dataset_records[model_b]
                detail_metrics: dict[str, Any] = {}
                for metric_name, higher_is_better in _PAIRWISE_METRICS:
                    val_a = (record_a.get("summary") or {}).get(metric_name)
                    val_b = (record_b.get("summary") or {}).get(metric_name)
                    if val_a is None or val_b is None:
                        continue
                    value_a = float(val_a)
                    value_b = float(val_b)
                    winner = "tie"
                    if abs(value_a - value_b) < 1e-12:
                        metric_summary[metric_name]["ties"] += 1
                    elif (value_a > value_b) if higher_is_better else (value_a < value_b):
                        winner = model_a
                        metric_summary[metric_name]["wins"][model_a] += 1
                    else:
                        winner = model_b
                        metric_summary[metric_name]["wins"][model_b] += 1
                    metric_summary[metric_name]["total"] += 1
                    detail_metrics[metric_name] = {
                        model_a: value_a,
                        model_b: value_b,
                        "winner": winner,
                    }
                details.append(
                    {
                        "dataset": dataset_name,
                        "metrics": detail_metrics,
                    }
                )
            if not details:
                continue
            comparisons.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "task_count": len(details),
                    "metrics": metric_summary,
                    "details": details,
                }
            )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark karakana on OpenML classification tasks.")
    parser.add_argument("--suite", default="OpenML-CC18", help="OpenML benchmark suite alias")
    parser.add_argument("--task-limit", type=int, default=10, help="Maximum number of OpenML tasks to evaluate")
    parser.add_argument(
        "--model",
        choices=["dummy", "logreg"],
        default="logreg",
        help="Single baseline model to benchmark; kept for backward compatibility",
    )
    parser.add_argument(
        "--models", default="", help="Comma-separated baseline models to benchmark on each task, e.g. dummy,logreg"
    )
    parser.add_argument("--positive-gain", type=float, default=1.0, help="Magnitude assigned to a correct prediction")
    parser.add_argument("--failure-cost", type=float, default=1.0, help="Magnitude assigned to an incorrect prediction")
    parser.add_argument(
        "--cost-mode",
        choices=["symmetric", "balanced"],
        default="balanced",
        help="How to weight class outcomes: balanced is the recommended default because it upweights rarer true classes",
    )
    parser.add_argument(
        "--confidence-mode",
        choices=["none", "predicted_max"],
        default="predicted_max",
        help="How to scale magnitudes by model confidence; predicted_max is the recommended default",
    )
    parser.add_argument(
        "--confidence-power",
        type=float,
        default=0.5,
        help="Exponent applied to confidence weighting when enabled; 0.5 is the recommended OpenML default",
    )
    parser.add_argument(
        "--confidence-floor",
        type=float,
        default=0.05,
        help="Minimum confidence weight when confidence weighting is enabled",
    )
    parser.add_argument("--ranking-profile", default="balanced", help="karakana ranking profile")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split and models")
    parser.add_argument("--output-dir", default="benchmark_runs/openml", help="Directory for benchmark outputs")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional local OpenML cache directory; defaults to <output-dir>/.openml_cache",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / ".openml_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(cache_dir))
    model_names = _parse_models(args.models, args.model)
    for model_name in model_names:
        _build_model(model_name)

    try:
        suite = openml.study.get_suite(args.suite)
    except Exception as exc:
        raise SystemExit(
            "OpenML benchmark fetch failed. "
            "Check internet/DNS access and retry. "
            f"cache_dir={cache_dir} error={type(exc).__name__}: {exc}"
        ) from exc
    task_ids = list(suite.tasks)[: args.task_limit]
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task_id in task_ids:
        for model_name in model_names:
            try:
                records.append(
                    _run_task(
                        task_id=task_id,
                        model_name=model_name,
                        positive_gain=args.positive_gain,
                        failure_cost=args.failure_cost,
                        cost_mode=args.cost_mode,
                        confidence_mode=args.confidence_mode,
                        confidence_power=args.confidence_power,
                        confidence_floor=args.confidence_floor,
                        seed=args.seed,
                        ranking_profile=args.ranking_profile,
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "task_id": task_id,
                        "model": model_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    summary = summarize_benchmark_records(records)
    summary["suite"] = args.suite
    summary["cache_dir"] = str(cache_dir.resolve())
    summary["models"] = model_names
    summary["cost_mode"] = args.cost_mode
    summary["confidence_mode"] = args.confidence_mode
    summary["confidence_power"] = float(args.confidence_power)
    summary["confidence_floor"] = float(args.confidence_floor)
    summary["ranking_profile"] = args.ranking_profile
    summary["preferred_structural_metrics"] = list(_PREFERRED_STRUCTURAL_METRICS)
    summary["secondary_structural_metric"] = "normalized_cofactors_norm_diff_abs"
    summary["recommended_interpretation"] = (
        "Use normalized_mse_cofactors and relative_cofactors_fro_error as the "
        "primary structural comparison metrics for OpenML runs. Treat "
        "normalized_cofactors_norm_diff_abs as a secondary tie-breaker."
    )
    summary["requested_task_limit"] = int(args.task_limit)
    summary["completed_task_count"] = len(records)
    summary["failed_task_count"] = len(failures)
    if len(model_names) > 1:
        summary["pairwise_comparisons"] = _build_pairwise_summary(records)
    if failures:
        summary["failures"] = failures
    outputs = write_benchmark_outputs(args.output_dir, records, summary)
    print(json.dumps({"summary": summary, "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
