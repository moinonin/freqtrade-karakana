"""Shared output helpers for benchmark runners."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from karakana.metrics import evaluate_distributional_geometry


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _metric_or_diagnostic(metrics: dict[str, Any], key: str) -> Any:
    if key in metrics:
        return metrics.get(key)
    diagnostics = metrics.get("diagnostics") or {}
    if isinstance(diagnostics, dict):
        return diagnostics.get(key)
    return None


def build_benchmark_record(
    *,
    benchmark: str,
    benchmark_id: str,
    subject: str,
    positives: np.ndarray,
    negatives: np.ndarray,
    neutral_count: int = 0,
    metadata: dict[str, Any] | None = None,
    ranking_profile: str = "conservative",
) -> dict[str, Any]:
    """Create one normalized benchmark record."""
    positives = np.asarray(positives, dtype=float)
    negatives = np.asarray(negatives, dtype=float)
    # Use distributional geometry and regime-aware ideals for higher fidelity
    metrics = evaluate_distributional_geometry(
        positives,
        negatives,
        regime_aware=True,
        ranking_profile=ranking_profile,
        neutral_count=neutral_count,
    )
    total_positive = float(np.sum(positives)) if positives.size else 0.0
    total_negative = float(np.sum(negatives)) if negatives.size else 0.0
    net_total = total_positive + total_negative
    record = {
        "benchmark": benchmark,
        "benchmark_id": benchmark_id,
        "subject": subject,
        "positive_count": int(positives.size),
        "negative_count": int(negatives.size),
        "neutral_count": int(neutral_count),
        "positive_total": total_positive,
        "negative_total": total_negative,
        "net_total": net_total,
        "metrics": metrics,
        "metadata": metadata or {},
    }
    record["summary"] = {
        "success_rate": metrics["success_rate"],
        "expectancy": metrics["expectancy"],
        "ranking_score": metrics["ranking_score"],
        "credibility": metrics["credibility"],
        "alpha_s_sil": metrics["alpha_s_sil"],
        "smse": metrics["smse"],
        "normalized_mse_cofactors": metrics["normalized_mse_cofactors"],
        "cofactors_norm_diff": metrics["cofactors_norm_diff"],
        "normalized_cofactors_norm_diff": metrics["normalized_cofactors_norm_diff"],
        "normalized_cofactors_norm_diff_abs": metrics["normalized_cofactors_norm_diff_abs"],
        "mse_cofactors": metrics["mse_cofactors"],
        "relative_cofactors_fro_error": metrics["relative_cofactors_fro_error"],
        "structural_alpha": metrics["structural_alpha"],
        "positive_alpha_weight": metrics.get("positive_alpha_weight"),
        "positive_alpha_penalty": metrics.get("positive_alpha_penalty"),
        "positive_alpha_scale": metrics.get("positive_alpha_scale"),
        "rate_weighted_balance_gap": _metric_or_diagnostic(metrics, "rate_weighted_balance_gap"),
        "rate_weighted_balance_gap_abs": _metric_or_diagnostic(metrics, "rate_weighted_balance_gap_abs"),
        "distributional_structural_term": metrics.get("distributional_structural_term"),
        "distributional_alpha_s_sil": metrics.get("distributional_alpha_s_sil"),
        "variance": _metric_or_diagnostic(metrics, "variance"),
        "skewness": _metric_or_diagnostic(metrics, "skewness"),
        "total_outcomes": metrics["total_outcomes"],
    }
    return record


def summarize_benchmark_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an aggregate summary across normalized benchmark records."""
    if not records:
        return {
            "record_count": 0,
            "benchmarks": [],
        }

    keys = [
        "success_rate",
        "expectancy",
        "ranking_score",
        "credibility",
        "alpha_s_sil",
        "smse",
        "normalized_mse_cofactors",
        "cofactors_norm_diff",
        "normalized_cofactors_norm_diff",
        "normalized_cofactors_norm_diff_abs",
        "mse_cofactors",
        "relative_cofactors_fro_error",
        "structural_alpha",
        "positive_alpha_weight",
        "positive_alpha_penalty",
        "positive_alpha_scale",
        "rate_weighted_balance_gap",
        "rate_weighted_balance_gap_abs",
        "distributional_structural_term",
        "distributional_alpha_s_sil",
        "variance",
        "skewness",
    ]
    summary: dict[str, Any] = {
        "record_count": len(records),
        "benchmarks": sorted({str(record.get("benchmark") or "") for record in records}),
        "subjects": [str(record.get("subject") or "") for record in records],
    }
    for key in keys:
        values = [_to_float((record.get("summary") or {}).get(key)) for record in records]
        values = [value for value in values if value is not None]
        if not values:
            continue
        summary[key] = {
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(sum(values) / len(values)),
        }
    return summary


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    summary = record.get("summary") or {}
    flat = {
        "benchmark": record.get("benchmark"),
        "benchmark_id": record.get("benchmark_id"),
        "subject": record.get("subject"),
        "positive_count": record.get("positive_count"),
        "negative_count": record.get("negative_count"),
        "neutral_count": record.get("neutral_count"),
        "positive_total": record.get("positive_total"),
        "negative_total": record.get("negative_total"),
        "net_total": record.get("net_total"),
        "success_rate": summary.get("success_rate"),
        "expectancy": summary.get("expectancy"),
        "cofactors_norm_diff": summary.get("cofactors_norm_diff"),
        "normalized_cofactors_norm_diff": summary.get("normalized_cofactors_norm_diff"),
        "normalized_cofactors_norm_diff_abs": summary.get("normalized_cofactors_norm_diff_abs"),
        "mse_cofactors": summary.get("mse_cofactors"),
        "normalized_mse_cofactors": summary.get("normalized_mse_cofactors"),
        "relative_cofactors_fro_error": summary.get("relative_cofactors_fro_error"),
        "structural_alpha": summary.get("structural_alpha"),
        "positive_alpha_weight": summary.get("positive_alpha_weight"),
        "positive_alpha_penalty": summary.get("positive_alpha_penalty"),
        "positive_alpha_scale": summary.get("positive_alpha_scale"),
        "rate_weighted_balance_gap": summary.get("rate_weighted_balance_gap"),
        "rate_weighted_balance_gap_abs": summary.get("rate_weighted_balance_gap_abs"),
        "total_outcomes": summary.get("total_outcomes"),
    }
    for key, value in metadata.items():
        flat[f"meta_{key}"] = value
    return flat


def write_benchmark_outputs(
    output_dir: str | Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write JSON and CSV outputs for benchmark runs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = summary or summarize_benchmark_records(records)
    summary_path = output_path / "summary.json"
    records_path = output_path / "records.json"
    csv_path = output_path / "records.csv"

    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    records_path.write_text(json.dumps(records, indent=2) + "\n")

    flattened = [_flatten_record(record) for record in records]
    fieldnames: list[str] = []
    for row in flattened:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)

    return {
        "summary": str(summary_path),
        "records_json": str(records_path),
        "records_csv": str(csv_path),
    }
