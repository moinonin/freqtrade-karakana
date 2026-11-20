"""Interpretation helpers for turning benchmark metrics into product signals."""

from __future__ import annotations

from typing import Any

LOWER_IS_BETTER = {
    "ranking_score",
    "alpha_s_sil",
    "alpha_s_mse",
    "normalized_mse_cofactors",
    "relative_cofactors_fro_error",
    "distributional_structural_term",
    "normalized_cofactors_norm_diff_abs",
}

HIGHER_IS_BETTER = {
    "success_rate",
    "expectancy",
    "credibility",
    "accuracy",
    "balanced_accuracy",
}


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def score_band(record: dict[str, Any]) -> str:
    """Return a conservative, human-readable quality band for one record."""
    summary = record.get("summary") or {}
    ranking = _to_float(summary.get("ranking_score"))
    nmse = _to_float(summary.get("normalized_mse_cofactors"))
    rel_fro = _to_float(summary.get("relative_cofactors_fro_error"))
    credibility = _to_float(summary.get("credibility"))
    success_rate = _to_float(summary.get("success_rate"))
    expectancy = _to_float(summary.get("expectancy"))

    if credibility is not None and credibility < 0.25:
        return "low-confidence"
    if success_rate is not None and success_rate <= 0.0:
        return "fragile"
    if expectancy is not None and expectancy < 0.0:
        return "watch"
    if nmse is not None and rel_fro is not None:
        if nmse <= 0.25 and rel_fro <= 0.35:
            return "excellent"
        if nmse <= 1.0 and rel_fro <= 1.0:
            return "usable"
        if nmse <= 1.75 and rel_fro <= 1.35:
            return "watch"
        return "fragile"
    if ranking is not None and ranking <= 1.5:
        return "usable"
    return "watch"


def record_warnings(record: dict[str, Any]) -> list[str]:
    """Return concise warnings that explain why a result needs caution."""
    summary = record.get("summary") or {}
    warnings: list[str] = []

    credibility = _to_float(summary.get("credibility"))
    success_rate = _to_float(summary.get("success_rate"))
    expectancy = _to_float(summary.get("expectancy"))
    nmse = _to_float(summary.get("normalized_mse_cofactors"))
    rel_fro = _to_float(summary.get("relative_cofactors_fro_error"))
    dist_term = _to_float(summary.get("distributional_structural_term"))

    if credibility is not None and credibility < 0.5:
        warnings.append("low sample credibility")
    if success_rate is not None and success_rate <= 0.0:
        warnings.append("no positive outcomes")
    if expectancy is not None and expectancy < 0.0:
        warnings.append("negative expectancy")
    if nmse is not None and nmse > 1.5:
        warnings.append("high cofactor mismatch")
    if rel_fro is not None and rel_fro > 1.25:
        warnings.append("high relative structural error")
    if dist_term is not None and dist_term > 3.0:
        warnings.append("distributional instability")

    return warnings


def record_strengths(record: dict[str, Any]) -> list[str]:
    """Return concise strengths that explain why a record ranked well."""
    summary = record.get("summary") or {}
    metadata = record.get("metadata") or {}
    strengths: list[str] = []

    success_rate = _to_float(summary.get("success_rate"))
    expectancy = _to_float(summary.get("expectancy"))
    credibility = _to_float(summary.get("credibility"))
    nmse = _to_float(summary.get("normalized_mse_cofactors"))
    rel_fro = _to_float(summary.get("relative_cofactors_fro_error"))
    dist_term = _to_float(summary.get("distributional_structural_term"))
    accuracy = _to_float(metadata.get("accuracy"))

    if success_rate is not None and success_rate >= 0.8:
        strengths.append("high success rate")
    if expectancy is not None and expectancy > 0.0:
        strengths.append("positive expectancy")
    if credibility is not None and credibility >= 0.75:
        strengths.append("high sample credibility")
    if nmse is not None and nmse <= 1.0:
        strengths.append("low cofactor mismatch")
    if rel_fro is not None and rel_fro <= 1.0:
        strengths.append("low relative structural error")
    if dist_term is not None and dist_term <= 1.5:
        strengths.append("distributionally stable")
    if accuracy is not None and accuracy >= 0.8:
        strengths.append("strong conventional accuracy")

    return strengths


def data_quality_warnings(summary: dict[str, Any]) -> list[str]:
    """Return dataset-level warnings from benchmark summaries."""
    warnings: list[str] = []
    error_count = _to_float(summary.get("error_count"))
    completed = _to_float(summary.get("completed_subject_count") or summary.get("record_count"))
    if error_count and error_count > 0:
        warnings.append(f"{int(error_count)} input rows were skipped or rejected")
    neutral_count = _to_float(summary.get("neutral_count"))
    if neutral_count and neutral_count > 0:
        warnings.append(f"{int(neutral_count)} neutral rows were retained as context")
    if completed is not None and completed <= 0:
        warnings.append("no benchmark records completed")
    failed = _to_float(summary.get("failed_task_count"))
    if failed and failed > 0:
        warnings.append(f"{int(failed)} benchmark tasks failed")
    return warnings


def sort_records(records: list[dict[str, Any]], metric: str = "ranking_score") -> list[dict[str, Any]]:
    """Sort benchmark records by a metric using the expected direction."""
    reverse = metric in HIGHER_IS_BETTER

    def key(record: dict[str, Any]) -> tuple[int, float]:
        value = _to_float((record.get("summary") or {}).get(metric))
        if value is None:
            return (1, 0.0)
        return (0, value)

    return sorted(records, key=key, reverse=reverse)


def summarize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a compact product-facing representation of one benchmark record."""
    summary = record.get("summary") or {}
    metadata = record.get("metadata") or {}
    return {
        "benchmark": record.get("benchmark"),
        "subject": record.get("subject"),
        "band": score_band(record),
        "ranking_score": _to_float(summary.get("ranking_score")),
        "success_rate": _to_float(summary.get("success_rate")),
        "expectancy": _to_float(summary.get("expectancy")),
        "credibility": _to_float(summary.get("credibility")),
        "normalized_mse_cofactors": _to_float(summary.get("normalized_mse_cofactors")),
        "relative_cofactors_fro_error": _to_float(summary.get("relative_cofactors_fro_error")),
        "distributional_structural_term": _to_float(summary.get("distributional_structural_term")),
        "accuracy": _to_float(metadata.get("accuracy")),
        "balanced_accuracy": _to_float(metadata.get("balanced_accuracy")),
        "strengths": record_strengths(record),
        "warnings": record_warnings(record),
    }
