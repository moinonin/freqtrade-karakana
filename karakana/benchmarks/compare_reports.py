"""Compare two benchmark or feasibility report outputs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SUMMARY_METRICS = [
    "success_rate",
    "expectancy",
    "ranking_score",
    "credibility",
    "alpha_s_sil",
    "normalized_mse_cofactors",
    "relative_cofactors_fro_error",
    "distributional_structural_term",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _resolve_summary(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        summary = p / "summary.json"
        if summary.exists():
            return _load_json(summary)
        report = p / "feasibility_report.json"
        if report.exists():
            return _load_json(report)
    return _load_json(p)


def _metric_mean(summary: dict[str, Any], metric: str) -> float | None:
    value = summary.get(metric)
    if isinstance(value, dict) and "mean" in value:
        try:
            return float(value["mean"])
        except Exception:
            return None
    return None


def _direction(metric: str) -> str:
    if metric in {"success_rate", "expectancy", "credibility"}:
        return "higher"
    return "lower"


def _delta_status(metric: str, delta: float | None) -> str:
    if delta is None:
        return "n/a"
    direction = _direction(metric)
    if abs(delta) < 1e-12:
        return "flat"
    improved = delta > 0 if direction == "higher" else delta < 0
    return "improved" if improved else "regressed"


def _pairwise_index(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    comparisons = summary.get("pairwise_comparisons") or []
    indexed: dict[str, dict[str, Any]] = {}
    for comparison in comparisons:
        label = f"{comparison.get('model_a')} vs {comparison.get('model_b')}"
        for metric, stats in (comparison.get("metrics") or {}).items():
            indexed[f"{label}:{metric}"] = stats
    return indexed


def build_comparison_report(
    baseline_path: str | Path,
    latest_path: str | Path,
    *,
    baseline_label: str = "baseline",
    latest_label: str = "latest",
) -> dict[str, Any]:
    """Build a comparison report between two summary/report artifacts."""
    baseline = _resolve_summary(baseline_path)
    latest = _resolve_summary(latest_path)

    metric_rows = []
    for metric in SUMMARY_METRICS:
        before = _metric_mean(baseline, metric)
        after = _metric_mean(latest, metric)
        delta = None if before is None or after is None else after - before
        metric_rows.append(
            {
                "metric": metric,
                "direction": _direction(metric),
                "baseline_mean": before,
                "latest_mean": after,
                "delta": delta,
                "status": _delta_status(metric, delta),
            }
        )

    baseline_pairwise = _pairwise_index(baseline)
    latest_pairwise = _pairwise_index(latest)
    pairwise_rows = []
    for key in sorted(set(baseline_pairwise) | set(latest_pairwise)):
        before = baseline_pairwise.get(key) or {}
        after = latest_pairwise.get(key) or {}
        pairwise_rows.append(
            {
                "metric": key,
                "baseline_wins": before.get("wins"),
                "latest_wins": after.get("wins"),
                "baseline_total": before.get("total"),
                "latest_total": after.get("total"),
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline_label": baseline_label,
        "latest_label": latest_label,
        "baseline_path": str(baseline_path),
        "latest_path": str(latest_path),
        "baseline_record_count": baseline.get("record_count"),
        "latest_record_count": latest.get("record_count"),
        "metric_deltas": metric_rows,
        "pairwise_deltas": pairwise_rows,
        "assessment": _assessment(metric_rows),
    }


def _assessment(metric_rows: list[dict[str, Any]]) -> str:
    improved = sum(1 for row in metric_rows if row["status"] == "improved")
    regressed = sum(1 for row in metric_rows if row["status"] == "regressed")
    if improved > regressed:
        return "latest improved on more aggregate metrics than it regressed"
    if regressed > improved:
        return "latest regressed on more aggregate metrics than it improved"
    return "latest is mixed or flat versus baseline"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if abs(number) >= 1000 or (0 < abs(number) < 0.0001):
        return f"{number:.3e}"
    return f"{number:.4f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# karakana Benchmark Comparison",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        f"Baseline: `{report['baseline_label']}` ({report['baseline_record_count']} records)",
        f"Latest: `{report['latest_label']}` ({report['latest_record_count']} records)",
        "",
        f"Assessment: **{report['assessment']}**",
        "",
        "## Aggregate Metric Deltas",
        "",
        "| Metric | Direction | Baseline | Latest | Delta | Status |",
        "| :--- | :--- | ---: | ---: | ---: | :--- |",
    ]
    for row in report["metric_deltas"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["metric"],
                    row["direction"],
                    _fmt(row["baseline_mean"]),
                    _fmt(row["latest_mean"]),
                    _fmt(row["delta"]),
                    row["status"],
                ]
            )
            + " |"
        )
    if report["pairwise_deltas"]:
        lines.extend(["", "## Pairwise Metric Coverage", ""])
        lines.extend(
            [
                "| Metric | Baseline Wins | Latest Wins | Baseline Total | Latest Total |",
                "| :--- | :--- | :--- | ---: | ---: |",
            ]
        )
        for row in report["pairwise_deltas"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["metric"],
                        str(row.get("baseline_wins") or "n/a"),
                        str(row.get("latest_wins") or "n/a"),
                        str(row.get("baseline_total") or "n/a"),
                        str(row.get("latest_total") or "n/a"),
                    ]
                )
                + " |"
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two karakana benchmark summaries.")
    parser.add_argument("--baseline", required=True, help="Baseline summary/report file or run directory")
    parser.add_argument("--latest", required=True, help="Latest summary/report file or run directory")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--latest-label", default="latest")
    parser.add_argument("--output-dir", default="benchmark_runs/comparison_report")
    args = parser.parse_args()

    report = build_comparison_report(
        args.baseline,
        args.latest,
        baseline_label=args.baseline_label,
        latest_label=args.latest_label,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison_report.json"
    markdown_path = output_dir / "comparison_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(render_markdown(report))

    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
