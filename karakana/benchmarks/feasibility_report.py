"""Generate a consolidated feasibility report from benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .interpretation import (
    data_quality_warnings,
    sort_records,
    summarize_record,
)

DEFAULT_RUNS = {
    "gymnasium": "benchmark_runs/feasibility_gym_cartpole",
    "payoff_csv": "benchmark_runs/feasibility_payoff_trade_history",
    "openml": "benchmark_runs/feasibility_openml_cc18",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_run(name: str, path: str | Path) -> dict[str, Any]:
    run_path = Path(path)
    summary_path = run_path / "summary.json"
    records_path = run_path / "records.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json for {name}: {summary_path}")
    if not records_path.exists():
        raise FileNotFoundError(f"Missing records.json for {name}: {records_path}")
    summary = _load_json(summary_path)
    records = _load_json(records_path)
    return {
        "name": name,
        "path": str(run_path),
        "summary": summary,
        "records": records,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if abs(number) >= 1000 or (0 < abs(number) < 0.0001):
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def _top_records(records: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    return [summarize_record(record) for record in sort_records(records)[:limit]]


def _benchmark_section(run: dict[str, Any]) -> dict[str, Any]:
    records = run["records"]
    summary = run["summary"]
    best = _top_records(records, limit=min(3, len(records)))
    warnings = data_quality_warnings(summary)
    bands: dict[str, int] = {}
    compact_records = []
    for record in records:
        compact = summarize_record(record)
        compact_records.append(compact)
        bands[compact["band"]] = bands.get(compact["band"], 0) + 1
    return {
        "name": run["name"],
        "path": run["path"],
        "record_count": len(records),
        "bands": bands,
        "metric_agreement": _metric_agreement(summary),
        "best_records": best,
        "data_quality_warnings": warnings,
        "records": compact_records,
    }


def build_feasibility_report(runs: dict[str, str | Path]) -> dict[str, Any]:
    """Build a consolidated feasibility report from benchmark output folders."""
    loaded = [_load_run(name, path) for name, path in runs.items()]
    sections = [_benchmark_section(run) for run in loaded]
    all_records = [record for run in loaded for record in run["records"]]
    all_compact = [summarize_record(record) for record in sort_records(all_records)]
    warnings: list[str] = []
    if not all_records:
        warnings.append("No records were available across benchmark runs")

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_count": len(loaded),
        "record_count": len(all_records),
        "benchmarks": sections,
        "overall_best": all_compact[:5],
        "warnings": warnings,
        "assessment": _assessment(sections, all_compact, warnings),
        "interpretation_notes": [
            "Ranking scores are most reliable within the same benchmark family and configuration.",
            "Cross-domain ordering is useful for smoke-screening, but should not be treated as a universal leaderboard.",
            "Quality bands combine structural metrics with basic business viability checks such as positive outcomes and expectancy.",
        ],
    }


def _assessment(
    sections: list[dict[str, Any]],
    best_records: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, str]:
    has_openml = any(section["name"] == "openml" for section in sections)
    has_gym = any(section["name"] == "gymnasium" for section in sections)
    has_payoff = any(section["name"] == "payoff_csv" for section in sections)
    best = best_records[0] if best_records else {}
    band = str(best.get("band") or "unknown")

    feasibility = "promising"
    if not best_records:
        feasibility = "blocked"
    elif band in {"excellent", "usable"} and has_openml and has_gym and has_payoff:
        feasibility = "strong"
    elif warnings:
        feasibility = "promising with cautions"
    if warnings and feasibility == "strong":
        feasibility = "strong with data-quality cautions"

    return {
        "technical_feasibility": feasibility,
        "best_candidate": str(best.get("subject") or "n/a"),
        "primary_use_case": (
            "cross-domain outcome auditing for ML, RL, and payoff systems"
            if has_openml and has_gym and has_payoff
            else "benchmark-specific structural evaluation"
        ),
        "next_validation_step": ("repeat benchmarks across more seeds/tasks and compare against domain-native metrics"),
    }


def _metric_agreement(summary: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons = summary.get("pairwise_comparisons") or []
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        model_a = comparison.get("model_a")
        model_b = comparison.get("model_b")
        metrics = comparison.get("metrics") or {}
        for metric, stats in metrics.items():
            wins = stats.get("wins") or {}
            rows.append(
                {
                    "comparison": f"{model_a} vs {model_b}",
                    "metric": metric,
                    "higher_is_better": bool(stats.get("higher_is_better")),
                    "wins": wins,
                    "ties": int(stats.get("ties") or 0),
                    "total": int(stats.get("total") or 0),
                    "winner": _winner_from_wins(wins),
                }
            )
    return rows


def _winner_from_wins(wins: dict[str, Any]) -> str:
    if not wins:
        return "n/a"
    ordered = sorted(wins.items(), key=lambda item: int(item[1]), reverse=True)
    if len(ordered) > 1 and int(ordered[0][1]) == int(ordered[1][1]):
        return "tie"
    return str(ordered[0][0])


def render_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable Markdown report."""
    lines = [
        "# karakana Feasibility Report",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Executive Assessment",
        "",
        f"- Technical feasibility: **{report['assessment']['technical_feasibility']}**",
        f"- Best candidate: `{report['assessment']['best_candidate']}`",
        f"- Primary use case: {report['assessment']['primary_use_case']}",
        f"- Next validation step: {report['assessment']['next_validation_step']}",
        "",
    ]

    if report["warnings"]:
        lines.extend(["## Report Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    lines.extend(["## Interpretation Notes", ""])
    for note in report["interpretation_notes"]:
        lines.append(f"- {note}")
    lines.append("")

    lines.extend(["## Overall Best Candidates", ""])
    lines.extend(_record_table(report["overall_best"]))
    lines.append("")

    for section in report["benchmarks"]:
        lines.extend([f"## {section['name']}", ""])
        lines.append(f"Records: `{section['record_count']}`")
        lines.append(f"Quality bands: `{section['bands']}`")
        lines.append("")
        if section["data_quality_warnings"]:
            lines.append("Data-quality warnings:")
            for warning in section["data_quality_warnings"]:
                lines.append(f"- {warning}")
            lines.append("")
        if section["metric_agreement"]:
            lines.extend(_agreement_table(section["metric_agreement"]))
            lines.append("")
        lines.extend(_record_table(section["best_records"]))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _record_table(records: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Subject | Band | Ranking | Success | Expectancy | Credibility | NMSE | Rel. Fro | Why | Warnings |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |",
    ]
    for record in records:
        strengths = ", ".join(record.get("strengths") or []) or "-"
        warnings = ", ".join(record.get("warnings") or []) or "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.get("subject") or ""),
                    str(record.get("band") or ""),
                    _fmt(record.get("ranking_score")),
                    _fmt(record.get("success_rate")),
                    _fmt(record.get("expectancy")),
                    _fmt(record.get("credibility")),
                    _fmt(record.get("normalized_mse_cofactors")),
                    _fmt(record.get("relative_cofactors_fro_error")),
                    strengths,
                    warnings,
                ]
            )
            + " |"
        )
    return lines


def _agreement_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Comparison | Metric | Winner | Wins | Ties | Total |",
        "| :--- | :--- | :--- | :--- | ---: | ---: |",
    ]
    for row in rows:
        wins = ", ".join(f"{key}: {value}" for key, value in (row.get("wins") or {}).items())
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("comparison") or ""),
                    str(row.get("metric") or ""),
                    str(row.get("winner") or ""),
                    wins,
                    str(row.get("ties") or 0),
                    str(row.get("total") or 0),
                ]
            )
            + " |"
        )
    return lines


def _parse_run(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = Path(value)
        return path.name, value
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a consolidated karakana feasibility report.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Benchmark run folder as name=path. May be repeated. Defaults to current feasibility runs.",
    )
    parser.add_argument(
        "--output-dir", default="benchmark_runs/feasibility_report", help="Directory for report outputs"
    )
    args = parser.parse_args()

    runs = dict(_parse_run(value) for value in args.run) if args.run else DEFAULT_RUNS
    report = build_feasibility_report(runs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "feasibility_report.json"
    markdown_path = output_dir / "feasibility_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(render_markdown(report))

    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
