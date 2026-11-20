"""Run or assemble public validation benchmarks for karakana."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .compare_reports import build_comparison_report
from .compare_reports import render_markdown as render_comparison_markdown
from .feasibility_report import build_feasibility_report
from .feasibility_report import render_markdown as render_feasibility_markdown

UCI_WINE_URLS = {
    "red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "white": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
}


def _run_command(args: list[str]) -> None:
    subprocess.run(args, check=True)


def build_uci_wine_payoff_csv(output_dir: str | Path) -> Path:
    """Download UCI Wine Quality data and convert it to payoff-style CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frames = []
    for color, url in UCI_WINE_URLS.items():
        df = pd.read_csv(url, sep=";")
        df["subject"] = color
        df["outcome"] = df["quality"].apply(lambda q: "success" if q >= 7 else "failure")
        df["magnitude"] = df["quality"].apply(lambda q: abs(float(q) - 6.0) or 0.5)
        frames.append(df[["subject", "magnitude", "outcome", "quality"]])
    combined = pd.concat(frames, ignore_index=True)
    csv_path = output_path / "wine_quality_payoff.csv"
    combined.to_csv(csv_path, index=False)
    return csv_path


def _write_report(report: dict, output_dir: str | Path, stem: str = "feasibility_report") -> dict[str, str]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / f"{stem}.json"
    markdown_path = path / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    if stem == "comparison_report":
        markdown_path.write_text(render_comparison_markdown(report))
    else:
        markdown_path.write_text(render_feasibility_markdown(report))
    return {"json": str(json_path), "markdown": str(markdown_path)}


def run_public_validation(
    *,
    mode: str = "smoke",
    output_dir: str | Path = "benchmark_runs/public_validation",
    reuse_existing: bool = False,
    baseline_openml_dir: str | Path | None = "benchmark_runs/public_openml_cc18_10",
) -> dict[str, str]:
    """Run public validation benchmarks and generate reports."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    task_limit = 10 if mode == "smoke" else 72
    openml_dir = output_path / f"openml_cc18_{mode}"
    wine_dir = output_path / "uci_wine_quality"

    if not reuse_existing or not (openml_dir / "summary.json").exists():
        _run_command(
            [
                sys.executable,
                "-m",
                "karakana.benchmarks.openml_runner",
                "--suite",
                "OpenML-CC18",
                "--task-limit",
                str(task_limit),
                "--models",
                "dummy,logreg",
                "--output-dir",
                str(openml_dir),
            ]
        )

    if not reuse_existing or not (wine_dir / "summary.json").exists():
        csv_path = build_uci_wine_payoff_csv(wine_dir)
        _run_command(
            [
                sys.executable,
                "-m",
                "karakana.benchmarks.payoff_csv_runner",
                "--input",
                str(csv_path),
                "--subject-col",
                "subject",
                "--magnitude-col",
                "magnitude",
                "--outcome-col",
                "outcome",
                "--output-dir",
                str(wine_dir),
            ]
        )

    report = build_feasibility_report(
        {
            "openml": openml_dir,
            "uci_wine_quality": wine_dir,
        }
    )
    report_paths = _write_report(report, output_path / "report")

    if baseline_openml_dir and Path(baseline_openml_dir).exists():
        comparison = build_comparison_report(
            baseline_openml_dir,
            openml_dir,
            baseline_label=str(Path(baseline_openml_dir).name),
            latest_label=str(openml_dir.name),
        )
        comparison_paths = _write_report(comparison, output_path / "comparison", stem="comparison_report")
    else:
        comparison_paths = {}

    return {
        "openml_dir": str(openml_dir),
        "uci_wine_dir": str(wine_dir),
        "report_markdown": report_paths["markdown"],
        "report_json": report_paths["json"],
        "comparison_markdown": comparison_paths.get("markdown", ""),
        "comparison_json": comparison_paths.get("json", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run public validation benchmarks and reports.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", default="benchmark_runs/public_validation")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing output folders when present")
    parser.add_argument("--baseline-openml-dir", default="benchmark_runs/public_openml_cc18_10")
    args = parser.parse_args()

    outputs = run_public_validation(
        mode=args.mode,
        output_dir=args.output_dir,
        reuse_existing=args.reuse_existing,
        baseline_openml_dir=args.baseline_openml_dir,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
