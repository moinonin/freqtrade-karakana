"""CSV payoff benchmark runner for karakana."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import build_benchmark_record, summarize_benchmark_records, write_benchmark_outputs


def _parse_label_set(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _normalize_outcome(
    row: dict[str, str],
    *,
    magnitude_col: str,
    outcome_col: str | None,
    positive_labels: set[str],
    negative_labels: set[str],
    neutral_labels: set[str],
) -> tuple[float | None, str | None]:
    raw_magnitude = row.get(magnitude_col, "")
    if raw_magnitude is None or str(raw_magnitude).strip() == "":
        return None, "missing magnitude"
    try:
        magnitude = float(raw_magnitude)
    except Exception:
        return None, f"invalid magnitude={raw_magnitude!r}"

    if outcome_col:
        raw_outcome = str(row.get(outcome_col, "")).strip().lower()
        if raw_outcome in positive_labels:
            return abs(magnitude), None
        if raw_outcome in negative_labels:
            return -abs(magnitude), None
        if raw_outcome in neutral_labels:
            return 0.0, None
        return None, f"unknown outcome label={raw_outcome!r}"

    if magnitude > 0:
        return magnitude, None
    if magnitude < 0:
        return magnitude, None
    return 0.0, None


def _subject_name(row: dict[str, str], subject_col: str | None, fallback: str) -> str:
    if subject_col:
        value = str(row.get(subject_col, "")).strip()
        if value:
            return value
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark karakana on payoff-style CSV data.")
    parser.add_argument("--input", required=True, help="CSV file containing payoff events")
    parser.add_argument("--benchmark-id", default=None, help="Optional benchmark id; defaults to CSV stem")
    parser.add_argument("--subject-col", default="subject", help="Column used to group records into subjects")
    parser.add_argument("--magnitude-col", default="magnitude", help="Column containing payoff magnitudes")
    parser.add_argument(
        "--outcome-col",
        default="outcome",
        help="Optional outcome label column. Use an empty string to infer sign directly from magnitude",
    )
    parser.add_argument(
        "--positive-labels",
        default="success,win,positive,1,true",
        help="Comma-separated labels treated as positive when --outcome-col is used",
    )
    parser.add_argument(
        "--negative-labels",
        default="failure,loss,negative,0,false",
        help="Comma-separated labels treated as negative when --outcome-col is used",
    )
    parser.add_argument(
        "--neutral-labels",
        default="neutral,flat,none,no_action,no-action,0.0",
        help="Comma-separated labels treated as neutral when --outcome-col is used",
    )
    parser.add_argument("--output-dir", default="benchmark_runs/payoff_csv", help="Directory for benchmark outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"CSV input not found: {input_path}")

    benchmark_id = str(args.benchmark_id or input_path.stem)
    outcome_col = str(args.outcome_col).strip() or None
    positive_labels = _parse_label_set(args.positive_labels)
    negative_labels = _parse_label_set(args.negative_labels)
    neutral_labels = _parse_label_set(args.neutral_labels)
    grouped_values: dict[str, list[float]] = defaultdict(list)
    errors: list[dict[str, Any]] = []

    with input_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV file has no header: {input_path}")
        for row_idx, row in enumerate(reader, start=2):
            subject = _subject_name(row, args.subject_col, benchmark_id)
            normalized, error = _normalize_outcome(
                row,
                magnitude_col=args.magnitude_col,
                outcome_col=outcome_col,
                positive_labels=positive_labels,
                negative_labels=negative_labels,
                neutral_labels=neutral_labels,
            )
            if error is not None or normalized is None:
                errors.append({"row": row_idx, "subject": subject, "error": error})
                continue
            grouped_values[subject].append(float(normalized))

    records: list[dict[str, Any]] = []
    for subject, values in sorted(grouped_values.items()):
        positives = np.asarray([value for value in values if value > 0], dtype=float)
        negatives = np.asarray([value for value in values if value < 0], dtype=float)
        neutral_count = sum(1 for value in values if value == 0)
        if positives.size + negatives.size == 0:
            continue
        records.append(
            build_benchmark_record(
                benchmark="payoff_csv",
                benchmark_id=benchmark_id,
                subject=subject,
                positives=positives,
                negatives=negatives,
                neutral_count=neutral_count,
                metadata={
                    "source_csv": str(input_path.resolve()),
                    "subject_col": args.subject_col,
                    "magnitude_col": args.magnitude_col,
                    "outcome_col": outcome_col,
                    "row_count": len(values),
                    "neutral_count": neutral_count,
                },
            )
        )

    summary = summarize_benchmark_records(records)
    summary["input_csv"] = str(input_path.resolve())
    summary["benchmark_id"] = benchmark_id
    summary["completed_subject_count"] = len(records)
    summary["neutral_count"] = int(sum(record.get("neutral_count", 0) for record in records))
    summary["error_count"] = len(errors)
    if errors:
        summary["errors"] = errors[:100]
        summary["errors_truncated"] = len(errors) > 100
    outputs = write_benchmark_outputs(args.output_dir, records, summary)
    print(json.dumps({"summary": summary, "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
