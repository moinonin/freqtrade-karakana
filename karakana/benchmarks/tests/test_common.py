import csv

import numpy as np

from karakana.benchmarks.common import build_benchmark_record, write_benchmark_outputs
from karakana.benchmarks.payoff_csv_runner import _normalize_outcome, _parse_label_set


def test_benchmark_csv_contains_structural_fields(tmp_path):
    record = build_benchmark_record(
        benchmark="unit",
        benchmark_id="unit",
        subject="subject",
        positives=np.array([2.0, 3.0]),
        negatives=np.array([-1.0]),
    )

    outputs = write_benchmark_outputs(tmp_path, [record])

    with open(outputs["records_csv"], newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["cofactors_norm_diff"] != ""
    assert row["relative_cofactors_fro_error"] != ""
    assert row["rate_weighted_balance_gap"] != ""


def test_payoff_csv_zero_magnitude_is_neutral_not_error():
    normalized, error = _normalize_outcome(
        {"magnitude": "0"},
        magnitude_col="magnitude",
        outcome_col=None,
        positive_labels=set(),
        negative_labels=set(),
        neutral_labels=set(),
    )

    assert normalized == 0.0
    assert error is None


def test_payoff_csv_neutral_label_is_neutral_not_error():
    normalized, error = _normalize_outcome(
        {"magnitude": "10", "outcome": "flat"},
        magnitude_col="magnitude",
        outcome_col="outcome",
        positive_labels=_parse_label_set("win"),
        negative_labels=_parse_label_set("loss"),
        neutral_labels=_parse_label_set("flat"),
    )

    assert normalized == 0.0
    assert error is None
