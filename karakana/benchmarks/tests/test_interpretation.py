from karakana.benchmarks.interpretation import (
    data_quality_warnings,
    record_strengths,
    record_warnings,
    score_band,
    sort_records,
)


def _record(subject, ranking, nmse, rel_fro, success_rate=1.0, expectancy=1.0, credibility=1.0):
    return {
        "subject": subject,
        "summary": {
            "ranking_score": ranking,
            "normalized_mse_cofactors": nmse,
            "relative_cofactors_fro_error": rel_fro,
            "success_rate": success_rate,
            "expectancy": expectancy,
            "credibility": credibility,
        },
    }


def test_score_band_marks_excellent_record():
    record = _record("good", 0.2, 0.1, 0.2)

    assert score_band(record) == "excellent"


def test_score_band_marks_no_positive_outcomes_as_fragile():
    record = _record("no-win", 1.0, 1.0, 1.0, success_rate=0.0, expectancy=-0.1)

    assert score_band(record) == "fragile"


def test_record_warnings_explain_fragility():
    record = _record("bad", 10.0, 2.0, 1.5, success_rate=0.0, expectancy=-1.0)

    warnings = record_warnings(record)

    assert "no positive outcomes" in warnings
    assert "negative expectancy" in warnings
    assert "high cofactor mismatch" in warnings


def test_record_strengths_explain_good_candidate():
    record = _record("good", 0.2, 0.1, 0.2, success_rate=0.9, expectancy=1.0)

    strengths = record_strengths(record)

    assert "high success rate" in strengths
    assert "positive expectancy" in strengths
    assert "low cofactor mismatch" in strengths


def test_sort_records_uses_lower_ranking_score_as_better():
    records = [_record("worse", 2.0, 1.0, 1.0), _record("better", 1.0, 1.0, 1.0)]

    assert [record["subject"] for record in sort_records(records)] == ["better", "worse"]


def test_data_quality_warnings_include_skipped_rows():
    warnings = data_quality_warnings({"error_count": 3, "completed_subject_count": 2})

    assert warnings == ["3 input rows were skipped or rejected"]
