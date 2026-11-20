import json

from karakana.benchmarks.compare_reports import build_comparison_report, render_markdown


def _write_summary(path, ranking_mean, success_mean, total=1):
    path.mkdir()
    summary = {
        "record_count": total,
        "ranking_score": {"mean": ranking_mean},
        "success_rate": {"mean": success_mean},
        "pairwise_comparisons": [
            {
                "model_a": "a",
                "model_b": "b",
                "metrics": {
                    "ranking_score": {
                        "wins": {"a": 0, "b": total},
                        "ties": 0,
                        "total": total,
                    }
                },
            }
        ],
    }
    (path / "summary.json").write_text(json.dumps(summary))


def test_build_comparison_report_calculates_metric_deltas(tmp_path):
    baseline = tmp_path / "baseline"
    latest = tmp_path / "latest"
    _write_summary(baseline, ranking_mean=2.0, success_mean=0.5)
    _write_summary(latest, ranking_mean=1.0, success_mean=0.75)

    report = build_comparison_report(baseline, latest)

    statuses = {row["metric"]: row["status"] for row in report["metric_deltas"]}
    assert statuses["ranking_score"] == "improved"
    assert statuses["success_rate"] == "improved"


def test_render_comparison_markdown_has_pairwise_section(tmp_path):
    baseline = tmp_path / "baseline"
    latest = tmp_path / "latest"
    _write_summary(baseline, ranking_mean=2.0, success_mean=0.5)
    _write_summary(latest, ranking_mean=1.0, success_mean=0.75)
    report = build_comparison_report(baseline, latest)

    markdown = render_markdown(report)

    assert "# karakana Benchmark Comparison" in markdown
    assert "Pairwise Metric Coverage" in markdown
