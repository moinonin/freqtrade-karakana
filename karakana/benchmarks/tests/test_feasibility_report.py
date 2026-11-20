import json

from karakana.benchmarks.feasibility_report import build_feasibility_report, render_markdown


def _write_run(path, benchmark, subject, ranking):
    path.mkdir()
    summary = {
        "record_count": 1,
        "benchmarks": [benchmark],
    }
    records = [
        {
            "benchmark": benchmark,
            "subject": subject,
            "summary": {
                "ranking_score": ranking,
                "success_rate": 1.0,
                "expectancy": 1.0,
                "credibility": 1.0,
                "normalized_mse_cofactors": 0.1,
                "relative_cofactors_fro_error": 0.2,
            },
            "metadata": {},
        }
    ]
    (path / "summary.json").write_text(json.dumps(summary))
    (path / "records.json").write_text(json.dumps(records))


def test_build_feasibility_report_selects_overall_best(tmp_path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    _write_run(run_a, "gymnasium", "a", 2.0)
    _write_run(run_b, "openml", "b", 1.0)

    report = build_feasibility_report({"gymnasium": run_a, "openml": run_b})

    assert report["record_count"] == 2
    assert report["overall_best"][0]["subject"] == "b"


def test_render_markdown_contains_assessment_and_table(tmp_path):
    run = tmp_path / "run"
    _write_run(run, "payoff_csv", "candidate", 1.0)
    report = build_feasibility_report({"payoff_csv": run})

    markdown = render_markdown(report)

    assert "# karakana Feasibility Report" in markdown
    assert "| Subject | Band | Ranking | Success |" in markdown
    assert "Why" in markdown
    assert "candidate" in markdown


def test_data_quality_warnings_stay_benchmark_scoped(tmp_path):
    clean = tmp_path / "clean"
    noisy = tmp_path / "noisy"
    _write_run(clean, "openml", "clean", 1.0)
    _write_run(noisy, "payoff_csv", "noisy", 2.0)

    summary = json.loads((noisy / "summary.json").read_text())
    summary["error_count"] = 3
    (noisy / "summary.json").write_text(json.dumps(summary))

    report = build_feasibility_report({"openml": clean, "payoff_csv": noisy})

    assert report["warnings"] == []
    sections = {section["name"]: section for section in report["benchmarks"]}
    assert sections["openml"]["data_quality_warnings"] == []
    assert sections["payoff_csv"]["data_quality_warnings"] == ["3 input rows were skipped or rejected"]
