from karakana.benchmarks.public_validation import build_uci_wine_payoff_csv


def test_build_uci_wine_payoff_csv_with_mocked_pandas(monkeypatch, tmp_path):
    import csv

    import pandas as pd

    def fake_read_csv(url, sep):
        return pd.DataFrame({"quality": [5, 6, 7, 8]})

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    csv_path = build_uci_wine_payoff_csv(tmp_path)
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["subject"] for row in rows} == {"red", "white"}
    assert {row["outcome"] for row in rows} == {"success", "failure"}
    assert "magnitude" in rows[0]
