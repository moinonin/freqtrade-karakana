"""Monitor and optionally compact Karakana runtime CSV artifacts.

The monitor writes a compact JSON status file for operational dashboards and can
archive + cap append-only CSVs so long-running Freqtrade deployments do not
accumulate unbounded runtime files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import tempfile
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TIME_COLUMNS = (
    "timestamp",
    "date",
    "datetime",
    "open_date",
    "close_date",
    "signal_time",
    "entry_time",
    "exit_time",
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _count_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _read_head_tail(path: Path) -> tuple[list[str], dict[str, str] | None, dict[str, str] | None]:
    if not path.exists() or path.stat().st_size == 0:
        return [], None, None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        first: dict[str, str] | None = None
        last: dict[str, str] | None = None
        for row in reader:
            if first is None:
                first = dict(row)
            last = dict(row)
    return header, first, last


def _first_present(row: dict[str, str] | None, columns: tuple[str, ...]) -> str | None:
    if not row:
        return None
    for column in columns:
        value = row.get(column)
        if value not in (None, ""):
            return value
    return None


def _counter_for_column(path: Path, column: str, max_rows: int) -> dict[str, int]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = deque(reader, maxlen=max_rows) if max_rows > 0 else reader
        for row in rows:
            value = row.get(column)
            if value not in (None, ""):
                counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _summarize_csv(path: Path, *, count_tail_rows: int) -> dict[str, Any]:
    stat = path.stat() if path.exists() else None
    row_count = _count_rows(path)
    header, first, last = _read_head_tail(path)
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "row_count": row_count,
        "size_bytes": stat.st_size if stat else 0,
        "size_mb": round((stat.st_size if stat else 0) / (1024 * 1024), 4),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat() if stat else None,
        "columns": header,
        "oldest_timestamp": _first_present(first, TIME_COLUMNS),
        "newest_timestamp": _first_present(last, TIME_COLUMNS),
    }
    for column in ("action", "trade_position", "position", "side", "is_short", "armed"):
        if column in header:
            summary[f"{column}_counts_tail"] = _counter_for_column(path, column, count_tail_rows)
    if last:
        threshold_keys = [key for key in last if "threshold" in key.lower()]
        summary["latest_thresholds"] = {key: last.get(key) for key in threshold_keys}
        for key in ("armed", "regime_confidence", "long_count", "short_count"):
            if key in last:
                summary[f"latest_{key}"] = last.get(key)
    return summary


def _archive_csv(path: Path, archive_dir: Path, label: str) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{label}_{_utc_stamp()}.csv.gz"
    with path.open("rb") as src, gzip.open(archive_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return archive_path


def _cap_csv_rows(path: Path, max_rows: int, archive_dir: Path, label: str, archive: bool) -> dict[str, Any]:
    row_count = _count_rows(path)
    result: dict[str, Any] = {
        "path": str(path),
        "max_rows": max_rows,
        "before_rows": row_count,
        "capped": False,
        "archive_path": None,
    }
    if max_rows <= 0 or row_count <= max_rows or not path.exists():
        return result

    if archive:
        result["archive_path"] = str(_archive_csv(path, archive_dir, label))

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        tail_rows = deque(reader, maxlen=max_rows)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(path.parent)) as tmp:
        writer = csv.writer(tmp)
        if header:
            writer.writerow(header)
        writer.writerows(tail_rows)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)

    result["after_rows"] = _count_rows(path)
    result["capped"] = True
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals-csv", type=Path, default=Path("user_data/freqtrade_signals_karakana.csv"))
    parser.add_argument("--shadow-csv", type=Path, default=Path("user_data/reports/karakana_shadow_trade_history.csv"))
    parser.add_argument("--health-csv", type=Path, default=Path("user_data/reports/directional_strategy_health.csv"))
    parser.add_argument("--status-json", type=Path, default=Path("user_data/reports/karakana_runtime_status.json"))
    parser.add_argument("--archive-dir", type=Path, default=Path("user_data/reports/archive"))
    parser.add_argument("--signals-max-rows", type=int, default=0)
    parser.add_argument("--shadow-max-rows", type=int, default=0)
    parser.add_argument("--health-max-rows", type=int, default=0)
    parser.add_argument("--count-tail-rows", type=int, default=10000)
    parser.add_argument("--no-archive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = not args.no_archive
    compactions = {
        "signals": _cap_csv_rows(args.signals_csv, args.signals_max_rows, args.archive_dir, "freqtrade_signals_karakana", archive),
        "shadow_trades": _cap_csv_rows(args.shadow_csv, args.shadow_max_rows, args.archive_dir, "karakana_shadow_trade_history", archive),
        "health": _cap_csv_rows(args.health_csv, args.health_max_rows, args.archive_dir, "directional_strategy_health", archive),
    }
    status = {
        "timestamp": _utc_now(),
        "compactions": compactions,
        "files": {
            "signals": _summarize_csv(args.signals_csv, count_tail_rows=args.count_tail_rows),
            "shadow_trades": _summarize_csv(args.shadow_csv, count_tail_rows=args.count_tail_rows),
            "health": _summarize_csv(args.health_csv, count_tail_rows=args.count_tail_rows),
        },
    }
    args.status_json.parent.mkdir(parents=True, exist_ok=True)
    args.status_json.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
