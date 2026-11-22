"""Periodic live Karakana threshold calibration scheduler.

This wrapper is designed for the external Freqtrade root.  It performs safe
preflight checks, runs the live threshold sweep/apply script, and optionally
restarts Karakana sidecars through the Docker Unix socket.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import json
import shutil
import logging
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str):
        super().__init__("localhost")
        self.socket_path = socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        self.sock = sock


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _parse_int_values(spec: str) -> list[int]:
    raw = str(spec).strip()
    if not raw:
        return []
    if "," in raw:
        return sorted({int(float(part.strip())) for part in raw.split(",") if part.strip()})
    parts = raw.split(":")
    if len(parts) == 1:
        return [int(float(raw))]
    if len(parts) != 3:
        raise ValueError(f"Expected comma list or start:stop:step range, received {spec!r}")
    start, stop, step = (int(float(part)) for part in parts)
    if step <= 0:
        raise ValueError("interval step must be positive")
    values = []
    current = start
    while current <= stop:
        values.append(current)
        current += step
    return values


def _count_csv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("rb") as handle:
        rows = sum(1 for _ in handle)
    return max(0, rows - 1)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_fingerprint(policy_root: Path) -> dict[str, Any] | None:
    checkpoint = policy_root / "best_policy.pt"
    preprocessing = policy_root / "offline_preprocessing.json"
    if not checkpoint.exists() or not preprocessing.exists():
        return None
    files = {}
    for path in (checkpoint, preprocessing):
        stat = path.stat()
        files[path.name] = {
            "path": str(path),
            "size_bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256": _sha256_file(path),
        }
    combined = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"policy_root": str(policy_root), "fingerprint": combined, "files": files}


def _archive_path(path: Path, archive_dir: Path, reason: str) -> str | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    safe_name = path.name.replace(".csv", "")
    archive_path = archive_dir / f"{safe_name}_{reason}_{_utc_stamp()}.csv.gz"
    with path.open("rb") as src, gzip.open(archive_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return str(archive_path)


def _unlink_if_exists(path: Path) -> bool:
    if not path.exists():
        return False
    path.unlink()
    return True


def _reset_policy_derived_artifacts(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    archive_dir = args.policy_reset_archive_dir
    archives = {
        "shadow_csv": _archive_path(args.monitor_shadow_csv, archive_dir, reason),
        "health_csv": _archive_path(args.monitor_health_csv, archive_dir, reason),
    }
    removed = {
        "shadow_csv": _unlink_if_exists(args.monitor_shadow_csv),
        "health_csv": _unlink_if_exists(args.monitor_health_csv),
        "shadow_state": _unlink_if_exists(args.policy_reset_shadow_state),
        "shadow_summary": _unlink_if_exists(args.policy_reset_shadow_summary),
    }
    return {
        "reason": reason,
        "archives": archives,
        "removed": removed,
    }


def _maybe_reset_on_policy_change(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any] | None:
    if not args.policy_reset_enabled:
        return None
    current = _policy_fingerprint(args.policy_root)
    if current is None:
        return {"action": "skip", "reason": "missing_policy_artifacts"}

    previous = state.get("policy_fingerprint")
    changed = bool(previous and previous.get("fingerprint") != current["fingerprint"])
    first_seen = previous is None
    force = bool(args.policy_reset_force)
    should_reset = force or changed or (first_seen and args.policy_reset_on_first_seen)

    result: dict[str, Any] = {
        "action": "baseline",
        "changed": changed,
        "first_seen": first_seen,
        "force": force,
        "current": current,
    }
    if should_reset:
        reason = "policy_force_reset" if force else "policy_first_seen" if first_seen else "policy_changed"
        result.update(_reset_policy_derived_artifacts(args, reason))
        result["action"] = "reset_policy_derived_artifacts"
        logging.warning("Policy artifact change detected; reset policy-derived artifacts: %s", reason)

    state["policy_fingerprint"] = current
    state["policy_fingerprint_checked_at"] = _utc_now()
    state["last_policy_reset"] = result if result["action"] == "reset_policy_derived_artifacts" else state.get("last_policy_reset")
    _write_json(args.state_json, state)
    return result


def _setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
    )


def _choose_interval(args: argparse.Namespace, state: dict[str, Any], current_rows: int) -> int:
    if args.interval_seconds is not None:
        return int(args.interval_seconds)
    candidates = _parse_int_values(args.interval_seconds_values)
    if not candidates:
        return 7200

    previous_rows = int(state.get("last_signal_rows") or 0)
    previous_time = float(state.get("last_run_epoch") or 0.0)
    now = time.time()
    elapsed = max(0.0, now - previous_time)
    row_delta = max(0, current_rows - previous_rows)
    if elapsed <= 0.0 or row_delta <= 0:
        return int(args.default_interval_seconds)

    rows_per_second = row_delta / elapsed
    target_seconds = int(args.min_new_rows_per_run / max(rows_per_second, 1e-9))
    for candidate in candidates:
        if candidate >= target_seconds:
            return int(candidate)
    return int(candidates[-1])


def _restart_sidecars(args: argparse.Namespace) -> list[str]:
    """Recreate sidecar containers via 'docker compose up -d' so new env vars take effect.

    This replaces the old simple restart which only re-executed the existing container
    config. Recreating is required because thresholds are passed as env vars / CLI args
    at container create time.
    """
    if not args.restart_sidecars:
        return []
    if not args.compose_file.exists():
        raise FileNotFoundError(f"Compose file not found: {args.compose_file}")
    if not args.docker_socket.exists():
        raise FileNotFoundError(f"Docker socket not mounted: {args.docker_socket}")

    containers = [c.strip() for c in args.sidecar_containers.split(",") if c.strip()]
    if not containers:
        return []

    # Use docker compose (not the raw socket) so the updated compose file is read
    # and containers are recreated with new env vars / command args.
    import subprocess
    cmd = [
        "docker",
        "--host", f"unix://{args.docker_socket}",
        "compose",
        "-f", str(args.compose_file),
        "up", "-d",
        "--no-deps",
    ] + containers
    logging.info("Recreating sidecars with updated config: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"docker compose up failed: {result.stderr.strip() or result.stdout.strip()}")
    return containers


def _build_sweep_command(args: argparse.Namespace) -> list[str]:
    command = [
        "python",
        str(args.sweep_script),
        "--repo-root",
        str(args.repo_root),
        "--source",
        str(args.source),
        "--policy-root",
        str(args.policy_root),
        "--long-thresholds",
        args.long_thresholds,
        "--short-thresholds",
        args.short_thresholds,
        "--health-window-sizes",
        args.health_window_sizes,
        "--health-window-steps",
        args.health_window_steps,
        "--health-poll-seconds-values",
        args.health_poll_seconds_values,
        "--tail-rows",
        str(args.tail_rows),
        "--horizon-rows",
        str(args.horizon_rows),
        "--horizon-mode",
        args.horizon_mode,
        "--fee-bps",
        str(args.fee_bps),
        "--min-trades",
        str(args.min_trades),
        "--min-side-trades",
        str(args.min_side_trades),
        "--win-rate-weight",
        str(args.win_rate_weight),
        "--balance-weight",
        str(args.balance_weight),
        "--health-weight",
        str(args.health_weight),
        "--poll-seconds-weight",
        str(args.poll_seconds_weight),
        "--output-csv",
        str(args.output_csv),
        "--output-json",
        str(args.output_json),
    ]
    if args.max_rows is not None:
        command.extend(["--max-rows", str(args.max_rows)])
    if args.apply_compose:
        command.extend(["--compose-file", str(args.compose_file), "--apply-compose"])
    return command


def _build_monitor_command(args: argparse.Namespace) -> list[str]:
    return [
        "python",
        str(args.monitor_script),
        "--signals-csv",
        str(args.source),
        "--shadow-csv",
        str(args.monitor_shadow_csv),
        "--health-csv",
        str(args.monitor_health_csv),
        "--status-json",
        str(args.monitor_status_json),
        "--archive-dir",
        str(args.monitor_archive_dir),
        "--signals-max-rows",
        str(args.monitor_signals_max_rows),
        "--shadow-max-rows",
        str(args.monitor_shadow_max_rows),
        "--health-max-rows",
        str(args.monitor_health_max_rows),
        "--count-tail-rows",
        str(args.monitor_count_tail_rows),
    ]


def _run_monitor(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.monitor_enabled:
        return None
    if not args.monitor_script.exists():
        logging.warning("Skipping runtime monitor; script missing: %s", args.monitor_script)
        return {"returncode": None, "reason": f"missing_monitor_script:{args.monitor_script}"}
    command = _build_monitor_command(args)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    payload = {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }
    if result.returncode != 0:
        logging.warning("Runtime monitor failed: %s", result.stderr[-1000:])
    return payload


def _preflight(args: argparse.Namespace) -> tuple[bool, str, int]:
    rows = _count_csv_rows(args.source)
    if rows < args.min_signal_rows:
        return False, f"insufficient_signal_rows:{rows}<{args.min_signal_rows}", rows
    checkpoint = args.policy_root / "best_policy.pt"
    preprocessing = args.policy_root / "offline_preprocessing.json"
    if not checkpoint.exists():
        return False, f"missing_policy:{checkpoint}", rows
    if not preprocessing.exists():
        return False, f"missing_preprocessing:{preprocessing}", rows
    if args.apply_compose and not args.compose_file.exists():
        return False, f"missing_compose:{args.compose_file}", rows
    return True, "ready", rows


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    state = _load_json(args.state_json)
    policy_reset_result = _maybe_reset_on_policy_change(args, state)
    monitor_result = _run_monitor(args)
    ready, reason, signal_rows = _preflight(args)
    decision: dict[str, Any] = {
        "timestamp": _utc_now(),
        "signal_rows": signal_rows,
        "ready": ready,
        "reason": reason,
        "apply_compose": bool(args.apply_compose),
        "restart_sidecars": bool(args.restart_sidecars),
    }
    if policy_reset_result is not None:
        decision["policy_reset"] = policy_reset_result
    if monitor_result is not None:
        decision["runtime_monitor"] = monitor_result
    if not ready:
        logging.info("Skipping threshold scheduler run: %s", reason)
        decision["action"] = "skip"
        state.update(decision)
        state["last_signal_rows"] = signal_rows
        state["last_run_epoch"] = time.time()
        state["next_interval_seconds"] = _choose_interval(args, state, signal_rows)
        _write_json(args.state_json, state)
        return decision

    command = _build_sweep_command(args)
    logging.info("Running threshold sweep: %s", " ".join(command))
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    decision["sweep_returncode"] = result.returncode
    decision["sweep_stdout_tail"] = result.stdout[-4000:]
    decision["sweep_stderr_tail"] = result.stderr[-4000:]
    if result.returncode != 0:
        logging.error("Threshold sweep failed: %s", result.stderr[-2000:])
        decision["action"] = "sweep_failed"
        state.update(decision)
        state["last_signal_rows"] = signal_rows
        state["last_run_epoch"] = time.time()
        state["next_interval_seconds"] = _choose_interval(args, state, signal_rows)
        _write_json(args.state_json, state)
        return decision

    sweep_summary = _load_json(args.output_json)
    decision["viable"] = bool(sweep_summary.get("viable"))
    decision["recommended"] = sweep_summary.get("recommended")
    if not decision["viable"]:
        logging.info("Sweep produced no viable recommendation; no restart.")
        decision["action"] = "no_viable_recommendation"
        state.update(decision)
        state["last_signal_rows"] = signal_rows
        state["last_run_epoch"] = time.time()
        state["next_interval_seconds"] = _choose_interval(args, state, signal_rows)
        _write_json(args.state_json, state)
        return decision

    if not args.apply_compose:
        logging.info("Viable recommendation written; apply disabled, so no restart.")
        decision["action"] = "recommendation_only"
        state.update(decision)
        state["last_signal_rows"] = signal_rows
        state["last_run_epoch"] = time.time()
        state["next_interval_seconds"] = _choose_interval(args, state, signal_rows)
        _write_json(args.state_json, state)
        return decision

    restarted = _restart_sidecars(args)
    decision["restarted_sidecars"] = restarted
    decision["action"] = "applied_and_restarted" if restarted else "applied"
    state.update(decision)
    state["last_signal_rows"] = signal_rows
    state["last_run_epoch"] = time.time()
    state["next_interval_seconds"] = _choose_interval(args, state, signal_rows)
    _write_json(args.state_json, state)
    logging.info("Threshold scheduler decision: %s", decision["action"])
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("rl-trade-trainer"))
    parser.add_argument(
        "--sweep-script",
        type=Path,
        default=Path("rl-trade-trainer/artifacts/karakana_analysis/sweep_live_shadow_thresholds.py"),
    )
    parser.add_argument("--source", type=Path, default=Path("user_data/freqtrade_signals_karakana.csv"))
    parser.add_argument("--policy-root", type=Path, default=Path("user_data/karakana_policy/watch"))
    parser.add_argument("--compose-file", type=Path, default=Path("docker-compose-karakana.yml"))
    parser.add_argument("--output-csv", type=Path, default=Path("user_data/reports/karakana_live_threshold_sweep.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("user_data/reports/karakana_live_threshold_sweep.json"))
    parser.add_argument("--state-json", type=Path, default=Path("user_data/reports/karakana_threshold_scheduler_state.json"))
    parser.add_argument("--log-file", type=Path, default=Path("user_data/reports/karakana_threshold_scheduler.log"))
    parser.add_argument(
        "--monitor-script",
        type=Path,
        default=Path("rl-trade-trainer/artifacts/karakana_analysis/monitor_karakana_runtime.py"),
    )
    parser.add_argument("--monitor-enabled", action="store_true")
    parser.add_argument("--monitor-shadow-csv", type=Path, default=Path("user_data/reports/karakana_shadow_trade_history.csv"))
    parser.add_argument("--monitor-health-csv", type=Path, default=Path("user_data/reports/directional_strategy_health.csv"))
    parser.add_argument("--monitor-status-json", type=Path, default=Path("user_data/reports/karakana_runtime_status.json"))
    parser.add_argument("--monitor-archive-dir", type=Path, default=Path("user_data/reports/archive"))
    parser.add_argument("--monitor-signals-max-rows", type=int, default=250000)
    parser.add_argument("--monitor-shadow-max-rows", type=int, default=100000)
    parser.add_argument("--monitor-health-max-rows", type=int, default=20000)
    parser.add_argument("--monitor-count-tail-rows", type=int, default=10000)
    parser.add_argument("--policy-reset-enabled", action="store_true")
    parser.add_argument("--policy-reset-force", action="store_true")
    parser.add_argument("--policy-reset-on-first-seen", action="store_true")
    parser.add_argument("--policy-reset-archive-dir", type=Path, default=Path("user_data/reports/archive"))
    parser.add_argument(
        "--policy-reset-shadow-state",
        type=Path,
        default=Path("user_data/reports/karakana_shadow_trader_state.json"),
    )
    parser.add_argument(
        "--policy-reset-shadow-summary",
        type=Path,
        default=Path("user_data/reports/karakana_shadow_trade_history_summary.json"),
    )
    parser.add_argument("--min-signal-rows", type=int, default=20000)
    parser.add_argument("--min-new-rows-per-run", type=int, default=25000)
    parser.add_argument("--tail-rows", type=int, default=50000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--long-thresholds", default="1.0:3.0:0.1")
    parser.add_argument("--short-thresholds", default="0.05:1.5:0.05")
    parser.add_argument("--health-window-sizes", default="128,256,512,1024")
    parser.add_argument("--health-window-steps", default="32,64,128,256")
    parser.add_argument("--health-poll-seconds-values", default="60")
    parser.add_argument("--horizon-rows", type=int, default=12)
    parser.add_argument("--horizon-mode", choices=["pair", "row"], default="pair")
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=0.0,
        help="Round-trip trading fee in basis points passed to the sweep (0 = fee-free).",
    )
    parser.add_argument("--min-trades", type=int, default=128)
    parser.add_argument("--min-side-trades", type=int, default=8)
    parser.add_argument("--win-rate-weight", type=float, default=0.1)
    parser.add_argument("--balance-weight", type=float, default=0.1)
    parser.add_argument("--health-weight", type=float, default=0.05)
    parser.add_argument("--poll-seconds-weight", type=float, default=0.0)
    parser.add_argument("--interval-seconds", type=int, default=None)
    parser.add_argument("--default-interval-seconds", type=int, default=7200)
    parser.add_argument("--interval-seconds-values", default="3600,7200,10800,21600")
    parser.add_argument("--apply-compose", action="store_true")
    parser.add_argument("--restart-sidecars", action="store_true")
    parser.add_argument("--sidecar-containers", default="karakana-shadow-trader,karakana-shadow-health")
    parser.add_argument("--docker-socket", type=Path, default=Path("/var/run/docker.sock"))
    parser.add_argument("--restart-timeout-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--run-on-start", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_file)
    logging.info("Starting Karakana threshold scheduler")

    if args.once or args.run_on_start:
        decision = run_once(args)
        if args.once:
            print(json.dumps(decision, indent=2))
            return

    while True:
        state = _load_json(args.state_json)
        signal_rows = _count_csv_rows(args.source)
        interval = _choose_interval(args, state, signal_rows)
        logging.info("Sleeping %s seconds before next threshold scheduler run", interval)
        time.sleep(max(1, int(interval)))
        run_once(args)


if __name__ == "__main__":
    main()
