"""Directional Strategy Health probe for Karakana trade histories."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from karakana.metrics import AlphaMomentumTracker, evaluate_distributional_geometry

BASE_LONG_THRESHOLD = 1.3
BASE_SHORT_THRESHOLD = 0.07


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(min(high, max(low, value)))


def _exp_decay_weights(size: int, decay: float) -> np.ndarray:
    raw = np.exp(-float(decay) * np.arange(size, dtype=float))
    return raw / raw.sum()


def _side_from_frame(frame: pd.DataFrame) -> pd.Series:
    if "position_status" in frame:
        return pd.to_numeric(frame["position_status"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    if "trade_position" in frame:
        return (
            frame["trade_position"]
            .astype(str)
            .str.lower()
            .map({"short": 1, "go_short": 1, "long": 0, "go_long": 0})
            .fillna(0)
            .astype(int)
        )
    raise ValueError("Expected position_status or trade_position in trade history")


def _evaluate_side(rewards: pd.Series) -> dict[str, float | int | bool]:
    values = pd.to_numeric(rewards, errors="coerce").dropna().astype(float)
    if values.empty:
        return {
            "count": 0,
            "reward_mean": 0.0,
            "reward_sum": 0.0,
            "win_rate": 0.0,
            "alpha_health": 0.0,
            "alpha_s_sil": 1.0,
            "normalized_structure_term": 1.0,
            "normalized_mse_cofactors": 1.0,
            "entropy": 1.0,
            "is_distributionally_stable": False,
        }
    positives = values[values > 0].to_numpy(dtype=float)
    negatives = np.abs(values[values <= 0].to_numpy(dtype=float))
    report = evaluate_distributional_geometry(
        positives,
        negatives,
        ranking_profile="balanced",
        regime_aware=True,
        bins=5,
    )
    alpha_s_sil = float(report.get("distributional_alpha_s_sil", report.get("alpha_s_sil", 1.0)))
    alpha_health = _clip(1.0 - alpha_s_sil)
    return {
        "count": int(len(values)),
        "reward_mean": float(values.mean()) if len(values) else 0.0,
        "reward_sum": float(values.sum()) if len(values) else 0.0,
        "win_rate": float((values > 0).mean()) if len(values) else 0.0,
        "alpha_health": alpha_health,
        "alpha_s_sil": alpha_s_sil,
        "normalized_structure_term": float(report.get("normalized_structure_term", 1.0)),
        "normalized_mse_cofactors": float(report.get("normalized_mse_cofactors", 1.0)),
        "entropy": float(report.get("entropy", 1.0)),
        "is_distributionally_stable": bool(report.get("is_distributionally_stable", False)),
    }


def _pressure_from_horizon(horizon: float, horizon_floor: float) -> float:
    if horizon <= 0:
        return 1.0
    if horizon >= 999:
        return 0.0
    return _clip((float(horizon_floor) - float(horizon)) / float(horizon_floor))


def _fragility(
    *,
    alpha_health: float,
    v_alpha: float,
    horizon: float,
    alpha_s_sil: float,
    floor: float,
    v_panic: float,
    horizon_floor: float,
    decay: float,
) -> float:
    components = np.asarray(
        [
            _clip((floor - alpha_health) / max(floor, 1e-8)),
            _clip(abs(min(v_alpha, 0.0)) / max(abs(v_panic), 1e-8)),
            _pressure_from_horizon(horizon, horizon_floor),
            _clip(alpha_s_sil),
        ],
        dtype=float,
    )
    weights = _exp_decay_weights(len(components), decay)
    return float(np.dot(weights, components))


def _project_regime(long_stats: dict[str, Any], short_stats: dict[str, Any]) -> tuple[str, float, float, float]:
    net_health = float(long_stats["alpha_health"]) - float(short_stats["alpha_health"])
    net_velocity = float(long_stats["v_alpha"]) - float(short_stats["v_alpha"])
    alpha_divergence = abs(net_health)
    velocity_divergence = abs(net_velocity)
    asymmetry = alpha_divergence / max(
        abs(float(long_stats["alpha_health"])) + abs(float(short_stats["alpha_health"])),
        1e-8,
    )
    if abs(net_health) < 0.05 and abs(net_velocity) < 0.005:
        regime = "transitioning"
    elif net_health > 0.05 and net_velocity >= 0:
        regime = "bullish"
    elif net_health < -0.05 and net_velocity <= 0:
        regime = "bearish"
    else:
        regime = "chaotic"
    confidence = _clip(0.5 * min(abs(net_velocity) / 0.02, 1.0) + 0.5 * asymmetry)
    return regime, confidence, alpha_divergence, velocity_divergence


def _effective_thresholds(
    *,
    regime: str,
    confidence: float,
    asymmetry: float,
    long_fragility: float,
    short_fragility: float,
    base_long: float,
    base_short: float,
) -> tuple[float, float]:
    long_threshold = base_long
    short_threshold = base_short
    if regime == "bullish":
        short_threshold += confidence * asymmetry * 0.50
        long_threshold += max(0.0, long_fragility - 0.30) * 0.50
    elif regime == "bearish":
        long_threshold += confidence * asymmetry * 1.50
        short_threshold += max(0.0, short_fragility - 0.30) * 0.10
    elif regime == "chaotic":
        long_threshold += 1.0
        short_threshold += 0.50
    else:
        long_threshold += max(0.0, long_fragility - 0.50) * 0.25
        short_threshold += max(0.0, short_fragility - 0.50) * 0.05
    return float(long_threshold), float(short_threshold)


def build_health_report(args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_csv(args.source)
    frame["reward"] = pd.to_numeric(frame["reward"], errors="coerce")
    frame["side"] = _side_from_frame(frame)

    long_tracker = AlphaMomentumTracker(window_size=args.tracker_window, alpha_ema=args.ema_weight)
    short_tracker = AlphaMomentumTracker(window_size=args.tracker_window, alpha_ema=args.ema_weight)

    rows: list[dict[str, Any]] = []
    for end in range(args.window_size, len(frame) + 1, args.window_step):
        start = end - args.window_size
        window = frame.iloc[start:end].copy()
        long_eval = _evaluate_side(window.loc[window["side"] == 0, "reward"])
        short_eval = _evaluate_side(window.loc[window["side"] == 1, "reward"])
        long_tracker.push(float(long_eval["alpha_health"]))
        short_tracker.push(float(short_eval["alpha_health"]))
        long_momentum = long_tracker.get_stats(v_panic=args.v_panic, floor=args.alpha_floor)
        short_momentum = short_tracker.get_stats(v_panic=args.v_panic, floor=args.alpha_floor)

        long_stats = {**long_eval, **long_momentum}
        short_stats = {**short_eval, **short_momentum}
        regime, confidence, alpha_divergence, velocity_divergence = _project_regime(long_stats, short_stats)
        asymmetry = alpha_divergence / max(
            abs(float(long_stats["alpha_health"])) + abs(float(short_stats["alpha_health"])),
            1e-8,
        )
        long_fragility = _fragility(
            alpha_health=float(long_stats["alpha_health"]),
            v_alpha=float(long_stats["v_alpha"]),
            horizon=float(long_stats["horizon"]),
            alpha_s_sil=float(long_stats["alpha_s_sil"]),
            floor=args.alpha_floor,
            v_panic=args.v_panic,
            horizon_floor=args.horizon_floor,
            decay=args.fragility_decay,
        )
        short_fragility = _fragility(
            alpha_health=float(short_stats["alpha_health"]),
            v_alpha=float(short_stats["v_alpha"]),
            horizon=float(short_stats["horizon"]),
            alpha_s_sil=float(short_stats["alpha_s_sil"]),
            floor=args.alpha_floor,
            v_panic=args.v_panic,
            horizon_floor=args.horizon_floor,
            decay=args.fragility_decay,
        )
        long_threshold, short_threshold = _effective_thresholds(
            regime=regime,
            confidence=confidence,
            asymmetry=asymmetry,
            long_fragility=long_fragility,
            short_fragility=short_fragility,
            base_long=args.base_long_threshold,
            base_short=args.base_short_threshold,
        )
        rows.append(
            {
                "start_row": start,
                "end_row": end,
                "rows": int(len(window)),
                "regime": regime,
                "regime_confidence": confidence,
                "alpha_divergence": alpha_divergence,
                "velocity_divergence": velocity_divergence,
                "asymmetry_index": asymmetry,
                "long_count": long_eval["count"],
                "short_count": short_eval["count"],
                "long_alpha_health": long_stats["alpha_health"],
                "short_alpha_health": short_stats["alpha_health"],
                "long_v_alpha": long_stats["v_alpha"],
                "short_v_alpha": short_stats["v_alpha"],
                "long_m_alpha": long_stats["m_alpha"],
                "short_m_alpha": short_stats["m_alpha"],
                "long_horizon": long_stats["horizon"],
                "short_horizon": short_stats["horizon"],
                "long_alpha_s_sil": long_stats["alpha_s_sil"],
                "short_alpha_s_sil": short_stats["alpha_s_sil"],
                "long_fragility": long_fragility,
                "short_fragility": short_fragility,
                "base_long_threshold": args.base_long_threshold,
                "base_short_threshold": args.base_short_threshold,
                "effective_long_threshold": long_threshold,
                "effective_short_threshold": short_threshold,
                "long_reward_mean": long_eval["reward_mean"],
                "short_reward_mean": short_eval["reward_mean"],
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("user_data/reports/karakana_shadow_trade_history.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("user_data/reports/directional_strategy_health.csv"))
    parser.add_argument(
        "--output-jsonl", type=Path, default=Path("user_data/reports/directional_strategy_health.jsonl")
    )
    parser.add_argument(
        "--summary-json", type=Path, default=Path("user_data/reports/directional_strategy_health_summary.json")
    )
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--window-step", type=int, default=256)
    parser.add_argument("--tracker-window", type=int, default=20)
    parser.add_argument("--ema-weight", type=float, default=0.3)
    parser.add_argument("--alpha-floor", type=float, default=0.10)
    parser.add_argument("--v-panic", type=float, default=-0.02)
    parser.add_argument("--horizon-floor", type=float, default=25.0)
    parser.add_argument("--fragility-decay", type=float, default=0.35)
    parser.add_argument("--base-long-threshold", type=float, default=BASE_LONG_THRESHOLD)
    parser.add_argument("--base-short-threshold", type=float, default=BASE_SHORT_THRESHOLD)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def write_health_outputs(args: argparse.Namespace) -> dict[str, Any]:
    report = build_health_report(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_csv, index=False)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in report.to_dict(orient="records"):
            handle.write(json.dumps(row, allow_nan=True) + "\n")

    summary = {
        "windows": int(len(report)),
        "latest": report.iloc[-1].to_dict() if not report.empty else None,
        "regime_counts": report["regime"].value_counts().to_dict() if not report.empty else {},
        "mean_effective_long_threshold": float(report["effective_long_threshold"].mean())
        if not report.empty
        else math.nan,
        "mean_effective_short_threshold": float(report["effective_short_threshold"].mean())
        if not report.empty
        else math.nan,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"Wrote {args.output_csv} ({len(report)} windows)")
    print(f"Wrote {args.output_jsonl}")
    print(f"Wrote {args.summary_json}")
    print(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def main() -> int:
    args = parse_args()
    last_mtime: float | None = None
    while True:
        if not args.source.exists():
            if not args.watch:
                raise FileNotFoundError(f"Health source not found: {args.source}")
            print(f"Waiting for health source: {args.source}", flush=True)
        else:
            mtime = args.source.stat().st_mtime
            if last_mtime != mtime:
                try:
                    write_health_outputs(args)
                    last_mtime = mtime
                except Exception as exc:
                    if not args.watch:
                        raise
                    print(f"Could not build health report yet: {exc}", flush=True)
            elif not args.watch:
                write_health_outputs(args)
        if not args.watch:
            break
        time.sleep(max(1, int(args.poll_seconds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
