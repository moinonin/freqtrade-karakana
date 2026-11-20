"""Generate virtual Karakana trades for directional Strategy Health."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd

from karakana.runtime.policy import (
    DEFAULT_LONG_ACTION_MARGIN,
    DEFAULT_SHORT_ACTION_MARGIN,
    KarakanaOfflinePolicy,
)

ACTION_BY_SIDE = {0: "go_long", 1: "go_short"}
TRADE_POSITION_BY_SIDE = {0: "go_long", 1: "go_short"}


def _load_state(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"last_source_row": 0}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"last_source_row": 0}
    return {"last_source_row": int(payload.get("last_source_row", 0))}


def _save_state(path: Path, payload: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_model_dir(policy_root: Path) -> Path:
    latest_path = policy_root / "latest.json"
    if latest_path.exists():
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
        latest_dir = payload.get("latest_export_dir")
        if latest_dir:
            return Path(latest_dir)
    if (policy_root / "best_policy.pt").exists():
        return policy_root
    raise FileNotFoundError(
        f"Could not resolve Karakana policy from {policy_root}. Expected best_policy.pt or latest.json."
    )


def _parse_state(value: Any) -> tuple[float, float, int]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if len(parsed) < 3:
        raise ValueError(f"Expected state with at least three values, received {value!r}")
    return float(parsed[0]), float(parsed[1]), int(float(parsed[2]))


def _series_value(row: pd.Series, names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return None


def _extract_market_state(row: pd.Series) -> tuple[float, float, int]:
    if "state" in row and pd.notna(row["state"]):
        return _parse_state(row["state"])

    ask_value = _series_value(row, ("ask", "Ask", "best_ask"))
    bid_value = _series_value(row, ("bid", "Bid", "best_bid"))
    sma_value = _series_value(row, ("sma-compare", "sma_compare", "sma_compare_flag", "sma-trend-flag"))
    if ask_value is None or bid_value is None or sma_value is None:
        raise ValueError("Source rows must contain either state or ask/bid/sma-compare compatible columns")
    return float(ask_value), float(bid_value), int(float(sma_value))


def _mid_price(row: pd.Series, ask: float, bid: float) -> float:
    for column in ("reward-price", "reward_price", "close", "Close", "mid-price", "mid_price", "mid"):
        value = _series_value(row, (column,))
        if value is not None:
            price = float(value)
            if math.isfinite(price) and price > 0.0:
                return price
    price = (float(ask) + float(bid)) / 2.0
    if not math.isfinite(price) or price <= 0.0:
        raise ValueError(f"Invalid mid price: {price!r}")
    return price


def _reward_for_side(side: int, entry_price: float, exit_price: float, fee_bps: float) -> float:
    raw_return = (exit_price - entry_price) / entry_price
    if int(side) == 1:
        raw_return *= -1.0
    return float(raw_return - (float(fee_bps) / 10_000.0))


def _next_state(frame: pd.DataFrame, exit_index: int, side: int) -> str:
    ask, bid, sma_compare = _extract_market_state(frame.iloc[exit_index])
    return str((float(ask), float(bid), int(sma_compare), int(side)))


def _safe_float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def build_shadow_trades(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not args.source.exists():
        raise FileNotFoundError(f"Shadow source not found: {args.source}")

    source = pd.read_csv(args.source)
    if source.empty:
        return pd.DataFrame(), {"source_rows": 0, "virtual_trades": 0}

    state = {"last_source_row": 0} if args.reset_state else _load_state(args.state_path)
    start_row = max(0, int(state["last_source_row"]))
    horizon = int(args.horizon_rows)
    has_pair_col = "pair" in source.columns
    exit_index_by_row: dict[int, int] | None = None
    if has_pair_col:
        exit_index_by_row = {}
        pair_groups: dict[str, list[int]] = {}
        for idx, pair_val in enumerate(source["pair"].astype(str).tolist()):
            pair_groups.setdefault(pair_val, []).append(idx)
        for idxs in pair_groups.values():
            for pos in range(0, len(idxs) - horizon):
                exit_index_by_row[idxs[pos]] = idxs[pos + horizon]
        end_limit = len(source)
        if args.max_rows is not None:
            end_limit = min(end_limit, start_row + int(args.max_rows))
        last_eligible_row = start_row
        while last_eligible_row < end_limit and last_eligible_row in exit_index_by_row:
            last_eligible_row += 1
    else:
        last_eligible_row = max(0, len(source) - horizon)
        if args.max_rows is not None:
            last_eligible_row = min(last_eligible_row, start_row + int(args.max_rows))

    if start_row >= last_eligible_row:
        return pd.DataFrame(), {
            "source_rows": int(len(source)),
            "start_row": start_row,
            "last_eligible_row": last_eligible_row,
            "virtual_trades": 0,
            "reason": "no_new_horizon_complete_rows",
        }

    model_dir = _resolve_model_dir(args.policy_root)
    policy = KarakanaOfflinePolicy(
        model_dir,
        checkpoint_type=args.checkpoint_type,
        side_consistent=True,
        min_long_action_margin=args.long_threshold,
        min_short_action_margin=args.short_threshold,
        threshold_governor_enabled=False,
    )

    rows: list[dict[str, Any]] = []
    evaluated = 0
    margin_failures = {0: 0, 1: 0}
    admitted = {0: 0, 1: 0}

    for source_index in range(start_row, last_eligible_row):
        row = source.iloc[source_index]
        exit_index = exit_index_by_row[source_index] if exit_index_by_row is not None else source_index + horizon
        exit_row = source.iloc[exit_index]
        ask, bid, sma_compare = _extract_market_state(row)
        entry_mid = _mid_price(row, ask, bid)
        exit_ask, exit_bid, _exit_sma = _extract_market_state(exit_row)
        exit_mid = _mid_price(exit_row, exit_ask, exit_bid)

        candidates: list[tuple[int, dict[str, Any], float, float]] = []
        for side in (0, 1):
            evaluated += 1
            result = policy.predict(ask=ask, bid=bid, sma_compare=sma_compare, is_short=side)
            required_margin = args.short_threshold if side else args.long_threshold
            margin = _safe_float(result, "action_margin")
            action = str(result.get("action", "do_nothing"))
            if action != ACTION_BY_SIDE[side] or margin < required_margin:
                margin_failures[side] += 1
                continue
            candidates.append((side, result, margin, float(required_margin)))

        if args.admission_mode == "winner" and len(candidates) > 1:
            candidates = [max(candidates, key=lambda item: item[2] - item[3])]

        for side, result, margin, required_margin in candidates:
            reward = _reward_for_side(side, entry_mid, exit_mid, args.fee_bps)
            admitted[side] += 1
            rows.append(
                {
                    "source_row": int(source_index),
                    "exit_source_row": int(exit_index),
                    "episode_number": int(source_index),
                    "state": str((float(ask), float(bid), int(sma_compare), int(side))),
                    "next_state": _next_state(source, exit_index, side),
                    "action": ACTION_BY_SIDE[side],
                    "reward": reward,
                    "position_status": int(side),
                    "trade_position": TRADE_POSITION_BY_SIDE[side],
                    "entry_mid": entry_mid,
                    "exit_mid": exit_mid,
                    "horizon_rows": int(args.horizon_rows),
                    "q_margin": margin,
                    "required_q_margin": float(required_margin),
                    "directional_q": _safe_float(result, "directional_q"),
                    "do_nothing_q": _safe_float(result, "do_nothing_q"),
                    "long_threshold": float(args.long_threshold),
                    "short_threshold": float(args.short_threshold),
                    "policy_dir": str(model_dir),
                }
            )

    shadow = pd.DataFrame(rows)
    summary = {
        "source": str(args.source),
        "policy_dir": str(model_dir),
        "source_rows": int(len(source)),
        "start_row": int(start_row),
        "end_row": int(last_eligible_row),
        "horizon_rows": int(args.horizon_rows),
        "evaluated_candidates": int(evaluated),
        "virtual_trades": int(len(shadow)),
        "long_virtual_trades": int(admitted[0]),
        "short_virtual_trades": int(admitted[1]),
        "long_margin_failures": int(margin_failures[0]),
        "short_margin_failures": int(margin_failures[1]),
        "admission_mode": args.admission_mode,
        "long_threshold": float(args.long_threshold),
        "short_threshold": float(args.short_threshold),
    }
    if not shadow.empty:
        summary.update(
            {
                "reward_mean": float(shadow["reward"].mean()),
                "reward_sum": float(shadow["reward"].sum()),
                "long_reward_mean": float(shadow.loc[shadow["position_status"] == 0, "reward"].mean())
                if int(admitted[0])
                else 0.0,
                "short_reward_mean": float(shadow.loc[shadow["position_status"] == 1, "reward"].mean())
                if int(admitted[1])
                else 0.0,
                "q_margin_p50": float(shadow["q_margin"].quantile(0.50)),
                "q_margin_p90": float(shadow["q_margin"].quantile(0.90)),
            }
        )
    return shadow, summary


def write_outputs(args: argparse.Namespace, shadow: pd.DataFrame, summary: dict[str, Any]) -> None:
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not shadow.empty:
        append = bool(args.append and args.output_csv.exists() and not args.reset_state)
        shadow.to_csv(args.output_csv, mode="a" if append else "w", header=not append, index=False)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary.get("end_row") is not None:
        _save_state(args.state_path, {"last_source_row": int(summary["end_row"])})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("user_data/freqtrade_signals_karakana.csv"))
    parser.add_argument("--policy-root", type=Path, default=Path("user_data/karakana_policy/watch"))
    parser.add_argument("--checkpoint-type", choices=["best", "latest"], default="best")
    parser.add_argument("--output-csv", type=Path, default=Path("user_data/reports/karakana_shadow_trade_history.csv"))
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("user_data/reports/karakana_shadow_trade_history_summary.json"),
    )
    parser.add_argument("--state-path", type=Path, default=Path("user_data/reports/karakana_shadow_trader_state.json"))
    parser.add_argument("--horizon-rows", type=int, default=12)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--long-threshold", type=float, default=DEFAULT_LONG_ACTION_MARGIN)
    parser.add_argument("--short-threshold", type=float, default=DEFAULT_SHORT_ACTION_MARGIN)
    parser.add_argument(
        "--admission-mode",
        choices=["winner", "both"],
        default="winner",
        help="winner admits only the strongest passing side per row; both admits every passing side.",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--append", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    while True:
        try:
            shadow, summary = build_shadow_trades(args)
            write_outputs(args, shadow, summary)
            logging.info(
                "Shadow Karakana: trades=%s long=%s short=%s reward_mean=%s q_p50=%s q_p90=%s source_rows=%s",
                summary.get("virtual_trades", 0),
                summary.get("long_virtual_trades", 0),
                summary.get("short_virtual_trades", 0),
                summary.get("reward_mean", 0.0),
                summary.get("q_margin_p50", 0.0),
                summary.get("q_margin_p90", 0.0),
                summary.get("source_rows", 0),
            )
        except Exception as exc:
            if not args.watch:
                raise
            logging.warning("Shadow Karakana waiting for ready source/policy: %s", exc)
        if not args.watch:
            return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
