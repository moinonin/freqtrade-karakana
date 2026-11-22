"""Sweep live Karakana shadow thresholds and optionally patch compose defaults."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _ensure_karakana_importable(repo_root: Path) -> None:
    import sys

    karakana_root = repo_root / "karakana"
    if karakana_root.exists():
        sys.path.insert(0, str(karakana_root))


def _float_range(spec: str) -> list[float]:
    raw = str(spec).strip()
    if not raw:
        raise ValueError("range spec cannot be empty")
    if "," in raw:
        return [float(part.strip()) for part in raw.split(",") if part.strip()]
    parts = raw.split(":")
    if len(parts) == 1:
        return [float(raw)]
    if len(parts) != 3:
        raise ValueError(f"Expected comma list or start:stop:step range, received {spec!r}")
    start, stop, step = (float(part) for part in parts)
    if step <= 0:
        raise ValueError("range step must be positive")
    values: list[float] = []
    current = start
    epsilon = step / 10.0
    while current <= stop + epsilon:
        values.append(round(current, 12))
        current += step
    return values


def _int_range(spec: str) -> list[int]:
    values = _float_range(spec)
    return [int(round(value)) for value in values]


def _series_value(row: pd.Series, names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return None


def _extract_market_state(row: pd.Series) -> tuple[float, float, int]:
    ask_value = _series_value(row, ("ask", "Ask", "best_ask"))
    bid_value = _series_value(row, ("bid", "Bid", "best_bid"))
    sma_value = _series_value(row, ("sma-compare", "sma_compare", "sma_compare_flag", "sma-trend-flag"))
    if ask_value is None or bid_value is None or sma_value is None:
        raise ValueError("Signal rows must contain ask, bid, and sma-compare compatible columns")
    return float(ask_value), float(bid_value), int(float(sma_value))


def _reward_price(row: pd.Series, ask: float, bid: float) -> float:
    for name in ("reward-price", "mark-price", "close", "Close", "mid-price", "mid_price", "mid"):
        value = _series_value(row, (name,))
        if value is None:
            continue
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and price > 0.0:
            return price
    feature_mid = (float(ask) + float(bid)) / 2.0
    if math.isfinite(feature_mid) and feature_mid > 0.0:
        return feature_mid
    raise ValueError("Could not resolve positive reward price")


def _reward_for_side(side: int, entry_price: float, exit_price: float, fee_bps: float) -> float:
    raw_return = (exit_price - entry_price) / entry_price
    if int(side) == 1:
        raw_return *= -1.0
    return float(raw_return - (float(fee_bps) / 10_000.0))


def _safe_float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _load_signals(path: Path, max_rows: int | None, tail_rows: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if tail_rows:
        frame = frame.tail(int(tail_rows)).copy()
    if max_rows:
        frame = frame.head(int(max_rows)).copy()
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No signal rows found in {path}")
    return frame


def _exit_index_by_row(frame: pd.DataFrame, horizon_rows: int) -> dict[int, int]:
    """Map each row to the row `horizon_rows` later for the same pair.

    Replaces a per-row forward scan (O(n^2) pandas indexing) with one grouping
    pass; a 50k-row sweep otherwise takes hours instead of minutes.
    """
    groups: dict[str, list[int]] = {}
    for idx, pair in enumerate(frame["pair"].astype(str).tolist()):
        groups.setdefault(pair, []).append(idx)
    mapping: dict[int, int] = {}
    for idxs in groups.values():
        for pos in range(0, len(idxs) - horizon_rows):
            mapping[idxs[pos]] = idxs[pos + horizon_rows]
    return mapping


@dataclass(frozen=True)
class Candidate:
    source_index: int
    side: int
    margin: float
    reward: float


def _build_candidates(args: argparse.Namespace, frame: pd.DataFrame) -> list[Candidate]:
    from karakana.runtime.policy import KarakanaOfflinePolicy

    policy = KarakanaOfflinePolicy(
        args.policy_root,
        checkpoint_type=args.checkpoint_type,
        side_consistent=True,
        min_long_action_margin=0.0,
        min_short_action_margin=0.0,
        threshold_governor_enabled=False,
    )
    candidates: list[Candidate] = []
    exit_index_by_row = (
        _exit_index_by_row(frame, int(args.horizon_rows))
        if args.horizon_mode == "pair" and "pair" in frame
        else None
    )
    for source_index in range(len(frame)):
        row = frame.iloc[source_index]
        if exit_index_by_row is not None:
            exit_index = exit_index_by_row.get(source_index)
        else:
            exit_index = source_index + int(args.horizon_rows)
        if exit_index is None or exit_index >= len(frame):
            continue

        try:
            ask, bid, sma_compare = _extract_market_state(row)
            entry_price = _reward_price(row, ask, bid)
            exit_ask, exit_bid, _ = _extract_market_state(frame.iloc[exit_index])
            exit_price = _reward_price(frame.iloc[exit_index], exit_ask, exit_bid)
        except Exception:
            continue

        source_candidates: list[Candidate] = []
        for side in (0, 1):
            result = policy.predict(ask=ask, bid=bid, sma_compare=sma_compare, is_short=side)
            action = str(result.get("directional_action", ""))
            expected_action = "go_short" if side else "go_long"
            if action != expected_action:
                continue
            margin = _safe_float(result, "action_margin")
            reward = _reward_for_side(side, entry_price, exit_price, args.fee_bps)
            source_candidates.append(Candidate(source_index, side, margin, reward))
        candidates.extend(source_candidates)
    return candidates


def _summarize_selected(selected: list[Candidate], long_threshold: float, short_threshold: float) -> dict[str, Any]:
    rewards = [item.reward for item in selected]
    long_rewards = [item.reward for item in selected if item.side == 0]
    short_rewards = [item.reward for item in selected if item.side == 1]
    total = len(selected)
    long_count = len(long_rewards)
    short_count = len(short_rewards)
    balance = min(long_count, short_count) / max(long_count, short_count) if max(long_count, short_count) else 0.0
    win_rate = sum(1 for value in rewards if value > 0.0) / total if total else 0.0
    long_win_rate = sum(1 for value in long_rewards if value > 0.0) / long_count if long_count else 0.0
    short_win_rate = sum(1 for value in short_rewards if value > 0.0) / short_count if short_count else 0.0
    reward_mean = float(sum(rewards) / total) if total else 0.0
    long_reward_mean = float(sum(long_rewards) / long_count) if long_count else 0.0
    short_reward_mean = float(sum(short_rewards) / short_count) if short_count else 0.0
    return {
        "long_threshold": float(long_threshold),
        "short_threshold": float(short_threshold),
        "trades": int(total),
        "long_count": int(long_count),
        "short_count": int(short_count),
        "directional_balance": float(balance),
        "win_rate": float(win_rate),
        "long_win_rate": float(long_win_rate),
        "short_win_rate": float(short_win_rate),
        "reward_mean": reward_mean,
        "reward_sum": float(sum(rewards)) if rewards else 0.0,
        "long_reward_mean": long_reward_mean,
        "short_reward_mean": short_reward_mean,
    }


def _score(row: dict[str, Any], args: argparse.Namespace) -> float:
    if row["trades"] < args.min_trades:
        return -1e12 + row["trades"]
    side_penalty = 0.0
    if row["long_count"] < args.min_side_trades:
        side_penalty += (args.min_side_trades - row["long_count"]) * 10.0
    if row["short_count"] < args.min_side_trades:
        side_penalty += (args.min_side_trades - row["short_count"]) * 10.0
    return (
        row["reward_mean"]
        + args.win_rate_weight * row["win_rate"]
        + args.balance_weight * row["directional_balance"]
        + args.health_weight * row.get("health_score", 0.0)
        - args.poll_seconds_weight * row.get("health_poll_seconds", 0)
        - side_penalty
    )


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return float((sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5)


def _health_metrics(selected: list[Candidate], window_size: int, window_step: int) -> dict[str, Any]:
    ordered = sorted(selected, key=lambda item: item.source_index)
    if window_size <= 0 or window_step <= 0 or len(ordered) < window_size:
        return {
            "health_windows": 0,
            "health_window_size": int(window_size),
            "health_window_step": int(window_step),
            "health_mean_reward_mean": 0.0,
            "health_reward_volatility": 0.0,
            "health_mean_balance": 0.0,
            "health_min_long_count": 0,
            "health_min_short_count": 0,
            "health_latest_long_count": 0,
            "health_latest_short_count": 0,
            "health_score": -1000.0,
        }

    reward_means: list[float] = []
    balances: list[float] = []
    long_counts: list[int] = []
    short_counts: list[int] = []
    for end in range(window_size, len(ordered) + 1, window_step):
        window = ordered[end - window_size : end]
        rewards = [item.reward for item in window]
        long_count = sum(1 for item in window if item.side == 0)
        short_count = len(window) - long_count
        reward_means.append(_mean(rewards))
        balances.append(min(long_count, short_count) / max(long_count, short_count) if max(long_count, short_count) else 0.0)
        long_counts.append(long_count)
        short_counts.append(short_count)

    min_long = min(long_counts) if long_counts else 0
    min_short = min(short_counts) if short_counts else 0
    support_penalty = 0.0
    if min_long < 1:
        support_penalty += 1.0
    if min_short < 1:
        support_penalty += 1.0
    volatility = _std(reward_means)
    mean_reward = _mean(reward_means)
    mean_balance = _mean(balances)
    health_score = mean_reward + mean_balance - args_health_volatility_penalty(volatility) - support_penalty
    return {
        "health_windows": int(len(reward_means)),
        "health_window_size": int(window_size),
        "health_window_step": int(window_step),
        "health_mean_reward_mean": mean_reward,
        "health_reward_volatility": volatility,
        "health_mean_balance": mean_balance,
        "health_min_long_count": int(min_long),
        "health_min_short_count": int(min_short),
        "health_latest_long_count": int(long_counts[-1]) if long_counts else 0,
        "health_latest_short_count": int(short_counts[-1]) if short_counts else 0,
        "health_score": float(health_score),
    }


def args_health_volatility_penalty(volatility: float) -> float:
    return float(min(10.0, volatility))


def _sweep(args: argparse.Namespace, candidates: list[Candidate]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    long_values = _float_range(args.long_thresholds)
    short_values = _float_range(args.short_thresholds)
    window_values = _int_range(args.health_window_sizes)
    step_values = _int_range(args.health_window_steps)
    poll_values = _int_range(args.health_poll_seconds_values)
    rows: list[dict[str, Any]] = []
    for long_threshold in long_values:
        for short_threshold in short_values:
            passing = [
                item
                for item in candidates
                if (item.side == 0 and item.margin >= long_threshold)
                or (item.side == 1 and item.margin >= short_threshold)
            ]
            if args.admission_mode == "winner":
                by_row: dict[int, Candidate] = {}
                for item in passing:
                    required = short_threshold if item.side else long_threshold
                    edge = item.margin - required
                    current = by_row.get(item.source_index)
                    if current is None:
                        by_row[item.source_index] = item
                        continue
                    current_required = short_threshold if current.side else long_threshold
                    if edge > current.margin - current_required:
                        by_row[item.source_index] = item
                passing = list(by_row.values())
            for window_size in window_values:
                for window_step in step_values:
                    if window_step > window_size:
                        continue
                    health = _health_metrics(passing, window_size, window_step)
                    for poll_seconds in poll_values:
                        row = _summarize_selected(passing, long_threshold, short_threshold)
                        row.update(health)
                        row["health_poll_seconds"] = int(poll_seconds)
                        row["score"] = _score(row, args)
                        rows.append(row)

    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)
    best = ranked[0] if ranked else {}
    summary = {
        "recommended": best,
        "viable": bool(best and best.get("trades", 0) >= args.min_trades and best.get("score", -1e12) > -1e11),
        "candidate_count": len(candidates),
        "grid_size": len(rows),
        "long_thresholds": long_values,
        "short_thresholds": short_values,
        "health_window_sizes": window_values,
        "health_window_steps": step_values,
        "health_poll_seconds_values": poll_values,
        "objective": {
            "min_trades": args.min_trades,
            "min_side_trades": args.min_side_trades,
            "win_rate_weight": args.win_rate_weight,
            "balance_weight": args.balance_weight,
            "health_weight": args.health_weight,
            "poll_seconds_weight": args.poll_seconds_weight,
        },
        "top": ranked[: min(args.top, len(ranked))],
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _replace_line_value(text: str, key: str, value: float) -> str:
    text = re.sub(rf"({re.escape(key)}=)[^\s]+", rf"\g<1>{value}", text)
    return text


def _replace_cli_arg(text: str, arg_name: str, value: float) -> str:
    return re.sub(rf"({re.escape(arg_name)}\s+)[^\s]+", rf"\g<1>{value}", text)


def _replace_optional_line_value(text: str, key: str, value: int | None) -> str:
    if value is None:
        return text
    return _replace_line_value(text, key, float(value) if isinstance(value, float) else value)


def _replace_optional_cli_arg(text: str, arg_name: str, value: int | None) -> str:
    if value is None:
        return text
    return _replace_cli_arg(text, arg_name, float(value) if isinstance(value, float) else value)


def _replace_cli_arg_in_service(text: str, service: str, arg_name: str, value: float) -> str:
    """Replace a CLI arg only inside one compose service block.

    A bare regex over the whole file would also hit the scheduler's own
    --long-thresholds / --short-thresholds sweep grid, corrupting it.
    """
    marker = f"  {service}:"
    start = text.find(marker)
    if start < 0:
        return text
    next_match = re.search(r"\n  [A-Za-z0-9_-]+:", text[start + len(marker) :])
    if next_match is None:
        head, block, tail = text[:start], text[start:], ""
    else:
        next_service = start + len(marker) + next_match.start()
        head, block, tail = text[:start], text[start:next_service], text[next_service:]
    return head + _replace_cli_arg(block, arg_name, value) + tail


def _replace_health_cli_arg(text: str, arg_name: str, value: int | None) -> str:
    if value is None:
        return text
    marker = "  karakana-shadow-health:"
    start = text.find(marker)
    if start < 0:
        return text
    next_match = re.search(r"\n  [A-Za-z0-9_-]+:", text[start + len(marker) :])
    if next_match is None:
        head, block, tail = text[:start], text[start:], ""
    else:
        next_service = start + len(marker) + next_match.start()
        head, block, tail = text[:start], text[start:next_service], text[next_service:]
    return head + _replace_cli_arg(block, arg_name, value) + tail


def _patch_compose(
    path: Path,
    long_threshold: float,
    short_threshold: float,
    *,
    health_window_size: int | None,
    health_window_step: int | None,
    health_poll_seconds: int | None,
) -> None:
    """Patch the compose file with new threshold and health parameters.

    Updates both env vars and CLI args for:
    - Shadow trader: KARAKANA_SHADOW_LONG_THRESHOLD, KARAKANA_SHADOW_SHORT_THRESHOLD,
      --long-threshold, --short-threshold
    - Shadow health: KARAKANA_HEALTH_BASE_LONG_THRESHOLD, KARAKANA_HEALTH_BASE_SHORT_THRESHOLD,
      --base-long-threshold, --base-short-threshold
    - Health window: KARAKANA_HEALTH_WINDOW_SIZE, KARAKANA_HEALTH_WINDOW_STEP,
      KARAKANA_HEALTH_POLL_SECONDS, --window-size, --window-step, --poll-seconds

    Does NOT patch KARAKANA_HISTORICAL_MIN_ARMED_ROWS (strategy governor parameter)
    as it is a separate tuning knob from the health window size.
    """
    text = path.read_text(encoding="utf-8")
    updated = text
    # Shadow trader thresholds (env + CLI)
    for key in ("KARAKANA_SHADOW_LONG_THRESHOLD", "KARAKANA_HEALTH_BASE_LONG_THRESHOLD"):
        updated = _replace_line_value(updated, key, long_threshold)
    for key in ("KARAKANA_SHADOW_SHORT_THRESHOLD", "KARAKANA_HEALTH_BASE_SHORT_THRESHOLD"):
        updated = _replace_line_value(updated, key, short_threshold)
    # Health window parameters (env + CLI)
    updated = _replace_optional_line_value(updated, "KARAKANA_HEALTH_WINDOW_SIZE", health_window_size)
    updated = _replace_optional_line_value(updated, "KARAKANA_HEALTH_WINDOW_STEP", health_window_step)
    updated = _replace_optional_line_value(updated, "KARAKANA_HEALTH_POLL_SECONDS", health_poll_seconds)
    # Shadow trader CLI args (scoped to the shadow-trader service block so the
    # scheduler's own --long-thresholds/--short-thresholds grid is untouched)
    updated = _replace_cli_arg_in_service(updated, "karakana-shadow-trader", "--long-threshold", long_threshold)
    updated = _replace_cli_arg_in_service(updated, "karakana-shadow-trader", "--short-threshold", short_threshold)
    # Shadow health CLI args
    updated = _replace_cli_arg_in_service(updated, "karakana-shadow-health", "--base-long-threshold", long_threshold)
    updated = _replace_cli_arg_in_service(updated, "karakana-shadow-health", "--base-short-threshold", short_threshold)
    updated = _replace_health_cli_arg(updated, "--window-size", health_window_size)
    updated = _replace_health_cli_arg(updated, "--window-step", health_window_step)
    updated = _replace_health_cli_arg(updated, "--poll-seconds", health_poll_seconds)
    if updated != text:
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(text, encoding="utf-8")
        path.write_text(updated, encoding="utf-8")
        logging.info("Patched compose thresholds in %s", path)
        changed = True
    else:
        # Nothing to change is a legitimate steady state (the recommended values
        # already match the compose file). Do not fail the run in that case.
        logging.info("Compose thresholds already match the recommendation; no changes needed.")
        changed = False
    return {
        "long_threshold": float(long_threshold),
        "short_threshold": float(short_threshold),
        "health_window_size": health_window_size,
        "health_window_step": health_window_step,
        "health_poll_seconds": health_poll_seconds,
        "changed": changed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source", type=Path, default=Path("user_data/freqtrade_signals_karakana.csv"))
    parser.add_argument("--policy-root", type=Path, default=Path("user_data/karakana_policy/watch"))
    parser.add_argument("--checkpoint-type", choices=["best", "latest"], default="best")
    parser.add_argument("--horizon-rows", type=int, default=12)
    parser.add_argument("--horizon-mode", choices=["pair", "row"], default="pair")
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--long-thresholds", default="1.0:3.0:0.1")
    parser.add_argument("--short-thresholds", default="0.05:1.5:0.05")
    parser.add_argument("--health-window-sizes", default="128,256,512,1024")
    parser.add_argument("--health-window-steps", default="32,64,128,256")
    parser.add_argument("--health-poll-seconds-values", default="60")
    parser.add_argument("--admission-mode", choices=["winner", "both"], default="winner")
    parser.add_argument("--tail-rows", type=int, default=20000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--min-trades", type=int, default=128)
    parser.add_argument("--min-side-trades", type=int, default=8)
    parser.add_argument("--win-rate-weight", type=float, default=0.1)
    parser.add_argument("--balance-weight", type=float, default=0.1)
    parser.add_argument("--health-weight", type=float, default=0.05)
    parser.add_argument("--poll-seconds-weight", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output-csv", type=Path, default=Path("user_data/reports/karakana_live_threshold_sweep.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("user_data/reports/karakana_live_threshold_sweep.json"))
    parser.add_argument("--compose-file", type=Path, default=Path("custom-docker-compose.yml"))
    parser.add_argument("--apply-compose", action="store_true")
    parser.add_argument("--apply-health-window-size", type=int, default=None)
    parser.add_argument("--apply-health-window-step", type=int, default=None)
    parser.add_argument("--apply-health-poll-seconds", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    _ensure_karakana_importable(repo_root)
    frame = _load_signals(args.source, args.max_rows, args.tail_rows)
    candidates = _build_candidates(args, frame)
    rows, summary = _sweep(args, candidates)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    recommended = summary.get("recommended") or {}
    if args.apply_compose:
        if not summary.get("viable"):
            raise RuntimeError(
                "Refusing to patch compose: sweep did not produce a viable recommendation. "
                "Increase tail rows, loosen min trades, or inspect the output JSON."
            )
        applied = _patch_compose(
            args.compose_file,
            float(recommended["long_threshold"]),
            float(recommended["short_threshold"]),
            health_window_size=args.apply_health_window_size
            if args.apply_health_window_size is not None
            else int(recommended["health_window_size"]),
            health_window_step=args.apply_health_window_step
            if args.apply_health_window_step is not None
            else int(recommended["health_window_step"]),
            health_poll_seconds=args.apply_health_poll_seconds
            if args.apply_health_poll_seconds is not None
            else int(recommended["health_poll_seconds"]),
        )
        summary["applied_compose_file"] = str(args.compose_file)
        summary["applied_long_threshold"] = applied["long_threshold"]
        summary["applied_short_threshold"] = applied["short_threshold"]
        summary["applied_health_window_size"] = (
            args.apply_health_window_size
            if args.apply_health_window_size is not None
            else int(recommended["health_window_size"])
        )
        summary["applied_health_window_step"] = (
            args.apply_health_window_step if args.apply_health_window_step is not None else int(recommended["health_window_step"])
        )
        summary["applied_health_poll_seconds"] = (
            args.apply_health_poll_seconds
            if args.apply_health_poll_seconds is not None
            else int(recommended["health_poll_seconds"])
        )
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
