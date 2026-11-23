#!/usr/bin/env python3
"""Check live unrealized profit of open Freqtrade futures trades from SQLite + Bybit tickers."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BYBIT_TICKERS_URL = "https://api.bybit.com/v5/market/tickers?category=linear"
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
BINANCE_TICKERS_URL = "https://fapi.binance.com/fapi/v1/ticker/price"
GATE_TICKERS_URL = "https://api.gateio.ws/api/v4/futures/usdt/tickers"


def _base_coin(pair: str) -> str:
    return pair.split("/")[0]


def _pair_to_compact(pair: str) -> str:
    # "BTC/USDT:USDT" -> "BTCUSDT"
    return pair.split(":")[0].replace("/", "")


def _pair_to_underscore(pair: str) -> str:
    # "BTC/USDT:USDT" -> "BTC_USDT"
    return pair.split(":")[0].replace("/", "_")


def _fetch_exchange_prices(exchange: str, timeout: float = 5.0) -> tuple[str, dict[str, float]]:
    ex = (exchange or "bybit").lower()
    headers = {"User-Agent": "karakana-pnl-check/1.0"}
    prices: dict[str, float] = {}

    if ex == "hyperliquid":
        body = json.dumps({"type": "allMids"}).encode("utf-8")
        req = urllib.request.Request(
            HYPERLIQUID_INFO_URL,
            data=body,
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        for coin, mid in payload.items():
            try:
                val = float(mid)
                if val > 0:
                    prices[coin] = val
            except (TypeError, ValueError):
                pass
        return "hyperliquid", prices

    if ex == "binance":
        req = urllib.request.Request(BINANCE_TICKERS_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        for item in payload:
            sym = item.get("symbol")
            raw = item.get("price")
            if sym and raw:
                try:
                    val = float(raw)
                    if val > 0:
                        prices[sym] = val
                except ValueError:
                    pass
        return "binance", prices

    if ex in ("gate", "gateio"):
        req = urllib.request.Request(GATE_TICKERS_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        for item in payload:
            sym = item.get("contract")
            raw = item.get("last") or item.get("mark_price")
            if sym and raw:
                try:
                    val = float(raw)
                    if val > 0:
                        prices[sym] = val
                except ValueError:
                    pass
        return "gate", prices

    # Default: Bybit linear perpetuals
    req = urllib.request.Request(BYBIT_TICKERS_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    for item in payload.get("result", {}).get("list", []):
        sym = item.get("symbol")
        last = item.get("lastPrice") or item.get("markPrice")
        if sym and last:
            try:
                val = float(last)
                if val > 0:
                    prices[sym] = val
            except ValueError:
                pass
    return "bybit", prices


def _lookup_price(exchange: str, pair: str, prices: dict[str, float]) -> float | None:
    ex = (exchange or "bybit").lower()
    if ex == "hyperliquid":
        return prices.get(_base_coin(pair))
    if ex in ("gate", "gateio"):
        return prices.get(_pair_to_underscore(pair))
    return prices.get(_pair_to_compact(pair))


def _fallback_csv_prices(signals_csv: Path) -> dict[str, float]:
    prices: dict[str, float] = {}
    if not signals_csv.is_file():
        return prices
    with signals_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            pair = row.get("pair")
            raw = row.get("close") or row.get("reward-price")
            if pair and raw:
                try:
                    val = float(raw)
                    if val > 0:
                        prices[pair] = val
                except ValueError:
                    pass
    return prices


def _load_trades(db_path: Path) -> tuple[list[dict], float, int]:
    # Copy DB (+ WAL if present) to a local temp dir so host macOS never touches
    # live SQLite locks/SHM across Docker's VirtioFS bind mount.
    with tempfile.TemporaryDirectory(prefix="ft_pnl_") as tmpdir:
        tmp_db = Path(tmpdir) / db_path.name
        shutil.copy2(db_path, tmp_db)
        wal = db_path.with_name(db_path.name + "-wal")
        if wal.is_file():
            shutil.copy2(wal, Path(tmpdir) / wal.name)
        con = sqlite3.connect(str(tmp_db), timeout=5.0)
        con.row_factory = sqlite3.Row
        try:
            open_rows = [
                dict(r)
                for r in con.execute(
                    """
                    SELECT id, exchange, pair, is_short, open_rate, stake_amount, amount,
                           COALESCE(contract_size, 1.0) AS contract_size,
                           COALESCE(leverage, 1.0) AS leverage,
                           COALESCE(fee_open, 0.0) AS fee_open,
                           fee_open_cost,
                           COALESCE(fee_close, fee_open, 0.0) AS fee_close,
                           COALESCE(funding_fees, 0.0) AS funding_fees,
                           stop_loss, initial_stop_loss, stop_loss_pct,
                           open_date, enter_tag
                    FROM trades
                    WHERE is_open = 1
                    ORDER BY id ASC
                    """
                ).fetchall()
            ]
            closed = con.execute(
                "SELECT COUNT(*), COALESCE(SUM(close_profit_abs), 0.0) FROM trades WHERE is_open = 0"
            ).fetchone()
            closed_count = int(closed[0] or 0)
            closed_pnl = float(closed[1] or 0.0)
        finally:
            con.close()
    return open_rows, closed_pnl, closed_count


def _age_str(open_date: str) -> str:
    try:
        dt = datetime.fromisoformat(open_date.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        mins = max(0, int((datetime.now(timezone.utc) - dt).total_seconds() // 60))
        if mins < 60:
            return f"{mins}m"
        return f"{mins // 60}h{mins % 60:02d}m"
    except Exception:
        return "?"


def compute_snapshot(db_path: Path, signals_csv: Path, offline: bool = False) -> dict:
    open_trades, closed_pnl, closed_count = _load_trades(db_path)
    exchange = str(open_trades[0].get("exchange") or "bybit").lower() if open_trades else "bybit"
    price_source = exchange
    exchange_prices: dict[str, float] = {}
    csv_prices: dict[str, float] = {}

    if not offline and open_trades:
        try:
            price_source, exchange_prices = _fetch_exchange_prices(exchange)
        except Exception:
            price_source = "signals_csv"
            csv_prices = _fallback_csv_prices(signals_csv)
    else:
        price_source = "signals_csv"
        csv_prices = _fallback_csv_prices(signals_csv)

    rows = []
    total_stake = 0.0
    total_pnl_abs = 0.0
    long_pnl = 0.0
    short_pnl = 0.0
    long_n = 0
    short_n = 0

    for t in open_trades:
        pair = str(t["pair"])
        trade_ex = str(t.get("exchange") or exchange).lower()
        curr = _lookup_price(trade_ex, pair, exchange_prices) or csv_prices.get(pair)
        if curr is None and not csv_prices:
            csv_prices = _fallback_csv_prices(signals_csv)
            curr = csv_prices.get(pair)

        open_rate = float(t["open_rate"])
        stake = float(t["stake_amount"])
        qty = float(t["amount"]) * float(t["contract_size"])
        is_short = bool(t["is_short"])
        lev = float(t["leverage"])
        fee_open_cost = (
            float(t["fee_open_cost"])
            if t["fee_open_cost"] is not None
            else open_rate * qty * float(t["fee_open"])
        )
        funding = float(t["funding_fees"])

        if curr is not None and open_rate > 0:
            gross = (open_rate - curr) * qty if is_short else (curr - open_rate) * qty
            fee_close_est = curr * qty * float(t["fee_close"])
            pnl_abs = gross - fee_open_cost - fee_close_est + funding
            pnl_pct = (pnl_abs / stake * 100.0) if stake > 0 else 0.0
        else:
            pnl_abs = 0.0
            pnl_pct = 0.0

        total_stake += stake
        total_pnl_abs += pnl_abs
        if is_short:
            short_n += 1
            short_pnl += pnl_abs
        else:
            long_n += 1
            long_pnl += pnl_abs

        stop_loss = float(t["stop_loss"]) if t["stop_loss"] is not None else None
        init_sl = float(t["initial_stop_loss"]) if t["initial_stop_loss"] is not None else None
        sl_adjusted = (
            stop_loss is not None
            and init_sl is not None
            and abs(stop_loss - init_sl) > max(abs(init_sl) * 1e-6, 1e-12)
        )
        if stop_loss is not None and curr is not None and curr > 0:
            sl_dist_pct = ((curr - stop_loss) / curr * 100.0) if not is_short else ((stop_loss - curr) / curr * 100.0)
        else:
            sl_dist_pct = None

        rows.append(
            {
                "id": int(t["id"]),
                "pair": pair,
                "side": "SHORT" if is_short else "LONG",
                "leverage": lev,
                "open_rate": open_rate,
                "current_rate": curr,
                "stop_loss": stop_loss,
                "initial_stop_loss": init_sl,
                "sl_adjusted": sl_adjusted,
                "sl_dist_pct": sl_dist_pct,
                "stake": stake,
                "pnl_abs": pnl_abs,
                "pnl_pct": pnl_pct,
                "age": _age_str(str(t["open_date"] or "")),
            }
        )

    rows.sort(key=lambda r: r["pnl_pct"], reverse=True)
    total_pnl_pct = (total_pnl_abs / total_stake * 100.0) if total_stake > 0 else 0.0
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "price_source": price_source,
        "open_count": len(rows),
        "long_count": long_n,
        "short_count": short_n,
        "total_stake": total_stake,
        "unrealized_pnl_abs": total_pnl_abs,
        "unrealized_pnl_pct": total_pnl_pct,
        "long_pnl_abs": long_pnl,
        "short_pnl_abs": short_pnl,
        "closed_count": closed_count,
        "closed_pnl_abs": closed_pnl,
        "trades": rows,
    }


def render_table(snap: dict, top: int | None = None) -> None:
    print(f"\n[{snap['timestamp']}] Open Trades: {snap['open_count']} "
          f"(Long: {snap['long_count']}, Short: {snap['short_count']}) | Prices: {snap['price_source']}")
    print(f"Total Stake: {snap['total_stake']:.2f} USDT | "
          f"Unrealized PnL: {snap['unrealized_pnl_abs']:+.4f} USDT ({snap['unrealized_pnl_pct']:+.2f}%) | "
          f"Long: {snap['long_pnl_abs']:+.4f} | Short: {snap['short_pnl_abs']:+.4f}")
    if snap["closed_count"]:
        print(f"Closed Trades: {snap['closed_count']} | Realized PnL: {snap['closed_pnl_abs']:+.4f} USDT")

    trades = snap["trades"]
    if not trades:
        print("No open trades.")
        return

    if top is not None and len(trades) > 2 * top:
        shown = trades[:top] + [None] + trades[-top:]
    else:
        shown = trades

    header = (
        f"{'ID':>4}  {'PAIR':<22} {'SIDE':<5} {'LEV':>4} {'OPEN':>11} {'CURR':>11} "
        f"{'STOP_LOSS':>12} {'SL_DIST':>8} {'STAKE':>8} {'PNL USDT':>10} {'PNL %':>8} {'AGE':>6}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in shown:
        if r is None:
            print("  ...")
            continue
        curr_s = f"{r['current_rate']:.6g}" if r["current_rate"] is not None else "n/a"
        if r["stop_loss"] is not None:
            sl_s = f"{r['stop_loss']:.6g}{'*' if r['sl_adjusted'] else ' '}"
        else:
            sl_s = "n/a "
        dist_s = f"{r['sl_dist_pct']:+.2f}%" if r["sl_dist_pct"] is not None else "n/a"
        print(
            f"{r['id']:>4}  {r['pair']:<22} {r['side']:<5} {r['leverage']:>3.0f}x "
            f"{r['open_rate']:>11.6g} {curr_s:>11} {sl_s:>12} {dist_s:>8} {r['stake']:>8.2f} "
            f"{r['pnl_abs']:>+10.4f} {r['pnl_pct']:>+7.2f}% {r['age']:>6}"
        )
    print("  (* = stop_loss adjusted from initial_stop_loss)")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Check live unrealized profit of open Freqtrade trades.")
    parser.add_argument("--db", type=Path, default=root / "user_data" / "tradesv3.sqlite")
    parser.add_argument("--signals", type=Path, default=root / "user_data" / "freqtrade_signals_karakana.csv")
    parser.add_argument("--offline", action="store_true", help="Use signals CSV prices instead of live Bybit tickers.")
    parser.add_argument("--top", type=int, default=None, help="Show only top N and bottom N trades.")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output JSON.")
    parser.add_argument("--watch", type=float, default=0.0, help="Refresh interval in seconds (0 = run once).")
    args = parser.parse_args()

    try:
        while True:
            snap = compute_snapshot(args.db, args.signals, offline=args.offline)
            if args.as_json:
                print(json.dumps(snap, indent=2))
            else:
                render_table(snap, top=args.top)
            if args.watch <= 0:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
