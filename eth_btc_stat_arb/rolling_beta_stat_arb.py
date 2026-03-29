import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

API_URL = "https://api.hyperliquid.xyz/info"


def symbol_to_filename(symbol: str) -> str:
    return symbol.lower().replace(":", "_").replace("/", "-")


def read_candles_csv(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "t": int(row["t"]),
                    "time": row["time"],
                    "close": float(row["close"]),
                }
            )
    rows.sort(key=lambda x: x["t"])
    return rows


def write_candles_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t", "time", "close"])
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    "t": int(row["t"]),
                    "time": row["time"],
                    "close": float(row["close"]),
                }
            )


def dedup_rows(rows):
    d = {}
    for r in rows:
        d[int(r["t"])] = {
            "t": int(r["t"]),
            "time": r["time"],
            "close": float(r["close"]),
        }
    return [d[t] for t in sorted(d.keys())]


def filter_range(rows, start_ms: int, end_ms: int):
    return [r for r in rows if int(r["t"]) >= int(start_ms) and int(r["t"]) <= int(end_ms)]


def load_candles_with_cache(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    cache_dir: Path,
    refresh_cache: bool,
    allow_network: bool,
):
    cache_path = cache_dir / f"{symbol_to_filename(symbol)}_{interval}.csv"

    existing = [] if refresh_cache else read_candles_csv(cache_path)
    existing_in_range = filter_range(existing, start_ms, end_ms)
    if existing_in_range:
        return existing_in_range, "cache", cache_path

    if not allow_network:
        return [], "cache-miss", cache_path

    fetched = fetch_candles(symbol, interval, start_ms, end_ms)
    merged = dedup_rows(existing + fetched)
    if merged:
        write_candles_csv(cache_path, merged)
    return filter_range(merged, start_ms, end_ms), "api", cache_path


def post_info(payload: dict):
    req = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None


def fetch_candles(symbol: str, interval: str, start_ms: int, end_ms: int):
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": interval,
            "startTime": int(start_ms),
            "endTime": int(end_ms),
        },
    }
    data = post_info(payload)
    if not isinstance(data, list):
        return []

    rows = []
    for x in data:
        rows.append(
            {
                "t": int(x["t"]),
                "time": datetime.fromtimestamp(x["t"] / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "close": float(x["c"]),
            }
        )
    rows.sort(key=lambda r: r["t"])

    dedup = {}
    for r in rows:
        dedup[r["t"]] = r
    return [dedup[t] for t in sorted(dedup.keys())]


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def std_sample(vals):
    n = len(vals)
    if n < 2:
        return float("nan")
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))


def corr(x, y):
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mx = mean(x)
    my = mean(y)
    sxx = sum((v - mx) ** 2 for v in x)
    syy = sum((v - my) ** 2 for v in y)
    if sxx == 0 or syy == 0:
        return float("nan")
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(min(len(x), len(y))))
    return sxy / math.sqrt(sxx * syy)


def ols_alpha_beta(y, x):
    mx = mean(x)
    my = mean(y)
    sxx = sum((v - mx) ** 2 for v in x)
    if sxx == 0:
        return my, 0.0
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(len(y)))
    b = sxy / sxx
    a = my - b * mx
    return a, b


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def rolling_beta_past_only(asset_ret, btc_ret, window):
    n = len(asset_ret)
    rb = [None] * n
    for i in range(window, n):
        # Use only past returns [i-window, i) so beta at i does not include return i.
        y = asset_ret[i - window : i]
        x = btc_ret[i - window : i]
        _, b = ols_alpha_beta(y, x)
        rb[i] = b
    return rb


def safe_num(x):
    if x is None:
        return float("nan")
    return float(x)


def interval_hours(interval: str) -> float:
    if interval.endswith("m"):
        return float(interval[:-1]) / 60.0
    if interval.endswith("h"):
        return float(interval[:-1])
    if interval.endswith("d"):
        return float(interval[:-1]) * 24.0
    if interval.endswith("w"):
        return float(interval[:-1]) * 24.0 * 7.0
    if interval.endswith("M"):
        return float(interval[:-1]) * 24.0 * 30.0
    return 24.0


def holding_durations(position):
    durs = []
    side = 0
    bars = 0
    for p in position:
        if p == 0:
            if side != 0:
                durs.append(bars)
                side = 0
                bars = 0
            continue
        if side == 0:
            side = p
            bars = 1
            continue
        if p == side:
            bars += 1
            continue
        durs.append(bars)
        side = p
        bars = 1
    if side != 0:
        durs.append(bars)
    return durs


def max_drawdown(equity):
    if not equity:
        return float("nan")
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        dd = (v / peak) - 1.0 if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd
    return mdd


def save_graphs(
    base: Path,
    prefix: str,
    ret_dates,
    spread_ret,
    z_ret,
    rb,
    position,
    equity_gross,
    equity_net,
    interval: str,
):
    x = list(range(len(ret_dates)))
    z_vals = [safe_num(v) for v in z_ret]
    rb_vals = [safe_num(v) for v in rb]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(x, spread_ret, color="#1f4e79", linewidth=1.0, label="Spread")
    axes[0].plot(x, z_vals, color="#c0392b", linewidth=1.0, alpha=0.7, label="Z-score")
    axes[0].axhline(2.0, color="#7f0000", linestyle="--", linewidth=0.8)
    axes[0].axhline(-2.0, color="#7f0000", linestyle="--", linewidth=0.8)
    axes[0].axhline(0.5, color="#8c6239", linestyle=":", linewidth=0.8)
    axes[0].axhline(-0.5, color="#8c6239", linestyle=":", linewidth=0.8)
    axes[0].set_title(f"{prefix} spread and z-score ({interval})")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].plot(x, rb_vals, color="#006d77", linewidth=1.1, label="Rolling beta")
    axes[1].plot(x, position, color="#e67e22", linewidth=0.9, label="Position")
    axes[1].axhline(0.0, color="#555555", linewidth=0.7)
    axes[1].set_title("Rolling beta and position")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper left")

    axes[2].plot(x, equity_gross, color="#2c3e50", linewidth=1.1, label="Equity gross")
    axes[2].plot(x, equity_net, color="#2a9d8f", linewidth=1.4, label="Equity net")
    axes[2].axhline(1.0, color="#555555", linewidth=0.7)
    axes[2].set_title("Equity curves")
    axes[2].set_xlabel("Bar index")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper left")

    fig.tight_layout()
    out = base / f"{prefix}_{interval}_rolling_beta_charts.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="ETH")
    parser.add_argument("--btc", default="BTC")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--start-ms", type=int, default=1727308800000)
    parser.add_argument("--end-ms", type=int, default=1772236800000)
    parser.add_argument("--beta-window", type=int, default=60)
    # Hyperliquid docs userFees example: feeSchedule.cross = 0.00045 (4.5 bps taker perp fee).
    parser.add_argument("--taker-fee-rate", type=float, default=0.00045)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-network", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    out_prefix = args.asset.lower().replace("/", "-")
    cache_dir = (base / args.cache_dir).resolve()
    allow_network = not args.no_network

    asset_rows, asset_source, asset_cache_path = load_candles_with_cache(
        symbol=args.asset,
        interval=args.interval,
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        cache_dir=cache_dir,
        refresh_cache=args.refresh_cache,
        allow_network=allow_network,
    )
    if not asset_rows:
        msg = {
            "error": "No candle data for requested asset symbol on Hyperliquid",
            "requested_symbol": args.asset,
            "requested_interval": args.interval,
            "cache_path": str(asset_cache_path),
            "asset_source": asset_source,
            "hint": "Check symbol naming and availability in Hyperliquid candleSnapshot.",
        }
        (base / f"{out_prefix}_{args.interval}_rolling_beta_error.json").write_text(json.dumps(msg, indent=2), encoding="utf-8")
        print(json.dumps(msg, indent=2))
        return

    asset_start_ms = min(r["t"] for r in asset_rows)
    asset_end_ms = max(r["t"] for r in asset_rows)

    # Fetch hedge leg only for the available asset window to avoid oversized requests.
    btc_rows, btc_source, btc_cache_path = load_candles_with_cache(
        symbol=args.btc,
        interval=args.interval,
        start_ms=asset_start_ms,
        end_ms=asset_end_ms,
        cache_dir=cache_dir,
        refresh_cache=args.refresh_cache,
        allow_network=allow_network,
    )
    if not btc_rows:
        raise RuntimeError(f"No BTC data returned. cache={btc_cache_path}")

    asset_map = {r["time"]: r["close"] for r in asset_rows}
    btc_map = {r["time"]: r["close"] for r in btc_rows}
    dates = sorted(set(asset_map.keys()) & set(btc_map.keys()))
    if len(dates) < args.beta_window + 40:
        raise RuntimeError("Not enough aligned rows for rolling-beta test")

    asset_close = [asset_map[d] for d in dates]
    btc_close = [btc_map[d] for d in dates]

    log_a = [math.log(v) for v in asset_close]
    log_b = [math.log(v) for v in btc_close]
    a_p, b_p = ols_alpha_beta(log_a, log_b)
    spread = [log_a[i] - (a_p + b_p * log_b[i]) for i in range(len(dates))]

    a_ret = [asset_close[i] / asset_close[i - 1] - 1.0 for i in range(1, len(asset_close))]
    b_ret = [btc_close[i] / btc_close[i - 1] - 1.0 for i in range(1, len(btc_close))]
    ret_dates = dates[1:]

    # Rolling beta on returns, used one-day lag to avoid lookahead.
    rb = rolling_beta_past_only(a_ret, b_ret, args.beta_window)
    hedged_ret = []
    for i in range(len(a_ret)):
        beta_used = rb[i]
        if beta_used is None:
            hedged_ret.append(0.0)
        else:
            hedged_ret.append(a_ret[i] - beta_used * b_ret[i])

    # Mean reversion test: ds = a + b*s_lag + e
    x = spread[:-1]
    y = [spread[i] - spread[i - 1] for i in range(1, len(spread))]
    a_mr, b_mr = ols_alpha_beta(y, x)
    n = len(x)
    mx = mean(x)
    sxx = sum((v - mx) ** 2 for v in x)
    resid = [y[i] - (a_mr + b_mr * x[i]) for i in range(n)]
    sigma2 = sum(r * r for r in resid) / max(1, n - 2)
    se = math.sqrt(sigma2 / sxx) if sxx > 0 else float("nan")
    t_stat = b_mr / se if se and not math.isnan(se) else float("nan")
    p_one = norm_cdf(t_stat) if not math.isnan(t_stat) else float("nan")
    half_life = math.log(2) / (-b_mr) if b_mr < 0 else float("inf")

    # Z-score strategy on spread
    win = 30
    z = [None] * len(dates)
    for i in range(win - 1, len(dates)):
        w = spread[i - win + 1 : i + 1]
        m = mean(w)
        s = std_sample(w)
        z[i] = (spread[i] - m) / s if s and not math.isnan(s) and s > 0 else None

    # Align strategy arrays to returns timeline
    z_ret = z[1:]
    spread_ret = spread[1:]
    position = [0] * len(ret_dates)
    cur = 0
    for i in range(len(ret_dates)):
        zi = z_ret[i]
        if zi is None:
            position[i] = cur
            continue
        if cur == 0:
            if zi >= 2:
                cur = -1
            elif zi <= -2:
                cur = 1
        else:
            if abs(zi) <= 0.5:
                cur = 0
        position[i] = cur

    strategy_ret_gross = [0.0] * len(ret_dates)
    for i in range(1, len(ret_dates)):
        strategy_ret_gross[i] = position[i - 1] * hedged_ret[i]

    fee_cost = [0.0] * len(ret_dates)
    strategy_ret_net = [0.0] * len(ret_dates)
    for i in range(len(ret_dates)):
        turnover = 0.0
        if i >= 1:
            prev_pos = position[i - 2] if i >= 2 else 0
            curr_pos = position[i - 1]
            turnover = abs(curr_pos - prev_pos)
        fee_cost[i] = turnover * args.taker_fee_rate
        strategy_ret_net[i] = strategy_ret_gross[i] - fee_cost[i]

    equity_gross = []
    eq_g = 1.0
    for r in strategy_ret_gross:
        eq_g *= 1.0 + r
        equity_gross.append(eq_g)

    equity_net = []
    eq_n = 1.0
    for r in strategy_ret_net:
        eq_n *= 1.0 + r
        equity_net.append(eq_n)

    bars_per_year = (24.0 / interval_hours(args.interval)) * 365.0

    sharpe_gross = float("nan")
    st_g = std_sample(strategy_ret_gross[1:])
    if st_g and not math.isnan(st_g) and st_g > 0:
        sharpe_gross = (mean(strategy_ret_gross[1:]) / st_g) * math.sqrt(bars_per_year)

    sharpe_net = float("nan")
    st_n = std_sample(strategy_ret_net[1:])
    if st_n and not math.isnan(st_n) and st_n > 0:
        sharpe_net = (mean(strategy_ret_net[1:]) / st_n) * math.sqrt(bars_per_year)

    beta_points = sum(1 for b in rb if b is not None)
    beta_coverage = beta_points / len(rb) if rb else float("nan")
    holds = holding_durations(position)
    avg_hold_bars = mean(holds) if holds else 0.0
    avg_hold_hours = avg_hold_bars * interval_hours(args.interval)
    avg_hold_days = avg_hold_hours / 24.0

    metrics = {
        "asset_symbol": args.asset,
        "btc_symbol": args.btc,
        "interval": args.interval,
        "start_date": dates[0],
        "end_date": dates[-1],
        "rows_aligned": len(dates),
        "return_rows": len(ret_dates),
        "price_beta_static": b_p,
        "rolling_beta_window": args.beta_window,
        "rolling_beta_points": beta_points,
        "rolling_beta_coverage": beta_coverage,
        "rolling_beta_last": rb[-1],
        "bars_per_year": bars_per_year,
        "returns_corr": corr(a_ret, b_ret),
        "mean_reversion_b": b_mr,
        "mean_reversion_t": t_stat,
        "mean_reversion_p_one_sided": p_one,
        "half_life_days": half_life,
        "taker_fee_rate": args.taker_fee_rate,
        "asset_data_source": asset_source,
        "btc_data_source": btc_source,
        "asset_cache_file": str(asset_cache_path),
        "btc_cache_file": str(btc_cache_path),
        "strategy_total_return_gross": equity_gross[-1] - 1.0,
        "strategy_total_return_net": equity_net[-1] - 1.0,
        "max_drawdown_net": max_drawdown(equity_net),
        "strategy_sharpe_365_gross": sharpe_gross,
        "strategy_sharpe_365_net": sharpe_net,
        "total_fee_drag_return": (equity_gross[-1] - equity_net[-1]),
        "avg_holding_bars": avg_hold_bars,
        "avg_holding_hours": avg_hold_hours,
        "avg_holding_days": avg_hold_days,
        "holding_count": len(holds),
        "signal_count": sum(abs(p) for p in position),
    }
    metrics["opportunity_flag"] = bool(
        metrics["mean_reversion_b"] < 0
        and metrics["mean_reversion_p_one_sided"] < 0.05
        and metrics["strategy_sharpe_365_net"] > 0.75
        and metrics["strategy_total_return_net"] > 0
        and metrics["signal_count"] > 10
    )

    with open(base / f"{out_prefix}_{args.interval}_rolling_beta_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(base / f"{out_prefix}_{args.interval}_rolling_beta_signals.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "date",
                "asset_close",
                "btc_close",
                "spread",
                "z",
                "rolling_beta",
                "position",
                "hedged_ret",
                "fee_cost",
                "strategy_ret_gross",
                "strategy_ret_net",
                "equity_curve_gross",
                "equity_curve_net",
            ]
        )
        for i in range(len(ret_dates)):
            w.writerow(
                [
                    ret_dates[i],
                    asset_close[i + 1],
                    btc_close[i + 1],
                    spread_ret[i],
                    z_ret[i],
                    rb[i],
                    position[i],
                    hedged_ret[i],
                    fee_cost[i],
                    strategy_ret_gross[i],
                    strategy_ret_net[i],
                    equity_gross[i],
                    equity_net[i],
                ]
            )

    chart_path = save_graphs(
        base=base,
        prefix=out_prefix,
        ret_dates=ret_dates,
        spread_ret=spread_ret,
        z_ret=z_ret,
        rb=rb,
        position=position,
        equity_gross=equity_gross,
        equity_net=equity_net,
        interval=args.interval,
    )

    verdict = "Potential opportunity" if metrics["opportunity_flag"] else "No robust statistical-arbitrage opportunity"
    report = "\n".join(
        [
            f"# {args.asset} vs {args.btc} Stat-Arb (Rolling Beta)",
            "",
            f"- Interval: {args.interval}",
            f"- Period: {metrics['start_date']} to {metrics['end_date']}",
            f"- Aligned bars: {metrics['rows_aligned']}",
            f"- Static price beta: {metrics['price_beta_static']:.4f}",
            f"- Rolling beta window: {metrics['rolling_beta_window']}",
            f"- Rolling beta points: {metrics['rolling_beta_points']} ({metrics['rolling_beta_coverage']:.2%} coverage)",
            f"- Last rolling beta: {metrics['rolling_beta_last']:.4f}",
            f"- Return correlation: {metrics['returns_corr']:.4f}",
            f"- Mean-reversion b: {metrics['mean_reversion_b']:.6f}",
            f"- Mean-reversion p(one-sided): {metrics['mean_reversion_p_one_sided']:.6f}",
            f"- Half-life (days): {metrics['half_life_days']:.2f}",
            f"- Taker fee rate used: {metrics['taker_fee_rate']:.5f} ({metrics['taker_fee_rate'] * 100:.3f}%)",
            f"- Strategy total return (gross): {metrics['strategy_total_return_gross']:.4%}",
            f"- Strategy total return (net): {metrics['strategy_total_return_net']:.4%}",
            f"- Strategy Sharpe(365) gross/net: {metrics['strategy_sharpe_365_gross']:.4f} / {metrics['strategy_sharpe_365_net']:.4f}",
            f"- Fee drag on ending equity: {metrics['total_fee_drag_return']:.4f}",
            f"- Holding count: {metrics['holding_count']}",
            f"- Avg holding time: {metrics['avg_holding_bars']:.2f} bars ({metrics['avg_holding_hours']:.2f} hours, {metrics['avg_holding_days']:.2f} days)",
            f"- Signal count: {metrics['signal_count']}",
            f"- Chart: {chart_path.name}",
            "",
            f"## Verdict: {verdict}",
        ]
    ) + "\n"

    (base / f"{out_prefix}_{args.interval}_rolling_beta_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
