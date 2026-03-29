import argparse
import bisect
import csv
import json
import math
from pathlib import Path

import yfinance as yf


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def std_sample(vals):
    n = len(vals)
    if n < 2:
        return float("nan")
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))


def corr(x, y):
    if len(x) < 2:
        return float("nan")
    mx = mean(x)
    my = mean(y)
    sxx = sum((v - mx) ** 2 for v in x)
    syy = sum((v - my) ** 2 for v in y)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(len(x)))
    return sxy / math.sqrt(sxx * syy)


def ols_alpha_beta(y, x):
    mx = mean(x)
    my = mean(y)
    sxx = sum((v - mx) ** 2 for v in x)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(len(y)))
    b = sxy / sxx
    a = my - b * mx
    return a, b


def rolling_beta_past_only(asset_ret, btc_ret, window):
    out = [None] * len(asset_ret)
    for i in range(window, len(asset_ret)):
        _, b = ols_alpha_beta(asset_ret[i - window : i], btc_ret[i - window : i])
        out[i] = b
    return out


def max_drawdown(eq):
    if not eq:
        return float("nan")
    peak = eq[0]
    mdd = 0.0
    for v in eq:
        peak = max(peak, v)
        dd = (v / peak) - 1.0 if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd
    return mdd


def fetch_close_series(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return []
    out = []
    for idx, row in df.iterrows():
        # Handle scalar extraction robustly across pandas versions.
        close_val = row["Close"]
        if hasattr(close_val, "iloc"):
            close_val = close_val.iloc[0]
        out.append((idx.strftime("%Y-%m-%d"), float(close_val)))
    return out


def compute_strategy_returns(a_close, b_close, beta_window, z_window, z_entry, z_exit, fee_rate):
    dates = list(range(len(a_close)))

    a_ret = [a_close[i] / a_close[i - 1] - 1.0 for i in range(1, len(a_close))]
    b_ret = [b_close[i] / b_close[i - 1] - 1.0 for i in range(1, len(b_close))]

    log_a = [math.log(v) for v in a_close]
    log_b = [math.log(v) for v in b_close]
    a0, b0 = ols_alpha_beta(log_a, log_b)
    spread = [log_a[i] - (a0 + b0 * log_b[i]) for i in range(len(dates))]

    rb = rolling_beta_past_only(a_ret, b_ret, beta_window)
    hedged_ret = [0.0 if rb[i] is None else (a_ret[i] - rb[i] * b_ret[i]) for i in range(len(a_ret))]

    z = [None] * len(dates)
    for i in range(z_window - 1, len(dates)):
        w = spread[i - z_window + 1 : i + 1]
        m = mean(w)
        s = std_sample(w)
        z[i] = (spread[i] - m) / s if s and not math.isnan(s) and s > 0 else None

    ret_dates = list(range(len(dates) - 1))
    z_ret = z[1:]
    pos = [0] * len(ret_dates)
    cur = 0
    for i in range(len(ret_dates)):
        zi = z_ret[i]
        if zi is None:
            pos[i] = cur
            continue
        if cur == 0:
            if zi >= z_entry:
                cur = -1
            elif zi <= -z_entry:
                cur = 1
        else:
            if abs(zi) <= z_exit:
                cur = 0
        pos[i] = cur

    gross = [0.0] * len(ret_dates)
    for i in range(1, len(ret_dates)):
        gross[i] = pos[i - 1] * hedged_ret[i]

    net = [0.0] * len(ret_dates)
    for i in range(len(ret_dates)):
        turnover = 0.0
        if i >= 1:
            prev = pos[i - 2] if i >= 2 else 0
            curr = pos[i - 1]
            turnover = abs(curr - prev)
        fee_cost = turnover * fee_rate
        net[i] = gross[i] - fee_cost

    return {
        "ret": net,
        "pos": pos,
        "a_ret": a_ret,
        "b_ret": b_ret,
        "price_beta_static": b0,
        "returns_corr": corr(a_ret, b_ret),
    }


def metrics_from_returns(ret, bars_per_year=365.0):
    if not ret:
        return {
            "total_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }

    eq = []
    v = 1.0
    for r in ret:
        v *= 1.0 + r
        eq.append(v)

    if len(ret) > 1:
        mu = mean(ret[1:])
        sigma = std_sample(ret[1:])
    else:
        mu = float("nan")
        sigma = float("nan")

    sharpe = float("nan")
    if sigma and not math.isnan(sigma) and sigma > 0:
        sharpe = (mu / sigma) * math.sqrt(bars_per_year)

    return {
        "total_return": eq[-1] - 1.0,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(eq),
    }


def parse_int_list(v):
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def parse_float_list(v):
    return [float(x.strip()) for x in v.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="MSTR")
    parser.add_argument("--btc", default="BTC-USD")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-03-23")
    parser.add_argument("--z-window", type=int, default=30)
    parser.add_argument("--fee-rate", type=float, default=0.00045)
    parser.add_argument("--split-ratio", type=float, default=0.7)
    parser.add_argument(
        "--split-date",
        default="",
        help="Chronological split date (YYYY-MM-DD). Train uses dates < split-date, OOS uses >= split-date.",
    )
    parser.add_argument("--beta-windows", default="40,60,90,120")
    parser.add_argument("--z-entries", default="1.5,1.75,2.0,2.25")
    parser.add_argument("--z-exits", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    split_tag = str(args.split_ratio).replace(".", "p")
    if args.split_date:
        split_tag = f"date_{args.split_date.replace('-', '')}"
    prefix = f"yahoo_{args.asset.lower()}_oos_sweep_split_{split_tag}"

    beta_windows = parse_int_list(args.beta_windows)
    z_entries = parse_float_list(args.z_entries)
    z_exits = parse_float_list(args.z_exits)

    a_rows = fetch_close_series(args.asset, args.start, args.end)
    b_rows = fetch_close_series(args.btc, args.start, args.end)
    if not a_rows or not b_rows:
        raise RuntimeError("Failed to fetch Yahoo data")

    a_map = {d: c for d, c in a_rows}
    b_map = {d: c for d, c in b_rows}
    dates = sorted(set(a_map) & set(b_map))

    a_close = [a_map[d] for d in dates]
    b_close = [b_map[d] for d in dates]

    if len(dates) < max(beta_windows) + args.z_window + 60:
        raise RuntimeError("Not enough aligned rows for requested sweep")

    ret_len = len(dates) - 1
    if args.split_date:
        split_idx = bisect.bisect_left(dates, args.split_date)
        if split_idx >= len(dates):
            raise RuntimeError(
                f"split-date {args.split_date} is after last aligned market date {dates[-1]}"
            )
        split_method = "time_cutoff_date"
    else:
        split_idx = int(ret_len * args.split_ratio)
        split_method = "time_contiguous_ratio"

    split_idx = max(20, min(ret_len - 20, split_idx))

    results = []
    for bw in beta_windows:
        for ze in z_entries:
            for zx in z_exits:
                if zx >= ze:
                    continue

                run = compute_strategy_returns(
                    a_close=a_close,
                    b_close=b_close,
                    beta_window=bw,
                    z_window=args.z_window,
                    z_entry=ze,
                    z_exit=zx,
                    fee_rate=args.fee_rate,
                )

                full_m = metrics_from_returns(run["ret"])
                is_ret = run["ret"][:split_idx]
                oos_ret = run["ret"][split_idx:]
                is_m = metrics_from_returns(is_ret)
                oos_m = metrics_from_returns(oos_ret)

                results.append(
                    {
                        "beta_window": bw,
                        "z_entry": ze,
                        "z_exit": zx,
                        "rows_aligned": len(dates),
                        "split_ratio": args.split_ratio,
                        "split_date": dates[split_idx],
                        "split_method": split_method,
                        "returns_corr": run["returns_corr"],
                        "price_beta_static": run["price_beta_static"],
                        "full_return": full_m["total_return"],
                        "full_sharpe": full_m["sharpe"],
                        "full_mdd": full_m["max_drawdown"],
                        "is_return": is_m["total_return"],
                        "is_sharpe": is_m["sharpe"],
                        "is_mdd": is_m["max_drawdown"],
                        "oos_return": oos_m["total_return"],
                        "oos_sharpe": oos_m["sharpe"],
                        "oos_mdd": oos_m["max_drawdown"],
                    }
                )

    if not results:
        raise RuntimeError("No valid parameter combinations")

    def sharpe_key(x, field):
        v = x[field]
        if v is None or math.isnan(v):
            return -1e18
        return v

    by_oos = sorted(results, key=lambda x: sharpe_key(x, "oos_sharpe"), reverse=True)
    by_is = sorted(results, key=lambda x: sharpe_key(x, "is_sharpe"), reverse=True)

    best_is = by_is[0]

    csv_path = base / f"{prefix}.csv"
    fields = list(results[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    summary = {
        "asset": args.asset,
        "btc": args.btc,
        "start": dates[0],
        "end": dates[-1],
        "rows_aligned": len(dates),
        "split_ratio": args.split_ratio,
        "split_date": dates[split_idx],
        "split_method": split_method,
        "requested_split_date": args.split_date or None,
        "z_window": args.z_window,
        "fee_rate": args.fee_rate,
        "search_space": {
            "beta_windows": beta_windows,
            "z_entries": z_entries,
            "z_exits": z_exits,
        },
        "best_by_oos_sharpe": by_oos[: args.top_n],
        "best_by_is_sharpe": by_is[: args.top_n],
        "best_is_then_oos": best_is,
        "csv": csv_path.name,
    }

    json_path = base / f"{prefix}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# Yahoo OOS Parameter Sweep ({args.asset} vs {args.btc})",
        "",
        f"- Period: {dates[0]} to {dates[-1]}",
        f"- Rows aligned: {len(dates)}",
        f"- Split method: {split_method}",
        f"- Split date: {dates[split_idx]} (IS uses < split date, OOS uses >= split date)",
        f"- Z window: {args.z_window}",
        f"- Fee rate: {args.fee_rate}",
        f"- Search combinations: {len(results)}",
        "",
        "## Top by OOS Sharpe",
    ]

    for i, r in enumerate(by_oos[: args.top_n], start=1):
        lines.append(
            f"{i}. bw={r['beta_window']}, z_entry={r['z_entry']}, z_exit={r['z_exit']} | "
            f"OOS Sharpe={r['oos_sharpe']:.4f}, OOS Return={r['oos_return']:.2%}, OOS MDD={r['oos_mdd']:.2%} | "
            f"IS Sharpe={r['is_sharpe']:.4f}, IS Return={r['is_return']:.2%}"
        )

    lines.append("")
    lines.append("## Best IS Model And Its OOS")
    lines.append(
        f"- bw={best_is['beta_window']}, z_entry={best_is['z_entry']}, z_exit={best_is['z_exit']}"
    )
    lines.append(
        f"- IS: Sharpe={best_is['is_sharpe']:.4f}, Return={best_is['is_return']:.2%}, MDD={best_is['is_mdd']:.2%}"
    )
    lines.append(
        f"- OOS: Sharpe={best_is['oos_sharpe']:.4f}, Return={best_is['oos_return']:.2%}, MDD={best_is['oos_mdd']:.2%}"
    )
    lines.append("")
    lines.append(f"- Full CSV: {csv_path.name}")
    lines.append(f"- Full JSON summary: {json_path.name}")

    md_path = base / f"{prefix}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "csv": csv_path.name,
        "json": json_path.name,
        "md": md_path.name,
        "best_oos": by_oos[0],
        "best_is": best_is,
    }, indent=2))


if __name__ == "__main__":
    main()
