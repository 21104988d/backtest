import argparse
import bisect
import csv
import json
import math
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        close_val = row["Close"]
        if hasattr(close_val, "iloc"):
            close_val = close_val.iloc[0]
        out.append((idx.strftime("%Y-%m-%d"), float(close_val)))
    return out


def metrics_from_returns(ret, bars_per_year=365.0):
    if not ret:
        return {"total_return": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan")}

    eq = []
    v = 1.0
    for r in ret:
        v *= 1.0 + r
        eq.append(v)

    mu = mean(ret)
    sigma = std_sample(ret)
    sharpe = float("nan")
    if sigma and not math.isnan(sigma) and sigma > 0:
        sharpe = (mu / sigma) * math.sqrt(bars_per_year)

    return {
        "total_return": eq[-1] - 1.0,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(eq),
    }


def fit_ou_ar1(series):
    # Fit AR(1): x_t = a + b*x_{t-1} + e_t, then map to OU speed.
    if len(series) < 30:
        return {"kappa": float("nan"), "half_life": float("nan"), "sigma": float("nan"), "b": float("nan")}

    x0 = series[:-1]
    x1 = series[1:]
    mx0 = mean(x0)
    mx1 = mean(x1)
    sxx = sum((v - mx0) ** 2 for v in x0)
    if sxx <= 0:
        return {"kappa": float("nan"), "half_life": float("nan"), "sigma": float("nan"), "b": float("nan")}

    sxy = sum((x0[i] - mx0) * (x1[i] - mx1) for i in range(len(x0)))
    b = sxy / sxx
    a = mx1 - b * mx0

    resid = [x1[i] - (a + b * x0[i]) for i in range(len(x0))]
    sigma = std_sample(resid)

    kappa = float("nan")
    half_life = float("nan")
    if 0.0 < b < 1.0:
        kappa = -math.log(b)
        if kappa > 0:
            half_life = math.log(2.0) / kappa

    return {"kappa": kappa, "half_life": half_life, "sigma": sigma, "b": b}


def parse_int_list(v):
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def parse_float_list(v):
    return [float(x.strip()) for x in v.split(",") if x.strip()]


def add_months(d, months):
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = min(d.day, month_end_day(y, m))
    return date(y, m, day)


def month_end_day(y, m):
    if m in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if m in (4, 6, 9, 11):
        return 30
    leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
    return 29 if leap else 28


def build_net_returns(a_close, b_close, beta_window, z_window, z_entry, z_exit, fee_rate):
    a_ret = [a_close[i] / a_close[i - 1] - 1.0 for i in range(1, len(a_close))]
    b_ret = [b_close[i] / b_close[i - 1] - 1.0 for i in range(1, len(b_close))]

    log_a = [math.log(v) for v in a_close]
    log_b = [math.log(v) for v in b_close]
    a0, b0 = ols_alpha_beta(log_a, log_b)
    spread = [log_a[i] - (a0 + b0 * log_b[i]) for i in range(len(a_close))]

    rb = rolling_beta_past_only(a_ret, b_ret, beta_window)
    hedged_ret = [0.0 if rb[i] is None else (a_ret[i] - rb[i] * b_ret[i]) for i in range(len(a_ret))]

    z = [None] * len(a_close)
    for i in range(z_window - 1, len(a_close)):
        w = spread[i - z_window + 1 : i + 1]
        m = mean(w)
        s = std_sample(w)
        z[i] = (spread[i] - m) / s if s and not math.isnan(s) and s > 0 else None

    z_ret = z[1:]
    pos = [0] * len(z_ret)
    cur = 0
    for i, zi in enumerate(z_ret):
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

    gross = [0.0] * len(pos)
    for i in range(1, len(pos)):
        gross[i] = pos[i - 1] * hedged_ret[i]

    net = [0.0] * len(pos)
    for i in range(len(pos)):
        turnover = 0.0
        if i >= 1:
            prev = pos[i - 2] if i >= 2 else 0
            curr = pos[i - 1]
            turnover = abs(curr - prev)
        net[i] = gross[i] - turnover * fee_rate

    # Parameter-dependent mean-reversion proxy series from rolling-beta hedged returns.
    mr_series = []
    csum = 0.0
    for r in hedged_ret:
        csum += r
        mr_series.append(csum)

    return {
        "returns": net,
        "mr_series": mr_series,
        "returns_corr": corr(a_ret, b_ret),
        "static_beta": b0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="MSTR")
    parser.add_argument("--btc", default="BTC-USD")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-03-23")
    parser.add_argument("--wf-start-date", default="2024-01-01")
    parser.add_argument("--oos-months", type=int, default=3)
    parser.add_argument("--min-train-bars", type=int, default=252)
    parser.add_argument("--min-oos-bars", type=int, default=40)
    parser.add_argument("--z-window", type=int, default=30)
    parser.add_argument("--fee-rate", type=float, default=0.00045)
    parser.add_argument("--selection-objective", choices=["is_sharpe", "ou_sharpe"], default="is_sharpe")
    parser.add_argument("--ou-min-half-life", type=float, default=2.0)
    parser.add_argument("--ou-max-half-life", type=float, default=120.0)
    parser.add_argument("--ou-min-kappa", type=float, default=0.0)
    parser.add_argument(
        "--ou-lookback-bars",
        type=int,
        default=252,
        help="When using OU gating, fit OU on the last N in-sample bars instead of full expanding train.",
    )
    parser.add_argument("--beta-windows", default="40,60,90,120")
    parser.add_argument("--z-entries", default="1.5,1.75,2.0,2.25")
    parser.add_argument("--z-exits", default="0.25,0.5,0.75,1.0")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    tag = f"wf_{args.asset.lower()}_{args.wf_start_date.replace('-', '')}_{args.oos_months}m_{args.selection_objective}"

    beta_windows = parse_int_list(args.beta_windows)
    z_entries = parse_float_list(args.z_entries)
    z_exits = parse_float_list(args.z_exits)

    params = []
    for bw in beta_windows:
        for ze in z_entries:
            for zx in z_exits:
                if zx >= ze:
                    continue
                params.append((bw, ze, zx))

    a_rows = fetch_close_series(args.asset, args.start, args.end)
    b_rows = fetch_close_series(args.btc, args.start, args.end)
    if not a_rows or not b_rows:
        raise RuntimeError("Failed to fetch Yahoo data")

    a_map = {d: c for d, c in a_rows}
    b_map = {d: c for d, c in b_rows}
    dates = sorted(set(a_map) & set(b_map))
    if len(dates) < max(beta_windows) + args.z_window + args.min_train_bars + args.min_oos_bars:
        raise RuntimeError("Not enough aligned rows for walk-forward constraints")

    a_close = [a_map[d] for d in dates]
    b_close = [b_map[d] for d in dates]

    ret_dates = dates[1:]

    precomputed = {}
    ref_stats = None
    for bw, ze, zx in params:
        run = build_net_returns(
            a_close=a_close,
            b_close=b_close,
            beta_window=bw,
            z_window=args.z_window,
            z_entry=ze,
            z_exit=zx,
            fee_rate=args.fee_rate,
        )
        precomputed[(bw, ze, zx)] = {"returns": run["returns"], "mr_series": run["mr_series"]}
        if ref_stats is None:
            ref_stats = {
                "returns_corr": run["returns_corr"],
                "static_beta": run["static_beta"],
            }

    wf_start_idx = bisect.bisect_left(ret_dates, args.wf_start_date)
    if wf_start_idx >= len(ret_dates):
        raise RuntimeError(f"wf-start-date {args.wf_start_date} is after available return dates")

    folds = []
    combined_oos_returns = []
    selected_counter = Counter()

    fold_start_idx = wf_start_idx
    while fold_start_idx < len(ret_dates):
        train_end_idx = fold_start_idx
        if train_end_idx < args.min_train_bars:
            fold_start_idx += 1
            continue

        fold_start_dt = datetime.strptime(ret_dates[fold_start_idx], "%Y-%m-%d").date()
        oos_end_dt = add_months(fold_start_dt, args.oos_months)
        fold_end_idx = bisect.bisect_left(ret_dates, oos_end_dt.isoformat())
        if fold_end_idx <= fold_start_idx:
            fold_start_idx += 1
            continue

        oos_len = fold_end_idx - fold_start_idx
        if oos_len < args.min_oos_bars:
            break

        best_key = None
        best_is_sharpe = -1e18
        best_is_ret = -1e18
        best_ou = None
        ou_candidate_count = 0

        for key, payload in precomputed.items():
            ret = payload["returns"]
            mr_series = payload["mr_series"]
            is_slice = ret[:train_end_idx]
            is_metrics = metrics_from_returns(is_slice)
            is_sharpe = is_metrics["sharpe"]
            is_ret = is_metrics["total_return"]
            if math.isnan(is_sharpe):
                continue

            ou_start_idx = max(0, train_end_idx - args.ou_lookback_bars)
            ou_fit = fit_ou_ar1(mr_series[ou_start_idx:train_end_idx])
            use_candidate = True
            if args.selection_objective == "ou_sharpe":
                hl = ou_fit["half_life"]
                kappa = ou_fit["kappa"]
                if (
                    math.isnan(hl)
                    or math.isnan(kappa)
                    or kappa <= 0
                    or kappa < args.ou_min_kappa
                    or hl < args.ou_min_half_life
                    or hl > args.ou_max_half_life
                ):
                    use_candidate = False
                else:
                    ou_candidate_count += 1

            if not use_candidate:
                continue

            if is_sharpe > best_is_sharpe or (is_sharpe == best_is_sharpe and is_ret > best_is_ret):
                best_key = key
                best_is_sharpe = is_sharpe
                best_is_ret = is_ret
                best_ou = ou_fit

        if args.selection_objective == "ou_sharpe" and best_key is None:
            # Fallback: if no OU-valid candidates, revert to Sharpe-only for this fold.
            for key, payload in precomputed.items():
                ret = payload["returns"]
                is_slice = ret[:train_end_idx]
                is_metrics = metrics_from_returns(is_slice)
                is_sharpe = is_metrics["sharpe"]
                is_ret = is_metrics["total_return"]
                if math.isnan(is_sharpe):
                    continue
                if is_sharpe > best_is_sharpe or (is_sharpe == best_is_sharpe and is_ret > best_is_ret):
                    best_key = key
                    best_is_sharpe = is_sharpe
                    best_is_ret = is_ret
                    ou_start_idx = max(0, train_end_idx - args.ou_lookback_bars)
                    best_ou = fit_ou_ar1(payload["mr_series"][ou_start_idx:train_end_idx])

        if best_key is None:
            fold_start_idx = fold_end_idx
            continue

        bw, ze, zx = best_key
        selected_counter[f"bw={bw}|ze={ze}|zx={zx}"] += 1

        chosen_ret = precomputed[best_key]["returns"]
        is_slice = chosen_ret[:train_end_idx]
        oos_slice = chosen_ret[fold_start_idx:fold_end_idx]

        is_metrics = metrics_from_returns(is_slice)
        oos_metrics = metrics_from_returns(oos_slice)

        combined_oos_returns.extend(oos_slice)

        folds.append(
            {
                "fold": len(folds) + 1,
                "train_start": ret_dates[0],
                "train_end": ret_dates[train_end_idx - 1],
                "oos_start": ret_dates[fold_start_idx],
                "oos_end": ret_dates[fold_end_idx - 1],
                "train_bars": train_end_idx,
                "oos_bars": oos_len,
                "beta_window": bw,
                "z_entry": ze,
                "z_exit": zx,
                "is_return": is_metrics["total_return"],
                "is_sharpe": is_metrics["sharpe"],
                "is_mdd": is_metrics["max_drawdown"],
                "oos_return": oos_metrics["total_return"],
                "oos_sharpe": oos_metrics["sharpe"],
                "oos_mdd": oos_metrics["max_drawdown"],
                "ou_kappa_is": best_ou["kappa"] if best_ou else float("nan"),
                "ou_half_life_is": best_ou["half_life"] if best_ou else float("nan"),
                "ou_sigma_is": best_ou["sigma"] if best_ou else float("nan"),
                "ou_candidates_in_fold": ou_candidate_count,
            }
        )

        # Non-overlapping walk-forward windows.
        fold_start_idx = fold_end_idx

    if not folds:
        raise RuntimeError("No valid walk-forward folds generated")

    agg = metrics_from_returns(combined_oos_returns)
    avg_oos_sharpe = mean([r["oos_sharpe"] for r in folds if not math.isnan(r["oos_sharpe"])])
    pos_oos = sum(1 for r in folds if r["oos_return"] > 0)

    summary = {
        "asset": args.asset,
        "btc": args.btc,
        "period_start": dates[0],
        "period_end": dates[-1],
        "rows_aligned": len(dates),
        "returns_corr": ref_stats["returns_corr"],
        "static_beta": ref_stats["static_beta"],
        "wf_start_date_requested": args.wf_start_date,
        "wf_start_date_used": ret_dates[wf_start_idx],
        "oos_months": args.oos_months,
        "min_train_bars": args.min_train_bars,
        "min_oos_bars": args.min_oos_bars,
        "z_window": args.z_window,
        "fee_rate": args.fee_rate,
        "selection_objective": args.selection_objective,
        "ou_min_half_life": args.ou_min_half_life,
        "ou_max_half_life": args.ou_max_half_life,
        "ou_min_kappa": args.ou_min_kappa,
        "ou_lookback_bars": args.ou_lookback_bars,
        "param_space": {
            "beta_windows": beta_windows,
            "z_entries": z_entries,
            "z_exits": z_exits,
            "count": len(params),
        },
        "fold_count": len(folds),
        "positive_oos_folds": pos_oos,
        "positive_oos_fold_ratio": pos_oos / len(folds),
        "average_oos_sharpe": avg_oos_sharpe,
        "combined_oos_return": agg["total_return"],
        "combined_oos_sharpe": agg["sharpe"],
        "combined_oos_mdd": agg["max_drawdown"],
        "avg_ou_candidates_per_fold": mean([r["ou_candidates_in_fold"] for r in folds]),
        "most_selected_params": selected_counter.most_common(10),
    }

    csv_path = base / f"{tag}_folds.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(folds[0].keys()))
        w.writeheader()
        w.writerows(folds)

    json_path = base / f"{tag}_summary.json"
    json_path.write_text(json.dumps({"summary": summary, "folds": folds}, indent=2), encoding="utf-8")

    lines = [
        f"# Walk-Forward OOS Validation ({args.asset} vs {args.btc})",
        "",
        "Leakage control:",
        "- Each fold selects parameters using only data strictly before OOS start.",
        "- Chosen parameters are then applied to the immediately following OOS window.",
        "- Folds are non-overlapping in OOS.",
        "",
        f"- Period: {summary['period_start']} to {summary['period_end']}",
        f"- Walk-forward start requested: {summary['wf_start_date_requested']}",
        f"- Walk-forward start used: {summary['wf_start_date_used']}",
        f"- OOS window size: {summary['oos_months']} months",
        f"- Selection objective: {summary['selection_objective']}",
        f"- OU half-life filter (for ou_sharpe): [{summary['ou_min_half_life']}, {summary['ou_max_half_life']}]",
        f"- OU min kappa (for ou_sharpe): {summary['ou_min_kappa']}",
        f"- OU lookback bars (for ou_sharpe): {summary['ou_lookback_bars']}",
        f"- Fold count: {summary['fold_count']}",
        f"- Positive OOS folds: {summary['positive_oos_folds']}/{summary['fold_count']} ({summary['positive_oos_fold_ratio']:.2%})",
        f"- Average OOS Sharpe across folds: {summary['average_oos_sharpe']:.4f}",
        f"- Combined OOS Return: {summary['combined_oos_return']:.2%}",
        f"- Combined OOS Sharpe: {summary['combined_oos_sharpe']:.4f}",
        f"- Combined OOS Max Drawdown: {summary['combined_oos_mdd']:.2%}",
        f"- Avg OU-valid candidates per fold: {summary['avg_ou_candidates_per_fold']:.2f}",
        "",
        "## Most Selected Parameter Sets",
    ]

    for i, (k, cnt) in enumerate(summary["most_selected_params"], start=1):
        lines.append(f"{i}. {k}, selected {cnt} folds")

    lines.append("")
    lines.append("## Fold Results")
    for r in folds:
        lines.append(
            f"{r['fold']}. OOS {r['oos_start']} to {r['oos_end']} | "
            f"bw={r['beta_window']}, z_entry={r['z_entry']}, z_exit={r['z_exit']} | "
            f"OOS Sharpe={r['oos_sharpe']:.4f}, OOS Return={r['oos_return']:.2%}, OOS MDD={r['oos_mdd']:.2%} | "
            f"OU hl(IS)={r['ou_half_life_is']:.2f}, kappa(IS)={r['ou_kappa_is']:.4f}"
        )

    lines.append("")
    lines.append(f"- Fold CSV: {csv_path.name}")
    lines.append(f"- Full JSON: {json_path.name}")

    # Render fold-level observability charts.
    fold_labels = [str(r["fold"]) for r in folds]
    oos_returns = [r["oos_return"] for r in folds]
    oos_sharpes = [r["oos_sharpe"] for r in folds]

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    x = list(range(len(folds)))
    ax1.bar(x, oos_returns, color=["#2a9d8f" if v >= 0 else "#e76f51" for v in oos_returns], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(fold_labels)
    ax1.axhline(0.0, color="#333", linewidth=0.8)
    ax1.set_title("Walk-Forward OOS Return by Fold")
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("OOS Return")
    ax1.grid(axis="y", alpha=0.25)
    fig1.tight_layout()
    chart_fold_return = base / f"{tag}_oos_return_by_fold.png"
    fig1.savefig(chart_fold_return, dpi=160)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(x, oos_sharpes, marker="o", color="#1d3557", linewidth=1.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(fold_labels)
    ax2.axhline(0.0, color="#333", linewidth=0.8)
    ax2.set_title("Walk-Forward OOS Sharpe by Fold")
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("OOS Sharpe")
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    chart_fold_sharpe = base / f"{tag}_oos_sharpe_by_fold.png"
    fig2.savefig(chart_fold_sharpe, dpi=160)
    plt.close(fig2)

    eq = []
    v = 1.0
    for r in combined_oos_returns:
        v *= 1.0 + r
        eq.append(v)
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(list(range(len(eq))), eq, color="#264653", linewidth=1.8)
    ax3.axhline(1.0, color="#555", linewidth=0.8)
    ax3.set_title("Combined OOS Equity Across Walk-Forward Folds")
    ax3.set_xlabel("OOS bar index (concatenated folds)")
    ax3.set_ylabel("Equity")
    ax3.grid(alpha=0.25)
    fig3.tight_layout()
    chart_oos_equity = base / f"{tag}_combined_oos_equity.png"
    fig3.savefig(chart_oos_equity, dpi=160)
    plt.close(fig3)

    lines.append(f"- Chart: {chart_fold_return.name}")
    lines.append(f"- Chart: {chart_fold_sharpe.name}")
    lines.append(f"- Chart: {chart_oos_equity.name}")

    md_path = base / f"{tag}_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "csv": csv_path.name,
                "json": json_path.name,
                "md": md_path.name,
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
