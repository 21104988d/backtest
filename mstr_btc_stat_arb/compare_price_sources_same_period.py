import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import optimize_oos_20_regime as core


def load_hyperliquid_csv(path: Path):
    if not path.exists():
        raise RuntimeError(f"Missing Hyperliquid cache file: {path}")
    df = pd.read_csv(path)
    if "time" not in df.columns or "close" not in df.columns:
        raise RuntimeError(f"Unexpected columns in {path}")
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(df["time"], utc=True, errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
        }
    ).dropna()
    return out.sort_values("datetime").reset_index(drop=True)


def aggregate_4h_to_daily_close(df_4h: pd.DataFrame):
    x = df_4h.copy()
    x["date"] = x["datetime"].dt.floor("D")
    out = x.groupby("date", as_index=False).tail(1).copy()
    out = out[["date", "close"]].rename(columns={"close": "hl4h_daily_close"})
    return out.reset_index(drop=True)


def load_yahoo_close(ticker: str, start_date: str, end_date: str, out_col: str):
    df = core.fetch_yahoo_close(ticker, start_date, end_date)
    if df.empty:
        raise RuntimeError(f"Failed to fetch Yahoo data: {ticker}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize("UTC")
    out = out.rename(columns={"close": out_col})[["date", out_col]]
    return out.sort_values("date").reset_index(drop=True)


def series_stats(a: pd.Series, b: pd.Series):
    z = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(z) < 3:
        return {
            "rows": int(len(z)),
            "corr": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "mean_abs_pct_diff": float("nan"),
        }

    d = z["a"] - z["b"]
    abs_pct = (d.abs() / z["b"].replace(0.0, np.nan)).dropna()

    return {
        "rows": int(len(z)),
        "corr": float(z["a"].corr(z["b"])),
        "mae": float(d.abs().mean()),
        "rmse": float(np.sqrt((d**2).mean())),
        "mean_abs_pct_diff": float(abs_pct.mean()) if len(abs_pct) else float("nan"),
    }


def return_stats(a: pd.Series, b: pd.Series):
    z = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(z) < 3:
        return {
            "rows": int(len(z)),
            "corr": float("nan"),
            "mean_diff": float("nan"),
            "std_diff": float("nan"),
            "rmse_diff": float("nan"),
            "cum_a": float("nan"),
            "cum_b": float("nan"),
        }

    diff = z["a"] - z["b"]
    return {
        "rows": int(len(z)),
        "corr": float(z["a"].corr(z["b"])),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else float("nan"),
        "rmse_diff": float(np.sqrt((diff**2).mean())),
        "cum_a": float((1.0 + z["a"]).prod() - 1.0),
        "cum_b": float((1.0 + z["b"]).prod() - 1.0),
    }


def main():
    base = Path(__file__).resolve().parent

    m4_path = base / "xyz:mstr_4h_rolling_beta_metrics.json"
    if not m4_path.exists():
        raise RuntimeError("Missing 4H reference metrics JSON")
    m4 = json.loads(m4_path.read_text(encoding="utf-8"))

    start_dt = datetime.strptime(m4["start_date"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(m4["end_date"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    start_date = start_dt.date().isoformat()
    end_date = end_dt.date().isoformat()
    yahoo_end_exclusive = (end_dt.date() + timedelta(days=1)).isoformat()

    # Hyperliquid cached prices.
    mstr_4h = load_hyperliquid_csv(base / "data/xyz_mstr_4h.csv")
    btc_4h = load_hyperliquid_csv(base / "data/btc_4h.csv")
    mstr_1d = load_hyperliquid_csv(base / "data/xyz_mstr_1d.csv")
    btc_1d = load_hyperliquid_csv(base / "data/btc_1d.csv")

    # Restrict to same calendar window.
    mstr_4h = mstr_4h[(mstr_4h["datetime"] >= start_dt) & (mstr_4h["datetime"] <= end_dt)].copy()
    btc_4h = btc_4h[(btc_4h["datetime"] >= start_dt) & (btc_4h["datetime"] <= end_dt)].copy()
    mstr_1d = mstr_1d[(mstr_1d["datetime"] >= start_dt) & (mstr_1d["datetime"] <= end_dt)].copy()
    btc_1d = btc_1d[(btc_1d["datetime"] >= start_dt) & (btc_1d["datetime"] <= end_dt)].copy()

    mstr_4h_day = aggregate_4h_to_daily_close(mstr_4h)
    btc_4h_day = aggregate_4h_to_daily_close(btc_4h)
    mstr_1d_day = mstr_1d.assign(date=mstr_1d["datetime"].dt.floor("D"))[["date", "close"]].rename(columns={"close": "hl1d_close"})
    btc_1d_day = btc_1d.assign(date=btc_1d["datetime"].dt.floor("D"))[["date", "close"]].rename(columns={"close": "hl1d_close"})

    # Yahoo prices in the same date range.
    y_mstr = load_yahoo_close("MSTR", start_date, yahoo_end_exclusive, "y_mstr_close")
    y_btc = load_yahoo_close("BTC-USD", start_date, yahoo_end_exclusive, "y_btc_close")

    mstr = (
        mstr_4h_day.merge(mstr_1d_day, on="date", how="outer")
        .merge(y_mstr, on="date", how="outer")
        .sort_values("date")
        .reset_index(drop=True)
    )
    btc = (
        btc_4h_day.merge(btc_1d_day, on="date", how="outer")
        .merge(y_btc, on="date", how="outer")
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Keep only the exact same-period calendar window.
    s_utc = pd.Timestamp(start_date).tz_localize("UTC")
    e_utc = pd.Timestamp(end_date).tz_localize("UTC")
    mstr = mstr[(mstr["date"] >= s_utc) & (mstr["date"] <= e_utc)].copy()
    btc = btc[(btc["date"] >= s_utc) & (btc["date"] <= e_utc)].copy()

    # Price-level comparisons.
    mstr_price_hl1d_vs_y = series_stats(mstr["hl1d_close"], mstr["y_mstr_close"])
    mstr_price_hl4d_vs_y = series_stats(mstr["hl4h_daily_close"], mstr["y_mstr_close"])
    btc_price_hl1d_vs_y = series_stats(btc["hl1d_close"], btc["y_btc_close"])

    # Return-level comparisons.
    for col in ["hl4h_daily_close", "hl1d_close", "y_mstr_close"]:
        mstr[f"ret_{col}"] = mstr[col].pct_change(fill_method=None)
    for col in ["hl4h_daily_close", "hl1d_close", "y_btc_close"]:
        btc[f"ret_{col}"] = btc[col].pct_change(fill_method=None)

    mstr_ret_hl1d_vs_y = return_stats(mstr["ret_hl1d_close"], mstr["ret_y_mstr_close"])
    mstr_ret_hl4d_vs_y = return_stats(mstr["ret_hl4h_daily_close"], mstr["ret_y_mstr_close"])
    btc_ret_hl1d_vs_y = return_stats(btc["ret_hl1d_close"], btc["ret_y_btc_close"])

    # Normalized series chart.
    mstr_plot = mstr[["date", "hl4h_daily_close", "hl1d_close", "y_mstr_close"]].dropna()
    if not mstr_plot.empty:
        for c in ["hl4h_daily_close", "hl1d_close", "y_mstr_close"]:
            mstr_plot[f"n_{c}"] = mstr_plot[c] / mstr_plot[c].iloc[0]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(mstr_plot["date"], mstr_plot["n_hl4h_daily_close"], label="HL 4H->Daily", color="#1f77b4")
        ax.plot(mstr_plot["date"], mstr_plot["n_hl1d_close"], label="HL 1D", color="#2ca02c")
        ax.plot(mstr_plot["date"], mstr_plot["n_y_mstr_close"], label="Yahoo Daily", color="#d62728")
        ax.set_title("MSTR Price Sources (Normalized, Same Period)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Price")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(base / "price_source_mstr_same_period.png", dpi=160)
        plt.close(fig)

    # Return gap chart.
    gap_plot = mstr[["date", "ret_hl1d_close", "ret_y_mstr_close"]].dropna().copy()
    if not gap_plot.empty:
        gap_plot["ret_gap_hl1d_minus_y"] = gap_plot["ret_hl1d_close"] - gap_plot["ret_y_mstr_close"]
        fig2, ax2 = plt.subplots(figsize=(12, 4.5))
        ax2.bar(gap_plot["date"], gap_plot["ret_gap_hl1d_minus_y"], color="#9467bd", width=0.8)
        ax2.axhline(0.0, color="#333", linewidth=0.8)
        ax2.set_title("Daily Return Gap: Hyperliquid MSTR 1D - Yahoo MSTR")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Return Gap")
        ax2.grid(axis="y", alpha=0.2)
        fig2.tight_layout()
        fig2.savefig(base / "price_source_mstr_return_gap_same_period.png", dpi=160)
        plt.close(fig2)

    result = {
        "period": {"start": start_date, "end": end_date},
        "mstr_price_compare": {
            "hl1d_vs_yahoo": mstr_price_hl1d_vs_y,
            "hl4h_daily_vs_yahoo": mstr_price_hl4d_vs_y,
        },
        "mstr_return_compare": {
            "hl1d_vs_yahoo": mstr_ret_hl1d_vs_y,
            "hl4h_daily_vs_yahoo": mstr_ret_hl4d_vs_y,
        },
        "btc_price_compare": {
            "hl1d_vs_yahoo": btc_price_hl1d_vs_y,
        },
        "btc_return_compare": {
            "hl1d_vs_yahoo": btc_ret_hl1d_vs_y,
        },
        "row_counts": {
            "mstr_days": int(len(mstr)),
            "btc_days": int(len(btc)),
        },
    }

    out_json = base / "price_source_same_period_summary.json"
    out_md = base / "price_source_same_period_report.md"

    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Price Source Comparison (Same Period)")
    lines.append("")
    lines.append(f"- Period: {start_date} to {end_date}")
    lines.append("- Compare Hyperliquid (1D and 4H aggregated to daily) vs Yahoo daily")
    lines.append("")

    mp = result["mstr_price_compare"]["hl1d_vs_yahoo"]
    mr = result["mstr_return_compare"]["hl1d_vs_yahoo"]
    lines.append("## MSTR: Hyperliquid 1D vs Yahoo Daily")
    lines.append(
        f"- Price corr={mp['corr']:.4f}, mean abs pct diff={mp['mean_abs_pct_diff']:.2%}, MAE={mp['mae']:.4f}"
    )
    lines.append(
        f"- Return corr={mr['corr']:.4f}, mean return diff={mr['mean_diff']:.4%}, RMSE diff={mr['rmse_diff']:.4%}"
    )
    lines.append(
        f"- Cum return HL1D={mr['cum_a']:.2%} vs Yahoo={mr['cum_b']:.2%}"
    )

    mr4 = result["mstr_return_compare"]["hl4h_daily_vs_yahoo"]
    lines.append("")
    lines.append("## MSTR: Hyperliquid 4H->Daily vs Yahoo Daily")
    lines.append(
        f"- Return corr={mr4['corr']:.4f}, mean return diff={mr4['mean_diff']:.4%}, RMSE diff={mr4['rmse_diff']:.4%}"
    )
    lines.append(
        f"- Cum return HL4H(daily-close)={mr4['cum_a']:.2%} vs Yahoo={mr4['cum_b']:.2%}"
    )

    br = result["btc_return_compare"]["hl1d_vs_yahoo"]
    lines.append("")
    lines.append("## BTC: Hyperliquid 1D vs Yahoo Daily")
    lines.append(
        f"- Return corr={br['corr']:.4f}, mean return diff={br['mean_diff']:.4%}, RMSE diff={br['rmse_diff']:.4%}"
    )

    lines.append("")
    lines.append("- Chart: price_source_mstr_same_period.png")
    lines.append("- Chart: price_source_mstr_return_gap_same_period.png")
    lines.append(f"- JSON: {out_json.name}")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
