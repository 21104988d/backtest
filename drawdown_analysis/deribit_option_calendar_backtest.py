from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

BASE_URL = "https://www.deribit.com/api/v2/public"
INSTRUMENT_PATTERN = re.compile(
    r"^(?P<underlying>[A-Z]+)-(?P<day>\d{1,2})(?P<mon>[A-Z]{3})(?P<year>\d{2})-(?P<strike>\d+(?:\.\d+)?)-(?P<cp>[CP])$"
)
MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


@dataclass
class OptionLeg:
    instrument: str
    strike: float
    expiry_date: datetime
    quantity: float


@dataclass
class CyclePosition:
    anchor_open: float
    strike: float
    opened_at: datetime
    long_leg: OptionLeg
    short_legs: List[OptionLeg] = field(default_factory=list)
    closed_at: Optional[datetime] = None
    close_reason: Optional[str] = None


class DeribitClient:
    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, endpoint: str, params: Dict) -> Dict:
        response = self.session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(f"Deribit error for {endpoint}: {payload['error']}")
        return payload.get("result", {})

    def get_instruments(self, currency: str, kind: str, include_expired: bool = True) -> List[Dict]:
        result = self._get(
            "get_instruments",
            {
                "currency": currency,
                "kind": kind,
                "include_expired": str(include_expired).lower(),
            },
        )
        return result

    def get_daily_ohlc(self, instrument_name: str, start_ts_ms: int, end_ts_ms: int) -> pd.DataFrame:
        result = self._get(
            "get_tradingview_chart_data",
            {
                "instrument_name": instrument_name,
                "start_timestamp": start_ts_ms,
                "end_timestamp": end_ts_ms,
                "resolution": "1D",
            },
        )
        if result.get("status") != "ok":
            return pd.DataFrame(columns=["date", "open", "high", "low", "close"])

        ticks = result.get("ticks", [])
        opens = result.get("open", [])
        highs = result.get("high", [])
        lows = result.get("low", [])
        closes = result.get("close", [])

        if not ticks:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close"])

        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ticks, unit="ms", utc=True),
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
            }
        )
        frame["date"] = frame["timestamp"].dt.date
        frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        return frame[["date", "open", "high", "low", "close"]]


def parse_option_instrument(name: str) -> Optional[Dict]:
    match = INSTRUMENT_PATTERN.match(name)
    if not match:
        return None
    year = 2000 + int(match.group("year"))
    month = MONTH_MAP.get(match.group("mon"))
    if not month:
        return None
    day = int(match.group("day"))
    expiry = datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)
    return {
        "instrument_name": name,
        "strike": float(match.group("strike")),
        "cp": "call" if match.group("cp") == "C" else "put",
        "expiry": expiry,
    }


def standard_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_delta(spot: float, strike: float, tau_years: float, sigma: float) -> float:
    if spot <= 0 or strike <= 0:
        return 0.0
    if tau_years <= 0:
        return 1.0 if spot > strike else 0.0
    if sigma <= 1e-8:
        return 1.0 if spot > strike else 0.0
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * tau_years) / (sigma * math.sqrt(tau_years))
    return standard_norm_cdf(d1)


def compute_expected_recovery_days(history: pd.DataFrame, bucket_floor: int) -> Optional[float]:
    records = []
    for idx in range(len(history) - 1):
        open_px = float(history.iloc[idx]["open"])
        close_px = float(history.iloc[idx]["close"])
        if close_px >= open_px:
            continue
        ret = (close_px - open_px) / open_px * 100.0
        recovered_days = None
        for j in range(idx + 1, len(history)):
            if float(history.iloc[j]["close"]) >= open_px:
                d0 = pd.Timestamp(history.iloc[idx]["date"])
                d1 = pd.Timestamp(history.iloc[j]["date"])
                recovered_days = int((d1 - d0).days)
                break
        if recovered_days is not None:
            records.append((math.floor(ret), recovered_days))

    if not records:
        return None

    bucket_map: Dict[int, List[int]] = {}
    for b, days in records:
        bucket_map.setdefault(b, []).append(days)

    if bucket_floor in bucket_map and bucket_map[bucket_floor]:
        return float(sum(bucket_map[bucket_floor]) / len(bucket_map[bucket_floor]))

    nearest_bucket = min(bucket_map.keys(), key=lambda x: abs(x - bucket_floor))
    vals = bucket_map[nearest_bucket]
    return float(sum(vals) / len(vals))


def choose_expiry(
    option_catalog: pd.DataFrame,
    trade_date: datetime,
    strike: float,
    min_days: int,
    target_days: Optional[float] = None,
    strict_strike: bool = True,
) -> Optional[pd.Series]:
    df = option_catalog[(option_catalog["cp"] == "call") & (option_catalog["expiry"] > trade_date)]
    if strict_strike:
        df = df[df["strike"] == strike]
    else:
        if df.empty:
            return None
        nearest = min(df["strike"].unique(), key=lambda x: abs(x - strike))
        df = df[df["strike"] == nearest]

    if df.empty:
        return None

    min_expiry = trade_date + timedelta(days=min_days)
    df = df[df["expiry"] >= min_expiry]
    if df.empty:
        return None

    if target_days is None:
        return df.sort_values("expiry").iloc[0]

    target_expiry = trade_date + timedelta(days=float(target_days))
    later = df[df["expiry"] >= target_expiry].sort_values("expiry")
    if not later.empty:
        return later.iloc[0]
    return df.sort_values("expiry").iloc[-1]


def normalize_option_price(raw_price: float, spot: float) -> float:
    if pd.isna(raw_price):
        return float("nan")
    if raw_price < 5:
        return float(raw_price * spot)
    return float(raw_price)


def get_price_on_date(cache: Dict[str, pd.DataFrame], instrument: str, date_obj) -> Optional[float]:
    frame = cache.get(instrument)
    if frame is None or frame.empty:
        return None
    row = frame.loc[frame["date"] == date_obj]
    if row.empty:
        return None
    return float(row.iloc[0]["close"])


def prefetch_option_history(
    client: DeribitClient,
    instruments: List[str],
    start_ts_ms: int,
    end_ts_ms: int,
) -> Dict[str, pd.DataFrame]:
    history = {}
    for name in instruments:
        history[name] = client.get_daily_ohlc(name, start_ts_ms, end_ts_ms)
    return history


def close_cycle_legs(
    cycle: CyclePosition,
    option_cache: Dict[str, pd.DataFrame],
    cash_usd: float,
    date_obj,
    spot: float,
) -> float:
    all_legs = [cycle.long_leg] + cycle.short_legs
    for leg in all_legs:
        px = get_price_on_date(option_cache, leg.instrument, date_obj)
        if px is None:
            intrinsic = max(spot - leg.strike, 0.0)
            px_usd = intrinsic
        else:
            px_usd = normalize_option_price(px, spot)
        cash_usd += leg.quantity * px_usd
    return cash_usd


def run_backtest(
    currency: str,
    start_date: str,
    end_date: str,
    strict_strike: bool,
    output_dir: Path,
) -> Dict:
    client = DeribitClient()
    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    spot = client.get_daily_ohlc(f"{currency}-PERPETUAL", start_ts, end_ts)
    spot = spot.dropna(subset=["open", "close"]).sort_values("date").reset_index(drop=True)
    if len(spot) < 10:
        raise RuntimeError("Insufficient spot history for backtest.")

    raw_instruments = client.get_instruments(currency=currency, kind="option", include_expired=True)
    parsed = []
    for row in raw_instruments:
        info = parse_option_instrument(row.get("instrument_name", ""))
        if info is not None:
            parsed.append(info)
    option_catalog = pd.DataFrame(parsed)
    if option_catalog.empty:
        raise RuntimeError("No parsable option instruments from Deribit.")

    option_catalog = option_catalog.sort_values(["expiry", "strike"]).reset_index(drop=True)

    cash_usd = 0.0
    hedge_units = 0.0
    active_cycles: List[CyclePosition] = []
    equity_curve = []
    trade_log = []

    prefetched = {}

    for i in range(2, len(spot)):
        t_date = pd.Timestamp(spot.iloc[i]["date"]).to_pydatetime().replace(tzinfo=timezone.utc)
        t_day = spot.iloc[i]["date"]
        t_spot_open = float(spot.iloc[i]["open"])
        t_spot_close = float(spot.iloc[i]["close"])

        prev_open = float(spot.iloc[i - 1]["open"])
        prev_close = float(spot.iloc[i - 1]["close"])
        prev_ret = (prev_close - prev_open) / prev_open * 100.0

        remaining_cycles = []
        for cycle in active_cycles:
            live_short_legs = []
            for leg in cycle.short_legs:
                if t_date >= leg.expiry_date:
                    px_raw = get_price_on_date(prefetched, leg.instrument, t_day)
                    if px_raw is None:
                        px_usd = max(t_spot_close - leg.strike, 0.0)
                    else:
                        px_usd = normalize_option_price(px_raw, t_spot_close)
                    cash_usd += leg.quantity * px_usd
                    trade_log.append(
                        {
                            "date": t_day,
                            "event": "short_expire_settlement",
                            "instrument": leg.instrument,
                            "strike": leg.strike,
                            "settle_usd": px_usd,
                        }
                    )
                else:
                    live_short_legs.append(leg)
            cycle.short_legs = live_short_legs

            recovered = t_spot_close >= cycle.anchor_open
            expired = t_date >= cycle.long_leg.expiry_date
            if recovered or expired:
                reason = "recovered" if recovered else "long_expired"
                cash_usd = close_cycle_legs(cycle, prefetched, cash_usd, t_day, t_spot_close)
                cycle.closed_at = t_date
                cycle.close_reason = reason
                trade_log.append(
                    {
                        "date": t_day,
                        "event": "cycle_close",
                        "reason": reason,
                        "strike": cycle.strike,
                        "anchor_open": cycle.anchor_open,
                        "long_instrument": cycle.long_leg.instrument,
                    }
                )
            else:
                remaining_cycles.append(cycle)
        active_cycles = remaining_cycles

        if prev_ret < 0:
            history = spot.iloc[:i].copy()
            bucket_floor = math.floor(prev_ret)
            exp_days = compute_expected_recovery_days(history, bucket_floor)
            if exp_days is not None:
                strike = prev_open
                long_row = choose_expiry(
                    option_catalog,
                    t_date,
                    strike,
                    min_days=2,
                    target_days=exp_days,
                    strict_strike=strict_strike,
                )
                short_row = choose_expiry(
                    option_catalog,
                    t_date,
                    strike,
                    min_days=2,
                    target_days=None,
                    strict_strike=strict_strike,
                )

                if long_row is not None and short_row is not None:
                    if pd.Timestamp(short_row["expiry"]).to_pydatetime() >= pd.Timestamp(long_row["expiry"]).to_pydatetime():
                        same_or_later = option_catalog[
                            (option_catalog["cp"] == "call")
                            & (option_catalog["strike"] == float(short_row["strike"]))
                            & (option_catalog["expiry"] < long_row["expiry"])
                            & (option_catalog["expiry"] >= (t_date + timedelta(days=2)))
                        ].sort_values("expiry")
                        if not same_or_later.empty:
                            short_row = same_or_later.iloc[0]

                    selected_instruments = [long_row["instrument_name"], short_row["instrument_name"]]
                    for inst in selected_instruments:
                        if inst not in prefetched:
                            prefetched[inst] = client.get_daily_ohlc(inst, start_ts, end_ts)

                    long_px_raw = get_price_on_date(prefetched, long_row["instrument_name"], t_day)
                    short_px_raw = get_price_on_date(prefetched, short_row["instrument_name"], t_day)

                    if long_px_raw is not None and short_px_raw is not None:
                        long_px = normalize_option_price(long_px_raw, t_spot_close)
                        short_px = normalize_option_price(short_px_raw, t_spot_close)

                        cash_usd -= long_px
                        cash_usd += short_px

                        cycle = CyclePosition(
                            anchor_open=strike,
                            strike=float(long_row["strike"]),
                            opened_at=t_date,
                            long_leg=OptionLeg(
                                instrument=long_row["instrument_name"],
                                strike=float(long_row["strike"]),
                                expiry_date=pd.Timestamp(long_row["expiry"]).to_pydatetime(),
                                quantity=1.0,
                            ),
                            short_legs=[
                                OptionLeg(
                                    instrument=short_row["instrument_name"],
                                    strike=float(short_row["strike"]),
                                    expiry_date=pd.Timestamp(short_row["expiry"]).to_pydatetime(),
                                    quantity=-1.0,
                                )
                            ],
                        )
                        active_cycles.append(cycle)
                        trade_log.append(
                            {
                                "date": t_day,
                                "event": "cycle_open",
                                "prev_return_pct": prev_ret,
                                "expected_recovery_days": exp_days,
                                "requested_strike": strike,
                                "selected_strike": float(long_row["strike"]),
                                "long_instrument": long_row["instrument_name"],
                                "short_instrument": short_row["instrument_name"],
                                "long_price_usd": long_px,
                                "short_price_usd": short_px,
                            }
                        )

            for cycle in active_cycles:
                if t_spot_close >= cycle.anchor_open:
                    continue
                today_ret = (t_spot_close - t_spot_open) / t_spot_open * 100.0
                if today_ret < 0:
                    short_roll = choose_expiry(
                        option_catalog,
                        t_date,
                        cycle.strike,
                        min_days=2,
                        target_days=None,
                        strict_strike=True,
                    )
                    if short_roll is not None:
                        inst = short_roll["instrument_name"]
                        if inst not in prefetched:
                            prefetched[inst] = client.get_daily_ohlc(inst, start_ts, end_ts)
                        px_raw = get_price_on_date(prefetched, inst, t_day)
                        if px_raw is not None:
                            px = normalize_option_price(px_raw, t_spot_close)
                            cash_usd += px
                            cycle.short_legs.append(
                                OptionLeg(
                                    instrument=inst,
                                    strike=cycle.strike,
                                    expiry_date=pd.Timestamp(short_roll["expiry"]).to_pydatetime(),
                                    quantity=-1.0,
                                )
                            )
                            trade_log.append(
                                {
                                    "date": t_day,
                                    "event": "add_short_call",
                                    "strike": cycle.strike,
                                    "instrument": inst,
                                    "price_usd": px,
                                }
                            )

        spot_returns = spot.iloc[: i + 1]["close"].pct_change().dropna()
        if len(spot_returns) >= 20:
            sigma = float(spot_returns.tail(30).std() * math.sqrt(365))
        else:
            sigma = 0.6

        net_option_delta = 0.0
        option_mtm = 0.0
        for cycle in active_cycles:
            legs = [cycle.long_leg] + cycle.short_legs
            for leg in legs:
                px_raw = get_price_on_date(prefetched, leg.instrument, t_day)
                if px_raw is None:
                    px_usd = max(t_spot_close - leg.strike, 0.0)
                else:
                    px_usd = normalize_option_price(px_raw, t_spot_close)
                option_mtm += leg.quantity * px_usd

                tau_days = max((leg.expiry_date - t_date).days, 0)
                tau_years = tau_days / 365.0
                delta = bs_call_delta(t_spot_close, leg.strike, tau_years, sigma)
                net_option_delta += leg.quantity * delta

        target_hedge = -net_option_delta
        hedge_trade = target_hedge - hedge_units
        cash_usd -= hedge_trade * t_spot_close
        hedge_units = target_hedge

        total_equity = cash_usd + option_mtm + hedge_units * t_spot_close
        equity_curve.append(
            {
                "date": t_day,
                "equity_usd": total_equity,
                "cash_usd": cash_usd,
                "option_mtm_usd": option_mtm,
                "hedge_units": hedge_units,
                "spot_close": t_spot_close,
                "active_cycles": len(active_cycles),
            }
        )

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trade_log)

    output_dir.mkdir(parents=True, exist_ok=True)
    equity_path = output_dir / "deribit_calendar_equity.csv"
    trades_path = output_dir / "deribit_calendar_trades.csv"
    equity_df.to_csv(equity_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    if equity_df.empty:
        summary = {
            "start": start_date,
            "end": end_date,
            "trades": 0,
            "final_equity_usd": 0.0,
            "return_pct": 0.0,
            "strict_strike": strict_strike,
            "equity_csv": str(equity_path),
            "trades_csv": str(trades_path),
        }
    else:
        start_equity = float(equity_df.iloc[0]["equity_usd"])
        final_equity = float(equity_df.iloc[-1]["equity_usd"])
        denom = abs(start_equity) if abs(start_equity) > 1e-9 else 1.0
        summary = {
            "start": start_date,
            "end": end_date,
            "trades": int((trades_df["event"] == "cycle_open").sum()) if not trades_df.empty else 0,
            "final_equity_usd": final_equity,
            "return_pct": (final_equity - start_equity) / denom * 100.0,
            "strict_strike": strict_strike,
            "equity_csv": str(equity_path),
            "trades_csv": str(trades_path),
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Deribit option calendar backtest based on negative candle recovery.")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-12-15")
    parser.add_argument("--allow-nearest-strike", action="store_true", help="Allow nearest listed strike instead of strict K = T-1 open.")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    summary = run_backtest(
        currency=args.currency,
        start_date=args.start,
        end_date=args.end,
        strict_strike=not args.allow_nearest_strike,
        output_dir=Path(args.output_dir),
    )

    print("Backtest finished")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
