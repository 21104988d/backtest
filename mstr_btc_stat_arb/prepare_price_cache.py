import argparse
import json
import time
from pathlib import Path

from rolling_beta_stat_arb import load_candles_with_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="xyz:MSTR")
    parser.add_argument("--btc", default="BTC")
    parser.add_argument("--intervals", default="1h,4h,1d")
    parser.add_argument("--start-ms", type=int, default=0)
    parser.add_argument("--end-ms", type=int, default=0)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    end_ms = args.end_ms if args.end_ms > 0 else int(time.time() * 1000)
    intervals = [x.strip() for x in args.intervals.split(",") if x.strip()]

    summary = {
        "asset": args.asset,
        "btc": args.btc,
        "start_ms": args.start_ms,
        "end_ms": end_ms,
        "cache_dir": str((base / args.cache_dir).resolve()),
        "intervals": {},
    }

    for interval in intervals:
        a_rows, a_source, a_path = load_candles_with_cache(
            symbol=args.asset,
            interval=interval,
            start_ms=args.start_ms,
            end_ms=end_ms,
            cache_dir=(base / args.cache_dir).resolve(),
            refresh_cache=args.refresh_cache,
            allow_network=True,
        )
        b_rows, b_source, b_path = load_candles_with_cache(
            symbol=args.btc,
            interval=interval,
            start_ms=args.start_ms,
            end_ms=end_ms,
            cache_dir=(base / args.cache_dir).resolve(),
            refresh_cache=args.refresh_cache,
            allow_network=True,
        )

        summary["intervals"][interval] = {
            "asset_rows": len(a_rows),
            "btc_rows": len(b_rows),
            "asset_source": a_source,
            "btc_source": b_source,
            "asset_cache_file": str(a_path),
            "btc_cache_file": str(b_path),
        }

    out = base / "price_cache_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
