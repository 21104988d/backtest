#!/usr/bin/env python3
"""Run the full MAG7 vs Hyperliquid basis arb pipeline.

Steps:
  1. fetch_data.py       -- Fetch Yahoo + HL daily data
  2. check_hl_margin.py  -- Check cross-margin capabilities
  3. backtest_basis_arb.py -- Run backtest
  4. stress_test_fees.py   -- Fee/spread stress test

Usage:
  python run_pipeline.py
  python run_pipeline.py --start 2024-09-01 --coins AAPL,MSFT,NVDA
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full MAG7 basis arb pipeline.")
    parser.add_argument("--start", default="2024-09-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--coins", default=None, help="Subset of MAG7 tickers")
    parser.add_argument("--z-window", type=int, default=20)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--fee", type=float, default=0.00045)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-margin-check", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], step: str) -> bool:
    print(f"\n{'='*70}")
    print(f"STEP: {step}")
    print(f"CMD:  {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed: {step} (exit code {result.returncode})")
        return False
    return True


def main() -> None:
    args = parse_args()
    py = sys.executable

    # Step 1: Fetch data
    fetch_cmd = [py, "fetch_data.py", "--start", args.start, "--output-dir", args.output_dir]
    if args.end:
        fetch_cmd += ["--end", args.end]
    if args.coins:
        fetch_cmd += ["--coins", args.coins]
    if args.refresh:
        fetch_cmd.append("--refresh")

    if not run(fetch_cmd, "Fetch Yahoo + Hyperliquid daily data"):
        sys.exit(1)

    # Step 2: Margin check (optional)
    if not args.skip_margin_check:
        margin_cmd = [py, "check_hl_margin.py", "--output-dir", args.output_dir]
        if args.coins:
            margin_cmd += ["--coins", args.coins]
        if not run(margin_cmd, "Check Hyperliquid cross-margin capabilities"):
            print("[WARNING] Margin check failed, continuing...")

    # Step 3: Backtest
    backtest_cmd = [
        py, "backtest_basis_arb.py",
        "--input", f"{args.output_dir}/aligned_mag7_daily.csv",
        "--output-dir", args.output_dir,
        "--z-window", str(args.z_window),
        "--entry-z", str(args.entry_z),
        "--exit-z", str(args.exit_z),
        "--fee", str(args.fee),
    ]
    if not run(backtest_cmd, "Run basis arb backtest"):
        sys.exit(1)

    # Step 4: Stress test
    stress_cmd = [
        py, "stress_test_fees.py",
        "--input", f"{args.output_dir}/aligned_mag7_daily.csv",
        "--output-dir", args.output_dir,
        "--z-window", str(args.z_window),
        "--entry-z", str(args.entry_z),
        "--exit-z", str(args.exit_z),
    ]
    if not run(stress_cmd, "Fee and spread stress test"):
        sys.exit(1)

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutputs in {args.output_dir}/:")
    for f in sorted(Path(args.output_dir).glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
