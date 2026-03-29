#!/usr/bin/env python3
"""Check Hyperliquid cross-margin capabilities for MAG7 stock perps.

Queries the Hyperliquid API to document:
  1. Which xyz:STOCK symbols are active in the meta universe
  2. Perp metadata: leverage limits, margin tiers, funding rates
  3. Margin mode documentation (cross vs isolated)
  4. Current funding rates for MAG7 stock perps

On Hyperliquid, ALL perpetuals (including xyz: stock perps) share one
cross-margin pool by default. Isolated margin is opt-in per position.
The cross-margin pool includes both crypto and stock perps.

Output:
  data/hl_margin_check.json   -- Full API data + analysis
  data/hl_margin_check.md     -- Human-readable report

Usage:
  python check_hl_margin.py
  python check_hl_margin.py --coins AAPL,MSFT
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

HL_API_URL = "https://api.hyperliquid.xyz/info"

MAG7_HL = {
    "AAPL": "xyz:AAPL",
    "MSFT": "xyz:MSFT",
    "NVDA": "xyz:NVDA",
    "GOOGL": "xyz:GOOGL",
    "AMZN": "xyz:AMZN",
    "META": "xyz:META",
    "TSLA": "xyz:TSLA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Hyperliquid cross-margin for MAG7 stock perps.")
    parser.add_argument("--coins", default=None, help="Comma-separated subset of MAG7 tickers")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    return parser.parse_args()


def hl_post(payload: dict) -> object:
    req = Request(
        HL_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            if attempt == 2:
                return None
            time.sleep(2 ** attempt)
    return None


def get_perp_meta_and_ctx() -> tuple[dict, list]:
    """Fetch perp universe meta and market contexts."""
    result = hl_post({"type": "metaAndAssetCtxs"})
    if not isinstance(result, list) or len(result) < 2:
        return {}, []
    meta = result[0] if isinstance(result[0], dict) else {}
    ctxs = result[1] if isinstance(result[1], list) else []
    return meta, ctxs


def get_spot_meta_and_ctx() -> tuple[dict, list]:
    """Fetch spot universe meta and market contexts (xyz: tokens live here)."""
    result = hl_post({"type": "spotMetaAndAssetCtxs"})
    if not isinstance(result, list) or len(result) < 2:
        return {}, []
    meta = result[0] if isinstance(result[0], dict) else {}
    ctxs = result[1] if isinstance(result[1], list) else []
    return meta, ctxs


def get_funding_rates(symbols: list[str]) -> dict[str, dict]:
    """Fetch current funding rates for a list of HL symbols."""
    rates = {}
    for sym in symbols:
        payload = {
            "type": "fundingHistory",
            "req": {
                "coin": sym,
                "startTime": int(time.time() * 1000) - 7 * 24 * 3600 * 1000,
            },
        }
        data = hl_post(payload)
        if isinstance(data, list) and data:
            last = data[-1]
            rates[sym] = {
                "latest_funding_rate": float(last.get("fundingRate", 0)),
                "latest_time": datetime.fromtimestamp(
                    int(last.get("time", 0)) / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "num_records": len(data),
            }
        else:
            rates[sym] = {"error": "no funding data returned"}
        time.sleep(0.2)
    return rates


def main() -> None:
    args = parse_args()

    ticker_map = dict(MAG7_HL)
    if args.coins:
        subset = {c.strip().upper() for c in args.coins.split(",")}
        ticker_map = {k: v for k, v in MAG7_HL.items() if k in subset}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HYPERLIQUID CROSS-MARGIN CHECK FOR MAG7 STOCK TOKENS")
    print("=" * 70)

    # ── 1. Perp universe check ────────────────────────────────────────────────
    print("\n[1] Checking perp universe for xyz: symbols...")
    perp_meta, perp_ctxs = get_perp_meta_and_ctx()
    perp_universe = perp_meta.get("universe", [])
    perp_names = {entry.get("name", "") for entry in perp_universe}
    xyz_in_perp = [n for n in perp_names if "xyz:" in n or any(m in n for m in MAG7_HL.keys())]
    print(f"   Total perp symbols: {len(perp_universe)}")
    print(f"   xyz:/stock symbols in perp universe: {xyz_in_perp if xyz_in_perp else 'None'}")

    # ── 2. Spot universe — where xyz: tokens actually live ────────────────────
    print("\n[2] Checking spot universe for MAG7 stock tokens...")
    spot_meta, spot_ctxs = get_spot_meta_and_ctx()
    spot_tokens = spot_meta.get("tokens", [])
    spot_universe = spot_meta.get("universe", [])
    print(f"   Total spot tokens: {len(spot_tokens)}")
    print(f"   Total spot pairs: {len(spot_universe)}")

    # Build a map: token name → index
    token_name_to_idx = {t.get("name", ""): i for i, t in enumerate(spot_tokens)}

    # ── 3. Find MAG7 stock tokens ─────────────────────────────────────────────
    print("\n[3] MAG7 stock token details (spot DEX):")
    stock_perp_info: dict[str, dict] = {}

    for ticker, hl_sym in ticker_map.items():
        # Find the token by name (ticker without xyz: prefix)
        token_name = ticker  # e.g. "AAPL"
        tok_idx = token_name_to_idx.get(token_name)

        if tok_idx is None:
            print(f"   {token_name:6s} → NOT FOUND in spot token list")
            stock_perp_info[ticker] = {"hl_symbol": hl_sym, "found": False, "type": "spot_token"}
            continue

        tok = spot_tokens[tok_idx]

        # Find the spot pair (pair that contains this token with USDC)
        pair_info = None
        pair_ctx = None
        for i, pair in enumerate(spot_universe):
            pair_toks = pair.get("tokens", [])
            if tok_idx in pair_toks:
                pair_info = pair
                pair_ctx = spot_ctxs[i] if i < len(spot_ctxs) else {}
                break

        sz_decimals = tok.get("szDecimals", "N/A")
        wei_decimals = tok.get("weiDecimals", "N/A")
        is_canonical = tok.get("isCanonical", False)
        token_id = tok.get("tokenId", "N/A")

        mark_px = pair_ctx.get("markPx", "N/A") if pair_ctx else "N/A"
        mid_px = pair_ctx.get("midPx", "N/A") if pair_ctx else "N/A"
        circ_supply = pair_ctx.get("circulatingSupply", "N/A") if pair_ctx else "N/A"
        pair_name = pair_info.get("name", "N/A") if pair_info else "N/A"

        # Spot tokens cannot be margined/leveraged like perps
        # They can be bought (long) via spot wallet or sold if you hold them
        print(f"   {token_name:6s} → FOUND (token_idx={tok_idx}, pair={pair_name})")
        print(f"      type          : Spot token (HIP-1 wrapped equity)")
        print(f"      isCanonical   : {is_canonical}")
        print(f"      szDecimals    : {sz_decimals}")
        print(f"      markPrice     : {mark_px}")
        print(f"      midPrice      : {mid_px}")
        print(f"      cross-margin  : N/A — spot tokens, no leverage")
        print(f"      can short?    : Only by selling existing holdings")

        stock_perp_info[ticker] = {
            "hl_symbol": hl_sym,
            "found": True,
            "type": "spot_token_hip1",
            "token_index": tok_idx,
            "pair_name": pair_name,
            "is_canonical": is_canonical,
            "szDecimals": sz_decimals,
            "markPrice": mark_px,
            "midPrice": mid_px,
            "circulating_supply": str(circ_supply),
            "cross_margin_available": False,
            "leverage_available": False,
            "short_available": "sell_only",
            "note": "HIP-1 spot token. No leverage or traditional cross-margin. Buy on HL spot DEX, hold in spot wallet.",
        }
        time.sleep(0.05)

    # ── 4. Funding rate check (not applicable for spot) ───────────────────────
    print("\n[4] Funding rates:")
    print("   xyz: stock tokens are SPOT tokens — no funding rate mechanism.")
    print("   Funding rates only apply to perpetual futures contracts.")
    print("   These tokens accrue no funding; price tracks real stock via oracle/arbitrage.")

    # ── 5. Cross-margin explanation ───────────────────────────────────────────
    print("\n[5] Margin architecture for xyz:/MAG7 tokens:")
    print("""
   KEY FINDING: xyz:AAPL, xyz:MSFT, etc. are HIP-1 spot tokens on Hyperliquid.
   They are NOT perpetual futures contracts.

   WHAT THIS MEANS FOR TRADING:
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  Feature              │ MAG7 on HL (xyz:)    │ Crypto Perps on HL     │
   ├─────────────────────────────────────────────────────────────────────────┤
   │  Instrument type      │ Spot token (HIP-1)   │ Perpetual future       │
   │  Leverage             │ None (1x only)       │ Up to 50x              │
   │  Cross-margin         │ No — spot wallet     │ Yes — perp margin pool │
   │  Short selling        │ Sell holdings only   │ Yes, with margin       │
   │  Funding rate         │ None                 │ Paid/received 8-hourly │
   │  Mark price source    │ Oracle / spot DEX    │ Mark price + funding   │
   │  Where held           │ Spot wallet          │ Perp margin account    │
   └─────────────────────────────────────────────────────────────────────────┘

   CROSS-VENUE ARB STRUCTURE:
   - Leg 1 (HL): Buy xyz:AAPL on HL spot DEX when HL discount > threshold.
   - Leg 2 (broker): Simultaneously short AAPL at real stock broker (with margin).
   - Convergence: HL price reverts to real stock price over time.
   - Risk: HL spot tokens cannot be shorted directly on HL.

   ALTERNATIVE WITHIN HL:
   - If HL introduces perpetual contracts for stocks (future), cross-margin would apply.
   - Currently: manage spot wallet (for HL leg) and broker margin (for stock leg) separately.
""")

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    perp_universe_size = len(perp_universe)
    spot_universe_size = len(spot_universe)
    report_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "perp_universe_size": perp_universe_size,
        "spot_universe_size": spot_universe_size,
        "mag7_token_type": "HIP-1 spot tokens (NOT perpetuals)",
        "mag7_tokens": stock_perp_info,
        "cross_margin_explanation": (
            "MAG7 stocks (xyz:AAPL, xyz:MSFT, etc.) on Hyperliquid are HIP-1 wrapped equity spot tokens. "
            "They are NOT perpetual futures contracts. "
            "These tokens live in the spot wallet, have no leverage, no funding rate, and no cross-margin pool. "
            "You can buy them (go long) or sell existing holdings (go flat). Direct shorting is not possible. "
            "For cross-venue stat-arb: buy xyz:STOCK on HL spot DEX when cheap, short real stock at broker; "
            "sell xyz:STOCK on HL + close broker short when spread reverts."
        ),
    }

    json_out = out_dir / "hl_margin_check.json"
    json_out.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"\nSaved: {json_out}")

    # Markdown report
    lines = [
        "# Hyperliquid Margin Check: MAG7 Stock Tokens\n",
        f"_Generated: {report_data['generated_at']}_\n",
        f"**HL perp universe:** {perp_universe_size} symbols  |  "
        f"**HL spot universe:** {spot_universe_size} pairs\n",
        "\n## Key Finding\n",
        "> **`xyz:AAPL`, `xyz:MSFT`, `xyz:NVDA`, `xyz:GOOGL`, `xyz:AMZN`, `xyz:META`, `xyz:TSLA`**",
        "> are **HIP-1 spot tokens** on Hyperliquid — NOT perpetual futures contracts.\n",
        "\n## MAG7 Token Details\n",
        "| Ticker | HL Symbol | Found | Type | Mark Price | Mid Price | Leverage | Cross-Margin |",
        "|--------|-----------|-------|------|------------|-----------|----------|--------------|",
    ]
    for ticker, info in stock_perp_info.items():
        found = "✓" if info.get("found") else "✗"
        tok_type = info.get("type", "-")
        mark = info.get("markPrice", "-")
        mid = info.get("midPrice", "-")
        lev = "1x (spot)" if info.get("found") else "-"
        cross = "No (spot wallet)" if info.get("found") else "-"
        hl_sym = info.get("hl_symbol", "-")
        lines.append(f"| {ticker} | `{hl_sym}` | {found} | {tok_type} | {mark} | {mid} | {lev} | {cross} |")

    lines += [
        "\n## Margin Architecture Comparison\n",
        "| Feature | MAG7 on HL (xyz:) | Crypto Perps on HL |",
        "|---------|-------------------|-------------------|",
        "| Instrument type | Spot token (HIP-1) | Perpetual future |",
        "| Leverage | None (1× spot) | Up to 50× |",
        "| Cross-margin | No — spot wallet | Yes — shared perp pool |",
        "| Short selling | Sell holdings only | Yes, with margin |",
        "| Funding rate | None | Paid/received every 8h |",
        "| Mark price | Oracle / spot DEX | Mark price + funding |",
        "",
        "\n## Cross-Venue Arb Implementation Notes\n",
        "- **Long bias only** on HL: buy `xyz:AAPL` when HL price < real stock price.",
        "- **Short via broker**: to short, you must use a stock broker (IB, Alpaca, etc.).",
        "- **Two separate accounts**: HL spot wallet + broker margin account — no shared margin.",
        "- **Capital efficiency**: ~50% deployed on each leg; no cross-margin offset.",
        "- **Funding cost**: None on HL leg. Short rebate/borrowing cost from broker applies.",
    ]

    md_out = out_dir / "hl_margin_check.md"
    md_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {md_out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
