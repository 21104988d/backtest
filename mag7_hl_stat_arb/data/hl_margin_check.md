# Hyperliquid Margin Check: MAG7 Stock Tokens

_Generated: 2026-03-29T04:07:50.091132+00:00_

**HL perp universe:** 229 symbols  |  **HL spot universe:** 286 pairs


## Key Finding

> **`xyz:AAPL`, `xyz:MSFT`, `xyz:NVDA`, `xyz:GOOGL`, `xyz:AMZN`, `xyz:META`, `xyz:TSLA`**
> are **HIP-1 spot tokens** on Hyperliquid — NOT perpetual futures contracts.


## MAG7 Token Details

| Ticker | HL Symbol | Found | Type | Mark Price | Mid Price | Leverage | Cross-Margin |
|--------|-----------|-------|------|------------|-----------|----------|--------------|
| AAPL | `xyz:AAPL` | ✓ | spot_token_hip1 | 0.047978 | 0.047894 | 1x (spot) | No (spot wallet) |
| MSFT | `xyz:MSFT` | ✓ | spot_token_hip1 | 78.0 | None | 1x (spot) | No (spot wallet) |
| NVDA | `xyz:NVDA` | ✓ | spot_token_hip1 | N/A | N/A | 1x (spot) | No (spot wallet) |
| GOOGL | `xyz:GOOGL` | ✓ | spot_token_hip1 | 40.452 | 40.463 | 1x (spot) | No (spot wallet) |
| AMZN | `xyz:AMZN` | ✓ | spot_token_hip1 | 31.001 | 246.8905 | 1x (spot) | No (spot wallet) |
| META | `xyz:META` | ✓ | spot_token_hip1 | 65.936 | 66.0715 | 1x (spot) | No (spot wallet) |
| TSLA | `xyz:TSLA` | ✓ | spot_token_hip1 | 217.9 | 218.115 | 1x (spot) | No (spot wallet) |

## Margin Architecture Comparison

| Feature | MAG7 on HL (xyz:) | Crypto Perps on HL |
|---------|-------------------|-------------------|
| Instrument type | Spot token (HIP-1) | Perpetual future |
| Leverage | None (1× spot) | Up to 50× |
| Cross-margin | No — spot wallet | Yes — shared perp pool |
| Short selling | Sell holdings only | Yes, with margin |
| Funding rate | None | Paid/received every 8h |
| Mark price | Oracle / spot DEX | Mark price + funding |


## Cross-Venue Arb Implementation Notes

- **Long bias only** on HL: buy `xyz:AAPL` when HL price < real stock price.
- **Short via broker**: to short, you must use a stock broker (IB, Alpaca, etc.).
- **Two separate accounts**: HL spot wallet + broker margin account — no shared margin.
- **Capital efficiency**: ~50% deployed on each leg; no cross-margin offset.
- **Funding cost**: None on HL leg. Short rebate/borrowing cost from broker applies.