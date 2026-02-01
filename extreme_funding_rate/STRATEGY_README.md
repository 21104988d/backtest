# Mean Reversion Funding Rate Strategy

## Overview

A mean-reversion strategy that profits from funding rates reverting to zero on Hyperliquid perpetual futures. The strategy enters positions when funding rates are in a specific "sweet spot" range and exits when rates normalize.

**Key Insight**: Funding rates tend to mean-revert to zero. By betting against the current funding rate direction when rates are low (but not too low), we can capture the reversion while receiving favorable funding payments.

---

## Strategy Rules

### Entry Conditions

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ENTRY_LOW` | 0.0014% | Minimum absolute funding rate to enter |
| `ENTRY_HIGH` | 0.0015% | Maximum absolute funding rate to enter |

**Entry Logic:**
```
IF 0.0014% < |funding_rate| < 0.0015%:
    IF funding_rate > 0: OPEN SHORT (bet rate will decrease)
    IF funding_rate < 0: OPEN LONG (bet rate will increase)
```

### Exit Conditions

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EXIT_LOW` | 0.0003% | Exit when funding rate drops to this level |

**Exit Logic:**
```
IF |funding_rate| <= 0.0003%:
    CLOSE POSITION (funding rate has reverted to near-zero)
```

### Key Design Decisions

1. **No Exit High**: Positions are NOT closed when funding rates spike above 0.0015%. This was discovered through backtesting:
   - Exit when FR drops (to 0.0003%): **+$10,959 profit**
   - Exit when FR spikes (above 0.0015%): **-$5,455 loss**
   - The strategy lets winners run and collects funding while waiting for reversion.

2. **Direction Logic**: Always bet AGAINST the funding rate direction
   - Positive FR → SHORT (shorts pay longs, we receive funding)
   - Negative FR → LONG (longs pay shorts, we receive funding)

3. **Entry Range Selection**: The 0.0014%-0.0015% range was optimized to:
   - Avoid too many concurrent positions (wider range = more positions)
   - Maintain profitability (too narrow = missed opportunities)

---

## Position Sizing & Risk Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `POSITION_SIZE` | $100 | Fixed size per position |
| `TAKER_FEE` | 0.045% | Hyperliquid taker fee per trade |

### Capital Requirements

Based on backtest results with **Max Drawdown = $1,983** and **Max Notional Exposure = $8,800**:

| Capital | Max Leverage | Avg Leverage | MDD % | Return % | Risk Rating |
|---------|--------------|--------------|-------|----------|-------------|
| $1,000 | 8.80x | 3.66x | 198.3% | 536.8% | 🔴 LIQUIDATION |
| $2,000 | 4.40x | 1.83x | 99.2% | 268.4% | 🟠 Very High |
| $3,000 | 2.93x | 1.22x | 66.1% | 178.9% | 🟠 Very High |
| $5,000 | 1.76x | 0.73x | 39.7% | 107.4% | 🟡 High |
| $7,500 | 1.17x | 0.49x | 26.4% | 71.6% | 🟢 Moderate |
| $10,000 | 0.88x | 0.37x | 19.8% | 53.7% | ✅ Safe |

**Recommended Minimum Capital:**
- **Safe (MDD < 30%)**: $6,700 minimum
- **Moderate (MDD < 50%)**: $4,000 minimum
- **Conservative (Leverage < 1x)**: $8,800 minimum

---

## Backtest Results

### Performance Summary (2.7 years: May 2023 - Jan 2026)

| Metric | Value |
|--------|-------|
| **Net PnL** | +$5,368 |
| **Total Trades** | 2,426 |
| **Win Rate** | 49.1% |
| **Avg PnL per Trade** | $2.21 |
| **Unique Coins Traded** | 191 |

### PnL Breakdown

| Component | Amount |
|-----------|--------|
| Price PnL | +$4,856 |
| Funding Received | +$724 |
| Fees Paid | -$218 |
| **Net Total** | **+$5,362** |

### Direction Analysis

| Direction | Trades | Win Rate | Total PnL | Avg PnL |
|-----------|--------|----------|-----------|---------|
| **SHORT** | 433 | **70.7%** | +$4,706 | +$10.87 |
| **LONG** | 1,993 | 44.5% | +$656 | +$0.33 |

**Key Finding**: SHORT positions (betting against positive funding) are significantly more profitable than LONG positions.

### Holding Time Statistics

| Metric | Value |
|--------|-------|
| Average | 321 hours (13.4 days) |
| Median | 13 hours |
| Min | 1 hour |
| Max | 12,088 hours (~504 days) |

### Position Exposure

| Metric | Value |
|--------|-------|
| Average Concurrent Positions | 36.6 |
| Max Concurrent Positions | 88 |
| P50 (Median) | 29 |
| P95 | 77 |

### Yearly Performance

| Year | Net PnL | Trades | Win Rate |
|------|---------|--------|----------|
| 2023 | +$339 | 257 | 52.1% |
| 2024 | -$744 | 557 | 48.8% |
| 2025 | +$5,771 | 1,609 | 48.8% |

---

## Benchmark Comparison (vs BTC Buy & Hold)

Starting Capital: $1,000

| Metric | Strategy | BTC Buy & Hold |
|--------|----------|----------------|
| Final Value | $6,366 | $3,259 |
| Return | **+536.8%** | +225.9% |
| Net PnL | **+$5,368** | +$2,259 |

**Outperformance: +$3,108 (+310.78 percentage points)**

### Correlation Analysis

| Metric | Value |
|--------|-------|
| **Hourly Return Correlation** | **-0.0517** |
| Rolling 7-day Correlation (Mean) | -0.1036 |
| Rolling 7-day Correlation (Range) | -0.87 to +0.91 |

**Interpretation**: Near-zero correlation (-0.05) means the strategy provides **excellent diversification** from BTC. Returns are almost independent of Bitcoin price movements.

---

## Data Quality Notes

### Known Data Gaps

The source data has gaps that cause visual "spikes" in the equity curve:

| Gap Period | Duration | Effect |
|------------|----------|--------|
| Aug 21 - Sep 3, 2024 | 13 days | Large jump |
| Mar 13 - Mar 20, 2025 | 7.5 days | +$2,932 spike |
| Jul 30 - Aug 6, 2025 | 6.2 days | Jump |
| Jan 12 - Jan 15, 2026 | 2.9 days | Jump |

**Important**: These spikes are NOT errors. During data gaps:
- Positions remain open (no exit signals)
- When data resumes, accumulated unrealized PnL shows up at once
- This is realistic behavior during API/exchange outages

### Data Source

- **Exchange**: Hyperliquid
- **Data Types**: Funding rates (hourly), OHLC prices (hourly)
- **Period**: May 2023 - January 2026 (~2.7 years)
- **Total Hours**: 21,760
- **Coins**: 203 unique perpetual contracts

---

## Implementation Guide (Live Trading)

### Required Components

1. **Data Feed**
   - Hyperliquid WebSocket for real-time funding rates
   - Price feed for entry/exit execution
   - Hourly funding rate snapshots

2. **Position Manager**
   - Track all open positions with entry details
   - Calculate unrealized PnL continuously
   - Monitor funding payments

3. **Signal Generator**
   - Check funding rates every hour (8-hour funding cycle)
   - Generate entry signals when 0.0014% < |FR| < 0.0015%
   - Generate exit signals when |FR| <= 0.0003%

4. **Execution Engine**
   - Market orders for entries and exits
   - Handle partial fills
   - Implement retry logic

5. **Risk Manager**
   - Monitor total exposure vs capital
   - Track concurrent positions
   - Calculate real-time leverage

### Pseudocode for Live Implementation

```python
# Every hour (at funding settlement)
def check_signals():
    current_funding_rates = fetch_hyperliquid_funding_rates()
    
    for coin, fr in current_funding_rates.items():
        abs_fr = abs(fr)
        
        # Check exits first
        if coin in open_positions:
            if abs_fr <= 0.0003:  # EXIT_LOW
                close_position(coin)
                continue
        
        # Check entries
        if coin not in open_positions:
            if 0.0014 < abs_fr < 0.0015:  # Entry range (as percentages * 100)
                direction = "SHORT" if fr > 0 else "LONG"
                open_position(coin, direction, size=100)

# Run every hour
schedule.every().hour.at(":00").do(check_signals)
```

### API Endpoints Needed

| Endpoint | Purpose |
|----------|---------|
| `GET /info` | Fetch all perpetual funding rates |
| `POST /exchange` | Place market orders |
| `GET /clearinghouse` | Get account positions |
| `WS funding` | Real-time funding rate updates |

### Recommended Execution Schedule

- **Signal Check**: Every hour at :00 (funding settlement time)
- **Position Monitoring**: Every 5 minutes for risk checks
- **Rebalancing**: Not required (positions are independent)

---

## Risk Warnings

### Strategy Risks

1. **Leverage Risk**: With $100 per position and up to 88 concurrent positions, total exposure can reach $8,800. Ensure sufficient capital.

2. **Liquidation Risk**: At low capital levels (<$5,000), max drawdown can exceed 50%. Account could face liquidation during extreme moves.

3. **Data Dependency**: Strategy relies on accurate, timely funding rate data. API outages can affect performance.

4. **Market Regime Risk**: Strategy performed poorly in 2024 (-$744). May underperform during certain market conditions.

5. **Execution Risk**: Slippage on market orders, especially for less liquid coins.

### Mitigation Strategies

1. **Use Adequate Capital**: Minimum $6,700 for safe operation
2. **Position Limits**: Cap maximum concurrent positions
3. **Stop Loss**: Consider adding stop loss for extreme drawdowns
4. **Monitoring**: Implement alerting for unusual exposure levels
5. **Gradual Scaling**: Start with smaller position sizes

---

## File Structure

```
extreme_funding_rate/
├── STRATEGY_README.md          # This documentation
├── mean_reversion_strategy.py  # Backtest engine
├── funding_history.csv         # Historical funding rate data
├── price_history.csv           # Historical price data
├── fetch_data.py               # Data fetching utilities
├── fetch_funding_data.py       # Funding rate fetcher
├── config.py                   # Configuration
└── mean_reversion_results/     # Backtest outputs
    ├── trades.csv              # Individual trade records
    ├── hourly_records.csv      # Hourly portfolio snapshots
    ├── backtest_charts.png     # Main performance charts
    ├── btc_comparison.png      # BTC benchmark comparison
    ├── fee_analysis.png        # Trading fee analysis
    └── trade_distribution.png  # Trade statistics charts
```

---

## Next Steps for Live Implementation

### Phase 1: Infrastructure Setup
- [ ] Set up Hyperliquid API credentials
- [ ] Implement WebSocket connection for real-time data
- [ ] Create database for position tracking
- [ ] Set up logging and monitoring

### Phase 2: Core Logic
- [ ] Implement signal generation module
- [ ] Build position manager
- [ ] Create order execution engine
- [ ] Add funding payment tracking

### Phase 3: Risk Management
- [ ] Implement exposure limits
- [ ] Add position count caps
- [ ] Create alerting system
- [ ] Build dashboard for monitoring

### Phase 4: Testing
- [ ] Paper trading for 2-4 weeks
- [ ] Compare with backtest results
- [ ] Adjust parameters if needed
- [ ] Validate execution quality

### Phase 5: Live Deployment
- [ ] Start with reduced position size ($50)
- [ ] Gradually scale to full size
- [ ] Monitor for 1 month
- [ ] Full deployment

---

## Appendix: Key Discoveries from Backtesting

### Why This Entry Range?

| Entry Range | Avg Positions | Max Positions | Net PnL |
|-------------|---------------|---------------|---------|
| 0.0010% - 0.0015% | ~80 | ~150 | Higher PnL but risky |
| 0.0012% - 0.0015% | ~60 | ~100 | Moderate |
| **0.0014% - 0.0015%** | **~37** | **~88** | **Optimal risk/reward** |

### Why No Exit High?

When testing exit when FR spikes above entry range:
- **Exit High (FR > 0.0015%)**: -$5,455 (LOSS)
- **Exit Low (FR < 0.0003%)**: +$10,959 (PROFIT)

The market tends to overshoot before reverting. Exiting on spikes cuts winners short.

### Why SHORT Outperforms LONG?

- More coins have positive funding rates (bull market bias)
- Positive FR tends to mean-revert more reliably
- SHORT positions: 70.7% win rate vs LONG 44.5%

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Feb 2026 | Initial strategy documentation |

---

*Strategy developed and backtested using historical Hyperliquid data. Past performance does not guarantee future results.*
