# Delta-Hedged Strategy Comparison

## Strategy Overview

### Standard Strategy (Original)
**Setup**: Buy coin with most extreme negative funding rate each hour

**Position**: Long only
- Entry: Buy $10,000 of extreme negative funding coin
- Exit: Sell after 1 hour
- P&L: Price change - 2x transaction costs (buy + sell)

**Exposure**: Full market exposure (beta ≈ 1 to crypto market)

### Delta-Hedged Strategy (New)
**Setup**: Buy extreme negative funding coin + Short BTC hedge

**Position**: Market-neutral
- Entry: Buy $10,000 of extreme funding coin + Short $10,000 BTC
- Exit: Sell coin + Cover BTC short after 1 hour
- P&L: (Coin return - BTC return) - 4x transaction costs

**Exposure**: Zero market exposure (beta ≈ 0 to crypto market)

## Key Differences

| Aspect | Standard | Delta-Hedged |
|--------|----------|--------------|
| **Market Exposure** | Yes (directional) | No (market-neutral) |
| **Risk Source** | Coin price + Market risk | Coin vs BTC performance only |
| **Transaction Costs** | 0.10% (2 sides) | 0.20% (4 sides) |
| **Profit Driver** | Absolute price movement | Relative outperformance vs BTC |
| **Best Case** | Coin rallies strongly | Coin rallies while BTC falls |
| **Worst Case** | Coin crashes | Coin falls while BTC rallies |

## Why Delta Hedging?

### Problem with Standard Strategy
The standard strategy suffers from **market correlation**:
- When crypto market crashes, ALL coins fall (including extreme negative funding coins)
- Extreme negative funding may signal oversold, but market-wide sell-offs override this
- We can't isolate whether losses are from bad signal or bad market timing

### Solution: Delta Hedging
By shorting BTC, we eliminate market beta:
- If crypto market rises: Coin gains offset by BTC short losses = neutral
- If crypto market falls: Coin losses offset by BTC short gains = neutral
- **Only** profit/loss comes from whether coin outperforms or underperforms BTC

This isolates the **funding rate signal** from **market direction**.

## Expected Outcomes

### If Delta-Hedged Performs Better:
✓ Extreme negative funding is a valid signal
✓ Strategy works but was masked by market downtrends
✓ Should consider trading with hedge in live environment

### If Delta-Hedged Performs Worse:
✗ The signal itself is flawed (not just market timing)
✗ Selected coins underperform even relative to BTC
✗ Higher transaction costs (4 sides) hurt without adding value

### If Both Lose Money:
⚠️ Fundamental strategy issue regardless of hedging
⚠️ Extreme negative funding may not be a reliable mean reversion signal
⚠️ Need additional filters or different approach

## Cost Analysis

### Standard Strategy Transaction Costs
- Buy coin: 0.05% × $10,000 = $5
- Sell coin: 0.05% × $10,000 = $5
- **Total**: $10 per trade (0.10%)

### Delta-Hedged Transaction Costs
- Buy coin: 0.05% × $10,000 = $5
- Short BTC: 0.05% × $10,000 = $5
- Sell coin: 0.05% × $10,000 = $5
- Cover BTC: 0.05% × $10,000 = $5
- **Total**: $20 per trade (0.20%)

The delta-hedged strategy must generate **0.10% extra return per trade** just to break even after costs.

## Break-Even Analysis

For 558 trades (30-day backtest):

**Standard Strategy**:
- Total costs: 558 × $10 = $5,580
- Must generate $5,580 in gross returns to break even

**Delta-Hedged Strategy**:
- Total costs: 558 × $20 = $11,160
- Must generate $11,160 in gross returns to break even

The delta-hedged strategy needs **$5,580 more in gross returns** to match standard strategy performance.

## What We're Testing

### Hypothesis
"Extreme negative funding rates signal temporary oversold conditions that lead to outperformance vs the broader market (BTC)"

### Test
- **Standard strategy** tests: Does buying extreme negative funding generate positive returns?
- **Delta-hedged strategy** tests: Do these coins outperform BTC specifically?

### Interpretation Guide

| Standard Result | Delta-Hedged Result | Interpretation |
|----------------|-------------------|----------------|
| Loss | Loss | Signal is bad, coins underperform absolutely and relatively |
| Loss | Profit | Signal is good but bad market timing, hedge reveals value |
| Profit | Loss | Signal benefited from market rally, not actual edge |
| Profit | Profit | Signal has true alpha, works in both absolute and relative terms |

## Running the Comparison

Once both backtests complete, run:
```bash
python compare_strategies.py
```

This will generate:
1. `strategy_comparison.csv` - Side-by-side metrics
2. `strategy_comparison.png` - Visual comparison charts
3. Console output with improvement analysis

## Files Generated

**Standard Strategy**:
- `backtest_trades.csv` - 558 trades
- `backtest_metrics.csv` - Performance summary
- `backtest_results.png` - Charts
- `equity_performance.png` - Equity curve

**Delta-Hedged Strategy**:
- `backtest_delta_hedged_trades.csv` - Hedged trades
- `backtest_delta_hedged_metrics.csv` - Hedged performance
- (Charts to be added)

**Comparison**:
- `strategy_comparison.csv` - Head-to-head comparison
- `strategy_comparison.png` - Comparative visualizations

---

*This analysis helps determine if the strategy has genuine alpha or just market beta exposure.*
