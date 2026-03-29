# BTC Nearest-Strike Recovery Calendar Strategy (Beginner Execution Guide)

## Read This First
This is a rules-based trading playbook. It is designed to be simple to execute, but it does **not** guarantee profit.

If you are new, start with small size and follow the checklist exactly.

## Strategy Goal
When BTC has a negative daily candle, we expect mean-reversion/recovery behavior. We trade this with a call calendar structure and daily management.

## Core Version Used Here: **Nearest Strike**
Because exact strike = prior open is often not listed on Deribit, we use nearest listed strike.

- Anchor price: prior day open \(O_{T-1}\)
- Strike used:
  \[
  K^* = \arg\min_K |K - O_{T-1}|
  \]
  where \(K\) is available listed BTC call strike

## Hard Rules
1. Daily decision time: **UTC 00:00**.
2. Entry trigger: prior day return is negative:
   \[
   r_{T-1}=\frac{C_{T-1}-O_{T-1}}{O_{T-1}} < 0
   \]
3. Short leg expiry must be the **shortest listed expiry with DTE > 1 day**.
4. Long leg expiry must be the nearest listed expiry with DTE >= expected recovery days for the return bucket (optionally + 1 to 3 day safety buffer).
5. Long and short calls must use the **same strike** \(K^*\).

## What To Buy/Sell (At Entry)
At signal time (UTC 00:00):

1. Compute the return bucket from \(r_{T-1}\) (for example 1% bins).
2. Recompute historical negative-candle recovery distribution up to T-1 only.
3. Get expected recovery days for that bucket.
4. Select strike \(K^*\) nearest to \(O_{T-1}\).
5. Open calendar:
   - Buy 1 call at \(K^*\), expiry \(E_L\) (back expiry)
   - Sell 1 call at \(K^*\), expiry \(E_S\) (nearest expiry with DTE > 1 day)

## Daily Management Rules (Every UTC 00:00)
For each active cycle:

1. **Recovery check**:
   \[
   S_T \ge O_{anchor}
   \]
   - If true: close all option legs and hedge for that cycle.

2. If not recovered and long leg still alive:
   - If current day is another negative candle, sell one more short call at same strike \(K^*\), same shortest DTE > 1 day rule.
   - Re-hedge delta after any change.

3. If long expiry is reached before recovery:
   - Close remaining legs and hedge.
   - End that cycle.

## Delta Hedge Rule (Simple)
- Hedge instrument: BTC perpetual.
- Target: net portfolio delta near 0.
- Practical rule:
  - Re-hedge at least once daily at UTC 00:00.
  - Optional intraday re-hedge if |net delta| exceeds your threshold.

## Position Sizing (For Beginners)
Use these conservative defaults:

- Start capital example: 100,000 USD
- Max risk per new cycle: 1% of capital
- Max concurrent cycles: 2
- If 3 consecutive losing cycles occur: pause strategy and review

## Risk Controls (Mandatory)
1. Liquidity filter:
   - Avoid wide bid/ask options
   - Avoid very low volume/open interest strikes
2. Event filter:
   - Optionally skip entries before high-impact macro events
3. Exposure caps:
   - Cap total short call count per strike
   - Cap net vega and net gamma exposure
4. Stop policy:
   - If cycle drawdown breaches pre-set threshold, force close

## Beginner Daily Checklist
Use this exact checklist each day:

1. Confirm UTC 00:00 data is complete.
2. Compute \(r_{T-1}\).
3. If \(r_{T-1} >= 0\): no new entry; manage existing cycles only.
4. If \(r_{T-1} < 0\):
   - Rebuild recovery table
   - Get expected recovery days
   - Pick nearest strike to \(O_{T-1}\)
   - Open long/back + short/front calendar
5. For every open cycle:
   - Check recovery \(S_T \ge O_{anchor}\)
   - If recovered: close cycle
   - If not recovered and day negative: add one short call (same strike)
6. Recompute net delta and hedge.
7. Log everything (prices, strikes, expiries, hedge, PnL, reason codes).

## Logging Fields You Must Keep
- Date/time (UTC)
- Cycle ID
- Anchor open
- Selected nearest strike
- Long instrument, short instrument(s)
- Entry/exit prices
- Hedge size changes
- Daily PnL and cumulative PnL
- Exit reason (recovered / long_expired / risk_stop)

## Backtest and Execution Files in This Workspace
- Strategy backtest engine:
  - [drawdown_analysis/deribit_option_calendar_backtest.py](drawdown_analysis/deribit_option_calendar_backtest.py)
- Validation matrix runner:
  - [drawdown_analysis/run_validation_matrix.py](drawdown_analysis/run_validation_matrix.py)
- Matrix output folder:
  - [drawdown_analysis/matrix_results](drawdown_analysis/matrix_results)

## How to Run (Nearest Mode)
From workspace root:

python drawdown_analysis/deribit_option_calendar_backtest.py --start 2025-01-01 --end 2025-12-31 --allow-nearest-strike --output-dir drawdown_analysis

## Final Notes
- Nearest-strike mode is practical and executable on real historical chains.
- Exact-strike mode can under-trade because exact strikes are often unavailable.
- Keep discipline: this strategy works only if entry, management, and hedge rules are followed consistently.