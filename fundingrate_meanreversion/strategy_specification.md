# Funding Rate Mean Reversion Strategy

## 1. Objective
To profit from the mean-reverting nature of crypto perpetual funding rates. The strategy assumes that extreme funding rates (highly positive or negative) are unsustainable and will eventually revert to zero. By positioning against the crowd (counter-trend to the funding rate), we aim to capture price reversals and potentially earn funding fees.

## 2. Core Logic
- **Signal**: Deribit `BTC-PERPETUAL` Funding Rate (8-hour interest rate).
- **Direction**: Fade the funding rate.
    - **Negative Funding** (Shorts paying Longs) $\rightarrow$ Market is overly bearish $\rightarrow$ **Open Long**.
    - **Positive Funding** (Longs paying Shorts) $\rightarrow$ Market is overly bullish $\rightarrow$ **Open Short**.

## 3. Trading Rules

### A. Open Position (When Flat)
1.  **Check Funding Rate (FR)**.
2.  If **FR < 0**:
    - Open **Long** position (Size: 10 USD).
    - Store `min_funding_rate` = Current FR.
3.  If **FR > 0**:
    - Open **Short** position (Size: 10 USD).
    - Store `max_funding_rate` = Current FR.

### B. Add Position (Scaling In)
*Used when the funding rate moves further against the crowd, implying a more extreme sentiment.*

1.  **If Holding Long**:
    - If **Current FR < `min_funding_rate`** (More negative):
        - Open another **Long** (Size: 10 USD).
        - Update `min_funding_rate` = Current FR.
2.  **If Holding Short**:
    - If **Current FR > `max_funding_rate`** (More positive):
        - Open another **Short** (Size: 10 USD).
        - Update `max_funding_rate` = Current FR.

### C. Close / Reverse Position
*Used when the funding rate normalizes or flips.*

1.  **If Holding Long**:
    - If **FR >= 0**:
        - **Close** entire Long position.
        - **Check Reversal**: If FR > 0, immediately Open **Short** (Size: 10 USD) and set `max_funding_rate`.
2.  **If Holding Short**:
    - If **FR <= 0**:
        - **Close** entire Short position.
        - **Check Reversal**: If FR < 0, immediately Open **Long** (Size: 10 USD) and set `min_funding_rate`.

## 4. Funding Fees
- **Schedule**: Funding is exchanged every 8 hours (04:00, 12:00, 20:00 UTC).
- **Impact**:
    - If Long and FR < 0: **Receive** Funding (Shorts pay Longs).
    - If Short and FR > 0: **Receive** Funding (Longs pay Shorts).
    - *Note: Since we always trade against the funding rate sign, we are positioned to RECEIVE funding fees if we hold through the funding timestamp.*

## 5. Real-Time Execution Plan
Since historical data is limited to 1-hour snapshots, real-time execution allows for higher granularity:
- **Monitor**: Poll Funding Rate every 1 minute or 5 minutes.
- **Advantage**: Capture intra-hour spikes in funding that might disappear by the hourly close.
- **Platform**: Deribit API (WebSocket or REST).
