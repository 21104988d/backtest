# Hyperliquid Mainnet Data Source

## Important Notice

This project **exclusively uses Hyperliquid MAINNET data**. 

### Why Mainnet Only?

- **Real Trading Data**: Mainnet reflects actual trading conditions and funding rates
- **Production Ready**: Suitable for live trading strategy analysis
- **Accurate Liquidity**: Real volume and market depth data
- **No Testnet Noise**: Testnet may have artificial/test conditions

### API Endpoint

All scripts use the Hyperliquid mainnet public API:
```
https://api.hyperliquid.xyz/info
```

### Included Mainnet Data

The following mainnet funding snapshot is included for reference:
- **File**: `hyperliquid_mainnet_funding.csv`
- **Contents**: Current funding rates for all pairs on mainnet
- **Columns**:
  - Coin: Trading pair symbol
  - Price: Current price
  - Funding Rate (1h): Hourly funding rate
  - Funding Rate (Apr %): Annualized funding rate percentage

### Historical Data Fetching

When you run `fetch_funding_data.py`, it fetches historical funding rates from:
- **Source**: Hyperliquid mainnet API
- **Default Period**: Last 30 days
- **All Pairs**: All perpetual contracts on mainnet

### Verification

To verify you're using mainnet data:
1. Check that the API endpoint in scripts is `https://api.hyperliquid.xyz/info`
2. Review the funding snapshot file for realistic funding rates
3. Cross-reference with Hyperliquid's official mainnet interface

### Testnet Data

**Testnet data is NOT used in this project.** If you need testnet data:
- You would need to use a different API endpoint
- Funding rates and conditions would be different
- Not recommended for backtesting real trading strategies

---

**Last Updated**: January 28, 2026
