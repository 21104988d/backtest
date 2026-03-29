"""Central configuration for the daily returns analysis pipeline."""

from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

# Data paths
HOURLY_PRICE_FILE = SCRIPT_DIR / "price_history.csv"
DAILY_OHLC_FILE = SCRIPT_DIR / "daily_ohlc.csv"
RESULTS_FILE = SCRIPT_DIR / "strategy_comparison_results.csv"

# Data integrity gates
MIN_REQUIRED_HOURS_PER_DAY = 24
MIN_ASSETS_PER_DAY = 6
MAX_STALENESS_DAYS = 2
DROP_INCOMPLETE_DAYS = True

# Backtest controls
INITIAL_CAPITAL = 1000.0
POSITION_SIZE_FIXED = 100.0
POSITION_FRACTION = 0.10

# Strategy and execution controls
DEFAULT_N = 5
N_SWEEP_VALUES = list(range(1, 13))
DYNAMIC_SL_MULTIPLIER = 0.50
MIN_SL_PCT = 0.5
MAX_SL_PCT = 3.0

# Hyperliquid conservative default: taker-only round-trip
TAKER_FEE_ONE_WAY_PCT = 0.05
ROUND_TRIP_FEE_PCT = TAKER_FEE_ONE_WAY_PCT * 2

# Execution realism controls (percent values expressed in basis points inputs)
SPREAD_BPS_PER_SIDE = 2.0
SLIPPAGE_BPS_PER_SIDE = 3.0
STOP_EXTRA_SLIPPAGE_BPS = 8.0

# Optimizer controls
DYNAMIC_MULTIPLIERS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
FIXED_SL_PCTS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00]

# Train/validation split for N sweep
VALIDATION_DAYS = 90


@dataclass(frozen=True)
class BacktestConfig:
    n: int = DEFAULT_N
    initial_capital: float = INITIAL_CAPITAL
    position_size_fixed: float = POSITION_SIZE_FIXED
    position_fraction: float = POSITION_FRACTION
    dynamic_sl_multiplier: float = DYNAMIC_SL_MULTIPLIER
    min_sl_pct: float = MIN_SL_PCT
    max_sl_pct: float = MAX_SL_PCT
    round_trip_fee_pct: float = ROUND_TRIP_FEE_PCT
