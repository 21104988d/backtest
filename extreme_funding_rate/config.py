"""
Configuration loader for extreme funding rate backtest.
Reads settings from .env file.
"""
import os
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """Load configuration from .env file."""
    config = {
        # Data fetching
        'days_back': 7,
        'fetch_delay': 0.2,
        
        # Backtest parameters
        'initial_capital': 10000,
        'position_size_pct': 1.0,
        'transaction_cost': 0.0005,
        
        # Strategy configuration
        'num_positions': 1,
        'holding_period_hours': 1,
        
        # Risk management
        'stop_loss_pct': 0,
        'take_profit_pct': 0,
        
        # Filters
        'min_funding_threshold': -1.0,
        'max_funding_threshold': 1.0,
    }
    
    # Try to load from .env file
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove comments from value
                    if '#' in value:
                        value = value.split('#')[0].strip()
                    
                    # Map env keys to config keys
                    env_to_config = {
                        'DAYS_BACK': 'days_back',
                        'FETCH_DELAY': 'fetch_delay',
                        'INITIAL_CAPITAL': 'initial_capital',
                        'POSITION_SIZE_PCT': 'position_size_pct',
                        'TRANSACTION_COST': 'transaction_cost',
                        'NUM_POSITIONS': 'num_positions',
                        'HOLDING_PERIOD_HOURS': 'holding_period_hours',
                        'STOP_LOSS_PCT': 'stop_loss_pct',
                        'TAKE_PROFIT_PCT': 'take_profit_pct',
                        'MIN_FUNDING_THRESHOLD': 'min_funding_threshold',
                        'MAX_FUNDING_THRESHOLD': 'max_funding_threshold',
                    }
                    
                    if key in env_to_config:
                        config_key = env_to_config[key]
                        # Convert to appropriate type
                        try:
                            if config_key in ['days_back', 'num_positions', 'holding_period_hours']:
                                config[config_key] = int(value)
                            else:
                                config[config_key] = float(value)
                        except ValueError:
                            print(f"Warning: Invalid value for {key}: {value}")
    
    return config


def print_config(config: Dict[str, Any]):
    """Print current configuration."""
    print("="*80)
    print("BACKTEST CONFIGURATION")
    print("="*80)
    print("\n📊 Data Fetching:")
    print(f"  Days of history:        {config['days_back']} days")
    print(f"  API delay:              {config['fetch_delay']} seconds")
    
    print("\n💰 Capital & Costs:")
    print(f"  Initial capital:        ${config['initial_capital']:,.2f}")
    print(f"  Position size:          {config['position_size_pct']*100:.1f}% of capital")
    print(f"  Transaction cost:       {config['transaction_cost']*100:.3f}%")
    
    print("\n🎯 Strategy:")
    print(f"  Number of positions:    {config['num_positions']} (top {config['num_positions']} extreme negative)")
    print(f"  Holding period:         {config['holding_period_hours']} hour(s)")
    
    print("\n⚠️  Risk Management:")
    if config['stop_loss_pct'] > 0:
        print(f"  Stop loss:              {config['stop_loss_pct']*100:.1f}%")
    else:
        print(f"  Stop loss:              Disabled")
    
    if config['take_profit_pct'] > 0:
        print(f"  Take profit:            {config['take_profit_pct']*100:.1f}%")
    else:
        print(f"  Take profit:            Disabled")
    
    print("\n🔍 Filters:")
    if config['min_funding_threshold'] > -1.0:
        print(f"  Min funding threshold:  {config['min_funding_threshold']*100:.3f}%")
    else:
        print(f"  Min funding threshold:  Disabled")
    
    if config['max_funding_threshold'] < 1.0:
        print(f"  Max funding threshold:  {config['max_funding_threshold']*100:.3f}%")
    else:
        print(f"  Max funding threshold:  Disabled")
    
    print("="*80)


if __name__ == "__main__":
    config = load_config()
    print_config(config)
