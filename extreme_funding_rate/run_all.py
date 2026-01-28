#!/usr/bin/env python3
"""
Master script to execute the complete extreme funding rate analysis pipeline.

This script runs all steps in sequence:
1. Fetch funding rate data from Hyperliquid
2. Analyze extreme funding events
3. Backtest the trading strategy
"""
import subprocess
import sys
import os


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print_banner(f"STEP: {description}")
    print(f"Running: {script_name}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error in {description}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n✗ {description} interrupted by user")
        return False


def check_file_exists(filename):
    """Check if a required file exists."""
    if os.path.exists(filename):
        print(f"✓ Found: {filename}")
        return True
    else:
        print(f"✗ Missing: {filename}")
        return False


def main():
    """Main execution function."""
    print_banner("EXTREME FUNDING RATE ANALYSIS PIPELINE")
    
    print("This script will run the complete analysis pipeline:")
    print("  1. Fetch funding rate data from Hyperliquid")
    print("  2. Identify extreme funding events")
    print("  3. Backtest the trading strategy")
    print()
    
    # Check if we should skip data fetching
    skip_fetch = False
    if check_file_exists('funding_history.csv'):
        response = input("\nfunding_history.csv already exists. Skip fetching new data? (y/n): ")
        if response.lower() == 'y':
            skip_fetch = True
            print("Skipping data fetch step...")
    
    # Step 1: Fetch funding data
    if not skip_fetch:
        success = run_script('fetch_funding_data.py', 'Fetch Funding Rate Data')
        if not success:
            print("\n⚠ Data fetching failed. Cannot proceed with analysis.")
            return
    
    # Verify data file exists
    if not check_file_exists('funding_history.csv'):
        print("\n⚠ funding_history.csv not found. Cannot proceed with analysis.")
        print("Please run fetch_funding_data.py first.")
        return
    
    # Step 2: Analyze extreme funding (skip interactive performance calculation)
    print_banner("STEP: Analyze Extreme Funding Events")
    print("Note: Skipping optional performance calculation for now.")
    print("You can run analyze_extreme_funding.py manually to calculate performance.\n")
    
    # We'll extract just the extreme identification part
    try:
        import pandas as pd
        from analyze_extreme_funding import load_funding_data, identify_extreme_funding_per_hour, save_extreme_events
        
        print("Loading funding rate data...")
        df = load_funding_data()
        print(f"Loaded {len(df):,} funding rate records")
        
        print("\nIdentifying extreme funding rates per hour...")
        extreme_df = identify_extreme_funding_per_hour(df)
        print(f"Identified {len(extreme_df):,} extreme funding events")
        
        save_extreme_events(extreme_df)
        print("\n✓ Extreme funding analysis completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error in extreme funding analysis: {e}")
        print("You may need to run analyze_extreme_funding.py manually.")
    
    # Step 3: Backtest strategy
    success = run_script('backtest_strategy.py', 'Backtest Trading Strategy')
    if not success:
        print("\n⚠ Backtesting failed.")
        return
    
    # Summary
    print_banner("PIPELINE COMPLETE")
    
    print("Generated Files:")
    files_to_check = [
        'funding_history.csv',
        'extreme_funding_events.csv',
        'backtest_trades.csv',
        'backtest_metrics.csv',
        'backtest_results.png'
    ]
    
    for filename in files_to_check:
        check_file_exists(filename)
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("  1. Review backtest_results.png for visual analysis")
    print("  2. Examine backtest_trades.csv for individual trade details")
    print("  3. Check backtest_metrics.csv for performance summary")
    print("  4. Run analyze_extreme_funding.py interactively for detailed performance analysis")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user. Exiting...")
        sys.exit(1)
