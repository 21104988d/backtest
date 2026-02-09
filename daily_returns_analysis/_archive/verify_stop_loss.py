"""
Verify Stop Loss Logic - Deep Analysis
Understand why tighter stop losses appear to perform better
"""

import pandas as pd
import numpy as np

def main():
    ohlc_df = pd.read_csv('daily_ohlc.csv')
    ohlc_df['date'] = pd.to_datetime(ohlc_df['date']).dt.date

    # Calculate intraday drawdown from open
    # For LONG: how much did price drop? (low vs open)
    ohlc_df['long_drawdown'] = (ohlc_df['low'] / ohlc_df['open'] - 1) * 100  # negative values
    # For SHORT: how much did price rise? (high vs open) - this is a loss for shorts
    ohlc_df['short_drawdown'] = -(ohlc_df['high'] / ohlc_df['open'] - 1) * 100  # negative values

    print('='*80)
    print('INTRADAY DRAWDOWN ANALYSIS')
    print('='*80)

    print('\nLONG Position Intraday Drawdown (Low vs Open):')
    print(f'  Mean:    {ohlc_df["long_drawdown"].mean():.2f}%')
    print(f'  Median:  {ohlc_df["long_drawdown"].median():.2f}%')
    print(f'  5th %:   {ohlc_df["long_drawdown"].quantile(0.05):.2f}%')
    print(f'  Min:     {ohlc_df["long_drawdown"].min():.2f}%')

    print('\nSHORT Position Intraday Drawdown (High vs Open):')
    print(f'  Mean:    {ohlc_df["short_drawdown"].mean():.2f}%')
    print(f'  Median:  {ohlc_df["short_drawdown"].median():.2f}%')
    print(f'  5th %:   {ohlc_df["short_drawdown"].quantile(0.05):.2f}%')
    print(f'  Min:     {ohlc_df["short_drawdown"].min():.2f}%')

    print('\n' + '='*80)
    print('STOP LOSS HIT RATE BY LEVEL')
    print('='*80)
    print(f'\nSL Level   Long Hit%   Short Hit%   Total Hit%')
    print('-'*50)
    for sl in [0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        long_hit = (ohlc_df['long_drawdown'] <= -sl).mean() * 100
        short_hit = (ohlc_df['short_drawdown'] <= -sl).mean() * 100
        total_hit = (long_hit + short_hit) / 2
        print(f'  {sl}%       {long_hit:5.1f}%       {short_hit:5.1f}%       {total_hit:5.1f}%')

    print('\n' + '='*80)
    print('KEY INSIGHT: What happens to positions that hit different SL levels?')
    print('='*80)

    # For positions that DON'T get stopped at 0.5%, what's their close return?
    not_stopped_05 = ohlc_df[ohlc_df['long_drawdown'] > -0.5]
    print(f'\nLONG positions NOT stopped at 0.5% SL ({len(not_stopped_05)} positions):')
    print(f'  Avg Close Return: {not_stopped_05["daily_return"].mean():.2f}%')
    print(f'  % Profitable:     {(not_stopped_05["daily_return"] > 0).mean()*100:.1f}%')

    not_stopped_10 = ohlc_df[ohlc_df['long_drawdown'] > -1.0]
    print(f'\nLONG positions NOT stopped at 1.0% SL ({len(not_stopped_10)} positions):')
    print(f'  Avg Close Return: {not_stopped_10["daily_return"].mean():.2f}%')
    print(f'  % Profitable:     {(not_stopped_10["daily_return"] > 0).mean()*100:.1f}%')

    print('\n' + '='*80)
    print('THE CRITICAL QUESTION: Positions stopped at 0.5% but not at 1%')
    print('='*80)
    
    # Positions stopped at 0.5% but NOT at 1.0% 
    # These are the ones we "miss out on" by using tighter SL
    saved_by_tight = ohlc_df[(ohlc_df['long_drawdown'] <= -0.5) & (ohlc_df['long_drawdown'] > -1.0)]
    print(f'\nPositions stopped at 0.5% but NOT at 1.0% ({len(saved_by_tight)} positions):')
    print(f'  These are positions we EXIT at -0.5% with tight SL')
    print(f'  But would have stayed in with 1% SL')
    print(f'  What would have happened if we stayed?')
    print(f'    Avg Close Return: {saved_by_tight["daily_return"].mean():.2f}%')
    print(f'    % that would have been profitable: {(saved_by_tight["daily_return"] > 0).mean()*100:.1f}%')
    print(f'    Max Profit Missed: {saved_by_tight["daily_return"].max():.2f}%')
    print(f'    Max Loss Avoided: {saved_by_tight["daily_return"].min():.2f}%')

    # Calculate the tradeoff
    print('\n' + '='*80)
    print('DOLLAR IMPACT ANALYSIS (per 100 positions, $100 each)')
    print('='*80)
    
    # Simulate what happens with 0.5% SL vs 1% SL
    # Using saved_by_tight positions only (the difference between the two)
    n_diff = len(saved_by_tight)
    total_positions = len(ohlc_df)
    pct_diff = n_diff / total_positions * 100
    
    print(f'\nPositions in the "difference zone" (-0.5% to -1% drawdown): {n_diff} ({pct_diff:.1f}%)')
    
    # With 0.5% SL: we lose 0.5% on these
    loss_with_05 = n_diff * 0.5
    print(f'\nWith 0.5% SL: Exit these {n_diff} positions at -0.5%')
    print(f'  Total loss: {loss_with_05:.1f}% (across portfolio)')
    
    # With 1.0% SL: we get the actual returns
    actual_returns = saved_by_tight['daily_return'].sum()
    print(f'\nWith 1.0% SL: Hold these {n_diff} positions to close')
    print(f'  Total return: {actual_returns:.1f}%')
    
    diff = actual_returns - (-loss_with_05)
    print(f'\nNet difference: {diff:.1f}%')
    if diff > 0:
        print('  -> 1.0% SL is better for these positions')
    else:
        print('  -> 0.5% SL is better for these positions')

    print('\n' + '='*80)
    print('POSITIONS THAT HIT 1% STOP LOSS')
    print('='*80)
    
    # Positions that hit 1% SL - what's their actual close return?
    hit_10 = ohlc_df[ohlc_df['long_drawdown'] <= -1.0]
    print(f'\nPositions that HIT 1% SL ({len(hit_10)} positions):')
    print(f'  Avg Close Return (if no SL): {hit_10["daily_return"].mean():.2f}%')
    print(f'  % that would have recovered to profit: {(hit_10["daily_return"] > 0).mean()*100:.1f}%')
    
    # Compare outcomes
    print('\n  If we use 1% SL: we lose 1% on these')
    print(f'  If we use NO SL: avg return is {hit_10["daily_return"].mean():.2f}%')
    
    # The key question: do recovered positions make up for the losses?
    recovered = hit_10[hit_10['daily_return'] > 0]
    not_recovered = hit_10[hit_10['daily_return'] <= 0]
    print(f'\n  Breakdown of {len(hit_10)} positions that hit 1% SL:')
    print(f'    {len(recovered)} ({len(recovered)/len(hit_10)*100:.1f}%) recovered to profit, avg: +{recovered["daily_return"].mean():.2f}%')
    print(f'    {len(not_recovered)} ({len(not_recovered)/len(hit_10)*100:.1f}%) stayed negative, avg: {not_recovered["daily_return"].mean():.2f}%')

if __name__ == '__main__':
    main()
