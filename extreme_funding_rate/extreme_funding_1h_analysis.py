"""
Analyze distribution of extreme funding rates and 1-hour after price performance.
Outputs summary stats and visualization charts.
"""
import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_funding(path):
    funding = pd.read_csv(path)
    if 'datetime' in funding.columns:
        funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed')
    else:
        funding['datetime'] = pd.to_datetime(funding['timestamp'], unit='ms')
    funding['hour'] = funding['datetime'].dt.floor('h')
    return funding


def load_prices(path):
    price = pd.read_csv(path)
    price['timestamp'] = pd.to_datetime(price['timestamp'])
    return price


def build_price_lookup(price_df):
    return dict(zip(price_df['coin'] + '_' + price_df['timestamp'].astype(str), price_df['price']))


def get_price(price_lookup, coin, timestamp):
    return price_lookup.get(f"{coin}_{timestamp}")


def main():
    parser = argparse.ArgumentParser(description="Extreme funding distribution and 1h performance analysis")
    parser.add_argument("--funding-file", default="funding_history.csv")
    parser.add_argument("--price-file", default="price_cache_with_beta_history.csv")
    parser.add_argument("--output-prefix", default="extreme_funding")
    args = parser.parse_args()

    funding = load_funding(args.funding_file)
    price = load_prices(args.price_file)
    price_lookup = build_price_lookup(price)

    hours = sorted(funding['hour'].unique())
    events = []

    for hour in hours[:-1]:
        hour_data = funding[funding['hour'] == hour]
        if hour_data.empty:
            continue

        min_row = hour_data.loc[hour_data['funding_rate'].idxmin()]
        max_row = hour_data.loc[hour_data['funding_rate'].idxmax()]

        for label, row in [("negative", min_row), ("positive", max_row)]:
            coin = row['coin']
            fr = row['funding_rate']
            entry = get_price(price_lookup, coin, hour)
            exit_ = get_price(price_lookup, coin, hour + timedelta(hours=1))
            if entry is None or exit_ is None:
                continue
            ret = (exit_ / entry - 1) * 100
            events.append({
                'hour': hour,
                'coin': coin,
                'type': label,
                'funding_rate': fr,
                'return_1h_pct': ret
            })

    events_df = pd.DataFrame(events)
    if events_df.empty:
        print("No events with valid 1h prices found.")
        return

    out_csv = f"{args.output_prefix}_1h_events.csv"
    events_df.to_csv(out_csv, index=False)

    # Summary stats
    def summarize(df, label):
        return {
            'type': label,
            'count': len(df),
            'funding_mean': df['funding_rate'].mean(),
            'funding_median': df['funding_rate'].median(),
            'ret_mean': df['return_1h_pct'].mean(),
            'ret_median': df['return_1h_pct'].median(),
            'win_rate': (df['return_1h_pct'] > 0).mean() * 100
        }

    neg = events_df[events_df['type'] == 'negative']
    pos = events_df[events_df['type'] == 'positive']

    summary = pd.DataFrame([summarize(neg, 'negative'), summarize(pos, 'positive')])
    corr = events_df['funding_rate'].corr(events_df['return_1h_pct'])

    print("Summary (extreme funding -> 1h return)")
    print(summary[['type', 'count', 'funding_mean', 'funding_median', 'ret_mean', 'ret_median', 'win_rate']])
    print(f"Correlation funding vs 1h return: {corr:.4f}")
    print(f"Saved events: {out_csv}")

    # Plot 1: Funding rate distribution for extremes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(neg['funding_rate'], bins=50, color='#e74c3c', alpha=0.7)
    axes[0].set_title('Extreme Negative Funding Rate Distribution')
    axes[0].set_xlabel('Funding Rate')
    axes[0].set_ylabel('Count')

    axes[1].hist(pos['funding_rate'], bins=50, color='#2ecc71', alpha=0.7)
    axes[1].set_title('Extreme Positive Funding Rate Distribution')
    axes[1].set_xlabel('Funding Rate')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    dist_png = f"{args.output_prefix}_distribution.png"
    plt.savefig(dist_png, dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Funding rate vs 1h return scatter + binned boxplot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    axes[0].scatter(neg['funding_rate'], neg['return_1h_pct'], s=10, alpha=0.4, color='#e74c3c', label='Negative extreme')
    axes[0].scatter(pos['funding_rate'], pos['return_1h_pct'], s=10, alpha=0.4, color='#2ecc71', label='Positive extreme')
    axes[0].axhline(0, color='gray', linewidth=0.8)
    axes[0].set_title('Funding Rate vs 1h Return')
    axes[0].set_xlabel('Funding Rate')
    axes[0].set_ylabel('1h Return (%)')
    axes[0].legend(loc='upper right')

    # Boxplot by funding bins
    bins = pd.qcut(events_df['funding_rate'], q=8, duplicates='drop')
    events_df['funding_bin'] = bins.astype(str)
    grouped = [events_df[events_df['funding_bin'] == b]['return_1h_pct'] for b in events_df['funding_bin'].unique()]
    axes[1].boxplot(grouped, labels=events_df['funding_bin'].unique(), showfliers=False)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_title('1h Return by Funding Rate Quantile')
    axes[1].set_ylabel('1h Return (%)')

    plt.tight_layout()
    perf_png = f"{args.output_prefix}_1h_performance.png"
    plt.savefig(perf_png, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved charts: {dist_png}, {perf_png}")


if __name__ == "__main__":
    main()
