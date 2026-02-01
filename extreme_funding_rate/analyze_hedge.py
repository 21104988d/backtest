import pandas as pd

events = pd.read_csv('extreme_funding_1h_events.csv')

neg = events[events['type'] == 'negative']
pos = events[events['type'] == 'positive']

print('=== HEDGED FUNDING CAPTURE STRATEGY ===')
print(f'Negative FR: mean={neg["funding_rate"].mean()*100:.4f}%, you receive={abs(neg["funding_rate"].mean())*100:.4f}%/hr, n={len(neg)}')
print(f'Positive FR: mean={pos["funding_rate"].mean()*100:.4f}%, you receive={pos["funding_rate"].mean()*100:.4f}%/hr, n={len(pos)}')

total = len(events)
profit = abs(neg['funding_rate']).sum()*100 + pos['funding_rate'].sum()*100
print(f'\nTotal: {total} trades, {profit:.2f}% earned, avg {profit/total:.4f}%/trade')
print(f'Annualized (if hourly): {profit/total*8760:.1f}%/yr before costs')

print('\n=== WITH THRESHOLD FILTERS ===')
for t in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
    nf = neg[neg['funding_rate'] < -t]
    pf = pos[pos['funding_rate'] > t]
    n = len(nf) + len(pf)
    if n > 0:
        p = (abs(nf['funding_rate']).sum() + pf['funding_rate'].sum())*100
        print(f'|FR|>{t*100:.2f}%: {n:5d} trades, avg {p/n:.4f}%/trade, total {p:.2f}%')
