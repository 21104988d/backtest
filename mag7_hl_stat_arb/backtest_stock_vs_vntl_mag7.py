#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, time
from pathlib import Path
from datetime import datetime, timezone
import urllib.request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAG7=['AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA']
API='https://api.hyperliquid.xyz/info'


def post(payload):
    req=urllib.request.Request(API,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=30) as r:
        return json.loads(r.read().decode())


def mean(v): return sum(v)/len(v) if v else float('nan')

def std(v):
    n=len(v)
    if n<2:return float('nan')
    m=mean(v)
    return math.sqrt(sum((x-m)**2 for x in v)/(n-1))

def sharpe(r,ppy=252.0):
    if len(r)<2:return float('nan')
    s=std(r)
    return float('nan') if (math.isnan(s) or s==0) else (mean(r)/s)*math.sqrt(ppy)

def mdd(eq):
    if not eq:return float('nan')
    p=eq[0];d=0.0
    for v in eq:
        p=max(p,v)
        d=min(d,(v/p)-1 if p>0 else 0.0)
    return d

def ols_beta(y,x):
    mx,my=mean(x),mean(y)
    sxx=sum((i-mx)**2 for i in x)
    if sxx<=0:return 0.0
    sxy=sum((x[i]-mx)*(y[i]-my) for i in range(len(y)))
    return sxy/sxx

def rolling_beta(y,x,w):
    o=[None]*len(y)
    for i in range(w,len(y)):o[i]=ols_beta(y[i-w:i],x[i-w:i])
    return o

def rolling_z(s,w):
    o=[None]*len(s)
    for i in range(w,len(s)):
        win=s[i-w:i]; mu=mean(win); sd=std(win)
        o[i]=0.0 if (math.isnan(sd) or sd==0) else (s[i]-mu)/sd
    return o


def run_pair(dates, r_s, r_i, beta_w, z_w, entry, exit_, fee, cap):
    b=rolling_beta(r_s,r_i,beta_w)
    h=[0.0 if b[i] is None else (r_s[i]-b[i]*r_i[i]) for i in range(len(r_s))]
    sp=[]; lv=0.0
    for v in h: lv+=v; sp.append(lv)
    z=rolling_z(sp,z_w)

    pos=[0]*len(r_s); eg=[cap]*len(r_s); en=[cap]*len(r_s); br=[]; tr=0
    for i in range(1,len(r_s)):
        zp=z[i-1]; pp=pos[i-1]
        if zp is None:
            pos[i]=0; eg[i]=eg[i-1]; en[i]=en[i-1]; br.append(0.0); continue
        if pp==0:
            np=-1 if zp>=entry else (1 if zp<=-entry else 0)
        else:
            if abs(zp)<=exit_: np=0
            elif pp==-1 and zp<=-entry: np=1
            elif pp==1 and zp>=entry: np=-1
            else: np=pp
        pos[i]=np
        turn=abs(np-pp)
        if turn>0: tr+=1
        gross=np*h[i]
        net=gross-turn*fee
        eg[i]=eg[i-1]*(1+gross)
        en[i]=en[i-1]*(1+net)
        br.append(net)

    return {
        'dates':dates,'beta':[float('nan') if x is None else x for x in b],
        'spread':sp,'zscore':[float('nan') if x is None else x for x in z],
        'position':pos,'equity_net':en,'equity_gross':eg,
        'metrics':{
            'bars':len(r_s),'trades':tr,
            'total_return_net_pct':round((en[-1]/cap-1)*100,3),
            'ann_sharpe_net':round(sharpe(br),3),
            'max_drawdown_net_pct':round(mdd(en)*100,3)
        }
    }


def parse_args():
    p=argparse.ArgumentParser(description='Backtest stock vs vntl:MAG7 using HL closes')
    p.add_argument('--stocks-input',default='data/hl_mag7_daily.csv')
    p.add_argument('--output-dir',default='data')
    p.add_argument('--beta-window',type=int,default=30)
    p.add_argument('--z-window',type=int,default=20)
    p.add_argument('--entry-z',type=float,default=2.0)
    p.add_argument('--exit-z',type=float,default=0.5)
    p.add_argument('--fee',type=float,default=0.00045)
    p.add_argument('--capital',type=float,default=10000.0)
    return p.parse_args()


def main():
    a=parse_args(); od=Path(a.output_dir); od.mkdir(parents=True,exist_ok=True)
    sdf=pd.read_csv(a.stocks_input)
    sm=sdf[sdf['ticker'].isin(MAG7)].pivot_table(index='date',columns='ticker',values='close',aggfunc='last').sort_index().dropna(subset=MAG7)

    # Fetch vntl:MAG7 daily closes over wide window
    now=int(time.time()*1000)
    start=int((time.time()-240*86400)*1000)
    c=post({'type':'candleSnapshot','req':{'coin':'vntl:MAG7','interval':'1d','startTime':start,'endTime':now}})
    v=[]
    for x in c:
        d=datetime.fromtimestamp(int(x['t'])/1000,timezone.utc).strftime('%Y-%m-%d')
        v.append({'date':d,'vntl_mag7_close':float(x['c'])})
    vdf=pd.DataFrame(v).sort_values('date').drop_duplicates('date',keep='last')

    # Build aligned panel
    panel=sm.reset_index().merge(vdf,on='date',how='inner').sort_values('date').reset_index(drop=True)
    ret=panel[['date']+MAG7+['vntl_mag7_close']].copy()
    for col in MAG7+['vntl_mag7_close']:
        ret[col]=ret[col].pct_change()
    ret=ret.dropna().reset_index(drop=True)

    if len(ret)<max(a.beta_window+5,a.z_window+5):
        raise RuntimeError(f'Not enough overlap bars: {len(ret)}')

    print('='*74)
    print('ARBITRAGE BACKTEST: each stock vs vntl:MAG7')
    print('='*74)
    print(f"Overlap bars: {len(ret)} | Range: {ret['date'].min()} -> {ret['date'].max()}")
    print(f"Params: beta_window={a.beta_window}, z_window={a.z_window}, entry_z={a.entry_z}, exit_z={a.exit_z}, fee={a.fee*10000:.2f} bps")
    print()

    results={}
    for t in MAG7:
        out=run_pair(
            dates=ret['date'].tolist(),
            r_s=ret[t].tolist(),
            r_i=ret['vntl_mag7_close'].tolist(),
            beta_w=a.beta_window,z_w=a.z_window,entry=a.entry_z,exit_=a.exit_z,fee=a.fee,cap=a.capital
        )
        results[t]=out
        m=out['metrics']
        print(f"{t:6s}: trades={m['trades']:2d} ret={m['total_return_net_pct']:+6.2f}% sharpe={m['ann_sharpe_net']:+5.2f} mdd={m['max_drawdown_net_pct']:6.2f}%")

        with (od/f'stock_vs_vntl_mag7_signals_{t}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['date','position','zscore','spread','beta','equity_net'])
            w.writeheader()
            for i,d in enumerate(out['dates']):
                z=out['zscore'][i]; b=out['beta'][i]
                w.writerow({'date':d,'position':out['position'][i],'zscore':'' if math.isnan(z) else round(z,6),'spread':round(out['spread'][i],8),'beta':'' if math.isnan(b) else round(b,6),'equity_net':round(out['equity_net'][i],6)})

    min_len=min(len(results[t]['equity_net']) for t in MAG7)
    p_eq=[sum(results[t]['equity_net'][i] for t in MAG7) for i in range(min_len)]
    p_ret=p_eq[-1]/(a.capital*len(MAG7))-1.0
    p_bar=[p_eq[i]/p_eq[i-1]-1.0 for i in range(1,len(p_eq))]
    p_sh=sharpe(p_bar); p_dd=mdd(p_eq)
    results['__portfolio__']={'metrics':{'total_return_net_pct':round(p_ret*100,3),'ann_sharpe_net':round(p_sh,3),'max_drawdown_net_pct':round(p_dd*100,3)}}

    print('-'*74)
    print(f'PORTFOLIO: ret={p_ret*100:+.2f}% sharpe={p_sh:+.2f} mdd={p_dd*100:.2f}%')

    # Save overlap panel for transparency
    panel.to_csv(od/'stock_vs_vntl_overlap_prices.csv',index=False)

    (od/'stock_vs_vntl_mag7_backtest_results.json').write_text(json.dumps({
        'params':{'beta_window':a.beta_window,'z_window':a.z_window,'entry_z':a.entry_z,'exit_z':a.exit_z,'fee':a.fee,'capital':a.capital},
        'overlap_bars':len(ret),
        'results':{k:v['metrics'] for k,v in results.items()}
    },indent=2),encoding='utf-8')

    md=['# Stock vs vntl:MAG7 Arbitrage Backtest\n',
        'Uses HL daily closes for 7 stocks and live `vntl:MAG7` daily candles.\n',
        f"**Overlap bars:** {len(ret)}  |  **Range:** {ret['date'].min()} -> {ret['date'].max()}  \\n",
        f"**Params:** beta_window={a.beta_window}, z_window={a.z_window}, entry_z={a.entry_z}, exit_z={a.exit_z}, fee={a.fee*10000:.2f} bps\n",
        '\n## Per-stock\n','| Stock | Trades | Net Return | Ann Sharpe | Max DD |','|---|---:|---:|---:|---:|']
    for t in MAG7:
        m=results[t]['metrics']; md.append(f"| {t} | {m['trades']} | {m['total_return_net_pct']:+.2f}% | {m['ann_sharpe_net']:+.2f} | {m['max_drawdown_net_pct']:.2f}% |")
    pm=results['__portfolio__']['metrics']
    md += ['\n## Portfolio\n',f"- Net return: **{pm['total_return_net_pct']:+.2f}%**",f"- Annualized Sharpe: **{pm['ann_sharpe_net']:+.2f}**",f"- Max drawdown: **{pm['max_drawdown_net_pct']:.2f}%**"]
    (od/'stock_vs_vntl_mag7_backtest_results.md').write_text('\n'.join(md),encoding='utf-8')

    fig,axs=plt.subplots(4,2,figsize=(14,12)); ax=axs.flatten()
    for i,t in enumerate(MAG7):
        eq=results[t]['equity_net']; m=results[t]['metrics']
        ax[i].plot(range(len(eq)),eq,color='#1f77b4',linewidth=1.4)
        ax[i].axhline(a.capital,color='#666',linewidth=0.6)
        ax[i].set_title(f"{t} | S={m['ann_sharpe_net']:+.2f} R={m['total_return_net_pct']:+.1f}%")
        ax[i].grid(alpha=0.25)
    ax[7].plot(range(len(p_eq)),p_eq,color='#d62728',linewidth=1.6)
    ax[7].axhline(a.capital*len(MAG7),color='#666',linewidth=0.6); ax[7].set_title('Portfolio'); ax[7].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(od/'stock_vs_vntl_mag7_backtest_equity.png',dpi=160); plt.close(fig)

    print('Saved: data/stock_vs_vntl_overlap_prices.csv')
    print('Saved: data/stock_vs_vntl_mag7_backtest_results.json')
    print('Saved: data/stock_vs_vntl_mag7_backtest_results.md')
    print('Saved: data/stock_vs_vntl_mag7_backtest_equity.png')

if __name__=='__main__':
    main()
