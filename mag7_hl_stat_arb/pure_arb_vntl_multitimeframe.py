#!/usr/bin/env python3
from __future__ import annotations
import json, math, time
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import pandas as pd

API='https://api.hyperliquid.xyz/info'
MAG7=['AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA']
INTERVALS=['1h','4h','1d']


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
def sharpe(r,ppy):
    if len(r)<2:return float('nan')
    s=std(r)
    return float('nan') if (math.isnan(s) or s==0) else (mean(r)/s)*math.sqrt(ppy)
def mdd(eq):
    p=eq[0]; out=0.0
    for v in eq:
        p=max(p,v); out=min(out,(v/p)-1 if p>0 else 0.0)
    return out


def fetch_candles(coin, interval, start_ms, end_ms):
    data=post({'type':'candleSnapshot','req':{'coin':coin,'interval':interval,'startTime':int(start_ms),'endTime':int(end_ms)}})
    rows=[]
    if isinstance(data,list):
        for x in data:
            t=int(x['t'])
            dt=datetime.fromtimestamp(t/1000,timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            rows.append({'ts':t,'dt':dt,'close':float(x['c'])})
    rows=sorted(rows,key=lambda r:r['ts'])
    # dedup ts
    d={r['ts']:r for r in rows}
    return [d[k] for k in sorted(d.keys())]


def backtest_pure(ret_s, ret_i, z_w, entry, exit_, fee, cap):
    h=[ret_s[i]-ret_i[i] for i in range(len(ret_s))]
    spread=[]; lv=0.0
    for v in h: lv+=v; spread.append(lv)

    z=[None]*len(spread)
    for i in range(z_w,len(spread)):
        win=spread[i-z_w:i]; mu=mean(win); sd=std(win)
        z[i]=0.0 if (math.isnan(sd) or sd==0) else (spread[i]-mu)/sd

    pos=[0]*len(ret_s); eq=[cap]*len(ret_s); br=[]; trades=0
    for i in range(1,len(ret_s)):
        zp=z[i-1]; pp=pos[i-1]
        if zp is None:
            pos[i]=0; eq[i]=eq[i-1]; br.append(0.0); continue
        if pp==0:
            np=-1 if zp>=entry else (1 if zp<=-entry else 0)
        else:
            if abs(zp)<=exit_: np=0
            elif pp==-1 and zp<=-entry: np=1
            elif pp==1 and zp>=entry: np=-1
            else: np=pp
        pos[i]=np
        turn=abs(np-pp)
        if turn>0: trades+=1
        net=np*h[i]-turn*fee
        eq[i]=eq[i-1]*(1+net)
        br.append(net)

    return {'ret_pct':(eq[-1]/cap-1)*100,'mdd_pct':mdd(eq)*100,'trades':trades,'bar_returns':br,'equity':eq}


def periods_per_year(interval):
    if interval=='1h': return 24*365
    if interval=='4h': return 6*365
    return 365


def main():
    base=Path('/Users/leeisaackaiyui/Desktop/backtest.worktrees/copilot-worktree-2026-03-28T16-33-21/mag7_hl_stat_arb')
    out=base/'data'
    out.mkdir(parents=True,exist_ok=True)

    now=int(time.time()*1000)
    start=int((time.time()-240*86400)*1000)

    summary=[]
    by_interval={}

    for interval in INTERVALS:
        print('\n'+'='*76)
        print('PURE ARB STOCK vs vntl:MAG7 | interval',interval)
        print('='*76)

        # fetch vntl index
        v=fetch_candles('vntl:MAG7',interval,start,now)
        if not v:
            print('No vntl candles for',interval)
            continue
        vdf=pd.DataFrame(v).rename(columns={'close':'vntl'})[['ts','dt','vntl']]

        # fetch all stocks
        mats=[]
        for t in MAG7:
            c=fetch_candles(f'xyz:{t}',interval,start,now)
            if not c:
                print('No candles for',t,interval)
                continue
            df=pd.DataFrame(c).rename(columns={'close':t})[['ts',t]]
            mats.append(df)

        if len(mats)<7:
            print('Missing stock series for',interval)
            continue

        panel=vdf.copy()
        for m in mats:
            panel=panel.merge(m,on='ts',how='inner')
        panel=panel.sort_values('ts').reset_index(drop=True)
        # keep dt from vntl ts mapping
        panel['dt']=panel['ts'].apply(lambda x: datetime.fromtimestamp(x/1000,timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))

        # returns
        ret=panel[['dt']+MAG7+['vntl']].copy()
        for c in MAG7+['vntl']:
            ret[c]=ret[c].pct_change()
        ret=ret.dropna().reset_index(drop=True)

        # dynamic z-window by interval (roughly 20 days lookback)
        if interval=='1h': z_w=24*5  # ~5d to avoid too sparse start
        elif interval=='4h': z_w=6*7 # ~1w
        else: z_w=20

        ppy=periods_per_year(interval)
        res={}
        for t in MAG7:
            bt=backtest_pure(ret[t].tolist(),ret['vntl'].tolist(),z_w=z_w,entry=2.0,exit_=0.5,fee=0.00045,cap=10000)
            bt['sharpe']=sharpe(bt['bar_returns'],ppy)
            res[t]=bt
            print(f"{t:6s}: trades={bt['trades']:3d} ret={bt['ret_pct']:+7.2f}% sharpe={bt['sharpe']:+6.2f} mdd={bt['mdd_pct']:7.2f}%")

        L=min(len(res[t]['equity']) for t in MAG7)
        p_eq=[sum(res[t]['equity'][i] for t in MAG7) for i in range(L)]
        p_br=[p_eq[i]/p_eq[i-1]-1 for i in range(1,L)]
        p={
            'ret_pct':(p_eq[-1]/(10000*7)-1)*100,
            'sharpe':sharpe(p_br,ppy),
            'mdd_pct':mdd(p_eq)*100,
            'bars':len(ret),
            'start':ret['dt'].min(),
            'end':ret['dt'].max(),
            'z_window':z_w,
        }
        print('-'*76)
        print(f"PORTFOLIO: ret={p['ret_pct']:+.2f}% sharpe={p['sharpe']:+.2f} mdd={p['mdd_pct']:.2f}% bars={p['bars']}")

        by_interval[interval]={'portfolio':p,'by_ticker':{k:{'ret_pct':round(v['ret_pct'],3),'sharpe':round(v['sharpe'],3),'mdd_pct':round(v['mdd_pct'],3),'trades':v['trades']} for k,v in res.items()}}
        summary.append({'interval':interval,**p})

    sdf=pd.DataFrame(summary).sort_values('interval')
    sdf.to_csv(out/'pure_arb_vntl_multitimeframe_summary.csv',index=False)
    (out/'pure_arb_vntl_multitimeframe_summary.json').write_text(json.dumps(by_interval,indent=2),encoding='utf-8')

    if not sdf.empty:
        print('\n'+'='*76)
        print('PORTFOLIO SUMMARY BY TIMEFRAME')
        print('='*76)
        print(sdf[['interval','bars','start','end','ret_pct','sharpe','mdd_pct','z_window']].to_string(index=False))

if __name__=='__main__':
    main()
