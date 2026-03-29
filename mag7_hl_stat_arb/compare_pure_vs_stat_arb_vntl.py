#!/usr/bin/env python3
from __future__ import annotations
import json, math, time
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import pandas as pd

API='https://api.hyperliquid.xyz/info'
MAG7=['AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA']


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
    p=eq[0]; out=0.0
    for v in eq:
        p=max(p,v); out=min(out,(v/p)-1 if p>0 else 0.0)
    return out

def rolling_beta(y,x,w):
    out=[None]*len(y)
    for i in range(w,len(y)):
        yy=y[i-w:i]; xx=x[i-w:i]
        mx,my=mean(xx),mean(yy)
        sxx=sum((q-mx)**2 for q in xx)
        out[i]=0.0 if sxx<=0 else sum((xx[k]-mx)*(yy[k]-my) for k in range(len(yy)))/sxx
    return out

def rolling_z(s,w):
    out=[None]*len(s)
    for i in range(w,len(s)):
        win=s[i-w:i]; mu=mean(win); sd=std(win)
        out[i]=0.0 if (math.isnan(sd) or sd==0) else (s[i]-mu)/sd
    return out

def backtest(ret_s, ret_i, model='stat', beta_w=30, z_w=20, entry=2.0, exit_=0.5, fee=0.00045, cap=10000):
    # model='pure': spread uses simple return diff; model='stat': rolling-beta hedged return
    if model=='stat':
        b=rolling_beta(ret_s,ret_i,beta_w)
        h=[0.0 if b[i] is None else ret_s[i]-b[i]*ret_i[i] for i in range(len(ret_s))]
    else:
        b=[1.0]*len(ret_s)
        h=[ret_s[i]-ret_i[i] for i in range(len(ret_s))]

    spread=[]; lv=0.0
    for v in h: lv+=v; spread.append(lv)
    z=rolling_z(spread,z_w)

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

    return {
        'ret_pct':(eq[-1]/cap-1)*100,
        'sharpe':sharpe(br),
        'mdd_pct':mdd(eq)*100,
        'trades':trades,
        'equity':eq,
    }


def main():
    base=Path('/Users/leeisaackaiyui/Desktop/backtest.worktrees/copilot-worktree-2026-03-28T16-33-21/mag7_hl_stat_arb')
    data=base/'data'

    sdf=pd.read_csv(data/'hl_mag7_daily.csv')
    sm=sdf[sdf['ticker'].isin(MAG7)].pivot_table(index='date',columns='ticker',values='close',aggfunc='last').sort_index().dropna(subset=MAG7)

    now=int(time.time()*1000)
    start=int((time.time()-240*86400)*1000)
    c=post({'type':'candleSnapshot','req':{'coin':'vntl:MAG7','interval':'1d','startTime':start,'endTime':now}})
    rows=[]
    for x in c:
        rows.append({'date':datetime.fromtimestamp(int(x['t'])/1000,timezone.utc).strftime('%Y-%m-%d'),'vntl':float(x['c'])})
    vdf=pd.DataFrame(rows).sort_values('date').drop_duplicates('date',keep='last')

    panel=sm.reset_index().merge(vdf,on='date',how='inner').sort_values('date').reset_index(drop=True)
    ret=panel[['date']+MAG7+['vntl']].copy()
    for c in MAG7+['vntl']:
        ret[c]=ret[c].pct_change()
    ret=ret.dropna().reset_index(drop=True)

    pure={}; stat={}
    for t in MAG7:
        rs=ret[t].tolist(); ri=ret['vntl'].tolist()
        pure[t]=backtest(rs,ri,model='pure')
        stat[t]=backtest(rs,ri,model='stat')

    def port(res):
        L=min(len(res[t]['equity']) for t in MAG7)
        p=[sum(res[t]['equity'][i] for t in MAG7) for i in range(L)]
        rb=[p[i]/p[i-1]-1 for i in range(1,L)]
        return {'ret_pct':(p[-1]/(10000*7)-1)*100,'sharpe':sharpe(rb),'mdd_pct':mdd(p)*100}

    pp=port(pure); sp=port(stat)

    rows=[]
    for t in MAG7:
        rows.append({
            'ticker':t,
            'pure_ret_pct':round(pure[t]['ret_pct'],3),
            'pure_sharpe':round(pure[t]['sharpe'],3),
            'stat_ret_pct':round(stat[t]['ret_pct'],3),
            'stat_sharpe':round(stat[t]['sharpe'],3),
            'delta_sharpe':round(stat[t]['sharpe']-pure[t]['sharpe'],3),
            'delta_ret_pct':round(stat[t]['ret_pct']-pure[t]['ret_pct'],3),
        })

    cmp_df=pd.DataFrame(rows).sort_values('delta_sharpe',ascending=False)
    cmp_df.to_csv(data/'pure_vs_stat_vntl_comparison.csv',index=False)

    out={
        'period':{'bars':len(ret),'start':ret['date'].min(),'end':ret['date'].max()},
        'portfolio':{
            'pure':{'ret_pct':round(pp['ret_pct'],3),'sharpe':round(pp['sharpe'],3),'mdd_pct':round(pp['mdd_pct'],3)},
            'stat':{'ret_pct':round(sp['ret_pct'],3),'sharpe':round(sp['sharpe'],3),'mdd_pct':round(sp['mdd_pct'],3)},
            'delta_stat_minus_pure':{
                'ret_pct':round(sp['ret_pct']-pp['ret_pct'],3),
                'sharpe':round(sp['sharpe']-pp['sharpe'],3),
                'mdd_pct':round(sp['mdd_pct']-pp['mdd_pct'],3),
            }
        },
        'by_ticker':rows,
    }
    (data/'pure_vs_stat_vntl_comparison.json').write_text(json.dumps(out,indent=2),encoding='utf-8')

    print('Period:',out['period'])
    print('Portfolio PURE:',out['portfolio']['pure'])
    print('Portfolio STAT:',out['portfolio']['stat'])
    print('Delta STAT-PURE:',out['portfolio']['delta_stat_minus_pure'])

if __name__=='__main__':
    main()
