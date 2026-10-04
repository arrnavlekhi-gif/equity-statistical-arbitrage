import numpy as np

def stats(returns,position):
    r=returns.dropna()
    vol=r.std(ddof=1)*np.sqrt(252)
    ann=(1+r).prod()**(252/len(r))-1 if len(r) else np.nan
    sharpe=r.mean()/r.std(ddof=1)*np.sqrt(252) if len(r) and r.std(ddof=1)>0 else np.nan
    wealth=(1+r.fillna(0)).cumprod(); dd=wealth/wealth.cummax()-1
    tim=(position.reindex(r.index).fillna(0)!=0).mean() if len(r) else np.nan
    return {"annual_return":float(ann),"annual_volatility":float(vol),
            "sharpe":float(sharpe),"max_drawdown":float(dd.min()),
            "time_in_market":float(tim)}

def pair_quality(p,hl,r2):
    pscore=np.clip((0.10-p)/0.10,0,1)*50
    if not np.isfinite(hl) or hl<=0: h=0
    elif hl<=30: h=30
    elif hl<=60: h=20
    else: h=5
    return float(np.clip(pscore+h+np.clip(r2,0,1)*20,0,100))
