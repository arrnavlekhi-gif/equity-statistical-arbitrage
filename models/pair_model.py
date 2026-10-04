import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def fit_orientation(dep_price, indep_price):
    df = pd.concat([dep_price, indep_price], axis=1).dropna()
    y = np.log(df.iloc[:,0]); x = np.log(df.iloc[:,1])
    X = sm.add_constant(x)
    m = sm.OLS(y, X).fit()
    alpha = float(m.params.iloc[0]); beta = float(m.params.iloc[1])
    spread = y - alpha - beta*x
    lagged = spread.shift(1); delta = spread.diff()
    reg = pd.concat([delta.rename("d"),lagged.rename("l")],axis=1).dropna()
    hl = np.inf
    if len(reg) > 10:
        mm = sm.OLS(reg["d"], sm.add_constant(reg["l"])).fit()
        lam = float(mm.params["l"])
        if lam < 0:
            hl = float(-np.log(2)/lam)
    stat,pval,crit = coint(y,x)
    return {"alpha":alpha,"beta":beta,"r_squared":float(m.rsquared),
            "cointegration_pvalue":float(pval),"cointegration_stat":float(stat),
            "half_life":hl,"spread":spread}

def choose_orientation(a, b, max_half_life=60):
    ab = fit_orientation(a,b)
    ba = fit_orientation(b,a)
    items = [("A_on_B",ab),("B_on_A",ba)]
    admissible = [(n,r) for n,r in items
                  if np.isfinite(r["half_life"]) and 0 < r["half_life"] <= max_half_life]
    pool = admissible if admissible else items
    name,result = min(pool, key=lambda kv: kv[1]["cointegration_pvalue"])
    return name,result,{"A_on_B":ab,"B_on_A":ba}

def rolling_market_beta(stock_ret, market_ret, window=126):
    return stock_ret.rolling(window).cov(market_ret) / market_ret.rolling(window).var()
