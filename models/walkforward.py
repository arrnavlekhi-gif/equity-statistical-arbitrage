import numpy as np
import pandas as pd
from .pair_model import choose_orientation

def build_walkforward(close_a, close_b, formation_window=252, trading_window=21,
                      z_window=60, coint_threshold=0.05, max_half_life=60):
    df = pd.concat([close_a,close_b],axis=1).dropna()
    df.columns = ["A","B"]
    out = pd.DataFrame(index=df.index)
    out["price_a"]=df.A; out["price_b"]=df.B
    for c in ["alpha","beta","r_squared","spread","zscore",
              "cointegration_pvalue","half_life"]:
        out[c]=np.nan
    out["orientation"]=None
    out["pair_valid"]=False

    start = formation_window
    while start < len(df):
        end = min(start+trading_window,len(df))
        formation = df.iloc[start-formation_window:start]
        trade = df.iloc[start:end]
        orientation,res,_ = choose_orientation(formation.A, formation.B, max_half_life)
        alpha,beta = res["alpha"],res["beta"]

        if orientation=="A_on_B":
            fdep,find = np.log(formation.A),np.log(formation.B)
            tdep,tind = np.log(trade.A),np.log(trade.B)
        else:
            fdep,find = np.log(formation.B),np.log(formation.A)
            tdep,tind = np.log(trade.B),np.log(trade.A)

        fspread = fdep-alpha-beta*find
        mu = fspread.tail(z_window).mean()
        sd = fspread.tail(z_window).std(ddof=1)
        tspread = tdep-alpha-beta*tind
        z = (tspread-mu)/sd if np.isfinite(sd) and sd>0 else np.nan
        idx = trade.index
        out.loc[idx,"alpha"]=alpha
        out.loc[idx,"beta"]=beta
        out.loc[idx,"r_squared"]=res["r_squared"]
        out.loc[idx,"spread"]=tspread.values
        if isinstance(z,pd.Series): out.loc[idx,"zscore"]=z.values
        out.loc[idx,"cointegration_pvalue"]=res["cointegration_pvalue"]
        out.loc[idx,"half_life"]=res["half_life"]
        out.loc[idx,"orientation"]=orientation
        valid = (res["cointegration_pvalue"]<coint_threshold and
                 np.isfinite(res["half_life"]) and 0<res["half_life"]<=max_half_life)
        out.loc[idx,"pair_valid"]=valid
        start=end
    return out
