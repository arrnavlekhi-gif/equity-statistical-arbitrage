import pandas as pd
from strategy.execution import run_execution
from analytics.performance import stats

def run_sensitivity(model_df, engine_kwargs):
    rows=[]
    for entry in [1.5,1.75,2.0,2.25,2.5]:
        for exit_ in [0.0,0.25,0.5]:
            daily,ledger=run_execution(model_df,z_entry=entry,z_exit=exit_,**engine_kwargs)
            p=stats(daily.net_return,daily.position)
            rows.append({"entry_z":entry,"exit_z":exit_,"sharpe":p["sharpe"],
                         "annual_return":p["annual_return"],"max_drawdown":p["max_drawdown"],
                         "trades":len(ledger),"time_in_market":p["time_in_market"]})
    return pd.DataFrame(rows)
