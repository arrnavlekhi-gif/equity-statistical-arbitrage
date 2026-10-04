from datetime import timedelta
import pandas as pd
import yfinance as yf

def normalize_symbol(symbol, market):
    s = symbol.strip().upper()
    if market == "India (NSE)" and not (s.endswith(".NS") or s.endswith(".BO")):
        s += ".NS"
    return s

def download_pair(a, b, start, end, market):
    a = normalize_symbol(a, market)
    b = normalize_symbol(b, market)
    data = yf.download([a,b], start=start, end=end+timedelta(days=1),
                       auto_adjust=True, progress=False, group_by="column")
    if data.empty or not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("No valid multi-ticker data returned.")
    out = pd.DataFrame(index=data.index)
    for f in ["Open","Close","Volume"]:
        out[f"{f.lower()}_a"] = data[f][a]
        out[f"{f.lower()}_b"] = data[f][b]
    out = out.dropna(subset=["open_a","open_b","close_a","close_b"])
    if len(out) < 300:
        raise ValueError("Not enough overlapping observations.")
    out.attrs["symbol_a"] = a
    out.attrs["symbol_b"] = b
    return out
