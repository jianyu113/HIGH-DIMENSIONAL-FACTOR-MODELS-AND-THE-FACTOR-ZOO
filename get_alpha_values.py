import numpy as np
import pandas as pd

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def _cum_ret(x):
    return (1 + x).prod() - 1

def past_1m_return(panel):
    tmp = panel.sort_values(["etf_id", "Date"]).copy()
    tmp["past_1m_return"] = (tmp.groupby("etf_id")["Adj Close"].pct_change())
    return tmp[["Date", "etf_id", "past_1m_return"]]

def past_3m_return(panel):
    tmp = panel.sort_values(["etf_id", "Date"]).copy()
    r1m = tmp.groupby("etf_id")["Adj Close"].pct_change()
    tmp["past_3m_return"] = (
        r1m.groupby(tmp["etf_id"])
        .rolling(window=3, min_periods=3)
        .apply(_cum_ret, raw=False)
        .reset_index(level=0, drop=True)
    )
    return tmp[["Date", "etf_id", "past_3m_return"]]

def past_6m_return(panel):
    tmp = panel.sort_values(["etf_id", "Date"]).copy()
    r1m = tmp.groupby("etf_id")["Adj Close"].pct_change()
    tmp["past_6m_return"] = (
        r1m.groupby(tmp["etf_id"])
        .rolling(window=6, min_periods=6)
        .apply(_cum_ret, raw=False)
        .reset_index(level=0, drop=True)
    )
    return tmp[["Date", "etf_id", "past_6m_return"]]

def volatility_3m(panel):
    tmp = panel.sort_values(["etf_id", "Date"]).copy()
    r1m = tmp.groupby("etf_id")["Adj Close"].pct_change()

    tmp["volatility_3m"] = (
        r1m.groupby(tmp["etf_id"])
        .rolling(window=3, min_periods=3)
        .std()
        .reset_index(level=0, drop=True)
    )
    return tmp[["Date", "etf_id", "volatility_3m"]]

def volatility_6m(panel):
    tmp = panel.sort_values(["etf_id", "Date"]).copy()
    tmp["r_1m"] = tmp.groupby("etf_id")["Adj Close"].pct_change()
    tmp["volatility_6m"] = (
        tmp["r_1m"]
        .groupby(tmp["etf_id"])
        .rolling(window=6, min_periods=6)
        .std()
        .reset_index(level=0, drop=True)
    )
    return tmp[["Date", "etf_id", "volatility_6m"]]

def turnover_proxy(panel):
    tmp = panel.sort_values(["etf_id", "Date"]).copy()
    tmp["turnover"] = tmp["Adj Close"] * tmp["Volume"]
    return tmp[["Date", "etf_id", "turnover"]]

def size_factor(panel):
    tmp = panel.sort_values(["etf_id", "Date"]).copy()
    tmp["size"] = np.log(tmp["Adj Close"] * tmp["Volume"] + 1.0)
    return tmp[["Date", "etf_id", "size"]]

def amihud(panel):
    tmp = panel.sort_values(["etf_id", "Date"]).copy()
    tmp["r_1m"] = tmp.groupby("etf_id")["Adj Close"].pct_change()
    dollar_vol = tmp["Adj Close"] * tmp["Volume"]
    dollar_vol = dollar_vol.replace(0, np.nan)

    tmp["amihud"] = tmp["r_1m"].abs() / dollar_vol
    return tmp[["Date", "etf_id", "amihud"]]

def beta_6m(panel, mkt_id="SPY"):
    tmp = panel.sort_values(["etf_id", "Date"]).copy()
    tmp["r_1m"] = tmp.groupby("etf_id")["Adj Close"].pct_change()
    ret_wide = tmp.pivot(index="Date", columns="etf_id", values="r_1m").sort_index()
    if mkt_id not in ret_wide.columns:
        raise ValueError(f"market id {mkt_id} not in etf_id columns")

    mkt = ret_wide[mkt_id]
    beta_wide = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns, dtype=float)
    for col in ret_wide.columns:
        r_i = ret_wide[col]
        cov = r_i.rolling(window=6, min_periods=6).cov(mkt)
        var = mkt.rolling(window=6, min_periods=6).var()
        beta_wide[col] = cov / var
    beta_long = (
        beta_wide.stack()
        .reset_index()
        .rename(columns={"Date": "Date", "etf_id": "etf_id", 0: "beta_6m"})
    )
    beta_long.columns = ["Date", "etf_id", "beta_6m"]
    return beta_long[["Date", "etf_id", "beta_6m"]]

def sharpe_6m(panel):
    tmp = panel.sort_values(["etf_id", "Date"]).copy()
    tmp["r_1m"] = tmp.groupby("etf_id")["Adj Close"].pct_change()
    rolling_mean = (
        tmp["r_1m"]
        .groupby(tmp["etf_id"])
        .rolling(window=6, min_periods=6)
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolling_std = (
        tmp["r_1m"]
        .groupby(tmp["etf_id"])
        .rolling(window=6, min_periods=6)
        .std()
        .reset_index(level=0, drop=True)
    )
    tmp["sharpe_6m"] = rolling_mean / rolling_std.replace(0, np.nan)
    return tmp[["Date", "etf_id", "sharpe_6m"]]

def momentum_12m(panel):
    tmp = panel.sort_values(["etf_id", "Date"]).copy()
    tmp["r_1m"] = tmp.groupby("etf_id")["Adj Close"].pct_change()
    r_shift = tmp.groupby("etf_id")["r_1m"].shift(1)

    def _cum_ret(x):
        return (1 + x).prod() - 1
    tmp["momentum_12m"] = (
        r_shift
        .groupby(tmp["etf_id"])
        .rolling(window=11, min_periods=11)
        .apply(_cum_ret, raw=False)
        .reset_index(level=0, drop=True)
    )
    return tmp[["Date", "etf_id", "momentum_12m"]]

def volume_6m_pct_change(panel):
    tmp = panel.sort_values(["etf_id", "Date"]).copy()
    tmp["volume_6m_pct_change"] = (
        tmp.groupby("etf_id")["Volume"].pct_change(periods=6)
    )
    return tmp[["Date", "etf_id", "volume_6m_pct_change"]]
