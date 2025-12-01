import yfinance as yf
import pandas as pd

def get_etf_price_panel(etf_id, start, end, interval, auto_adjust):

    data = yf.download(
        etf_id,
        start = start,
        end = end,
        interval = interval,
        auto_adjust = auto_adjust
    )

    #print(data.columns)
    panel = data.stack(level=1).reset_index()
    panel = panel.rename(columns={'Ticker': 'etf_id'})
    panel = panel.sort_values(['etf_id', 'Date'])
    return panel
