import yfinance as yf
import pandas as pd

etf_id = [
    "SPY","IVV","VOO",
    "QQQ","IWM","DIA",
    "EFA","EEM","EWJ",
    "XLK","XLF","XLY","XLP","XLI","XLE","XLV","XLU","XLB",
    "TLT","IEF","SHY",
    "GLD","SLV","USO",
    "MTUM","QUAL","SIZE","USMV","VLUE",
]

data = yf.download(
    etf_id,
    start="2010-01-01",
    end="2024-01-01",
    interval="1mo",
    auto_adjust=False
)

#print(data.columns)
panel = data.stack(level=1).reset_index()
panel = panel.rename(columns={'Ticker': 'etf_id'})
panel = panel.sort_values(['etf_id', 'Date'])
