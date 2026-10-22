"""Run this on YOUR computer (the cloud workspace cannot reach market-data sites).
    pip install yfinance pandas
    python data/download_indices.py
Writes data/indices/<name>.csv with columns Date, Close (daily closing level of the price index;
no split adjustment is needed). Then commit data/indices/ or send the folder back.
Do not inspect or plot the series before the multi-market pre-registration is committed.
"""
from pathlib import Path
import pandas as pd, yfinance as yf

TICKERS = {  # name: Yahoo symbol
    "sp500": "^GSPC", "nikkei225": "^N225", "dax": "^GDAXI", "ftse100": "^FTSE",
    "hangseng": "^HSI", "kospi": "^KS11", "sensex": "^BSESN", "asx200": "^AXJO",
    "eurostoxx50": "^STOXX50E", "tsx": "^GSPTSE", "nifty50": "^NSEI", "india_vix": "^INDIAVIX", "vix": "^VIX",
}
out = Path(__file__).parent / "indices"; out.mkdir(exist_ok=True)
for name, sym in TICKERS.items():
    df = yf.download(sym, start="1990-01-01", auto_adjust=False, progress=False)
    if df.empty:
        print(f"{name}: no data"); continue
    s = df["Close"].squeeze().dropna()
    pd.DataFrame({"Date": s.index.strftime("%Y-%m-%d"), "Close": s.values}).to_csv(out / f"{name}.csv", index=False)
    print(f"{name}: {len(s)} rows, {s.index.min().date()} .. {s.index.max().date()}")
