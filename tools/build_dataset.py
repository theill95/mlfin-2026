"""One-time builder for the course dataset. NOT course material.

Downloads daily price data from Yahoo Finance via the yfinance package and
writes the CSV files in ../data/ that all course sessions use. Students
never run this script and the course never downloads live data: the
resulting CSVs are committed to the repository.

Source     : Yahoo Finance daily quotes via yfinance (auto-adjusted closes,
             i.e. split- and dividend-adjusted)
Fetched    : 2026-07-19
Coverage   : 2015-01-01 .. 2024-12-31
Terms note : Data obtained through the public Yahoo Finance service for
             non-commercial teaching. Recorded for provenance.

Usage (from the repository root):
    pip install yfinance
    python tools/build_dataset.py

Outputs:
    data/prices.csv            tidy long format: date,ticker,close,volume
    data/aapl_2024_closes.csv  date + close for AAPL in 2024
    data/ko_2024_closes.csv    date + close for KO in 2024
                               (small files used by Session 1 before pandas)
"""

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "KO", "PG", "XOM", "JNJ", "WMT", "DIS", "SPY"]
START, END = "2015-01-01", "2025-01-01"  # yfinance end date is exclusive
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Trading-day count sanity band for 2015-2024 (NYSE has ~252 per year).
MIN_ROWS, MAX_ROWS = 2480, 2540


def fetch_one(ticker: str) -> pd.DataFrame:
    hist = yf.Ticker(ticker).history(start=START, end=END, auto_adjust=True)
    if hist.empty:
        raise RuntimeError(f"{ticker}: Yahoo returned no data")
    df = hist.reset_index()[["Date", "Close", "Volume"]]
    df.columns = ["date", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["ticker"] = ticker
    return df[["date", "ticker", "close", "volume"]]


def sanity_check(df: pd.DataFrame) -> None:
    problems = []
    for ticker, g in df.groupby("ticker"):
        n = len(g)
        if not (MIN_ROWS <= n <= MAX_ROWS):
            problems.append(f"{ticker}: {n} rows outside [{MIN_ROWS}, {MAX_ROWS}]")
        if g["close"].isna().any() or (g["close"] <= 0).any():
            problems.append(f"{ticker}: missing or non-positive closes")
        gaps = g["date"].sort_values().diff().dt.days.max()
        if gaps and gaps > 7:  # allow long weekends/holidays, flag real holes
            problems.append(f"{ticker}: largest calendar gap {gaps} days")
    if problems:
        raise RuntimeError("Sanity check failed:\n  " + "\n  ".join(problems))


def main() -> None:
    frames = []
    for ticker in TICKERS:
        print(f"fetching {ticker} ...", flush=True)
        frames.append(fetch_one(ticker))
        time.sleep(1.0)  # be polite to the free service
    prices = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    sanity_check(prices)

    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "prices.csv"
    prices.to_csv(out, index=False, float_format="%.4f", date_format="%Y-%m-%d")
    print(f"wrote {out} ({len(prices)} rows, {prices['ticker'].nunique()} tickers)")

    for ticker in ["AAPL", "KO"]:
        small = prices[
            (prices["ticker"] == ticker) & (prices["date"].dt.year == 2024)
        ].sort_values("date")
        out_small = DATA_DIR / f"{ticker.lower()}_2024_closes.csv"
        small[["date", "close"]].to_csv(
            out_small, index=False, float_format="%.2f", date_format="%Y-%m-%d"
        )
        print(f"wrote {out_small} ({len(small)} rows)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
