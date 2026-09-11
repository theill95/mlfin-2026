# -*- coding: utf-8 -*-
"""Build the wide table Session 6 works on, from data/prices.csv.

One row per trading day. Nineteen feature columns, all looking back from that
day, and one target looking forward:

    vol_5d .. vol_120d   the S&P 500 fund's daily-return volatility over six windows
    ret_5d .. ret_60d    its average daily return over three windows
    AAPL_vol .. DIS_vol  the 20-day volatility of each of the ten stocks
    vol_next             the fund's volatility over the NEXT twenty trading days

Returns are in percent. The deck loads this file rather than building it, so
the lecture spends its time on the models; building it is an exercise. A copy
goes to the root data/ folder as well, which is where the notebooks' Colab
fallback URL points.

    python tools/build_session06_table.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTS = [ROOT / "session_06" / "data" / "market_features.csv",
        ROOT / "data" / "market_features.csv"]

prices = pd.read_csv(ROOT / "data" / "prices.csv", parse_dates=["date"])
rets = prices.pivot(index="date", columns="ticker", values="close").pct_change().dropna() * 100
r = rets["SPY"]

table = pd.DataFrame()
for w in [5, 10, 20, 40, 60, 120]:
    table["vol_" + str(w) + "d"] = r.rolling(w).std()
for w in [5, 20, 60]:
    table["ret_" + str(w) + "d"] = r.rolling(w).mean()
for t in ["AAPL", "MSFT", "NVDA", "JPM", "KO", "PG", "XOM", "JNJ", "WMT", "DIS"]:
    table[t + "_vol"] = rets[t].rolling(20).std()
table["vol_next"] = r.rolling(20).std().shift(-20)
table = table.dropna()

for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index_label="date")
    print("wrote", out.relative_to(ROOT), table.shape)
