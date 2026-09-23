# -*- coding: utf-8 -*-
"""One-time builder for the credit table Session 9 works on. NOT course material.

Downloads the UCI "default of credit card clients" data set, renames its
columns into plain English, keeps seven of them, and writes one CSV. Students
never run this script: the resulting CSV is committed to the repository.

Source     : UCI Machine Learning Repository, data set 350, "default of
             credit card clients" (Yeh & Lien 2009). Payment records of
             30,000 credit card holders at a Taiwanese bank, April to
             September 2005. Amounts are New Taiwan dollars.
             https://archive.ics.uci.edu/dataset/350/
Licence    : Creative Commons Attribution 4.0 International (CC BY 4.0).
Fetched    : 2026-09-23

One row per borrower. Seven columns describing the borrower in September
2005, and one label:

    limit         the credit limit on the account
    age           the borrower's age in years
    late_now      months behind on the most recent bill, 0 if not behind
    months_late   how many of the last six months were at least one month late
    bill          the most recent bill
    paid          the amount paid against that bill
    utilisation   the most recent bill divided by the credit limit
    default       1 if the borrower defaulted the following month

The original PAY_* columns code "paid in full" and "paid on time" as -2, -1
and 0. Those three groups default at about the same rate, so the script
clips them to 0 and the column reads as months late.

The four demographic columns (sex, education, marriage, and their codes) are
left out. Their documented categories do not cover every value in the file,
and the session's subject is imbalance and cost rather than what may be used
in a credit decision.

    python tools/build_session09_table.py

Outputs:
    session_09/data/credit.csv
    data/credit.csv             the copy the notebooks' Colab fallback reads
"""
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
MEMBER = "default of credit card clients.xls"
ROOT = Path(__file__).resolve().parents[1]
OUTS = [ROOT / "session_09" / "data" / "credit.csv",
        ROOT / "data" / "credit.csv"]

print("downloading", URL)
with urlopen(URL, timeout=180) as response:
    archive = zipfile.ZipFile(io.BytesIO(response.read()))
with archive.open(MEMBER) as member:
    raw = pd.read_excel(member, header=1)

if len(raw) != 30000:
    raise RuntimeError(f"expected 30,000 rows, got {len(raw)}")

pay = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

table = pd.DataFrame({
    "limit": raw["LIMIT_BAL"],
    "age": raw["AGE"],
    "late_now": raw["PAY_0"].clip(lower=0),
    "months_late": (raw[pay] >= 1).sum(axis=1),
    "bill": raw["BILL_AMT1"],
    "paid": raw["PAY_AMT1"],
    "utilisation": (raw["BILL_AMT1"] / raw["LIMIT_BAL"]).round(4),
    "default": raw["default payment next month"],
})

if table.isna().any().any():
    raise RuntimeError("missing values in the finished table")
share = table["default"].mean()
if not 0.21 < share < 0.23:
    raise RuntimeError(f"default rate {share:.4f} is not the documented 0.2212")

for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print("wrote", out.relative_to(ROOT), table.shape,
          f"{out.stat().st_size / 1e6:.2f} MB, default rate {share:.4f}")
