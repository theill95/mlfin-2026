# -*- coding: utf-8 -*-
"""Build session_04_exercises.ipynb.

Same conventions as Sessions 1 to 3: pleasant intro, 4-star badges, toolkit card
with <abbr> hover docs, task -> work cell (blank-safe `...`) -> 1-2 folded hints
-> folded solution, no em-dashes, plain academic tone.

Session 4 is the conceptual session, so the exercises deliberately turn every
idea into something you compute rather than something you recite: the metrics
are written out as functions, leakage is found by fixing broken code, and
over-fitting is measured rather than described.

Only tools taught by the end of Session 4. New this session: shift / rolling
used as feature builders, reindex / isna / fillna / ffill / interpolate,
quantile, get_dummies, np.log, np.polyfit / np.polyval, and metrics written by
hand out of numpy and pandas. NOT taught, so never required: sklearn anything,
seaborn, statsmodels, apply, resample.

BLANK-SAFE RULES:
- every work cell assigns `x = ...` and then displays `x` bare. Never call a
  method on, index into, format, or do arithmetic with a placeholder.
- nothing depends on an earlier exercise having been solved: the setup cell
  provides every variable the tasks use.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "session_04" / "session_04_exercises.ipynb"

cells = []
LABELS = {1: "Warm-up", 2: "Core", 3: "Challenge", 4: "Stretch"}


def badge(n):
    return ("★" * n + "☆" * (4 - n)) + " " + LABELS[n]


def md(text):
    cells.append(new_markdown_cell(text))


def code(text, raises=False):
    c = new_code_cell(text)
    if raises:
        c.metadata["tags"] = ["raises-exception"]
    cells.append(c)


def _hints_solution(hints, sol_code, sol_note):
    if isinstance(hints, str):
        hints = [hints]
    for i, h in enumerate(hints):
        label = "Hint" if len(hints) == 1 else f"Hint {i+1}"
        md(f"<details>\n<summary>\U0001f4a1 {label}</summary>\n\n{h}\n\n</details>")
    md(f"<details>\n<summary>✅ Solution</summary>\n\n```python\n{sol_code}\n```\n\n{sol_note}\n\n</details>")
    md("---")


def ex(sid, title, n, task, work, hints, sol_code, sol_note):
    md(f"### {sid} · {title}  {badge(n)}\n\n{task}")
    code(work)
    _hints_solution(hints, sol_code, sol_note)


def section(header):
    md(header)


# ---- real numbers, computed here so every solution note is exact -----------
PX = pd.read_csv(ROOT / "data" / "prices.csv", parse_dates=["date"])
WIDE = PX.pivot(index="date", columns="ticker", values="close")
RET = WIDE.pct_change()

# A: the feature table
TBL = pd.DataFrame({"ret_1d": RET["AAPL"]})
TBL["ret_5d"] = RET["AAPL"].rolling(5).mean()
TBL["vol_20d"] = RET["AAPL"].rolling(20).std()
TBL["ret_next"] = RET["AAPL"].shift(-1)
TBL_CLEAN = TBL.dropna()
N_ROWS, N_COLS = TBL_CLEAN.shape
UP_SHARE = float((RET["AAPL"].shift(-1).dropna() > 0).mean())

# C: missing values
CAL = pd.date_range(WIDE.index.min(), WIDE.index.max(), freq="D")
DAILY = WIDE.reindex(CAL)
N_CAL, N_TRADING = len(CAL), len(WIDE)
N_MISSING = int(DAILY["AAPL"].isna().sum())
FFILL_RET = DAILY["AAPL"].ffill().pct_change().dropna()
N_ZERO_RET = int((FFILL_RET == 0).sum())
SD_TRUE = RET["AAPL"].std()
SD_FFILL = FFILL_RET.std()

# D: extremes
A = RET["AAPL"].dropna()
Q1, Q3 = A.quantile([0.25, 0.75])
IQR = Q3 - Q1
LO, HI = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
N_IQR = int(((A < LO) | (A > HI)).sum())
N_3SD = int(((A - A.mean()).abs() > 3 * A.std()).sum())
N_NORMAL = 0.0027 * len(A)
BEYOND = {}
for _t in RET.columns:
    _r = RET[_t].dropna()
    BEYOND[_t] = int(((_r - _r.mean()).abs() > 3 * _r.std()).sum())
BEYOND_S = pd.Series(BEYOND).sort_values(ascending=False)

# E: transformations
LOGR = np.log(WIDE["AAPL"]).diff().dropna()
MAX_GAP = float((LOGR - A).abs().max())
VOL_SD = PX[PX["ticker"] == "AAPL"]["volume"].std()
RET_SD = A.std()

# F: regression metrics
ACTUAL = RET["AAPL"].dropna()
MARKET = RET["SPY"].dropna()
PREDICTED = 0.00043 + 1.2073 * MARKET
RESID = ACTUAL - PREDICTED
MAE = float(RESID.abs().mean())
MSE = float((RESID ** 2).mean())
RMSE = float(np.sqrt(MSE))
R2 = float(1 - (RESID ** 2).sum() / ((ACTUAL - ACTUAL.mean()) ** 2).sum())
BAD = PREDICTED.copy()
BAD.iloc[-30] = 0.40
BAD_RESID = ACTUAL - BAD
MAE_BAD = float(BAD_RESID.abs().mean())
RMSE_BAD = float(np.sqrt((BAD_RESID ** 2).mean()))

# G: classification metrics
WENT_UP = ACTUAL > 0
SAID_UP = MARKET > 0
TP = int((SAID_UP & WENT_UP).sum())
FP = int((SAID_UP & ~WENT_UP).sum())
FN = int((~SAID_UP & WENT_UP).sum())
TN = int((~SAID_UP & ~WENT_UP).sum())
ACC = (TP + TN) / (TP + FP + FN + TN)
PREC = TP / (TP + FP)
REC = TP / (TP + FN)
F1 = 2 * PREC * REC / (PREC + REC)
CRASH = ACTUAL < -0.05
N_CRASH = int(CRASH.sum())
ACC_LAZY = float((~CRASH).mean())
SAID_UP_HI = MARKET > 0.01
TP_HI = int((SAID_UP_HI & WENT_UP).sum())
FP_HI = int((SAID_UP_HI & ~WENT_UP).sum())
PREC_HI = TP_HI / (TP_HI + FP_HI)
REC_HI = TP_HI / int(WENT_UP.sum())

# H: train and test
TRAIN = TBL_CLEAN.loc[:"2022-12-31"]
TEST = TBL_CLEAN.loc["2023-01-01":]
N_TRAIN, N_TEST = len(TRAIN), len(TEST)
TEST_SHARE = N_TEST / (N_TRAIN + N_TEST)
TRAIN_MEAN = TRAIN["ret_1d"].mean()
TRAIN_SD = TRAIN["ret_1d"].std()
FULL_SD = TBL_CLEAN["ret_1d"].std()


def _rmse(y, yhat):
    return float(np.sqrt(((np.asarray(y) - np.asarray(yhat)) ** 2).mean()))


RMSE_TRAIN = _rmse(TRAIN["ret_next"], TRAIN["ret_1d"])
RMSE_TEST = _rmse(TEST["ret_next"], TEST["ret_1d"])

# I: flexibility
_rng = np.random.default_rng(11)
XTR = np.sort(_rng.uniform(0, 1, 30))
YTR = np.sin(2 * np.pi * XTR) + _rng.normal(0, 0.28, 30)
XTE = np.sort(_rng.uniform(0, 1, 400))
YTE = np.sin(2 * np.pi * XTE) + _rng.normal(0, 0.28, 400)


def _poly_mse(deg):
    c = np.polyfit(XTR, YTR, deg)
    return (float(((np.polyval(c, XTR) - YTR) ** 2).mean()),
            float(((np.polyval(c, XTE) - YTE) ** 2).mean()))


TR1, TE1 = _poly_mse(1)
TR15, TE15 = _poly_mse(15)
CURVE = {d: _poly_mse(d) for d in range(1, 13)}
BEST_DEG = min(CURVE, key=lambda d: CURVE[d][1])

# J: end to end
MTBL = pd.DataFrame({"ret_1d": RET["MSFT"]})
MTBL["ret_5d"] = RET["MSFT"].rolling(5).mean()
MTBL["vol_20d"] = RET["MSFT"].rolling(20).std()
MTBL["ret_next"] = RET["MSFT"].shift(-1)
MTBL = MTBL.dropna()
MTRAIN = MTBL.loc[:"2022-12-31"]
MTEST = MTBL.loc["2023-01-01":]
NAIVE_RMSE = _rmse(MTEST["ret_next"], MTEST["ret_1d"])
BASE_RMSE = _rmse(MTEST["ret_next"], np.full(len(MTEST), MTRAIN["ret_next"].mean()))
NAIVE_R2 = 1 - ((MTEST["ret_next"] - MTEST["ret_1d"]) ** 2).sum() / \
    ((MTEST["ret_next"] - MTRAIN["ret_next"].mean()) ** 2).sum()

# ---------------------------------------------------------------- top matter
md(
"# \U0001f4d3 Session 4 · Exercises\n"
"### The statistical learning framework, data problems, metrics, and overfitting\n\n"
"Session 4 was mostly ideas. These exercises turn every one of them into "
"something you compute, because a metric you have written out yourself is a "
"metric you will never misread again.\n\n"
"You will build a feature table, break it deliberately and repair it, write the "
"four regression metrics and the four classification counts from scratch, split "
"a sample by time, and measure overfitting rather than take my word for it."
)

md(
"## How to use this notebook\n\n"
"- Run the **setup cell** below first. It loads the price data and prepares the "
"few extra objects the exercises use.\n"
"- Each exercise has a **task**, then a **code cell** for your work. Cells with "
"`...` are blanks to fill in. Replace them with real code.\n"
"- Stuck? Open the **\U0001f4a1 Hint**, but only after a genuine attempt. Open the "
"**✅ Solution** to *check* yourself, not to skip the thinking.\n"
"- Every cell runs cleanly even with the blanks still in place, so pressing "
"**Run all** never floods you with errors.\n"
"- Exercises do not depend on each other. If one defeats you, move on.\n\n"
"**You are not expected to finish all of these.** Do what you can, and come back "
"to the rest when you revise. Short on time? Read the hint, then the solution. A "
"worked solution you genuinely understand is real learning too.\n\n"
"**No model is fitted here**, with one deliberate exception in section I, where "
"a curve is fitted only so that you can watch it overfit."
)

md(
"### Difficulty\n\n"
"| badge | level | what to expect |\n"
"|:--|:--|:--|\n"
"| ★☆☆☆ | **Warm-up** | one idea, straight from the lecture |\n"
"| ★★☆☆ | **Core** | the skill you will actually use. Most exercises live here |\n"
"| ★★★☆ | **Challenge** | combine a few ideas, or think one step past what you were shown |\n"
"| ★★★★ | **Stretch** | a real puzzle for those who want one. Rare, and always solvable with today's tools |\n"
)

md(
"## \U0001f9f0 Your toolkit for today\n\n"
"Everything from Sessions 1 to 3 still applies. This card holds what Session 4 "
"added, plus the handful of pandas methods the lecture used along the way.\n\n"
"> Names in brackets (`values`, `frame`, `column`, ...) are **placeholders**: put "
"your own variable there. **Hover any tool** to see what it does.\n\n"
'<p><strong>Building features and a target</strong><br>\n'
'<abbr title="Move a column down by n rows. shift(1) is yesterday\'s value on today\'s row."><code>s.shift(1)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Move a column UP: shift(-1) puts tomorrow\'s value on today\'s row. This is how a target is built."><code>s.shift(-1)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="A moving window of n rows. Follow it with .mean(), .std() or .sum()."><code>s.rolling(20)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Build a table out of named columns. Series are lined up on their labels for you."><code>pd.DataFrame({\'a\': s1, \'b\': s2})</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Turn True and False into 1 and 0."><code>(s &gt; 0).astype(int)</code></abbr></p>\n\n'
'<p><strong>Missing values</strong><br>\n'
'<abbr title="Force a table onto a different set of row labels. Rows that did not exist arrive empty."><code>frame.reindex(index)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Every calendar day between two dates."><code>pd.date_range(start, end, freq=\'D\')</code></abbr> &nbsp;·&nbsp; '
'<abbr title="True wherever the value is missing."><code>frame.isna()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Count the missing values in each column."><code>frame.isna().sum()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Throw away rows with missing values."><code>frame.dropna()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Fill missing values with something you choose."><code>s.fillna(value)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Carry the last known value forward. It only ever looks backwards, which is what forecasting needs."><code>s.ffill()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Draw a straight line across the gap. Careful: it uses the value AFTER the gap."><code>s.interpolate()</code></abbr></p>\n\n'
'<p><strong>Extreme values</strong><br>\n'
'<abbr title="Count, mean, standard deviation, min, quartiles and max, all at once."><code>s.describe()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="The value below which that fraction of the data lies. Pass a list to get several at once."><code>s.quantile([0.25, 0.75])</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Absolute value, element by element."><code>s.abs()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Combine two conditions: and, or. Each side needs its own brackets."><code>(a) &amp; (b)  ·  (a) | (b)</code></abbr></p>\n\n'
'<p><strong>Transformations</strong><br>\n'
'<abbr title="The natural logarithm, element by element."><code>np.log(values)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="The difference between each row and the one before it."><code>s.diff()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Standardise: subtract the mean, divide by the standard deviation."><code>(s - s.mean()) / s.std()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="One 0/1 column per category. This is how text gets into a feature table."><code>pd.get_dummies(frame, columns=[\'sector\'], dtype=int)</code></abbr></p>\n\n'
'<p><strong>Scoring a prediction</strong><br>\n'
'<abbr title="Mean absolute error: the average size of a mistake, in the units of the target."><code>(y - yhat).abs().mean()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Mean squared error: punishes a big miss far harder than several small ones."><code>((y - yhat) ** 2).mean()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Root mean squared error: back in the units of the target."><code>np.sqrt(mse)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="How much better than always predicting the average. Zero means no better at all."><code>1 - rss / tss</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Count the days where both conditions hold. This is one cell of a confusion matrix."><code>(said &amp; happened).sum()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="The opposite of a boolean Series: True becomes False."><code>~said</code></abbr></p>\n\n'
'<p><strong>Splitting, and measuring flexibility</strong><br>\n'
'<abbr title="Rows up to a date, and rows from a date. A DataFrame with a date index slices with dates."><code>frame.loc[:\'2022-12-31\']  ·  frame.loc[\'2023-01-01\':]</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Fit a polynomial of the given degree and hand back its coefficients."><code>np.polyfit(x, y, degree)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Evaluate a fitted polynomial at some x values."><code>np.polyval(coef, x)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="An array of the same value, repeated. Useful as a baseline prediction."><code>np.full(n, value)</code></abbr></p>\n\n'
"**Formulas you will reach for**\n\n"
r"| what | formula |"
"\n|:--|:--|\n"
r"| Mean absolute error | $\text{MAE}=\dfrac{1}{n}\sum_i \lvert y_i-\hat{y}_i \rvert$ |"
"\n"
r"| Mean squared error | $\text{MSE}=\dfrac{1}{n}\sum_i (y_i-\hat{y}_i)^2$ |"
"\n"
r"| Root mean squared error | $\text{RMSE}=\sqrt{\text{MSE}}$ |"
"\n"
r"| R squared | $R^2=1-\dfrac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}$ |"
"\n"
r"| Log return | $r^{\log}_t=\log P_t-\log P_{t-1}$ |"
"\n"
r"| Standardised value | $z_i=\dfrac{x_i-\bar{x}}{s}$ |"
"\n"
r"| IQR rule | flag outside $[\,Q_1-1.5\,\text{IQR},\; Q_3+1.5\,\text{IQR}\,]$ |"
"\n"
r"| Precision, recall | $\dfrac{TP}{TP+FP}$, $\dfrac{TP}{TP+FN}$ |"
"\n"
)

md("---")

# ---------------------------------------------------------------- setup cell
md(
"## ⚙️ Setup: run this first\n\n"
"This loads the price data and prepares the handful of extra objects the "
"exercises use. If you are in Google Colab it downloads the data by itself."
)

code(
'''import os
import numpy as np
import pandas as pd

CANDIDATE_DIRS = ["data", os.path.join("..", "data"), "."]
REPO_RAW_URL = "https://raw.githubusercontent.com/theill95/mlfin-2026/main/data/"   # used when the CSV files are not next to the notebook


def load_csv(filename, **kwargs):
    """Read one of the course CSV files, wherever it happens to be."""
    for folder in CANDIDATE_DIRS:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return pd.read_csv(path, **kwargs)
    if REPO_RAW_URL is not None:
        return pd.read_csv(REPO_RAW_URL + filename, **kwargs)
    raise FileNotFoundError(
        f"Could not find {filename}. Run this notebook from the course folder, "
        f"upload the CSV into Colab, or set REPO_RAW_URL."
    )


# Eleven stocks, 2015 to 2024
prices = load_csv("prices.csv", parse_dates=["date"])
wide = prices.pivot(index="date", columns="ticker", values="close")
rets = wide.pct_change()

# The running example of Sessions 4: Apple against the market, same day
actual = rets["AAPL"].dropna()
market = rets["SPY"].dropna()

# A tiny pair of arrays, small enough to check any metric by hand
y_true = np.array([2.0, 4.0, 6.0, 8.0])
y_pred = np.array([3.0, 4.0, 5.0, 11.0])

# Four firms and their sectors, for the encoding exercise
firms = pd.DataFrame({
    "ticker": ["AAPL", "KO", "JPM", "XOM"],
    "sector": ["Tech", "Staples", "Financials", "Energy"],
})

# Invented data with a known truth, for section I only
rng = np.random.default_rng(11)
x_train = np.sort(rng.uniform(0, 1, 30))
y_train = np.sin(2 * np.pi * x_train) + rng.normal(0, 0.28, 30)
x_test = np.sort(rng.uniform(0, 1, 400))
y_test = np.sin(2 * np.pi * x_test) + rng.normal(0, 0.28, 400)

print("prices:", prices.shape, "rows x columns")
print("wide  :", wide.shape, "trading days x tickers")
print("actual:", len(actual), "daily returns")'''
)

md("---")

# ============================================== A. the table a model starts from
section(
"## \U0001f9f1 A · The table a model starts from\n\n"
"One row per observation, one column per feature, and one column that is the "
"target. Everything else in machine learning assumes you have built this first."
)

ex("A1", "One column of returns", 1,
   "Take Apple's daily returns out of `rets` into a Series called `aapl_ret`.",
   "aapl_ret = ...\naapl_ret",
   "`rets` is a table with one column per ticker. Select a column by name.",
   "aapl_ret = rets['AAPL']\naapl_ret",
   "A Series with a date on every value. The first one is `NaN`, because there is no day before the first day.")

ex("A2", "The beginnings of a feature table", 1,
   "Put that Series into a one-column DataFrame called `table`, with the column "
   "named `ret_1d`.",
   "table = ...\ntable",
   "`pd.DataFrame({'name': series})` builds a table from named columns.",
   "table = pd.DataFrame({'ret_1d': rets['AAPL']})\ntable",
   "One column so far. The date index came along for free, which is what makes the next few steps line up.")

ex("A3", "A slower-moving feature", 2,
   "Add a column `ret_5d` to `table`: the **average of the last five daily "
   "returns**. Then show the last three rows.",
   "table = pd.DataFrame({'ret_1d': rets['AAPL']})\n\ntable['ret_5d'] = ...\nresult = table.tail(3)\nresult",
   ["`s.rolling(5)` gives you a five-row window. Follow it with `.mean()`.",
    "The window ends on the current row, so this feature only ever uses today and the four days before it."],
   "table = pd.DataFrame({'ret_1d': rets['AAPL']})\n\ntable['ret_5d'] = rets['AAPL'].rolling(5).mean()\nresult = table.tail(3)\nresult",
   "A five-day average moves far more smoothly than a single day. The first four rows are `NaN`, because a five-day window needs five days.")

ex("A4", "A risk feature", 2,
   "Add `vol_20d` to the same table: the **standard deviation of the last twenty "
   "daily returns**.",
   "table = pd.DataFrame({'ret_1d': rets['AAPL']})\n\ntable['vol_20d'] = ...\nresult = table.tail(3)\nresult",
   "Same `rolling` idea as before, with `.std()` instead of `.mean()`.",
   "table = pd.DataFrame({'ret_1d': rets['AAPL']})\n\ntable['vol_20d'] = rets['AAPL'].rolling(20).std()\nresult = table.tail(3)\nresult",
   "This is the volatility you built by hand in Session 2, recomputed on a moving window. As a feature it says how nervous the recent past has been.")

ex("A5", "The target", 2,
   "Add the column you actually want to predict: `ret_next`, **tomorrow's return**, "
   "on today's row.",
   "table = pd.DataFrame({'ret_1d': rets['AAPL']})\n\ntable['ret_next'] = ...\nresult = table.tail(4)\nresult",
   ["`shift` moves a column up or down. Which direction brings the future backwards?",
    "`shift(-1)` moves values up by one row, so tomorrow's return lands on today's row."],
   "table = pd.DataFrame({'ret_1d': rets['AAPL']})\n\ntable['ret_next'] = rets['AAPL'].shift(-1)\nresult = table.tail(4)\nresult",
   "Look down the two columns: `ret_next` on one row is `ret_1d` on the next. The very last row has no tomorrow, so its target is `NaN`.")

ex("A6", "Rows a model can use", 2,
   "Build the full table (all three features and the target), drop every row that "
   "has a missing value, and store the result as `clean`.",
   "table = pd.DataFrame({\n    'ret_1d': rets['AAPL'],\n    'ret_5d': rets['AAPL'].rolling(5).mean(),\n    'vol_20d': rets['AAPL'].rolling(20).std(),\n})\ntable['ret_next'] = rets['AAPL'].shift(-1)\n\nclean = ...\nclean",
   "`frame.dropna()` removes any row with a missing value anywhere in it.",
   "table = pd.DataFrame({\n    'ret_1d': rets['AAPL'],\n    'ret_5d': rets['AAPL'].rolling(5).mean(),\n    'vol_20d': rets['AAPL'].rolling(20).std(),\n})\ntable['ret_next'] = rets['AAPL'].shift(-1)\n\nclean = table.dropna()\nclean",
   f"**{N_ROWS:,} rows survive.** The twenty-day window costs you the first twenty days, and the target costs you the last one.")

ex("A7", "n and p", 3,
   "Report the two numbers that describe the shape of a learning problem: `n`, the "
   "number of observations, and `p`, the number of **features**. Careful: the "
   "target is not a feature.",
   "clean = pd.DataFrame({\n    'ret_1d': rets['AAPL'],\n    'ret_5d': rets['AAPL'].rolling(5).mean(),\n    'vol_20d': rets['AAPL'].rolling(20).std(),\n    'ret_next': rets['AAPL'].shift(-1),\n}).dropna()\n\nn = ...\np = ...\nprint('n =', n)\nprint('p =', p)",
   ["`clean.shape` gives `(rows, columns)`.",
    "One of those columns is the target, so `p` is one less than the number of columns."],
   "clean = pd.DataFrame({\n    'ret_1d': rets['AAPL'],\n    'ret_5d': rets['AAPL'].rolling(5).mean(),\n    'vol_20d': rets['AAPL'].rolling(20).std(),\n    'ret_next': rets['AAPL'].shift(-1),\n}).dropna()\n\nn = clean.shape[0]\np = clean.shape[1] - 1\nprint('n =', n)\nprint('p =', p)",
   f"`n = {N_ROWS:,}` and `p = {N_COLS - 1}`. Almost every paper you read will describe its data with exactly these two numbers.")

# ================================================== B. features that cannot cheat
section(
"## \U0001f6a7 B · Features that cannot cheat\n\n"
"A feature is only allowed to contain information you would genuinely have had "
"at the moment you needed to predict. Breaking that rule is the single easiest "
"way to build a model that looks brilliant and is worthless."
)

ex("B1", "Yesterday, on today's row", 2,
   "Build `ret_prev`: the return from **the day before**, sitting on today's row.",
   "ret_prev = ...\nret_prev",
   "`shift(1)` moves values down by one row, so yesterday's number arrives on today's.",
   "ret_prev = rets['AAPL'].shift(1)\nret_prev",
   "Positive shift reaches backwards, negative shift reaches forwards. Targets use a negative shift; features almost never do.")

ex("B2", "Fix the leak", 3,
   "This table is supposed to predict tomorrow from what is known today. One of "
   "the two columns is cheating. Find it and repair it.",
   "leaky = pd.DataFrame({\n    'feature': rets['AAPL'].shift(-2),\n    'target': rets['AAPL'].shift(-1),\n})\n\nfixed = ...\nfixed",
   ["Read the shifts. `shift(-2)` puts the return from **two days ahead** on today's row.",
    "A feature has to come from today or earlier, so its shift must be zero or positive."],
   "leaky = pd.DataFrame({\n    'feature': rets['AAPL'].shift(-2),\n    'target': rets['AAPL'].shift(-1),\n})\n\nfixed = pd.DataFrame({\n    'feature': rets['AAPL'],\n    'target': rets['AAPL'].shift(-1),\n})\nfixed",
   "The broken version predicts tomorrow using the day after tomorrow. It would score beautifully and could never be used for anything.")

ex("B3", "Check it yourself", 3,
   "Prove that the target really is tomorrow's return: compare `target` on the "
   "first row with `feature` on the second row of the fixed table.",
   "tbl = pd.DataFrame({\n    'feature': rets['AAPL'],\n    'target': rets['AAPL'].shift(-1),\n}).dropna()\n\nsame = ...\nprint('target[0] equals feature[1]:', same)",
   ["Take single values out with `.iloc[0]` and `.iloc[1]`.",
    "Compare `tbl['target'].iloc[0]` with `tbl['feature'].iloc[1]`."],
   "tbl = pd.DataFrame({\n    'feature': rets['AAPL'],\n    'target': rets['AAPL'].shift(-1),\n}).dropna()\n\nsame = tbl['target'].iloc[0] == tbl['feature'].iloc[1]\nprint('target[0] equals feature[1]:', same)",
   "`True`. A one-line check like this is worth writing every time you build a target, because a shift in the wrong direction is invisible otherwise.")

ex("B4", "The same question, as a label", 2,
   "Build `up_next`: **1** when tomorrow's return is positive, **0** when it is not.",
   "up_next = ...\nup_next",
   "Compare the shifted Series to zero, then turn the True/False into numbers with `.astype(int)`.",
   "up_next = (rets['AAPL'].shift(-1) > 0).astype(int)\nup_next",
   "The same underlying data as `ret_next`, asked as a classification problem instead of a regression one.")

ex("B5", "How balanced is it", 2,
   "What share of the days in `up_next` are up days? Store it as `share_up`.",
   "up_next = (rets['AAPL'].shift(-1) > 0).astype(int).dropna()\n\nshare_up = ...\nprint('share of up days:', share_up)",
   "The mean of a column of 1s and 0s is the share of 1s.",
   "up_next = (rets['AAPL'].shift(-1) > 0).astype(int).dropna()\n\nshare_up = up_next.mean()\nprint('share of up days:', share_up)",
   f"About **{UP_SHARE:.1%}**. Remember this number: any classifier that cannot beat it is doing nothing at all.")

# ================================================================ C. missing
section(
"## \U0001f573️ C · Missing values\n\n"
"Real tables have holes in them. What you do about the holes changes your "
"answer, so it is a decision, not a formality."
)

ex("C1", "Make the gaps appear", 2,
   "Force `wide` onto **every calendar day** between its first and last date, and "
   "call the result `daily`. Print its shape.",
   "calendar = ...\ndaily = ...\ndaily",
   ["`pd.date_range(start, end, freq='D')` builds every day between two dates.",
    "`wide.index.min()` and `wide.index.max()` are the first and last dates."],
   "calendar = pd.date_range(wide.index.min(), wide.index.max(), freq='D')\ndaily = wide.reindex(calendar)\ndaily",
   f"**{N_CAL:,} rows instead of {N_TRADING:,}.** The extra rows are weekends and market holidays, and they are all empty.")

ex("C2", "Count the holes", 1,
   "How many missing values are there in `daily['AAPL']`?",
   "daily = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq='D'))\n\nn_missing = ...\nprint('missing:', n_missing)",
   "`.isna()` gives True where a value is missing, and `.sum()` counts the Trues.",
   "daily = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq='D'))\n\nn_missing = daily['AAPL'].isna().sum()\nprint('missing:', n_missing)",
   f"**{N_MISSING:,} of {N_CAL:,} days**, which is about {N_MISSING / N_CAL:.0%} of the calendar.")

ex("C3", "Column by column", 1,
   "Count the missing values in **every** column of `daily` at once.",
   "daily = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq='D'))\n\nper_column = ...\nper_column",
   "`frame.isna().sum()` counts down each column and gives you a Series.",
   "daily = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq='D'))\n\nper_column = daily.isna().sum()\nper_column",
   "Identical for every ticker, because they all trade on the same exchange calendar. Unequal counts would have told you something was wrong with one of them.")

ex("C4", "Drop them", 2,
   "Drop every row of `daily` that has a missing value, and check that you are back "
   "where you started.",
   "daily = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq='D'))\n\ndropped = ...\nn_after = ...\nprint('rows after dropping:', n_after)\nprint('rows in wide       :', len(wide))",
   "`frame.dropna()` again.",
   "daily = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq='D'))\n\ndropped = daily.dropna()\nn_after = len(dropped)\nprint('rows after dropping:', n_after)\nprint('rows in wide       :', len(wide))",
   f"Both **{N_TRADING:,}**. Here dropping is exactly right: those rows were never real observations in the first place.")

ex("C5", "What forward filling invents", 3,
   "Forward fill `daily['AAPL']`, compute daily returns from the filled series, and "
   "count how many of them are **exactly zero**.",
   "daily = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq='D'))\n\nfilled_ret = ...\nn_zero = ...\nprint('returns that are exactly zero:', n_zero)",
   ["`.ffill()` carries the last known value forward, then `.pct_change()` as usual.",
    "A return of exactly 0.0 means the price did not move, which is what a repeated price looks like. Count with `(s == 0).sum()`."],
   "daily = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq='D'))\n\nfilled_ret = daily['AAPL'].ffill().pct_change().dropna()\nn_zero = (filled_ret == 0).sum()\nprint('returns that are exactly zero:', n_zero)",
   f"**{N_ZERO_RET:,} zero returns**, one for every weekend and holiday. Forward filling is honest about the past but it invents days on which nothing happened.")

ex("C6", "And what it does to the risk number", 3,
   "Compare the standard deviation of the filled returns from C5 with the "
   "standard deviation of the real trading-day returns in `rets['AAPL']`.",
   "daily = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq='D'))\nfilled_ret = daily['AAPL'].ffill().pct_change().dropna()\n\nsd_filled = ...\nsd_real = ...\nprint('filled:', sd_filled)\nprint('real  :', sd_real)",
   "`.std()` on each of the two Series.",
   "daily = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq='D'))\nfilled_ret = daily['AAPL'].ffill().pct_change().dropna()\n\nsd_filled = filled_ret.std()\nsd_real = rets['AAPL'].std()\nprint('filled:', sd_filled)\nprint('real  :', sd_real)",
   f"**{SD_FFILL:.4f} against {SD_TRUE:.4f}.** The invented zero days drag the volatility down by about {100 * (1 - SD_FFILL / SD_TRUE):.0f}%. A cleaning step you thought was harmless has changed the answer to the question you were asking.")

ex("C7", "Record the fact", 2,
   "Sometimes the missingness itself is information. Build a table with two "
   "columns: `close`, the forward-filled price, and `was_missing`, a 1/0 column "
   "saying whether that day's price had to be invented.",
   "daily = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq='D'))\n\nflagged = ...\nflagged",
   ["The first column is `daily['AAPL'].ffill()`.",
    "The second is `daily['AAPL'].isna().astype(int)`, computed **before** the filling."],
   "daily = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq='D'))\n\nflagged = pd.DataFrame({\n    'close': daily['AAPL'].ffill(),\n    'was_missing': daily['AAPL'].isna().astype(int),\n})\nflagged",
   "Now nothing is hidden: the filled value is there to use, and the flag says it was filled. A model can even learn from the flag.")

# =============================================================== D. extremes
section(
"## \U0001f4a5 D · Extreme values\n\n"
"Finding them is arithmetic. Deciding what they are is judgement, and in finance "
"the answer is usually \"real, and the most important rows in the table\"."
)

ex("D1", "The quick look", 1,
   "Get count, mean, standard deviation, min, quartiles and max for Apple's "
   "returns in one call.",
   "summary = ...\nsummary",
   "`s.describe()` gives all of them at once.",
   "summary = actual.describe()\nsummary",
   f"The mean is close to zero and the extremes are enormous: min {A.min():.4f}, max {A.max():.4f}. That gap is the whole subject of this section.")

ex("D2", "The middle half", 2,
   "Compute `q1` and `q3`, the 25th and 75th percentiles of `actual`, and the "
   "interquartile range `iqr` between them.",
   "q1 = ...\nq3 = ...\niqr = ...\nprint('q1 :', q1)\nprint('q3 :', q3)\nprint('iqr:', iqr)",
   "`s.quantile([0.25, 0.75])` returns both at once, and you can unpack them into two names.",
   "q1 = actual.quantile(0.25)\nq3 = actual.quantile(0.75)\niqr = q3 - q1\nprint('q1 :', q1)\nprint('q3 :', q3)\nprint('iqr:', iqr)",
   f"`q1 = {Q1:.4f}`, `q3 = {Q3:.4f}`, `iqr = {IQR:.4f}`. Half of all days sit inside a band {IQR:.2%} wide.")

ex("D3", "The IQR rule", 2,
   "Flag every day outside $[\\,Q_1 - 1.5\\,\\text{IQR},\\; Q_3 + 1.5\\,\\text{IQR}\\,]$ "
   "and count them.",
   "q1, q3 = actual.quantile([0.25, 0.75])\niqr = q3 - q1\n\nflagged = ...\nn_flagged = ...\nprint('flagged:', n_flagged, 'of', len(actual))",
   ["The two fences are `q1 - 1.5 * iqr` and `q3 + 1.5 * iqr`.",
    "Combine two conditions with `|`, and give each side its own brackets: `(actual < lo) | (actual > hi)`."],
   "q1, q3 = actual.quantile([0.25, 0.75])\niqr = q3 - q1\n\nflagged = (actual < q1 - 1.5 * iqr) | (actual > q3 + 1.5 * iqr)\nn_flagged = flagged.sum()\nprint('flagged:', n_flagged, 'of', len(actual))",
   f"**{N_IQR} of {len(A):,} days, or {N_IQR / len(A):.1%}.** Far too many to be mistakes.")

ex("D4", "The other rule", 2,
   "Count the days more than **three standard deviations** from the mean.",
   "n_extreme = ...\nprint('beyond three sd:', n_extreme)",
   "`(actual - actual.mean()).abs() > 3 * actual.std()`, then `.sum()`.",
   "n_extreme = ((actual - actual.mean()).abs() > 3 * actual.std()).sum()\nprint('beyond three sd:', n_extreme)",
   f"**{N_3SD} days.**")

ex("D5", "What a bell curve would have predicted", 3,
   "A normal distribution puts about 0.27% of its observations beyond three "
   "standard deviations. How many days is that here, and how does it compare with "
   "D4?",
   "n_extreme = ((actual - actual.mean()).abs() > 3 * actual.std()).sum()\n\nexpected = ...\nprint('observed:', n_extreme)\nprint('expected:', expected)",
   "0.0027 times the number of observations.",
   "n_extreme = ((actual - actual.mean()).abs() > 3 * actual.std()).sum()\n\nexpected = 0.0027 * len(actual)\nprint('observed:', n_extreme)\nprint('expected:', round(expected, 1))",
   f"**{N_3SD} observed against {N_NORMAL:.1f} expected**, about {N_3SD / N_NORMAL:.0f} times as many. Returns have fatter tails than the bell curve, which is why both rules of thumb mislabel so much.")

ex("D6", "Every stock at once", 3,
   "Loop over the columns of `rets` and count each stock's days beyond three "
   "standard deviations. Collect the counts into a Series, sorted with the worst "
   "first.",
   "counts = {}\nfor ticker in rets.columns:\n    r = rets[ticker].dropna()\n    counts[ticker] = ...\n\nresult = ...\nresult",
   ["Inside the loop, the count is the same expression as D4 applied to `r`.",
    "`pd.Series(a_dict)` turns the dictionary into a Series, and `.sort_values(ascending=False)` orders it."],
   "counts = {}\nfor ticker in rets.columns:\n    r = rets[ticker].dropna()\n    counts[ticker] = ((r - r.mean()).abs() > 3 * r.std()).sum()\n\nresult = pd.Series(counts).sort_values(ascending=False)\nresult",
   f"**{BEYOND_S.index[0]} tops the list with {BEYOND_S.iloc[0]} days.** Every one of the eleven has several times what a bell curve allows, so this is a property of markets and not of one stock.")

# ========================================================= E. transformations
section(
"## \U0001f504 E · Transformations\n\n"
"Changing the scale of a column changes nothing about the information in it, and "
"everything about how a model reads it."
)

ex("E1", "Log returns", 2,
   "Compute Apple's **log returns** from `wide['AAPL']`, into `log_ret`.",
   "log_ret = ...\nlog_ret",
   ["A log return is the difference of the logs: $\\log P_t - \\log P_{t-1}$.",
    "`np.log(wide['AAPL'])` takes the log of every price, and `.diff()` differences it."],
   "log_ret = np.log(wide['AAPL']).diff().dropna()\nlog_ret",
   "Log returns add up over time, which simple returns do not. That one property is why most of the academic literature uses them.")

ex("E2", "How different are they, really", 3,
   "Find the **largest absolute difference** between the log returns and the "
   "simple returns in `actual`.",
   "log_ret = np.log(wide['AAPL']).diff().dropna()\n\nbiggest_gap = ...\nprint('largest difference:', biggest_gap)",
   "Subtract the two Series, take `.abs()`, then `.max()`. pandas lines them up on their dates for you.",
   "log_ret = np.log(wide['AAPL']).diff().dropna()\n\nbiggest_gap = (log_ret - actual).abs().max()\nprint('largest difference:', biggest_gap)",
   f"**{MAX_GAP:.4f}**, and that is on the worst day of the decade. On an ordinary day the two agree to three decimals, which is why people switch between them so casually.")

ex("E3", "Standardise a column", 2,
   "Standardise the 20-day volatility of Apple: subtract its mean and divide by its "
   "standard deviation. Then check the result really has mean 0 and standard "
   "deviation 1.",
   "vol20 = rets['AAPL'].rolling(20).std().dropna()\n\nz = ...\nprint('mean:', ...)\nprint('sd  :', ...)",
   "$z_i = (x_i - \\bar{x}) / s$, written straight out with `vol20.mean()` and `vol20.std()`.",
   "vol20 = rets['AAPL'].rolling(20).std().dropna()\n\nz = (vol20 - vol20.mean()) / vol20.std()\nprint('mean:', round(z.mean(), 8))\nprint('sd  :', round(z.std(), 8))",
   "Mean 0 and standard deviation 1, by construction. Nothing about the shape of the column has changed, only its units.")

ex("E4", "Standardise without cheating", 3,
   "Standardise the **whole** column using only the mean and standard deviation of "
   "the rows up to the end of 2022. Then report the mean of the standardised test "
   "rows, which will not be zero.",
   "vol20 = rets['AAPL'].rolling(20).std().dropna()\ntrain = vol20.loc[:'2022-12-31']\n\nz_all = ...\nprint('mean of standardised test rows:', ...)",
   ["The mean and standard deviation must come from `train`, but the subtraction applies to `vol20`.",
    "`(vol20 - train.mean()) / train.std()`."],
   "vol20 = rets['AAPL'].rolling(20).std().dropna()\ntrain = vol20.loc[:'2022-12-31']\n\nz_all = (vol20 - train.mean()) / train.std()\nz_test = z_all.loc['2023-01-01':]\nprint('mean of standardised test rows:', z_test.mean())",
   "Not zero, and that is the point: the test rows were standardised with numbers that knew nothing about them. Using the full-sample mean would have quietly leaked the test period into every training row.")

ex("E5", "Text into numbers", 2,
   "Turn the `sector` column of `firms` into 0/1 columns.",
   "encoded = ...\nencoded",
   "`pd.get_dummies(frame, columns=['sector'], dtype=int)`.",
   "encoded = pd.get_dummies(firms, columns=['sector'], dtype=int)\nencoded",
   "Four sectors become four columns, each holding a 1 in exactly one row. This is one-hot encoding, and it is how any category gets into a feature table.")

ex("E6", "Columns from different worlds", 2,
   "Compare the standard deviation of Apple's **volume** with the standard "
   "deviation of its **returns**, and compute how many times larger the first is.",
   "aapl_rows = prices[prices['ticker'] == 'AAPL']\n\nsd_volume = ...\nsd_return = ...\nratio = ...\nprint('volume sd:', sd_volume)\nprint('return sd:', sd_return)\nprint('ratio    :', ratio)",
   "`aapl_rows['volume'].std()` and `actual.std()`, then divide one by the other.",
   "aapl_rows = prices[prices['ticker'] == 'AAPL']\n\nsd_volume = aapl_rows['volume'].std()\nsd_return = actual.std()\nratio = sd_volume / sd_return\nprint('volume sd:', sd_volume)\nprint('return sd:', sd_return)\nprint('ratio    :', ratio)",
   f"About **{VOL_SD / RET_SD:.2e}**, which is roughly four billion. Any method that measures distances between rows would hear the volume column and nothing else.")

# ==================================================== F. scoring a number
section(
"## \U0001f4cf F · Scoring a prediction of a number\n\n"
"You are given the predictions. The job here is to turn a column of mistakes "
"into one number, four different ways, and to understand why the four disagree."
)

ex("F1", "Residuals", 1,
   "Compute the residuals of the small pair `y_true` and `y_pred` from the setup "
   "cell.",
   "residuals = ...\nresiduals",
   "A residual is actual minus predicted, element by element.",
   "residuals = y_true - y_pred\nresiduals",
   "`array([-1., 0., 1., -3.])`. Three small mistakes and one large one, which is exactly what you need to tell the metrics apart.")

ex("F2", "Write MAE", 2,
   "Write a function `mae(y, yhat)` that returns the mean absolute error, then run "
   "it on `y_true` and `y_pred`.",
   "def mae(y, yhat):\n    ...\n\nresult = ...\nprint('MAE:', result)",
   ["$\\text{MAE} = \\frac{1}{n}\\sum_i |y_i - \\hat{y}_i|$.",
    "`np.abs(y - yhat).mean()` does the whole thing in one expression."],
   "def mae(y, yhat):\n    return np.abs(y - yhat).mean()\n\nresult = mae(y_true, y_pred)\nprint('MAE:', result)",
   "**1.25.** The four mistakes were 1, 0, 1 and 3, and their average size is 1.25.")

ex("F3", "Write MSE", 2,
   "Write `mse(y, yhat)` and run it on the same pair.",
   "def mse(y, yhat):\n    ...\n\nresult = ...\nprint('MSE:', result)",
   "$\\text{MSE} = \\frac{1}{n}\\sum_i (y_i - \\hat{y}_i)^2$. Square before you average.",
   "def mse(y, yhat):\n    return ((y - yhat) ** 2).mean()\n\nresult = mse(y_true, y_pred)\nprint('MSE:', result)",
   "**2.75.** The single mistake of 3 contributes 9 of the total 11, which is why one bad day dominates this metric.")

ex("F4", "Write RMSE", 1,
   "Write `rmse(y, yhat)`, using the `mse` function you just wrote.",
   "def mse(y, yhat):\n    return ((y - yhat) ** 2).mean()\n\ndef rmse(y, yhat):\n    ...\n\nresult = ...\nprint('RMSE:', result)",
   "The square root of the mean squared error. `np.sqrt(mse(y, yhat))`.",
   "def mse(y, yhat):\n    return ((y - yhat) ** 2).mean()\n\ndef rmse(y, yhat):\n    return np.sqrt(mse(y, yhat))\n\nresult = rmse(y_true, y_pred)\nprint('RMSE:', result)",
   "**1.658.** Larger than the MAE of 1.25, and the gap between them is a signal that the errors are uneven.")

ex("F5", "Write R squared", 3,
   "Write `r2(y, yhat)`: one minus the sum of squared residuals divided by the sum "
   "of squared deviations from the mean of `y`.",
   "def r2(y, yhat):\n    ...\n\nresult = ...\nprint('R2:', result)",
   ["$R^2 = 1 - \\dfrac{\\sum_i (y_i-\\hat{y}_i)^2}{\\sum_i (y_i-\\bar{y})^2}$.",
    "Build the two sums separately first: `rss = ((y - yhat) ** 2).sum()` and `tss = ((y - y.mean()) ** 2).sum()`."],
   "def r2(y, yhat):\n    rss = ((y - yhat) ** 2).sum()\n    tss = ((y - y.mean()) ** 2).sum()\n    return 1 - rss / tss\n\nresult = r2(y_true, y_pred)\nprint('R2:', result)",
   "**0.45.** The predictions beat the average of `y_true`, but not by a great deal.")

ex("F6", "All four, on the real thing", 2,
   "The Session 4 rule is $\\hat{y} = 0.00043 + 1.2073\\,x$, where $x$ is the "
   "market's return. Apply it to `market` and score it against `actual` with all "
   "four metrics.",
   "predicted = ...\nresid = ...\n\nprint('MAE :', ...)\nprint('MSE :', ...)\nprint('RMSE:', ...)\nprint('R2  :', ...)",
   ["The prediction is `0.00043 + 1.2073 * market`, applied to the whole Series at once.",
    "Reuse the expressions from F2 to F5: `resid.abs().mean()`, `(resid ** 2).mean()`, and so on."],
   "predicted = 0.00043 + 1.2073 * market\nresid = actual - predicted\n\nprint('MAE :', resid.abs().mean())\nprint('MSE :', (resid ** 2).mean())\nprint('RMSE:', np.sqrt((resid ** 2).mean()))\nprint('R2  :', 1 - (resid ** 2).sum() / ((actual - actual.mean()) ** 2).sum())",
   f"MAE **{MAE:.5f}** ({MAE:.2%} on a typical day), RMSE **{RMSE:.5f}**, R² **{R2:.4f}**. The market explains a bit over half of Apple's daily movement.")

ex("F7", "The baseline everybody forgets", 3,
   "Score the laziest possible model: predict the **mean of `actual`** on every "
   "single day. Report its RMSE and its R².",
   "baseline = ...\n\nprint('RMSE:', ...)\nprint('R2  :', ...)",
   ["`np.full(len(actual), actual.mean())` builds an array of the same value repeated.",
    "Work out what R² must be before you run it: the model *is* the mean."],
   "baseline = np.full(len(actual), actual.mean())\nresid = actual - baseline\n\nprint('RMSE:', np.sqrt((resid ** 2).mean()))\nprint('R2  :', 1 - (resid ** 2).sum() / ((actual - actual.mean()) ** 2).sum())",
   f"RMSE **{A.std(ddof=0):.5f}**, and R² **exactly 0**. That is what R² measures: how far you have moved from this. Any model with a negative R² is worse than this one.")

ex("F8", "One catastrophic day", 3,
   "Take the predictions from F6, replace the 30th-from-last one with a forecast of "
   "`0.40`, and see what each metric does. Report the ratio of the damaged metric "
   "to the clean one.",
   "predicted = 0.00043 + 1.2073 * market\nbroken = predicted.copy()\nbroken.iloc[-30] = 0.40\n\nmae_ratio = ...\nrmse_ratio = ...\nprint('MAE  multiplied by:', mae_ratio)\nprint('RMSE multiplied by:', rmse_ratio)",
   ["Compute each metric twice: once with `predicted`, once with `broken`.",
    "The ratio is the damaged value divided by the clean value."],
   "predicted = 0.00043 + 1.2073 * market\nbroken = predicted.copy()\nbroken.iloc[-30] = 0.40\n\nclean_resid = actual - predicted\nbroken_resid = actual - broken\n\nmae_ratio = broken_resid.abs().mean() / clean_resid.abs().mean()\nrmse_ratio = np.sqrt((broken_resid ** 2).mean()) / np.sqrt((clean_resid ** 2).mean())\nprint('MAE  multiplied by:', mae_ratio)\nprint('RMSE multiplied by:', rmse_ratio)",
   f"MAE grows by a factor of **{MAE_BAD / MAE:.2f}**, RMSE by **{RMSE_BAD / RMSE:.2f}**. One day in {len(A):,} moves the average mistake by 2% and the root mean squared one by ten times as much.")

ex("F9", "Pick the better rule", 3,
   "Two candidate rules: rule A is $0.00043 + 1.2073\\,x$, rule B is simply "
   "$1.0\\,x$. Score both with RMSE and print the winner.",
   "rule_a = ...\nrule_b = ...\n\nrmse_a = ...\nrmse_b = ...\nprint('rule A RMSE:', rmse_a)\nprint('rule B RMSE:', rmse_b)\nprint('better:', 'A' if ... else 'B')",
   ["Rule B is just `1.0 * market`.",
    "The lower RMSE wins, so the condition is `rmse_a < rmse_b`."],
   "rule_a = 0.00043 + 1.2073 * market\nrule_b = 1.0 * market\n\nrmse_a = np.sqrt(((actual - rule_a) ** 2).mean())\nrmse_b = np.sqrt(((actual - rule_b) ** 2).mean())\nprint('rule A RMSE:', rmse_a)\nprint('rule B RMSE:', rmse_b)\nprint('better:', 'A' if rmse_a < rmse_b else 'B')",
   f"**A wins**, {RMSE:.5f} against {_rmse(ACTUAL, MARKET):.5f}. Apple moves about 1.2 times as much as the market, so a rule that assumes 1.0 systematically under-predicts.")

# ================================================== G. scoring a label
section(
"## \U0001f3af G · Scoring a prediction of a label\n\n"
"No residuals now. Everything is built out of four counts, and the whole art is "
"knowing which of the four you care about."
)

ex("G1", "Two boolean columns", 1,
   "Build `went_up` (Apple really rose) and `said_up` (the market rose, which is "
   "what our rule predicts with).",
   "went_up = ...\nsaid_up = ...\nsaid_up",
   "Each one is a comparison to zero on one of the two Series from the setup cell.",
   "went_up = actual > 0\nsaid_up = market > 0\npd.DataFrame({'said_up': said_up, 'went_up': went_up}).tail(4)",
   "Two columns of True and False. Everything in this section is counting the four ways those two columns can line up.")

ex("G2", "True positives", 2,
   "Count the days where the rule said up **and** the stock went up.",
   "went_up = actual > 0\nsaid_up = market > 0\n\ntp = ...\nprint('true positives:', tp)",
   "Combine the two with `&`, giving each side brackets, then `.sum()`.",
   "went_up = actual > 0\nsaid_up = market > 0\n\ntp = (said_up & went_up).sum()\nprint('true positives:', tp)",
   f"**{TP:,} days.**")

ex("G3", "The other three", 2,
   "Count the false positives, false negatives and true negatives.",
   "went_up = actual > 0\nsaid_up = market > 0\n\nfp = ...\nfn = ...\ntn = ...\nprint('FP:', fp, ' FN:', fn, ' TN:', tn)",
   ["`~` flips a boolean Series: `~said_up` is every day the rule said down.",
    "A false positive is `said_up & ~went_up`; a false negative is `~said_up & went_up`."],
   "went_up = actual > 0\nsaid_up = market > 0\n\nfp = (said_up & ~went_up).sum()\nfn = (~said_up & went_up).sum()\ntn = (~said_up & ~went_up).sum()\nprint('FP:', fp, ' FN:', fn, ' TN:', tn)",
   f"FP **{FP}**, FN **{FN}**, TN **{TN:,}**. Those four numbers add up to {TP + FP + FN + TN:,}, every day in the sample.")

ex("G4", "The confusion matrix as a table", 3,
   "Arrange the four counts into a 2 by 2 DataFrame with readable labels.",
   "went_up = actual > 0\nsaid_up = market > 0\ntp = (said_up & went_up).sum()\nfp = (said_up & ~went_up).sum()\nfn = (~said_up & went_up).sum()\ntn = (~said_up & ~went_up).sum()\n\nmatrix = ...\nmatrix",
   ["`pd.DataFrame(...)` accepts a dictionary of columns, and `index=` names the rows.",
    "Columns are what the model said, rows are what happened: `{'said up': [tp, fp], 'said down': [fn, tn]}` with `index=['went up', 'went down']`."],
   "went_up = actual > 0\nsaid_up = market > 0\ntp = (said_up & went_up).sum()\nfp = (said_up & ~went_up).sum()\nfn = (~said_up & went_up).sum()\ntn = (~said_up & ~went_up).sum()\n\nmatrix = pd.DataFrame(\n    {'said up': [tp, fp], 'said down': [fn, tn]},\n    index=['went up', 'went down'],\n)\nmatrix",
   "The diagonal is what the rule got right. Getting the layout the right way round matters: a transposed confusion matrix silently swaps precision and recall.")

ex("G5", "Accuracy", 2,
   "Write `accuracy(tp, fp, fn, tn)` and run it on the four counts.",
   "def accuracy(tp, fp, fn, tn):\n    ...\n\nwent_up = actual > 0\nsaid_up = market > 0\nresult = ...\nprint('accuracy:', result)",
   "The share of all days that were called correctly: $(TP+TN)/(TP+TN+FP+FN)$.",
   "def accuracy(tp, fp, fn, tn):\n    return (tp + tn) / (tp + fp + fn + tn)\n\nwent_up = actual > 0\nsaid_up = market > 0\nresult = accuracy(\n    (said_up & went_up).sum(), (said_up & ~went_up).sum(),\n    (~said_up & went_up).sum(), (~said_up & ~went_up).sum(),\n)\nprint('accuracy:', result)",
   f"**{ACC:.4f}**, so three days in four. Section G8 shows why that number can be worthless.")

ex("G6", "Precision and recall", 2,
   "Write both, and run them on the same counts.",
   "def precision(tp, fp):\n    ...\n\ndef recall(tp, fn):\n    ...\n\nwent_up = actual > 0\nsaid_up = market > 0\ntp = (said_up & went_up).sum()\nfp = (said_up & ~went_up).sum()\nfn = (~said_up & went_up).sum()\nprint('precision:', ...)\nprint('recall   :', ...)",
   ["Precision is $TP/(TP+FP)$: of the days I called up, how many were.",
    "Recall is $TP/(TP+FN)$: of the days that were up, how many did I catch."],
   "def precision(tp, fp):\n    return tp / (tp + fp)\n\ndef recall(tp, fn):\n    return tp / (tp + fn)\n\nwent_up = actual > 0\nsaid_up = market > 0\ntp = (said_up & went_up).sum()\nfp = (said_up & ~went_up).sum()\nfn = (~said_up & went_up).sum()\nprint('precision:', precision(tp, fp))\nprint('recall   :', recall(tp, fn))",
   f"Precision **{PREC:.4f}**, recall **{REC:.4f}**. Notice that neither of them uses the true negatives at all.")

ex("G7", "F1", 2,
   "Write `f1(precision, recall)`, the harmonic mean of the two, and run it on the "
   "values from G6.",
   "def f1(p, r):\n    ...\n\nresult = f1(0.7589, 0.7782)\nprint('F1:', result)",
   "$F_1 = 2\\,\\dfrac{p \\cdot r}{p + r}$.",
   "def f1(p, r):\n    return 2 * p * r / (p + r)\n\nresult = f1(0.7589, 0.7782)\nprint('F1:', result)",
   f"**{F1:.4f}**, sitting between the two as it must. Try `f1(1.0, 0.01)`: the answer is 0.02, not 0.5, which is exactly the point of using a harmonic mean.")

ex("G8", "When accuracy lies", 3,
   "Now predict something rare: a day when Apple falls more than 5%. Score the "
   "model that **always says no**, on accuracy and on recall.",
   "crash = actual < -0.05\nnever_warns = ...\n\nprint('crash days :', crash.sum())\nprint('accuracy   :', ...)\nprint('recall     :', ...)",
   ["`never_warns` predicts False on every single day. `crash & False` is never True.",
    "Its accuracy is the share of days that are not crashes; its recall is 0 true positives divided by all the real crashes."],
   "crash = actual < -0.05\nnever_warns = pd.Series(False, index=actual.index)\n\ntp = (never_warns & crash).sum()\ntn = (~never_warns & ~crash).sum()\nprint('crash days :', crash.sum())\nprint('accuracy   :', (tp + tn) / len(actual))\nprint('recall     :', tp / crash.sum())",
   f"**{ACC_LAZY:.2%} accurate, and it catches none of the {N_CRASH} crashes.** A model with no skill of any kind beats almost anything you could build, measured by accuracy alone.")

ex("G9", "Move the threshold", 3,
   "Instead of \"the market rose\", demand that the market rose **more than 1%** "
   "before calling a day up. Compute precision and recall for the stricter rule.",
   "went_up = actual > 0\nstrict = ...\n\ntp = ...\nfp = ...\nfn = ...\nprint('precision:', ...)\nprint('recall   :', ...)",
   ["The stricter rule is `market > 0.01`.",
    "The three counts are built exactly as in G3, using `strict` in place of `said_up`."],
   "went_up = actual > 0\nstrict = market > 0.01\n\ntp = (strict & went_up).sum()\nfp = (strict & ~went_up).sum()\nfn = (~strict & went_up).sum()\nprint('precision:', tp / (tp + fp))\nprint('recall   :', tp / (tp + fn))",
   f"Precision rises to **{PREC_HI:.3f}** and recall falls to **{REC_HI:.3f}**. Raising the bar makes you righter about the days you call, and blind to most of the days you do not.")

# ============================================================= H. train and test
section(
"## ✂️ H · Training rows and test rows\n\n"
"A score computed on the rows a model learned from is not evidence. These are "
"the mechanics of keeping the two apart, and of noticing when you have failed to."
)

ex("H1", "Split by date", 1,
   "Split the Apple feature table into `train` (up to the end of 2022) and `test` "
   "(2023 onwards), and print the size of each.",
   "table = pd.DataFrame({\n    'ret_1d': rets['AAPL'],\n    'ret_next': rets['AAPL'].shift(-1),\n}).dropna()\n\ntrain = ...\ntest = ...\nprint('train rows:', ...)\nprint('test rows :', ...)",
   "`table.loc[:'2022-12-31']` and `table.loc['2023-01-01':]`.",
   "table = pd.DataFrame({\n    'ret_1d': rets['AAPL'],\n    'ret_next': rets['AAPL'].shift(-1),\n}).dropna()\n\ntrain = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\nprint('train rows:', len(train))\nprint('test rows :', len(test))",
   "Eight years to learn from and two to be judged on. The split is a date, not a fraction of the rows chosen at random.")

ex("H2", "Prove they do not overlap", 2,
   "Show that the last date in `train` is earlier than the first date in `test`.",
   "table = pd.DataFrame({'ret_1d': rets['AAPL'], 'ret_next': rets['AAPL'].shift(-1)}).dropna()\ntrain = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\n\nno_overlap = ...\nprint('last train date:', train.index.max().date())\nprint('first test date:', test.index.min().date())\nprint('clean split    :', no_overlap)",
   "Compare `train.index.max()` with `test.index.min()`.",
   "table = pd.DataFrame({'ret_1d': rets['AAPL'], 'ret_next': rets['AAPL'].shift(-1)}).dropna()\ntrain = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\n\nno_overlap = train.index.max() < test.index.min()\nprint('last train date:', train.index.max().date())\nprint('first test date:', test.index.min().date())\nprint('clean split    :', no_overlap)",
   "`True`. Worth checking every time, because an off-by-one in a date filter is silent and flattering.")

ex("H3", "Score on both sides", 2,
   "Score the naive rule \"tomorrow will repeat today\" with RMSE, once on the "
   "training rows and once on the test rows.",
   "table = pd.DataFrame({'ret_1d': rets['AAPL'], 'ret_next': rets['AAPL'].shift(-1)}).dropna()\ntrain = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\n\nrmse_train = ...\nrmse_test = ...\nprint('train RMSE:', rmse_train)\nprint('test  RMSE:', rmse_test)",
   ["The prediction is the `ret_1d` column and the truth is the `ret_next` column.",
    "`np.sqrt(((frame['ret_next'] - frame['ret_1d']) ** 2).mean())` for each half."],
   "table = pd.DataFrame({'ret_1d': rets['AAPL'], 'ret_next': rets['AAPL'].shift(-1)}).dropna()\ntrain = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\n\nrmse_train = np.sqrt(((train['ret_next'] - train['ret_1d']) ** 2).mean())\nrmse_test = np.sqrt(((test['ret_next'] - test['ret_1d']) ** 2).mean())\nprint('train RMSE:', rmse_train)\nprint('test  RMSE:', rmse_test)",
   f"**{RMSE_TRAIN:.5f} and {RMSE_TEST:.5f}.** This rule has no parameters to fit, so it cannot overfit, and the two numbers differ only because the two periods differ.")

ex("H4", "Fix the leaking scaler", 3,
   "This code standardises before splitting, so the training rows have already been "
   "told about the test period. Repair it so the mean and standard deviation come "
   "from the training rows only.",
   "vol20 = rets['AAPL'].rolling(20).std().dropna()\n\n# leaky: statistics computed on everything\nz_leaky = (vol20 - vol20.mean()) / vol20.std()\n\nz_fixed = ...\nprint('leaky test mean:', z_leaky.loc['2023-01-01':].mean())\nprint('fixed test mean:', ...)",
   ["Cut the training slice out first: `train = vol20.loc[:'2022-12-31']`.",
    "Then subtract `train.mean()` and divide by `train.std()`, applying it to the whole column."],
   "vol20 = rets['AAPL'].rolling(20).std().dropna()\n\n# leaky: statistics computed on everything\nz_leaky = (vol20 - vol20.mean()) / vol20.std()\n\ntrain = vol20.loc[:'2022-12-31']\nz_fixed = (vol20 - train.mean()) / train.std()\nprint('leaky test mean:', z_leaky.loc['2023-01-01':].mean())\nprint('fixed test mean:', z_fixed.loc['2023-01-01':].mean())",
   "The two answers differ, which is the proof that the leak was doing something. Every cleaning number, not just this one, has to be computed on the training rows.")

ex("H5", "Fix the shuffled split", 3,
   "This code splits a time series at random, so the model would train on Friday to "
   "predict Thursday. Replace it with a chronological split that keeps the same "
   "number of test rows.",
   "table = pd.DataFrame({'ret_1d': rets['AAPL'], 'ret_next': rets['AAPL'].shift(-1)}).dropna()\nn_test = 500\n\n# wrong: a random quarter of the days\nshuffled_test = table.sample(n_test, random_state=0)\n\nproper_test = ...\nproper_train = ...\nprint('random split, first test date:', shuffled_test.index.min().date())\nprint('proper split, first test date:', ...)",
   ["The last `n_test` rows are the future. `table.iloc[-n_test:]` takes them.",
    "The training rows are everything before that: `table.iloc[:-n_test]`."],
   "table = pd.DataFrame({'ret_1d': rets['AAPL'], 'ret_next': rets['AAPL'].shift(-1)}).dropna()\nn_test = 500\n\n# wrong: a random quarter of the days\nshuffled_test = table.sample(n_test, random_state=0)\n\nproper_test = table.iloc[-n_test:]\nproper_train = table.iloc[:-n_test]\nprint('random split, first test date:', shuffled_test.index.min().date())\nprint('proper split, first test date:', proper_test.index.min().date())",
   "The random split's test set starts in 2015 and is scattered through the whole sample, so the model would be trained on days that come after the days it is tested on.")

ex("H6", "How much did you hold back", 2,
   "Compute the share of rows in the test set, as a percentage, for the 2023 split "
   "from H1.",
   "table = pd.DataFrame({'ret_1d': rets['AAPL'], 'ret_next': rets['AAPL'].shift(-1)}).dropna()\ntrain = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\n\nshare = ...\nprint('test share:', share)",
   "Rows in the test set divided by rows in both, times 100.",
   "table = pd.DataFrame({'ret_1d': rets['AAPL'], 'ret_next': rets['AAPL'].shift(-1)}).dropna()\ntrain = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\n\nshare = 100 * len(test) / (len(train) + len(test))\nprint('test share:', share)",
   "About **20%**, which is a common choice. Too small and the score is noisy; too large and the model has little to learn from.")

ex("H7", "A function you will reuse", 3,
   "Write `split_by_date(frame, cutoff)` that returns the training frame and the "
   "test frame, and use it on the Apple table.",
   "def split_by_date(frame, cutoff):\n    ...\n\ntable = pd.DataFrame({'ret_1d': rets['AAPL'], 'ret_next': rets['AAPL'].shift(-1)}).dropna()\ntrain, test = ..., ...\nprint('train rows:', ...)\nprint('test rows :', ...)",
   ["A function can return two things at once: `return a, b`.",
    "Inside, slice with `frame.loc[:cutoff]` and `frame.loc[cutoff:]`, but make the second one start the day after so no row appears twice."],
   "def split_by_date(frame, cutoff):\n    train = frame.loc[:cutoff]\n    test = frame.loc[frame.index > cutoff]\n    return train, test\n\ntable = pd.DataFrame({'ret_1d': rets['AAPL'], 'ret_next': rets['AAPL'].shift(-1)}).dropna()\ntrain, test = split_by_date(table, '2022-12-31')\nprint('train rows:', len(train))\nprint('test rows :', len(test))",
   "Using `frame.index > cutoff` for the second half guarantees the cutoff day itself cannot land in both. Small detail, and it is exactly the kind of thing that silently inflates a score.")

# ================================================================ I. flexibility
section(
"## \U0001f39a️ I · Flexibility, and too much of it\n\n"
"This is the one section where something is fitted, and only so that you can "
"watch it go wrong. The data is invented, so the truth is known: "
"$y=\\sin(2\\pi x)$ plus noise."
)

ex("I1", "Two very different curves", 2,
   "Fit a degree-1 and a degree-15 polynomial to `x_train` and `y_train`, and "
   "report the **training** MSE of each.",
   "coef_1 = ...\ncoef_15 = ...\n\nmse_1 = ...\nmse_15 = ...\nprint('degree  1 train MSE:', mse_1)\nprint('degree 15 train MSE:', mse_15)",
   ["`np.polyfit(x, y, degree)` returns the coefficients.",
    "`np.polyval(coef, x_train)` gives the fitted values, and the MSE is the mean squared difference from `y_train`."],
   "coef_1 = np.polyfit(x_train, y_train, 1)\ncoef_15 = np.polyfit(x_train, y_train, 15)\n\nmse_1 = ((np.polyval(coef_1, x_train) - y_train) ** 2).mean()\nmse_15 = ((np.polyval(coef_15, x_train) - y_train) ** 2).mean()\nprint('degree  1 train MSE:', mse_1)\nprint('degree 15 train MSE:', mse_15)",
   f"**{TR1:.3f} against {TR15:.3f}.** On the data it was fitted to, the flexible curve is far better. If you stopped here you would ship it.")

ex("I2", "Now on data they have not seen", 2,
   "Score the same two curves on `x_test` and `y_test`.",
   "coef_1 = np.polyfit(x_train, y_train, 1)\ncoef_15 = np.polyfit(x_train, y_train, 15)\n\ntest_1 = ...\ntest_15 = ...\nprint('degree  1 test MSE:', test_1)\nprint('degree 15 test MSE:', test_15)",
   "Same expression as I1, with `x_test` and `y_test` in place of the training arrays.",
   "coef_1 = np.polyfit(x_train, y_train, 1)\ncoef_15 = np.polyfit(x_train, y_train, 15)\n\ntest_1 = ((np.polyval(coef_1, x_test) - y_test) ** 2).mean()\ntest_15 = ((np.polyval(coef_15, x_test) - y_test) ** 2).mean()\nprint('degree  1 test MSE:', test_1)\nprint('degree 15 test MSE:', test_15)",
   f"**{TE1:.3f} against {TE15:.3f}.** The flexible curve was thirty times better on its own data and is now several times worse. That reversal is overfitting, measured.")

ex("I3", "The whole curve", 3,
   "Loop over degrees 1 to 12, collecting the training and test MSE of each into a "
   "DataFrame with one row per degree.",
   "rows = []\nfor degree in range(1, 13):\n    coef = ...\n    rows.append({\n        'degree': degree,\n        'train': ...,\n        'test': ...,\n    })\n\ncurve = pd.DataFrame(rows).set_index('degree')\ncurve",
   ["Inside the loop, fit with `np.polyfit(x_train, y_train, degree)`.",
    "`pd.DataFrame(list_of_dicts)` builds a table with one row per dictionary, exactly as in Session 3."],
   "rows = []\nfor degree in range(1, 13):\n    coef = np.polyfit(x_train, y_train, degree)\n    rows.append({\n        'degree': degree,\n        'train': ((np.polyval(coef, x_train) - y_train) ** 2).mean(),\n        'test': ((np.polyval(coef, x_test) - y_test) ** 2).mean(),\n    })\n\ncurve = pd.DataFrame(rows).set_index('degree')\ncurve",
   "Read down the two columns: `train` falls all the way, `test` falls, flattens, and then turns back up. That is the U from the lecture, in numbers.")

ex("I4", "Where the bottom is", 3,
   "From that table, find the degree with the **lowest test MSE**.",
   "rows = []\nfor degree in range(1, 13):\n    coef = np.polyfit(x_train, y_train, degree)\n    rows.append({'degree': degree,\n                 'train': ((np.polyval(coef, x_train) - y_train) ** 2).mean(),\n                 'test': ((np.polyval(coef, x_test) - y_test) ** 2).mean()})\ncurve = pd.DataFrame(rows).set_index('degree')\n\nbest = ...\nprint('best degree:', best)",
   "`curve['test'].idxmin()` gives the index label where the column is smallest.",
   "rows = []\nfor degree in range(1, 13):\n    coef = np.polyfit(x_train, y_train, degree)\n    rows.append({'degree': degree,\n                 'train': ((np.polyval(coef, x_train) - y_train) ** 2).mean(),\n                 'test': ((np.polyval(coef, x_test) - y_test) ** 2).mean()})\ncurve = pd.DataFrame(rows).set_index('degree')\n\nbest = curve['test'].idxmin()\nprint('best degree:', best)",
   f"**Degree {BEST_DEG}.** The true function is a sine, so a low-order polynomial can bend enough to follow it, and everything beyond that is fitting noise.")

ex("I5", "One thing that is always true", 4,
   "Show that the **training** error never increases as the degree grows: compare "
   "each row of the `train` column with the one before it, and count how many times "
   "it rose.",
   "rows = []\nfor degree in range(1, 13):\n    coef = np.polyfit(x_train, y_train, degree)\n    rows.append({'degree': degree,\n                 'train': ((np.polyval(coef, x_train) - y_train) ** 2).mean()})\ncurve = pd.DataFrame(rows).set_index('degree')\n\nrises = ...\nprint('times the training error rose:', rises)",
   ["`s.diff()` gives the change from each row to the next.",
    "Count the positive changes: `(curve['train'].diff() > 0).sum()`. Allow a tiny tolerance if you like, since these are floating point numbers."],
   "rows = []\nfor degree in range(1, 13):\n    coef = np.polyfit(x_train, y_train, degree)\n    rows.append({'degree': degree,\n                 'train': ((np.polyval(coef, x_train) - y_train) ** 2).mean()})\ncurve = pd.DataFrame(rows).set_index('degree')\n\nrises = (curve['train'].diff() > 1e-12).sum()\nprint('times the training error rose:', rises)",
   "**Zero.** A degree-8 polynomial can do everything a degree-7 one can, so it can never fit the training data worse. That is precisely why training error cannot be used to choose a model.")

# ========================================================== J. putting it together
section(
"## \U0001f3d7️ J · Putting it together\n\n"
"One short end-to-end pass, with every step of today in it."
)

ex("J1", "Build the table", 3,
   "Build a feature table for **Microsoft**: `ret_1d`, `ret_5d`, `vol_20d`, and the "
   "target `ret_next`. Drop the incomplete rows and report the shape.",
   "msft = ...\nmsft",
   "Exactly the construction from section A, with `'MSFT'` in place of `'AAPL'`.",
   "msft = pd.DataFrame({\n    'ret_1d': rets['MSFT'],\n    'ret_5d': rets['MSFT'].rolling(5).mean(),\n    'vol_20d': rets['MSFT'].rolling(20).std(),\n    'ret_next': rets['MSFT'].shift(-1),\n}).dropna()\nmsft",
   f"**{MTBL.shape[0]:,} rows and {MTBL.shape[1]} columns**, so three features and a target.")

ex("J2", "Split it, then standardise it properly", 3,
   "Split at the end of 2022, then standardise `vol_20d` across the whole table "
   "using the **training** mean and standard deviation only.",
   "msft = pd.DataFrame({\n    'ret_1d': rets['MSFT'],\n    'ret_5d': rets['MSFT'].rolling(5).mean(),\n    'vol_20d': rets['MSFT'].rolling(20).std(),\n    'ret_next': rets['MSFT'].shift(-1),\n}).dropna()\n\ntrain = ...\nvol_z = ...\nvol_z",
   ["`train = msft.loc[:'2022-12-31']`.",
    "`(msft['vol_20d'] - train['vol_20d'].mean()) / train['vol_20d'].std()`."],
   "msft = pd.DataFrame({\n    'ret_1d': rets['MSFT'],\n    'ret_5d': rets['MSFT'].rolling(5).mean(),\n    'vol_20d': rets['MSFT'].rolling(20).std(),\n    'ret_next': rets['MSFT'].shift(-1),\n}).dropna()\n\ntrain = msft.loc[:'2022-12-31']\nvol_z = (msft['vol_20d'] - train['vol_20d'].mean()) / train['vol_20d'].std()\nvol_z",
   "The order matters: split first, then compute the statistics, then apply them everywhere. Doing it the other way round is the most common leak there is.")

ex("J3", "Score a naive rule on the test rows", 3,
   "On the **test** rows only, score the rule \"tomorrow repeats today\" with MAE, "
   "RMSE and R². Use the **training** mean as the baseline inside R².",
   "msft = pd.DataFrame({\n    'ret_1d': rets['MSFT'],\n    'ret_next': rets['MSFT'].shift(-1),\n}).dropna()\ntrain = msft.loc[:'2022-12-31']\ntest = msft.loc['2023-01-01':]\n\nresid = ...\nprint('MAE :', ...)\nprint('RMSE:', ...)\nprint('R2  :', ...)",
   ["The prediction on the test rows is `test['ret_1d']`, and the truth is `test['ret_next']`.",
    "For R², the total sum of squares uses the training mean: `((test['ret_next'] - train['ret_next'].mean()) ** 2).sum()`."],
   "msft = pd.DataFrame({\n    'ret_1d': rets['MSFT'],\n    'ret_next': rets['MSFT'].shift(-1),\n}).dropna()\ntrain = msft.loc[:'2022-12-31']\ntest = msft.loc['2023-01-01':]\n\nresid = test['ret_next'] - test['ret_1d']\nrss = (resid ** 2).sum()\ntss = ((test['ret_next'] - train['ret_next'].mean()) ** 2).sum()\nprint('MAE :', resid.abs().mean())\nprint('RMSE:', np.sqrt((resid ** 2).mean()))\nprint('R2  :', 1 - rss / tss)",
   f"RMSE **{NAIVE_RMSE:.5f}** and R² **{NAIVE_R2:.4f}**. A negative R² means this rule is worse than predicting the training average every single day.")

ex("J4", "Beat it with nothing at all", 4,
   "Score the baseline that predicts the **training mean** on every test row, and "
   "print which of the two rules has the lower RMSE.",
   "msft = pd.DataFrame({'ret_1d': rets['MSFT'], 'ret_next': rets['MSFT'].shift(-1)}).dropna()\ntrain = msft.loc[:'2022-12-31']\ntest = msft.loc['2023-01-01':]\n\nbaseline = ...\nrmse_naive = ...\nrmse_base = ...\nprint('naive rule RMSE:', rmse_naive)\nprint('baseline   RMSE:', rmse_base)\nprint('winner:', 'naive rule' if ... else 'baseline')",
   ["`np.full(len(test), train['ret_next'].mean())` repeats one number for every test row.",
    "The winner is whichever RMSE is smaller."],
   "msft = pd.DataFrame({'ret_1d': rets['MSFT'], 'ret_next': rets['MSFT'].shift(-1)}).dropna()\ntrain = msft.loc[:'2022-12-31']\ntest = msft.loc['2023-01-01':]\n\nbaseline = np.full(len(test), train['ret_next'].mean())\nrmse_naive = np.sqrt(((test['ret_next'] - test['ret_1d']) ** 2).mean())\nrmse_base = np.sqrt(((test['ret_next'] - baseline) ** 2).mean())\nprint('naive rule RMSE:', rmse_naive)\nprint('baseline   RMSE:', rmse_base)\nprint('winner:', 'naive rule' if rmse_naive < rmse_base else 'baseline')",
   f"**The baseline wins**, {BASE_RMSE:.5f} against {NAIVE_RMSE:.5f}. Doing nothing beats a plausible-sounding rule, which is the honest starting point for the rest of this course: you must beat the boring thing before you have anything at all.")

# ---------------------------------------------------------------- closing
md(
"## \U0001f3c1 Done\n\n"
"You have built a feature table that cannot see the future, broken it and "
"repaired it, written every metric in the session out by hand, kept a test set "
"honest, and measured overfitting rather than taking it on trust.\n\n"
"That is the entire workflow that sits underneath every model in the rest of "
"the course. What changes from here is only the middle step: instead of a rule "
"somebody handed you, an algorithm will choose one.\n\n"
"**Next:** open `session_04_case.ipynb` (*The Analyst's Notebook, Part 4*) and "
"turn the risk report into a prediction problem.\n\n"
"*Stuck for more than 15 minutes on anything? Ask a friend, ask an AI for a hint "
"(not the answer), or email me at `jobo@econ.au.dk`.*"
)

# ---------------------------------------------------------------- write
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
# Deterministic cell ids: nbformat assigns a fresh random uuid to every cell,
# which would make every regeneration look like a rewrite in git.
for _i, _c in enumerate(nb.cells):
    _c["id"] = f"c{_i:04d}"

OUT.write_text(nbf.writes(nb), encoding="utf-8")

n_ex = sum(1 for c in cells if c.cell_type == "markdown" and c.source.startswith("### "))
tiers = {}
for c in cells:
    if c.cell_type == "markdown" and c.source.startswith("### "):
        for k, lab in LABELS.items():
            if lab in c.source.splitlines()[0]:
                tiers[lab] = tiers.get(lab, 0) + 1
print(f"wrote {OUT}  ({len(cells)} cells, {n_ex} exercises)")
print("difficulty:", tiers)
