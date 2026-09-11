# -*- coding: utf-8 -*-
"""Build session_06_case.ipynb  (The Analyst's Notebook, Part 6).

Conventions (approved for Sessions 1 to 5): no star badges, one cumulative
investigation where later questions reuse what earlier ones stored, folded
hints and solutions, stated formulas, plain explanatory tone, no em-dashes.
Opens with a QUICK LOAD restoring Part 5's findings.

Part 5 ended with a one-column model chosen by cross-validation and beating the
persistence rule on all eleven instruments. Part 6 gives the risk report every
column the desk can offer: Apple's own history over six windows, three return
windows, the Part 4 up-day share, and the 20-day volatility of the ten other
instruments. Ordinary least squares makes that worse than guessing the average.
The penalty repairs it, the grid finds the same alpha the lecture found once the
columns are standardised, and across the desk the penalised wide model beats the
single column on most instruments. On Apple itself it does not, and the report
says so.

Returns stay in plain decimals here, as in Parts 1 to 5, so the numbers carry
forward. The lecture used percent; that difference is stated in the quick load,
and it is why the lasso grid here is a hundred times smaller.

BLANK-SAFE, and this one needs care because the case is cumulative: no
pre-written line may CALL anything on a variable an earlier question produced.
Every such dependency sits inside the student's own blank.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "session_06" / "session_06_case.ipynb"

cells = []


def md(text):
    cells.append(new_markdown_cell(text))


def code(text, raises=False):
    c = new_code_cell(text)
    if raises:
        c.metadata["tags"] = ["raises-exception"]
    cells.append(c)


def q(qid, title, task, work, hints, sol_code, sol_note):
    md("### " + qid + " · " + title + "\n\n" + task)
    code(work)
    if isinstance(hints, str):
        hints = [hints]
    for i, h in enumerate(hints):
        label = "Hint" if len(hints) == 1 else "Hint " + str(i + 1)
        md("<details>\n<summary>\U0001f4a1 " + label + "</summary>\n\n" + h + "\n\n</details>")
    md("<details>\n<summary>✅ Solution</summary>\n\n```python\n" + sol_code +
       "\n```\n\n" + sol_note + "\n\n</details>")
    md("---")


# ---- the real numbers, so every note is exact ------------------------------
PX = pd.read_csv(ROOT / "data" / "prices.csv", parse_dates=["date"])
W = PX.pivot(index="date", columns="ticker", values="close")
R = W.pct_change()
TICKERS = sorted(W.columns)
TS5 = TimeSeriesSplit(n_splits=5)


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(np.asarray(y), np.asarray(p))))


def part5_table(ticker):
    frame = pd.DataFrame({
        "vol_20d": R[ticker].rolling(20).std(),
        "ret_20d": R[ticker].rolling(20).mean(),
        "up_20d": (R[ticker] > 0).rolling(20).mean(),
    })
    frame["vol_next"] = R[ticker].rolling(20).std().shift(-20)
    return frame.dropna()


def wide_table(ticker):
    r = R[ticker]
    frame = pd.DataFrame()
    for w in [5, 10, 20, 40, 60, 120]:
        frame["vol_" + str(w) + "d"] = r.rolling(w).std()
    for w in [5, 20, 60]:
        frame["ret_" + str(w) + "d"] = r.rolling(w).mean()
    frame["up_20d"] = (r > 0).rolling(20).mean()
    for t in TICKERS:
        if t != ticker:
            frame[t + "_vol"] = R[t].rolling(20).std()
    frame["vol_next"] = r.rolling(20).std().shift(-20)
    return frame.dropna()


# Part 5, restated
T5 = part5_table("AAPL")
TR5, TE5 = T5.loc[:"2022-12-31"], T5.loc["2023-01-01":]
M5 = LinearRegression().fit(TR5[["vol_20d"]], TR5["vol_next"])
P5_RMSE = rmse(TE5["vol_next"], M5.predict(TE5[["vol_20d"]]))
P5_PERS = rmse(TE5["vol_next"], TE5["vol_20d"])
P5_BEAT = 0
for _t in TICKERS:
    _tb = part5_table(_t)
    _tr, _te = _tb.loc[:"2022-12-31"], _tb.loc["2023-01-01":]
    _m = LinearRegression().fit(_tr[["vol_20d"]], _tr["vol_next"])
    P5_BEAT += rmse(_te["vol_next"], _m.predict(_te[["vol_20d"]])) < rmse(_te["vol_next"], _te["vol_20d"])

# Part 6
TBL = wide_table("AAPL")
TRAIN, TEST = TBL.loc[:"2022-12-31"], TBL.loc["2023-01-01":]
COLS = list(TBL.columns[:-1])
N_TBL, N_TRAIN, N_TEST = len(TBL), len(TRAIN), len(TEST)
ONE = LinearRegression().fit(TRAIN[["vol_20d"]], TRAIN["vol_next"])
ONE_TR, ONE_TE = rmse(TRAIN["vol_next"], ONE.predict(TRAIN[["vol_20d"]])), rmse(TEST["vol_next"], ONE.predict(TEST[["vol_20d"]]))
PERS_TE = rmse(TEST["vol_next"], TEST["vol_20d"])
MEAN_TE = rmse(TEST["vol_next"], np.full(N_TEST, TRAIN["vol_next"].mean()))
OLS = LinearRegression().fit(TRAIN[COLS], TRAIN["vol_next"])
OLS_TR, OLS_TE = rmse(TRAIN["vol_next"], OLS.predict(TRAIN[COLS])), rmse(TEST["vol_next"], OLS.predict(TEST[COLS]))
N_NEG = int((OLS.coef_ < 0).sum())
NEG_NAMES = [c for c, b in zip(COLS, OLS.coef_) if b < 0]
NEG_VOL = [c for c in NEG_NAMES if not c.startswith("ret_")]
RAW1000 = Ridge(alpha=1000).fit(TRAIN[COLS], TRAIN["vol_next"])
RAW1000_TE = rmse(TEST["vol_next"], RAW1000.predict(TEST[COLS]))
RAW1000_MAXCOEF = float(np.abs(RAW1000.coef_).max())
PIPE1000 = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1000))]).fit(TRAIN[COLS], TRAIN["vol_next"])
PIPE1000_TE = rmse(TEST["vol_next"], PIPE1000.predict(TEST[COLS]))
GRID = [1, 10, 100, 1000, 10000]
CV_BY_ALPHA = {a: float(-cross_val_score(Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=a))]),
                                         TRAIN[COLS], TRAIN["vol_next"], cv=TS5,
                                         scoring="neg_root_mean_squared_error").mean()) for a in GRID}
CV_BEST = min(CV_BY_ALPHA, key=CV_BY_ALPHA.get)
SEARCH = GridSearchCV(Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]), {"ridge__alpha": GRID},
                      cv=TS5, scoring="neg_root_mean_squared_error").fit(TRAIN[COLS], TRAIN["vol_next"])
RIDGE_TE = rmse(TEST["vol_next"], SEARCH.predict(TEST[COLS]))
RIDGE_ALPHA, RIDGE_CV = SEARCH.best_params_["ridge__alpha"], float(-SEARCH.best_score_)
ONE_CV = float(-cross_val_score(LinearRegression(), TRAIN[["vol_20d"]], TRAIN["vol_next"], cv=TS5,
                                scoring="neg_root_mean_squared_error").mean())
LGRID = [0.00001, 0.0001, 0.001, 0.01]
LSEARCH = GridSearchCV(Pipeline([("scale", StandardScaler()), ("lasso", Lasso(max_iter=20000))]), {"lasso__alpha": LGRID},
                       cv=TS5, scoring="neg_root_mean_squared_error").fit(TRAIN[COLS], TRAIN["vol_next"])
LASSO_ALPHA, LASSO_CV = LSEARCH.best_params_["lasso__alpha"], float(-LSEARCH.best_score_)
LASSO_TE = rmse(TEST["vol_next"], LSEARCH.predict(TEST[COLS]))
LASSO_COEF = LSEARCH.best_estimator_.named_steps["lasso"].coef_
KEPT = [c for c, b in zip(COLS, LASSO_COEF) if b != 0]
KEPT_BY_FOLD = []
for _fi, _sc in TS5.split(TRAIN):
    _blk = TRAIN.iloc[_fi]
    _m = Pipeline([("scale", StandardScaler()), ("lasso", Lasso(alpha=LASSO_ALPHA, max_iter=20000))]).fit(_blk[COLS], _blk["vol_next"])
    KEPT_BY_FOLD.append([c for c, b in zip(COLS, _m.named_steps["lasso"].coef_) if b != 0])
ALWAYS_KEPT = [c for c in COLS if all(c in k for k in KEPT_BY_FOLD)]
EGRID = {"enet__alpha": [0.0001, 0.001, 0.01], "enet__l1_ratio": [0.1, 0.5, 0.9]}
ESEARCH = GridSearchCV(Pipeline([("scale", StandardScaler()), ("enet", ElasticNet(max_iter=20000))]), EGRID,
                       cv=TS5, scoring="neg_root_mean_squared_error").fit(TRAIN[COLS], TRAIN["vol_next"])
ENET_BEST, ENET_CV = ESEARCH.best_params_, float(-ESEARCH.best_score_)
ENET_TE = rmse(TEST["vol_next"], ESEARCH.predict(TEST[COLS]))
ENET_NZ = int((ESEARCH.best_estimator_.named_steps["enet"].coef_ != 0).sum())

DESK = {}
for _t in TICKERS:
    _tb = wide_table(_t)
    _tr, _te = _tb.loc[:"2022-12-31"], _tb.loc["2023-01-01":]
    _cc = list(_tb.columns[:-1])
    _g = GridSearchCV(Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]), {"ridge__alpha": GRID},
                      cv=TS5, scoring="neg_root_mean_squared_error").fit(_tr[_cc], _tr["vol_next"])
    _o = LinearRegression().fit(_tr[["vol_20d"]], _tr["vol_next"])
    DESK[_t] = (rmse(_te["vol_next"], _g.predict(_te[_cc])), rmse(_te["vol_next"], _o.predict(_te[["vol_20d"]])))
N_DESK = sum(1 for t in TICKERS if DESK[t][0] < DESK[t][1])
DESK_LOSERS = [t for t in TICKERS if DESK[t][0] >= DESK[t][1]]
DESK_BEST = max(TICKERS, key=lambda t: (DESK[t][1] - DESK[t][0]) / DESK[t][1])
DESK_BEST_PCT = 100 * (DESK[DESK_BEST][1] - DESK[DESK_BEST][0]) / DESK[DESK_BEST][1]

_calm = TEST["vol_20d"] < TEST["vol_20d"].median()
N_CALM = int(_calm.sum())
_e_ridge = TEST["vol_next"] - SEARCH.predict(TEST[COLS])
_e_one = TEST["vol_next"] - ONE.predict(TEST[["vol_20d"]])
CALM_RIDGE, BUSY_RIDGE = float(np.sqrt((_e_ridge[_calm] ** 2).mean())), float(np.sqrt((_e_ridge[~_calm] ** 2).mean()))
CALM_ONE, BUSY_ONE = float(np.sqrt((_e_one[_calm] ** 2).mean())), float(np.sqrt((_e_one[~_calm] ** 2).mean()))

# ---------------------------------------------------------------- top matter
md(
"# \U0001f4bc The Analyst's Notebook · Part 6\n"
"### Every column the desk can offer\n\n"
"Part 5 ended with a model of one column, chosen by cross-validation, beating "
f"the persistence rule on {P5_BEAT} of the eleven instruments. It also ended "
"with a question left open: the desk has far more information than Apple's own "
"last twenty days. Six windows of its history, the direction of its recent "
"returns, and the volatility of every other instrument on the desk are all "
"known on the day the forecast is made.\n\n"
"Part 6 puts all of that in. Ordinary least squares makes the forecast worse, "
"and then a penalty on the coefficients, chosen with the folds from Part 5, "
"repairs it. Whether the repaired model beats the one-column model is a "
"question with a different answer on Apple and on the desk as a whole, and the "
"report at the end has to say both."
)

md(
"## How to work through this\n\n"
"- Run the **quick load** cell first. It brings back what Part 5 established and "
"loads the price table.\n"
"- Each question builds on the last, so keep them in order and keep your "
"variables. Later questions use the names earlier ones created.\n"
"- Cells with `...` are blanks. The notebook runs cleanly even before you fill "
"them in, so **Run all** is always safe.\n"
"- Hints and solutions are folded under each question. Work first, then check.\n\n"
"**A note on units.** The lecture worked in percent. This notebook keeps the "
"plain decimals of Parts 1 to 5, so an RMSE of `0.004` means 0.4 percentage "
"points of daily volatility. One consequence matters today: the lasso's alpha "
"scales with the target, so its grid here is a hundred times smaller than the "
"lecture's. Ridge's does not, once the columns are standardised.\n\n"
"*Stuck for more than 15 minutes? Ask a friend, ask an AI for a hint (not the "
"answer), or email me at `jobo@econ.au.dk`.*"
)

md("---")

# ------------------------------------------------------------- quick load
md(
"## ⚙️ Quick load\n\n"
"The packages, the price table, and what Part 5 left you. Run it and read what "
"it prints."
)

code(
'''import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV

CANDIDATE_DIRS = ["data", os.path.join("..", "data"), "."]
REPO_RAW_URL = "https://raw.githubusercontent.com/theill95/mlfin-2026/main/data/"   # used when the CSV files are not next to the notebook


def data_path(filename):
    """Where the course CSV files are, wherever you happen to be running."""
    for folder in CANDIDATE_DIRS:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path
    if REPO_RAW_URL is not None:
        return REPO_RAW_URL + filename
    raise FileNotFoundError(
        f"Could not find {filename}. Run this notebook from the course folder, "
        f"upload the CSV into Colab, or set REPO_RAW_URL."
    )


def rmse(actual, predicted):
    """Root mean squared error, as in the lecture, as a plain number."""
    return float(np.sqrt(mean_squared_error(actual, predicted)))


# The whole universe: eleven instruments, 2015 to 2024, returns in plain decimals
prices = pd.read_csv(data_path("prices.csv"), parse_dates=["date"])
wide = prices.pivot(index="date", columns="ticker", values="close")
rets = wide.pct_change()
TICKERS = sorted(prices["ticker"].unique())

folds = TimeSeriesSplit(n_splits=5)

# --- What Part 5 established ---
part5_target = "sd of daily returns over the next 20 trading days"
part5_features = ["vol_20d", "ret_20d", "up_20d"]     # the three candidates' columns
part5_chosen = ["vol_20d"]                            # the one cross-validation kept
part5_split = "by date: train to 2022-12-31, test from 2023-01-01"
part5_rmse = ''' + f"{P5_RMSE:.5f}" + '''          # the chosen model, on the test block
part5_pers_rmse = ''' + f"{P5_PERS:.5f}" + '''     # repeat the last 20 days
part5_beat = ''' + f"{P5_BEAT}" + '''               # instruments where the model beat persistence, of 11

print("Loaded prices:", prices.shape[0], "rows")
print("Instruments  :", ", ".join(TICKERS))
print()
print("Part 5 left you a chosen model:")
print("  target  :", part5_target)
print("  columns :", part5_chosen, "chosen from", part5_features)
print("  split   :", part5_split)
print(f"  test RMSE {part5_rmse:.5f}  against {part5_pers_rmse:.5f} for persistence")
print(f"  beats persistence on {part5_beat} of {len(TICKERS)} instruments")
print()
print("Today the desk hands you every column it has.")'''
)

md("---")

# ==================================================================== Q1
q("Q1", "Where Part 5 stopped",
  "Rebuild Part 5's table for Apple, the three features and the target with "
  "incomplete rows dropped, split it at the end of 2022, and refit the chosen "
  "one-column model. Print its test RMSE and check it matches `part5_rmse`.\n\n"
  "$$\\text{vol\\_next}_t = \\text{sd}\\big(r_{t+1},\\, \\ldots,\\, r_{t+20}\\big)$$",
  "table5 = ...\ntrain5 = ...\ntest5 = ...\n\nmodel5 = LinearRegression()\n...\n\n"
  "check = ...\nprint(check)\nprint('matches Part 5:', ...)",
  ["The three features are `rets['AAPL'].rolling(20).std()`, "
   "`.rolling(20).mean()` and `(rets['AAPL'] > 0).rolling(20).mean()`; the target "
   "is the first of those with `.shift(-20)`.",
   "Fit on `train5[['vol_20d']]`, score on `test5`. `abs(check - part5_rmse) < 0.00001` "
   "is True when the two agree."],
  "table5 = pd.DataFrame({\n    'vol_20d': rets['AAPL'].rolling(20).std(),\n"
  "    'ret_20d': rets['AAPL'].rolling(20).mean(),\n"
  "    'up_20d': (rets['AAPL'] > 0).rolling(20).mean(),\n})\n"
  "table5['vol_next'] = rets['AAPL'].rolling(20).std().shift(-20)\ntable5 = table5.dropna()\n"
  "train5 = table5.loc[:'2022-12-31']\ntest5 = table5.loc['2023-01-01':]\n\n"
  "model5 = LinearRegression()\nmodel5.fit(train5[['vol_20d']], train5['vol_next'])\n\n"
  "check = rmse(test5['vol_next'], model5.predict(test5[['vol_20d']]))\nprint(check)\n"
  "print('matches Part 5:', abs(check - part5_rmse) < 0.00001)",
  f"{P5_RMSE:.5f}, and `True`. Everything in this part is measured against that "
  "number, so it is worth one cell to see it come back.")

# ==================================================================== Q2
q("Q2", "Every column the desk can offer",
  "Build the wide table, `table`, from three loops: Apple's volatility over "
  "`[5, 10, 20, 40, 60, 120]` days as `vol_<w>d`, its average return over "
  "`[5, 20, 60]` days as `ret_<w>d`, then Part 4's `up_20d`, then the 20-day "
  "volatility of every **other** instrument as `<ticker>_vol`. Add the target and "
  "drop incomplete rows. Print the shape.",
  "table = pd.DataFrame()\n\nfor w in [5, 10, 20, 40, 60, 120]:\n    ...\n\n"
  "for w in [5, 20, 60]:\n    ...\n\ntable['up_20d'] = ...\n\n"
  "for t in TICKERS:\n    ...\n\ntable['vol_next'] = ...\ntable = ...\nprint(...)",
  ["Column names are text built from the number: `'vol_' + str(w) + 'd'`. "
   "Inside the last loop, `if t != 'AAPL':`.",
   "The target is `rets['AAPL'].rolling(20).std().shift(-20)`, then `.dropna()`."],
  "table = pd.DataFrame()\n\nfor w in [5, 10, 20, 40, 60, 120]:\n"
  "    table['vol_' + str(w) + 'd'] = rets['AAPL'].rolling(w).std()\n\n"
  "for w in [5, 20, 60]:\n    table['ret_' + str(w) + 'd'] = rets['AAPL'].rolling(w).mean()\n\n"
  "table['up_20d'] = (rets['AAPL'] > 0).rolling(20).mean()\n\n"
  "for t in TICKERS:\n    if t != 'AAPL':\n        table[t + '_vol'] = rets[t].rolling(20).std()\n\n"
  "table['vol_next'] = rets['AAPL'].rolling(20).std().shift(-20)\ntable = table.dropna()\nprint(table.shape)",
  f"{N_TBL:,} rows and 21 columns: twenty features and the target. The row count "
  "fell from Part 5's 2,476 because the 120-day window needs a hundred more days "
  "to fill. That matters for the next question.")

# ==================================================================== Q3
q("Q3", "The same split, and a fair one-column baseline",
  "Split `table` at the end of 2022 into `train` and `test`, and store the twenty "
  "feature names in `columns`. Then refit the one-column model on the **new** "
  "training rows and store its test RMSE as `one_rmse`. Why refit rather than "
  "reuse `part5_rmse`?",
  "train = ...\ntest = ...\ncolumns = ...\n\none_model = LinearRegression()\n...\n"
  "one_rmse = ...\nprint(one_rmse)",
  ["`list(table.columns[:-1])` is every column but the target.",
   "The rows changed in Q2, so the fair comparison is a one-column model fitted "
   "and scored on exactly the rows the wide models will use."],
  "train = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\n"
  "columns = list(table.columns[:-1])\n\n"
  "one_model = LinearRegression()\none_model.fit(train[['vol_20d']], train['vol_next'])\n"
  "one_rmse = rmse(test['vol_next'], one_model.predict(test[['vol_20d']]))\nprint(one_rmse)",
  f"{ONE_TE:.5f}, a little above Part 5's {P5_RMSE:.5f} because the training "
  "block lost its first hundred days. Every comparison from here on is against "
  "this number, on these rows.")

# ==================================================================== Q4
q("Q4", "Everything in",
  "Fit ordinary least squares on all twenty columns as `ols_model`. Print its "
  "training RMSE and its test RMSE next to the one-column model's, and store the "
  "test RMSE as `ols_rmse`. Then print the error of guessing the training average.",
  "ols_model = LinearRegression()\n...\n\n"
  "print('one column, train:', ...)\nprint('all columns, train:', ...)\n"
  "print('one column, test :', ...)\nols_rmse = ...\nprint('all columns, test :', ...)\n\n"
  "print('guess the average:', ...)",
  ["Training RMSE scores predictions on `train` against `train['vol_next']`.",
   "The average guess is `np.full(len(test), train['vol_next'].mean())`."],
  "ols_model = LinearRegression()\nols_model.fit(train[columns], train['vol_next'])\n\n"
  "print('one column, train:', rmse(train['vol_next'], one_model.predict(train[['vol_20d']])))\n"
  "print('all columns, train:', rmse(train['vol_next'], ols_model.predict(train[columns])))\n"
  "print('one column, test :', one_rmse)\n"
  "ols_rmse = rmse(test['vol_next'], ols_model.predict(test[columns]))\n"
  "print('all columns, test :', ols_rmse)\n\n"
  "print('guess the average:', rmse(test['vol_next'], np.full(len(test), train['vol_next'].mean())))",
  f"Training error falls from {ONE_TR:.5f} to {OLS_TR:.5f}; test error rises "
  f"from {ONE_TE:.5f} to {OLS_TE:.5f}. **The twenty-column OLS is worse than "
  f"guessing the average**, which scores {MEAN_TE:.5f}. Nineteen extra columns "
  "did not add information the model could use; they added freedom to fit noise, "
  "and it used all of it.")

# ==================================================================== Q5
q("Q5", "Count the wrong signs",
  "Every **volatility** column in this table should, if anything, raise the "
  "forecast when it rises. Count the negative coefficients in `ols_model` with a "
  "mask, then print their names with a loop. Which of them are volatility columns?",
  "n_negative = ...\nprint(n_negative)\n\n...",
  "`(ols_model.coef_ < 0).sum()` counts. Then `for name, b in zip(columns, "
  "ols_model.coef_):` with `if b < 0: print(name)` inside.",
  "n_negative = (ols_model.coef_ < 0).sum()\nprint(n_negative, 'of', len(columns))\n\n"
  "for name, b in zip(columns, ols_model.coef_):\n    if b < 0:\n        print(name)",
  f"{N_NEG} of the twenty are negative. The two return columns are allowed to "
  f"be, since a falling market is a volatile one. The other {len(NEG_VOL)} are "
  f"volatility columns, `{NEG_VOL[0]}` and `{NEG_VOL[1]}` among them: more than "
  "half of the desk's volatility columns lower the forecast of Apple's "
  "volatility, which no one believes. The columns are near-copies of each other, "
  "and OLS is trading coefficient between them.")

# ==================================================================== Q6
q("Q6", "The lecture's alpha, on raw columns",
  "Fit `Ridge(alpha=1000)` on the twenty raw columns and print its test RMSE. It "
  "will look familiar. Which number from Q4 does it match, and why?",
  "raw_ridge = ...\n...\nprint(...)",
  ["Fit and predict exactly as with `LinearRegression`.",
   "The columns are in decimals, so their values are around 0.01. A coefficient "
   "of 1 on such a column moves the forecast by 0.01, and alpha 1000 charges 1000 "
   "for it. Every slope is crushed and the forecast is the intercept."],
  "raw_ridge = Ridge(alpha=1000)\nraw_ridge.fit(train[columns], train['vol_next'])\n"
  "print(rmse(test['vol_next'], raw_ridge.predict(test[columns])))",
  f"{RAW1000_TE:.5f}, which is the average guess from Q4 to four decimals. The "
  f"largest coefficient left is {RAW1000_MAXCOEF:.4f}. In decimals every column "
  "is a hundred times smaller than in the lecture, so it needs a coefficient a "
  "hundred times larger, which the penalty charges ten thousand times more for. "
  "This is the units problem, and it is why the next step is not a smaller alpha.")

# ==================================================================== Q7
q("Q7", "Standardise, then penalise",
  "Build a `Pipeline` with a `StandardScaler` step named `'scale'` and a "
  "`Ridge(alpha=1000)` step named `'ridge'`. Fit it and print the test RMSE.",
  "pipe = Pipeline([...])\n...\nprint(...)",
  "`Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1000))])`, then "
  "`.fit` and `.predict` on the raw columns; the scaling happens inside.",
  "pipe = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1000))])\n"
  "pipe.fit(train[columns], train['vol_next'])\n"
  "print(rmse(test['vol_next'], pipe.predict(test[columns])))",
  f"{PIPE1000_TE:.5f}. The same alpha that flattened everything in Q6 now gives "
  f"a forecast better than the twenty-column OLS ({OLS_TE:.5f}) and still worse "
  f"than the single column ({ONE_TE:.5f}). With every column in standard "
  "deviations, the penalty judges columns by their usefulness rather than their "
  "units.")

# ==================================================================== Q8
q("Q8", "Choose alpha on the folds",
  "Loop over `[1, 10, 100, 1000, 10000]`: build the pipeline with that alpha, "
  "cross-validate it on the training rows with `folds`, and store the mean RMSE "
  "in `cv_by_alpha`. Print the dictionary and the best alpha.",
  "cv_by_alpha = {}\n\nfor alpha in [1, 10, 100, 1000, 10000]:\n    ...\n\n"
  "print(cv_by_alpha)\nprint('best:', ...)",
  ["`cross_val_score(pipe, train[columns], train['vol_next'], cv=folds, "
   "scoring='neg_root_mean_squared_error')`, then `-scores.mean()`.",
   "`min(cv_by_alpha, key=cv_by_alpha.get)` is the key with the smallest value."],
  "cv_by_alpha = {}\n\nfor alpha in [1, 10, 100, 1000, 10000]:\n"
  "    pipe = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=alpha))])\n"
  "    scores = cross_val_score(pipe, train[columns], train['vol_next'],\n"
  "                             cv=folds, scoring='neg_root_mean_squared_error')\n"
  "    cv_by_alpha[alpha] = round(float(-scores.mean()), 5)\n\n"
  "print(cv_by_alpha)\nprint('best:', min(cv_by_alpha, key=cv_by_alpha.get))",
  f"Best at {CV_BEST}, the same alpha the lecture found in percent. Scaling the "
  "target by a hundred scales both the residual sum of squares and the squared "
  "coefficients by ten thousand, so the trade-off between them is unchanged. "
  "That is a property of ridge; it is not true of the lasso.")

# ==================================================================== Q9
q("Q9", "GridSearchCV, and the test block once",
  "Let `GridSearchCV` do Q8: the pipeline with `Ridge()` and no alpha, the grid "
  "`{'ridge__alpha': [1, 10, 100, 1000, 10000]}`, `folds`, and the RMSE scoring. "
  "Fit it as `search`, print the best alpha and score, then predict the test rows "
  "**once** and store the RMSE as `ridge_rmse`.",
  "pipe = Pipeline([('scale', StandardScaler()), ('ridge', Ridge())])\ngrid = ...\n\n"
  "search = ...\n...\n\nprint(...)\nprint(...)\n\nridge_rmse = ...\nprint('test:', ridge_rmse, ' one column:', one_rmse)",
  ["`GridSearchCV(pipe, grid, cv=folds, scoring='neg_root_mean_squared_error')`, "
   "then `.fit(train[columns], train['vol_next'])`.",
   "`search.predict(test[columns])` uses the best pipeline refitted on all the "
   "training rows."],
  "pipe = Pipeline([('scale', StandardScaler()), ('ridge', Ridge())])\n"
  "grid = {'ridge__alpha': [1, 10, 100, 1000, 10000]}\n\n"
  "search = GridSearchCV(pipe, grid, cv=folds, scoring='neg_root_mean_squared_error')\n"
  "search.fit(train[columns], train['vol_next'])\n\n"
  "print(search.best_params_)\nprint(-search.best_score_)\n\n"
  "ridge_rmse = rmse(test['vol_next'], search.predict(test[columns]))\n"
  "print('test:', ridge_rmse, ' one column:', one_rmse)",
  f"Alpha {RIDGE_ALPHA} at a cross-validated {RIDGE_CV:.5f}, and {RIDGE_TE:.5f} "
  f"on the test block against {ONE_TE:.5f} for the single column. On Apple, "
  "twenty penalised columns lose to one unpenalised column. The folds already "
  f"said so: the one-column model cross-validates at {ONE_CV:.5f}, below every "
  "row of the grid. The penalty repaired the damage of Q4; it did not turn the "
  "extra columns into signal.")

# ==================================================================== Q10
q("Q10", "Lasso, on its own scale",
  "Search a lasso pipeline (step named `'lasso'`) over "
  "`{'lasso__alpha': [0.00001, 0.0001, 0.001, 0.01]}` as `lasso_search`. Print the "
  "best alpha and score, store the test RMSE as `lasso_rmse`, and print the names "
  "of the columns the winning lasso kept as the list `kept`.",
  "lasso_pipe = Pipeline([('scale', StandardScaler()), ('lasso', Lasso(max_iter=20000))])\n"
  "lasso_grid = ...\n\nlasso_search = ...\n...\n\nprint(...)\nprint(...)\n"
  "lasso_rmse = ...\nprint('test:', lasso_rmse)\n\n"
  "coefs = ...\nkept = []\n...\nprint(kept)",
  ["The lecture's grid ran from 0.001 to 1 in percent. The lasso's penalty is on "
   "$|\\beta|$, which scales with the target, so in decimals the grid is a hundred "
   "times smaller.",
   "`lasso_search.best_estimator_.named_steps['lasso'].coef_` are the "
   "coefficients. Loop over `zip(columns, coefs)` and append `name` when `b != 0`."],
  "lasso_pipe = Pipeline([('scale', StandardScaler()), ('lasso', Lasso(max_iter=20000))])\n"
  "lasso_grid = {'lasso__alpha': [0.00001, 0.0001, 0.001, 0.01]}\n\n"
  "lasso_search = GridSearchCV(lasso_pipe, lasso_grid, cv=folds, scoring='neg_root_mean_squared_error')\n"
  "lasso_search.fit(train[columns], train['vol_next'])\n\n"
  "print(lasso_search.best_params_)\nprint(-lasso_search.best_score_)\n"
  "lasso_rmse = rmse(test['vol_next'], lasso_search.predict(test[columns]))\nprint('test:', lasso_rmse)\n\n"
  "coefs = lasso_search.best_estimator_.named_steps['lasso'].coef_\nkept = []\n"
  "for name, b in zip(columns, coefs):\n    if b != 0:\n        kept.append(name)\nprint(kept)",
  f"Alpha {LASSO_ALPHA} at {LASSO_CV:.5f}, and {LASSO_TE:.5f} on the test block: "
  f"a little better than ridge, still behind the single column. It keeps "
  f"{len(KEPT)} of the twenty columns: {', '.join(KEPT)}. `max_iter=20000` is "
  "there because small alphas need more passes than the default, and the "
  "convergence warning would otherwise appear.")

# ==================================================================== Q11
q("Q11", "The kept set, fold by fold",
  "Fit the winning lasso (its alpha is in `lasso_search.best_params_`) on the "
  "fitting rows of each fold and print the columns it keeps each time. Is the "
  "list from Q10 stable?",
  "best_alpha = ...\n\n...",
  ["`best_alpha = lasso_search.best_params_['lasso__alpha']`. Then "
   "`for fit_rows, score_rows in folds.split(train):`, with "
   "`block = train.iloc[fit_rows]` and a fresh pipeline fitted on the block.",
   "Collect the kept names with the same loop as Q10 and print the list."],
  "best_alpha = lasso_search.best_params_['lasso__alpha']\n\n"
  "for fit_rows, score_rows in folds.split(train):\n"
  "    block = train.iloc[fit_rows]\n"
  "    fold_lasso = Pipeline([('scale', StandardScaler()), ('lasso', Lasso(alpha=best_alpha, max_iter=20000))])\n"
  "    fold_lasso.fit(block[columns], block['vol_next'])\n"
  "    fold_kept = []\n"
  "    for name, b in zip(columns, fold_lasso.named_steps['lasso'].coef_):\n"
  "        if b != 0:\n            fold_kept.append(name)\n"
  "    print(fold_kept)",
  "Five different lists, with " +
  (f"only `{'`, `'.join(ALWAYS_KEPT)}` in every one" if ALWAYS_KEPT else "no column in every one") +
  ". The lasso's zeros say which near-copy it happened to keep on those rows, "
  "not which columns carry information. Report the kept list as a compact "
  "forecast, never as a finding about the columns.")

# ==================================================================== Q12
q("Q12", "Both penalties",
  "Search an elastic net pipeline (step `'enet'`) over `alpha` in "
  "`[0.0001, 0.001, 0.01]` and `l1_ratio` in `[0.1, 0.5, 0.9]` as `enet_search`. "
  "Print the best pair and score, and store the test RMSE as `enet_rmse`.",
  "enet_pipe = Pipeline([('scale', StandardScaler()), ('enet', ElasticNet(max_iter=20000))])\n"
  "enet_grid = {...}\n\nenet_search = ...\n...\n\nprint(...)\nprint(...)\nenet_rmse = ...\nprint('test:', enet_rmse)",
  "Two keys in one dictionary: `{'enet__alpha': [...], 'enet__l1_ratio': [...]}`. "
  "Nine combinations, forty-five fits.",
  "enet_pipe = Pipeline([('scale', StandardScaler()), ('enet', ElasticNet(max_iter=20000))])\n"
  "enet_grid = {'enet__alpha': [0.0001, 0.001, 0.01], 'enet__l1_ratio': [0.1, 0.5, 0.9]}\n\n"
  "enet_search = GridSearchCV(enet_pipe, enet_grid, cv=folds, scoring='neg_root_mean_squared_error')\n"
  "enet_search.fit(train[columns], train['vol_next'])\n\n"
  "print(enet_search.best_params_)\nprint(-enet_search.best_score_)\n"
  "enet_rmse = rmse(test['vol_next'], enet_search.predict(test[columns]))\nprint('test:', enet_rmse)",
  f"Alpha {ENET_BEST['enet__alpha']} with l1_ratio {ENET_BEST['enet__l1_ratio']}, "
  f"at {ENET_CV:.5f} on the folds and {ENET_TE:.5f} on the test block, keeping "
  f"{ENET_NZ} columns. Three penalised models within a hair of each other, all "
  "behind the single column on Apple. The folds ranked them the same way, which "
  "is the point of having the folds.")

# ==================================================================== Q13
q("Q13", "Across the desk",
  "Apple is one instrument. Write `penalised_vs_one(ticker)`: build the wide "
  "table for that ticker (its own windows, `up_20d`, the other ten instruments' "
  "volatility), split at the end of 2022, fit the one-column model, run the ridge "
  "grid search from Q9, and return the pair `(ridge_rmse, one_rmse)` on the test "
  "block. Run it for every ticker into a dictionary `desk` and count the wins.",
  "def penalised_vs_one(ticker):\n    ...\n\ndesk = {}\nfor ticker in TICKERS:\n    ...\n\n"
  "wins = ...\nprint('penalised wins on', ..., 'of', ...)",
  ["The function is Q2, Q3 and Q9 with `ticker` in place of `'AAPL'`, including "
   "in the `if` that skips the instrument itself.",
   "`sum(1 for t in desk if desk[t][0] < desk[t][1])` counts the wins. Eleven "
   "grid searches take a little while."],
  "def penalised_vs_one(ticker):\n    frame = pd.DataFrame()\n"
  "    for w in [5, 10, 20, 40, 60, 120]:\n        frame['vol_' + str(w) + 'd'] = rets[ticker].rolling(w).std()\n"
  "    for w in [5, 20, 60]:\n        frame['ret_' + str(w) + 'd'] = rets[ticker].rolling(w).mean()\n"
  "    frame['up_20d'] = (rets[ticker] > 0).rolling(20).mean()\n"
  "    for t in TICKERS:\n        if t != ticker:\n            frame[t + '_vol'] = rets[t].rolling(20).std()\n"
  "    frame['vol_next'] = rets[ticker].rolling(20).std().shift(-20)\n    frame = frame.dropna()\n\n"
  "    tr = frame.loc[:'2022-12-31']\n    te = frame.loc['2023-01-01':]\n    cols = list(frame.columns[:-1])\n\n"
  "    one = LinearRegression()\n    one.fit(tr[['vol_20d']], tr['vol_next'])\n"
  "    pipe = Pipeline([('scale', StandardScaler()), ('ridge', Ridge())])\n"
  "    grid_search = GridSearchCV(pipe, {'ridge__alpha': [1, 10, 100, 1000, 10000]},\n"
  "                               cv=folds, scoring='neg_root_mean_squared_error')\n"
  "    grid_search.fit(tr[cols], tr['vol_next'])\n\n"
  "    return (rmse(te['vol_next'], grid_search.predict(te[cols])),\n"
  "            rmse(te['vol_next'], one.predict(te[['vol_20d']])))\n\n"
  "desk = {}\nfor ticker in TICKERS:\n    desk[ticker] = penalised_vs_one(ticker)\n\n"
  "wins = sum(1 for t in desk if desk[t][0] < desk[t][1])\n"
  "print('penalised wins on', wins, 'of', len(desk))",
  f"{N_DESK} of {len(TICKERS)}. The penalised wide model beats the single column "
  f"on most of the desk, by {DESK_BEST_PCT:.0f}% on {DESK_BEST}, and loses on "
  f"{', '.join(DESK_LOSERS)}. Apple, the instrument every part of this case has "
  "been built on, is one of the exceptions. A result checked on one stock is one "
  "observation.")

# ==================================================================== Q14
q("Q14", "Calm days and busy days",
  "Split Apple's test block with a mask, days where `vol_20d` is below its "
  "median and the rest, and print the RMSE of both the ridge `search` and "
  "`one_model` on each half. Does the wide model help where it matters?",
  "ridge_errors = ...\none_errors = ...\ncalm = ...\n\n"
  "print('calm, ridge:', ...)\nprint('calm, one  :', ...)\n"
  "print('busy, ridge:', ...)\nprint('busy, one  :', ...)",
  ["`test['vol_next'] - search.predict(test[columns])` and the same with "
   "`one_model.predict(test[['vol_20d']])`.",
   "`calm = test['vol_20d'] < test['vol_20d'].median()`; `errors[~calm]` is the "
   "busy half; `np.sqrt((errors[calm] ** 2).mean())`."],
  "ridge_errors = test['vol_next'] - search.predict(test[columns])\n"
  "one_errors = test['vol_next'] - one_model.predict(test[['vol_20d']])\n"
  "calm = test['vol_20d'] < test['vol_20d'].median()\n\n"
  "print('calm, ridge:', round(np.sqrt((ridge_errors[calm] ** 2).mean()), 5))\n"
  "print('calm, one  :', round(np.sqrt((one_errors[calm] ** 2).mean()), 5))\n"
  "print('busy, ridge:', round(np.sqrt((ridge_errors[~calm] ** 2).mean()), 5))\n"
  "print('busy, one  :', round(np.sqrt((one_errors[~calm] ** 2).mean()), 5))",
  f"Calm: {CALM_RIDGE:.5f} against {CALM_ONE:.5f}. Busy: {BUSY_RIDGE:.5f} against "
  f"{BUSY_ONE:.5f}. The single column wins both halves on Apple, so there is no "
  "regime in which the wide model earns its place here. That is a cleaner "
  "statement than \"it loses on average\", and it took one mask.")

# ==================================================================== Q15
q("Q15", "Draw it",
  "One figure: what happened in the test years as a line, the one-column "
  "forecast and the ridge forecast as two more lines, with a legend and a title.",
  "fig, ax = plt.subplots(figsize=(10, 3.5))\n...\nplt.show()",
  ["`ax.plot(test.index, test['vol_next'], label='what happened')`, then the two "
   "forecasts with their own labels.",
   "`ax.legend()`, `ax.set_ylabel('20-day volatility')`, `ax.set_title(..., loc='left')`."],
  "fig, ax = plt.subplots(figsize=(10, 3.5))\n"
  "ax.plot(test.index, test['vol_next'], label='what happened', color='black', linewidth=1.2)\n"
  "ax.plot(test.index, one_model.predict(test[['vol_20d']]), label='one column')\n"
  "ax.plot(test.index, search.predict(test[columns]), label='twenty columns, ridge')\n"
  "ax.set_ylabel('20-day volatility')\nax.legend()\n"
  "ax.set_title('Apple, 2023 to 2024: two forecasts against what happened', loc='left')\n"
  "plt.show()",
  "The two forecasts track each other closely and both lag the turns, as every "
  "model of this kind does. The wide model is not doing anything different from "
  "the narrow one on Apple; it is doing the same thing with more noise.")

# ==================================================================== Q16
q("Q16", "Write down what you would defend",
  "Finish the way Parts 4 and 5 finished: one dictionary and a function that "
  "prints it with a verdict. Fill in `report`, then write `summarise(report)`: "
  "it prints each entry on its own line and ends with one sentence on whether "
  "the wide model replaces the one-column model **on Apple**, and one on the "
  "desk as a whole.",
  "report = {\n    'target': ...,\n    'columns_offered': ...,\n    'alpha': ...,\n"
  "    'chosen_by': ...,\n    'one_column_rmse': ...,\n    'ols_rmse': ...,\n"
  "    'ridge_rmse': ...,\n    'lasso_rmse': ...,\n    'desk_wins': ...,\n}\n\n"
  "def summarise(report):\n    ...\n\nsummarise(report)",
  ["Most values are already in variables: `len(columns)`, "
   "`search.best_params_['ridge__alpha']`, `one_rmse`, `ols_rmse`, `ridge_rmse`, "
   "`lasso_rmse`, and `wins` from Q13.",
   "Inside the function, `for key in report:` then "
   "`print(f'{key:18} {report[key]}')`. The verdicts are two `if`s: one on "
   "`ridge_rmse < one_column_rmse`, one on `desk_wins`."],
  "report = {\n    'target': part5_target,\n    'columns_offered': len(columns),\n"
  "    'alpha': search.best_params_['ridge__alpha'],\n"
  "    'chosen_by': 'grid search on five time-ordered folds, training rows only',\n"
  "    'one_column_rmse': round(one_rmse, 5),\n    'ols_rmse': round(ols_rmse, 5),\n"
  "    'ridge_rmse': round(ridge_rmse, 5),\n    'lasso_rmse': round(lasso_rmse, 5),\n"
  "    'desk_wins': f'{wins} of {len(desk)}',\n}\n\n"
  "def summarise(report):\n"
  "    \"\"\"Print a finished comparison, and say what replaces what.\"\"\"\n"
  "    for key in report:\n        print(f'{key:18} {report[key]}')\n\n"
  "    if report['ridge_rmse'] < report['one_column_rmse']:\n"
  "        print('\\nOn Apple, the penalised wide model replaces the one-column model.')\n"
  "    else:\n"
  "        print('\\nOn Apple, the one-column model stays. The extra columns did not help.')\n"
  "    print(f\"Across the desk, the penalised wide model wins on {report['desk_wins']} instruments.\")\n\n"
  "summarise(report)",
  "Nine lines and two verdicts, and they point in different directions. That is "
  "the honest state of things: on Apple the twenty columns were tested and did "
  "not help, and on most of the desk they did. Note that `ols_rmse` is in the "
  "report too. A reader deserves to know what the columns cost before the "
  "penalty, because it says how much of the wide model's score is the penalty's "
  "doing.")

# ---------------------------------------------------------------- closing
md(
"## \U0001f9ed What you have now\n\n"
"| what | where it lives |\n"
"|:--|:--|\n"
"| Part 5's model, rebuilt and checked | `table5`, `model5` |\n"
"| the wide table and its split | `table`, `train`, `test`, `columns` |\n"
"| the fair one-column baseline | `one_model`, `one_rmse` |\n"
"| what OLS does with twenty columns | `ols_model`, `ols_rmse` |\n"
"| the alphas on the folds | `cv_by_alpha` |\n"
"| the ridge search, refitted | `search`, `ridge_rmse` |\n"
"| the lasso search and its kept columns | `lasso_search`, `lasso_rmse`, `kept` |\n"
"| the elastic net search | `enet_search`, `enet_rmse` |\n"
"| the whole desk | `desk`, `wins` |\n"
"| the calm and busy halves | `calm` |\n"
"| the thing you would defend | `report` |"
)

md(
"## What changed since Part 5\n\n"
"- **The model got every column the desk has.** Twenty instead of one. "
f"Unpenalised, that was worse than guessing the average ({OLS_TE:.5f} against "
f"{MEAN_TE:.5f}). With a penalty chosen on the folds it was {RIDGE_TE:.5f}, "
"which repairs the damage without beating the single column on Apple.\n"
"- **The units problem showed up in earnest.** The lecture's alpha on raw "
"decimal columns flattened the model to the average (Q6). Standardising inside "
"a pipeline brought the lecture's alpha back as the right one (Q8), because "
"ridge's trade-off does not depend on the target's units. The lasso's does, "
"which is why its grid moved by a factor of a hundred (Q10).\n"
"- **The zeros are not findings.** Q11 showed the lasso keeping a different set "
"of columns on every fold block.\n"
f"- **The desk disagrees with Apple.** Q13 found the penalised wide model "
f"winning on {N_DESK} of {len(TICKERS)} instruments. The report carries both "
"numbers."
)

md(
"## Where this leaves the risk report\n\n"
"Six parts ago this was two stocks compared with a subtraction. It is now a "
"forecast with a chosen model, a documented search over a setting, a baseline "
"that beats it on the stock it was built on, and a desk-wide result that says "
"the extra columns are worth having on most of the others.\n\n"
"The tools from this part carry forward unchanged: a pipeline, a grid, the "
"folds, and the test block opened once. Every later model in the course has "
"settings of its own, and they are chosen exactly this way.\n\n"
"**Next block:** the target becomes a label rather than a number, and the same "
"penalty appears in a classifier under a different name."
)

# ---------------------------------------------------------------- write
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
for _i, _c in enumerate(nb.cells):
    _c["id"] = f"c{_i:04d}"

OUT.write_text(nbf.writes(nb), encoding="utf-8")

n_q = sum(1 for c in cells if c.cell_type == "markdown" and c.source.startswith("### Q"))
print("wrote", OUT, " (", len(cells), "cells,", n_q, "questions )")
