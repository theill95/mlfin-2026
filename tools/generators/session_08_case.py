# -*- coding: utf-8 -*-
"""Build session_08_case.ipynb  (The Analyst's Notebook, Part 8).

Conventions (approved for Sessions 1 to 6): no star badges, one cumulative
investigation where later questions reuse what earlier ones stored, folded
hints and solutions, stated formulas, plain explanatory tone, no em-dashes.
Opens with a QUICK LOAD restoring Part 6's findings. (Session 7 was a
stand-alone recap, so the case goes from Part 6 to Part 8.)

Part 6 ended with the wide table of twenty columns, a ridge penalty chosen on
the folds, and a verdict that differed on Apple and on the desk. Part 8 gives
the same table a label as its target: whether Apple's volatility over the next
20 days is above its volatility over the last 20. A straight line fails on the
label, logistic regression on the raw decimal column fails in a different way
(the default penalty crushes a coefficient that has to be a hundred times
larger in decimals than in percent, and the model predicts a rise on every
day), standardising repairs it, and the model is scored with the four counts
and the AUC, cross-validated, widened to twenty columns with C chosen on the
folds, and run across the desk.

Returns stay in plain decimals here, as in Parts 1 to 6. The label has no
units, but the penalty does; that is Q4 and Q5.

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
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error, accuracy_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "session_08" / "session_08_case.ipynb"

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


# Part 6, restated (the same code as Part 6's generator)
TBL = wide_table("AAPL")
TRAIN, TEST = TBL.loc[:"2022-12-31"], TBL.loc["2023-01-01":]
COLS = list(TBL.columns[:-1])
N_TBL, N_TRAIN, N_TEST = len(TBL), len(TRAIN), len(TEST)
ONE = LinearRegression().fit(TRAIN[["vol_20d"]], TRAIN["vol_next"])
P6_ONE = rmse(TEST["vol_next"], ONE.predict(TEST[["vol_20d"]]))
GRID6 = [1, 10, 100, 1000, 10000]
_s6 = GridSearchCV(Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]), {"ridge__alpha": GRID6},
                   cv=TS5, scoring="neg_root_mean_squared_error").fit(TRAIN[COLS], TRAIN["vol_next"])
P6_RIDGE, P6_ALPHA = rmse(TEST["vol_next"], _s6.predict(TEST[COLS])), _s6.best_params_["ridge__alpha"]
P6_WINS = 0
for _t in TICKERS:
    _tb = wide_table(_t)
    _tr, _te = _tb.loc[:"2022-12-31"], _tb.loc["2023-01-01":]
    _cc = list(_tb.columns[:-1])
    _g = GridSearchCV(Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]), {"ridge__alpha": GRID6},
                      cv=TS5, scoring="neg_root_mean_squared_error").fit(_tr[_cc], _tr["vol_next"])
    _o = LinearRegression().fit(_tr[["vol_20d"]], _tr["vol_next"])
    P6_WINS += rmse(_te["vol_next"], _g.predict(_te[_cc])) < rmse(_te["vol_next"], _o.predict(_te[["vol_20d"]]))

# Part 8
TBL["rising"] = (TBL["vol_next"] > TBL["vol_20d"]).astype(int)
TRAIN, TEST = TBL.loc[:"2022-12-31"], TBL.loc["2023-01-01":]
Y_TR, Y_TE = TRAIN["rising"], TEST["rising"]
SHARE_TR, SHARE_TE = float(Y_TR.mean()), float(Y_TE.mean())
MAJORITY = float(max(SHARE_TE, 1 - SHARE_TE))
N_RISE_TE = int(Y_TE.sum())
LINE = LinearRegression().fit(TRAIN[["vol_20d"]], Y_TR)
_fit = LINE.predict(TRAIN[["vol_20d"]])
N_OUTSIDE = int(((_fit < 0) | (_fit > 1)).sum())
MAX_VOL = float(TRAIN["vol_20d"].max())
FIT_AT_MAX = float(LINE.predict(pd.DataFrame({"vol_20d": [MAX_VOL]}))[0])
MAX_VOL_DAY = TRAIN["vol_20d"].idxmax().date()
RAW = LogisticRegression().fit(TRAIN[["vol_20d"]], Y_TR)
RAW_B0, RAW_B1 = float(RAW.intercept_[0]), float(RAW.coef_[0, 0])
RAW_CROSS = -RAW_B0 / RAW_B1
RAW_PRED = RAW.predict(TEST[["vol_20d"]])
RAW_SHARE_1, RAW_ACC = float(RAW_PRED.mean()), float(accuracy_score(Y_TE, RAW_PRED))
RAW_P = RAW.predict_proba(TEST[["vol_20d"]])[:, 1]
RAW_P_MIN, RAW_P_MAX = float(RAW_P.min()), float(RAW_P.max())
FREE = LogisticRegression(C=1000000).fit(TRAIN[["vol_20d"]], Y_TR)
FREE_B1, FREE_CROSS = float(FREE.coef_[0, 0]), float(-FREE.intercept_[0] / FREE.coef_[0, 0])
FREE_ACC = float(accuracy_score(Y_TE, FREE.predict(TEST[["vol_20d"]])))
PCT = LogisticRegression().fit(TRAIN[["vol_20d"]] * 100, Y_TR)
PCT_B1 = float(PCT.coef_[0, 0])
ONE_PIPE = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression())]).fit(TRAIN[["vol_20d"]], Y_TR)
SD_B1 = float(ONE_PIPE.named_steps["logit"].coef_[0, 0])
P_ONE = ONE_PIPE.predict_proba(TEST[["vol_20d"]])[:, 1]
PRED_ONE = ONE_PIPE.predict(TEST[["vol_20d"]])
AUC_ONE, ACC_ONE = float(roc_auc_score(Y_TE, P_ONE)), float(accuracy_score(Y_TE, PRED_ONE))
AUC_RAW = float(roc_auc_score(Y_TE, RAW_P))
SHARE_PRED_1 = float(PRED_ONE.mean())
CM = confusion_matrix(Y_TE, PRED_ONE)
TN, FP, FN, TP = int(CM[0, 0]), int(CM[0, 1]), int(CM[1, 0]), int(CM[1, 1])
PREC, REC = float(precision_score(Y_TE, PRED_ONE)), float(recall_score(Y_TE, PRED_ONE))
BY_THR = {}
for _t in [0.4, 0.5, 0.6, 0.7]:
    _pr = (P_ONE >= _t).astype(int)
    BY_THR[_t] = (round(float(precision_score(Y_TE, _pr, zero_division=0)), 3), round(float(recall_score(Y_TE, _pr)), 3),
                  round(float(accuracy_score(Y_TE, _pr)), 3))
BEST_THR_ACC = max(BY_THR, key=lambda t: BY_THR[t][2])
AUC_FOLDS = cross_val_score(Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression())]),
                            TRAIN[["vol_20d"]], Y_TR, cv=TS5, scoring="roc_auc")
def pipe(C=1.0):
    return Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(C=C, max_iter=1000))])
CV_WIDE = cross_val_score(pipe(), TRAIN[COLS], Y_TR, cv=TS5, scoring="roc_auc")
CGRID = [0.0001, 0.001, 0.01, 0.1, 1, 10]
SEARCH = GridSearchCV(pipe(), {"logit__C": CGRID}, cv=TS5, scoring="roc_auc").fit(TRAIN[COLS], Y_TR)
BEST_C, BEST_CV = SEARCH.best_params_["logit__C"], float(SEARCH.best_score_)
AUC_WIDE = float(roc_auc_score(Y_TE, SEARCH.predict_proba(TEST[COLS])[:, 1]))
ACC_WIDE = float(accuracy_score(Y_TE, SEARCH.predict(TEST[COLS])))
L1 = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(penalty="l1", solver="liblinear", C=0.01))]).fit(TRAIN[COLS], Y_TR)
KEPT = [c for c, b in zip(COLS, L1.named_steps["logit"].coef_[0]) if b != 0]
AUC_L1 = float(roc_auc_score(Y_TE, L1.predict_proba(TEST[COLS])[:, 1]))
DESK = {}
for _t in TICKERS:
    _tb = wide_table(_t)
    _tb["rising"] = (_tb["vol_next"] > _tb["vol_20d"]).astype(int)
    _tr, _te = _tb.loc[:"2022-12-31"], _tb.loc["2023-01-01":]
    _m = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression())]).fit(_tr[["vol_20d"]], _tr["rising"])
    DESK[_t] = (round(float(roc_auc_score(_te["rising"], _m.predict_proba(_te[["vol_20d"]])[:, 1])), 3),
                round(float(accuracy_score(_te["rising"], _m.predict(_te[["vol_20d"]]))), 3),
                round(float(max(_te["rising"].mean(), 1 - _te["rising"].mean())), 3))
N_BEAT = sum(1 for t in TICKERS if DESK[t][1] > DESK[t][2])
DESK_BEST, DESK_WORST = max(TICKERS, key=lambda t: DESK[t][0]), min(TICKERS, key=lambda t: DESK[t][0])
BUCKETS = []
for lo, hi in [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01)]:
    _m = (P_ONE >= lo) & (P_ONE < hi)
    BUCKETS.append((lo, hi, int(_m.sum()), float(Y_TE.values[_m].mean()) if _m.sum() else float("nan")))
P_ONE_MIN, P_ONE_MAX = float(P_ONE.min()), float(P_ONE.max())

# ---------------------------------------------------------------- top matter
md(
"# \U0001f4bc The Analyst's Notebook · Part 8\n"
"### A label for the risk report\n\n"
"Part 6 ended with every column the desk could offer, a ridge penalty chosen "
"on the folds, and a verdict that differed on Apple and on the desk: the wide "
f"model beat the single column on {P6_WINS} of the 11 instruments, and Apple "
"was not one of them.\n\n"
"Part 8 changes the question rather than the columns. The desk does not act "
"on a volatility forecast directly; it acts on whether next month will be "
"busier than this one, because that is when protection has to be bought. "
"That is a label, 1 or 0, and this part fits the first classifier to it: a "
"straight line, which fails; logistic regression on the raw decimal column, "
"which fails in a way that Part 6 prepared you for; and logistic regression "
"on a standardised column, which works. Then the four counts, the AUC, the "
"folds, the 20 columns with `C`, and the desk."
)

md(
"## How to work through this\n\n"
"- Run the **quick load** cell first. It brings back what Part 6 established and "
"loads the price table.\n"
"- Each question builds on the last, so keep them in order and keep your "
"variables. Later questions use the names earlier ones created.\n"
"- Cells with `...` are blanks. The notebook runs cleanly even before you fill "
"them in, so **Run all** is always safe.\n"
"- Hints and solutions are folded under each question. Work first, then check.\n\n"
"**A note on units.** The lecture worked in percent. This notebook keeps the "
"plain decimals of Parts 1 to 6. The label itself has no units, since a "
"comparison is the same in any units, but `LogisticRegression` has a penalty "
"on by default, and a penalty always has units. Q4 and Q5 are about that.\n\n"
"*Stuck for more than 15 minutes? Ask a friend, ask an AI for a hint (not the "
"answer), or email me at `jobo@econ.au.dk`.*"
)

md("---")

# ------------------------------------------------------------- quick load
md(
"## ⚙️ Quick load\n\n"
"The packages, the price table, and what Part 6 left you. Run it and read what "
"it prints."
)

code(
'''import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error, accuracy_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
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
    """Root mean squared error, as in Part 6, as a plain number."""
    return float(np.sqrt(mean_squared_error(actual, predicted)))


# The whole universe: eleven instruments, 2015 to 2024, returns in plain decimals
prices = pd.read_csv(data_path("prices.csv"), parse_dates=["date"])
wide = prices.pivot(index="date", columns="ticker", values="close")
rets = wide.pct_change()
TICKERS = sorted(prices["ticker"].unique())

folds = TimeSeriesSplit(n_splits=5)

# --- What Part 6 established ---
part6_target = "sd of daily returns over the next 20 trading days"
part6_columns = 20                                   # Apple's own windows, up_20d, the other ten instruments' volatility
part6_split = "by date: train to 2022-12-31, test from 2023-01-01"
part6_one_rmse = ''' + f"{P6_ONE:.5f}" + '''          # one column, vol_20d, on the test block
part6_ridge_rmse = ''' + f"{P6_RIDGE:.5f}" + '''        # twenty columns, ridge with alpha chosen on the folds
part6_alpha = ''' + f"{P6_ALPHA}" + '''                   # the alpha the folds chose
part6_desk_wins = ''' + f"{P6_WINS}" + '''                 # instruments where the wide model beat one column, of 11

print("Loaded prices:", prices.shape[0], "rows")
print("Instruments  :", ", ".join(TICKERS))
print()
print("Part 6 left you a wide model and a verdict:")
print("  target  :", part6_target)
print("  columns :", part6_columns, "with alpha", part6_alpha, "chosen on the folds")
print("  split   :", part6_split)
print(f"  test RMSE {part6_ridge_rmse:.5f} for the wide model, {part6_one_rmse:.5f} for one column")
print(f"  the wide model wins on {part6_desk_wins} of {len(TICKERS)} instruments, and not on Apple")
print()
print("Today the target becomes a label.")'''
)

md("---")

# ==================================================================== Q1
q("Q1", "Where Part 6 stopped",
  "Rebuild Part 6's wide table for Apple as `table`: the six volatility windows "
  "`[5, 10, 20, 40, 60, 120]` as `vol_<w>d`, the three return windows "
  "`[5, 20, 60]` as `ret_<w>d`, `up_20d`, the 20-day volatility of every "
  "**other** instrument as `<ticker>_vol`, the target `vol_next`, and "
  "incomplete rows dropped. Split it at the end of 2022 into `train` and `test`, "
  "store the twenty feature names in `columns`, refit the one-column regression "
  "and check its test RMSE matches `part6_one_rmse`.",
  "table = pd.DataFrame()\n\nfor w in [5, 10, 20, 40, 60, 120]:\n    ...\n\n"
  "for w in [5, 20, 60]:\n    ...\n\ntable['up_20d'] = ...\n\n"
  "for t in TICKERS:\n    ...\n\ntable['vol_next'] = ...\ntable = ...\n\n"
  "train = ...\ntest = ...\ncolumns = ...\n\n"
  "one_model = LinearRegression()\n...\ncheck = ...\nprint(check)\nprint('matches Part 6:', ...)",
  ["Column names are text built from the number: `'vol_' + str(w) + 'd'`. "
   "Inside the last loop, `if t != 'AAPL':`. `up_20d` is `(rets['AAPL'] > 0).rolling(20).mean()`.",
   "`columns = list(table.columns[:-1])`. Fit on `train[['vol_20d']]`, score on "
   "`test`, and `abs(check - part6_one_rmse) < 0.00001` is the check."],
  "table = pd.DataFrame()\n\nfor w in [5, 10, 20, 40, 60, 120]:\n"
  "    table['vol_' + str(w) + 'd'] = rets['AAPL'].rolling(w).std()\n\n"
  "for w in [5, 20, 60]:\n    table['ret_' + str(w) + 'd'] = rets['AAPL'].rolling(w).mean()\n\n"
  "table['up_20d'] = (rets['AAPL'] > 0).rolling(20).mean()\n\n"
  "for t in TICKERS:\n    if t != 'AAPL':\n        table[t + '_vol'] = rets[t].rolling(20).std()\n\n"
  "table['vol_next'] = rets['AAPL'].rolling(20).std().shift(-20)\ntable = table.dropna()\n\n"
  "train = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\ncolumns = list(table.columns[:-1])\n\n"
  "one_model = LinearRegression()\none_model.fit(train[['vol_20d']], train['vol_next'])\n"
  "check = rmse(test['vol_next'], one_model.predict(test[['vol_20d']]))\nprint(check)\n"
  "print('matches Part 6:', abs(check - part6_one_rmse) < 0.00001)",
  f"{P6_ONE:.5f}, and `True`: {N_TBL:,} rows, {N_TRAIN:,} to fit on and "
  f"{N_TEST} to check on, the same as Part 6. The regression stays in the "
  "notebook because the label is made from the same two columns it used.")

# ==================================================================== Q2
q("Q2", "The label",
  "Add `rising` to `table`: 1 where `vol_next` is larger than `vol_20d`, 0 "
  "otherwise. Split again so that `train` and `test` carry the label, print the "
  "share of days with a rise in each, and store the accuracy of predicting 0 on "
  "every test day as `majority`.",
  "# add the label to table, then split again\n...\ntrain = ...\ntest = ...\n\n"
  "print('train:', ...)\nprint('test :', ...)\nmajority = ...\nprint('majority rule:', majority)",
  ["`(table['vol_next'] > table['vol_20d']).astype(int)`: the comparison answers "
   "`True` or `False` on every row, and `.astype(int)` makes it 1 or 0.",
   "The share is the mean of the label. Predicting 0 everywhere is right on the "
   "days whose label is 0, so `majority = 1 - test['rising'].mean()`; write it "
   "as `max(share, 1 - share)` if you want the rule that also works when the 1s "
   "are the majority."],
  "table['rising'] = (table['vol_next'] > table['vol_20d']).astype(int)\n"
  "train = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\n\n"
  "print('train:', round(train['rising'].mean(), 4))\nprint('test :', round(test['rising'].mean(), 4))\n"
  "majority = max(test['rising'].mean(), 1 - test['rising'].mean())\nprint('majority rule:', round(majority, 4))",
  f"{SHARE_TR:.3f} of the training days and {SHARE_TE:.3f} of the test days are "
  f"followed by a rise, so predicting 0 everywhere scores {MAJORITY:.3f} on the "
  "test rows. The label is unit-free: multiplying both columns by 100 changes "
  "nothing about which is larger. Every model from here has to beat "
  "`majority`.")

# ==================================================================== Q3
q("Q3", "A straight line on the label",
  "Fit `LinearRegression` on `vol_20d` with `rising` as the target as `line`. "
  "Count the training days whose fitted value falls below 0 or above 1, and "
  "print the fitted value for the most volatile training day.",
  "line = LinearRegression()\n...\nfitted = ...\n\n"
  "print('outside [0, 1]:', ...)\nprint('most volatile day:', ...)",
  ["`fitted = line.predict(train[['vol_20d']])`; the count is "
   "`((fitted < 0) | (fitted > 1)).sum()`.",
   "`train['vol_20d'].max()` is the largest volatility; predict at it with a "
   "one-row DataFrame, `pd.DataFrame({'vol_20d': [train['vol_20d'].max()]})`; "
   "`[0]` takes the single number out of the array `predict` returns."],
  "line = LinearRegression()\nline.fit(train[['vol_20d']], train['rising'])\n"
  "fitted = line.predict(train[['vol_20d']])\n\n"
  "print('outside [0, 1]:', ((fitted < 0) | (fitted > 1)).sum())\n"
  "print('most volatile day:', line.predict(pd.DataFrame({'vol_20d': [train['vol_20d'].max()]}))[0])",
  f"{N_OUTSIDE} training days get a fitted value below zero, and the most "
  f"volatile day, {MAX_VOL_DAY}, at a daily volatility of {MAX_VOL:.3f}, gets "
  f"{FIT_AT_MAX:.2f}. Read as a probability, that is nonsense, and the line has "
  "no way to stop at the edge of the band.")

# ==================================================================== Q4
q("Q4", "Logistic regression on the raw column",
  "Fit `LogisticRegression()` on `vol_20d` as `raw_model`. Print its intercept "
  "and coefficient, the volatility at which its probability crosses one half, "
  "the share of test days on which it predicts a rise, and its test accuracy "
  "next to `majority`. Something is wrong; say what.",
  "raw_model = LogisticRegression()\n...\n\nprint(..., ...)\nprint('crosses one half at:', ...)\n"
  "raw_predicted = ...\nprint('share predicted as a rise:', ...)\nprint('accuracy:', ..., ' majority:', majority)",
  ["The crossing is `-raw_model.intercept_[0] / raw_model.coef_[0, 0]`.",
   "`raw_predicted.mean()` is the share of ones; `accuracy_score(test['rising'], raw_predicted)`."],
  "raw_model = LogisticRegression()\nraw_model.fit(train[['vol_20d']], train['rising'])\n\n"
  "print(raw_model.intercept_, raw_model.coef_)\n"
  "print('crosses one half at:', -raw_model.intercept_[0] / raw_model.coef_[0, 0])\n"
  "raw_predicted = raw_model.predict(test[['vol_20d']])\n"
  "print('share predicted as a rise:', raw_predicted.mean())\n"
  "print('accuracy:', accuracy_score(test['rising'], raw_predicted), ' majority:', round(majority, 4))",
  f"A coefficient of {RAW_B1:.2f}, a crossing at a volatility of {RAW_CROSS:.4f}, "
  f"which no test day reaches, so the model predicts a rise on **every** test "
  f"day: {RAW_SHARE_1:.0%} of them. Its accuracy is {RAW_ACC:.3f}, which is one "
  f"minus `majority`, the share of days with a rise. Every test probability lies "
  f"between {RAW_P_MIN:.3f} and {RAW_P_MAX:.3f}. The model has barely moved off "
  "the intercept.")

# ==================================================================== Q5
q("Q5", "Why: the penalty has units",
  "`LogisticRegression` has a ridge penalty on by default, at `C=1`. Fit the "
  "same model with the penalty switched off in effect, `C=1000000`, as "
  "`free_model`, and print its coefficient, its crossing and its test accuracy. "
  "Then explain the difference from Q4 in two sentences, using what Part 6 "
  "found about alpha on decimal columns.",
  "free_model = LogisticRegression(C=1000000)\n...\n\nprint(..., ...)\nprint('crosses one half at:', ...)\nprint('accuracy:', ...)",
  ["A large `C` is a weak penalty: `C = 1/alpha`.",
   "In decimals the column's values are around 0.01, a hundred times smaller "
   "than in percent, so the coefficient has to be a hundred times larger for "
   "the same curve, and the penalty charges its square: ten thousand times more."],
  "free_model = LogisticRegression(C=1000000)\nfree_model.fit(train[['vol_20d']], train['rising'])\n\n"
  "print(free_model.intercept_, free_model.coef_)\n"
  "print('crosses one half at:', -free_model.intercept_[0] / free_model.coef_[0, 0])\n"
  "print('accuracy:', accuracy_score(test['rising'], free_model.predict(test[['vol_20d']])))",
  f"A coefficient of {FREE_B1:.1f}, a crossing at {FREE_CROSS:.4f} (a daily "
  f"volatility of {100 * FREE_CROSS:.2f} percent), and an accuracy of "
  f"{FREE_ACC:.3f}, above `majority`. In percent the coefficient would be "
  f"{PCT_B1:.2f}; in decimals it has to be {FREE_B1:.0f}, and the default "
  f"penalty charges for {FREE_B1:.0f} squared, so at `C=1` it shrinks the "
  f"coefficient to {RAW_B1:.1f} and the curve flattens onto the intercept. This "
  "is Q6 of Part 6 again: the label has no units, the penalty does.")

# ==================================================================== Q6
q("Q6", "Standardise, then classify",
  "Build a `Pipeline` with a `StandardScaler` step `'scale'` and a "
  "`LogisticRegression()` step `'logit'`, fit it on `vol_20d` as `one_pipe`, and "
  "store the test probabilities of a rise as `p_one`. Print the coefficient per "
  "standard deviation, the test AUC as `auc_one`, and the test accuracy as "
  "`acc_one`. Then print the AUC of `raw_model` from Q4 as well.",
  "one_pipe = Pipeline([...])\n...\np_one = ...\n\n"
  "print('per sd:', ...)\nauc_one = ...\nacc_one = ...\n"
  "print('AUC:', auc_one, ' accuracy:', acc_one)\nprint('AUC of the raw model:', ...)",
  ["`one_pipe.named_steps['logit'].coef_[0, 0]` is the coefficient; "
   "`one_pipe.predict_proba(test[['vol_20d']])[:, 1]` the probabilities.",
   "`roc_auc_score(test['rising'], p_one)` and `accuracy_score(test['rising'], "
   "one_pipe.predict(test[['vol_20d']]))`. The raw model's AUC needs its own "
   "`predict_proba`."],
  "one_pipe = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
  "one_pipe.fit(train[['vol_20d']], train['rising'])\np_one = one_pipe.predict_proba(test[['vol_20d']])[:, 1]\n\n"
  "print('per sd:', one_pipe.named_steps['logit'].coef_[0, 0])\n"
  "auc_one = roc_auc_score(test['rising'], p_one)\n"
  "acc_one = accuracy_score(test['rising'], one_pipe.predict(test[['vol_20d']]))\n"
  "print('AUC:', auc_one, ' accuracy:', acc_one)\n"
  "print('AUC of the raw model:', roc_auc_score(test['rising'], raw_model.predict_proba(test[['vol_20d']])[:, 1]))",
  f"A coefficient of {SD_B1:.2f} per standard deviation, an AUC of {AUC_ONE:.3f} "
  f"and an accuracy of {ACC_ONE:.3f}, above `majority`. The raw model's AUC is "
  f"{AUC_RAW:.3f}, the same number. The AUC only asks whether the days are "
  "ranked in the right order, and a coefficient of −2.8 ranks them exactly as "
  "−130 does; it is the threshold at one half that the crushed coefficient "
  "broke. Standardising inside the pipeline is the fix, as it was for ridge.")

# ==================================================================== Q7
q("Q7", "The four kinds of day",
  "Turn `p_one` into predictions at one half, build the masks `rose` and "
  "`pred_rise`, and count the four kinds of test day. Then print "
  "`confusion_matrix` as `cm`, and the precision and recall. For a desk that "
  "buys protection on every predicted rise, say in one sentence what the two "
  "rates mean.",
  "predicted = ...\nrose = ...\npred_rise = ...\n\n"
  "print(..., ..., ..., ...)\n"
  "cm = ...\nprint(cm)\nprint('precision:', ...)\nprint('recall   :', ...)",
  ["`predicted = (p_one >= 0.5).astype(int)`, `rose = test['rising'].values == 1`, "
   "`pred_rise = predicted == 1`. The four counts are `(pred_rise & rose).sum()`, "
   "`(pred_rise & ~rose).sum()`, `(~pred_rise & rose).sum()` and "
   "`(~pred_rise & ~rose).sum()`.",
   "`confusion_matrix(test['rising'], predicted)`, then `precision_score` and "
   "`recall_score` with the true labels first."],
  "predicted = (p_one >= 0.5).astype(int)\nrose = test['rising'].values == 1\npred_rise = predicted == 1\n\n"
  "print((pred_rise & rose).sum(), (pred_rise & ~rose).sum(), (~pred_rise & rose).sum(), (~pred_rise & ~rose).sum())\n"
  "cm = confusion_matrix(test['rising'], predicted)\nprint(cm)\n"
  "print('precision:', precision_score(test['rising'], predicted))\n"
  "print('recall   :', recall_score(test['rising'], predicted))",
  f"{TP} predicted rises that came, {FP} that did not, {FN} rises missed and "
  f"{TN} calm days predicted as calm. The model predicts a rise on "
  f"{SHARE_PRED_1:.0%} of the test days: recall {REC:.2f}, precision {PREC:.2f}. "
  "Protection bought on every predicted rise would have been in place for "
  f"{REC:.0%} of the rises, and unnecessary on {1 - PREC:.0%} of the days it "
  "was bought. On Apple the crossing point sits above almost the whole of "
  "2023 and 2024, so at one half the model says \"rise\" nearly always.")

# ==================================================================== Q8
q("Q8", "The threshold",
  "Loop over the thresholds `[0.4, 0.5, 0.6, 0.7]`: turn `p_one` into "
  "predictions at each, and store the precision, the recall and the accuracy "
  "as a tuple in a dictionary `by_threshold`. Print it. Which threshold has "
  "the highest accuracy, and why is that not yet a choice?",
  "by_threshold = {}\n\nfor threshold in [0.4, 0.5, 0.6, 0.7]:\n    ...\n\n"
  "for threshold in by_threshold:\n    print(threshold, by_threshold[threshold])",
  ["Inside the loop, `pred_t = (p_one >= threshold).astype(int)` and the three "
   "functions on it, each wrapped in `round(float(...), 3)` and put in a tuple: "
   "`(precision, recall, accuracy)`.",
   "`precision_score(..., zero_division=0)` keeps a threshold that predicts no "
   "rise at all from printing a warning."],
  "by_threshold = {}\n\nfor threshold in [0.4, 0.5, 0.6, 0.7]:\n"
  "    pred_t = (p_one >= threshold).astype(int)\n"
  "    by_threshold[threshold] = (round(float(precision_score(test['rising'], pred_t, zero_division=0)), 3),\n"
  "                               round(float(recall_score(test['rising'], pred_t)), 3),\n"
  "                               round(float(accuracy_score(test['rising'], pred_t)), 3))\n\n"
  "for threshold in by_threshold:\n    print(threshold, by_threshold[threshold])",
  f"At 0.5 the accuracy is {BY_THR[0.5][2]:.3f}; at {BEST_THR_ACC} it is "
  f"{BY_THR[BEST_THR_ACC][2]:.3f}, with precision {BY_THR[BEST_THR_ACC][0]:.2f} "
  f"and recall {BY_THR[BEST_THR_ACC][1]:.2f}. The test rows say a higher "
  "threshold would have done better on Apple in 2023 and 2024, and that is "
  "exactly the kind of thing the test rows are not allowed to decide. The "
  "threshold is a setting, and the next part chooses it from the cost of each "
  "error, on the training rows.")

# ==================================================================== Q9
q("Q9", "The AUC on the folds",
  "Cross-validate the one-column pipeline (a fresh one, unfitted) on the "
  "training rows with `folds` and `scoring='roc_auc'` as `auc_folds`. Print the "
  "five scores and their mean next to `auc_one`.",
  "auc_folds = ...\nprint(...)\nprint('folds:', ..., ' test:', auc_one)",
  "`cross_val_score(Pipeline([...]), train[['vol_20d']], train['rising'], "
  "cv=folds, scoring='roc_auc')`. Larger is better, so no minus sign.",
  "auc_folds = cross_val_score(Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())]),\n"
  "                            train[['vol_20d']], train['rising'], cv=folds, scoring='roc_auc')\n"
  "print(auc_folds.round(3))\nprint('folds:', round(auc_folds.mean(), 4), ' test:', round(auc_one, 4))",
  f"A mean of {AUC_FOLDS.mean():.3f} on the folds, from {AUC_FOLDS.min():.2f} "
  f"to {AUC_FOLDS.max():.2f}, and {AUC_ONE:.3f} on the test rows. The fold "
  "spread is the size of the uncertainty around any one of these numbers.")

# ==================================================================== Q10
q("Q10", "Twenty columns, and C",
  "Cross-validate a scaling pipeline with `LogisticRegression(max_iter=1000)` "
  "on all twenty `columns` and print the mean AUC next to the one-column mean "
  "from Q9. Then search `{'logit__C': [0.0001, 0.001, 0.01, 0.1, 1, 10]}` with "
  "`GridSearchCV` as `search`, print the best C and score, and store the test "
  "AUC of the search as `auc_wide`.",
  "wide_pipe = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(max_iter=1000))])\n"
  "wide_scores = ...\nprint('twenty columns:', ..., ' one column:', ...)\n\n"
  "grid = ...\nsearch = ...\n...\nprint(...)\n\n"
  "auc_wide = ...\nprint('test AUC, twenty columns:', auc_wide, ' one column:', auc_one)",
  ["`cross_val_score(wide_pipe, train[columns], train['rising'], cv=folds, scoring='roc_auc')`.",
   "`GridSearchCV(wide_pipe, grid, cv=folds, scoring='roc_auc')`, then "
   "`search.predict_proba(test[columns])[:, 1]` for the test AUC."],
  "wide_pipe = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(max_iter=1000))])\n"
  "wide_scores = cross_val_score(wide_pipe, train[columns], train['rising'], cv=folds, scoring='roc_auc')\n"
  "print('twenty columns:', round(wide_scores.mean(), 4), ' one column:', round(auc_folds.mean(), 4))\n\n"
  "grid = {'logit__C': [0.0001, 0.001, 0.01, 0.1, 1, 10]}\n"
  "search = GridSearchCV(wide_pipe, grid, cv=folds, scoring='roc_auc')\n"
  "search.fit(train[columns], train['rising'])\nprint(search.best_params_, search.best_score_)\n\n"
  "auc_wide = roc_auc_score(test['rising'], search.predict_proba(test[columns])[:, 1])\n"
  "print('test AUC, twenty columns:', auc_wide, ' one column:', auc_one)",
  f"Twenty columns score {CV_WIDE.mean():.3f} on the folds against "
  f"{AUC_FOLDS.mean():.3f} for one. The grid picks C = {BEST_C} at "
  f"{BEST_CV:.3f}, and on the test rows the wide model scores {AUC_WIDE:.3f} "
  f"against {AUC_ONE:.3f} for one column. The verdict of Part 6 holds for the "
  "label: on Apple the extra columns do not help, penalty or no penalty.")

# ==================================================================== Q11
q("Q11", "Which columns the l1 penalty keeps",
  "Fit a scaling pipeline with `LogisticRegression(penalty='l1', "
  "solver='liblinear', C=0.01)` on all twenty columns, and store the names of "
  "the columns with a non-zero coefficient as the list `kept`. Print it and the "
  "test AUC.",
  "sparse = Pipeline([...])\n...\n\nkept = []\n...\nprint(kept)\nprint('test AUC:', ...)",
  ["`sparse.named_steps['logit'].coef_[0]` is the one row of coefficients; "
   "loop over `zip(columns, coefs)` and append the name when `b != 0`.",
   "`roc_auc_score(test['rising'], sparse.predict_proba(test[columns])[:, 1])`."],
  "sparse = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(penalty='l1', solver='liblinear', C=0.01))])\n"
  "sparse.fit(train[columns], train['rising'])\n\n"
  "kept = []\nfor name, b in zip(columns, sparse.named_steps['logit'].coef_[0]):\n"
  "    if b != 0:\n        kept.append(name)\nprint(kept)\n"
  "print('test AUC:', roc_auc_score(test['rising'], sparse.predict_proba(test[columns])[:, 1]))",
  f"{', '.join('`' + c + '`' for c in KEPT)}, and an AUC of {AUC_L1:.3f}. Of "
  "twenty columns the penalty keeps the one the lecture started with and the "
  "last week's return, which is the same pair it kept on the index. The l1 "
  "penalty on a classifier reads like the lasso on a regression.")

# ==================================================================== Q12
q("Q12", "Across the desk",
  "Write `classifier_vs_majority(ticker)`: build the two columns `vol_20d` and "
  "`vol_next` for that ticker from `rets`, drop incomplete rows, add the label, "
  "split at the end of 2022, fit the one-column scaling pipeline, and return "
  "the triple `(auc, accuracy, majority)` on the test rows. Run it for every "
  "ticker into a dictionary `desk`, and count the instruments on which the "
  "accuracy beats the majority rule.",
  "def classifier_vs_majority(ticker):\n    ...\n\ndesk = {}\nfor ticker in TICKERS:\n    ...\n\n"
  "beats = ...\nprint('beats the majority rule on', beats, 'of', len(desk))",
  ["Inside: `now = rets[ticker].rolling(20).std()`, then "
   "`frame = pd.DataFrame({'vol_20d': now, 'vol_next': now.shift(-20)}).dropna()`, "
   "the label, the split, and the pipeline from Q6 with `tr` and `te`.",
   "`sum(1 for t in desk if desk[t][1] > desk[t][2])` counts the instruments "
   "whose accuracy (second entry) beats the majority rule (third entry)."],
  "def classifier_vs_majority(ticker):\n"
  "    now = rets[ticker].rolling(20).std()\n"
  "    frame = pd.DataFrame({'vol_20d': now, 'vol_next': now.shift(-20)}).dropna()\n"
  "    frame['rising'] = (frame['vol_next'] > frame['vol_20d']).astype(int)\n"
  "    tr = frame.loc[:'2022-12-31']\n    te = frame.loc['2023-01-01':]\n\n"
  "    pipe_t = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
  "    pipe_t.fit(tr[['vol_20d']], tr['rising'])\n"
  "    auc = roc_auc_score(te['rising'], pipe_t.predict_proba(te[['vol_20d']])[:, 1])\n"
  "    accuracy = accuracy_score(te['rising'], pipe_t.predict(te[['vol_20d']]))\n"
  "    majority_t = max(te['rising'].mean(), 1 - te['rising'].mean())\n"
  "    return (round(auc, 3), round(accuracy, 3), round(majority_t, 3))\n\n"
  "desk = {}\nfor ticker in TICKERS:\n    desk[ticker] = classifier_vs_majority(ticker)\n\n"
  "beats = sum(1 for t in desk if desk[t][1] > desk[t][2])\n"
  "print('beats the majority rule on', beats, 'of', len(desk))",
  f"{N_BEAT} of {len(TICKERS)}. The AUC runs from {DESK[DESK_WORST][0]:.2f} on "
  f"{DESK_WORST} to {DESK[DESK_BEST][0]:.2f} on {DESK_BEST}, and Apple, at "
  f"{DESK['AAPL'][0]:.2f}, is among the lower ones. Unlike the wide regression "
  "of Part 6, the one-column classifier beats its baseline everywhere: whether "
  "next month is busier than this one is predictable on every instrument, "
  "because volatility mean-reverts on every instrument.")

# ==================================================================== Q13
q("Q13", "Does 0.6 mean 60 percent",
  "Group the test days by the probability `p_one` gave them: below 0.4, 0.4 to "
  "0.5, 0.5 to 0.6, and 0.6 and above. For each group, count the days and "
  "compute the share that actually rose, and store the results as a list of "
  "tuples `buckets`. Print them. Do the probabilities mean what they say?",
  "edges = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01)]\nbuckets = []\n\nfor low, high in edges:\n    ...\n\n"
  "for row in buckets:\n    print(row)",
  ["`mask = (p_one >= low) & (p_one < high)`; the count is `mask.sum()` and the "
   "share that rose is `test['rising'].values[mask].mean()`.",
   "Append `(low, high, int(mask.sum()), round(float(share), 3))`."],
  "edges = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01)]\nbuckets = []\n\nfor low, high in edges:\n"
  "    mask = (p_one >= low) & (p_one < high)\n"
  "    share = test['rising'].values[mask].mean()\n"
  "    buckets.append((low, high, int(mask.sum()), round(float(share), 3)))\n\n"
  "for row in buckets:\n    print(row)",
  f"Below 0.4, {BUCKETS[0][2]} days and {BUCKETS[0][3]:.0%} rose; between 0.5 "
  f"and 0.6, {BUCKETS[2][2]} days and only {BUCKETS[2][3]:.0%} rose; at 0.6 and "
  f"above, {BUCKETS[3][2]} days and {BUCKETS[3][3]:.0%} rose. The ranking is "
  "right, the levels are not: on Apple in 2023 and 2024 the probabilities run "
  "too high, because the training years were more volatile than the test "
  "years and the curve was fitted to them. A probability from a model is a "
  "claim to check, not a fact.")

# ==================================================================== Q14
q("Q14", "Draw it",
  "One figure of the test years: `p_one` as a line, a horizontal line at one "
  "half, and a dot at the top of the chart on every day that was followed by "
  "a rise. Label the axes and give it a title.",
  "fig, ax = plt.subplots(figsize=(10, 3.5))\n...\nplt.show()",
  ["`ax.plot(test.index, p_one, label='probability of a rise')` and "
   "`ax.axhline(0.5, color='grey', linestyle='--')`.",
   "`rose = test['rising'].values == 1`, then "
   "`ax.scatter(test.index[rose], np.full(rose.sum(), 1.0), s=6, label='a rise came')`."],
  "fig, ax = plt.subplots(figsize=(10, 3.5))\n"
  "ax.plot(test.index, p_one, label='probability of a rise')\n"
  "ax.axhline(0.5, color='grey', linestyle='--')\n"
  "rose = test['rising'].values == 1\n"
  "ax.scatter(test.index[rose], np.full(rose.sum(), 1.0), s=6, color='black', label='a rise came')\n"
  "ax.set_ylabel('probability')\nax.legend(loc='lower left')\n"
  "ax.set_title('Apple, 2023 to 2024: the probability of a rise, and the rises', loc='left')\n"
  "plt.show()",
  "The line sits above one half for most of the two years, and the dots come in "
  "stretches: rises cluster, as calm does. The stretches where the line is "
  "high and the dots are absent are the false alarms of Q7, drawn in time.")

# ==================================================================== Q15
q("Q15", "Write down what you would defend",
  "Finish the way Parts 5 and 6 finished: one dictionary and a function that "
  "prints it with a verdict. Fill in `report`, then write `summarise(report)`: "
  "it prints each entry on its own line and ends with one sentence on whether "
  "the classifier beats the majority rule **on Apple**, and one on the desk as "
  "a whole.",
  "report = {\n    'label': ...,\n    'share_of_rises_test': ...,\n    'majority_rule': ...,\n"
  "    'one_column_auc': ...,\n    'one_column_accuracy': ...,\n    'wide_auc': ...,\n"
  "    'C': ...,\n    'threshold': ...,\n    'desk_beats_majority': ...,\n}\n\n"
  "def summarise(report):\n    ...\n\nsummarise(report)",
  ["Most values are already in variables: `majority`, `auc_one`, `acc_one`, "
   "`auc_wide`, `search.best_params_['logit__C']`, `beats` from Q12. The "
   "threshold is `0.5`, with a note that it was not chosen.",
   "Inside the function, `for key in report:` then "
   "`print(f'{key:22} {report[key]}')`. The verdicts are an `if` on "
   "`one_column_accuracy > majority_rule` and a sentence with `desk_beats_majority`."],
  "report = {\n    'label': 'volatility over the next 20 days above the last 20',\n"
  "    'share_of_rises_test': round(test['rising'].mean(), 3),\n"
  "    'majority_rule': round(majority, 3),\n"
  "    'one_column_auc': round(auc_one, 3),\n    'one_column_accuracy': round(acc_one, 3),\n"
  "    'wide_auc': round(auc_wide, 3),\n    'C': search.best_params_['logit__C'],\n"
  "    'threshold': '0.5, the default, not yet chosen',\n"
  "    'desk_beats_majority': f'{beats} of {len(desk)}',\n}\n\n"
  "def summarise(report):\n"
  "    \"\"\"Print a finished comparison, and say what the classifier is worth.\"\"\"\n"
  "    for key in report:\n        print(f'{key:22} {report[key]}')\n\n"
  "    if report['one_column_accuracy'] > report['majority_rule']:\n"
  "        print('\\nOn Apple, the one-column classifier beats the majority rule; the twenty columns do not add to it.')\n"
  "    else:\n"
  "        print('\\nOn Apple, the classifier does not beat the majority rule at this threshold.')\n"
  "    print(f\"Across the desk, it beats the majority rule on {report['desk_beats_majority']} instruments.\")\n\n"
  "summarise(report)",
  "Nine lines and two verdicts, and this time they point the same way. The "
  "report carries the threshold as an open item on purpose: the model's "
  "probabilities rank the days well and sit too high, so what the desk does "
  "with a 0.55 is a decision about costs, and that is the next part.")

# ---------------------------------------------------------------- closing
md(
"## \U0001f9ed What you have now\n\n"
"| what | where it lives |\n"
"|:--|:--|\n"
"| Part 6's table and split, rebuilt | `table`, `train`, `test`, `columns`, `one_model` |\n"
"| the label and the share to beat | `table['rising']`, `majority` |\n"
"| the straight line that fails | `line` |\n"
"| logistic regression on the raw column, and without its penalty | `raw_model`, `free_model` |\n"
"| the standardised one-column classifier and its probabilities | `one_pipe`, `p_one`, `auc_one`, `acc_one` |\n"
"| the four counts | `cm` |\n"
"| the thresholds | `by_threshold` |\n"
"| the folds | `auc_folds` |\n"
"| twenty columns with C chosen | `search`, `auc_wide` |\n"
"| the l1 penalty's columns | `kept` |\n"
"| the whole desk | `desk`, `beats` |\n"
"| the probability check | `buckets` |\n"
"| the thing you would defend | `report` |"
)

md(
"## What changed since Part 6\n\n"
"- **The target became a label.** Whether next month is busier than this one, "
f"1 or 0, with a rise on {SHARE_TE:.0%} of the test days and a majority rule "
f"at {MAJORITY:.3f}.\n"
"- **The units problem came back in a new form.** The label has no units, but "
f"the default penalty does: on the raw decimal column it shrank the coefficient "
f"from {FREE_B1:.0f} to {RAW_B1:.1f} and the model predicted a rise on every "
"day (Q4 and Q5). Standardising in a pipeline repaired it (Q6), and the AUC, "
"which only ranks, never noticed.\n"
f"- **The one-column classifier beats its baseline everywhere.** {ACC_ONE:.3f} "
f"against {MAJORITY:.3f} on Apple, and on {N_BEAT} of {len(TICKERS)} "
"instruments across the desk (Q12).\n"
f"- **The twenty columns still do not help on Apple.** {AUC_WIDE:.3f} against "
f"{AUC_ONE:.3f}, with C chosen on the folds (Q10); the l1 penalty keeps "
f"{' and '.join('`' + c + '`' for c in KEPT)} (Q11).\n"
"- **The probabilities rank well and run high.** In the 0.5 to 0.6 group only "
f"{BUCKETS[2][3]:.0%} of the days rose (Q13), so the threshold at one half "
"predicts a rise far too often on these two years."
)

md(
"## Where this leaves the risk report\n\n"
"The report now answers a yes/no question with a probability, and it can say "
"how well that probability ranks the days: an AUC of about 0.8 on Apple and "
"between 0.74 and 0.90 across the desk. What it cannot yet say is what to do "
"with a probability of 0.55. At one half the model buys protection almost "
"every month; at 0.6 it buys it far less often and misses a quarter of the "
"rises. Which is right depends on what a false alarm costs and what a miss "
"costs.\n\n"
"**Next part:** the threshold as a decision, with the cost of each kind of "
"error, chosen on the training rows; and a label that is rare rather than "
"balanced."
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
