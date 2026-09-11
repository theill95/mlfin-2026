# -*- coding: utf-8 -*-
"""Build session_05_case.ipynb  (The Analyst's Notebook, Part 5).

Conventions (approved for Sessions 1 to 4): no star badges, one cumulative
investigation where later questions reuse what earlier ones stored, folded
hints and solutions, stated formulas, plain explanatory tone, no em-dashes.
Opens with a QUICK LOAD restoring Part 4's findings.

Part 4 stopped at a fully specified prediction problem: target, features, a
dated split, two baselines and a metric. Part 5 fits the first model of the
whole course, then spends most of its length on the harder question of whether
that model was chosen honestly. It closes by taking the chosen rule across all
eleven instruments, which is the test Part 4's persistence rule failed.

Returns stay in plain decimals here, as in Parts 1 to 4, so the numbers carry
forward. The lecture used percent; that difference is stated in the quick load.

BLANK-SAFE, and this one needs care because the case is cumulative: no
pre-written line may CALL anything on a variable an earlier question produced.
Every such dependency sits inside the student's own blank.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "session_05" / "session_05_case.ipynb"

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


def build(ticker):
    frame = pd.DataFrame({
        "vol_20d": R[ticker].rolling(20).std(),
        "ret_20d": R[ticker].rolling(20).mean(),
        "up_20d": (R[ticker] > 0).rolling(20).mean(),
    })
    frame["vol_next"] = R[ticker].rolling(20).std().shift(-20)
    return frame.dropna()


TBL = build("AAPL")
TRAIN, TEST = TBL.loc[:"2022-12-31"], TBL.loc["2023-01-01":]
N_TBL, N_TRAIN, N_TEST = len(TBL), len(TRAIN), len(TEST)

SET_A = ["vol_20d"]
SET_B = ["vol_20d", "ret_20d"]
SET_C = ["vol_20d", "ret_20d", "up_20d"]
SETS = {"A": SET_A, "B": SET_B, "C": SET_C}


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(np.asarray(y), np.asarray(p))))


# what Part 4 left behind
BASE_PRED = float(TRAIN["vol_next"].mean())
BASE_RMSE = rmse(TEST["vol_next"], np.full(N_TEST, BASE_PRED))
BASE_MAE = float((TEST["vol_next"] - BASE_PRED).abs().mean())
PERS_RMSE = rmse(TEST["vol_next"], TEST["vol_20d"])
PERS_R2 = float(1 - ((TEST["vol_next"] - TEST["vol_20d"]) ** 2).sum() /
                ((TEST["vol_next"] - BASE_PRED) ** 2).sum())

M1 = LinearRegression().fit(TRAIN[SET_A], TRAIN["vol_next"])
INTERCEPT, SLOPE = float(M1.intercept_), float(M1.coef_[0])
PRED = M1.predict(TEST[SET_A])
MODEL_RMSE = rmse(TEST["vol_next"], PRED)
MODEL_R2 = float(1 - ((TEST["vol_next"] - PRED) ** 2).sum() /
                 ((TEST["vol_next"] - BASE_PRED) ** 2).sum())

# Q15: does the advantage survive in the turbulent half?
ERR_MODEL = TEST["vol_next"] - PRED
ERR_PERS = TEST["vol_next"] - TEST["vol_20d"]
CALM = TEST["vol_20d"] < TEST["vol_20d"].median()
N_CALM = int(CALM.sum())


def _half(errors, mask):
    return float(np.sqrt((errors[mask] ** 2).mean()))


CALM_MODEL, CALM_PERS = _half(ERR_MODEL, CALM), _half(ERR_PERS, CALM)
BUSY_MODEL, BUSY_PERS = _half(ERR_MODEL, ~CALM), _half(ERR_PERS, ~CALM)
CALM_GAIN = 100 * (CALM_PERS - CALM_MODEL) / CALM_PERS
BUSY_GAIN = 100 * (BUSY_PERS - BUSY_MODEL) / BUSY_PERS

TR_RMSE = {k: rmse(TRAIN["vol_next"], LinearRegression().fit(TRAIN[v], TRAIN["vol_next"]).predict(TRAIN[v]))
           for k, v in SETS.items()}
TE_RMSE = {k: rmse(TEST["vol_next"], LinearRegression().fit(TRAIN[v], TRAIN["vol_next"]).predict(TEST[v]))
           for k, v in SETS.items()}

FIT20, VAL20 = TRAIN.loc[:"2020-12-31"], TRAIN.loc["2021-01-01":]
FIT19, VAL19 = TRAIN.loc[:"2019-12-31"], TRAIN.loc["2020-01-01":]
VAL20_RMSE = {k: rmse(VAL20["vol_next"], LinearRegression().fit(FIT20[v], FIT20["vol_next"]).predict(VAL20[v]))
              for k, v in SETS.items()}
VAL19_RMSE = {k: rmse(VAL19["vol_next"], LinearRegression().fit(FIT19[v], FIT19["vol_next"]).predict(VAL19[v]))
              for k, v in SETS.items()}

TS5 = TimeSeriesSplit(n_splits=5)
FOLDS_A = -cross_val_score(LinearRegression(), TRAIN[SET_A], TRAIN["vol_next"],
                           cv=TS5, scoring="neg_root_mean_squared_error")
CV_TS = {k: float(-cross_val_score(LinearRegression(), TRAIN[v], TRAIN["vol_next"],
                                   cv=TS5, scoring="neg_root_mean_squared_error").mean())
         for k, v in SETS.items()}
CV_KF = {k: float(-cross_val_score(LinearRegression(), TRAIN[v], TRAIN["vol_next"],
                                   cv=KFold(n_splits=5, shuffle=True, random_state=0),
                                   scoring="neg_root_mean_squared_error").mean())
         for k, v in SETS.items()}
CV_GAP = float(-cross_val_score(LinearRegression(), TRAIN[SET_A], TRAIN["vol_next"],
                                cv=TimeSeriesSplit(n_splits=5, gap=20),
                                scoring="neg_root_mean_squared_error").mean())
WORST_FOLD = int(np.argmax(FOLDS_A)) + 1
_wf = list(TS5.split(TRAIN))[WORST_FOLD - 1][1]
WORST_FROM, WORST_TO = TRAIN.index[_wf[0]].date(), TRAIN.index[_wf[-1]].date()

ACROSS = {}
for _t in TICKERS:
    _tb = build(_t)
    _tr, _te = _tb.loc[:"2022-12-31"], _tb.loc["2023-01-01":]
    _m = LinearRegression().fit(_tr[SET_A], _tr["vol_next"])
    ACROSS[_t] = (rmse(_te["vol_next"], _m.predict(_te[SET_A])),
                  rmse(_te["vol_next"], _te["vol_20d"]))
N_BEAT = sum(1 for t in TICKERS if ACROSS[t][0] < ACROSS[t][1])
BEST_GAIN = max(TICKERS, key=lambda t: (ACROSS[t][1] - ACROSS[t][0]) / ACROSS[t][1])
BEST_GAIN_PCT = 100 * (ACROSS[BEST_GAIN][1] - ACROSS[BEST_GAIN][0]) / ACROSS[BEST_GAIN][1]

# Part 4's persistence rule, per ticker, for the closing comparison
PERS_R2_BY = {}
for _t in TICKERS:
    _tb = build(_t)
    _tr, _te = _tb.loc[:"2022-12-31"], _tb.loc["2023-01-01":]
    _b = _tr["vol_next"].mean()
    PERS_R2_BY[_t] = float(1 - ((_te["vol_next"] - _te["vol_20d"]) ** 2).sum() /
                           ((_te["vol_next"] - _b) ** 2).sum())
N_PERS_POS = sum(1 for t in TICKERS if PERS_R2_BY[t] > 0)

# ---------------------------------------------------------------- top matter
md(
"# \U0001f4bc The Analyst's Notebook · Part 5\n"
"### The first model, and whether you are allowed to believe it\n\n"
"Part 4 finished with a question written down precisely enough to answer: a "
"target, three features that cannot see the future, a split by date, two "
"baselines and a metric. The one thing it never did was fit anything.\n\n"
"That is what Part 5 starts with, and it takes about four lines. The rest of "
"this notebook is the harder half: deciding which of several models to keep, "
"without letting the test years anywhere near the decision.\n\n"
"It ends with the test Part 4's simple rule failed. A rule that looked "
f"convincing on Apple was positive on only {N_PERS_POS} of the eleven "
"instruments. The model chosen here has to do better than that across the whole "
"desk, or it has not earned anything."
)

md(
"## How to work through this\n\n"
"- Run the **quick load** cell first. It brings back what Part 4 established and "
"loads the price table.\n"
"- Each question builds on the last, so keep them in order and keep your "
"variables. Later questions use the names earlier ones created.\n"
"- Cells with `...` are blanks. The notebook runs cleanly even before you fill "
"them in, so **Run all** is always safe.\n"
"- Hints and solutions are folded under each question. Work first, then check.\n\n"
"**A note on units.** The lecture worked in percent so the numbers read on a "
"slide. This notebook keeps the plain decimals of Parts 1 to 4, so an RMSE of "
"`0.004` means 0.4 percentage points of daily volatility. Everything else is "
"identical.\n\n"
"*Stuck for more than 15 minutes? Ask a friend, ask an AI for a hint (not the "
"answer), or email me at `jobo@econ.au.dk`.*"
)

md("---")

# ------------------------------------------------------------- quick load
md(
"## ⚙️ Quick load\n\n"
"The packages, the price table, and what Part 4 left you. Run it and read what "
"it prints."
)

code(
'''import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit

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


# The whole universe: eleven instruments, 2015 to 2024
prices = pd.read_csv(data_path("prices.csv"), parse_dates=["date"])
wide = prices.pivot(index="date", columns="ticker", values="close")
rets = wide.pct_change()
TICKERS = sorted(prices["ticker"].unique())

# --- What Part 4 established ---
part4_target = "sd of daily returns over the next 20 trading days"
part4_features = ["vol_20d", "ret_20d", "up_20d"]
part4_split = "by date: train to 2022-12-31, test from 2023-01-01"
part4_base_rmse = ''' + f"{BASE_RMSE:.5f}" + '''      # predict the training average
part4_pers_rmse = ''' + f"{PERS_RMSE:.5f}" + '''      # repeat the last 20 days
part4_pers_r2 = ''' + f"{PERS_R2:.3f}" + '''         # the bar a model has to clear

print("Loaded prices:", prices.shape[0], "rows")
print("Instruments  :", ", ".join(TICKERS))
print()
print("Part 4 left you a fully specified problem:")
print("  target  :", part4_target)
print("  features:", part4_features)
print("  split   :", part4_split)
print()
print("and two rules that use no model at all:")
print(f"  predict the average     RMSE {part4_base_rmse:.5f}")
print(f"  repeat the last 20 days RMSE {part4_pers_rmse:.5f}   R2 {part4_pers_r2:.3f}")
print()
print("Anything you build today has to beat both of those.")'''
)

md("---")

# ==================================================================== Q1
q("Q1", "Rebuild the table",
  "Start from where Part 4 finished. Build the learning table for Apple again: "
  "the three features, then the target, with incomplete rows dropped. Call it "
  "`table`.\n\n"
  "The features look back over the last twenty trading days. The target looks "
  "forward over the next twenty.\n\n"
  "$$\\text{vol\\_next}_t = \\text{sd}\\big(r_{t+1},\\, \\ldots,\\, r_{t+20}\\big)$$",
  "table = ...\ntable",
  ["The three features are `rets['AAPL'].rolling(20).std()`, "
   "`.rolling(20).mean()`, and `(rets['AAPL'] > 0).rolling(20).mean()`.",
   "The target is the first of those shifted backwards by twenty rows: "
   "`.shift(-20)`. Finish with `.dropna()`."],
  "table = pd.DataFrame({\n    'vol_20d': rets['AAPL'].rolling(20).std(),\n"
  "    'ret_20d': rets['AAPL'].rolling(20).mean(),\n"
  "    'up_20d': (rets['AAPL'] > 0).rolling(20).mean(),\n})\n"
  "table['vol_next'] = rets['AAPL'].rolling(20).std().shift(-20)\n"
  "table = table.dropna()\ntable",
  f"{N_TBL:,} rows and four columns, exactly as in Part 4. Twenty rows are lost at "
  "the start, where the window had not filled, and twenty at the end, where the "
  "target reaches past the last date in the file.")

# ==================================================================== Q2
q("Q2", "The split, unchanged",
  "Cut the table at the same date Part 4 chose: everything up to the end of 2022 "
  "for training, 2023 onwards for the test block. Call them `train` and `test`.",
  "train = ...\ntest = ...\n\nprint('train:', ...)\nprint('test :', ...)",
  "`table.loc[:'2022-12-31']` and `table.loc['2023-01-01':]`.",
  "train = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\n\n"
  "print('train:', len(train))\nprint('test :', len(test))",
  f"{N_TRAIN:,} training rows and {N_TEST} test rows.\n\n"
  "From here to Q13, the word `test` does not appear again. That is not an "
  "accident: every decision in between is made without it.")

# ==================================================================== Q3
q("Q3", "Fit the first model of the course",
  "Fit a linear regression that predicts `vol_next` from `vol_20d` alone, on the "
  "training rows. Then print what it learned.",
  "X_train = ...\ny_train = ...\n\nmodel = LinearRegression()\n\n...        # fit it\n\n"
  "print('intercept:', ...)\nprint('coefficient:', ...)",
  ["`X` needs a list of column names inside the brackets so it stays a table: "
   "`train[['vol_20d']]`. `y` is one column: `train['vol_next']`.",
   "`model.fit(X_train, y_train)`, then `model.intercept_` and `model.coef_`."],
  "X_train = train[['vol_20d']]\ny_train = train['vol_next']\n\n"
  "model = LinearRegression()\nmodel.fit(X_train, y_train)\n\n"
  "print('intercept:', model.intercept_)\nprint('coefficient:', model.coef_)",
  f"An intercept of about {INTERCEPT:.5f} and a slope of about {SLOPE:.3f}.\n\n"
  "Read the slope out loud. It is not 1. A month that is unusually volatile is "
  "followed by a month only about half as far from normal, which is mean "
  "reversion. Four sessions of setup, and the first model you fit tells you "
  "something true about markets.")

# ==================================================================== Q4
q("Q4", "Score it against Part 4's two rules",
  "Predict on the test block and compute the RMSE. Put it next to the two numbers "
  "the quick load restored.\n\n"
  "$$\\text{RMSE}=\\sqrt{\\tfrac{1}{n}\\sum_i (y_i-\\hat{y}_i)^2}$$",
  "predictions = ...\nmodel_rmse = ...\n\nprint('model      :', ...)\n"
  "print('persistence:', part4_pers_rmse)\nprint('average    :', part4_base_rmse)",
  ["`model.predict(test[['vol_20d']])`, with the same double brackets as the fit.",
   "`np.sqrt(mean_squared_error(test['vol_next'], predictions))`, true values "
   "first."],
  "predictions = model.predict(test[['vol_20d']])\n"
  "model_rmse = np.sqrt(mean_squared_error(test['vol_next'], predictions))\n\n"
  "print('model      :', round(model_rmse, 5))\n"
  "print('persistence:', part4_pers_rmse)\nprint('average    :', part4_base_rmse)",
  f"{MODEL_RMSE:.5f} against {PERS_RMSE:.5f} and {BASE_RMSE:.5f}. The model is best "
  f"of the three, by about {100 * (PERS_RMSE - MODEL_RMSE) / PERS_RMSE:.0f}% over "
  "persistence.\n\n"
  "Hold the celebration for now. That number came from the test rows, which is "
  "fine because no choice has been made yet. From Q6 onwards choices start, and "
  "then it stops being fine.")

# ==================================================================== Q5
q("Q5", "Clear the bar Part 4 set",
  "Part 4 measured the persistence rule at $R^2 = " + f"{PERS_R2:.3f}" + "$ and called "
  "it the bar. Compute the same $R^2$ for your model.\n\n"
  "$$R^2 = 1-\\frac{\\sum_i (y_i-\\hat{y}_i)^2}{\\sum_i (y_i-\\bar{y}_{\\text{train}})^2}$$\n\n"
  "The denominator uses the **training** mean, because that is what you would have "
  "predicted with no model at all.",
  "rss = ...\ntss = ...\nmodel_r2 = ...\n\nprint('model R2      :', ...)\n"
  "print('the bar (Part 4):', part4_pers_r2)",
  ["`rss` is `((test['vol_next'] - predictions) ** 2).sum()`.",
   "`tss` is the same with `train['vol_next'].mean()` in place of `predictions`."],
  "rss = ((test['vol_next'] - predictions) ** 2).sum()\n"
  "tss = ((test['vol_next'] - train['vol_next'].mean()) ** 2).sum()\n"
  "model_r2 = 1 - rss / tss\n\nprint('model R2      :', round(model_r2, 3))\n"
  "print('the bar (Part 4):', part4_pers_r2)",
  f"{MODEL_R2:.3f} against a bar of {PERS_R2:.3f}. Cleared, and not by much.\n\n"
  "A model that takes a fitting procedure and a feature and beats a one-line rule "
  "by four hundredths of an $R^2$ is a normal result in this subject. Anything "
  "dramatically better would be worth checking for a leak.")

# ==================================================================== Q6
q("Q6", "Three candidates",
  "You used one feature. Part 4 built three. Define the three candidate feature "
  "sets, then score each one on the **training** rows.\n\n"
  "- `set_a`: `vol_20d` alone\n"
  "- `set_b`: `vol_20d` and `ret_20d`\n"
  "- `set_c`: all three\n\n"
  "Which fits the training rows best?",
  "set_a = ...\nset_b = ...\nset_c = ...\n\nfor columns in [set_a, set_b, set_c]:\n"
  "    ...\n    print(columns, ...)",
  ["Each set is a plain list of column names, for example `['vol_20d']`.",
   "Inside the loop: fit on `train[columns]`, predict on `train[columns]`, and "
   "score against `train['vol_next']`."],
  "set_a = ['vol_20d']\nset_b = ['vol_20d', 'ret_20d']\nset_c = ['vol_20d', 'ret_20d', 'up_20d']\n\n"
  "for columns in [set_a, set_b, set_c]:\n    m = LinearRegression()\n"
  "    m.fit(train[columns], train['vol_next'])\n"
  "    error = np.sqrt(mean_squared_error(train['vol_next'], m.predict(train[columns])))\n"
  "    print(columns, round(error, 5))",
  f"{TR_RMSE['A']:.5f}, {TR_RMSE['B']:.5f}, {TR_RMSE['C']:.5f}. More features, lower "
  "error, as always. Adding a column can never make a fit worse, so this ordering "
  "was guaranteed before you ran anything.")

# ==================================================================== Q7
q("Q7", "The tempting mistake",
  "Score the same three candidates on the **test** block instead, and see which "
  "wins.\n\n"
  "Then read the note underneath carefully, because this is the question the whole "
  "notebook turns on.",
  "for columns in [set_a, set_b, set_c]:\n    ...\n    print(columns, ...)",
  "The same loop as Q6 with the scoring moved to `test`. The fit stays on `train`.",
  "for columns in [set_a, set_b, set_c]:\n    m = LinearRegression()\n"
  "    m.fit(train[columns], train['vol_next'])\n"
  "    error = np.sqrt(mean_squared_error(test['vol_next'], m.predict(test[columns])))\n"
  "    print(columns, round(error, 5))",
  f"{TE_RMSE['A']:.5f}, {TE_RMSE['B']:.5f}, {TE_RMSE['C']:.5f}. The ranking reverses: "
  "one feature is best and the extra columns were fitting noise.\n\n"
  "**And you are now in trouble.** Those three numbers were used to pick a model, "
  "so the winner's 0.00402 is no longer an honest estimate of anything. Part of "
  "being lowest of three is being genuinely better and part of it is luck, and "
  "there is no way to tell how much of each.\n\n"
  "Three candidates is mild. A real project compares dozens, and every comparison "
  "is another look. The rest of this notebook does the job without them.")

# ==================================================================== Q8
q("Q8", "A block for choosing",
  "Split the **training** block again by date: fit on everything up to the end of "
  "2020, and compare the candidates on 2021 and 2022. Call them `fit_rows` and "
  "`val_rows`.\n\n"
  "Which candidate wins now?",
  "fit_rows = ...\nval_rows = ...\n\nfor columns in [set_a, set_b, set_c]:\n    ...\n"
  "    print(columns, ...)",
  ["`train.loc[:'2020-12-31']` and `train.loc['2021-01-01':]`.",
   "Fit on `fit_rows`, score on `val_rows`. The test block is not mentioned "
   "anywhere in this cell."],
  "fit_rows = train.loc[:'2020-12-31']\nval_rows = train.loc['2021-01-01':]\n\n"
  "for columns in [set_a, set_b, set_c]:\n    m = LinearRegression()\n"
  "    m.fit(fit_rows[columns], fit_rows['vol_next'])\n"
  "    error = np.sqrt(mean_squared_error(val_rows['vol_next'], m.predict(val_rows[columns])))\n"
  "    print(columns, round(error, 5))",
  f"{VAL20_RMSE['A']:.5f}, {VAL20_RMSE['B']:.5f}, {VAL20_RMSE['C']:.5f}. Two features "
  "win, and the test block played no part in it.\n\n"
  "Note that this is not the answer the test rows gave in Q7. Hold that thought.")

# ==================================================================== Q9
q("Q9", "Move the cut",
  "Part 4 warned that one split is one experiment. Test it. Run the same "
  "comparison with the cut at the end of **2019** instead.",
  "early_fit = ...\nearly_val = ...\n\nfor columns in [set_a, set_b, set_c]:\n    ...\n"
  "    print(columns, ...)",
  "Only the two dates change: `train.loc[:'2019-12-31']` and "
  "`train.loc['2020-01-01':]`.",
  "early_fit = train.loc[:'2019-12-31']\nearly_val = train.loc['2020-01-01':]\n\n"
  "for columns in [set_a, set_b, set_c]:\n    m = LinearRegression()\n"
  "    m.fit(early_fit[columns], early_fit['vol_next'])\n"
  "    error = np.sqrt(mean_squared_error(early_val['vol_next'], m.predict(early_val[columns])))\n"
  "    print(columns, round(error, 5))",
  f"{VAL19_RMSE['A']:.5f}, {VAL19_RMSE['B']:.5f}, {VAL19_RMSE['C']:.5f}. One feature "
  "wins now, clearly.\n\n"
  "The candidates did not change. The training block did not change. Only the cut "
  "date moved, and the answer moved with it. This is exactly the defect Part 4 "
  "listed and could not fix.")

# ==================================================================== Q10
q("Q10", "Every cut date at once",
  "Instead of choosing a cut date, use five of them and average. Cross-validate "
  "the three candidates on the training block with `TimeSeriesSplit`, so every "
  "scored block comes after the rows it was fitted on.\n\n"
  "$$\\text{CV}_K=\\frac{1}{K}\\sum_{k=1}^{K} e_k$$\n\n"
  "Store the three averages in a dictionary called `cv_scores`.",
  "folds = ...\ncv_scores = {}\n\n"
  "for name, columns in [('a', set_a), ('b', set_b), ('c', set_c)]:\n    ...\n\ncv_scores",
  ["`TimeSeriesSplit(n_splits=5)`, then `cross_val_score(LinearRegression(), "
   "train[columns], train['vol_next'], cv=folds, "
   "scoring='neg_root_mean_squared_error')`.",
   "The scores come back negative, because scikit-learn reports everything so that "
   "larger is better. Store `-scores.mean()`."],
  "folds = TimeSeriesSplit(n_splits=5)\ncv_scores = {}\n\n"
  "for name, columns in [('a', set_a), ('b', set_b), ('c', set_c)]:\n"
  "    scores = cross_val_score(LinearRegression(), train[columns], train['vol_next'],\n"
  "                             cv=folds, scoring='neg_root_mean_squared_error')\n"
  "    cv_scores[name] = round(float(-scores.mean()), 5)\n\ncv_scores",
  f"{CV_TS['A']:.5f}, {CV_TS['B']:.5f}, {CV_TS['C']:.5f}. One feature wins, and the "
  "answer is now an average over five cut dates rather than whatever one arbitrary "
  "cut happened to say.\n\n"
  "Every number is larger than the test RMSE in Q7, and that is expected: each "
  "fold fits on fewer rows than the final model will, and scores on an older, "
  "harder stretch of history.")

# ==================================================================== Q11
q("Q11", "Read the folds, not just the average",
  "An average can hide a disaster. Print the five fold errors for `set_a`, and "
  "find which period the worst one was scored on.",
  "scores = ...\nfold_errors = ...\n\nworst = ...\nblocks = ...\n\n"
  "print('folds:', ...)\nprint('worst fold:', ...)\nprint('scored from', ..., 'to', ...)",
  ["`fold_errors = -scores` is the array of five errors, and "
   "`fold_errors.argmax()` gives the position of the largest.",
   "`blocks = list(folds.split(train))`. Each entry is a pair, and the second half "
   "holds the scored row positions, so `train.index[blocks[worst][1][0]]` is its "
   "first date."],
  "scores = cross_val_score(LinearRegression(), train[set_a], train['vol_next'],\n"
  "                         cv=folds, scoring='neg_root_mean_squared_error')\n"
  "fold_errors = -scores\n\nworst = fold_errors.argmax()\nblocks = list(folds.split(train))\n"
  "scored_rows = blocks[worst][1]\n\nprint('folds:', fold_errors.round(5))\n"
  "print('worst fold:', worst + 1)\n"
  "print('scored from', train.index[scored_rows[0]].date(), 'to', train.index[scored_rows[-1]].date())",
  f"Fold {WORST_FOLD} is roughly twice as bad as the rest, scored from {WORST_FROM} "
  f"to {WORST_TO}. It contains March 2020.\n\n"
  "A model fitted on the quiet years before it had never seen a month like that. "
  "This is worth reporting next to the average, because a risk model that fails "
  "precisely when risk arrives is a different object from one that is mediocre "
  "throughout.")

# ==================================================================== Q12
q("Q12", "The folds most people would have used",
  "Run the same comparison with `KFold(n_splits=5, shuffle=True, random_state=0)`, "
  "the version of cross-validation found in most textbooks. Store the winners of "
  "both methods in a dictionary called `verdicts`.",
  "shuffled = ...\nshuffled_scores = {}\n\n"
  "for name, columns in [('a', set_a), ('b', set_b), ('c', set_c)]:\n    ...\n\n"
  "verdicts = {\n    'TimeSeriesSplit': ...,\n    'shuffled KFold': ...,\n}\nverdicts",
  ["Only the `cv=` object changes from Q10.",
   "`min(d, key=d.get)` returns the key with the smallest value."],
  "shuffled = KFold(n_splits=5, shuffle=True, random_state=0)\nshuffled_scores = {}\n\n"
  "for name, columns in [('a', set_a), ('b', set_b), ('c', set_c)]:\n"
  "    scores = cross_val_score(LinearRegression(), train[columns], train['vol_next'],\n"
  "                             cv=shuffled, scoring='neg_root_mean_squared_error')\n"
  "    shuffled_scores[name] = round(float(-scores.mean()), 5)\n\n"
  "verdicts = {\n    'TimeSeriesSplit': min(cv_scores, key=cv_scores.get),\n"
  "    'shuffled KFold': min(shuffled_scores, key=shuffled_scores.get),\n}\nverdicts",
  f"Shuffled folds give {CV_KF['A']:.5f}, {CV_KF['B']:.5f}, {CV_KF['C']:.5f}, so they "
  "choose three features. Time-ordered folds choose one.\n\n"
  "Two neighbouring rows share nineteen of their twenty days in both the feature "
  "and the target. Shuffling puts one in the fitting block and the other in the "
  "scored block, so the model is asked about a row it has effectively already "
  "seen. Every score comes back flattering, and the most flexible candidate "
  "benefits most.\n\n"
  "This is the same dependence Part 4 listed as the first thing still wrong with "
  "the setup. It has not gone away; the folds are simply built so that it cannot "
  "do any damage.")

# ==================================================================== Q13
q("Q13", "Close the last gap",
  "One overlap survives even in time-ordered folds. The last rows a fold fits on "
  "have targets reaching twenty days into the block about to be scored. Drop them "
  "with the `gap` argument, and see what it costs.",
  "gapped = ...\n\n# then cross-validate set_a with `gapped` and print the average\n...",
  "`TimeSeriesSplit(n_splits=5, gap=20)`, and twenty is the length of the target "
  "window.",
  "gapped = TimeSeriesSplit(n_splits=5, gap=20)\n\n"
  "scores = cross_val_score(LinearRegression(), train[set_a], train['vol_next'],\n"
  "                         cv=gapped, scoring='neg_root_mean_squared_error')\n"
  "print('no gap:', cv_scores['a'])\nprint('gap 20:', round(float(-scores.mean()), 5))",
  f"{CV_TS['A']:.5f} against {CV_GAP:.5f}. A small change, because twenty rows are "
  "little against fitting blocks of several hundred.\n\n"
  "It moves in the direction it has to: removing a leak can only make an estimate "
  "worse, never better. If a correction like this ever improves your score, you "
  "have implemented it backwards.")

# ==================================================================== Q14
q("Q14", "Refit, and open the test block once",
  "The choosing is finished. Cross-validation chose `set_a`. Fit it on **all** the "
  "training rows, including the ones the folds used for scoring, and score it once "
  "on the test block.",
  "final_model = ...\nfinal_pred = ...\nfinal_rmse = ...\n\n"
  "print('test RMSE  :', ...)\nprint('persistence:', part4_pers_rmse)\n"
  "print('average    :', part4_base_rmse)",
  "Nothing new here. Fit `LinearRegression()` on `train[set_a]`, predict on "
  "`test[set_a]`, take the root mean squared error.",
  "final_model = LinearRegression()\nfinal_model.fit(train[set_a], train['vol_next'])\n"
  "final_pred = final_model.predict(test[set_a])\n"
  "final_rmse = np.sqrt(mean_squared_error(test['vol_next'], final_pred))\n\n"
  "print('test RMSE  :', round(final_rmse, 5))\nprint('persistence:', part4_pers_rmse)\n"
  "print('average    :', part4_base_rmse)",
  f"{MODEL_RMSE:.5f}, the same number Q4 produced, and now it means something "
  "different.\n\n"
  "In Q4 it was the score of a model nobody had chosen. Here it is the score of a "
  "model chosen by a procedure that never saw 2023 or 2024. The arithmetic is "
  "identical; the claim you are entitled to make is not.")

# ==================================================================== Q15
q("Q15", "Does it help when it matters",
  "A risk model exists for the turbulent half of the sample. So far every score "
  "has averaged the calm days and the busy ones together.\n\n"
  "Split the test block with a **boolean mask**: days whose `vol_20d` is below "
  "its median, and the rest. Then score the model and the persistence rule "
  "separately on each half.",
  "model_errors = ...\npers_errors = ...\ncalm = ...\n\n"
  "print('calm, model      :', ...)\nprint('calm, persistence:', ...)\n"
  "print('busy, model      :', ...)\nprint('busy, persistence:', ...)",
  ["The two error columns are `test['vol_next'] - final_pred` and "
   "`test['vol_next'] - test['vol_20d']`. Subtracting whole columns at once is "
   "the vectorised arithmetic from Session 3.",
   "`calm = test['vol_20d'] < test['vol_20d'].median()` is a mask of True and "
   "False, and `~calm` is its opposite. `errors[calm]` keeps one half."],
  "model_errors = test['vol_next'] - final_pred\n"
  "pers_errors = test['vol_next'] - test['vol_20d']\n"
  "calm = test['vol_20d'] < test['vol_20d'].median()\n\n"
  "print('calm, model      :', round(np.sqrt((model_errors[calm] ** 2).mean()), 5))\n"
  "print('calm, persistence:', round(np.sqrt((pers_errors[calm] ** 2).mean()), 5))\n"
  "print('busy, model      :', round(np.sqrt((model_errors[~calm] ** 2).mean()), 5))\n"
  "print('busy, persistence:', round(np.sqrt((pers_errors[~calm] ** 2).mean()), 5))",
  f"On the {N_CALM} calmest test days the model beats persistence by "
  f"{CALM_GAIN:.0f}%, {CALM_MODEL:.5f} against {CALM_PERS:.5f}. On the busy half "
  f"the two are level: {BUSY_MODEL:.5f} against {BUSY_PERS:.5f}.\n\n"
  "**The whole of the model's advantage comes from the quiet days.** Those are "
  "the days a risk report needs a forecast for least.\n\n"
  "This is not a reason to throw the model away, and it does not undo Q14. It is "
  "the first line of the limitations section, and you found it with one mask and "
  "no loop. A single RMSE averages the two halves together and shows none of it.")

# ==================================================================== Q16
q("Q16", "The test Part 4's rule failed",
  "Part 4 took the persistence rule across all eleven instruments and found it "
  f"positive on only {N_PERS_POS} of them. Do the same with the model.\n\n"
  "For every ticker, build the two-column table, split it at the end of 2022, fit "
  "`vol_20d` on the training rows, and record the model's test RMSE next to the "
  "persistence rule's. Store the pairs in a dictionary called `across_desk`.",
  "across_desk = {}\n\nfor ticker in TICKERS:\n    ...\n\n"
  "beat = ...\nprint('model beats persistence on', ..., 'of', ...)",
  ["Inside the loop, build `frame` with `vol_20d` and `vol_next` only, then "
   "`.dropna()` and split it the same way as Q2.",
   "Store a pair: `across_desk[ticker] = (model_rmse, pers_rmse)`. Then "
   "`sum(1 for t in across_desk if across_desk[t][0] < across_desk[t][1])`."],
  "across_desk = {}\n\nfor ticker in TICKERS:\n"
  "    frame = pd.DataFrame({'vol_20d': rets[ticker].rolling(20).std()})\n"
  "    frame['vol_next'] = rets[ticker].rolling(20).std().shift(-20)\n"
  "    frame = frame.dropna()\n\n"
  "    tr = frame.loc[:'2022-12-31']\n    te = frame.loc['2023-01-01':]\n\n"
  "    m = LinearRegression()\n    m.fit(tr[['vol_20d']], tr['vol_next'])\n"
  "    model_rmse = np.sqrt(mean_squared_error(te['vol_next'], m.predict(te[['vol_20d']])))\n"
  "    pers_rmse = np.sqrt(mean_squared_error(te['vol_next'], te['vol_20d']))\n"
  "    across_desk[ticker] = (model_rmse, pers_rmse)\n\n"
  "beat = sum(1 for t in across_desk if across_desk[t][0] < across_desk[t][1])\n"
  "print('model beats persistence on', beat, 'of', len(across_desk))",
  f"**{N_BEAT} of {len(TICKERS)}.**\n\n"
  "This is the most important number in the notebook. Part 4 ended with a rule "
  f"that looked convincing on Apple and was positive on {N_PERS_POS} of eleven "
  "names. A model chosen by cross-validation on one stock, with the choice made "
  "without ever looking at 2023 or 2024, transfers to the whole desk.\n\n"
  "That is what separates a result from a coincidence, and it took five sessions "
  "of setup to be able to say it honestly.")

# ==================================================================== Q17
q("Q17", "Draw it",
  "Draw the improvement, `persistence - model`, for the eleven instruments, sorted "
  "from largest to smallest, with a line at zero.",
  "improvement = ...\nranked = ...\n\n"
  "fig, ax = plt.subplots(figsize=(9, 3.2))\n...\nplt.show()",
  ["`across_desk[t]` is a pair, so the improvement is "
   "`across_desk[t][1] - across_desk[t][0]`.",
   "`pd.Series(improvement).sort_values(ascending=False)`, then `ax.bar(...)` and "
   "`ax.axhline(0, color='black')`."],
  "improvement = {t: across_desk[t][1] - across_desk[t][0] for t in across_desk}\n"
  "ranked = pd.Series(improvement).sort_values(ascending=False)\n\n"
  "fig, ax = plt.subplots(figsize=(9, 3.2))\nax.bar(ranked.index, ranked.values)\n"
  "ax.axhline(0, color='black', linewidth=1)\n"
  "ax.set_ylabel('RMSE saved')\n"
  "ax.set_title('How much the model beats persistence by, 2023 to 2024', loc='left')\n"
  "plt.show()",
  f"Every bar sits above the line. {BEST_GAIN} gains the most in relative terms, "
  f"about {BEST_GAIN_PCT:.0f}% off the persistence error.\n\n"
  "Compare this with the equivalent figure in Part 3, where the ranking of "
  "volatility barely carried from one period to the next. A forecast that helps "
  "everywhere is a very different object from a description that held once.")

# ==================================================================== Q18
q("Q18", "Write down what you would defend",
  "Finish the investigation the way Part 4 finished: one dictionary that somebody "
  "else could pick up, and a function that prints it with a verdict.\n\n"
  "Fill in `report`, then write `summarise(report)`: it prints each entry on its "
  "own line and ends with one sentence on whether the model earned its place.\n\n"
  "A dictionary, a loop, an f-string and an `if`. All of it comes from Sessions 1 "
  "and 2.",
  "report = {\n    'target': ...,\n    'features_kept': ...,\n"
  "    'features_tried': ...,\n    'chosen_by': ...,\n    'folds': ...,\n"
  "    'test_rmse': ...,\n    'persistence_rmse': ...,\n    'beat_across_desk': ...,\n}\n\n"
  "def summarise(report):\n    ...\n\nsummarise(report)",
  ["Most of the values are already in variables: `final_rmse` from Q14, "
   "`part4_pers_rmse` from the quick load, and `beat` from Q16.",
   "Inside the function, `for key in report:` then "
   "`print(f'{key:18} {report[key]}')`. For the verdict, an `if` on whether "
   "`test_rmse` is below `persistence_rmse`."],
  "report = {\n    'target': part4_target,\n    'features_kept': set_a,\n"
  "    'features_tried': part4_features,\n"
  "    'chosen_by': 'cross-validation on the training block only',\n"
  "    'folds': 'TimeSeriesSplit, 5 splits, expanding window',\n"
  "    'test_rmse': round(final_rmse, 5),\n"
  "    'persistence_rmse': part4_pers_rmse,\n"
  "    'beat_across_desk': f'{beat} of {len(across_desk)}',\n}\n\n"
  "def summarise(report):\n"
  "    \"\"\"Print a finished model selection, and say whether it earned its place.\"\"\"\n"
  "    for key in report:\n        print(f'{key:18} {report[key]}')\n\n"
  "    if report['test_rmse'] < report['persistence_rmse']:\n"
  "        print('\\nThe model beats the no-model rule on data used in no decision.')\n"
  "    else:\n"
  "        print('\\nThe model does not beat the no-model rule. Report the rule.')\n\n"
  "summarise(report)",
  "Eight lines that another analyst could act on. Note what is in there besides "
  "the score: which features were **tried** as well as which were kept, and how "
  "the choice was made.\n\n"
  "A reader who knows only that `vol_20d` was kept cannot tell whether it won "
  "against two rivals or against two hundred. That difference is the whole content "
  "of this session, and it belongs in the report.")

# ---------------------------------------------------------------- closing
md(
"## \U0001f9ed What you have now\n\n"
"| what | where it lives |\n"
"|:--|:--|\n"
"| the learning table, rebuilt | `table`, `train`, `test` |\n"
"| the first fitted model | `model`, `final_model` |\n"
"| its honest test score | `final_rmse` |\n"
"| the three candidates | `set_a`, `set_b`, `set_c` |\n"
"| what one validation block said | Q8 and Q9, and they disagree |\n"
"| what five time-ordered folds said | `cv_scores` |\n"
"| what shuffled folds would have said | `verdicts` |\n"
"| where the advantage comes from | `calm`, and the two halves of Q15 |\n"
"| the whole desk | `across_desk` |\n"
"| the thing you would defend | `report` |"
)

md(
"## What changed since Part 4\n\n"
"Part 4 ended with three complaints about its own setup. Two of them now have "
"answers.\n\n"
"- **One split was one experiment.** Q9 showed the winner changing when the cut "
"moved, and Q10 replaced the single cut with five and an average.\n"
"- **The rows were not independent.** Q12 showed what that dependence does to "
"shuffled folds, and Q13 closed the last overlap at the fold boundary. The "
"dependence is still there; the folds are simply built so that it cannot flatter "
"the score.\n"
"- **The regimes differ.** This one is not solved and probably cannot be. Q11 "
"found it sitting in fold " + str(WORST_FOLD) + ", where the error is twice the "
"others because that block contains March 2020, and Q15 found it again in the "
"test years, where the model's advantage over persistence is entirely in the "
"calm half. The honest response is to report those numbers next to the average "
"rather than to hide behind it."
)

md(
"## Where this leaves the risk report\n\n"
"Five parts ago this was two stocks compared with a subtraction. It is now a "
"forecast, with a model chosen by a procedure that never saw the years it is "
f"reported on, beating a no-model rule on {N_BEAT} of {len(TICKERS)} "
"instruments.\n\n"
"The model itself is the least interesting thing in it. One feature and two "
"coefficients, and the slope near a half that says volatility reverts. "
"Everything that makes the number believable is the apparatus around it.\n\n"
"**Next block:** models that bring settings of their own, and a search over those "
"settings that uses exactly the folds you built here."
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
