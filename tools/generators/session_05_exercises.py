# -*- coding: utf-8 -*-
"""Build session_05_exercises.ipynb.

Same conventions as Sessions 1 to 4: pleasant intro, 1-5 star badges, toolkit
card with title= hover docs, task -> work cell (blank-safe `...`) -> 1-2 folded
hints -> folded solution, no em-dashes, plain explanatory tone.

Session 5 is the first session where a model is fitted, so the exercises walk
the whole workflow: build the table, split it, fit, score against baselines,
compare candidates, discover that one validation split is unstable, then
cross-validate with folds that respect the calendar.

Only tools taught by the end of Session 5. New this session: LinearRegression
with .fit / .predict / .coef_ / .intercept_, mean_squared_error, cross_val_score,
KFold, TimeSeriesSplit (n_splits, max_train_size, gap), and the double-bracket
X. NOT taught, so never required: pipelines, scalers, k-NN, GridSearchCV,
r2_score, cross_validate, any model other than LinearRegression.

Returns are in PERCENT here, exactly as in the lecture, so the numbers read.

BLANK-SAFE RULES:
- every blank is the right-hand side of an assignment, a bare `...` statement,
  or an argument to print(). Never call a method on, index into, or do
  arithmetic with a placeholder.
- nothing depends on an earlier exercise having been solved: the setup cell and
  each work cell provide every variable the task uses.
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
OUT = ROOT / "session_05" / "session_05_exercises.ipynb"

cells = []


def badge(n, revisits=None):
    """Five unnamed stars, plus an optional note that this one reaches back.

    The scale restarts each session: it rates the work against what THIS
    session has taught.
    """
    stars = "★" * n + "☆" * (5 - n)
    return stars + (f"  · revisits {revisits}" if revisits else "")


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


# Exercises whose earlier-session tool is genuinely load-bearing.
REVISITS = {
    "A2": "S4", "A3": "S4", "A5": "S3", "A6": "S3",
    "B5": "S1", "B7": "S3",
    "C5": "S2", "C6": "S4", "C7": "S2",
    "D2": "S2", "D6": "S3",
    "E4": "S2",
    "F5": "S2", "F8": "S3",
    "G3": "S3",
    "I1": "S4",
    "K1": "S2", "K2": "S2", "K3": "S3", "K4": "S1", "K5": "S2",
}


def ex(sid, title, n, task, work, hints, sol_code, sol_note, revisits=None):
    md(f"### {sid} · {title}  {badge(n, revisits or REVISITS.get(sid))}\n\n{task}")
    code(work)
    _hints_solution(hints, sol_code, sol_note)


def section(header):
    md(header)


# ---- real numbers, computed here so every solution note is exact -----------
PX = pd.read_csv(ROOT / "data" / "prices.csv", parse_dates=["date"])
WIDE = PX.pivot(index="date", columns="ticker", values="close")
RET = WIDE.pct_change().dropna() * 100


def build(ticker):
    s = RET[ticker]
    frame = pd.DataFrame({
        "vol_20d": s.rolling(20).std(),
        "vol_60d": s.rolling(60).std(),
        "ret_20d": s.rolling(20).mean(),
    })
    frame["vol_next"] = s.rolling(20).std().shift(-20)
    return frame.dropna()


TBL = build("AAPL")
TRAIN, TEST = TBL.loc[:"2022-12-31"], TBL.loc["2023-01-01":]
N_TBL, N_TRAIN, N_TEST = len(TBL), len(TRAIN), len(TEST)
LAST_TRAIN, FIRST_TEST = TRAIN.index[-1].date(), TEST.index[0].date()

A, B, C = ["vol_20d"], ["vol_20d", "vol_60d"], ["vol_20d", "vol_60d", "ret_20d"]
SETS = {"A": A, "B": B, "C": C}


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(np.asarray(y), np.asarray(p))))


M1 = LinearRegression().fit(TRAIN[A], TRAIN["vol_next"])
INTERCEPT, SLOPE = float(M1.intercept_), float(M1.coef_[0])
PRED = M1.predict(TEST[A])
MSE_TEST, RMSE_TEST = float(mean_squared_error(TEST["vol_next"], PRED)), rmse(TEST["vol_next"], PRED)
TRAIN_MEAN = float(TRAIN["vol_next"].mean())
RMSE_BASE = rmse(TEST["vol_next"], np.full(N_TEST, TRAIN_MEAN))
RMSE_PERS = rmse(TEST["vol_next"], TEST["vol_20d"])
R2_OOS = float(1 - ((TEST["vol_next"] - PRED) ** 2).sum() /
               ((TEST["vol_next"] - TRAIN_MEAN) ** 2).sum())

# C8 to C10: the residuals, done with plain array arithmetic
ERRORS = TEST["vol_next"] - PRED
RMSE_BY_HAND = float(np.sqrt((ERRORS ** 2).mean()))
SHARE_OVER = float((ERRORS < 0).mean())
MEAN_ERROR = float(ERRORS.mean())
MED_VOL = float(TEST["vol_20d"].median())
_calm = TEST["vol_20d"] < MED_VOL
RMSE_CALM = float(np.sqrt((ERRORS[_calm] ** 2).mean()))
RMSE_BUSY = float(np.sqrt((ERRORS[~_calm] ** 2).mean()))
N_CALM = int(_calm.sum())

TR_RMSE = {k: rmse(TRAIN["vol_next"], LinearRegression().fit(TRAIN[v], TRAIN["vol_next"]).predict(TRAIN[v]))
           for k, v in SETS.items()}
TE_RMSE = {k: rmse(TEST["vol_next"], LinearRegression().fit(TRAIN[v], TRAIN["vol_next"]).predict(TEST[v]))
           for k, v in SETS.items()}

# a deliberately useless extra column
_rng = np.random.default_rng(0)
NOISE = pd.Series(_rng.normal(0, 1, N_TRAIN), index=TRAIN.index)
_tn = TRAIN.copy()
_tn["noise"] = NOISE
RMSE_NOISE = rmse(TRAIN["vol_next"],
                  LinearRegression().fit(_tn[A + ["noise"]], _tn["vol_next"]).predict(_tn[A + ["noise"]]))

# validation blocks
FIT20, VAL20 = TRAIN.loc[:"2020-12-31"], TRAIN.loc["2021-01-01":]
FIT19, VAL19 = TRAIN.loc[:"2019-12-31"], TRAIN.loc["2020-01-01":]
VAL20_RMSE = {k: rmse(VAL20["vol_next"], LinearRegression().fit(FIT20[v], FIT20["vol_next"]).predict(VAL20[v]))
              for k, v in SETS.items()}
VAL19_RMSE = {k: rmse(VAL19["vol_next"], LinearRegression().fit(FIT19[v], FIT19["vol_next"]).predict(VAL19[v]))
              for k, v in SETS.items()}
CUTS = ["2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31"]
CUT_WINNER, CUT_SPREAD = {}, {}
for _cut in CUTS:
    _f, _v = TRAIN.loc[:_cut], TRAIN.loc[_cut:]
    _scores = {k: rmse(_v["vol_next"], LinearRegression().fit(_f[vv], _f["vol_next"]).predict(_v[vv]))
               for k, vv in SETS.items()}
    CUT_WINNER[_cut] = min(_scores, key=_scores.get)
    CUT_SPREAD[_cut] = _scores["A"]

# cross-validation
TS5 = TimeSeriesSplit(n_splits=5)
FOLD_SIZES = [(len(a), len(b)) for a, b in TS5.split(TRAIN)]
FOLDS_A = -cross_val_score(LinearRegression(), TRAIN[A], TRAIN["vol_next"],
                           cv=TS5, scoring="neg_root_mean_squared_error")
CV_TS = {k: float(-cross_val_score(LinearRegression(), TRAIN[v], TRAIN["vol_next"],
                                   cv=TS5, scoring="neg_root_mean_squared_error").mean())
         for k, v in SETS.items()}
CV_TS10 = {k: float(-cross_val_score(LinearRegression(), TRAIN[v], TRAIN["vol_next"],
                                     cv=TimeSeriesSplit(n_splits=10),
                                     scoring="neg_root_mean_squared_error").mean())
           for k, v in SETS.items()}
CV_KF = {k: float(-cross_val_score(LinearRegression(), TRAIN[v], TRAIN["vol_next"],
                                   cv=KFold(n_splits=5, shuffle=True, random_state=0),
                                   scoring="neg_root_mean_squared_error").mean())
         for k, v in SETS.items()}
CV_GAP = float(-cross_val_score(LinearRegression(), TRAIN[A], TRAIN["vol_next"],
                                cv=TimeSeriesSplit(n_splits=5, gap=20),
                                scoring="neg_root_mean_squared_error").mean())
CV_ROLL = float(-cross_val_score(LinearRegression(), TRAIN[A], TRAIN["vol_next"],
                                 cv=TimeSeriesSplit(n_splits=5, max_train_size=500),
                                 scoring="neg_root_mean_squared_error").mean())
WORST_FOLD = int(np.argmax(FOLDS_A)) + 1
_wf = list(TS5.split(TRAIN))[WORST_FOLD - 1][1]
WORST_FROM, WORST_TO = TRAIN.index[_wf[0]].date(), TRAIN.index[_wf[-1]].date()

# the overlap between neighbouring rows
STEP = float((TBL["vol_20d"] - TBL["vol_20d"].shift(1)).abs().mean())
SPREAD = float(TBL["vol_20d"].std())

# AIC and BIC on the training block
AIC, BIC, MSE_TR = {}, {}, {}
for _k, _v in SETS.items():
    _m = LinearRegression().fit(TRAIN[_v], TRAIN["vol_next"])
    _mse = float(mean_squared_error(TRAIN["vol_next"], _m.predict(TRAIN[_v])))
    _d = len(_v) + 1
    MSE_TR[_k] = _mse
    AIC[_k] = N_TRAIN * np.log(_mse) + 2 * _d
    BIC[_k] = N_TRAIN * np.log(_mse) + _d * np.log(N_TRAIN)
N_EFF = N_TRAIN // 20
BIC_EFF = {k: N_EFF * np.log(MSE_TR[k]) + (len(SETS[k]) + 1) * np.log(N_EFF) for k in SETS}

# other tickers
OTHER = {}
for _t in ["MSFT", "KO"]:
    _tb = build(_t)
    _tr, _te = _tb.loc[:"2022-12-31"], _tb.loc["2023-01-01":]
    _m = LinearRegression().fit(_tr[A], _tr["vol_next"])
    OTHER[_t] = {
        "cv": {k: float(-cross_val_score(LinearRegression(), _tr[v], _tr["vol_next"],
                                         cv=TS5, scoring="neg_root_mean_squared_error").mean())
               for k, v in SETS.items()},
        "test": rmse(_te["vol_next"], _m.predict(_te[A])),
        "pers": rmse(_te["vol_next"], _te["vol_20d"]),
        "base": rmse(_te["vol_next"], np.full(len(_te), _tr["vol_next"].mean())),
    }

ALL_T = sorted(RET.columns)
WIN = {}
for _t in ALL_T:
    _tb = build(_t)
    _tr, _te = _tb.loc[:"2022-12-31"], _tb.loc["2023-01-01":]
    _m = LinearRegression().fit(_tr[A], _tr["vol_next"])
    WIN[_t] = (rmse(_te["vol_next"], _m.predict(_te[A])), rmse(_te["vol_next"], _te["vol_20d"]))
N_WIN = sum(1 for t in ALL_T if WIN[t][0] < WIN[t][1])
BEST_T = min(ALL_T, key=lambda t: WIN[t][1] - WIN[t][0])
BIGGEST_T = max(ALL_T, key=lambda t: (WIN[t][1] - WIN[t][0]) / WIN[t][1])

# ---------------------------------------------------------------- top matter
md(
"# \U0001f4d3 Session 5 · Exercises\n"
"### Fitting a model, and choosing between models\n\n"
"This is the first set of exercises where you fit something. The fitting itself "
"is three lines, so most of the work here is the part that surrounds it: "
"scoring against a baseline, comparing candidates without spending the test "
"rows, and cross-validating with folds that respect the calendar.\n\n"
"You will build the volatility table from the lecture, fit a linear regression "
"to it, beat two rules that use no model at all, watch a single validation "
"split give two different answers, and finish by running the whole workflow on "
"a stock the lecture never touched."
)

md(
"## How to use this notebook\n\n"
"- Run the **setup cell** below first. It loads the price data, builds the "
"table the lecture used, and imports the three scikit-learn pieces.\n"
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
"**Returns are in percent here**, exactly as in the lecture, so an error of "
"`0.41` means 0.41 percentage points."
)

md(
"### Difficulty\n\n"
"| badge | what to expect |\n"
"|:--|:--|\n"
"| ★☆☆☆☆ | One step, straight from the lecture. You are checking that you can type it. |\n"
"| ★★☆☆☆ | The same idea on new data, or two steps in a row. Nothing to decide. |\n"
"| ★★★☆☆ | Combine two ideas, or adapt a pattern rather than copy it. |\n"
"| ★★★★☆ | You choose the approach. Several steps, and something has to be worked out before you type. |\n"
"| ★★★★★ | A genuine puzzle: an insight, or a constraint that rules out the obvious route. Always solvable with what you have. |\n\n"
"The stars rate the work against **this** session. A three-star task here "
"assumes everything from Sessions 1 to 4, so it is a bigger piece of work than a "
"three-star task in an earlier notebook.\n\n"
"Some exercises also carry a **revisits** tag. Those need something from an "
"earlier session as well as today's material, and they are there on purpose: "
"the skills are meant to accumulate."
)

md(
"## \U0001f9f0 Your toolkit for today\n\n"
"Everything from Sessions 1 to 4 still applies. This card holds what Session 5 "
"added.\n\n"
"> Names in brackets (`frame`, `columns`, `model`, ...) are **placeholders**: put "
"your own variable there. **Hover any tool** to see what it does."
)

md(
'<p style="line-height:2.1"><strong>Building X and y</strong><br>\n'
'<code style="cursor:help" title="A LIST of column names inside the brackets, so the result stays a table. This is what X has to be.">frame[[\'col_a\', \'col_b\']]</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="One pair of brackets gives a single column. That is what y has to be.">frame[\'target\']</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Rows up to a date, and rows from a date. A table with a date index slices with dates.">frame.loc[:\'2022-12-31\']  ·  frame.loc[\'2023-01-01\':]</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Rows and columns, as a pair. Handy for checking a split went where you meant.">frame.shape</code></p>\n\n'
'<p style="line-height:2.1"><strong>Fitting and predicting</strong><br>\n'
'<code style="cursor:help" title="Create a model. It knows nothing until you fit it.">LinearRegression()</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Read the training rows and compute the coefficients. Features first, target second.">model.fit(X, y)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="One prediction per row of X, in the same order, as a plain numpy array.">model.predict(X)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The constant term the fit found. The trailing underscore means it came from the data.">model.intercept_</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="One coefficient per feature, in the order the columns came in.">model.coef_</code></p>\n\n'
'<p style="line-height:2.1"><strong>Scoring</strong><br>\n'
'<code style="cursor:help" title="Mean squared error. The TRUE values go first and the predictions second.">mean_squared_error(y_true, y_pred)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Square root, which puts a mean squared error back into the units of the target.">np.sqrt(value)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="An array of one value repeated. A baseline that ignores the features looks like this.">np.full(n, value)</code></p>\n\n'
'<p style="line-height:2.1"><strong>Cross-validation</strong><br>\n'
'<code style="cursor:help" title="Folds that move forward in time: each scored block comes after the rows fitted on.">TimeSeriesSplit(n_splits=5)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Keep only the most recent rows for fitting, so old years are forgotten.">TimeSeriesSplit(n_splits=5, max_train_size=500)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Throw away the rows whose target reaches into the block about to be scored.">TimeSeriesSplit(n_splits=5, gap=20)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The textbook folds: the rows are shuffled first, which assumes they are interchangeable.">KFold(n_splits=5, shuffle=True, random_state=0)</code></p>\n\n'
'<p style="line-height:2.1"><strong>Running the folds</strong><br>\n'
'<code style="cursor:help" title="Fit and score once per fold, and hand back one number per fold.">cross_val_score(model, X, y, cv=folds, scoring=\'neg_root_mean_squared_error\')</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="scikit-learn reports scores so that larger is better, so errors come back negative. Flip the sign.">-scores</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The row positions of the fitted rows and the scored rows, one pair per fold.">folds.split(frame)</code></p>'
)

md(
"**Formulas you will reach for**\n\n"
r"| what | formula |" "\n"
r"|:--|:--|" "\n"
r"| Root mean squared error | $$\text{RMSE}=\sqrt{\dfrac{1}{n}\sum_i (y_i-\hat{y}_i)^2}$$ |" "\n"
r"| Out-of-sample R squared | $$R^2=1-\dfrac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y}_{\text{train}})^2}$$ |" "\n"
r"| Cross-validation error | $$\text{CV}_K=\dfrac{1}{K}\sum_{k=1}^{K} e_k$$ |" "\n"
r"| AIC | $$n\log(\text{MSE})+2d$$ |" "\n"
r"| BIC | $$n\log(\text{MSE})+d\log(n)$$ |" "\n"
)

md("---")

# ---------------------------------------------------------------- setup cell
md(
"## ⚙️ Setup: run this first\n\n"
"This loads the price data, rebuilds the table the lecture worked on, and "
"imports the scikit-learn pieces. If you are in Google Colab it downloads the "
"data by itself."
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


# Eleven instruments, 2015 to 2024. Returns in PERCENT, as in the lecture.
prices = load_csv("prices.csv", parse_dates=["date"])
wide = prices.pivot(index="date", columns="ticker", values="close")
rets = wide.pct_change().dropna() * 100

# The lecture's table: three features looking back, one target looking forward
apple = rets["AAPL"]
table = pd.DataFrame({
    "vol_20d": apple.rolling(20).std(),
    "vol_60d": apple.rolling(60).std(),
    "ret_20d": apple.rolling(20).mean(),
})
table["vol_next"] = apple.rolling(20).std().shift(-20)
table = table.dropna()

train = table.loc[:"2022-12-31"]
test = table.loc["2023-01-01":]

# The three candidate feature sets from the lecture
A = ["vol_20d"]
B = ["vol_20d", "vol_60d"]
C = ["vol_20d", "vol_60d", "ret_20d"]

# For one exercise that needs a column with nothing in it
rng = np.random.default_rng(0)

print("table:", table.shape, "rows x columns")
print("train:", len(train), " test:", len(test))
print("columns:", list(table.columns))'''
)

md("---")

# ============================================================ A
section(
"## \U0001f9f1 A · The table, and the split\n\n"
"The setup cell already built these, but building them yourself once is the "
"only way the rest of the notebook stops being magic."
)

ex("A1", "Apple's returns", 1,
   "Take Apple's daily returns out of `rets` into a Series called `apple_ret`.",
   "apple_ret = ...\napple_ret",
   "`rets` has one column per ticker. Select a column by name.",
   "apple_ret = rets['AAPL']\napple_ret",
   f"A return for every trading day, in percent. There are {len(RET):,} of them.")

ex("A2", "A feature that looks back", 2,
   "Build `vol_20d`: the standard deviation of the **last twenty** daily returns, "
   "as a Series.",
   "vol_20d = ...\nvol_20d",
   "`s.rolling(20).std()` gives a twenty-row window ending on the current row.",
   "vol_20d = rets['AAPL'].rolling(20).std()\nvol_20d",
   "The window ends on the current row, so this feature only ever uses today and "
   "the nineteen days before it. The first nineteen values are `NaN`.")

ex("A3", "A target that looks forward", 3,
   "Build `vol_next`: the volatility of the **next twenty** trading days, on "
   "today's row.\n\n"
   "$$\\text{vol\\_next}_t = \\text{sd}\\big(r_{t+1}, \\ldots, r_{t+20}\\big)$$",
   "vol_next = ...\nvol_next",
   ["Start from the same rolling standard deviation as A2. That gives the twenty "
    "days **ending** today.",
    "`shift(-20)` moves values up by twenty rows, so the window that ends twenty "
    "days from now lands on today's row."],
   "vol_next = rets['AAPL'].rolling(20).std().shift(-20)\nvol_next",
   "Backwards for the feature, forwards for the target, and the two windows never "
   "overlap. Getting the sign of the shift wrong is the most expensive mistake in "
   "this whole notebook, and it is invisible in the output.")

ex("A4", "Put them together", 2,
   "Assemble the full table: the three features from the lecture plus the target, "
   "with incomplete rows dropped. Call it `tbl`.",
   "tbl = ...\ntbl",
   "`pd.DataFrame({'name': series, ...})` builds the table, then `.dropna()` "
   "removes the rows where a window did not fit.",
   "tbl = pd.DataFrame({\n    'vol_20d': rets['AAPL'].rolling(20).std(),\n"
   "    'vol_60d': rets['AAPL'].rolling(60).std(),\n"
   "    'ret_20d': rets['AAPL'].rolling(20).mean(),\n})\n"
   "tbl['vol_next'] = rets['AAPL'].rolling(20).std().shift(-20)\n"
   "tbl = tbl.dropna()\ntbl",
   f"{N_TBL:,} rows and 4 columns. The rows lost at the start are where the "
   "sixty-day window had not filled, and the twenty at the end are where the "
   "target reaches past the last date in the file.")

ex("A5", "Split by date", 2,
   "Cut `table` into `tr` (everything up to the end of 2022) and `te` (2023 "
   "onwards), and print how many rows each has.",
   "tr = ...\nte = ...\nprint('train:', ...)\nprint('test :', ...)",
   "`frame.loc[:'2022-12-31']` and `frame.loc['2023-01-01':]`.",
   "tr = table.loc[:'2022-12-31']\nte = table.loc['2023-01-01':]\n"
   "print('train:', len(tr))\nprint('test :', len(te))",
   f"{N_TRAIN:,} training rows and {N_TEST} test rows, so the test block is about "
   f"{100 * N_TEST / N_TBL:.0f}% of the sample and it is the most recent part of it.")

ex("A6", "Prove the split does not overlap", 3,
   "Print the **last date in the training block** and the **first date in the "
   "test block**. They should not be the same day.",
   "last_train = ...\nfirst_test = ...\nprint('last train row:', ...)\nprint('first test row:', ...)",
   ["`frame.index` holds the dates, and `[-1]` and `[0]` pick the ends.",
    "`train.index[-1]` and `test.index[0]`. Add `.date()` if you want to lose the "
    "time part."],
   "last_train = train.index[-1]\nfirst_test = test.index[0]\n"
   "print('last train row:', last_train.date())\nprint('first test row:', first_test.date())",
   f"{LAST_TRAIN} then {FIRST_TEST}. Worth checking once: an off-by-one here puts "
   "a training day inside the test block, and nothing will warn you.")

# ============================================================ B
section(
"## \U0001f527 B · Your first fit\n\n"
"Three lines to fit a model, and rather more than three to understand what "
"came out."
)

ex("B1", "X and y", 1,
   "Build `X_train` from the single column `vol_20d`, and `y_train` from "
   "`vol_next`, both out of `train`. Print the shape of each.",
   "X_train = ...\ny_train = ...\nprint('X:', ...)\nprint('y:', ...)",
   "`X` needs a **list** of column names inside the brackets, so it stays a table. "
   "`y` is one column, so one pair of brackets.",
   "X_train = train[['vol_20d']]\ny_train = train['vol_next']\n"
   "print('X:', X_train.shape)\nprint('y:', y_train.shape)",
   f"`X` is `({N_TRAIN}, 1)` and `y` is `({N_TRAIN},)`. Two dimensions against one, "
   "which is the difference the double brackets make.")

ex("B2", "One bracket or two", 2,
   "Print the shape of `train['vol_20d']` and of `train[['vol_20d']]` side by side, "
   "so the difference is on the screen rather than in your memory.",
   "one_pair = ...\ntwo_pairs = ...\nprint('one pair :', ...)\nprint('two pairs:', ...)",
   "Both are just selections out of `train`. Ask each one for its `.shape`.",
   "one_pair = train['vol_20d']\ntwo_pairs = train[['vol_20d']]\n"
   "print('one pair :', one_pair.shape)\nprint('two pairs:', two_pairs.shape)",
   f"`({N_TRAIN},)` against `({N_TRAIN}, 1)`. The first is a Series and scikit-learn "
   "refuses it as `X`. The second is a table with one column, which is what `X` "
   "always has to be.")

ex("B3", "Fit it", 2,
   "Fit a `LinearRegression` on `X_train` and `y_train`. The model object is "
   "already made for you.",
   "X_train = train[['vol_20d']]\ny_train = train['vol_next']\n\n"
   "model = LinearRegression()\n\n...        # fit the model here\n\nmodel",
   "`model.fit(X, y)`. Features first, target second.",
   "X_train = train[['vol_20d']]\ny_train = train['vol_next']\n\n"
   "model = LinearRegression()\nmodel.fit(X_train, y_train)\n\nmodel",
   "`.fit()` prints nothing and returns the model. What changed is stored on the "
   "object, which is what B4 goes looking for.")

ex("B4", "What it learned", 1,
   "Print the intercept and the coefficients of the fitted model.",
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n\n"
   "print('intercept:', ...)\nprint('coefficients:', ...)",
   "Both are attributes with a trailing underscore, which marks them as computed "
   "from the data.",
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n\n"
   "print('intercept:', model.intercept_)\nprint('coefficients:', model.coef_)",
   f"An intercept of about {INTERCEPT:.4f} and one coefficient of about {SLOPE:.4f}. "
   "`coef_` is an array even when there is only one feature, because there usually "
   "is more than one.")

ex("B5", "Write the fitted rule out", 3,
   "Print the fitted model as an equation, rounded to two decimals, using an "
   "f-string. Aim for something like `vol_next = 0.88 + 0.49 * vol_20d`.",
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n\n"
   "equation = ...\nprint(equation)",
   ["`model.coef_[0]` is the single coefficient. `f'{value:.2f}'` rounds inside an "
    "f-string.",
    "f\"vol_next = {model.intercept_:.2f} + {model.coef_[0]:.2f} * vol_20d\""],
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n\n"
   "equation = f'vol_next = {model.intercept_:.2f} + {model.coef_[0]:.2f} * vol_20d'\n"
   "print(equation)",
   f"`vol_next = {INTERCEPT:.2f} + {SLOPE:.2f} * vol_20d`. A slope near a half rather "
   "than near one is mean reversion: only about half of this month's unusual "
   "volatility carries into next month.",
   revisits="S1")

ex("B6", "Predict", 2,
   "Predict on the test block and show the first five predictions.",
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n\n"
   "predictions = ...\npredictions",
   "`model.predict(X)` wants the same columns, in the same shape, out of `test`.",
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n\n"
   "predictions = model.predict(test[['vol_20d']])\npredictions[:5]",
   "A plain numpy array, one number per test row, in the same order as the rows. "
   "No dates attached, which is why the next exercise puts them back.")

ex("B7", "Actual, predicted, residual", 3,
   "Build a DataFrame called `check` with three columns, `actual`, `predicted` and "
   "`residual`, for the **first five** test rows, indexed by their dates.",
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n"
   "predictions = model.predict(test[['vol_20d']])\n\ncheck = ...\ncheck",
   ["`test['vol_next'].values[:5]` and `predictions[:5]` are the two number "
    "columns, and the residual is the first minus the second.",
    "Pass `index=test.index[:5]` to `pd.DataFrame` so the dates come back."],
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n"
   "predictions = model.predict(test[['vol_20d']])\n\n"
   "check = pd.DataFrame({\n    'actual': test['vol_next'].values[:5],\n"
   "    'predicted': predictions[:5],\n"
   "    'residual': test['vol_next'].values[:5] - predictions[:5],\n"
   "}, index=test.index[:5])\ncheck",
   "The residual column is the one Session 4 called $e_i = y_i - \\hat{y}_i$. Reading "
   "five of them tells you more about a model than any single score does.",
   revisits="S3")

# ============================================================ C
section(
"## \U0001f4cf C · Scoring, and something to beat\n\n"
"A score on its own says nothing. These exercises build the two rules any model "
"has to beat before it is worth reporting."
)

ex("C1", "Mean squared error", 1,
   "Compute the mean squared error of the model's predictions on the test block.",
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n"
   "predictions = model.predict(test[['vol_20d']])\n\nmse = ...\nprint(mse)",
   "`mean_squared_error(y_true, y_pred)`, with the true values first.",
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n"
   "predictions = model.predict(test[['vol_20d']])\n\n"
   "mse = mean_squared_error(test['vol_next'], predictions)\nprint(mse)",
   f"About {MSE_TEST:.4f}, in squared percentage points. Nobody can picture that "
   "unit, which is what C2 fixes.")

ex("C2", "Root mean squared error", 1,
   "Take the square root, so the number is back in percentage points.",
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n"
   "predictions = model.predict(test[['vol_20d']])\n\nrmse = ...\nprint(rmse)",
   "`np.sqrt` of the mean squared error.",
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n"
   "predictions = model.predict(test[['vol_20d']])\n\n"
   "rmse = np.sqrt(mean_squared_error(test['vol_next'], predictions))\nprint(rmse)",
   f"About {RMSE_TEST:.4f}. The forecast is out by roughly {RMSE_TEST:.2f} percentage "
   "points on a typical day.")

ex("C3", "Baseline one: guess the average", 2,
   "Score the rule that ignores the features and predicts the **training** average "
   "on every test day.",
   "guess = ...\nflat = ...\n\nbase_rmse = ...\nprint(base_rmse)",
   ["The guess is `train['vol_next'].mean()`. Careful: the training mean, not the "
    "test mean.",
    "`np.full(len(test), guess)` repeats it once per test row."],
   "guess = train['vol_next'].mean()\nflat = np.full(len(test), guess)\n\n"
   "base_rmse = np.sqrt(mean_squared_error(test['vol_next'], flat))\nprint(base_rmse)",
   f"{RMSE_BASE:.4f}. Using the test average instead would need information nobody "
   "had at the time, and it would make this baseline look better than it is.")

ex("C4", "Baseline two: repeat this month", 2,
   "Score the rule that predicts next month's volatility to be whatever the last "
   "twenty days delivered. No fitting at all.",
   "pers_rmse = ...\nprint(pers_rmse)",
   "The prediction column is already sitting in the test block: `test['vol_20d']`.",
   "pers_rmse = np.sqrt(mean_squared_error(test['vol_next'], test['vol_20d']))\n"
   "print(pers_rmse)",
   f"{RMSE_PERS:.4f}, which already beats the average by a distance. Volatility "
   "clusters, so repeating the recent past is a genuinely strong rule and the one "
   "a model really has to beat.")

ex("C5", "All three, in a dictionary", 3,
   "Collect the three scores into a dictionary called `scores`, then print them "
   "from best to worst.",
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n\n"
   "scores = {\n    'model': ...,\n    'average': ...,\n    'persistence': ...,\n}\n\n"
   "# print them smallest first\nfor name in scores:\n    ...",
   ["Each value is the same `np.sqrt(mean_squared_error(...))` you have written "
    "three times already.",
    "`sorted(scores, key=scores.get)` gives the keys in order of their values, "
    "smallest first."],
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n\n"
   "scores = {\n"
   "    'model': np.sqrt(mean_squared_error(test['vol_next'], model.predict(test[['vol_20d']]))),\n"
   "    'average': np.sqrt(mean_squared_error(test['vol_next'], np.full(len(test), train['vol_next'].mean()))),\n"
   "    'persistence': np.sqrt(mean_squared_error(test['vol_next'], test['vol_20d'])),\n"
   "}\n\n"
   "for name in sorted(scores, key=scores.get):\n    print(f'{name:12} {scores[name]:.4f}')",
   f"model {RMSE_TEST:.4f}, persistence {RMSE_PERS:.4f}, average {RMSE_BASE:.4f}. The "
   "model wins by about "
   f"{100 * (RMSE_PERS - RMSE_TEST) / RMSE_PERS:.0f}% over persistence, which is a "
   "normal margin when forecasting financial data.",
   revisits="S2")

ex("C6", "Out-of-sample R squared", 3,
   "Compute how much better the model is than the average, as an $R^2$. The "
   "baseline in the denominator must be the **training** mean.\n\n"
   "$$R^2 = 1 - \\frac{\\sum_i (y_i-\\hat{y}_i)^2}{\\sum_i (y_i-\\bar{y}_{\\text{train}})^2}$$",
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n"
   "predictions = model.predict(test[['vol_20d']])\n\n"
   "rss = ...\ntss = ...\nr2 = ...\nprint(r2)",
   ["`rss` is `((test['vol_next'] - predictions) ** 2).sum()`.",
    "`tss` is the same thing with `train['vol_next'].mean()` in place of the "
    "predictions."],
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n"
   "predictions = model.predict(test[['vol_20d']])\n\n"
   "rss = ((test['vol_next'] - predictions) ** 2).sum()\n"
   "tss = ((test['vol_next'] - train['vol_next'].mean()) ** 2).sum()\n"
   "r2 = 1 - rss / tss\nprint(r2)",
   f"About {R2_OOS:.3f}. On days it had never seen, the model accounts for roughly "
   f"{100 * R2_OOS:.0f}% of the variation that the training average leaves "
   "unexplained.",
   revisits="S4")

ex("C7", "A function that does all of it", 4,
   "Write `test_rmse(columns)`: it fits a linear regression on those columns of "
   "`train`, predicts on `test`, and returns the RMSE. Then call it on `A`, on `B` "
   "and on `C`.",
   "def test_rmse(columns):\n    ...\n\nprint('A:', ...)\nprint('B:', ...)\nprint('C:', ...)",
   ["The body is four lines: make the model, fit it on `train[columns]`, predict "
    "on `test[columns]`, return the root mean squared error.",
    "Remember `return`. A function with no `return` hands back `None`."],
   "def test_rmse(columns):\n    model = LinearRegression()\n"
   "    model.fit(train[columns], train['vol_next'])\n"
   "    predictions = model.predict(test[columns])\n"
   "    return np.sqrt(mean_squared_error(test['vol_next'], predictions))\n\n"
   "print('A:', test_rmse(A))\nprint('B:', test_rmse(B))\nprint('C:', test_rmse(C))",
   f"{TE_RMSE['A']:.4f}, {TE_RMSE['B']:.4f}, {TE_RMSE['C']:.4f}. Wrapping it in a "
   "function is worth the two minutes: everything from here compares feature sets, "
   "and you now do it in one line each.",
   revisits="S2")

ex("C8", "The same number, two ways", 3,
   "`mean_squared_error` is only a name for some arithmetic you can do yourself. "
   "Compute the RMSE from the residuals with plain array arithmetic, and check it "
   "against the library.",
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n"
   "predictions = model.predict(test[['vol_20d']])\n\n"
   "errors = ...\nby_hand = ...\nfrom_library = ...\n\n"
   "print('by hand    :', ...)\nprint('the library:', ...)",
   ["The residuals are `test['vol_next'] - predictions`. Subtracting a whole "
    "column from a whole array at once is the vectorised arithmetic from Session 3.",
    "Square them, take the mean, take the square root. No loop is needed anywhere: "
    "`np.sqrt((errors ** 2).mean())`."],
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n"
   "predictions = model.predict(test[['vol_20d']])\n\n"
   "errors = test['vol_next'] - predictions\nby_hand = np.sqrt((errors ** 2).mean())\n"
   "from_library = np.sqrt(mean_squared_error(test['vol_next'], predictions))\n\n"
   "print('by hand    :', by_hand)\nprint('the library:', from_library)",
   f"Both give {RMSE_BY_HAND:.10f}. Not close, identical, because they are the same "
   "four operations.\n\n"
   "Doing it twice and checking the two agree is a habit worth keeping. It is how "
   "you find out that you passed the arguments to a metric the wrong way round, "
   "which is silent and produces a plausible number.",
   revisits="S3")

ex("C9", "Which way is it wrong", 3,
   "An RMSE says how big the errors are and nothing about their direction. Using "
   "the same residuals, find the share of test days where the model predicted "
   "**too much** volatility, and the average error.",
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n"
   "predictions = model.predict(test[['vol_20d']])\nerrors = test['vol_next'] - predictions\n\n"
   "share_over = ...\nmean_error = ...\n\n"
   "print('over-predicted on:', ...)\nprint('mean error       :', ...)",
   ["An error is `actual - predicted`, so the model predicted too much wherever "
    "the error is **negative**.",
    "`(errors < 0)` is a column of True and False. Its `.mean()` is the share that "
    "are True, exactly as in Session 3."],
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n"
   "predictions = model.predict(test[['vol_20d']])\nerrors = test['vol_next'] - predictions\n\n"
   "share_over = (errors < 0).mean()\nmean_error = errors.mean()\n\n"
   "print('over-predicted on:', round(share_over, 3))\n"
   "print('mean error       :', round(mean_error, 5))",
   f"The model forecast too much volatility on **{100 * SHARE_OVER:.0f}% of test "
   f"days**, and the average error is {MEAN_ERROR:.5f} rather than zero.\n\n"
   "That is a systematic bias, not bad luck. The training years contain 2020 and "
   "the test years do not, so a model fitted on the louder period expects more "
   "noise than 2023 and 2024 delivered. An RMSE on its own would never have shown "
   "you this, which is why a residual is worth looking at directly.",
   revisits="S3")

# ============================================================ D
section(
"## \U0001f5f3️ D · Three candidates\n\n"
"Same model, same rows, same target. Only the columns change."
)

ex("D1", "Name them", 1,
   "The setup cell defined `A`, `B` and `C`. Print each one with how many features "
   "it holds.",
   "for name, columns in [('A', A), ('B', B), ('C', C)]:\n    ...",
   "Inside the loop, print the name, `len(columns)`, and the list itself.",
   "for name, columns in [('A', A), ('B', B), ('C', C)]:\n"
   "    print(name, len(columns), 'features:', columns)",
   "One, two and three features. Everything in this section and the next two is "
   "about deciding between them.")

ex("D2", "Score them on the rows they were fitted on", 2,
   "Loop over the three candidates and print the **training** RMSE of each.",
   "for columns in [A, B, C]:\n    model = LinearRegression()\n    ...\n    error = ...\n"
   "    print(len(columns), 'features:', ...)",
   "Fit on `train[columns]` and score the predictions against `train['vol_next']`, "
   "on the same rows.",
   "for columns in [A, B, C]:\n    model = LinearRegression()\n"
   "    model.fit(train[columns], train['vol_next'])\n"
   "    error = np.sqrt(mean_squared_error(train['vol_next'], model.predict(train[columns])))\n"
   "    print(len(columns), 'features:', round(error, 4))",
   f"{TR_RMSE['A']:.4f}, {TR_RMSE['B']:.4f}, {TR_RMSE['C']:.4f}. The error falls every "
   "time a column is added, which is what training error always does.",
   revisits="S2")

ex("D3", "Score them on rows they have never seen", 2,
   "Now the same loop, scored on the test block instead.",
   "for columns in [A, B, C]:\n    model = LinearRegression()\n    ...\n    error = ...\n"
   "    print(len(columns), 'features:', ...)",
   "Only the two places that say `train` in the scoring line change to `test`. The "
   "fit stays on the training rows.",
   "for columns in [A, B, C]:\n    model = LinearRegression()\n"
   "    model.fit(train[columns], train['vol_next'])\n"
   "    error = np.sqrt(mean_squared_error(test['vol_next'], model.predict(test[columns])))\n"
   "    print(len(columns), 'features:', round(error, 4))",
   f"{TE_RMSE['A']:.4f}, {TE_RMSE['B']:.4f}, {TE_RMSE['C']:.4f}. The ranking reverses. "
   "The two extra columns were fitting noise rather than signal.")

ex("D4", "The two winners, side by side", 3,
   "Build a DataFrame called `comparison` with one row per candidate and two "
   "columns, `train` and `test`.",
   "rows = []\nfor name, columns in [('A', A), ('B', B), ('C', C)]:\n    ...\n\n"
   "comparison = ...\ncomparison",
   ["Inside the loop, append a dictionary: "
    "`rows.append({'set': name, 'train': ..., 'test': ...})`.",
    "`pd.DataFrame(rows).set_index('set')` turns the list of dictionaries into a "
    "table."],
   "rows = []\nfor name, columns in [('A', A), ('B', B), ('C', C)]:\n"
   "    model = LinearRegression()\n    model.fit(train[columns], train['vol_next'])\n"
   "    rows.append({\n        'set': name,\n"
   "        'train': np.sqrt(mean_squared_error(train['vol_next'], model.predict(train[columns]))),\n"
   "        'test': np.sqrt(mean_squared_error(test['vol_next'], model.predict(test[columns]))),\n"
   "    })\n\ncomparison = pd.DataFrame(rows).set_index('set')\ncomparison",
   "The `train` column falls as you go down and the `test` column rises. Two "
   "columns, two opposite orderings, and only one of them was available while the "
   "choice was being made.")

ex("D5", "A column with nothing in it", 4,
   "Add a column of pure random numbers to the training block and fit `A` plus "
   "that column. Does the **training** error go up, down, or stay the same?",
   "noisy = train.copy()\nnoisy['noise'] = rng.normal(0, 1, len(train))\n\n"
   "plain_error = ...\nnoisy_error = ...\n\n"
   "print('vol_20d alone      :', ...)\nprint('vol_20d plus noise :', ...)",
   ["Fit twice: once on `train[A]`, once on `noisy[['vol_20d', 'noise']]`. Score "
    "both on the training rows.",
    "The noise column cannot possibly help on new data. Watch what it does to the "
    "fit on these rows anyway."],
   "noisy = train.copy()\nnoisy['noise'] = rng.normal(0, 1, len(train))\n\n"
   "m1 = LinearRegression().fit(train[A], train['vol_next'])\n"
   "m2 = LinearRegression().fit(noisy[['vol_20d', 'noise']], noisy['vol_next'])\n\n"
   "plain_error = np.sqrt(mean_squared_error(train['vol_next'], m1.predict(train[A])))\n"
   "noisy_error = np.sqrt(mean_squared_error(noisy['vol_next'], m2.predict(noisy[['vol_20d', 'noise']])))\n\n"
   "print('vol_20d alone      :', round(plain_error, 8))\n"
   "print('vol_20d plus noise :', round(noisy_error, 8))",
   f"{TR_RMSE['A']:.8f} against {RMSE_NOISE:.8f}. The difference is tiny, and the "
   "direction is the whole point: adding a column of pure noise made the training "
   "fit **better**, never worse. It can only ever go that way, which is why "
   "training error cannot choose a model.\n\n"
   "Eight decimals are needed to see it here because one random column against "
   f"{N_TRAIN:,} rows has very little room. Give a model thirty noise columns and "
   "the same effect becomes large enough to fool you.")

ex("D6", "Draw it", 4,
   "Plot training RMSE and test RMSE against the number of features, both on one "
   "axes, with a legend.",
   "n_features = []\ntrain_scores = []\ntest_scores = []\n\n"
   "for columns in [A, B, C]:\n    ...\n\n"
   "fig, ax = plt.subplots(figsize=(7, 3.5))\n...\nplt.show()",
   ["Fill the three lists inside the loop, then draw two `ax.plot(...)` lines with "
    "`marker='o'` and a `label=`.",
    "`ax.set_xticks([1, 2, 3])` keeps the axis honest, and `ax.legend()` shows the "
    "labels."],
   "n_features = []\ntrain_scores = []\ntest_scores = []\n\n"
   "for columns in [A, B, C]:\n    model = LinearRegression()\n"
   "    model.fit(train[columns], train['vol_next'])\n"
   "    n_features.append(len(columns))\n"
   "    train_scores.append(np.sqrt(mean_squared_error(train['vol_next'], model.predict(train[columns]))))\n"
   "    test_scores.append(np.sqrt(mean_squared_error(test['vol_next'], model.predict(test[columns]))))\n\n"
   "fig, ax = plt.subplots(figsize=(7, 3.5))\n"
   "ax.plot(n_features, train_scores, marker='o', label='training')\n"
   "ax.plot(n_features, test_scores, marker='o', label='test')\n"
   "ax.set_xticks([1, 2, 3])\nax.set_xlabel('number of features')\n"
   "ax.set_ylabel('RMSE (percentage points)')\n"
   "ax.set_title('Training error falls. Test error does not.', loc='left')\n"
   "ax.legend()\nplt.show()",
   "Two lines going opposite ways. This is the picture from Session 4, drawn from a "
   "model you fitted yourself rather than from an invented example.",
   revisits="S3")

# ============================================================ E
section(
"## ✂️ E · One split is one experiment\n\n"
"The test rows have now been used to compare three models, which is exactly what "
"they were not for. These exercises build the block that does the choosing."
)

ex("E1", "Cut the training block in two", 2,
   "Split `train` again by date: `fit_rows` up to the end of 2020, `val_rows` from "
   "2021 onwards. Print the size of each.",
   "fit_rows = ...\nval_rows = ...\nprint('fit       :', ...)\nprint('validation:', ...)",
   "Same `.loc` slicing as the first split, applied to `train` instead of `table`.",
   "fit_rows = train.loc[:'2020-12-31']\nval_rows = train.loc['2021-01-01':]\n"
   "print('fit       :', len(fit_rows))\nprint('validation:', len(val_rows))",
   f"{len(FIT20):,} rows to fit on and {len(VAL20)} to compare on. The test block is "
   "not mentioned anywhere in this exercise, which is the point of it.")

ex("E2", "Choose on the validation block", 2,
   "Score the three candidates on `val_rows` after fitting them on `fit_rows`, and "
   "say which wins.",
   "fit_rows = train.loc[:'2020-12-31']\nval_rows = train.loc['2021-01-01':]\n\n"
   "for columns in [A, B, C]:\n    ...\n    print(len(columns), 'features:', ...)",
   "The same loop as D3, with `fit_rows` where `train` was and `val_rows` where "
   "`test` was.",
   "fit_rows = train.loc[:'2020-12-31']\nval_rows = train.loc['2021-01-01':]\n\n"
   "for columns in [A, B, C]:\n    model = LinearRegression()\n"
   "    model.fit(fit_rows[columns], fit_rows['vol_next'])\n"
   "    error = np.sqrt(mean_squared_error(val_rows['vol_next'], model.predict(val_rows[columns])))\n"
   "    print(len(columns), 'features:', round(error, 4))",
   f"{VAL20_RMSE['A']:.4f}, {VAL20_RMSE['B']:.4f}, {VAL20_RMSE['C']:.4f}. Three features "
   "win, and no test row was involved in finding that out.")

ex("E3", "Move the cut one year", 3,
   "Run the same comparison with the cut at the end of **2019** instead. Does the "
   "same candidate win?",
   "fit_rows = ...\nval_rows = ...\n\nfor columns in [A, B, C]:\n    ...\n"
   "    print(len(columns), 'features:', ...)",
   "Only the two dates change: `train.loc[:'2019-12-31']` and "
   "`train.loc['2020-01-01':]`.",
   "fit_rows = train.loc[:'2019-12-31']\nval_rows = train.loc['2020-01-01':]\n\n"
   "for columns in [A, B, C]:\n    model = LinearRegression()\n"
   "    model.fit(fit_rows[columns], fit_rows['vol_next'])\n"
   "    error = np.sqrt(mean_squared_error(val_rows['vol_next'], model.predict(val_rows[columns])))\n"
   "    print(len(columns), 'features:', round(error, 4))",
   f"{VAL19_RMSE['A']:.4f}, {VAL19_RMSE['B']:.4f}, {VAL19_RMSE['C']:.4f}. One feature "
   "wins now, and by a clear margin. The candidates did not change and the training "
   "block did not change. Only the cut date moved.")

ex("E4", "Four cut dates, in a loop", 4,
   "Loop over the cuts `'2018-12-31'`, `'2019-12-31'`, `'2020-12-31'` and "
   "`'2021-12-31'`, and record which candidate wins at each one in a dictionary "
   "called `winner`.",
   "winner = {}\n\nfor cut in ['2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31']:\n"
   "    ...\n\nwinner",
   ["Inside the loop, build a small dictionary of the three scores first, then pick "
    "its smallest key.",
    "`min(scores, key=scores.get)` returns the key whose value is smallest."],
   "winner = {}\n\nfor cut in ['2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31']:\n"
   "    fit_rows = train.loc[:cut]\n    val_rows = train.loc[cut:]\n\n    scores = {}\n"
   "    for name, columns in [('A', A), ('B', B), ('C', C)]:\n"
   "        model = LinearRegression()\n"
   "        model.fit(fit_rows[columns], fit_rows['vol_next'])\n"
   "        scores[name] = np.sqrt(mean_squared_error(val_rows['vol_next'], model.predict(val_rows[columns])))\n\n"
   "    winner[cut] = min(scores, key=scores.get)\n\nwinner",
   "Two of the cuts choose " + CUT_WINNER[CUTS[0]] + " and two choose " +
   CUT_WINNER[CUTS[2]] + ". A single validation block is a single experiment, and "
   "running one experiment is not enough to separate candidates this close together.",
   revisits="S2")

ex("E5", "How far does one number move", 5,
   "For candidate `A` alone, find the **largest and smallest** validation RMSE "
   "across those four cut dates, and print the gap between them.",
   "a_scores = ...\n\nprint('lowest :', ...)\nprint('highest:', ...)\nprint('gap    :', ...)",
   ["Collect the four numbers into a list first, with the same loop as E4 but only "
    "candidate `A`.",
    "`min(a_scores)`, `max(a_scores)`, and the difference between them."],
   "a_scores = []\n\nfor cut in ['2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31']:\n"
   "    fit_rows = train.loc[:cut]\n    val_rows = train.loc[cut:]\n"
   "    model = LinearRegression()\n    model.fit(fit_rows[A], fit_rows['vol_next'])\n"
   "    a_scores.append(np.sqrt(mean_squared_error(val_rows['vol_next'], model.predict(val_rows[A]))))\n\n"
   "print('lowest :', round(min(a_scores), 4))\nprint('highest:', round(max(a_scores), 4))\n"
   "print('gap    :', round(max(a_scores) - min(a_scores), 4))",
   f"The same model, on the same training block, scores anywhere from "
   f"{min(CUT_SPREAD.values()):.4f} to {max(CUT_SPREAD.values()):.4f} depending on "
   "where the cut goes. The gap between the candidates is far smaller than that, "
   "which is why one cut cannot decide between them.")

# ============================================================ F
section(
"## \U0001f504 F · Cross-validation\n\n"
"Several cut dates instead of one, and the average of what they say."
)

ex("F1", "Five folds", 1,
   "Make a `TimeSeriesSplit` with five folds and print how many rows each fold "
   "fits on and scores on.",
   "folds = ...\n\n# then loop over folds.split(train) and print the two sizes\n...",
   "`folds.split(train)` yields a pair of row-position arrays per fold.",
   "folds = TimeSeriesSplit(n_splits=5)\n\n"
   "for fit_rows, score_rows in folds.split(train):\n"
   "    print(len(fit_rows), 'fitted', len(score_rows), 'scored')",
   "The fitted block grows by one step each time and the scored block stays the "
   "same size. Every scored block comes after the rows it was fitted on.")

ex("F2", "Cross-validate candidate A", 2,
   "Run `cross_val_score` on candidate `A` with those folds, and print what comes "
   "back.",
   "folds = TimeSeriesSplit(n_splits=5)\n\nscores = ...\nprint(scores)",
   "`cross_val_score(LinearRegression(), X, y, cv=folds, "
   "scoring='neg_root_mean_squared_error')`.",
   "folds = TimeSeriesSplit(n_splits=5)\n\n"
   "scores = cross_val_score(LinearRegression(), train[A], train['vol_next'],\n"
   "                         cv=folds, scoring='neg_root_mean_squared_error')\nprint(scores)",
   "Five numbers, all negative. scikit-learn reports every score so that larger is "
   "better, so it flips the sign of an error.")

ex("F3", "Turn it into an error", 1,
   "Flip the sign and take the average.",
   "folds = TimeSeriesSplit(n_splits=5)\nscores = cross_val_score(LinearRegression(), train[A], train['vol_next'],\n"
   "                         cv=folds, scoring='neg_root_mean_squared_error')\n\n"
   "cv_error = ...\nprint(cv_error)",
   "`-scores.mean()`, or `(-scores).mean()`. Both work.",
   "folds = TimeSeriesSplit(n_splits=5)\nscores = cross_val_score(LinearRegression(), train[A], train['vol_next'],\n"
   "                         cv=folds, scoring='neg_root_mean_squared_error')\n\n"
   "cv_error = -scores.mean()\nprint(cv_error)",
   f"{CV_TS['A']:.4f}. Larger than the test RMSE of {RMSE_TEST:.4f}, because every "
   "fold fits on less data than the final model will and scores on a harder stretch "
   "of history.")

ex("F4", "The spread, not just the average", 2,
   "Print the five fold errors, their average, and their standard deviation.",
   "folds = TimeSeriesSplit(n_splits=5)\nscores = cross_val_score(LinearRegression(), train[A], train['vol_next'],\n"
   "                         cv=folds, scoring='neg_root_mean_squared_error')\n\n"
   "fold_errors = ...\nprint('folds  :', ...)\nprint('average:', ...)\nprint('spread :', ...)",
   "`-scores` is the array of fold errors. It has `.mean()` and `.std()` like any "
   "numpy array.",
   "folds = TimeSeriesSplit(n_splits=5)\nscores = cross_val_score(LinearRegression(), train[A], train['vol_next'],\n"
   "                         cv=folds, scoring='neg_root_mean_squared_error')\n\n"
   "fold_errors = -scores\nprint('folds  :', fold_errors.round(3))\n"
   "print('average:', round(fold_errors.mean(), 4))\nprint('spread :', round(fold_errors.std(), 4))",
   "Four of the folds sit close together and one is roughly twice as large. An "
   "average that hides a fold like that is worth less than the two numbers together.")

ex("F5", "All three candidates", 3,
   "Cross-validate `A`, `B` and `C` in a loop and print the average error of each.",
   "folds = TimeSeriesSplit(n_splits=5)\n\nfor columns in [A, B, C]:\n    ...\n"
   "    print(len(columns), 'features:', ...)",
   "One `cross_val_score` call per candidate, then `-scores.mean()`.",
   "folds = TimeSeriesSplit(n_splits=5)\n\nfor columns in [A, B, C]:\n"
   "    scores = cross_val_score(LinearRegression(), train[columns], train['vol_next'],\n"
   "                             cv=folds, scoring='neg_root_mean_squared_error')\n"
   "    print(len(columns), 'features:', round(-scores.mean(), 4))",
   f"{CV_TS['A']:.4f}, {CV_TS['B']:.4f}, {CV_TS['C']:.4f}. One feature wins, and the "
   "answer is now an average over five cut dates rather than whatever one arbitrary "
   "cut happened to say.",
   revisits="S2")

ex("F6", "Which period is the bad fold", 4,
   "Find the worst fold for candidate `A`, and print the first and last date it "
   "was scored on.",
   "folds = TimeSeriesSplit(n_splits=5)\nscores = cross_val_score(LinearRegression(), train[A], train['vol_next'],\n"
   "                         cv=folds, scoring='neg_root_mean_squared_error')\nfold_errors = -scores\n\n"
   "worst = ...\nblocks = ...\n\nprint('worst fold:', ...)\nprint('scored from', ..., 'to', ...)",
   ["`fold_errors.argmax()` gives the position of the largest error.",
    "`blocks = list(folds.split(train))` lets you index the folds. Each entry is a "
    "pair, and the second half holds the scored row positions."],
   "folds = TimeSeriesSplit(n_splits=5)\nscores = cross_val_score(LinearRegression(), train[A], train['vol_next'],\n"
   "                         cv=folds, scoring='neg_root_mean_squared_error')\nfold_errors = -scores\n\n"
   "worst = fold_errors.argmax()\nblocks = list(folds.split(train))\n"
   "scored_rows = blocks[worst][1]\n\nprint('worst fold:', worst + 1)\n"
   "print('scored from', train.index[scored_rows[0]].date(), 'to', train.index[scored_rows[-1]].date())",
   f"Fold {WORST_FOLD}, scored from {WORST_FROM} to {WORST_TO}. It contains March "
   "2020. A model fitted on the quiet years before it had never seen a month like "
   "that, and the fold score says so.")

ex("F7", "Ten folds instead of five", 2,
   "Run the same comparison with `n_splits=10` and see whether the winner holds.",
   "folds = ...\n\nfor columns in [A, B, C]:\n    ...\n    print(len(columns), 'features:', ...)",
   "Only the `n_splits` argument changes.",
   "folds = TimeSeriesSplit(n_splits=10)\n\nfor columns in [A, B, C]:\n"
   "    scores = cross_val_score(LinearRegression(), train[columns], train['vol_next'],\n"
   "                             cv=folds, scoring='neg_root_mean_squared_error')\n"
   "    print(len(columns), 'features:', round(-scores.mean(), 4))",
   f"{CV_TS10['A']:.4f}, {CV_TS10['B']:.4f}, {CV_TS10['C']:.4f}. Every number is lower, "
   "because each fold now fits on more rows and scores on a shorter stretch.\n\n"
   "Note that the winner changed: with ten folds the three candidates sit within "
   "0.01 of each other and the order flips. That is worth knowing rather than "
   "hiding. It says these three are not really distinguishable on this data, so "
   "the honest report is that the simplest one is as good as any, not that it is "
   "provably best.")

ex("F8", "Draw the folds", 4,
   "Draw the five fold errors for candidate `A` as a bar chart, with a horizontal "
   "line at their average.",
   "folds = TimeSeriesSplit(n_splits=5)\nscores = cross_val_score(LinearRegression(), train[A], train['vol_next'],\n"
   "                         cv=folds, scoring='neg_root_mean_squared_error')\nfold_errors = -scores\n\n"
   "fig, ax = plt.subplots(figsize=(7, 3))\n...\nplt.show()",
   ["`ax.bar(range(1, 6), fold_errors)` draws the five bars.",
    "`ax.axhline(fold_errors.mean())` is the average line, and `label=` plus "
    "`ax.legend()` explains it."],
   "folds = TimeSeriesSplit(n_splits=5)\nscores = cross_val_score(LinearRegression(), train[A], train['vol_next'],\n"
   "                         cv=folds, scoring='neg_root_mean_squared_error')\nfold_errors = -scores\n\n"
   "fig, ax = plt.subplots(figsize=(7, 3))\nax.bar(range(1, 6), fold_errors)\n"
   "ax.axhline(fold_errors.mean(), color='black', linestyle='--', label='average')\n"
   "ax.set_xlabel('fold')\nax.set_ylabel('RMSE (percentage points)')\n"
   "ax.set_title('One fold is twice as hard as the others', loc='left')\n"
   "ax.legend()\nplt.show()",
   "One bar stands well above the line. A picture like this belongs next to any "
   "cross-validation number you report, because the average on its own cannot show "
   "it.",
   revisits="S3")

# ============================================================ G
section(
"## \U0001f4c6 G · Folds and the calendar\n\n"
"What the textbook version of cross-validation does to a table whose rows "
"overlap."
)

ex("G1", "The textbook folds", 2,
   "Run the same three-candidate comparison with "
   "`KFold(n_splits=5, shuffle=True, random_state=0)`.",
   "folds = ...\n\nfor columns in [A, B, C]:\n    ...\n    print(len(columns), 'features:', ...)",
   "Only the splitter changes. Everything else is the loop from F5.",
   "folds = KFold(n_splits=5, shuffle=True, random_state=0)\n\nfor columns in [A, B, C]:\n"
   "    scores = cross_val_score(LinearRegression(), train[columns], train['vol_next'],\n"
   "                             cv=folds, scoring='neg_root_mean_squared_error')\n"
   "    print(len(columns), 'features:', round(-scores.mean(), 4))",
   f"{CV_KF['A']:.4f}, {CV_KF['B']:.4f}, {CV_KF['C']:.4f}. Three features win, which is "
   "the candidate the test rows called worst.")

ex("G2", "Three verdicts on one table", 3,
   "Put the winner from each of the three methods into a dictionary: shuffled "
   "`KFold`, `TimeSeriesSplit`, and the test rows.",
   "def best_of(cv_object):\n    ...\n\nverdicts = {\n    'shuffled KFold': ...,\n"
   "    'TimeSeriesSplit': ...,\n    'the test rows': ...,\n}\nverdicts",
   ["`best_of` can build a dictionary of the three averages and return "
    "`min(scores, key=scores.get)`.",
    "For the test rows there are no folds: fit on all of `train` and score on "
    "`test`, then take the smallest."],
   "def best_of(cv_object):\n    scores = {}\n"
   "    for name, columns in [('A', A), ('B', B), ('C', C)]:\n"
   "        s = cross_val_score(LinearRegression(), train[columns], train['vol_next'],\n"
   "                            cv=cv_object, scoring='neg_root_mean_squared_error')\n"
   "        scores[name] = -s.mean()\n    return min(scores, key=scores.get)\n\n"
   "test_scores = {}\nfor name, columns in [('A', A), ('B', B), ('C', C)]:\n"
   "    model = LinearRegression().fit(train[columns], train['vol_next'])\n"
   "    test_scores[name] = np.sqrt(mean_squared_error(test['vol_next'], model.predict(test[columns])))\n\n"
   "verdicts = {\n    'shuffled KFold': best_of(KFold(n_splits=5, shuffle=True, random_state=0)),\n"
   "    'TimeSeriesSplit': best_of(TimeSeriesSplit(n_splits=5)),\n"
   "    'the test rows': min(test_scores, key=test_scores.get),\n}\nverdicts",
   "`TimeSeriesSplit` agrees with the test rows and shuffled `KFold` does not. On a "
   "table like this one, a shuffled score can leave you with the wrong model, not "
   "just the wrong number.")

ex("G3", "Why the shuffle helps the model cheat", 3,
   "Show how similar two neighbouring rows are. Print the average size of the "
   "change in `vol_20d` from one row to the next, next to the standard deviation "
   "of the column itself.",
   "step = ...\nspread = ...\n\nprint('average day-to-day change:', ...)\n"
   "print('standard deviation   :', ...)",
   ["`table['vol_20d'] - table['vol_20d'].shift(1)` is the change from one row to "
    "the next.",
    "Take `.abs().mean()` of that, and compare it with `table['vol_20d'].std()`."],
   "step = (table['vol_20d'] - table['vol_20d'].shift(1)).abs().mean()\n"
   "spread = table['vol_20d'].std()\n\n"
   "print('average day-to-day change:', round(step, 4))\n"
   "print('standard deviation   :', round(spread, 4))",
   f"A typical day-to-day change of {STEP:.4f} against a spread of {SPREAD:.4f}, so "
   f"consecutive rows differ by about {100 * STEP / SPREAD:.0f}% of the column's own "
   "variation. Two neighbouring rows share nineteen of their twenty days. Put one in "
   "the fitting block and the other in the scored block and the score comes out too "
   "good.",
   revisits="S3")

ex("G4", "Leave a gap", 2,
   "Cross-validate candidate `A` with a twenty-row gap between each fitted block "
   "and the block it is scored on.",
   "folds = ...\n\n"
   "# then cross-validate candidate A with those folds and print the average\n...",
   "`TimeSeriesSplit(n_splits=5, gap=20)`.",
   "folds = TimeSeriesSplit(n_splits=5, gap=20)\n\n"
   "scores = cross_val_score(LinearRegression(), train[A], train['vol_next'],\n"
   "                         cv=folds, scoring='neg_root_mean_squared_error')\n"
   "print(round(-scores.mean(), 4))",
   f"{CV_GAP:.4f}, against {CV_TS['A']:.4f} with no gap. A small change here, because "
   "twenty rows are little against fitting blocks of several hundred. The same "
   "correction matters far more when the target spans a longer window.")

ex("G5", "Forget the old years", 3,
   "Cross-validate candidate `A` with a **rolling** window of 500 rows instead of "
   "an expanding one, and compare.",
   "expanding = ...\nrolling = ...\n\nprint('expanding:', ...)\nprint('rolling  :', ...)",
   "`max_train_size=500` turns an expanding window into a rolling one.",
   "expanding = cross_val_score(LinearRegression(), train[A], train['vol_next'],\n"
   "                            cv=TimeSeriesSplit(n_splits=5),\n"
   "                            scoring='neg_root_mean_squared_error')\n"
   "rolling = cross_val_score(LinearRegression(), train[A], train['vol_next'],\n"
   "                          cv=TimeSeriesSplit(n_splits=5, max_train_size=500),\n"
   "                          scoring='neg_root_mean_squared_error')\n\n"
   "print('expanding:', round(-expanding.mean(), 4))\nprint('rolling  :', round(-rolling.mean(), 4))",
   f"{CV_TS['A']:.4f} expanding against {CV_ROLL:.4f} rolling. Very close here, which "
   "says that the older years are neither much help nor much harm. On data where "
   "the relationship really has moved, the two numbers separate.")

ex("G6", "Rank the estimates", 4,
   "Build a dictionary of the cross-validation error for candidate `A` under four "
   "arrangements: shuffled `KFold`, expanding, expanding with a gap, and rolling. "
   "Print them from most optimistic to most pessimistic.",
   "arrangements = {\n    'shuffled': ...,\n    'expanding': ...,\n"
   "    'expanding + gap': ...,\n    'rolling': ...,\n}\n\n# print them most optimistic first\nfor name in arrangements:\n    ...",
   ["Each value is one `cross_val_score` call with a different `cv=` object, "
    "followed by `-scores.mean()`.",
    "`sorted(arrangements, key=arrangements.get)` puts the smallest, most "
    "optimistic one first."],
   "def cv_error(cv_object):\n"
   "    s = cross_val_score(LinearRegression(), train[A], train['vol_next'],\n"
   "                        cv=cv_object, scoring='neg_root_mean_squared_error')\n"
   "    return -s.mean()\n\n"
   "arrangements = {\n    'shuffled': cv_error(KFold(n_splits=5, shuffle=True, random_state=0)),\n"
   "    'expanding': cv_error(TimeSeriesSplit(n_splits=5)),\n"
   "    'expanding + gap': cv_error(TimeSeriesSplit(n_splits=5, gap=20)),\n"
   "    'rolling': cv_error(TimeSeriesSplit(n_splits=5, max_train_size=500)),\n}\n\n"
   "for name in sorted(arrangements, key=arrangements.get):\n"
   "    print(f'{name:16} {arrangements[name]:.4f}')",
   f"The shuffled arrangement is the most optimistic at {CV_KF['A']:.4f}, and it is "
   "the only one of the four that lets a fold be scored by a model fitted on its "
   "neighbours. Every arrangement that respects the calendar gives a worse and more "
   "honest number.")

# ============================================================ H
section(
"## \U0001f9ee H · Counting the cost of a feature\n\n"
"AIC and BIC score fit and size in one number, without any folds at all."
)

ex("H1", "Training MSE for each candidate", 2,
   "Both formulas start from the mean squared error on the rows the model was "
   "fitted on. Collect it for the three candidates in a dictionary called "
   "`train_mse`.",
   "train_mse = {}\n\nfor name, columns in [('A', A), ('B', B), ('C', C)]:\n    ...\n\ntrain_mse",
   "`mean_squared_error(train['vol_next'], model.predict(train[columns]))`, with no "
   "square root this time.",
   "train_mse = {}\n\nfor name, columns in [('A', A), ('B', B), ('C', C)]:\n"
   "    model = LinearRegression()\n    model.fit(train[columns], train['vol_next'])\n"
   "    train_mse[name] = round(float(mean_squared_error(train['vol_next'], model.predict(train[columns]))), 5)\n\n"
   "train_mse",
   f"{MSE_TR['A']:.5f}, {MSE_TR['B']:.5f}, {MSE_TR['C']:.5f}. Falling as columns are "
   "added, as it always does.")

ex("H2", "AIC", 3,
   "Compute the AIC of each candidate.\n\n"
   "$$\\text{AIC} = n\\log(\\text{MSE}) + 2d$$\n\n"
   "Here $n$ is the number of training rows and $d$ is the number of parameters: "
   "one per feature, plus the intercept.",
   "n = ...\naic = {}\n\nfor name, columns in [('A', A), ('B', B), ('C', C)]:\n    ...\n\naic",
   ["`n` is `len(train)`, and `d` is `len(columns) + 1`.",
    "`np.log` is the natural logarithm. Fit the model, get its training MSE, then "
    "apply the formula."],
   "n = len(train)\naic = {}\n\nfor name, columns in [('A', A), ('B', B), ('C', C)]:\n"
   "    model = LinearRegression()\n    model.fit(train[columns], train['vol_next'])\n"
   "    mse = mean_squared_error(train['vol_next'], model.predict(train[columns]))\n"
   "    d = len(columns) + 1\n    aic[name] = round(float(n * np.log(mse) + 2 * d), 1)\n\naic",
   f"{AIC['A']:.0f}, {AIC['B']:.0f}, {AIC['C']:.0f}. Smaller is better, so AIC prefers "
   f"candidate {min(AIC, key=AIC.get)}.")

ex("H3", "BIC", 3,
   "Now BIC, which charges more per parameter.\n\n"
   "$$\\text{BIC} = n\\log(\\text{MSE}) + d\\log(n)$$",
   "n = len(train)\nbic = {}\n\nfor name, columns in [('A', A), ('B', B), ('C', C)]:\n    ...\n\nbic",
   "The only change from H2 is `2 * d` becoming `d * np.log(n)`.",
   "n = len(train)\nbic = {}\n\nfor name, columns in [('A', A), ('B', B), ('C', C)]:\n"
   "    model = LinearRegression()\n    model.fit(train[columns], train['vol_next'])\n"
   "    mse = mean_squared_error(train['vol_next'], model.predict(train[columns]))\n"
   "    d = len(columns) + 1\n    bic[name] = round(float(n * np.log(mse) + d * np.log(n)), 1)\n\nbic",
   f"{BIC['A']:.0f}, {BIC['B']:.0f}, {BIC['C']:.0f}. BIC prefers "
   f"{min(BIC, key=BIC.get)}. With {N_TRAIN:,} rows, $\\log(n)$ is about "
   f"{np.log(N_TRAIN):.1f}, so BIC charges roughly {np.log(N_TRAIN)/2:.1f} times as "
   "much per parameter as AIC does.")

ex("H4", "Who agrees with whom", 3,
   "Put the winner according to AIC, BIC, shuffled `KFold` and `TimeSeriesSplit` "
   "into one dictionary.",
   "picks = {\n    'AIC': ...,\n    'BIC': ...,\n    'shuffled KFold': ...,\n"
   "    'TimeSeriesSplit': ...,\n}\npicks",
   "You already wrote the AIC and BIC dictionaries in H2 and H3, and the two "
   "cross-validation loops in F5 and G1. `min(d, key=d.get)` picks the smallest "
   "from each.",
   "n = len(train)\naic, bic, kf, ts = {}, {}, {}, {}\n\n"
   "for name, columns in [('A', A), ('B', B), ('C', C)]:\n"
   "    model = LinearRegression()\n    model.fit(train[columns], train['vol_next'])\n"
   "    mse = mean_squared_error(train['vol_next'], model.predict(train[columns]))\n"
   "    d = len(columns) + 1\n    aic[name] = n * np.log(mse) + 2 * d\n"
   "    bic[name] = n * np.log(mse) + d * np.log(n)\n"
   "    kf[name] = -cross_val_score(LinearRegression(), train[columns], train['vol_next'],\n"
   "                                cv=KFold(n_splits=5, shuffle=True, random_state=0),\n"
   "                                scoring='neg_root_mean_squared_error').mean()\n"
   "    ts[name] = -cross_val_score(LinearRegression(), train[columns], train['vol_next'],\n"
   "                                cv=TimeSeriesSplit(n_splits=5),\n"
   "                                scoring='neg_root_mean_squared_error').mean()\n\n"
   "picks = {\n    'AIC': min(aic, key=aic.get),\n    'BIC': min(bic, key=bic.get),\n"
   "    'shuffled KFold': min(kf, key=kf.get),\n    'TimeSeriesSplit': min(ts, key=ts.get),\n}\npicks",
   f"AIC, BIC and shuffled `KFold` all choose {min(AIC, key=AIC.get)}. Only "
   "`TimeSeriesSplit` chooses "
   f"{min(CV_TS, key=CV_TS.get)}, and only `TimeSeriesSplit` agrees with the test "
   "rows. The three that disagree all treat the rows as independent pieces of "
   "evidence, and these rows are nothing of the kind.")

ex("H5", "How many observations are really there", 5,
   "Each row shares nineteen of its twenty days with the row before it, so "
   f"{N_TRAIN:,} rows are nowhere near {N_TRAIN:,} independent observations. Recompute "
   "BIC using `len(train) // 20` in place of $n$, and see whether it changes its "
   "mind.",
   "n_eff = ...\nbic_eff = {}\n\nfor name, columns in [('A', A), ('B', B), ('C', C)]:\n    ...\n\n"
   "print(bic_eff)\nprint('picks:', ...)",
   ["`n_eff = len(train) // 20`. Use it in **both** places where $n$ appears in the "
    "formula.",
    "The MSE itself does not change. Only the two $n$ terms do."],
   "n_eff = len(train) // 20\nbic_eff = {}\n\n"
   "for name, columns in [('A', A), ('B', B), ('C', C)]:\n"
   "    model = LinearRegression()\n    model.fit(train[columns], train['vol_next'])\n"
   "    mse = mean_squared_error(train['vol_next'], model.predict(train[columns]))\n"
   "    d = len(columns) + 1\n    bic_eff[name] = round(float(n_eff * np.log(mse) + d * np.log(n_eff)), 1)\n\n"
   "print(bic_eff)\nprint('picks:', min(bic_eff, key=bic_eff.get))",
   f"With $n = {N_EFF}$ instead of {N_TRAIN:,}, BIC picks "
   f"{min(BIC_EFF, key=BIC_EFF.get)}. The MSE never moved, so the whole change came "
   "from admitting how much evidence there really is.\n\n"
   "That is the honest reading of H4: AIC and BIC are not wrong, they were handed "
   "an $n$ that was twenty times too large. Cross-validation with folds that "
   "respect the calendar never needs that number at all, which is why it survives "
   "on data like this.")

# ============================================================ I
section(
"## \U0001f3d7️ I · The whole workflow, on a stock the lecture never used\n\n"
"Nothing new here. The point is to run the six steps end to end without the "
"lecture holding your hand."
)

ex("I1", "Build Microsoft's table", 2,
   "Build the same four-column table for `MSFT`, drop the incomplete rows, and "
   "split it at the end of 2022 into `m_train` and `m_test`.",
   "msft = ...\n\nm_train = ...\nm_test = ...\nprint(m_train)\nprint(m_test)",
   "This is exercise A4 and A5 again with a different ticker.",
   "msft = pd.DataFrame({\n    'vol_20d': rets['MSFT'].rolling(20).std(),\n"
   "    'vol_60d': rets['MSFT'].rolling(60).std(),\n"
   "    'ret_20d': rets['MSFT'].rolling(20).mean(),\n})\n"
   "msft['vol_next'] = rets['MSFT'].rolling(20).std().shift(-20)\nmsft = msft.dropna()\n\n"
   "m_train = msft.loc[:'2022-12-31']\nm_test = msft.loc['2023-01-01':]\n"
   "print(len(m_train), len(m_test))",
   "The same shape as Apple's table, because the windows and the date range are the "
   "same.",
   revisits="S4")

ex("I2", "Choose a candidate, honestly", 2,
   "Cross-validate the three candidates on `m_train` with five time-ordered folds, "
   "and print each average.",
   "msft = pd.DataFrame({\n    'vol_20d': rets['MSFT'].rolling(20).std(),\n"
   "    'vol_60d': rets['MSFT'].rolling(60).std(),\n"
   "    'ret_20d': rets['MSFT'].rolling(20).mean(),\n})\n"
   "msft['vol_next'] = rets['MSFT'].rolling(20).std().shift(-20)\nmsft = msft.dropna()\n"
   "m_train = msft.loc[:'2022-12-31']\n\n"
   "folds = TimeSeriesSplit(n_splits=5)\n\nfor columns in [A, B, C]:\n    ...",
   "Exactly the loop from F5, with `m_train` in place of `train`.",
   "msft = pd.DataFrame({\n    'vol_20d': rets['MSFT'].rolling(20).std(),\n"
   "    'vol_60d': rets['MSFT'].rolling(60).std(),\n"
   "    'ret_20d': rets['MSFT'].rolling(20).mean(),\n})\n"
   "msft['vol_next'] = rets['MSFT'].rolling(20).std().shift(-20)\nmsft = msft.dropna()\n"
   "m_train = msft.loc[:'2022-12-31']\n\nfolds = TimeSeriesSplit(n_splits=5)\n\n"
   "for columns in [A, B, C]:\n"
   "    scores = cross_val_score(LinearRegression(), m_train[columns], m_train['vol_next'],\n"
   "                             cv=folds, scoring='neg_root_mean_squared_error')\n"
   "    print(len(columns), 'features:', round(-scores.mean(), 4))",
   f"{OTHER['MSFT']['cv']['A']:.4f}, {OTHER['MSFT']['cv']['B']:.4f}, "
   f"{OTHER['MSFT']['cv']['C']:.4f}. One feature wins again, on a stock the lecture "
   "never mentioned.")

ex("I3", "Refit and score once", 3,
   "Refit the winning candidate on **all** of `m_train` and score it once on "
   "`m_test`.",
   "msft = pd.DataFrame({\n    'vol_20d': rets['MSFT'].rolling(20).std(),\n"
   "    'vol_60d': rets['MSFT'].rolling(60).std(),\n"
   "    'ret_20d': rets['MSFT'].rolling(20).mean(),\n})\n"
   "msft['vol_next'] = rets['MSFT'].rolling(20).std().shift(-20)\nmsft = msft.dropna()\n"
   "m_train = msft.loc[:'2022-12-31']\nm_test = msft.loc['2023-01-01':]\n\n"
   "final = ...\nm_rmse = ...\nprint(m_rmse)",
   "Fit `LinearRegression()` on `m_train[A]`, predict on `m_test[A]`, take the root "
   "mean squared error.",
   "msft = pd.DataFrame({\n    'vol_20d': rets['MSFT'].rolling(20).std(),\n"
   "    'vol_60d': rets['MSFT'].rolling(60).std(),\n"
   "    'ret_20d': rets['MSFT'].rolling(20).mean(),\n})\n"
   "msft['vol_next'] = rets['MSFT'].rolling(20).std().shift(-20)\nmsft = msft.dropna()\n"
   "m_train = msft.loc[:'2022-12-31']\nm_test = msft.loc['2023-01-01':]\n\n"
   "final = LinearRegression()\nfinal.fit(m_train[A], m_train['vol_next'])\n"
   "m_rmse = np.sqrt(mean_squared_error(m_test['vol_next'], final.predict(m_test[A])))\n"
   "print(m_rmse)",
   f"{OTHER['MSFT']['test']:.4f}. The folds were only ever a way of choosing. Once "
   "the choice is made the model may use every training row available.")

ex("I4", "Against the two baselines", 3,
   "Score the average and the persistence rule on `m_test` too, and print all three "
   "in order.",
   "msft = pd.DataFrame({\n    'vol_20d': rets['MSFT'].rolling(20).std(),\n"
   "    'vol_60d': rets['MSFT'].rolling(60).std(),\n"
   "    'ret_20d': rets['MSFT'].rolling(20).mean(),\n})\n"
   "msft['vol_next'] = rets['MSFT'].rolling(20).std().shift(-20)\nmsft = msft.dropna()\n"
   "m_train = msft.loc[:'2022-12-31']\nm_test = msft.loc['2023-01-01':]\n\n"
   "results = {\n    'model': ...,\n    'average': ...,\n    'persistence': ...,\n}\n\n"
   "# print them smallest first\nfor name in results:\n    ...",
   "This is exercise C5 with `m_train` and `m_test` in place of `train` and `test`.",
   "msft = pd.DataFrame({\n    'vol_20d': rets['MSFT'].rolling(20).std(),\n"
   "    'vol_60d': rets['MSFT'].rolling(60).std(),\n"
   "    'ret_20d': rets['MSFT'].rolling(20).mean(),\n})\n"
   "msft['vol_next'] = rets['MSFT'].rolling(20).std().shift(-20)\nmsft = msft.dropna()\n"
   "m_train = msft.loc[:'2022-12-31']\nm_test = msft.loc['2023-01-01':]\n\n"
   "model = LinearRegression().fit(m_train[A], m_train['vol_next'])\n\n"
   "results = {\n"
   "    'model': np.sqrt(mean_squared_error(m_test['vol_next'], model.predict(m_test[A]))),\n"
   "    'average': np.sqrt(mean_squared_error(m_test['vol_next'], np.full(len(m_test), m_train['vol_next'].mean()))),\n"
   "    'persistence': np.sqrt(mean_squared_error(m_test['vol_next'], m_test['vol_20d'])),\n"
   "}\n\nfor name in sorted(results, key=results.get):\n    print(f'{name:12} {results[name]:.4f}')",
   f"model {OTHER['MSFT']['test']:.4f}, persistence {OTHER['MSFT']['pers']:.4f}, average "
   f"{OTHER['MSFT']['base']:.4f}. The same ordering as Apple, on a stock chosen "
   "without looking first.")

ex("I5", "And on a quiet stock", 4,
   "Run the whole thing once more on `KO`: build the table, cross-validate the "
   "three candidates, then report the test RMSE of the winner next to persistence.",
   "ko = ...\n\nko_train = ...\nko_test = ...\n\n"
   "# cross-validate, then refit the winner and score once\n...",
   ["Everything you need is in I1 to I4. Change the ticker and nothing else.",
    "Coca-Cola is a much calmer stock, so expect every number to be smaller. What "
    "matters is whether the **ordering** survives."],
   "ko = pd.DataFrame({\n    'vol_20d': rets['KO'].rolling(20).std(),\n"
   "    'vol_60d': rets['KO'].rolling(60).std(),\n"
   "    'ret_20d': rets['KO'].rolling(20).mean(),\n})\n"
   "ko['vol_next'] = rets['KO'].rolling(20).std().shift(-20)\nko = ko.dropna()\n\n"
   "ko_train = ko.loc[:'2022-12-31']\nko_test = ko.loc['2023-01-01':]\n\n"
   "folds = TimeSeriesSplit(n_splits=5)\nfor columns in [A, B, C]:\n"
   "    scores = cross_val_score(LinearRegression(), ko_train[columns], ko_train['vol_next'],\n"
   "                             cv=folds, scoring='neg_root_mean_squared_error')\n"
   "    print(len(columns), 'features, CV:', round(-scores.mean(), 4))\n\n"
   "model = LinearRegression().fit(ko_train[A], ko_train['vol_next'])\n"
   "print('test RMSE  :', round(np.sqrt(mean_squared_error(ko_test['vol_next'], model.predict(ko_test[A]))), 4))\n"
   "print('persistence:', round(np.sqrt(mean_squared_error(ko_test['vol_next'], ko_test['vol_20d'])), 4))",
   f"Cross-validation picks one feature again ({OTHER['KO']['cv']['A']:.4f} against "
   f"{OTHER['KO']['cv']['C']:.4f}), and the model beats persistence "
   f"{OTHER['KO']['test']:.4f} to {OTHER['KO']['pers']:.4f}. Every number is smaller "
   "than Apple's because Coca-Cola moves less, which is why an RMSE always has to be "
   "read next to a baseline from the same data.")

# ============================================================ K
section(
"## \U0001f501 K · Everything you already had\n\n"
"Loops, functions, dictionaries and figures, pointed at the workflow you have "
"just learned."
)

ex("K1", "One function, any ticker", 3,
   "Write `evaluate(ticker)`: it builds the table for that ticker, splits it at the "
   "end of 2022, fits `A` on the training rows, and returns the test RMSE.",
   "def evaluate(ticker):\n    ...\n\nprint('AAPL:', ...)\nprint('NVDA:', ...)",
   ["The body is exercise I3 with `ticker` in place of `'MSFT'`.",
    "Only two columns are needed: `vol_20d` and `vol_next`."],
   "def evaluate(ticker):\n    frame = pd.DataFrame({'vol_20d': rets[ticker].rolling(20).std()})\n"
   "    frame['vol_next'] = rets[ticker].rolling(20).std().shift(-20)\n"
   "    frame = frame.dropna()\n\n"
   "    tr = frame.loc[:'2022-12-31']\n    te = frame.loc['2023-01-01':]\n\n"
   "    model = LinearRegression()\n    model.fit(tr[['vol_20d']], tr['vol_next'])\n"
   "    return np.sqrt(mean_squared_error(te['vol_next'], model.predict(te[['vol_20d']])))\n\n"
   "print('AAPL:', round(evaluate('AAPL'), 4))\nprint('NVDA:', round(evaluate('NVDA'), 4))",
   f"Apple {WIN['AAPL'][0]:.4f}, Nvidia {WIN['NVDA'][0]:.4f}. Apple's number differs "
   f"slightly from the {RMSE_TEST:.4f} earlier in the notebook, because dropping "
   "`vol_60d` also drops the rows where a sixty-day window had not yet filled. A "
   "function is the right shape here because the next exercise wants this done "
   "eleven times.",
   revisits="S2")

ex("K2", "Does it beat persistence everywhere", 4,
   "For every ticker in `rets`, compute the model's test RMSE and the persistence "
   "rule's test RMSE. Count how many of the eleven the model wins.",
   "results = {}\n\nfor ticker in rets.columns:\n    ...\n\n"
   "wins = ...\nprint('model wins on', ..., 'of', ...)",
   ["Store a pair in the dictionary: `results[ticker] = (model_rmse, pers_rmse)`.",
    "`sum(1 for t in results if results[t][0] < results[t][1])` counts the wins."],
   "results = {}\n\nfor ticker in rets.columns:\n"
   "    frame = pd.DataFrame({'vol_20d': rets[ticker].rolling(20).std()})\n"
   "    frame['vol_next'] = rets[ticker].rolling(20).std().shift(-20)\n"
   "    frame = frame.dropna()\n\n"
   "    tr = frame.loc[:'2022-12-31']\n    te = frame.loc['2023-01-01':]\n\n"
   "    model = LinearRegression()\n    model.fit(tr[['vol_20d']], tr['vol_next'])\n"
   "    model_rmse = np.sqrt(mean_squared_error(te['vol_next'], model.predict(te[['vol_20d']])))\n"
   "    pers_rmse = np.sqrt(mean_squared_error(te['vol_next'], te['vol_20d']))\n"
   "    results[ticker] = (model_rmse, pers_rmse)\n\n"
   "wins = sum(1 for t in results if results[t][0] < results[t][1])\n"
   "print('model wins on', wins, 'of', len(results))",
   f"{N_WIN} out of {len(ALL_T)}. A rule chosen on one stock, with the choice made "
   "without ever looking at 2023 or 2024, transfers to every other name in the "
   "universe. That is a far stronger result than a single number on a single stock.",
   revisits="S2")

ex("K3", "Draw the eleven", 3,
   "Draw a bar chart of the improvement, `persistence - model`, for the eleven "
   "tickers, sorted from largest to smallest.",
   "results = {}\nfor ticker in rets.columns:\n    ...\n\n"
   "improvement = ...\n\nfig, ax = plt.subplots(figsize=(8, 3.2))\n...\nplt.show()",
   ["Reuse the loop from K2, then "
    "`pd.Series(improvement).sort_values(ascending=False)`.",
    "`ax.bar(ranked.index, ranked.values)` draws it, and "
    "`ax.axhline(0, color='black')` marks the line a bar has to clear."],
   "improvement = {}\n\nfor ticker in rets.columns:\n"
   "    frame = pd.DataFrame({'vol_20d': rets[ticker].rolling(20).std()})\n"
   "    frame['vol_next'] = rets[ticker].rolling(20).std().shift(-20)\n"
   "    frame = frame.dropna()\n\n"
   "    tr = frame.loc[:'2022-12-31']\n    te = frame.loc['2023-01-01':]\n\n"
   "    model = LinearRegression().fit(tr[['vol_20d']], tr['vol_next'])\n"
   "    model_rmse = np.sqrt(mean_squared_error(te['vol_next'], model.predict(te[['vol_20d']])))\n"
   "    pers_rmse = np.sqrt(mean_squared_error(te['vol_next'], te['vol_20d']))\n"
   "    improvement[ticker] = pers_rmse - model_rmse\n\n"
   "ranked = pd.Series(improvement).sort_values(ascending=False)\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3.2))\nax.bar(ranked.index, ranked.values)\n"
   "ax.axhline(0, color='black', linewidth=1)\n"
   "ax.set_ylabel('RMSE saved (percentage points)')\n"
   "ax.set_title('How much the model beats persistence by, 2023 to 2024', loc='left')\n"
   "plt.show()",
   f"Every bar is above the line, and {BIGGEST_T} gains the most in relative terms. "
   "The stocks that move most are the ones where a forecast is worth the most, which "
   "is convenient rather than surprising.",
   revisits="S3")

ex("K4", "Say it in one sentence", 3,
   "Print a single line reporting the chosen model, its test RMSE and the "
   "persistence RMSE, each to three decimals, using an f-string.",
   "model = LinearRegression()\nmodel.fit(train[A], train['vol_next'])\n"
   "model_rmse = np.sqrt(mean_squared_error(test['vol_next'], model.predict(test[A])))\n"
   "pers_rmse = np.sqrt(mean_squared_error(test['vol_next'], test['vol_20d']))\n\n"
   "sentence = ...\nprint(sentence)",
   "`f'...{value:.3f}...'` rounds inside the string. Name the feature set, both "
   "numbers, and the years the test block covers.",
   "model = LinearRegression()\nmodel.fit(train[A], train['vol_next'])\n"
   "model_rmse = np.sqrt(mean_squared_error(test['vol_next'], model.predict(test[A])))\n"
   "pers_rmse = np.sqrt(mean_squared_error(test['vol_next'], test['vol_20d']))\n\n"
   "sentence = (f'A linear regression on vol_20d scores {model_rmse:.3f} on 2023 to 2024, '\n"
   "            f'against {pers_rmse:.3f} for repeating last month.')\nprint(sentence)",
   "One sentence with a model, a number and a comparison. That is the smallest "
   "honest unit of a result, and a surprising amount of published work never gets "
   "as far as the comparison.",
   revisits="S1")

ex("K5", "The whole workflow, as one table", 5,
   "Build a DataFrame with one row per ticker and three columns: the "
   "cross-validation error of the chosen candidate on the training block, the test "
   "RMSE, and the persistence RMSE. Sort it by test RMSE.",
   "rows = []\n\nfor ticker in rets.columns:\n    ...\n\nsummary = ...\nsummary",
   ["Per ticker: build the table, split it, cross-validate `A` on the training "
    "block, refit on all of it, then score both rules on the test block.",
    "Append a dictionary per ticker and finish with "
    "`pd.DataFrame(rows).set_index('ticker').sort_values('test')`."],
   "rows = []\n\nfor ticker in rets.columns:\n"
   "    frame = pd.DataFrame({'vol_20d': rets[ticker].rolling(20).std()})\n"
   "    frame['vol_next'] = rets[ticker].rolling(20).std().shift(-20)\n"
   "    frame = frame.dropna()\n\n"
   "    tr = frame.loc[:'2022-12-31']\n    te = frame.loc['2023-01-01':]\n\n"
   "    cv = cross_val_score(LinearRegression(), tr[['vol_20d']], tr['vol_next'],\n"
   "                         cv=TimeSeriesSplit(n_splits=5),\n"
   "                         scoring='neg_root_mean_squared_error')\n\n"
   "    model = LinearRegression().fit(tr[['vol_20d']], tr['vol_next'])\n\n"
   "    rows.append({\n        'ticker': ticker,\n        'cv': -cv.mean(),\n"
   "        'test': np.sqrt(mean_squared_error(te['vol_next'], model.predict(te[['vol_20d']]))),\n"
   "        'persistence': np.sqrt(mean_squared_error(te['vol_next'], te['vol_20d'])),\n"
   "    })\n\nsummary = pd.DataFrame(rows).set_index('ticker').sort_values('test')\nsummary",
   "The `cv` column is always larger than the `test` column, because each fold fits "
   "on fewer rows and scores on a harder stretch of history. That gap is normal and "
   "worth expecting: cross-validation is a conservative estimate, not a prediction "
   "of the test score.\n\n"
   "This table is the entire session in one object. If you can produce it from a "
   "blank cell, you can do everything Session 5 asked of you.",
   revisits="S2")

ex("K6", "Calm days and busy days", 4,
   "Split the test block in two with a **boolean mask**: days whose `vol_20d` is "
   "below its median, and the rest. Score the model separately on each half.\n\n"
   "Is the forecast equally good in quiet markets and busy ones?",
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n"
   "predictions = model.predict(test[['vol_20d']])\nerrors = test['vol_next'] - predictions\n\n"
   "calm = ...\n\nprint('calm days:', ...)\nprint('busy days:', ...)",
   ["`calm = test['vol_20d'] < test['vol_20d'].median()` is a column of True and "
    "False, one per test day.",
    "`errors[calm]` keeps the calm days and `errors[~calm]` keeps the rest. The "
    "`~` flips a mask, exactly as in Session 4's confusion matrix."],
   "model = LinearRegression()\nmodel.fit(train[['vol_20d']], train['vol_next'])\n"
   "predictions = model.predict(test[['vol_20d']])\nerrors = test['vol_next'] - predictions\n\n"
   "calm = test['vol_20d'] < test['vol_20d'].median()\n\n"
   "print('calm days:', round(np.sqrt((errors[calm] ** 2).mean()), 4))\n"
   "print('busy days:', round(np.sqrt((errors[~calm] ** 2).mean()), 4))",
   f"{RMSE_CALM:.4f} on the {N_CALM} calmest test days against {RMSE_BUSY:.4f} on "
   f"the rest, so the forecast is about {100 * (RMSE_BUSY / RMSE_CALM - 1):.0f}% "
   "worse when the market is busy.\n\n"
   "That is the direction you would expect and it is worth stating, because the "
   "busy half is the half a risk model exists for. A single RMSE averages the two "
   "together and hides it.\n\n"
   "Two masks, a `~`, and no loop anywhere. Everything in this exercise is "
   "Session 3 arithmetic pointed at a Session 5 question.",
   revisits="S3")

# ---------------------------------------------------------------- closing
md(
"## \U0001f3c1 Done\n\n"
"You have fitted a model, scored it against two rules that use no model at all, "
"watched a single validation split give two different answers, cross-validated "
"with folds that follow the calendar, and run the whole thing on stocks the "
"lecture never mentioned.\n\n"
"The one habit worth carrying out of this notebook: no score means anything "
"until it sits next to a baseline and a statement of what was allowed to see "
"what.\n\n"
"**Next:** open `session_05_case.ipynb` (*The Analyst's Notebook, Part 5*), where "
"the risk report finally gets a model and has to justify it.\n\n"
"*Stuck for more than 15 minutes on anything? Ask a friend, ask an AI for a hint "
"(not the answer), or email me at `jobo@econ.au.dk`.*"
)

# ---------------------------------------------------------------- write
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
for _i, _c in enumerate(nb.cells):
    _c["id"] = f"c{_i:04d}"

OUT.write_text(nbf.writes(nb), encoding="utf-8")

import re as _re

_heads = [c.source.splitlines()[0] for c in cells
          if c.cell_type == "markdown" and _re.match(r"^### [A-Z]\d+ ", c.source)]
n_ex = len(_heads)
tiers = {}
for _h in _heads:
    _m = _re.search(r"(★+)", _h)
    if _m:
        _k = len(_m.group(1))
        tiers[_k] = tiers.get(_k, 0) + 1
tiers = {f"{k}star": tiers[k] for k in sorted(tiers)}
n_revisit = sum(1 for _h in _heads if "revisits" in _h)
print(f"wrote {OUT}  ({len(cells)} cells, {n_ex} exercises)")
print("stars:", tiers, " revisits tags:", n_revisit)
