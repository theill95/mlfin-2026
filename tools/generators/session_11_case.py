# -*- coding: utf-8 -*-
"""Build session_11_case.ipynb  (The Analyst's Notebook, Part 11).

Conventions (approved for Sessions 1 to 9): no star badges, one cumulative
investigation where later questions reuse what earlier ones stored, folded
hints and solutions, stated formulas, plain explanatory tone, no em-dashes.
Part 9 is the last part before this one (Sessions 7 and 10 have no case part).
The QUICK LOAD restores what the risk report holds after Part 9, as the numbers
those parts printed: the volatility forecast (Part 5's target, Part 6's models,
test block and folds, and Part 6's desk) and the jump warning (Part 9's label,
AUCs on the test block and the folds, and threshold). Nothing is refitted to
get them back; Q1 refits the one-column forecast once and checks it with True.

The course has moved on from classification to a new kind of model, and so
does the case: trees and forests, for the number the report was built on.
Every model in the report so far is linear in its columns. Part 11 holds trees
and forests to what the report has established. On Apple: no single tree on
twenty columns beats forecasting the training average, the stump on vol_20d
comes within 0.00004 of the one column, a tree of depth 2 on vol_20d and
ret_20d edges it on the test block by 0.00001 and loses on the folds, and a
forest tuned on the folds draws level with ridge on both the test block and the
folds; across the desk it beats ridge on 5 of the 11. One light question at the
end takes the same forest to Part 9's jump label: 0.869 on the test block, below
the one column's 0.889 and above the twenty columns' 0.746, but 0.682 on the
folds, below both (0.792 and 0.755). The warning stays as Part 9 left it.

Returns stay in plain decimals here, as in Parts 1 to 9.

BLANK-SAFE, and this one needs care because the case is cumulative: no
pre-written line may CALL anything on a variable an earlier question produced.
Every such dependency sits inside the student's own blank.
"""
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "session_11" / "session_11_case.ipynb"

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


def rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def logit_pipe(**kw):
    return Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(**kw))])


# ---- the real numbers, so every note is exact ------------------------------
PX = pd.read_csv(ROOT / "data" / "prices.csv", parse_dates=["date"])
W = PX.pivot(index="date", columns="ticker", values="close")
R = W.pct_change()
TICKERS = sorted(W.columns)
TS5 = TimeSeriesSplit(n_splits=5)
GRID = [1, 10, 100, 1000, 10000]
SWEEP = np.arange(0.05, 0.60, 0.01)


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


def ridge_search():
    return GridSearchCV(Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]), {"ridge__alpha": GRID},
                        cv=TS5, scoring="neg_root_mean_squared_error")


def cost_at(y, p, threshold):
    tn, fp, fn, tp = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1]).ravel()
    return int(5 * fn + fp)


TBL = wide_table("AAPL")
COLS = list(TBL.columns[:-1])
TRAIN, TEST = TBL.loc[:"2022-12-31"], TBL.loc["2023-01-01":]
Y, YT = TRAIN["vol_next"], TEST["vol_next"]
J = (TRAIN["vol_next"] > 1.5 * TRAIN["vol_20d"]).astype(int)
JT = (TEST["vol_next"] > 1.5 * TEST["vol_20d"]).astype(int)
N_TBL, N_TRAIN, N_TEST = len(TBL), len(TRAIN), len(TEST)

# ---- what the report holds after Part 9: the warning, as Part 9 printed it ---
JUMP = logit_pipe().fit(TRAIN[["vol_20d"]], J)
P9_AUC = float(roc_auc_score(JT, JUMP.predict_proba(TEST[["vol_20d"]])[:, 1]))
_p_tr = JUMP.predict_proba(TRAIN[["vol_20d"]])[:, 1]
P9_CHOSEN = round(float(SWEEP[int(np.argmin([cost_at(J, _p_tr, t) for t in SWEEP]))]), 2)
P9_FOLDS = float(cross_val_score(logit_pipe(), TRAIN[["vol_20d"]], J, cv=TS5, scoring="roc_auc").mean())
_wide9 = GridSearchCV(logit_pipe(max_iter=1000), {"logit__C": [0.0001, 0.001, 0.01, 0.1, 1, 10]}, cv=TS5,
                      scoring="roc_auc").fit(TRAIN[COLS], J)
P9_WIDE_AUC = float(roc_auc_score(JT, _wide9.predict_proba(TEST[COLS])[:, 1]))
P9_WIDE_FOLDS = float(_wide9.best_score_)
assert P9_CHOSEN == 0.25 and round(P9_AUC, 4) == 0.8892 and round(P9_FOLDS, 4) == 0.7923
assert round(P9_WIDE_AUC, 4) == 0.7458 and round(P9_WIDE_FOLDS, 4) == 0.7547

# ---- and the forecast, as Part 6 printed it ---------------------------------
ONE = LinearRegression().fit(TRAIN[["vol_20d"]], Y)
ONE_TE = rmse(YT, ONE.predict(TEST[["vol_20d"]]))
ONE_CV = float(-cross_val_score(LinearRegression(), TRAIN[["vol_20d"]], Y, cv=TS5,
                                scoring="neg_root_mean_squared_error").mean())
MEAN_TE = rmse(YT, np.full(N_TEST, Y.mean()))
SEARCH = ridge_search().fit(TRAIN[COLS], Y)
RIDGE_TE = rmse(YT, SEARCH.predict(TEST[COLS]))
RIDGE_ALPHA, RIDGE_CV = SEARCH.best_params_["ridge__alpha"], float(-SEARCH.best_score_)
assert (round(ONE_TE, 5), round(ONE_CV, 5), round(RIDGE_TE, 5), round(RIDGE_CV, 5)) == (0.00412, 0.00755, 0.0046, 0.00761)
DESK = {}
for _t in TICKERS:
    _tb = wide_table(_t)
    _tr, _te = _tb.loc[:"2022-12-31"], _tb.loc["2023-01-01":]
    _cc = list(_tb.columns[:-1])
    _g = ridge_search().fit(_tr[_cc], _tr["vol_next"])
    _o = LinearRegression().fit(_tr[["vol_20d"]], _tr["vol_next"])
    DESK[_t] = (round(rmse(_te["vol_next"], _g.predict(_te[_cc])), 5),
                round(rmse(_te["vol_next"], _o.predict(_te[["vol_20d"]])), 5))
P6_WINS = sum(1 for t in TICKERS if DESK[t][0] < DESK[t][1])
assert RIDGE_ALPHA == 1000 and P6_WINS == 7 and DESK["AAPL"][0] > DESK["AAPL"][1]

# ---- Part 11 ------------------------------------------------------------------
# Q2 the stump on vol_20d
STUMP = DecisionTreeRegressor(max_depth=1, random_state=0).fit(TRAIN[["vol_20d"]], Y)
STUMP_CUT = float(STUMP.tree_.threshold[0])
_calm = TRAIN["vol_20d"] <= STUMP_CUT
N_CALM, N_BUSY = int(_calm.sum()), int((~_calm).sum())
CALM_MEAN, BUSY_MEAN = float(Y[_calm].mean()), float(Y[~_calm].mean())
STUMP_TE = rmse(YT, STUMP.predict(TEST[["vol_20d"]]))
TEST_BUSY = int((TEST["vol_20d"] > STUMP_CUT).sum())
STUMP_TR = rmse(Y, STUMP.predict(TRAIN[["vol_20d"]]))

# Q3 the stump on twenty columns
S20 = DecisionTreeRegressor(max_depth=1, random_state=0).fit(TRAIN[COLS], Y)
S20_COL = COLS[S20.tree_.feature[0]]
S20_CUT = float(S20.tree_.threshold[0])
S20_TE = rmse(YT, S20.predict(TEST[COLS]))
S20_TR = rmse(Y, S20.predict(TRAIN[COLS]))
S20_TEST_RIGHT = int((TEST[S20_COL] > S20_CUT).sum())
assert S20_COL == "DIS_vol" and S20_TR < STUMP_TR and S20_TE > MEAN_TE

# Q4 growing
GROWTH = {}
for _d in [1, 2, 3, 4, 6, 10, None]:
    _m = DecisionTreeRegressor(max_depth=_d, random_state=0).fit(TRAIN[COLS], Y)
    GROWTH[_d] = (int(_m.get_n_leaves()), int(_m.get_depth()), rmse(Y, _m.predict(TRAIN[COLS])), rmse(YT, _m.predict(TEST[COLS])))
FULL_LEAVES, FULL_DEPTH = GROWTH[None][0], GROWTH[None][1]
BEST_TREE_D = min(GROWTH, key=lambda d: GROWTH[d][3])
BEST_TREE_TE = GROWTH[BEST_TREE_D][3]
assert GROWTH[None][2] < 1e-9 and BEST_TREE_TE > MEAN_TE

# Q5 the folds choose a size
GD = GridSearchCV(DecisionTreeRegressor(random_state=0), {"max_depth": [1, 2, 3, 4, 6, 8]}, cv=TS5,
                  scoring="neg_root_mean_squared_error").fit(TRAIN[COLS], Y)
GL = GridSearchCV(DecisionTreeRegressor(random_state=0), {"min_samples_leaf": [20, 50, 100, 200, 400]}, cv=TS5,
                  scoring="neg_root_mean_squared_error").fit(TRAIN[COLS], Y)
GD_BEST, GD_CV, GD_TE = GD.best_params_["max_depth"], float(-GD.best_score_), rmse(YT, GD.predict(TEST[COLS]))
GL_BEST, GL_CV, GL_TE = GL.best_params_["min_samples_leaf"], float(-GL.best_score_), rmse(YT, GL.predict(TEST[COLS]))
GL_LEAVES = int(GL.best_estimator_.get_n_leaves())
assert GD_BEST == 1 and GD_CV > ONE_CV and GL_CV > ONE_CV

# Q6 two columns, four leaves, drawn
PAIR = ["vol_20d", "ret_20d"]
PT = DecisionTreeRegressor(max_depth=2, random_state=0).fit(TRAIN[PAIR], Y)
PAIR_ASKED = [PAIR[i] for i in PT.tree_.feature if i >= 0]
_inner = [i for i in range(PT.tree_.node_count) if PT.tree_.feature[i] >= 0]
_leaves = [i for i in range(PT.tree_.node_count) if PT.tree_.feature[i] < 0]
PT_CUTS = [float(PT.tree_.threshold[i]) for i in _inner]
PT_VALUES = [float(PT.tree_.value[i].ravel()[0]) for i in _leaves]
PT_BUSIEST_N = int(PT.tree_.n_node_samples[_leaves[-1]])
PAIR_TE = rmse(YT, PT.predict(TEST[PAIR]))
PAIR_CV = float(-cross_val_score(DecisionTreeRegressor(max_depth=2, random_state=0), TRAIN[PAIR], Y, cv=TS5,
                                 scoring="neg_root_mean_squared_error").mean())
assert PAIR_ASKED == ["vol_20d"] * 3 and PAIR_TE < ONE_TE and PAIR_CV > ONE_CV
assert _inner == [0, 1, 4] and PT_CUTS[0] == STUMP_CUT and PT_VALUES == sorted(PT_VALUES)

# Q7 bagging and a forest, timed
_t0 = time.perf_counter()
BAG = RandomForestRegressor(n_estimators=100, max_features=1.0, random_state=0).fit(TRAIN[COLS], Y)
BAG_S = time.perf_counter() - _t0
_t0 = time.perf_counter()
RF = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=0).fit(TRAIN[COLS], Y)
RF_S = time.perf_counter() - _t0
BAG_TE, RF_TE = rmse(YT, BAG.predict(TEST[COLS])), rmse(YT, RF.predict(TEST[COLS]))
assert BAG_TE > MEAN_TE > RF_TE > ONE_TE

# Q8 out-of-bag against the folds
RF_OOB_MODEL = RandomForestRegressor(n_estimators=100, max_features="sqrt", oob_score=True, random_state=0).fit(TRAIN[COLS], Y)
RF_OOB = rmse(Y, RF_OOB_MODEL.oob_prediction_)
RF_CV = float(-cross_val_score(RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=0),
                               TRAIN[COLS], Y, cv=TS5, scoring="neg_root_mean_squared_error").mean())
assert rmse(YT, RF_OOB_MODEL.predict(TEST[COLS])) == RF_TE          # oob_score changes nothing else

# Q9 what the grid will cost
FGRID = {"max_features": [1, 3, 5, 10, 20], "min_samples_leaf": [5, 20, 50, 100, 200]}
N_COMB = len(FGRID["max_features"]) * len(FGRID["min_samples_leaf"])
N_FITS = N_COMB * 5 + 1
TIMINGS = {}
for _mf, _leaf in [(20, 5), (1, 200)]:
    _f = RandomForestRegressor(n_estimators=100, max_features=_mf, min_samples_leaf=_leaf, random_state=0)
    _t0 = time.perf_counter()
    _f.fit(TRAIN[COLS], Y)
    TIMINGS[(_mf, _leaf)] = time.perf_counter() - _t0
T_SLOW, T_FAST = TIMINGS[(20, 5)], TIMINGS[(1, 200)]

# Q10 the search
_t0 = time.perf_counter()
FS = GridSearchCV(RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1), FGRID, cv=TS5,
                  scoring="neg_root_mean_squared_error").fit(TRAIN[COLS], Y)
FS_S = time.perf_counter() - _t0
FS_BEST = FS.best_params_
FS_CV, FS_TE = float(-FS.best_score_), rmse(YT, FS.predict(TEST[COLS]))
assert FS_BEST == {"max_features": 5, "min_samples_leaf": 100}
assert ONE_TE < RIDGE_TE < FS_TE < RF_TE and ONE_CV < FS_CV < RIDGE_CV
assert FS_TE - RIDGE_TE < 0.00003 and RIDGE_CV - FS_CV < 0.00005      # "level with ridge"

# Q11 the desk
DESK_FOREST = {}
for _t in TICKERS:
    _tb = wide_table(_t)
    _tr, _te = _tb.loc[:"2022-12-31"], _tb.loc["2023-01-01":]
    _cc = list(_tb.columns[:-1])
    _f = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1, max_features=FS_BEST["max_features"],
                               min_samples_leaf=FS_BEST["min_samples_leaf"]).fit(_tr[_cc], _tr["vol_next"])
    DESK_FOREST[_t] = rmse(_te["vol_next"], _f.predict(_te[_cc]))
BEATS_RIDGE = [t for t in TICKERS if DESK_FOREST[t] < DESK[t][0]]
BEATS_ONE = [t for t in TICKERS if DESK_FOREST[t] < DESK[t][1]]
assert len(BEATS_RIDGE) == 5 and len(BEATS_ONE) == 4 and "AAPL" not in BEATS_RIDGE

# Q12 the levels in the figure
F_MEAN = float(FS.predict(TEST[COLS]).mean())
O_MEAN = float(ONE.predict(TEST[["vol_20d"]]).mean())
A_MEAN = float(YT.mean())
_above = [float((p > YT.values).mean()) for p in (FS.predict(TEST[COLS]), ONE.predict(TEST[["vol_20d"]]),
                                                  STUMP.predict(TEST[["vol_20d"]]))]
assert all(0.70 <= a <= 0.80 for a in _above), _above      # "about three test days in four"
assert np.abs(np.diff(FS.predict(TEST[COLS]))).mean() > np.abs(np.diff(ONE.predict(TEST[["vol_20d"]]))).mean()

# Q13 the same forest, once, on Part 9's label
JF = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1, max_features=FS_BEST["max_features"],
                            min_samples_leaf=FS_BEST["min_samples_leaf"]).fit(TRAIN[COLS], J)
JF_AUC = float(roc_auc_score(JT, JF.predict_proba(TEST[COLS])[:, 1]))
JF_CV = float(cross_val_score(JF, TRAIN[COLS], J, cv=TS5, scoring="roc_auc").mean())
assert P9_WIDE_AUC < JF_AUC < P9_AUC and JF_CV < P9_WIDE_FOLDS < P9_FOLDS

# ---------------------------------------------------------------- top matter
md(
"# \U0001f4bc The Analyst's Notebook · Part 11\n"
"### Trees and forests for the forecast\n\n"
"After Part 9 the risk report holds two results. The first is the volatility "
"forecast it was built around: next month's volatility from this month's "
"alone, the one-column model chosen in Part 5, which in Part 6 held off every "
f"column the desk could offer, {ONE_TE:.5f} against {RIDGE_TE:.5f} for ridge on "
"twenty. The second is Part 9's jump warning: a one-column classifier with an "
f"AUC of {P9_AUC:.3f}, run at a threshold of {P9_CHOSEN} chosen from the desk's "
"costs.\n\n"
"Every model in the report so far is linear in its columns: one slope per "
"column, the same in a calm month as in a crash. This part tries a different "
"kind of model. A tree asks a question of one column and forecasts differently "
"on each side of the answer, and a forest averages many trees. Part 11 gives "
"them the question the report was built on, next month's volatility, and holds "
"them to what the report has already established: the one-column forecast, "
"ridge, the folds and the desk. The last question takes the forest to Part 9's "
"jump label, once."
)

md(
"## How to work through this\n\n"
"- Run the **quick load** cell first. It brings back what the risk report holds "
"after Part 9 and loads the price table.\n"
"- Each question builds on the last, so keep them in order and keep your "
"variables. Later questions use the names earlier ones created.\n"
"- Cells with `...` are blanks. The notebook runs cleanly even before you fill "
"them in, so **Run all** is always safe.\n"
"- Hints and solutions are folded under each question. Work first, then check.\n"
"- Q10 runs a grid search over forests and takes about a minute on Colab.\n\n"
"**A note on units.** The lecture worked in percent, on the index table. This "
"notebook keeps the plain decimals of Parts 1 to 9 and stays on Apple, so every "
"number compares directly with the report's. A tree does not mind the units: a "
"cut at 0.0175 in decimals is the same cut as 1.75 in percent, and no column "
"needs scaling.\n\n"
"*Stuck for more than 15 minutes? Ask a friend, ask an AI for a hint (not the "
"answer), or email me at `jobo@econ.au.dk`.*"
)

md("---")

# ------------------------------------------------------------- quick load
md(
"## ⚙️ Quick load\n\n"
"The packages, the price table, and what the risk report holds after Part 9. "
"Run it and read what it prints."
)

_desk_lines = ",\n".join(f"    '{t}': ({DESK[t][0]:.5f}, {DESK[t][1]:.5f})" for t in TICKERS)

code(
'''import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score
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

# --- What the risk report holds after Part 9, on Apple ---
part9_split = "by date: train to 2022-12-31, test from 2023-01-01"

# the forecast: Part 5's target, Part 6's models
part5_target = "sd of daily returns over the next 20 trading days"
part6_one_rmse = ''' + f"{ONE_TE:.5f}" + '''        # linear regression on vol_20d, the test block
part6_one_folds = ''' + f"{ONE_CV:.5f}" + '''       # the same model on the five time-ordered folds
part6_ridge_rmse = ''' + f"{RIDGE_TE:.5f}" + '''      # ridge on all twenty columns, alpha chosen on the folds
part6_ridge_folds = ''' + f"{RIDGE_CV:.5f}" + '''     # the same search, its score on the folds
part6_desk = {                  # (ridge, one column) on each instrument's test block
''' + _desk_lines + '''
}

# the warning: Part 9's label and model
part9_label = "jump: vol_next more than 1.5 times vol_20d"
part9_auc = ''' + f"{P9_AUC:.4f}" + '''              # one column, logistic regression, the test block
part9_folds_auc = ''' + f"{P9_FOLDS:.4f}" + '''        # the same model on the folds
part9_wide_auc = ''' + f"{P9_WIDE_AUC:.4f}" + '''         # twenty columns, C chosen on the folds, the test block
part9_wide_folds_auc = ''' + f"{P9_WIDE_FOLDS:.4f}" + '''   # the same model on the folds
part9_threshold = ''' + f"{P9_CHOSEN}" + '''          # chosen from the desk's costs: a miss 5, a false alarm 1

print("Loaded prices:", prices.shape[0], "rows")
print("Instruments  :", ", ".join(TICKERS))
print()
print("After Part 9 the risk report holds, on Apple:")
print(f"  a forecast: one column, RMSE {part6_one_rmse:.5f} on the test block and {part6_one_folds:.5f} on the folds")
print(f"              (ridge on twenty columns: {part6_ridge_rmse:.5f} and {part6_ridge_folds:.5f})")
print(f"  a warning : one column, AUC {part9_auc:.3f} on the test block, run at a threshold of {part9_threshold}")
print()
print("Every model in it is linear in its columns. Today the model changes.")'''
)

md("---")

# ==================================================================== Q1
q("Q1", "Where the report stands",
  "Write `wide_table(ticker)` returning the table Parts 6, 8 and 9 all worked "
  "on, for one instrument: the six volatility windows `[5, 10, 20, 40, 60, 120]` "
  "as `vol_<w>d`, the three return windows `[5, 20, 60]` as `ret_<w>d`, "
  "`up_20d`, the 20-day volatility of every **other** instrument as "
  "`<ticker>_vol`, the target `vol_next`, and incomplete rows dropped. Build "
  "Apple's table with it, store the twenty feature names in `columns`, and "
  "split at the end of 2022. Then refit the report's forecast, the one-column "
  "linear regression, as `one_model`, and check its test RMSE, `one_rmse`, "
  "against `part6_one_rmse`.",
  "def wide_table(ticker):\n    \"\"\"...\"\"\"\n    ...\n\n\n"
  "table = ...\ncolumns = ...\ntrain = ...\ntest = ...\n\n"
  "one_model = ...\n...\none_rmse = ...\n\nprint(one_rmse)\nprint('matches the report:', ...)",
  ["Column names are text built from the number: `'vol_' + str(w) + 'd'`. "
   "Inside the last loop, `if t != ticker:`. The target is "
   "`rets[ticker].rolling(20).std().shift(-20)`.",
   "`columns = list(table.columns[:-1])`, and the check is "
   "`abs(one_rmse - part6_one_rmse) < 0.00001`."],
  "def wide_table(ticker):\n"
  "    \"\"\"The report's twenty columns and the target for one instrument, in plain decimals.\"\"\"\n"
  "    frame = pd.DataFrame()\n"
  "    for w in [5, 10, 20, 40, 60, 120]:\n        frame['vol_' + str(w) + 'd'] = rets[ticker].rolling(w).std()\n"
  "    for w in [5, 20, 60]:\n        frame['ret_' + str(w) + 'd'] = rets[ticker].rolling(w).mean()\n"
  "    frame['up_20d'] = (rets[ticker] > 0).rolling(20).mean()\n"
  "    for t in TICKERS:\n        if t != ticker:\n            frame[t + '_vol'] = rets[t].rolling(20).std()\n"
  "    frame['vol_next'] = rets[ticker].rolling(20).std().shift(-20)\n"
  "    return frame.dropna()\n\n\n"
  "table = wide_table('AAPL')\ncolumns = list(table.columns[:-1])\n"
  "train = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\n\n"
  "one_model = LinearRegression()\none_model.fit(train[['vol_20d']], train['vol_next'])\n"
  "one_rmse = rmse(test['vol_next'], one_model.predict(test[['vol_20d']]))\n\n"
  "print(one_rmse)\nprint('matches the report:', abs(one_rmse - part6_one_rmse) < 0.00001)",
  f"{ONE_TE:.5f}, and `True`: {N_TBL:,} rows, {N_TRAIN:,} to fit on and {N_TEST} "
  "to check on. The table is a function now because Q11 runs it for every "
  "instrument; nothing else about it has changed. The forecast has two scores "
  f"in the report, {ONE_TE:.5f} on the test block, which is two calm years, and "
  f"{ONE_CV:.5f} on the folds, which include 2020. Every model in this part is "
  "held to both, and one that wins on one and loses on the other has not won.")

# ==================================================================== Q2
q("Q2", "One question, checked by hand",
  "Fit a tree with one question, `DecisionTreeRegressor(max_depth=1, "
  "random_state=0)`, on `vol_20d` alone, as `stump`. Its cut is "
  "`stump.tree_.threshold[0]`. With a mask on the training days, count the "
  "days on each side of the cut and print the mean `vol_next` on each side: "
  "those are the stump's two forecasts. Finish with its test RMSE beside "
  "`one_rmse`.",
  "stump = ...\n...\ncut = ...\ncalm = ...\n\n"
  "print('cut:', cut)\nprint('calm side:', ..., 'days, forecast', ...)\n"
  "print('busy side:', ..., 'days, forecast', ...)\n"
  "stump_rmse = ...\nprint('stump:', stump_rmse, ' one column:', one_rmse)",
  ["`calm = train['vol_20d'] <= cut` is True on the calm side. `calm.sum()` "
   "counts it and `~calm` is the other side.",
   "`train.loc[calm, 'vol_next'].mean()` is the calm forecast."],
  "stump = DecisionTreeRegressor(max_depth=1, random_state=0)\n"
  "stump.fit(train[['vol_20d']], train['vol_next'])\n"
  "cut = stump.tree_.threshold[0]\ncalm = train['vol_20d'] <= cut\n\n"
  "print('cut:', round(cut, 5))\n"
  "print('calm side:', calm.sum(), 'days, forecast', round(train.loc[calm, 'vol_next'].mean(), 5))\n"
  "print('busy side:', (~calm).sum(), 'days, forecast', round(train.loc[~calm, 'vol_next'].mean(), 5))\n"
  "stump_rmse = rmse(test['vol_next'], stump.predict(test[['vol_20d']]))\n"
  "print('stump:', stump_rmse, ' one column:', one_rmse)",
  f"A cut at {STUMP_CUT:.4f}: {N_CALM:,} calm days forecast {CALM_MEAN:.5f} and "
  f"{N_BUSY} busy days forecast {BUSY_MEAN:.5f}, the means of their boxes. The "
  f"stump scores {STUMP_TE:.5f} against {ONE_TE:.5f}. Two numbers come within "
  f"{STUMP_TE - ONE_TE:.5f} of a fitted slope, partly because only {TEST_BUSY} of "
  f"the {N_TEST} test days fell on the busy side.")

# ==================================================================== Q3
q("Q3", "Every column for one question",
  "Now offer the stump all twenty columns, as `stump20`. Print which column it "
  "asks about, `columns[stump20.tree_.feature[0]]`, its cut, how many test days "
  "land on the busy side of that cut, and its test RMSE beside the RMSE of "
  "forecasting the training average for every test day.",
  "stump20 = ...\n...\nasked = ...\n\nprint('asks about:', asked)\nprint('cut:', ...)\n"
  "print('test days on the busy side:', ...)\nprint('stump20:', ..., ' average:', ...)",
  ["The busy side is `test[asked] > stump20.tree_.threshold[0]`; `.sum()` counts it.",
   "The average forecast for every test day is "
   "`np.full(len(test), train['vol_next'].mean())`."],
  "stump20 = DecisionTreeRegressor(max_depth=1, random_state=0)\n"
  "stump20.fit(train[columns], train['vol_next'])\nasked = columns[stump20.tree_.feature[0]]\n\n"
  "print('asks about:', asked)\nprint('cut:', round(stump20.tree_.threshold[0], 5))\n"
  "print('test days on the busy side:', (test[asked] > stump20.tree_.threshold[0]).sum())\n"
  "print('stump20:', rmse(test['vol_next'], stump20.predict(test[columns])),\n"
  "      ' average:', rmse(test['vol_next'], np.full(len(test), train['vol_next'].mean())))",
  f"`{S20_COL}` at {S20_CUT:.5f}: Apple's first question is about Disney. On "
  f"the training days that cut left an RMSE of {S20_TR:.5f}, a little below the "
  f"{STUMP_TR:.5f} of Apple's own volatility, so the tree took it. On the test "
  f"days {S20_TEST_RIGHT} of {N_TEST} land on the busy side, and the stump "
  f"scores {S20_TE:.5f}, worse than the average at {MEAN_TE:.5f}. The best "
  "question on eight years was not the best question on the next two.")

# ==================================================================== Q4
q("Q4", "Grow it until it fits",
  "For depths 1, 2, 3, 4, 6, 10 and no limit (`None`), fit a tree on all "
  "twenty columns and collect the number of leaves, the training RMSE and the "
  "test RMSE in a DataFrame `growth`, one row per depth.",
  "rows = []\nfor depth in [1, 2, 3, 4, 6, 10, None]:\n    ...\n\ngrowth = ...\nprint(growth)",
  ["Inside the loop, fit the tree and append a dictionary: "
   "`{'depth': str(depth), 'leaves': t.get_n_leaves(), 'train': ..., 'test': ...}`.",
   "`pd.DataFrame(rows).set_index('depth')` turns the list of dictionaries into "
   "the table."],
  "rows = []\nfor depth in [1, 2, 3, 4, 6, 10, None]:\n"
  "    t = DecisionTreeRegressor(max_depth=depth, random_state=0)\n"
  "    t.fit(train[columns], train['vol_next'])\n"
  "    rows.append({'depth': str(depth), 'leaves': t.get_n_leaves(),\n"
  "                 'train': rmse(train['vol_next'], t.predict(train[columns])),\n"
  "                 'test': rmse(test['vol_next'], t.predict(test[columns]))})\n\n"
  "growth = pd.DataFrame(rows).set_index('depth')\nprint(growth.round(5))",
  f"With no limit the training error is zero to nine decimals: {FULL_LEAVES:,} "
  f"leaves for {N_TRAIN:,} days, at depth {FULL_DEPTH}. The test error is lowest "
  f"at depth {BEST_TREE_D}, {BEST_TREE_TE:.5f}, and {GROWTH[None][3]:.5f} with no "
  "limit. Every tree in the table is worse than forecasting the training "
  f"average ({MEAN_TE:.5f}). On twenty columns and eight years, one tree has too "
  "many ways to be wrong.")

# ==================================================================== Q5
q("Q5", "Let the folds choose the size",
  "Search `max_depth` over `[1, 2, 3, 4, 6, 8]` as `depth_search`, and "
  "separately `min_samples_leaf` over `[20, 50, 100, 200, 400]` as "
  "`leaf_search`, each on `folds`. For each search print the best setting, its "
  "mean fold RMSE and its test RMSE, and compare the fold scores with "
  "`part6_one_folds`.",
  "depth_search = ...\n...\nleaf_search = ...\n...\n\nprint(...)\nprint(...)\n"
  "print('one column on the folds:', part6_one_folds)",
  ["`GridSearchCV(DecisionTreeRegressor(random_state=0), {'max_depth': [...]}, "
   "cv=folds, scoring='neg_root_mean_squared_error')`, fitted on the twenty "
   "columns.",
   "A loop `for s in [depth_search, leaf_search]:` prints both with one line."],
  "depth_search = GridSearchCV(DecisionTreeRegressor(random_state=0), {'max_depth': [1, 2, 3, 4, 6, 8]},\n"
  "                            cv=folds, scoring='neg_root_mean_squared_error')\n"
  "depth_search.fit(train[columns], train['vol_next'])\n"
  "leaf_search = GridSearchCV(DecisionTreeRegressor(random_state=0),\n"
  "                           {'min_samples_leaf': [20, 50, 100, 200, 400]},\n"
  "                           cv=folds, scoring='neg_root_mean_squared_error')\n"
  "leaf_search.fit(train[columns], train['vol_next'])\n\n"
  "for s in [depth_search, leaf_search]:\n"
  "    print(s.best_params_, round(-s.best_score_, 5), round(rmse(test['vol_next'], s.predict(test[columns])), 5))\n"
  "print('one column on the folds:', part6_one_folds)",
  f"The folds choose depth {GD_BEST}, which is the Disney stump of Q3, at "
  f"{GD_CV:.5f}, and leaves of at least {GL_BEST} days at {GL_CV:.5f}: a tree of "
  f"{GL_LEAVES} leaves that scores {GL_TE:.5f} on the test block. Both are "
  f"behind the one column on the folds ({ONE_CV:.5f}). Choosing the size "
  "honestly limits the damage; it does not find a tree that helps.")

# ==================================================================== Q6
q("Q6", "Two columns, four leaves",
  "Go back to the two columns the lecture's tree used. Fit a tree of depth 2 "
  "on `pair` as `pair_tree` and draw it with `plot_tree`, without the error "
  "line and with four decimals. List the column behind each of its three "
  "questions, then compare it with the one-column model on the test block and "
  "on the folds.",
  "pair = ['vol_20d', 'ret_20d']\npair_tree = ...\n...\n\n"
  "plt.figure(figsize=(11, 3.6))\n...\nplt.show()\n\n"
  "pair_asked = ...\npair_rmse = ...\npair_folds = ...\n\n"
  "print('questions about:', pair_asked)\n"
  "print('test :', pair_rmse, ' one column:', one_rmse)\n"
  "print('folds:', pair_folds, ' one column:', part6_one_folds)",
  ["`plot_tree(pair_tree, feature_names=pair, filled=True, impurity=False, "
   "precision=4, fontsize=9)`. `pair_tree.tree_.feature` holds a column number "
   "for every box, and -2 for a leaf, so "
   "`[pair[i] for i in pair_tree.tree_.feature if i >= 0]` keeps the questions.",
   "Score it on the folds with `cross_val_score` on a fresh "
   "`DecisionTreeRegressor(max_depth=2, random_state=0)`, negated and averaged."],
  "pair = ['vol_20d', 'ret_20d']\npair_tree = DecisionTreeRegressor(max_depth=2, random_state=0)\n"
  "pair_tree.fit(train[pair], train['vol_next'])\n\n"
  "plt.figure(figsize=(11, 3.6))\n"
  "plot_tree(pair_tree, feature_names=pair, filled=True, impurity=False, precision=4, fontsize=9)\n"
  "plt.show()\n\n"
  "pair_asked = [pair[i] for i in pair_tree.tree_.feature if i >= 0]\n"
  "pair_rmse = rmse(test['vol_next'], pair_tree.predict(test[pair]))\n"
  "pair_folds = -cross_val_score(DecisionTreeRegressor(max_depth=2, random_state=0), train[pair],\n"
  "                              train['vol_next'], cv=folds, scoring='neg_root_mean_squared_error').mean()\n\n"
  "print('questions about:', pair_asked)\nprint('test :', pair_rmse, ' one column:', one_rmse)\n"
  "print('folds:', pair_folds, ' one column:', part6_one_folds)",
  f"All three questions are about `vol_20d`. The root asks the stump's question "
  f"again, at {PT_CUTS[0]:.4f}; the calm side then cuts at {PT_CUTS[1]:.4f} and "
  f"the busy side at {PT_CUTS[2]:.4f}, a level only {PT_BUSIEST_N} training days "
  f"reached. On Apple the tree never asks about the direction: it is a staircase "
  f"of four forecasts on one column, rising from {PT_VALUES[0]:.4f} to "
  f"{PT_VALUES[-1]:.4f}. It beats the one-column model on the test block by "
  f"{ONE_TE - PAIR_TE:.5f} and loses on the folds by {PAIR_CV - ONE_CV:.4f}. A win "
  "that small on two calm years, against a loss on the folds, is not a reason "
  "to change the report's model.")

# ==================================================================== Q7
q("Q7", "Bagging, then a forest",
  "Fit two forests of 100 trees on the twenty columns with `random_state=0`: "
  "`bagged` with `max_features=1.0`, where every question sees every column, "
  "which is plain bagging, and `forest` with `max_features='sqrt'`. Time each "
  "fit with `time.perf_counter()`, and print the seconds and the test RMSE of "
  "each.",
  "start = ...\nbagged = ...\n...\nbagged_seconds = ...\n\n"
  "start = ...\nforest = ...\n...\nforest_seconds = ...\n\n"
  "print('bagging:', ..., ...)\nprint('forest :', ..., ...)",
  "`start = time.perf_counter()` before the fit and "
  "`time.perf_counter() - start` after it.",
  "start = time.perf_counter()\n"
  "bagged = RandomForestRegressor(n_estimators=100, max_features=1.0, random_state=0)\n"
  "bagged.fit(train[columns], train['vol_next'])\nbagged_seconds = time.perf_counter() - start\n\n"
  "start = time.perf_counter()\n"
  "forest = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=0)\n"
  "forest.fit(train[columns], train['vol_next'])\nforest_seconds = time.perf_counter() - start\n\n"
  "print('bagging:', round(bagged_seconds, 2), 's', round(rmse(test['vol_next'], bagged.predict(test[columns])), 5))\n"
  "print('forest :', round(forest_seconds, 2), 's', round(rmse(test['vol_next'], forest.predict(test[columns])), 5))",
  f"Bagging scores {BAG_TE:.5f} and the forest {RF_TE:.5f}. The forest was "
  f"about {BAG_S / RF_S:.0f} times quicker to fit, because each question "
  "looks at 4 columns instead of 20; the seconds depend on your machine, the "
  "ratio much less. The forest is the first model built from trees in this "
  f"part to beat the training average ({MEAN_TE:.5f}) on twenty columns, and "
  "it is still well behind the one column.")

# ==================================================================== Q8
q("Q8", "Out-of-bag, or the folds",
  "Refit the forest from Q7 with `oob_score=True`, and print its out-of-bag "
  "RMSE on the training days beside its mean RMSE on the folds, stored as "
  "`forest_folds`. Say in a comment which of the two belongs in the report.",
  "forest = ...\n...\noob_rmse = ...\nforest_folds = ...\n\n"
  "print('out-of-bag:', oob_rmse)\nprint('folds     :', forest_folds, ' one column:', part6_one_folds)\n"
  "# which one belongs in the report, and why?",
  ["`forest.oob_prediction_` holds a forecast for every training day from the "
   "trees that did not see it.",
   "Score a fresh forest with the same settings on the folds; `oob_score` is "
   "not needed there."],
  "forest = RandomForestRegressor(n_estimators=100, max_features='sqrt', oob_score=True, random_state=0)\n"
  "forest.fit(train[columns], train['vol_next'])\n"
  "oob_rmse = rmse(train['vol_next'], forest.oob_prediction_)\n"
  "forest_folds = -cross_val_score(RandomForestRegressor(n_estimators=100, max_features='sqrt',\n"
  "                                                      random_state=0),\n"
  "                                train[columns], train['vol_next'], cv=folds,\n"
  "                                scoring='neg_root_mean_squared_error').mean()\n\n"
  "print('out-of-bag:', oob_rmse)\nprint('folds     :', forest_folds, ' one column:', part6_one_folds)\n"
  "# The folds. Out-of-bag scores each day with trees that saw the days either\n"
  "# side of it, and neighbouring days share most of a 20-day window.",
  f"{RF_OOB:.5f} out-of-bag against {RF_CV:.5f} on the folds. Out-of-bag makes "
  f"the forest look {RF_CV / RF_OOB:.0f} times better than it is, and better than "
  "anything else in the report; on the folds it is behind the one column "
  f"({ONE_CV:.5f}). A score the model computes on itself is honest only when "
  "the rows are independent, and days are not.")

# ==================================================================== Q9
q("Q9", "What a grid will cost, before you run it",
  "The grid below has two settings. Count its combinations and the fits it "
  "needs: five folds for each combination, plus one refit at the end. Then time "
  "one forest of 100 trees at the slowest corner of the grid, 20 columns and "
  "leaves of 5, and at the quickest, 1 column and leaves of 200, and turn the "
  "two into a range for the whole search on one core.",
  "grid = {'max_features': [1, 3, 5, 10, 20], 'min_samples_leaf': [5, 20, 50, 100, 200]}\n\n"
  "combinations = ...\nfits = ...\n\ntimings = {}\nfor setting in [(20, 5), (1, 200)]:\n    ...\n\n"
  "print(combinations, 'combinations,', fits, 'fits')\nprint(timings)\n"
  "print('between', ..., 'and', ..., 'seconds on one core')",
  ["`combinations = len(grid['max_features']) * len(grid['min_samples_leaf'])` "
   "and `fits = combinations * 5 + 1`.",
   "Inside the loop, build the forest with `max_features=setting[0]` and "
   "`min_samples_leaf=setting[1]`, time its fit, and store the seconds in "
   "`timings[setting]`."],
  "grid = {'max_features': [1, 3, 5, 10, 20], 'min_samples_leaf': [5, 20, 50, 100, 200]}\n\n"
  "combinations = len(grid['max_features']) * len(grid['min_samples_leaf'])\n"
  "fits = combinations * 5 + 1\n\ntimings = {}\nfor setting in [(20, 5), (1, 200)]:\n"
  "    f = RandomForestRegressor(n_estimators=100, max_features=setting[0],\n"
  "                              min_samples_leaf=setting[1], random_state=0)\n"
  "    start = time.perf_counter()\n    f.fit(train[columns], train['vol_next'])\n"
  "    timings[setting] = round(time.perf_counter() - start, 2)\n\n"
  "print(combinations, 'combinations,', fits, 'fits')\nprint(timings)\n"
  "print('between', round(fits * timings[(1, 200)]), 'and', round(fits * timings[(20, 5)]),\n"
  "      'seconds on one core')",
  f"{N_COMB} combinations and {N_FITS} fits, which is {N_FITS * 100:,} trees. The "
  f"slow corner took {T_SLOW:.2f} seconds here and the quick one {T_FAST:.2f}, "
  f"so the search lies somewhere between {N_FITS * T_FAST:.0f} and "
  f"{N_FITS * T_SLOW:.0f} seconds on one core. The fits on the folds use fewer "
  "rows than the full training block, so the real figure sits lower. A range "
  "is the honest answer, and it is enough to decide whether to wait.")

# ==================================================================== Q10
q("Q10", "Run it, on every core",
  "Run the search as `forest_search`, with `GridSearchCV` on `folds` and the "
  "forest set to `n_jobs=-1` so that its trees grow on every core, and time "
  "it. Print the best settings, the mean fold RMSE, the test RMSE as "
  "`forest_rmse`, and the seconds, with the report's two forecasts beside the "
  "scores. It takes about a minute on Colab.",
  "start = ...\nforest_search = ...\n...\nsearch_seconds = ...\nforest_rmse = ...\n\n"
  "print(...)\nprint('folds:', ..., ' ridge:', part6_ridge_folds, ' one column:', part6_one_folds)\n"
  "print('test :', forest_rmse, ' ridge:', part6_ridge_rmse, ' one column:', one_rmse)\n"
  "print('time :', search_seconds)",
  ["`GridSearchCV(RandomForestRegressor(n_estimators=100, random_state=0, "
   "n_jobs=-1), grid, cv=folds, scoring='neg_root_mean_squared_error')`.",
   "The refitted best forest forecasts with `forest_search.predict(...)`."],
  "start = time.perf_counter()\n"
  "forest_search = GridSearchCV(RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1),\n"
  "                             grid, cv=folds, scoring='neg_root_mean_squared_error')\n"
  "forest_search.fit(train[columns], train['vol_next'])\n"
  "search_seconds = time.perf_counter() - start\n"
  "forest_rmse = rmse(test['vol_next'], forest_search.predict(test[columns]))\n\n"
  "print(forest_search.best_params_)\n"
  "print('folds:', -forest_search.best_score_, ' ridge:', part6_ridge_folds, ' one column:', part6_one_folds)\n"
  "print('test :', forest_rmse, ' ridge:', part6_ridge_rmse, ' one column:', one_rmse)\n"
  "print('time :', search_seconds)",
  f"Five columns per question and leaves of at least 100 days. The tuned forest "
  f"scores {FS_CV:.5f} on the folds against ridge's {RIDGE_CV:.5f}, and "
  f"{FS_TE:.5f} on the test block against ridge's {RIDGE_TE:.5f}: level with "
  "ridge on both yardsticks. The settings alone took the forest from bagging's "
  f"{BAG_TE:.5f} to {FS_TE:.5f}, almost a third less error. The one column "
  f"still leads on both, at {ONE_CV:.5f} and {ONE_TE:.5f}. The search took "
  f"{FS_S:.0f} seconds on a machine with many cores.")

# ==================================================================== Q11
q("Q11", "The same forest across the desk",
  "Using `wide_table` from Q1, fit a forest with the settings the search chose "
  "on Apple for every instrument, and store its test RMSE in `desk_forest`. "
  "`part6_desk` holds each instrument's ridge and one-column RMSE, as the "
  "report established them. Count the instruments where the forest beats each.",
  "desk_forest = {}\nfor ticker in TICKERS:\n    ...\n\nbeats_ridge = ...\nbeats_one = ...\n\n"
  "print('forest beats ridge on', beats_ridge, 'of', len(TICKERS))\n"
  "print('forest beats one column on', beats_one, 'of', len(TICKERS))",
  ["Inside the loop: `frame = wide_table(ticker)`, split it with `.loc`, "
   "`cols = list(frame.columns[:-1])`, and build the forest with "
   "`max_features=forest_search.best_params_['max_features']` and the same "
   "for `min_samples_leaf`.",
   "`sum(1 for t in TICKERS if desk_forest[t] < part6_desk[t][0])` counts the "
   "wins over ridge; `[1]` is the one column."],
  "desk_forest = {}\nfor ticker in TICKERS:\n    frame = wide_table(ticker)\n"
  "    tr, te = frame.loc[:'2022-12-31'], frame.loc['2023-01-01':]\n"
  "    cols = list(frame.columns[:-1])\n"
  "    f = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1,\n"
  "                              max_features=forest_search.best_params_['max_features'],\n"
  "                              min_samples_leaf=forest_search.best_params_['min_samples_leaf'])\n"
  "    f.fit(tr[cols], tr['vol_next'])\n"
  "    desk_forest[ticker] = rmse(te['vol_next'], f.predict(te[cols]))\n\n"
  "beats_ridge = sum(1 for t in TICKERS if desk_forest[t] < part6_desk[t][0])\n"
  "beats_one = sum(1 for t in TICKERS if desk_forest[t] < part6_desk[t][1])\n\n"
  "print('forest beats ridge on', beats_ridge, 'of', len(TICKERS))\n"
  "print('forest beats one column on', beats_one, 'of', len(TICKERS))",
  f"The forest beats ridge on {len(BEATS_RIDGE)} of 11, "
  f"{', '.join(BEATS_RIDGE[:-1])} and {BEATS_RIDGE[-1]}, and the one column on "
  f"{len(BEATS_ONE)}. The settings were chosen on Apple and used everywhere, "
  "and they may not suit the other ten. Level with ridge on Apple and better "
  "on about half the desk: that is the forest's honest result on this target, "
  "not the win the lecture's forest had on the credit table, where the rows "
  "were separate borrowers.")

# ==================================================================== Q12
q("Q12", "Draw the forecasts",
  "One figure of the test years: what happened as a black line, and the "
  "one-column, stump and tuned forest forecasts as three more lines, with a "
  "legend and a title. Draw the stump with `drawstyle='steps-post'` so its two "
  "levels show as steps.",
  "fig, ax = plt.subplots(figsize=(10, 3.6))\n...\n...\n...\n...\n"
  "ax.set_ylabel('20-day volatility')\nplt.show()",
  ["`ax.plot(test.index, test['vol_next'], color='black', label='what happened')`, "
   "then each forecast with its own `label`.",
   "`ax.legend()` and `ax.set_title(..., loc='left')`."],
  "fig, ax = plt.subplots(figsize=(10, 3.6))\n"
  "ax.plot(test.index, test['vol_next'], color='black', linewidth=1.2, label='what happened')\n"
  "ax.plot(test.index, one_model.predict(test[['vol_20d']]), label='one column')\n"
  "ax.plot(test.index, stump.predict(test[['vol_20d']]), drawstyle='steps-post', label='stump')\n"
  "ax.plot(test.index, forest_search.predict(test[columns]), label='tuned forest')\n"
  "ax.legend()\nax.set_ylabel('20-day volatility')\n"
  "ax.set_title('Apple, 2023 to 2024: three forecasts against what happened', loc='left')\n"
  "plt.show()",
  "All three forecasts sit above what happened on about three test days in "
  "four, because the test years were calmer than the training years: on "
  f"average the one-column model forecast {O_MEAN:.4f} and the forest "
  f"{F_MEAN:.4f}, where {A_MEAN:.4f} happened. The stump holds one level on all but {TEST_BUSY} "
  "days. The forest moves more than the line and does not follow the turns "
  "any better.")

# ==================================================================== Q13
q("Q13", "The same forest, once, on the warning",
  "The report's other half is Part 9's jump warning. Make the jump label for "
  "the training and test days as `train_jump` and `test_jump`, and fit a "
  "`RandomForestClassifier` of 100 trees with the settings `forest_search` "
  "chose for the number, as `jump_forest`. Print its test AUC, `forest_auc`, "
  "and its mean fold AUC, `forest_folds_auc`, each beside Part 9's two models.",
  "train_jump = ...\ntest_jump = ...\n\njump_forest = ...\n...\nforest_auc = ...\nforest_folds_auc = ...\n\n"
  "print('test :', forest_auc, ' Part 9, one column:', part9_auc, ' twenty:', part9_wide_auc)\n"
  "print('folds:', forest_folds_auc, ' Part 9, one column:', part9_folds_auc, ' twenty:', part9_wide_folds_auc)",
  ["The label is `(train['vol_next'] > 1.5 * train['vol_20d']).astype(int)`, and "
   "the same on `test`.",
   "`roc_auc_score(test_jump, jump_forest.predict_proba(test[columns])[:, 1])`; "
   "on the folds, `cross_val_score(..., scoring='roc_auc')` with no minus sign, "
   "because a larger AUC is better."],
  "train_jump = (train['vol_next'] > 1.5 * train['vol_20d']).astype(int)\n"
  "test_jump = (test['vol_next'] > 1.5 * test['vol_20d']).astype(int)\n\n"
  "jump_forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1,\n"
  "                                     max_features=forest_search.best_params_['max_features'],\n"
  "                                     min_samples_leaf=forest_search.best_params_['min_samples_leaf'])\n"
  "jump_forest.fit(train[columns], train_jump)\n"
  "forest_auc = roc_auc_score(test_jump, jump_forest.predict_proba(test[columns])[:, 1])\n"
  "forest_folds_auc = cross_val_score(jump_forest, train[columns], train_jump, cv=folds,\n"
  "                                   scoring='roc_auc').mean()\n\n"
  "print('test :', forest_auc, ' Part 9, one column:', part9_auc, ' twenty:', part9_wide_auc)\n"
  "print('folds:', forest_folds_auc, ' Part 9, one column:', part9_folds_auc, ' twenty:', part9_wide_folds_auc)",
  f"{JF_AUC:.3f} on the test block, between Part 9's one column ({P9_AUC:.3f}) "
  f"and its twenty ({P9_WIDE_AUC:.3f}). On the folds {JF_CV:.3f}, below both "
  f"({P9_FOLDS:.3f} and {P9_WIDE_FOLDS:.3f}). The test block made the forest look "
  "like a better use of the twenty columns than Part 9's wide model; the folds "
  "do not agree, and they cover more years. The warning stays as Part 9 left "
  "it, threshold and all.")

# ==================================================================== Q14
q("Q14", "Write down what you would defend",
  "Collect what Part 11 established into a dictionary `report`, and write "
  "`summarise(report)`: it prints one line per entry and ends with two "
  "verdicts, one on the volatility forecast and one on the jump warning.",
  "report = {\n    'target': ...,\n    'one_column_rmse': ...,\n    'ridge_rmse': ...,\n"
  "    'stump_rmse': ...,\n    'best_single_tree_rmse': ...,\n    'forest_untuned_rmse': ...,\n"
  "    'forest_tuned_rmse': ...,\n    'forest_settings': ...,\n    'forest_beats_ridge_on': ...,\n"
  "    'warning_logistic_auc': ...,\n    'warning_forest_auc': ...,\n}\n\n\n"
  "def summarise(r):\n    \"\"\"...\"\"\"\n    ...\n\n\nsummarise(report)",
  ["Every value is a number or a short string you already have: `part5_target`, "
   "`one_rmse`, `part6_ridge_rmse`, `stump_rmse`, the smallest entry of "
   "`growth['test']`, the forest from Q8, `forest_rmse`, "
   "`forest_search.best_params_`, `beats_ridge`, `part9_auc` and `forest_auc`.",
   "Inside the function, `for key, value in r.items():` and a `print` with an "
   "f-string, then two `if` statements for the verdicts."],
  "report = {\n    'target': part5_target,\n    'one_column_rmse': round(one_rmse, 5),\n"
  "    'ridge_rmse': part6_ridge_rmse,\n    'stump_rmse': round(stump_rmse, 5),\n"
  "    'best_single_tree_rmse': round(growth['test'].min(), 5),\n"
  "    'forest_untuned_rmse': round(rmse(test['vol_next'], forest.predict(test[columns])), 5),\n"
  "    'forest_tuned_rmse': round(forest_rmse, 5),\n"
  "    'forest_settings': forest_search.best_params_,\n"
  "    'forest_beats_ridge_on': f'{beats_ridge} of {len(TICKERS)} instruments',\n"
  "    'warning_logistic_auc': part9_auc,\n"
  "    'warning_forest_auc': round(forest_auc, 4),\n}\n\n\n"
  "def summarise(r):\n    \"\"\"Print the report and the two verdicts it supports.\"\"\"\n"
  "    for key, value in r.items():\n        print(f\"{key:<24}{value}\")\n    print()\n"
  "    if r['forest_tuned_rmse'] < r['one_column_rmse']:\n"
  "        print('Forecast: the tuned forest replaces the one-column model.')\n"
  "    else:\n"
  "        print('Forecast: the one-column model stays.')\n"
  "    if r['warning_forest_auc'] > r['warning_logistic_auc']:\n"
  "        print('Warning: the forest replaces the one-column logistic regression.')\n"
  "    else:\n"
  "        print('Warning: the one-column logistic regression of Part 9 stays.')\n\n\n"
  "summarise(report)",
  "The report keeps both of its models, and now it can say why with a flexible "
  "model behind each verdict. The trees found patterns in the training years, "
  "Disney's volatility among them, that did not carry into the next two; the "
  "forest, tuned on the folds, drew level with ridge and no further. The "
  "untuned forest is in the report on purpose: next to the tuned one it shows "
  "how much of a forest's score comes from its settings.")

# ---------------------------------------------------------------- closing
md(
"## \U0001f4e6 What you now have\n\n"
"| what | where |\n|:--|:--|\n"
"| the report's table as a function, Apple's split and its forecast | `wide_table`, `table`, `columns`, `train`, `test`, `one_model`, `one_rmse` |\n"
"| one question, on one column and on twenty | `stump`, `cut`, `stump_rmse`, `stump20` |\n"
"| trees grown deeper, and sized on the folds | `growth`, `depth_search`, `leaf_search` |\n"
"| the staircase on two columns, drawn | `pair_tree`, `pair_rmse`, `pair_folds` |\n"
"| bagging and a forest, timed | `bagged`, `forest`, `bagged_seconds`, `forest_seconds` |\n"
"| out-of-bag against the folds | `oob_rmse`, `forest_folds` |\n"
"| the grid, costed and searched | `grid`, `timings`, `forest_search`, `forest_rmse`, `search_seconds` |\n"
"| the whole desk | `desk_forest`, `beats_ridge`, `beats_one` |\n"
"| the same forest on the jump label | `train_jump`, `test_jump`, `jump_forest`, `forest_auc`, `forest_folds_auc` |\n"
"| the thing you would defend | `report` |"
)

md(
"## What changed since Part 9\n\n"
"- **A new kind of model.** Everything in the report was linear in its "
"columns. A tree asks questions instead, and a forest averages many trees; "
"both were held to the numbers the report had already established, on the "
"test block and on the folds.\n"
"- **More columns made a single tree worse.** Offered twenty columns, a tree "
f"asks about Disney first (Q3) and is worse than the training average "
f"({MEAN_TE:.5f}) at every depth (Q4). A stump on `vol_20d` alone scores "
f"{STUMP_TE:.5f} with two numbers, against {ONE_TE:.5f} for the line (Q2).\n"
"- **A forest tuned on the folds draws level with ridge.** "
f"{FS_TE:.5f} against {RIDGE_TE:.5f} on the test block and {FS_CV:.5f} against "
f"{RIDGE_CV:.5f} on the folds (Q10), and better than ridge on "
f"{len(BEATS_RIDGE)} of the 11 instruments (Q11). The search was costed before "
"it ran (Q9).\n"
f"- **Out-of-bag flattered the forest.** {RF_OOB:.5f} against {RF_CV:.5f} on the "
"folds, because days are not independent rows (Q8).\n"
f"- **The warning is unchanged.** The same forest scores {JF_AUC:.3f} on Part 9's "
f"jump label and {JF_CV:.3f} on the folds, below the one column on both (Q13)."
)

md(
"## Where this leaves the risk report\n\n"
"Nothing in the report changes, and that is the finding. The volatility "
"forecast is still the one-column model of Part 5, and the jump warning is "
f"still Part 9's one-column logistic regression at a threshold of {P9_CHOSEN}. "
"What changed is how much stands behind them. The twenty columns have now "
"been offered to a model that can ask different questions of calm and busy "
"weeks, and on Apple they still do not beat this month's volatility. The desk "
"has a second family of models it can fit, tune, time and cost, and on about "
"half of its instruments the forest is the better of the two flexible "
"forecasts.\n\n"
"**Next part:** boosting, where each tree is grown to correct the ones before "
"it, and a look at which columns the models lean on."
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
