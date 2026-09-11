# -*- coding: utf-8 -*-
"""Build session_06_exercises.ipynb.

Same conventions as Sessions 1 to 5: pleasant intro, 1-5 star badges, toolkit
card with title= hover docs, task -> work cell (blank-safe `...`) -> 1-2 folded
hints -> folded solution, no em-dashes, plain explanatory tone.

Session 6 is penalised regression. The exercises build the lecture's wide
table from the price file (the lecture only loaded it), watch OLS overfit it,
then work through ridge, the units problem and StandardScaler, Pipeline,
choosing alpha by cross-validation and GridSearchCV, lasso, elastic net, the
arguments of these objects, and what the coefficients mean afterwards. Section
K takes the whole workflow across all eleven instruments.

Only tools taught by the end of Session 6. New this session: Ridge, Lasso,
ElasticNet (alpha, l1_ratio, max_iter, positive, fit_intercept),
StandardScaler (fit, transform, mean_, scale_), Pipeline (named steps,
named_steps, step__argument), GridSearchCV (param_grid, cv, scoring,
best_params_, best_score_, best_estimator_, cv_results_), get_params().
NOT taught, so never required: make_pipeline, RidgeCV/LassoCV,
RandomizedSearchCV, cross_validate, any non-linear model.

Returns are in PERCENT here, exactly as in the lecture, so the numbers read.

BLANK-SAFE RULES:
- every blank is the right-hand side of an assignment, a bare `...` statement,
  or an argument to print(). Never call a method on, index into, or do
  arithmetic with a placeholder.
- nothing depends on an earlier exercise having been solved: the setup cell and
  each work cell provide every variable the task uses.
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
OUT = ROOT / "session_06" / "session_06_exercises.ipynb"

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
    "A2": "S2", "A3": "S2", "A4": "S2", "A6": "S3", "A7": "S2",
    "B3": "S3", "B4": "S3", "B5": "S4", "B6": "S4", "B7": "S5",
    "C2": "S3", "C3": "S5", "C4": "S2", "C6": "S3",
    "D1": "S3", "D4": "S3", "D5": "S3", "D6": "S5",
    "E2": "S5", "E3": "S5", "E4": "S5",
    "F1": "S2", "F2": "S3", "F6": "S1", "F7": "S2",
    "G3": "S2", "G5": "S5", "G6": "S2",
    "H3": "S3",
    "I2": "S1",
    "J1": "S1", "J2": "S5", "J3": "S2", "J4": "S2", "J5": "S5",
    "K1": "S2", "K2": "S2", "K3": "S3",
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
TBL = pd.read_csv(ROOT / "data" / "market_features.csv", parse_dates=["date"]).set_index("date")
TRAIN, TEST = TBL.loc[:"2022-12-31"], TBL.loc["2023-01-01":]
COLS = list(TBL.columns[:-1])
N_TBL, N_TRAIN, N_TEST = len(TBL), len(TRAIN), len(TEST)
TS5 = TimeSeriesSplit(n_splits=5)
STOCKS = [t for t in RET.columns if t != "SPY"]


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(np.asarray(y), np.asarray(p))))


def cv(model, cols=COLS, folds=TS5, frame=TRAIN):
    return -cross_val_score(model, frame[cols], frame["vol_next"], cv=folds,
                            scoring="neg_root_mean_squared_error")


def build_table(ticker):
    r = RET[ticker]
    frame = pd.DataFrame()
    for w in [5, 10, 20, 40, 60, 120]:
        frame["vol_" + str(w) + "d"] = r.rolling(w).std()
    for w in [5, 20, 60]:
        frame["ret_" + str(w) + "d"] = r.rolling(w).mean()
    for t in RET.columns:
        if t != ticker:
            frame[t + "_vol"] = RET[t].rolling(20).std()
    frame["vol_next"] = r.rolling(20).std().shift(-20)
    return frame.dropna()


# A: the rebuilt table matches the file
_built = build_table("SPY")[TBL.columns]
BUILD_DIFF = float((_built - TBL).abs().max().max())

# B: one column and all nineteen
ONE = LinearRegression().fit(TRAIN[["vol_20d"]], TRAIN["vol_next"])
OLS = LinearRegression().fit(TRAIN[COLS], TRAIN["vol_next"])
ONE_TR, ONE_TE = rmse(TRAIN["vol_next"], ONE.predict(TRAIN[["vol_20d"]])), rmse(TEST["vol_next"], ONE.predict(TEST[["vol_20d"]]))
OLS_TR, OLS_TE = rmse(TRAIN["vol_next"], OLS.predict(TRAIN[COLS])), rmse(TEST["vol_next"], OLS.predict(TEST[COLS]))
PERS_TE = rmse(TEST["vol_next"], TEST["vol_20d"])
MEAN_TE = rmse(TEST["vol_next"], np.full(N_TEST, TRAIN["vol_next"].mean()))
N_NEG = int((OLS.coef_ < 0).sum())
NEG_NAMES = [c for c, b in zip(COLS, OLS.coef_) if b < 0]
NEG_VOL = [c for c in NEG_NAMES if not c.startswith("ret_")]
CORR_4060 = float(TRAIN["vol_40d"].corr(TRAIN["vol_60d"]))
_cm = TRAIN[COLS].corr().abs()
np.fill_diagonal(_cm.values, 0)
_pair = _cm.stack().idxmax()
TOP_PAIR, TOP_CORR = _pair, float(_cm.stack().max())
SPOT = COLS.index("vol_20d")
SPOT120 = COLS.index("vol_120d")
FOLD_COEF_120 = []
for _fi, _sc in TS5.split(TRAIN):
    _blk = TRAIN.iloc[_fi]
    FOLD_COEF_120.append(float(LinearRegression().fit(_blk[COLS], _blk["vol_next"]).coef_[SPOT120]))

# C: ridge on raw columns, validation block for the loop
RIDGE1000 = Ridge(alpha=1000).fit(TRAIN[COLS], TRAIN["vol_next"])
RIDGE1000_TE = rmse(TEST["vol_next"], RIDGE1000.predict(TEST[COLS]))
RIDGE0_TE = rmse(TEST["vol_next"], Ridge(alpha=0).fit(TRAIN[COLS], TRAIN["vol_next"]).predict(TEST[COLS]))
FIT, VAL = TRAIN.loc[:"2020-12-31"], TRAIN.loc["2021-01-01":]
C_ALPHAS = [1, 10, 100, 1000, 10000, 100000]
VAL_RMSE = {a: rmse(VAL["vol_next"], Ridge(alpha=a).fit(FIT[COLS], FIT["vol_next"]).predict(VAL[COLS])) for a in C_ALPHAS}
VAL_BEST = min(VAL_RMSE, key=VAL_RMSE.get)
INTERCEPTS = {a: float(Ridge(alpha=a).fit(TRAIN[COLS], TRAIN["vol_next"]).intercept_) for a in [1, 1000, 10_000_000]}
TRAIN_MEAN = float(TRAIN["vol_next"].mean())
SUMSQ = {a: float((Ridge(alpha=a).fit(TRAIN[COLS], TRAIN["vol_next"]).coef_ ** 2).sum()) for a in [0, 1000, 100000]}

# D: units and scaling
_tr_dec, _te_dec = TRAIN.copy(), TEST.copy()
_tr_dec["vol_20d"] = _tr_dec["vol_20d"] / 100
_te_dec["vol_20d"] = _te_dec["vol_20d"] / 100
OLS_DEC = LinearRegression().fit(_tr_dec[COLS], _tr_dec["vol_next"])
RIDGE_DEC = Ridge(alpha=1000).fit(_tr_dec[COLS], _tr_dec["vol_next"])
OLS_COEF, OLS_COEF_DEC = float(OLS.coef_[SPOT]), float(OLS_DEC.coef_[SPOT])
RIDGE_COEF, RIDGE_COEF_DEC = float(RIDGE1000.coef_[SPOT]), float(RIDGE_DEC.coef_[SPOT])
SCALER = StandardScaler().fit(TRAIN[COLS])
X_TRAIN, X_TEST = SCALER.transform(TRAIN[COLS]), SCALER.transform(TEST[COLS])
SC_MEAN, SC_SCALE = float(SCALER.mean_[SPOT]), float(SCALER.scale_[SPOT])
HAND_DIFF = float(np.abs((TRAIN["vol_20d"] - TRAIN["vol_20d"].mean()) / np.std(TRAIN["vol_20d"]) - X_TRAIN[:, SPOT]).max())
XTEST_MEAN, XTEST_STD = float(X_TEST[:, SPOT].mean()), float(X_TEST[:, SPOT].std())
RIDGE_SC = Ridge(alpha=1000).fit(X_TRAIN, TRAIN["vol_next"])
RIDGE_SC_TE = rmse(TEST["vol_next"], RIDGE_SC.predict(X_TEST))
_leak = StandardScaler().fit(TEST[COLS])
LEAK_TE = rmse(TEST["vol_next"], Ridge(alpha=1000).fit(SCALER.transform(TRAIN[COLS]), TRAIN["vol_next"]).predict(_leak.transform(TEST[COLS])))

# E: pipeline
PIPE = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1000))]).fit(TRAIN[COLS], TRAIN["vol_next"])
PIPE_TE = rmse(TEST["vol_next"], PIPE.predict(TEST[COLS]))
PIPE_COEF = PIPE.named_steps["ridge"].coef_
BIGGEST = COLS[int(np.argmax(np.abs(PIPE_COEF)))]
BIGGEST_VAL = float(PIPE_COEF[int(np.argmax(np.abs(PIPE_COEF)))])
PIPE_FOLDS = cv(Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1000))]))
PIPE_CV = float(PIPE_FOLDS.mean())
WORST_FOLD = int(np.argmax(PIPE_FOLDS)) + 1
_Xall = pd.DataFrame(X_TRAIN, index=TRAIN.index, columns=COLS)
LEAKY_CV = float(-cross_val_score(Ridge(alpha=1000), _Xall, TRAIN["vol_next"], cv=TS5,
                                  scoring="neg_root_mean_squared_error").mean())

# F: choosing alpha
GRID = [1, 10, 100, 1000, 10000]
CV_BY_ALPHA = {a: float(cv(Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=a))])).mean()) for a in GRID}
CV_BEST = min(CV_BY_ALPHA, key=CV_BY_ALPHA.get)
FINE = np.logspace(0, 5, 11)
CV_FINE = {float(a): float(cv(Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=a))])).mean()) for a in FINE}
CV_FINE_BEST = min(CV_FINE, key=CV_FINE.get)
_gs = GridSearchCV(Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]),
                   {"ridge__alpha": GRID}, cv=TS5, scoring="neg_root_mean_squared_error").fit(TRAIN[COLS], TRAIN["vol_next"])
GS_BEST, GS_SCORE = _gs.best_params_["ridge__alpha"], float(-_gs.best_score_)
GS_TE = rmse(TEST["vol_next"], _gs.predict(TEST[COLS]))
NARROW = [1, 2, 3, 4, 5]
_gn = GridSearchCV(Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]),
                   {"ridge__alpha": NARROW}, cv=TS5, scoring="neg_root_mean_squared_error").fit(TRAIN[COLS], TRAIN["vol_next"])
NARROW_BEST, NARROW_SCORE = _gn.best_params_["ridge__alpha"], float(-_gn.best_score_)
# the while loop: multiply by ten while the error falls
_a, _prev, WHILE_TRACE = 1, None, []
while True:
    _e = float(cv(Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=_a))])).mean())
    WHILE_TRACE.append((_a, _e))
    if _prev is not None and _e > _prev:
        break
    _prev, _a = _e, _a * 10
WHILE_BEST = WHILE_TRACE[-2][0]

# G: lasso
LASSO = Pipeline([("scale", StandardScaler()), ("lasso", Lasso(alpha=0.1))]).fit(TRAIN[COLS], TRAIN["vol_next"])
LASSO_TE = rmse(TEST["vol_next"], LASSO.predict(TEST[COLS]))
LASSO_COEF = LASSO.named_steps["lasso"].coef_
N_ZERO = int((LASSO_COEF == 0).sum())
KEPT = [c for c, b in zip(COLS, LASSO_COEF) if b != 0]
KEPT_VALS = {c: float(b) for c, b in zip(COLS, LASSO_COEF) if b != 0}
LGRID = [0.001, 0.01, 0.1, 1]
_gl = GridSearchCV(Pipeline([("scale", StandardScaler()), ("lasso", Lasso())]),
                   {"lasso__alpha": LGRID}, cv=TS5, scoring="neg_root_mean_squared_error").fit(TRAIN[COLS], TRAIN["vol_next"])
GL_BEST, GL_SCORE = _gl.best_params_["lasso__alpha"], float(-_gl.best_score_)
KEPT_BY_FOLD = []
for _fi, _sc in TS5.split(TRAIN):
    _blk = TRAIN.iloc[_fi]
    _m = Pipeline([("scale", StandardScaler()), ("lasso", Lasso(alpha=0.1))]).fit(_blk[COLS], _blk["vol_next"])
    KEPT_BY_FOLD.append([c for c, b in zip(COLS, _m.named_steps["lasso"].coef_) if b != 0])
ALWAYS_KEPT = [c for c in COLS if all(c in k for k in KEPT_BY_FOLD)]
SURVIVORS = {a: int((Pipeline([("scale", StandardScaler()), ("lasso", Lasso(alpha=a))]).fit(TRAIN[COLS], TRAIN["vol_next"]).named_steps["lasso"].coef_ != 0).sum())
             for a in [0.01, 0.03, 0.1, 0.3, 1]}

# H: elastic net
ENET = Pipeline([("scale", StandardScaler()), ("enet", ElasticNet(alpha=0.1, l1_ratio=0.5))]).fit(TRAIN[COLS], TRAIN["vol_next"])
ENET_TE = rmse(TEST["vol_next"], ENET.predict(TEST[COLS]))
ENET_NZ = int((ENET.named_steps["enet"].coef_ != 0).sum())
EGRID = {"enet__alpha": [0.01, 0.1, 1], "enet__l1_ratio": [0.1, 0.5, 0.9]}
_ge = GridSearchCV(Pipeline([("scale", StandardScaler()), ("enet", ElasticNet())]), EGRID,
                   cv=TS5, scoring="neg_root_mean_squared_error").fit(TRAIN[COLS], TRAIN["vol_next"])
GE_BEST, GE_SCORE, GE_N = _ge.best_params_, float(-_ge.best_score_), len(_ge.cv_results_["params"])
_en1 = Pipeline([("scale", StandardScaler()), ("enet", ElasticNet(alpha=0.1, l1_ratio=1.0))]).fit(TRAIN[COLS], TRAIN["vol_next"])
ENET_VS_LASSO = float(np.abs(_en1.named_steps["enet"].coef_ - LASSO_COEF).max())

# I: arguments
DEFAULT_MAX_ITER = Lasso().get_params()["max_iter"]
POS = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1000, positive=True))]).fit(TRAIN[COLS], TRAIN["vol_next"])
POS_TE = rmse(TEST["vol_next"], POS.predict(TEST[COLS]))
POS_NEG = int((POS.named_steps["ridge"].coef_ < 0).sum())
NOINT = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1000, fit_intercept=False))]).fit(TRAIN[COLS], TRAIN["vol_next"])
NOINT_TE = rmse(TEST["vol_next"], NOINT.predict(TEST[COLS]))

# J: reading the result
COEF_5D = float(PIPE_COEF[COLS.index("vol_5d")])
SMALL_OLS = LinearRegression().fit(TRAIN[KEPT], TRAIN["vol_next"])
SMALL_TE = rmse(TEST["vol_next"], SMALL_OLS.predict(TEST[KEPT]))
SIX = {"guess the average": MEAN_TE, "repeat this month": PERS_TE, "one column": ONE_TE,
       "nineteen columns, OLS": OLS_TE, "nineteen columns, ridge": PIPE_TE, "nineteen columns, lasso": LASSO_TE}
SIX_ORDER = sorted(SIX, key=SIX.get)
_err = TEST["vol_next"] - PIPE.predict(TEST[COLS])
_calm = TEST["vol_20d"] < TEST["vol_20d"].median()
CALM_RMSE, BUSY_RMSE = float(np.sqrt((_err[_calm] ** 2).mean())), float(np.sqrt((_err[~_calm] ** 2).mean()))
N_CALM = int(_calm.sum())

# J5: the worst test day of the ridge pipeline
_abs_err = np.abs(TEST["vol_next"].values - PIPE.predict(TEST[COLS]))
WORST_POS = int(_abs_err.argmax())
WORST_DAY = TEST.index[WORST_POS].date()
WORST_ACTUAL, WORST_PRED = float(TEST["vol_next"].iloc[WORST_POS]), float(PIPE.predict(TEST[COLS])[WORST_POS])

# K: across the desk. K1 is Nvidia; K2 the alpha each instrument picks; K3 the
# lasso's survivors per instrument. (The case does the ridge-against-one-column
# count, so the exercises do not.)
_nv = build_table("NVDA")
_nv_tr, _nv_te = _nv.loc[:"2022-12-31"], _nv.loc["2023-01-01":]
_nv_cols = list(_nv.columns[:-1])
NVDA_ONE = rmse(_nv_te["vol_next"], LinearRegression().fit(_nv_tr[["vol_20d"]], _nv_tr["vol_next"]).predict(_nv_te[["vol_20d"]]))
NVDA_OLS = rmse(_nv_te["vol_next"], LinearRegression().fit(_nv_tr[_nv_cols], _nv_tr["vol_next"]).predict(_nv_te[_nv_cols]))
_nv_gs = GridSearchCV(Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]), {"ridge__alpha": GRID},
                      cv=TS5, scoring="neg_root_mean_squared_error").fit(_nv_tr[_nv_cols], _nv_tr["vol_next"])
NVDA_RIDGE, NVDA_ALPHA = rmse(_nv_te["vol_next"], _nv_gs.predict(_nv_te[_nv_cols])), _nv_gs.best_params_["ridge__alpha"]
BEST_ALPHA = {}
SURVIVORS_BY = {}
DESK = {}
for _t in RET.columns:
    _tb = build_table(_t)
    _tr, _te = _tb.loc[:"2022-12-31"], _tb.loc["2023-01-01":]
    _cc = list(_tb.columns[:-1])
    _g = GridSearchCV(Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]), {"ridge__alpha": GRID},
                      cv=TS5, scoring="neg_root_mean_squared_error").fit(_tr[_cc], _tr["vol_next"])
    _o = LinearRegression().fit(_tr[["vol_20d"]], _tr["vol_next"])
    DESK[_t] = (rmse(_te["vol_next"], _g.predict(_te[_cc])), rmse(_te["vol_next"], _o.predict(_te[["vol_20d"]])),
                _g.best_params_["ridge__alpha"])
    if _t == "AAPL":
        AAPL_OLS = rmse(_te["vol_next"], LinearRegression().fit(_tr[_cc], _tr["vol_next"]).predict(_te[_cc]))
    BEST_ALPHA[_t] = _g.best_params_["ridge__alpha"]
    _lp = Pipeline([("scale", StandardScaler()), ("lasso", Lasso(alpha=0.1))]).fit(_tr[_cc], _tr["vol_next"])
    SURVIVORS_BY[_t] = int((_lp.named_steps["lasso"].coef_ != 0).sum())
N_DESK_WIN = sum(1 for t in DESK if DESK[t][0] < DESK[t][1])
DESK_LOSERS = [t for t in DESK if DESK[t][0] >= DESK[t][1]]
AAPL_RIDGE, AAPL_ONE, AAPL_ALPHA = DESK["AAPL"]
BEST_GAIN_T = max(DESK, key=lambda t: DESK[t][1] - DESK[t][0])
N_PICK_1000 = sum(1 for t in BEST_ALPHA if BEST_ALPHA[t] == 1000)
ALPHA_OTHERS = {t: a for t, a in BEST_ALPHA.items() if a != 1000}
MOST_KEPT = max(SURVIVORS_BY, key=SURVIVORS_BY.get)
FEWEST_KEPT = min(SURVIVORS_BY, key=SURVIVORS_BY.get)

# ---------------------------------------------------------------- top matter
md(
"# \U0001f4d3 Session 6 · Exercises\n"
"### Penalised regression\n\n"
"The lecture loaded a table with nineteen columns and showed what happens when "
"ordinary least squares is given all of them: a better fit on the training rows "
"and a worse forecast. These exercises start one step earlier, by building that "
"table yourself, and then work through the cure: a penalty on the size of the "
"coefficients, the scaling that penalty needs, a pipeline to hold the two "
"together, and a grid search to choose how strong the penalty should be.\n\n"
"By the end you will have run the whole workflow on every instrument in the "
"data, and found out on which of them nineteen penalised columns beat one "
"unpenalised one."
)

md(
"## How to use this notebook\n\n"
"- Run the **setup cell** below first. It loads the price data and the "
"lecture's table, and imports the scikit-learn pieces.\n"
"- Each exercise has a **task**, then a **code cell** for your work. Cells with "
"`...` are blanks to fill in. Replace them with real code.\n"
"- Stuck? Open the **\U0001f4a1 Hint**, but only after a genuine attempt. Open the "
"**✅ Solution** to *check* yourself, not to skip the thinking.\n"
"- Every cell runs cleanly even with the blanks still in place, so pressing "
"**Run all** never floods you with errors.\n"
"- Most exercises stand alone. A few short runs build on each other (A2 to "
"A6, B1 to B4, C3 to C4, D1 to D2, D3 to D6, E1 to E4, F3 to F5, G1 to G3); the task "
"says which earlier exercise it continues from. If one defeats you, open its "
"solution, run it, and carry on. Section K uses the function from A7.\n\n"
"**You are not expected to finish all of these.** Do what you can, and come back "
"to the rest when you revise. Short on time? Read the hint, then the solution. A "
"worked solution you genuinely understand is real learning too.\n\n"
"**Returns are in percent here**, exactly as in the lecture, so an error of "
"`0.22` means 0.22 percentage points."
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
"assumes everything from Sessions 1 to 5, so it is a bigger piece of work than a "
"three-star task in an earlier notebook.\n\n"
"Some exercises also carry a **revisits** tag. Those need something from an "
"earlier session as well as today's material, and they are there on purpose: "
"the skills are meant to accumulate."
)

md(
"## \U0001f9f0 Your toolkit for today\n\n"
"Everything from Sessions 1 to 5 still applies. This card holds what Session 6 "
"added.\n\n"
"> Names in brackets (`frame`, `columns`, `pipe`, ...) are **placeholders**: put "
"your own variable there. **Hover any tool** to see what it does."
)

md(
'<p style="line-height:2.1"><strong>Penalised models</strong><br>\n'
'<code style="cursor:help" title="Linear regression with a charge of alpha times the sum of squared slopes. alpha=0 is ordinary least squares.">Ridge(alpha=1000)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The same, with a charge on the sum of absolute slopes. Sets some coefficients to exactly zero.">Lasso(alpha=0.1)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Both penalties at once. l1_ratio=1 is the lasso, l1_ratio=0 is ridge.">ElasticNet(alpha=0.1, l1_ratio=0.5)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Fit, predict, coef_ and intercept_ work exactly as for LinearRegression.">.fit(X, y)  ·  .predict(X)  ·  .coef_</code></p>\n\n'
'<p style="line-height:2.1"><strong>Putting columns on the same scale</strong><br>\n'
'<code style="cursor:help" title="Learns each column\'s mean and standard deviation. Fit it on the TRAINING rows only.">StandardScaler().fit(X_train)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Subtract the learned mean and divide by the learned standard deviation. Returns a NumPy array.">scaler.transform(X)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The numbers the scaler learned, one per column, in column order.">scaler.mean_  ·  scaler.scale_</code></p>\n\n'
'<p style="line-height:2.1"><strong>Two steps in one object</strong><br>\n'
'<code style="cursor:help" title="A list of (name, step) pairs. The last step is the model; the ones before it transform.">Pipeline([(\'scale\', StandardScaler()), (\'ridge\', Ridge(alpha=1000))])</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="A pipeline fits, predicts and cross-validates like any model. The scaler is refitted inside every fold.">pipe.fit(X, y)  ·  pipe.predict(X)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The fitted object inside one step, so you can read its coef_.">pipe.named_steps[\'ridge\']</code></p>\n\n'
'<p style="line-height:2.1"><strong>Choosing a setting</strong><br>\n'
'<code style="cursor:help" title="The setting to search and the values to try. step name, two underscores, argument name.">{\'ridge__alpha\': [1, 10, 100, 1000, 10000]}</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Fit every value on every fold, keep the best mean score, refit it on all the training rows.">GridSearchCV(pipe, grid, cv=folds, scoring=\'neg_root_mean_squared_error\')</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The winning setting, its mean score (negative, for the usual reason), and the refitted winner.">search.best_params_  ·  search.best_score_  ·  search.best_estimator_</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The whole grid as a dictionary of lists. Wrap it in pd.DataFrame to read it.">search.cv_results_</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Every argument of an object and its current value.">model.get_params()</code></p>\n\n'
'<p style="line-height:2.1"><strong>Two helpers</strong><br>\n'
'<code style="cursor:help" title="Values spaced by a constant factor: here 1, 10, 100, 1000, 10000. The right shape for a grid of alphas.">np.logspace(0, 4, 5)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="A logarithmic axis, so factors of ten are evenly spaced.">ax.set_xscale(\'log\')</code></p>'
)

md(
"**Formulas you will reach for**\n\n"
r"| what | formula |" "\n"
r"|:--|:--|" "\n"
r"| Ordinary least squares | $$\text{RSS}=\sum_i (y_i-\hat{y}_i)^2$$ |" "\n"
r"| Ridge | $$\text{RSS}+\alpha\sum_j \beta_j^2$$ |" "\n"
r"| Lasso | $$\text{RSS}+\alpha\sum_j |\beta_j|$$ |" "\n"
r"| Elastic net | $$\text{RSS}+\alpha\left[\rho\sum_j |\beta_j| + \tfrac{1-\rho}{2}\sum_j \beta_j^2\right]$$ |" "\n"
r"| Standardising | $$z=\dfrac{x-\bar{x}}{s}$$ |" "\n"
r"| Root mean squared error | $$\text{RMSE}=\sqrt{\dfrac{1}{n}\sum_i (y_i-\hat{y}_i)^2}$$ |" "\n"
)

md("---")

# ---------------------------------------------------------------- setup cell
md(
"## ⚙️ Setup: run this first\n\n"
"This loads the price data and the lecture's nineteen-column table, splits it "
"by date, and imports the scikit-learn pieces. If you are in Google Colab it "
"downloads the data by itself."
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


def rmse(actual, predicted):
    """Root mean squared error, as in the lecture, as a plain number."""
    return float(np.sqrt(mean_squared_error(actual, predicted)))


# Eleven instruments, 2015 to 2024. Returns in PERCENT, as in the lecture.
prices = load_csv("prices.csv", parse_dates=["date"])
wide = prices.pivot(index="date", columns="ticker", values="close")
rets = wide.pct_change().dropna() * 100

# The lecture's table: the index's own history plus every stock's volatility
table = load_csv("market_features.csv", parse_dates=["date"]).set_index("date")
train = table.loc[:"2022-12-31"]
test = table.loc["2023-01-01":]
columns = list(table.columns[:-1])

folds = TimeSeriesSplit(n_splits=5)

print("table:", table.shape, "rows x columns")
print("train:", len(train), " test:", len(test), " columns:", len(columns))
print(columns)'''
)

md("---")

# ============================================================ A
section(
"## \U0001f9f1 A · Build the table yourself\n\n"
"The lecture loaded `market_features.csv`. Here you build it from the price "
"file, one group of columns at a time, each exercise adding to the last, and "
"check that the two agree."
)

ex("A1", "The index's returns", 1,
   "Take the daily returns of the index, `SPY`, out of `rets` into a Series called "
   "`spy`.",
   "spy = ...\nspy",
   "`rets` has one column per ticker. Select a column by name.",
   "spy = rets['SPY']\nspy",
   f"One return per trading day, in percent, {len(RET):,} of them.")

ex("A2", "Six volatility windows in a loop", 2,
   "Build an empty DataFrame called `own`, then loop over the windows `[5, 10, 20, "
   "40, 60, 120]` and add a column `vol_<w>d` for each: the standard deviation of "
   "the last `w` returns of `SPY`.",
   "own = pd.DataFrame()\n\nfor w in [5, 10, 20, 40, 60, 120]:\n    ...\n\nprint(list(own.columns))",
   ["The column name is text built from the number: `'vol_' + str(w) + 'd'`.",
    "Inside the loop: `own['vol_' + str(w) + 'd'] = rets['SPY'].rolling(w).std()`. "
    "Assigning to a name that does not exist yet creates the column."],
   "own = pd.DataFrame()\n\nfor w in [5, 10, 20, 40, 60, 120]:\n"
   "    own['vol_' + str(w) + 'd'] = rets['SPY'].rolling(w).std()\n\nprint(list(own.columns))",
   "Six columns from three lines. `str(w)` turns the number into text so it can "
   "be glued into the name, which is the string arithmetic from Session 1 doing "
   "real work.")

ex("A3", "Three return windows", 2,
   "Add three more columns to your `own` from A2, in another loop: `ret_<w>d` "
   "for `w` in `[5, 20, 60]`, the **average** return over the last `w` days.",
   "for w in [5, 20, 60]:\n    ...\n\nprint(list(own.columns))",
   "The same shape as A2 with `.mean()` in place of `.std()` and `'ret_'` in "
   "place of `'vol_'`.",
   "for w in [5, 20, 60]:\n    own['ret_' + str(w) + 'd'] = rets['SPY'].rolling(w).mean()\n\n"
   "print(list(own.columns))",
   "Nine columns. A falling market and a volatile one tend to go together, which "
   "is why the return windows earn their place next to the volatility ones.")

ex("A4", "Every other stock's volatility", 3,
   "Add one column per stock to `own`: `<ticker>_vol`, the 20-day volatility of "
   "that stock's returns, for every ticker in `rets` **except** `SPY`.",
   "for t in rets.columns:\n    ...\n\nprint(len(own.columns), 'columns')",
   ["Loop over `rets.columns` and skip one of them with `if t != 'SPY':`.",
    "The column is `own[t + '_vol'] = rets[t].rolling(20).std()`."],
   "for t in rets.columns:\n    if t != 'SPY':\n        own[t + '_vol'] = rets[t].rolling(20).std()\n\n"
   "print(len(own.columns), 'columns')",
   "Nineteen columns. The `if` inside the loop is the only new thing here, and "
   "it is the Session 2 way of saying \"all but one\".")

ex("A5", "The target, and the finished table", 2,
   "Add the target `vol_next` to `own`, the volatility of the **next** twenty "
   "days, then drop every incomplete row. Print the shape.\n\n"
   "$$\\text{vol\\_next}_t = \\text{sd}\\big(r_{t+1}, \\ldots, r_{t+20}\\big)$$",
   "own['vol_next'] = ...\nown = ...\nprint(...)",
   "The target is the 20-day rolling standard deviation shifted **up** by twenty "
   "rows: `.shift(-20)`. Then `.dropna()`.",
   "own['vol_next'] = rets['SPY'].rolling(20).std().shift(-20)\nown = own.dropna()\nprint(own.shape)",
   f"{N_TBL:,} rows and 20 columns, the same as the lecture's table. The 120-day "
   "window is what costs the most rows at the start; the target costs twenty at "
   "the end.")

ex("A6", "Prove it is the same table", 3,
   "The setup cell loaded the lecture's `table`. Check that your `own` matches it: "
   "the largest absolute difference across every cell should be zero, or as near "
   "as floating point gets.",
   "largest_gap = ...\nprint(largest_gap)",
   ["Subtracting two tables with the same index and columns subtracts cell by "
    "cell. `(own - table).abs()` is every gap.",
    "`.max()` on a table gives the largest per column; `.max()` again gives the "
    "largest overall. Select `table[own.columns]` first so the columns line up."],
   "largest_gap = (own - table[own.columns]).abs().max().max()\nprint(largest_gap)",
   f"About {BUILD_DIFF:.0e}, which is floating-point noise from writing the file "
   "to text and reading it back. No loop over cells was needed: the subtraction, "
   "the absolute value and both maxima are vectorised.",
   revisits="S3")

ex("A7", "A function that builds it for any ticker", 4,
   "Wrap A2 to A5 into `build_table(ticker)`: the six volatility windows and the "
   "three return windows of that ticker, the 20-day volatility of every **other** "
   "instrument, the target, and `dropna()`. Return the table.\n\n"
   "Check it on `'SPY'` and on `'AAPL'`.",
   "def build_table(ticker):\n    ...\n\nspy_table = build_table('SPY')\napple_table = build_table('AAPL')\nprint(spy_table)",
   ["Replace `'SPY'` by `ticker` everywhere, including in the `if`.",
    "The last line of the function is `return frame.dropna()`."],
   "def build_table(ticker):\n    frame = pd.DataFrame()\n"
   "    for w in [5, 10, 20, 40, 60, 120]:\n"
   "        frame['vol_' + str(w) + 'd'] = rets[ticker].rolling(w).std()\n"
   "    for w in [5, 20, 60]:\n"
   "        frame['ret_' + str(w) + 'd'] = rets[ticker].rolling(w).mean()\n"
   "    for t in rets.columns:\n        if t != ticker:\n"
   "            frame[t + '_vol'] = rets[t].rolling(20).std()\n"
   "    frame['vol_next'] = rets[ticker].rolling(20).std().shift(-20)\n"
   "    return frame.dropna()\n\n"
   "print(build_table('SPY').shape)\nprint(build_table('AAPL').shape)",
   "Both are 2,376 by 20. The function is the shape the last section of this "
   "notebook needs, where the whole workflow runs once per instrument.",
   revisits="S2")

# ============================================================ B
section(
"## \U0001f50d B · Nineteen columns, and what OLS does with them\n\n"
"The overfitting from last time, at a larger scale, and a look at why. B1 to "
"B4 share the two models B1 fits."
)

ex("B1", "One column, all columns", 2,
   "Fit two linear regressions on the training rows: one on `vol_20d` alone, one "
   "on all nineteen `columns`. Print the **test** RMSE of each.",
   "one = LinearRegression()\n...\n\nols = LinearRegression()\n...\n\n"
   "print('one column :', ...)\nprint('all columns:', ...)",
   "The setup cell defines `rmse(actual, predicted)`. Remember the double "
   "brackets for a single column: `train[['vol_20d']]`.",
   "one = LinearRegression()\none.fit(train[['vol_20d']], train['vol_next'])\n\n"
   "ols = LinearRegression()\nols.fit(train[columns], train['vol_next'])\n\n"
   "print('one column :', rmse(test['vol_next'], one.predict(test[['vol_20d']])))\n"
   "print('all columns:', rmse(test['vol_next'], ols.predict(test[columns])))",
   f"{ONE_TE:.4f} against {OLS_TE:.4f}. Eighteen extra columns, and the forecast "
   "got worse.")

ex("B2", "And on the training rows", 2,
   "Now print the **training** RMSE of `one` and `ols` from B1. Which is lower, "
   "and what does that say next to B1?",
   "print('one column :', ...)\nprint('all columns:', ...)",
   "Predict on `train[...]` instead of `test[...]`, and score against "
   "`train['vol_next']`.",
   "print('one column :', rmse(train['vol_next'], one.predict(train[['vol_20d']])))\n"
   "print('all columns:', rmse(train['vol_next'], ols.predict(train[columns])))",
   f"{ONE_TR:.4f} against {OLS_TR:.4f}: the nineteen-column model fits the "
   "training rows better and forecasts worse. That gap between training and "
   "test error is what overfitting looks like in numbers.")

ex("B3", "Draw the nineteen coefficients", 3,
   "Draw a horizontal bar chart of the coefficients of `ols` from B1, one bar per "
   "column, with the column names on the axis.",
   "fig, ax = plt.subplots(figsize=(8, 5))\n...\nplt.show()",
   ["`ax.barh(columns, ols.coef_)` draws one bar per name.",
    "`ax.axvline(0, color='black', linewidth=1)` marks zero, and "
    "`ax.invert_yaxis()` puts the first column at the top."],
   "fig, ax = plt.subplots(figsize=(8, 5))\nax.barh(columns, ols.coef_)\n"
   "ax.axvline(0, color='black', linewidth=1)\nax.invert_yaxis()\n"
   "ax.set_xlabel('coefficient')\nax.set_title('Nineteen columns, ordinary least squares', loc='left')\n"
   "plt.show()",
   "Bars pointing both ways, with the volatility of some stocks lowering the "
   "forecast of market volatility. The chart is the quickest way to see that "
   "something is wrong with the coefficients even before you read them.",
   revisits="S3")

ex("B4", "Count the wrong signs", 3,
   "Every **volatility** column in this table should, if anything, raise the "
   "forecast when it rises. Count how many coefficients of `ols` are negative, "
   "and print their names. Which of them are volatility columns?",
   "n_negative = ...\nprint(n_negative)\n\n...",
   ["`ols.coef_ < 0` is an array of True and False; `.sum()` counts the Trues.",
    "`for name, b in zip(columns, ols.coef_):` with `if b < 0: print(name)` inside."],
   "n_negative = (ols.coef_ < 0).sum()\nprint(n_negative)\n\n"
   "for name, b in zip(columns, ols.coef_):\n    if b < 0:\n        print(name)",
   f"{N_NEG} of the nineteen are negative. The two return columns are allowed "
   f"to be, since a falling market is a volatile one. The other {len(NEG_VOL)}, "
   f"{', '.join(NEG_VOL)}, are volatility columns with the wrong sign. The mask "
   "counts in one line; the loop names.",
   revisits="S3")

ex("B5", "Against the two rules that need no model", 3,
   "Score the two baselines from last time on the test rows: guessing the training "
   "average, and repeating the last twenty days (`vol_20d`). Where do the numbers "
   "from B1 sit among the four?",
   "guess = ...\nprint('guess the average :', ...)\nprint('repeat this month :', ...)",
   ["The average is `train['vol_next'].mean()`, repeated with "
    "`np.full(len(test), guess)`.",
    "The persistence forecast is simply the column `test['vol_20d']`."],
   "guess = train['vol_next'].mean()\n"
   "print('guess the average :', rmse(test['vol_next'], np.full(len(test), guess)))\n"
   "print('repeat this month :', rmse(test['vol_next'], test['vol_20d']))",
   f"{MEAN_TE:.4f} and {PERS_TE:.4f}. The nineteen-column OLS, at {OLS_TE:.4f}, "
   "still beats both rules, so it is not useless. It is worse than the single "
   "column, which is the point.",
   revisits="S4")

ex("B6", "How alike are the columns", 3,
   "Compute the correlation between `vol_40d` and `vol_60d` on the training rows. "
   "Then find the **most** correlated pair among all nineteen columns.",
   "corr_40_60 = ...\nprint(corr_40_60)\n\ncorr_table = ...\ncorr_table",
   ["`train['vol_40d'].corr(train['vol_60d'])` for one pair; "
    "`train[columns].corr()` for all of them at once.",
    "Reading the biggest off-diagonal entry by eye is fine. `.round(2)` makes the "
    "table easier to scan."],
   "corr_40_60 = train['vol_40d'].corr(train['vol_60d'])\nprint(corr_40_60)\n\n"
   "corr_table = train[columns].corr().round(2)\ncorr_table",
   f"{CORR_4060:.2f} for the pair in the question, and of the 171 pairs in the "
   f"table the most correlated is `{TOP_PAIR[0]}` with `{TOP_PAIR[1]}` at "
   f"{TOP_CORR:.2f}. Two columns that alike can trade coefficient between them "
   "almost freely, which is where the wrong signs come from.",
   revisits="S4")

ex("B7", "The same coefficient on five different blocks", 4,
   "Fit the nineteen-column OLS on the **fitting rows of each fold** of `folds` "
   "and collect the coefficient on `vol_120d` in a list. Print the list.",
   "spot = columns.index('vol_120d')\ncoefs_120 = []\n\n"
   "for fit_rows, score_rows in folds.split(train):\n    ...\n\nprint(coefs_120)",
   ["`train.iloc[fit_rows]` is the block of rows to fit on. Fit a fresh "
    "`LinearRegression` on it each time.",
    "Append `round(float(model.coef_[spot]), 3)` to the list. `float()` turns "
    "the NumPy number into a plain one, which prints more neatly."],
   "spot = columns.index('vol_120d')\ncoefs_120 = []\n\n"
   "for fit_rows, score_rows in folds.split(train):\n"
   "    block = train.iloc[fit_rows]\n    model = LinearRegression()\n"
   "    model.fit(block[columns], block['vol_next'])\n"
   "    coefs_120.append(round(float(model.coef_[spot]), 3))\n\nprint(coefs_120)",
   f"From {min(FOLD_COEF_120):.2f} to {max(FOLD_COEF_120):.2f}, changing sign "
   "along the way. A coefficient that depends this much on which years it was "
   "fitted on has no business forecasting the next year.",
   revisits="S5")

# ============================================================ C
section(
"## \U0001f3af C · Ridge\n\n"
"One new argument, and what it does to the coefficients."
)

ex("C1", "Ridge in three lines", 1,
   "Fit `Ridge(alpha=1000)` on all nineteen columns of the training rows and "
   "print its test RMSE.",
   "ridge = ...\n...\nprint(...)",
   "`Ridge` fits and predicts exactly like `LinearRegression`.",
   "ridge = Ridge(alpha=1000)\nridge.fit(train[columns], train['vol_next'])\n"
   "print(rmse(test['vol_next'], ridge.predict(test[columns])))",
   f"{RIDGE1000_TE:.4f}, against {OLS_TE:.4f} for the same columns unpenalised. "
   "The only change is the first line.")

ex("C2", "Alpha zero is OLS", 2,
   "Fit ordinary least squares and `Ridge(alpha=0)` on all nineteen columns, and "
   "print the two test RMSEs. Then print whether they agree to six decimals.",
   "plain = ...\n...\nplain_rmse = ...\n\nridge0 = ...\n...\nridge0_rmse = ...\n\nprint(plain_rmse, ridge0_rmse)\nprint(...)",
   "`abs(a - b) < 0.000001` is True or False.",
   "plain = LinearRegression()\nplain.fit(train[columns], train['vol_next'])\n"
   "plain_rmse = rmse(test['vol_next'], plain.predict(test[columns]))\n\n"
   "ridge0 = Ridge(alpha=0)\nridge0.fit(train[columns], train['vol_next'])\n"
   "ridge0_rmse = rmse(test['vol_next'], ridge0.predict(test[columns]))\n\n"
   "print(plain_rmse, ridge0_rmse)\nprint(abs(plain_rmse - ridge0_rmse) < 0.000001)",
   "True. With no penalty there is nothing to shrink, and ridge is ordinary least "
   "squares. Every number the penalty changes is therefore the penalty's doing.",
   revisits="S3")

ex("C3", "Alpha on a validation block", 3,
   "Choosing alpha needs held-out rows. Cut the training block at the end of 2020 "
   "into `fit` and `val`, then loop over `[1, 10, 100, 1000, 10000, 100000]`: fit "
   "a ridge on `fit`, score it on `val`, and print each alpha with its RMSE.",
   "fit = train.loc[:'2020-12-31']\nval = train.loc['2021-01-01':]\n\n"
   "for alpha in [1, 10, 100, 1000, 10000, 100000]:\n    ...",
   "Inside the loop: create `Ridge(alpha=alpha)`, fit on `fit[columns]`, score "
   "on `val`.",
   "fit = train.loc[:'2020-12-31']\nval = train.loc['2021-01-01':]\n\n"
   "for alpha in [1, 10, 100, 1000, 10000, 100000]:\n"
   "    model = Ridge(alpha=alpha)\n    model.fit(fit[columns], fit['vol_next'])\n"
   "    print(alpha, round(rmse(val['vol_next'], model.predict(val[columns])), 4))",
   f"The error falls to alpha {VAL_BEST} and rises after it. The test rows were "
   "not touched, so this is a legitimate way to choose; the folds in section F "
   "are the same idea with five cuts instead of one.",
   revisits="S5")

ex("C4", "A function for the loop body", 3,
   "Write `val_rmse(alpha)`: it fits a ridge with that alpha on `fit` and returns "
   "the RMSE on `val`, both from C3. Use it to store the six results in a "
   "dictionary keyed by alpha, and print the best key.",
   "def val_rmse(alpha):\n    ...\n\nby_alpha = {}\nfor alpha in [1, 10, 100, 1000, 10000, 100000]:\n    ...\n\n"
   "print(by_alpha)\nprint('best:', ...)",
   ["The function body is the loop body from C3 with `return` in place of "
    "`print`.",
    "`min(by_alpha, key=by_alpha.get)` is the key with the smallest value."],
   "def val_rmse(alpha):\n    model = Ridge(alpha=alpha)\n"
   "    model.fit(fit[columns], fit['vol_next'])\n"
   "    return rmse(val['vol_next'], model.predict(val[columns]))\n\n"
   "by_alpha = {}\nfor alpha in [1, 10, 100, 1000, 10000, 100000]:\n"
   "    by_alpha[alpha] = round(val_rmse(alpha), 4)\n\n"
   "print(by_alpha)\nprint('best:', min(by_alpha, key=by_alpha.get))",
   f"Best at {VAL_BEST}. A function, a dictionary and `min` with a key: the "
   "Session 2 toolkit, and it is exactly what `GridSearchCV` does internally.",
   revisits="S2")

ex("C5", "The intercept is not penalised", 3,
   "Fit ridge with alpha 1, 1000 and 10,000,000 and print each intercept. Then "
   "print the training mean of `vol_next`. What does the intercept become as the "
   "slopes are crushed?",
   "for alpha in [1, 1000, 10000000]:\n    ...\n\nprint('training mean:', ...)",
   "Print `model.intercept_` inside the loop. The mean is `train['vol_next'].mean()`.",
   "for alpha in [1, 1000, 10000000]:\n    model = Ridge(alpha=alpha)\n"
   "    model.fit(train[columns], train['vol_next'])\n"
   "    print(alpha, round(model.intercept_, 4))\n\n"
   "print('training mean:', round(train['vol_next'].mean(), 4))",
   f"The intercept moves from {INTERCEPTS[1]:.3f} to {INTERCEPTS[10_000_000]:.3f}, "
   f"and the training mean is {TRAIN_MEAN:.3f}. With every slope at zero the "
   "forecast is the intercept, and the intercept is free to be the average. That "
   "is why the penalty leaves it alone.")

ex("C6", "How much coefficient is left", 4,
   "The ridge penalty charges for $\\sum_j \\beta_j^2$. Compute that sum for alpha "
   "0, 1000 and 100000, without a loop over the coefficients.",
   "for alpha in [0, 1000, 100000]:\n    model = Ridge(alpha=alpha)\n"
   "    model.fit(train[columns], train['vol_next'])\n    size = ...\n    print(alpha, size)",
   "`model.coef_` is an array. Square it and sum it: `(model.coef_ ** 2).sum()`.",
   "for alpha in [0, 1000, 100000]:\n    model = Ridge(alpha=alpha)\n"
   "    model.fit(train[columns], train['vol_next'])\n"
   "    size = (model.coef_ ** 2).sum()\n    print(alpha, round(size, 5))",
   f"{SUMSQ[0]:.3f}, then {SUMSQ[1000]:.3f}, then {SUMSQ[100000]:.5f}. The penalty "
   "term is the thing being squeezed, and you can watch it shrink. Squaring and "
   "summing a whole array at once is the vectorised habit from Session 3.",
   revisits="S3")

# ============================================================ D
section(
"## \U0001f4cf D · The penalty and the units\n\n"
"Why a penalised model needs every column on the same scale, and how to put "
"them there without leaking the test rows. D1 and D2 share the decimal copies "
"D1 makes; D3 to D6 share the scaler D3 fits."
)

ex("D1", "OLS does not care about units", 2,
   "Make copies of `train` and `test` in which `vol_20d` is divided by 100 (so it "
   "is in decimals). Fit OLS on both versions and print the coefficient on "
   "`vol_20d` from each. Then print the ratio.",
   "train_dec = train.copy()\ntest_dec = test.copy()\n"
   "train_dec['vol_20d'] = train_dec['vol_20d'] / 100\ntest_dec['vol_20d'] = test_dec['vol_20d'] / 100\n"
   "spot = columns.index('vol_20d')\n\n"
   "ols_pct = ...\n...\nols_dec = ...\n...\n\n"
   "print(..., ...)\nprint('ratio:', ...)",
   "Fit `ols_pct` on `train[columns]` and `ols_dec` on `train_dec[columns]`, then "
   "print `ols_pct.coef_[spot]` and `ols_dec.coef_[spot]`. The ratio is one "
   "coefficient divided by the other.",
   "train_dec = train.copy()\ntest_dec = test.copy()\n"
   "train_dec['vol_20d'] = train_dec['vol_20d'] / 100\ntest_dec['vol_20d'] = test_dec['vol_20d'] / 100\n"
   "spot = columns.index('vol_20d')\n\n"
   "ols_pct = LinearRegression()\nols_pct.fit(train[columns], train['vol_next'])\n"
   "ols_dec = LinearRegression()\nols_dec.fit(train_dec[columns], train_dec['vol_next'])\n\n"
   "print(ols_pct.coef_[spot], ols_dec.coef_[spot])\nprint('ratio:', ols_dec.coef_[spot] / ols_pct.coef_[spot])",
   f"{OLS_COEF:.4f} and {OLS_COEF_DEC:.4f}: a ratio of exactly 100. The "
   "coefficient absorbs the change of units and the forecast is identical.",
   revisits="S3")

ex("D2", "Ridge does care", 2,
   "Repeat D1 with `Ridge(alpha=1000)` on both versions, using `train_dec` and "
   "`spot` from D1. What happens to the coefficient on `vol_20d` when the column "
   "is in decimals?",
   "ridge_pct = ...\n...\nridge_dec = ...\n...\n\nprint(..., ...)",
   "Same code as D1 with `Ridge(alpha=1000)` in place of `LinearRegression()`.",
   "ridge_pct = Ridge(alpha=1000)\nridge_pct.fit(train[columns], train['vol_next'])\n"
   "ridge_dec = Ridge(alpha=1000)\nridge_dec.fit(train_dec[columns], train_dec['vol_next'])\n\n"
   "print(ridge_pct.coef_[spot], ridge_dec.coef_[spot])",
   f"{RIDGE_COEF:.4f} in percent, {RIDGE_COEF_DEC:.5f} in decimals. The column "
   "would need a coefficient a hundred times larger, which costs ten thousand "
   "times more penalty, so ridge drops it. Nothing about its information changed.")

ex("D3", "StandardScaler", 2,
   "Fit a `StandardScaler` on the training columns, transform both blocks into "
   "`X_train` and `X_test`, and print the mean and standard deviation the scaler "
   "learned for `vol_20d`.",
   "spot = columns.index('vol_20d')\n\nscaler = ...\n...\nX_train = ...\nX_test = ...\n\nprint(..., ...)",
   "`scaler.fit(train[columns])`, then `scaler.transform(...)` on each block. The "
   "learned numbers are `mean_` and `scale_`, one per column.",
   "spot = columns.index('vol_20d')\n\nscaler = StandardScaler()\nscaler.fit(train[columns])\n"
   "X_train = scaler.transform(train[columns])\nX_test = scaler.transform(test[columns])\n\n"
   "print(scaler.mean_[spot], scaler.scale_[spot])",
   f"A mean of {SC_MEAN:.3f} and a standard deviation of {SC_SCALE:.3f}, in "
   "percent, learned from the training rows alone. `transform` returns a NumPy "
   "array, so the column names are gone and `spot` is how you find a column.")

ex("D4", "The same thing by hand", 3,
   "Standardise `vol_20d` yourself with the formula from Session 4, using "
   "`np.std` for the standard deviation, and check the result against column "
   "`spot` of `X_train` from D3: print the largest absolute difference.\n\n"
   "$$z = \\frac{x - \\bar{x}}{s}$$",
   "z_hand = ...\nlargest_gap = ...\nprint(largest_gap)",
   ["`(train['vol_20d'] - train['vol_20d'].mean()) / np.std(train['vol_20d'])`.",
    "`np.abs(z_hand - X_train[:, spot]).max()`. `X_train[:, spot]` is one column "
    "of the array, as in Session 3's matrices."],
   "z_hand = (train['vol_20d'] - train['vol_20d'].mean()) / np.std(train['vol_20d'])\n"
   "largest_gap = np.abs(z_hand - X_train[:, spot]).max()\nprint(largest_gap)",
   f"{'Zero' if HAND_DIFF < 1e-12 else 'About %.0e' % HAND_DIFF}, or a number of the order 1e-16: "
   "the same numbers. `np.std` is used rather than "
   "pandas' `.std()` because the scaler divides by $n$ and pandas divides by "
   "$n - 1$; the difference is in the fourth decimal, and it is the kind of thing "
   "a check like this is for.",
   revisits="S3")

ex("D5", "The test rows, in training units", 3,
   "Print the mean and standard deviation of column `spot` of `X_test` from D3. "
   "They are not 0 and 1. Why not, and what do the numbers say about 2023 and "
   "2024?",
   "print(..., ...)",
   "`X_test[:, spot].mean()` and `X_test[:, spot].std()`.",
   "print(X_test[:, spot].mean(), X_test[:, spot].std())",
   f"A mean of {XTEST_MEAN:.2f} and a standard deviation of {XTEST_STD:.2f}. The "
   "test rows were transformed with the **training** mean and spread, as they "
   "must be, and 2023 and 2024 were calmer and steadier than the training years. "
   "A mean of exactly zero here would be the sign of a leak.",
   revisits="S3")

ex("D6", "Fix the leak", 4,
   "The cell below fits a second scaler on the test rows. That uses the test "
   "block's own mean and spread, which is information from the future. Fit a "
   "ridge on `X_train` from D3, score it on the leaky array, then transform the "
   "test rows properly with `scaler` from D3 and score again.",
   "leaky = StandardScaler()\nleaky.fit(test[columns])          # the leak\nX_test_leaky = leaky.transform(test[columns])\n\n"
   "ridge_sc = ...\n...\nprint('leaky :', ...)\n\nX_test = ...\nprint('proper:', ...)",
   "The proper version is one line: `scaler.transform(test[columns])`, with the "
   "scaler that was fitted on `train`.",
   "leaky = StandardScaler()\nleaky.fit(test[columns])          # the leak\nX_test_leaky = leaky.transform(test[columns])\n\n"
   "ridge_sc = Ridge(alpha=1000)\nridge_sc.fit(X_train, train['vol_next'])\n"
   "print('leaky :', rmse(test['vol_next'], ridge_sc.predict(X_test_leaky)))\n\n"
   "X_test = scaler.transform(test[columns])\nprint('proper:', rmse(test['vol_next'], ridge_sc.predict(X_test)))",
   f"{LEAK_TE:.4f} with the leak and {RIDGE_SC_TE:.4f} without. The leaky "
   "version is worse here, not better, because it re-centres a calm test period "
   "as if it were average. Either way it is a number that could not have been "
   "computed on the day the forecast was made.",
   revisits="S5")

# ============================================================ E
section(
"## \U0001f517 E · Pipeline\n\n"
"The scaler and the model as one object, so the folds refit both. E2 to E4 "
"use the pipeline E1 builds."
)

ex("E1", "Scale and fit in one object", 1,
   "Build a `Pipeline` with a `StandardScaler` step called `'scale'` and a "
   "`Ridge(alpha=1000)` step called `'ridge'`. Fit it and print the test RMSE.",
   "pipe = Pipeline([...])\n...\nprint(...)",
   "`Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1000))])`, "
   "then `.fit` and `.predict` on the raw columns. The pipeline does the scaling.",
   "pipe = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1000))])\n"
   "pipe.fit(train[columns], train['vol_next'])\n"
   "print(rmse(test['vol_next'], pipe.predict(test[columns])))",
   f"{PIPE_TE:.4f}, the same as scaling by hand in D6. The raw columns go in; "
   "the pipeline transforms them with its own fitted scaler before the ridge sees "
   "them.")

ex("E2", "Reach inside", 2,
   "Get the fitted `Ridge` out of `pipe` from E1, and print the name of the "
   "column with the largest coefficient in absolute value.",
   "coefs = ...\nbiggest = ...\nprint(biggest)",
   ["`pipe.named_steps['ridge'].coef_`.",
    "`np.abs(coefs).argmax()` is the position of the largest absolute value; "
    "`columns[...]` turns it into a name."],
   "coefs = pipe.named_steps['ridge'].coef_\nbiggest = columns[np.abs(coefs).argmax()]\nprint(biggest)",
   f"`{BIGGEST}`, at {BIGGEST_VAL:.3f} per standard deviation. The pipeline "
   "itself has no `coef_`; the step inside it does.",
   revisits="S5")

ex("E3", "Cross-validate the pipeline", 3,
   "Run `cross_val_score` on `pipe` with `folds`, print the five fold RMSEs and "
   "their mean, and print the number of the worst fold.",
   "scores = ...\nprint(...)\nprint('mean :', ...)\nprint('worst:', ...)",
   ["Exactly as for `LinearRegression()` last time: "
    "`cross_val_score(pipe, train[columns], train['vol_next'], cv=folds, "
    "scoring='neg_root_mean_squared_error')`.",
    "`(-scores).argmax() + 1` is the fold number, counting from one."],
   "scores = cross_val_score(pipe, train[columns], train['vol_next'],\n"
   "                         cv=folds, scoring='neg_root_mean_squared_error')\n"
   "print((-scores).round(3))\nprint('mean :', round(-scores.mean(), 4))\n"
   "print('worst:', (-scores).argmax() + 1)",
   f"A mean of {PIPE_CV:.4f}, and fold {WORST_FOLD} is the worst at "
   f"{PIPE_FOLDS.max():.3f}, because it is scored on the block containing March "
   "2020. Inside each fold the scaler was refitted on that fold's fitting rows.",
   revisits="S5")

ex("E4", "What refitting the scaler is worth here", 4,
   "Do it the leaky way for comparison: fit one scaler on **all** the training "
   "rows and transform them once, then cross-validate a plain `Ridge(alpha=1000)` "
   "on the scaled array. Print that mean next to the pipeline's from E3. How large "
   "is the difference, and why?",
   "once = StandardScaler()\n...\nX_all = ...\nleaky = ...\n\nprint('proper:', ...)\nprint('leaky :', ...)",
   ["`once.fit(train[columns])`, `X_all = once.transform(train[columns])`, then "
    "`cross_val_score(Ridge(alpha=1000), X_all, train['vol_next'], cv=folds, ...)`.",
    "The difference is small because a mean and a standard deviation over "
    "hundreds of rows barely move when a block is added."],
   "once = StandardScaler()\nonce.fit(train[columns])\nX_all = once.transform(train[columns])\n"
   "leaky = cross_val_score(Ridge(alpha=1000), X_all, train['vol_next'],\n"
   "                        cv=folds, scoring='neg_root_mean_squared_error')\n\n"
   "print('proper:', round(-scores.mean(), 4))\nprint('leaky :', round(-leaky.mean(), 4))",
   f"{PIPE_CV:.4f} against {LEAKY_CV:.4f}. The leak is real and its effect is "
   "tiny, because the scaler's two numbers are stable. The pipeline costs nothing "
   "and removes the question, which is why it is the habit to build.",
   revisits="S5")

# ============================================================ F
section(
"## \U0001f39b️ F · Choosing alpha\n\n"
"A loop, a curve, and the object that does both for you."
)

ex("F1", "A loop over alphas", 2,
   "For each alpha in `[1, 10, 100, 1000, 10000]`, build the pipeline, "
   "cross-validate it with `folds`, and store the mean RMSE in a dictionary "
   "`cv_by_alpha`. Print the dictionary and the best key.",
   "cv_by_alpha = {}\n\nfor alpha in [1, 10, 100, 1000, 10000]:\n    ...\n\n"
   "print(cv_by_alpha)\nprint('best:', ...)",
   ["The loop body is E3 with a fresh pipeline, `Ridge(alpha=alpha)`, and the "
    "mean stored under `cv_by_alpha[alpha]`. Give the pipeline its own name so "
    "`pipe` from E1 is kept.",
    "`min(cv_by_alpha, key=cv_by_alpha.get)`."],
   "cv_by_alpha = {}\n\nfor alpha in [1, 10, 100, 1000, 10000]:\n"
   "    candidate = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=alpha))])\n"
   "    scores = cross_val_score(candidate, train[columns], train['vol_next'],\n"
   "                             cv=folds, scoring='neg_root_mean_squared_error')\n"
   "    cv_by_alpha[alpha] = round(float(-scores.mean()), 4)\n\n"
   "print(cv_by_alpha)\nprint('best:', min(cv_by_alpha, key=cv_by_alpha.get))",
   f"Best at {CV_BEST}, with the error falling until then and rising after. Five "
   "fits per alpha, twenty-five in all, and the test rows untouched.",
   revisits="S2")

ex("F2", "The validation curve", 3,
   "Repeat F1 on the finer grid `np.logspace(0, 5, 11)` and draw the mean RMSE "
   "against alpha as a line, with a logarithmic x-axis.",
   "alphas = np.logspace(0, 5, 11)\nerrors = []\n\nfor alpha in alphas:\n    ...\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3.2))\n...\nplt.show()",
   ["`np.logspace(0, 5, 11)` is eleven values from 1 to 100,000, each a factor of "
    "about 3.2 apart. Append each mean to `errors`.",
    "`ax.plot(alphas, errors)` then `ax.set_xscale('log')`, so the factors of ten "
    "are evenly spaced."],
   "alphas = np.logspace(0, 5, 11)\nerrors = []\n\nfor alpha in alphas:\n"
   "    candidate = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=alpha))])\n"
   "    scores = cross_val_score(candidate, train[columns], train['vol_next'],\n"
   "                             cv=folds, scoring='neg_root_mean_squared_error')\n"
   "    errors.append(-scores.mean())\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3.2))\nax.plot(alphas, errors, marker='o')\n"
   "ax.set_xscale('log')\nax.set_xlabel('alpha')\nax.set_ylabel('cross-validated RMSE')\n"
   "ax.set_title('The validation curve', loc='left')\nplt.show()",
   f"A U shape with its bottom near {CV_FINE_BEST:,.0f}. Left of it the penalty "
   "is too weak and the model overfits; right of it the penalty removes the "
   "signal. The log axis is what makes the curve readable.",
   revisits="S3")

ex("F3", "GridSearchCV", 2,
   "Let `GridSearchCV` run F1. Build a pipeline `base` with `Ridge()` and no "
   "alpha, a grid `{'ridge__alpha': [1, 10, 100, 1000, 10000]}`, fit the search "
   "on the training rows as `search`, and print the best alpha and the best score.",
   "base = ...\ngrid = ...\n\nsearch = ...\n...\n\nprint(...)\nprint(...)",
   "`GridSearchCV(base, grid, cv=folds, scoring='neg_root_mean_squared_error')`, "
   "then `.fit`. The best score is negative; print `-search.best_score_`.",
   "base = Pipeline([('scale', StandardScaler()), ('ridge', Ridge())])\n"
   "grid = {'ridge__alpha': [1, 10, 100, 1000, 10000]}\n\n"
   "search = GridSearchCV(base, grid, cv=folds, scoring='neg_root_mean_squared_error')\n"
   "search.fit(train[columns], train['vol_next'])\n\n"
   "print(search.best_params_)\nprint(-search.best_score_)",
   f"Alpha {GS_BEST} at {GS_SCORE:.4f}, exactly the numbers your loop in F1 "
   "produced. `'ridge__alpha'` is the step name, two underscores, then the "
   "argument.")

ex("F4", "Read the whole grid", 2,
   "Turn `search.cv_results_` from F3 into a DataFrame and show the columns "
   "`param_ridge__alpha`, `mean_test_score`, `std_test_score` and "
   "`rank_test_score`, sorted by rank.",
   "results = ...\nresults",
   "`pd.DataFrame(search.cv_results_)`, then select the four columns with a list "
   "inside the brackets and `.sort_values('rank_test_score')`.",
   "results = pd.DataFrame(search.cv_results_)\n"
   "results = results[['param_ridge__alpha', 'mean_test_score', 'std_test_score', 'rank_test_score']]\n"
   "results.sort_values('rank_test_score')",
   "One row per alpha, with the mean score, its spread across the folds, and a "
   "rank. The spread column is the fold-to-fold variation from last time, and it "
   "is far larger than the gaps between the top ranks.")

ex("F5", "Predict with the search", 3,
   "Use `search` directly to predict the test rows and print the RMSE. Then "
   "print the alpha inside `search.best_estimator_`.",
   "print('test RMSE:', ...)\nprint('alpha    :', ...)",
   ["`search.predict(test[columns])` uses the best pipeline, refitted on all the "
    "training rows.",
    "`search.best_estimator_.named_steps['ridge'].alpha` reads the setting back."],
   "print('test RMSE:', rmse(test['vol_next'], search.predict(test[columns])))\n"
   "print('alpha    :', search.best_estimator_.named_steps['ridge'].alpha)",
   f"{GS_TE:.4f} with alpha {GS_BEST}. The search chose alpha on the folds and "
   "refitted on every training row; the test rows are opened once, here.")

ex("F6", "A grid with the answer outside it", 4,
   "Search a scaling-plus-`Ridge()` pipeline over the grid `[1, 2, 3, 4, 5]`. "
   "Print the best alpha and score, then write an `if` that prints a warning when "
   "the best value is the largest one in the grid.",
   "narrow = {'ridge__alpha': [1, 2, 3, 4, 5]}\nnarrow_search = ...\n...\n\n"
   "best = ...\nprint(best, ...)\n\nif ...:\n    print('the best value is at the edge of the grid: widen it')",
   ["`best = narrow_search.best_params_['ridge__alpha']`.",
    "`if best == max(narrow['ridge__alpha']):`."],
   "narrow = {'ridge__alpha': [1, 2, 3, 4, 5]}\n"
   "narrow_search = GridSearchCV(Pipeline([('scale', StandardScaler()), ('ridge', Ridge())]),\n"
   "                             narrow, cv=folds, scoring='neg_root_mean_squared_error')\n"
   "narrow_search.fit(train[columns], train['vol_next'])\n\n"
   "best = narrow_search.best_params_['ridge__alpha']\nprint(best, -narrow_search.best_score_)\n\n"
   "if best == max(narrow['ridge__alpha']):\n"
   "    print('the best value is at the edge of the grid: widen it')",
   f"Best at {NARROW_BEST}, the edge, with a score of {NARROW_SCORE:.4f} against "
   f"{GS_SCORE:.4f} for the proper grid. A search can only pick from what it was "
   "given. A winner at the edge means the grid was wrong, not that the answer is "
   "5, and a two-line check catches it.",
   revisits="S1")

ex("F7", "Grow alpha until the error turns", 5,
   "Without a grid: start at alpha 1, cross-validate, multiply alpha by ten, and "
   "keep going **while** the error keeps falling. Stop at the first rise, with a "
   "`break`, and print the last alpha that improved. The loop below already stops "
   "at 100,000 so it can never run forever.",
   "alpha = 1\nprevious = None\nbest = None\n\nwhile alpha <= 100000:\n    ...\n    alpha = alpha * 10\n\n"
   "print('last improvement at alpha', best)",
   ["Compute the mean error for the current alpha. If `previous` is not `None` "
    "and the error is larger than `previous`, `break`.",
    "Otherwise store the error in `previous`, remember this alpha as the best so "
    "far, and multiply alpha by ten."],
   "alpha = 1\nprevious = None\nbest = None\n\nwhile alpha <= 100000:\n"
   "    candidate = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=alpha))])\n"
   "    scores = cross_val_score(candidate, train[columns], train['vol_next'],\n"
   "                             cv=folds, scoring='neg_root_mean_squared_error')\n"
   "    error = -scores.mean()\n    print(alpha, round(error, 4))\n"
   "    if previous is not None and error > previous:\n        break\n"
   "    previous = error\n    best = alpha\n    alpha = alpha * 10\n\n"
   "print('last improvement at alpha', best)",
   f"It stops after alpha {WHILE_TRACE[-1][0]:,}, and the last improvement was at "
   f"{WHILE_BEST:,}. A `while` loop with a `break` is the Session 2 tool for "
   "\"keep going until something happens\", and it finds the bottom of the "
   "validation curve without deciding the grid in advance.",
   revisits="S2")

# ============================================================ G
section(
"## ✂️ G · Lasso\n\n"
"The penalty that sets coefficients to exactly zero."
)

ex("G1", "Lasso in a pipeline", 1,
   "Build and fit a pipeline with a `StandardScaler` and `Lasso(alpha=0.1)`, and "
   "print the test RMSE.",
   "lasso = Pipeline([...])\n...\nprint(...)",
   "Same as E1 with `Lasso(alpha=0.1)` as the second step, named `'lasso'`.",
   "lasso = Pipeline([('scale', StandardScaler()), ('lasso', Lasso(alpha=0.1))])\n"
   "lasso.fit(train[columns], train['vol_next'])\n"
   "print(rmse(test['vol_next'], lasso.predict(test[columns])))",
   f"{LASSO_TE:.4f}, level with ridge at {PIPE_TE:.4f}.")

ex("G2", "Count the zeros", 2,
   "Get the coefficients out of `lasso` from G1 and count how many are exactly "
   "zero.",
   "coefs = ...\nn_zero = ...\nprint(n_zero, 'of', len(columns))",
   "`lasso.named_steps['lasso'].coef_`, then `(coefs == 0).sum()`. With ridge "
   "that count would be zero.",
   "coefs = lasso.named_steps['lasso'].coef_\nn_zero = (coefs == 0).sum()\nprint(n_zero, 'of', len(columns))",
   f"{N_ZERO} of 19. Not small; exactly zero. Those columns play no part in the "
   "forecast.")

ex("G3", "Which columns survived", 2,
   "Print the name and coefficient of every column the lasso kept, to three "
   "decimals, using `coefs` from G2.",
   "...",
   "`for name, b in zip(columns, coefs):` with `if b != 0: print(name, round(b, 3))` "
   "inside.",
   "for name, b in zip(columns, coefs):\n    if b != 0:\n        print(name, round(b, 3))",
   f"{', '.join(KEPT)}: the last few days of the index's own history, its recent "
   "return, and two of the stocks. Five columns out of nineteen, chosen by the "
   "penalty rather than by hand.",
   revisits="S2")

ex("G4", "Lasso's alpha lives on another scale", 3,
   "Build a lasso pipeline with `Lasso()` and no alpha, and run a grid search "
   "over `[0.001, 0.01, 0.1, 1]` as `lasso_search`. Print the best alpha and "
   "score. Why is this grid a thousand times smaller than ridge's?",
   "lasso_pipe = ...\nlasso_grid = ...\n\nlasso_search = ...\n...\n\nprint(...)\nprint(...)",
   "The grid key is `'lasso__alpha'` now, because that is the step's name.",
   "lasso_pipe = Pipeline([('scale', StandardScaler()), ('lasso', Lasso())])\n"
   "lasso_grid = {'lasso__alpha': [0.001, 0.01, 0.1, 1]}\n\n"
   "lasso_search = GridSearchCV(lasso_pipe, lasso_grid, cv=folds, scoring='neg_root_mean_squared_error')\n"
   "lasso_search.fit(train[columns], train['vol_next'])\n\n"
   "print(lasso_search.best_params_)\nprint(-lasso_search.best_score_)",
   f"Alpha {GL_BEST} at {GL_SCORE:.4f}, against {GS_SCORE:.4f} for ridge. The "
   "lasso charges $|\\beta|$ and ridge charges $\\beta^2$; for coefficients around "
   "0.1 the square is a hundredth of the absolute value, so the same alpha means "
   "a very different price. A grid for one is never a grid for the other.")

ex("G5", "The kept set, fold by fold", 4,
   "Fit the lasso pipeline (alpha 0.1) on the **fitting rows of each fold** and "
   "print the list of columns it keeps each time. Does the same set come back?",
   "for fit_rows, score_rows in folds.split(train):\n    ...",
   ["`block = train.iloc[fit_rows]`, fit a fresh pipeline on it, then read "
    "`named_steps['lasso'].coef_`.",
    "Build the list of kept names with a loop over `zip(columns, coefs)` and an "
    "`if`, or with `[n for n, b in zip(columns, coefs) if b != 0]`."],
   "for fit_rows, score_rows in folds.split(train):\n"
   "    block = train.iloc[fit_rows]\n"
   "    fold_lasso = Pipeline([('scale', StandardScaler()), ('lasso', Lasso(alpha=0.1))])\n"
   "    fold_lasso.fit(block[columns], block['vol_next'])\n"
   "    fold_coefs = fold_lasso.named_steps['lasso'].coef_\n    kept = []\n"
   "    for name, b in zip(columns, fold_coefs):\n        if b != 0:\n            kept.append(name)\n"
   "    print(kept)",
   f"Five different lists. Only {', '.join(ALWAYS_KEPT) if ALWAYS_KEPT else 'nothing'} "
   "survives every block. Among near-copies the lasso keeps one and drops the "
   "rest, and which one depends on the rows, so a zero is not a verdict on the "
   "column.",
   revisits="S5")

ex("G6", "How many survive as alpha grows", 4,
   "For alpha in `[0.01, 0.03, 0.1, 0.3, 1]`, fit the lasso pipeline and store the "
   "number of **non-zero** coefficients in a dictionary. Print it.",
   "survivors = {}\n\nfor alpha in [0.01, 0.03, 0.1, 0.3, 1]:\n    ...\n\nprint(survivors)",
   "`int((candidate.named_steps['lasso'].coef_ != 0).sum())` counts the "
   "survivors as a plain number. Give each fitted pipeline its own name so "
   "`lasso` from G1 is kept.",
   "survivors = {}\n\nfor alpha in [0.01, 0.03, 0.1, 0.3, 1]:\n"
   "    candidate = Pipeline([('scale', StandardScaler()), ('lasso', Lasso(alpha=alpha))])\n"
   "    candidate.fit(train[columns], train['vol_next'])\n"
   "    survivors[alpha] = int((candidate.named_steps['lasso'].coef_ != 0).sum())\n\nprint(survivors)",
   f"{SURVIVORS[0.01]}, {SURVIVORS[0.03]}, {SURVIVORS[0.1]}, {SURVIVORS[0.3]} and "
   f"then {SURVIVORS[1]}. At alpha 1 every coefficient is zero and the forecast is "
   "the training average. The lasso path from the lecture is this dictionary "
   "drawn as lines.",
   revisits="S2")

# ============================================================ H
section(
"## \U0001f9ec H · Elastic net\n\n"
"Both penalties at once, and a grid with two keys."
)

ex("H1", "Both penalties", 2,
   "Fit a pipeline with `ElasticNet(alpha=0.1, l1_ratio=0.5)`. Print the test RMSE "
   "and the number of non-zero coefficients.",
   "enet = Pipeline([...])\n...\n\nprint('test RMSE:', ...)\nprint('non-zero :', ...)",
   "Name the step `'enet'`. The coefficients are `enet.named_steps['enet'].coef_`.",
   "enet = Pipeline([('scale', StandardScaler()), ('enet', ElasticNet(alpha=0.1, l1_ratio=0.5))])\n"
   "enet.fit(train[columns], train['vol_next'])\n\n"
   "print('test RMSE:', rmse(test['vol_next'], enet.predict(test[columns])))\n"
   "print('non-zero :', (enet.named_steps['enet'].coef_ != 0).sum())",
   f"{ENET_TE:.4f} with {ENET_NZ} columns kept: some zeros, as with the lasso, and "
   "the rest shrunk, as with ridge.")

ex("H2", "A grid with two keys", 3,
   "Search `alpha` over `[0.01, 0.1, 1]` and `l1_ratio` over `[0.1, 0.5, 0.9]` at "
   "the same time. Print the best pair, the best score, and how many combinations "
   "were tried.",
   "enet = Pipeline([('scale', StandardScaler()), ('enet', ElasticNet())])\ngrid = {...}\n\n"
   "search = ...\n...\n\nprint(...)\nprint(...)\nprint('combinations:', ...)",
   ["Two keys in one dictionary: `{'enet__alpha': [...], 'enet__l1_ratio': [...]}`.",
    "`len(search.cv_results_['params'])` is the number of combinations."],
   "enet = Pipeline([('scale', StandardScaler()), ('enet', ElasticNet())])\n"
   "grid = {'enet__alpha': [0.01, 0.1, 1], 'enet__l1_ratio': [0.1, 0.5, 0.9]}\n\n"
   "search = GridSearchCV(enet, grid, cv=folds, scoring='neg_root_mean_squared_error')\n"
   "search.fit(train[columns], train['vol_next'])\n\n"
   "print(search.best_params_)\nprint(-search.best_score_)\n"
   "print('combinations:', len(search.cv_results_['params']))",
   f"Alpha {GE_BEST['enet__alpha']} with l1_ratio {GE_BEST['enet__l1_ratio']}, at "
   f"{GE_SCORE:.4f}, from {GE_N} combinations and {5 * GE_N} fits. Two keys "
   "multiply; three keys would multiply again, which is why grids stay short.")

ex("H3", "l1_ratio one is the lasso", 4,
   "Fit two scaling pipelines, one with `ElasticNet(alpha=0.1, l1_ratio=1.0)` and "
   "one with `Lasso(alpha=0.1)`, and print the largest absolute difference between "
   "their coefficient arrays. Then write an assertion that it is below `1e-6`.",
   "enet1 = ...\n...\nlasso1 = ...\n...\n\ngap = ...\nprint(gap)\nassert ...",
   "`np.abs(a - b).max()` on the two `coef_` arrays, then `assert gap < 1e-6`.",
   "enet1 = Pipeline([('scale', StandardScaler()), ('enet', ElasticNet(alpha=0.1, l1_ratio=1.0))])\n"
   "enet1.fit(train[columns], train['vol_next'])\n"
   "lasso1 = Pipeline([('scale', StandardScaler()), ('lasso', Lasso(alpha=0.1))])\n"
   "lasso1.fit(train[columns], train['vol_next'])\n\n"
   "gap = np.abs(enet1.named_steps['enet'].coef_ - lasso1.named_steps['lasso'].coef_).max()\n"
   "print(gap)\nassert gap < 1e-6",
   f"{'Zero' if ENET_VS_LASSO < 1e-12 else 'About %.0e' % ENET_VS_LASSO}, or within 1e-16 of it: "
   "the same model. `l1_ratio=1` switches the ridge "
   "part off, and `l1_ratio=0` would switch the lasso part off. An assertion is a "
   "check that shouts if it is ever wrong, and it costs one line.",
   revisits="S3")

# ============================================================ I
section(
"## \U0001f527 I · The arguments\n\n"
"Reading the settings, and the three you will actually change."
)

ex("I1", "What the defaults are", 1,
   "Print `max_iter` from `Lasso().get_params()`, and the value of "
   "`'ridge__alpha'` from the `get_params()` of a scaling pipeline whose ridge "
   "has `alpha=1000`.",
   "print(...)\nprint(...)",
   "`get_params()` returns a dictionary; read a key from it with square brackets. "
   "The pipeline can be built inside the print.",
   "print(Lasso().get_params()['max_iter'])\n"
   "print(Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1000))]).get_params()['ridge__alpha'])",
   f"{DEFAULT_MAX_ITER} passes by default, and the pipeline reports its ridge's "
   "alpha under the same `step__argument` name a grid uses.")

ex("I2", "Provoke the warning, then cure it", 3,
   "Fit `Lasso(alpha=0.001, max_iter=50)` in a scaling pipeline and watch the "
   "`ConvergenceWarning` appear. Then fit again with enough passes for it to go "
   "away, and print both test RMSEs.",
   "few = Pipeline([('scale', StandardScaler()), ('lasso', Lasso(alpha=0.001, max_iter=50))])\n"
   "few.fit(train[columns], train['vol_next'])\nprint('50 passes  :', rmse(test['vol_next'], few.predict(test[columns])))\n\n"
   "enough = ...\n...\nprint('more passes:', ...)",
   "The warning names the cure. `Lasso(alpha=0.001, max_iter=5000)` is enough here.",
   "few = Pipeline([('scale', StandardScaler()), ('lasso', Lasso(alpha=0.001, max_iter=50))])\n"
   "few.fit(train[columns], train['vol_next'])\nprint('50 passes  :', rmse(test['vol_next'], few.predict(test[columns])))\n\n"
   "enough = Pipeline([('scale', StandardScaler()), ('lasso', Lasso(alpha=0.001, max_iter=5000))])\n"
   "enough.fit(train[columns], train['vol_next'])\nprint('more passes:', rmse(test['vol_next'], enough.predict(test[columns])))",
   "The first fit prints a warning and returns a result anyway; the second is "
   "silent. The numbers are close, which is the dangerous part: a model that did "
   "not converge looks like one that did. Read your warnings.",
   revisits="S1")

ex("I3", "Force the signs", 2,
   "Fit a scaling pipeline with `Ridge(alpha=1000, positive=True)`. Print the test "
   "RMSE and count the negative coefficients.",
   "pos = Pipeline([...])\n...\n\nprint('test RMSE:', ...)\nprint('negative :', ...)",
   "`positive=True` is an argument of `Ridge`. Count with `(coefs < 0).sum()`.",
   "pos = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1000, positive=True))])\n"
   "pos.fit(train[columns], train['vol_next'])\n\n"
   "print('test RMSE:', rmse(test['vol_next'], pos.predict(test[columns])))\n"
   "print('negative :', (pos.named_steps['ridge'].coef_ < 0).sum())",
   f"{POS_TE:.4f} with {POS_NEG} negative coefficients. Use it when you know the "
   "sign, as with portfolio weights that cannot be short; here it costs almost "
   "nothing because ridge had already made the wrong signs tiny.")

ex("I4", "Why the intercept matters", 3,
   "Fit the scaling-plus-ridge pipeline twice, once as usual and once with "
   "`fit_intercept=False`, and print the two test RMSEs. Explain the difference "
   "in one sentence.",
   "usual = ...\n...\nprint('with intercept   :', ...)\n\nnoint = ...\n...\nprint('without intercept:', ...)",
   "`Ridge(alpha=1000, fit_intercept=False)`. The columns are standardised to "
   "mean zero, but the target is not.",
   "usual = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1000))])\n"
   "usual.fit(train[columns], train['vol_next'])\n"
   "print('with intercept   :', rmse(test['vol_next'], usual.predict(test[columns])))\n\n"
   "noint = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1000, fit_intercept=False))])\n"
   "noint.fit(train[columns], train['vol_next'])\nprint('without intercept:', rmse(test['vol_next'], noint.predict(test[columns])))",
   f"{PIPE_TE:.4f} against {NOINT_TE:.4f}. With the columns centred at zero and no "
   "intercept, the forecast for an average day is zero volatility, which is "
   "nonsense. The intercept carries the level of the target, and that is why the "
   "penalty leaves it alone.")

# ============================================================ J
section(
"## \U0001f4d6 J · Reading the result\n\n"
"What the coefficients mean now, and what the final numbers are. Each of these "
"fits what it needs; none depends on an earlier section."
)

ex("J1", "Say what a coefficient means", 2,
   "Fit the scaling-plus-ridge pipeline (alpha 1000) and print one sentence with "
   "an f-string: the standardised coefficient on `vol_5d` to three decimals, and "
   "what a one-standard-deviation rise in `vol_5d` does to the forecast.",
   "ridge_pipe = ...\n...\ncoefs = ...\n\nsentence = ...\nprint(sentence)",
   "`coefs[columns.index('vol_5d')]` is the number. `f'...{value:.3f}...'` rounds "
   "inside the string.",
   "ridge_pipe = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1000))])\n"
   "ridge_pipe.fit(train[columns], train['vol_next'])\ncoefs = ridge_pipe.named_steps['ridge'].coef_\n\n"
   "b = coefs[columns.index('vol_5d')]\n"
   "sentence = (f'One standard deviation more volatility over the last five days '\n"
   "            f'raises the forecast by {b:.3f} percentage points, other columns held fixed.')\n"
   "print(sentence)",
   f"The coefficient is {COEF_5D:.3f}. It is a forecasting statement, not a causal "
   "one: the model was shrunk on purpose, so there is no standard error to put "
   "next to it.",
   revisits="S1")

ex("J2", "A small OLS on the columns the lasso kept", 3,
   "The lasso with alpha 0.1 keeps five columns: `vol_5d`, `vol_10d`, `ret_5d`, "
   "`AAPL_vol` and `XOM_vol`. Fit an ordinary least squares on just those five and "
   "print its test RMSE next to the lasso's.",
   "kept = ['vol_5d', 'vol_10d', 'ret_5d', 'AAPL_vol', 'XOM_vol']\n\n"
   "small = LinearRegression()\n...\nlasso_pipe = ...\n...\n\n"
   "print('OLS on five columns:', ...)\nprint('the lasso         :', ...)",
   "Fit `small` on `train[kept]` and score on `test[kept]`; fit the lasso pipeline "
   "on all `columns` as in section G.",
   "kept = ['vol_5d', 'vol_10d', 'ret_5d', 'AAPL_vol', 'XOM_vol']\n\n"
   "small = LinearRegression()\nsmall.fit(train[kept], train['vol_next'])\n"
   "lasso_pipe = Pipeline([('scale', StandardScaler()), ('lasso', Lasso(alpha=0.1))])\n"
   "lasso_pipe.fit(train[columns], train['vol_next'])\n\n"
   "print('OLS on five columns:', rmse(test['vol_next'], small.predict(test[kept])))\n"
   "print('the lasso         :', rmse(test['vol_next'], lasso_pipe.predict(test[columns])))",
   f"{SMALL_TE:.4f}, against {LASSO_TE:.4f} for the lasso itself: better, because "
   "once the columns are chosen the shrinkage only costs accuracy. A small "
   "unpenalised model on chosen columns is also where standard errors and t-tests "
   "live. What would be dishonest is reading them as if the five columns had been "
   "chosen in advance: the lasso picked them from these same training rows.",
   revisits="S5")

ex("J3", "Six forecasts, ranked", 3,
   "Put the test RMSE of all six forecasts from the lecture into one dictionary: "
   "the training average, persistence, one column, nineteen columns with OLS, "
   "with ridge (alpha 1000, scaled) and with lasso (alpha 0.1, scaled). Print "
   "them from best to worst.",
   "six = {}\n\n...\n\nfor name in sorted(six, key=six.get):\n    print(f'{name:26} {six[name]:.4f}')",
   ["Each entry is one model fitted and scored, as in the earlier sections. The "
    "average and persistence need no fitting.",
    "`sorted(six, key=six.get)` orders the keys by their values."],
   "six = {}\n\n"
   "six['guess the average'] = rmse(test['vol_next'], np.full(len(test), train['vol_next'].mean()))\n"
   "six['repeat this month'] = rmse(test['vol_next'], test['vol_20d'])\n\n"
   "one_col = LinearRegression()\none_col.fit(train[['vol_20d']], train['vol_next'])\n"
   "six['one column'] = rmse(test['vol_next'], one_col.predict(test[['vol_20d']]))\n\n"
   "all_ols = LinearRegression()\nall_ols.fit(train[columns], train['vol_next'])\n"
   "six['nineteen columns, OLS'] = rmse(test['vol_next'], all_ols.predict(test[columns]))\n\n"
   "ridge_pipe = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1000))])\n"
   "ridge_pipe.fit(train[columns], train['vol_next'])\n"
   "six['nineteen columns, ridge'] = rmse(test['vol_next'], ridge_pipe.predict(test[columns]))\n\n"
   "lasso_pipe = Pipeline([('scale', StandardScaler()), ('lasso', Lasso(alpha=0.1))])\n"
   "lasso_pipe.fit(train[columns], train['vol_next'])\n"
   "six['nineteen columns, lasso'] = rmse(test['vol_next'], lasso_pipe.predict(test[columns]))\n\n"
   "for name in sorted(six, key=six.get):\n    print(f'{name:26} {six[name]:.4f}')",
   f"Best is \"{SIX_ORDER[0]}\" at {SIX[SIX_ORDER[0]]:.4f} and worst is "
   f"\"{SIX_ORDER[-1]}\" at {SIX[SIX_ORDER[-1]]:.4f}. The two penalised models are the only ones to beat "
   "the single column, and nineteen unpenalised columns sit below it.",
   revisits="S2")

ex("J4", "One function for any pipeline", 4,
   "Write `evaluate(p)`: it cross-validates the pipeline `p` on the training rows "
   "with `folds`, refits it on all of them, and returns the pair "
   "`(cv_rmse, test_rmse)`. Run it on a ridge pipeline (alpha 1000) and a lasso "
   "pipeline (alpha 0.1).",
   "def evaluate(p):\n    ...\n\nridge_pipe = ...\nlasso_pipe = ...\n\nprint('ridge:', ...)\nprint('lasso:', ...)",
   ["Inside: `cross_val_score(...)` for the first number, then `p.fit(...)` "
    "and `rmse(...)` on the test rows for the second.",
    "`return round(float(-scores.mean()), 4), round(test_rmse, 4)` returns a pair."],
   "def evaluate(p):\n"
   "    scores = cross_val_score(p, train[columns], train['vol_next'],\n"
   "                             cv=folds, scoring='neg_root_mean_squared_error')\n"
   "    p.fit(train[columns], train['vol_next'])\n"
   "    test_rmse = rmse(test['vol_next'], p.predict(test[columns]))\n"
   "    return round(float(-scores.mean()), 4), round(test_rmse, 4)\n\n"
   "ridge_pipe = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1000))])\n"
   "lasso_pipe = Pipeline([('scale', StandardScaler()), ('lasso', Lasso(alpha=0.1))])\n\n"
   "print('ridge:', evaluate(ridge_pipe))\nprint('lasso:', evaluate(lasso_pipe))",
   f"Ridge ({PIPE_CV:.4f}, {PIPE_TE:.4f}) and lasso ({GL_SCORE:.4f}, "
   f"{LASSO_TE:.4f}). One function, any pipeline: because every scikit-learn "
   "model fits and predicts the same way, the function never needs to know what "
   "is inside.",
   revisits="S2")

ex("J5", "The worst day", 4,
   "Fit the ridge pipeline (alpha 1000) and find the test day on which its forecast "
   "was furthest from what happened. Print the date, what happened, and the "
   "forecast.",
   "ridge_pipe = ...\n...\nforecast = ...\n\nworst = ...\nprint(..., ..., ...)",
   ["`errors = np.abs(test['vol_next'].values - forecast)` is the size of every "
    "miss; `errors.argmax()` is the position of the largest.",
    "`test.index[worst].date()`, `test['vol_next'].iloc[worst]` and "
    "`forecast[worst]` read the three things off at that position."],
   "ridge_pipe = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1000))])\n"
   "ridge_pipe.fit(train[columns], train['vol_next'])\nforecast = ridge_pipe.predict(test[columns])\n\n"
   "errors = np.abs(test['vol_next'].values - forecast)\nworst = errors.argmax()\n"
   "print(test.index[worst].date(), round(test['vol_next'].iloc[worst], 3), round(forecast[worst], 3))",
   f"{WORST_DAY}: the next twenty days turned out at {WORST_ACTUAL:.2f} and the "
   f"forecast said {WORST_PRED:.2f}. Look up what happened in the market in the "
   "weeks after that date; a volatility model made from past returns cannot see "
   "an event coming, and the biggest misses are always the days before one.",
   revisits="S5")

# ============================================================ K
section(
"## \U0001f501 K · Across the desk\n\n"
"The whole workflow, once per instrument. This section uses `build_table` from "
"A7; if you skipped A7, copy its solution into the first cell. Each exercise "
"builds the tables it needs."
)

ex("K1", "Nvidia, with everything", 4,
   "Use your `build_table` from A7 on `'NVDA'`, the most volatile stock in the "
   "data. Split at the end of 2022, then print three test RMSEs: one column "
   "(`vol_20d`), nineteen columns with OLS, and nineteen columns with a ridge "
   "whose alpha `GridSearchCV` chooses from `[1, 10, 100, 1000, 10000]`.",
   "nvda = build_table('NVDA')\ntr = ...\nte = ...\ncols = ...\n\n"
   "...\n\nprint('one column        :', ...)\nprint('all columns, OLS  :', ...)\nprint('all columns, ridge:', ...)",
   ["`cols = list(nvda.columns[:-1])`, then the three fits from earlier sections "
    "with `tr` and `te` in place of `train` and `test`.",
    "The search is F3 with the pipeline and grid unchanged."],
   "nvda = build_table('NVDA')\ntr = nvda.loc[:'2022-12-31']\nte = nvda.loc['2023-01-01':]\n"
   "cols = list(nvda.columns[:-1])\n\n"
   "one_n = LinearRegression()\none_n.fit(tr[['vol_20d']], tr['vol_next'])\n"
   "ols_n = LinearRegression()\nols_n.fit(tr[cols], tr['vol_next'])\n"
   "search_n = GridSearchCV(Pipeline([('scale', StandardScaler()), ('ridge', Ridge())]),\n"
   "                        {'ridge__alpha': [1, 10, 100, 1000, 10000]},\n"
   "                        cv=folds, scoring='neg_root_mean_squared_error')\n"
   "search_n.fit(tr[cols], tr['vol_next'])\n\n"
   "print('one column        :', rmse(te['vol_next'], one_n.predict(te[['vol_20d']])))\n"
   "print('all columns, OLS  :', rmse(te['vol_next'], ols_n.predict(te[cols])))\n"
   "print('all columns, ridge:', rmse(te['vol_next'], search_n.predict(te[cols])))",
   f"{NVDA_ONE:.3f}, {NVDA_OLS:.3f} and {NVDA_RIDGE:.3f}, with alpha {NVDA_ALPHA:,} "
   "chosen. The numbers are five times the index's because Nvidia moves five "
   "times as much, and the order is the same: the penalised nineteen beat the "
   "single column, and the unpenalised nineteen do not.",
   revisits="S2")

ex("K2", "Which alpha does each instrument pick", 5,
   "For every ticker in `rets`, build its table with `build_table`, run the same "
   "grid search on the training rows, and store the winning alpha in a dictionary "
   "`best_alpha`. Print it, and count how many instruments pick 1000.",
   "best_alpha = {}\n\nfor ticker in rets.columns:\n    ...\n\n"
   "print(best_alpha)\nprint('pick 1000:', ...)",
   ["The loop body is K1's search with `ticker` in place of `'NVDA'`, ending in "
    "`best_alpha[ticker] = search.best_params_['ridge__alpha']`.",
    "`sum(1 for t in best_alpha if best_alpha[t] == 1000)` counts. Eleven "
    "searches take a little while."],
   "best_alpha = {}\n\nfor ticker in rets.columns:\n"
   "    frame = build_table(ticker)\n    tr = frame.loc[:'2022-12-31']\n"
   "    cols = list(frame.columns[:-1])\n"
   "    search_t = GridSearchCV(Pipeline([('scale', StandardScaler()), ('ridge', Ridge())]),\n"
   "                            {'ridge__alpha': [1, 10, 100, 1000, 10000]},\n"
   "                            cv=folds, scoring='neg_root_mean_squared_error')\n"
   "    search_t.fit(tr[cols], tr['vol_next'])\n"
   "    best_alpha[ticker] = search_t.best_params_['ridge__alpha']\n\n"
   "print(best_alpha)\nprint('pick 1000:', sum(1 for t in best_alpha if best_alpha[t] == 1000))",
   f"{N_PICK_1000} of {len(BEST_ALPHA)} pick 1000" +
   (f"; {', '.join(f'{t} picks {a:,}' for t, a in ALPHA_OTHERS.items())}." if ALPHA_OTHERS else ".") +
   " One grid, one factor of ten apart, and nearly the same answer on every "
   "instrument: with standardised columns the right strength of penalty is a "
   "property of the problem, not of the stock. The test rows were never touched.",
   revisits="S2")

ex("K3", "How many columns the lasso keeps, per instrument", 3,
   "For every ticker, fit the scaling-plus-`Lasso(alpha=0.1)` pipeline on its "
   "training rows and count the non-zero coefficients. Draw the counts as a bar "
   "chart, sorted from most to fewest.",
   "survivors = {}\nfor ticker in rets.columns:\n    ...\n\n"
   "ranked = ...\n\nfig, ax = plt.subplots(figsize=(8, 3.2))\n...\nplt.show()",
   ["`survivors[ticker] = int((lasso_t.named_steps['lasso'].coef_ != 0).sum())`.",
    "`ranked = pd.Series(survivors).sort_values(ascending=False)`, then "
    "`ax.bar(ranked.index, ranked.values)`."],
   "survivors = {}\nfor ticker in rets.columns:\n"
   "    frame = build_table(ticker)\n    tr = frame.loc[:'2022-12-31']\n"
   "    cols = list(frame.columns[:-1])\n"
   "    lasso_t = Pipeline([('scale', StandardScaler()), ('lasso', Lasso(alpha=0.1))])\n"
   "    lasso_t.fit(tr[cols], tr['vol_next'])\n"
   "    survivors[ticker] = int((lasso_t.named_steps['lasso'].coef_ != 0).sum())\n\n"
   "ranked = pd.Series(survivors).sort_values(ascending=False)\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3.2))\nax.bar(ranked.index, ranked.values)\n"
   "ax.set_ylabel('columns kept of 19')\n"
   "ax.set_title('Lasso, alpha 0.1: how many columns survive, per instrument', loc='left')\n"
   "plt.show()",
   f"From {SURVIVORS_BY[MOST_KEPT]} on {MOST_KEPT} down to {SURVIVORS_BY[FEWEST_KEPT]} "
   f"on {FEWEST_KEPT}. The same alpha keeps a different number of columns on each "
   "instrument, because the lasso's alpha is a price in the units of the target, "
   "and the targets differ in size. A lasso grid, unlike ridge's, has to be "
   "searched per problem.",
   revisits="S3")

# ---------------------------------------------------------------- closing
md(
"## \U0001f3c1 Done\n\n"
"You built the lecture's table from the price file, watched nineteen "
"unpenalised columns forecast worse than one, and fixed it: a penalty, the "
"scaling it needs, a pipeline so the folds refit both, and a grid search to set "
"the penalty's strength. Then you did it eleven times.\n\n"
"The case takes the same tools back to the risk report, where the data is in "
"decimals and the results are not the same on every stock."
)

# ---------------------------------------------------------------- write
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
for _i, _c in enumerate(nb.cells):
    _c["id"] = f"c{_i:04d}"
OUT.write_text(nbf.writes(nb), encoding="utf-8")

n_ex = sum(1 for c in cells if c.cell_type == "markdown" and c.source.startswith("### ") and "★" in c.source)
print("wrote", OUT, " (", len(cells), "cells,", n_ex, "exercises )")
