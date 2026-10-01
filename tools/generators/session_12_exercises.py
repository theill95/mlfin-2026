# -*- coding: utf-8 -*-
"""Build session_12_exercises.ipynb.

Same conventions as Sessions 1 to 11: pleasant intro, 1-5 star badges, toolkit
card with title= hover docs, task -> work cell (blank-safe `...`) -> 1-2 folded
hints -> folded solution, no em-dashes, plain explanatory tone.

Session 12 is boosting, XGBoost and LightGBM, and reading a model. The
exercises use three tables from earlier sessions: the index table from Session 6
(returns in percent), where the target is the volatility over the next 20 days;
the credit table from Session 9, where the label is a default; and the daily
prices of the eleven instruments, for a target with nothing in it (section J).
They boost the lecture's six days by hand with NumPy, lists and loops, check
numerically that the trees are fitted to the negative gradient, run the
learning rate on a single number, watch the training error fall at every
stump, choose the number of trees on 2022 and on the folds, rebuild XGBoost's
early stopping and its leaf values by hand, time three libraries, boost the
credit label, measure importance four ways, compare every classifier of the
course, and fix five mistakes that boosting makes easy.

Sections A to H work through the session. Section I is five mistakes. Section
J is a target with nothing in it. Section K is five standalone small cases that
reach back to functions, loops, dictionaries and NumPy, in this session's
setting.

Only tools taught by the end of Session 12. New this session:
GradientBoostingRegressor, GradientBoostingClassifier, staged_predict,
n_iter_no_change, validation_fraction, n_estimators_, HistGradientBoosting*,
XGBRegressor, XGBClassifier, eval_set, early_stopping_rounds, best_iteration,
best_score, reg_lambda, gamma, subsample, colsample_bytree, LGBMRegressor,
LGBMClassifier, num_leaves, verbose=-1, early_stopping, best_iteration_,
feature_importances_, importance_type, permutation_importance. Named in the
task where used, because the lecture did not show them: evals_result(),
output_margin=True, base_score, min_child_weight, max_bin is not used, ** to
pass a dictionary of settings.

The index table is in PERCENT here, exactly as in the lecture. The credit table
is in New Taiwan dollars, as it comes.

BLANK-SAFE RULES:
- every blank is the right-hand side of an assignment, a bare `...` statement,
  or an argument to print(). Never call a method on, index into, or do
  arithmetic with a placeholder.
- nothing depends on an earlier exercise having been solved: the setup cell and
  each work cell provide every variable the task uses, except inside the short
  clusters the intro lists.
"""
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import nbformat as nbf
from _shared import XGB_GUARD
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "session_12" / "session_12_exercises.ipynb"

cells = []


def badge(n, revisits=None):
    """Five unnamed stars, plus an optional note that this one reaches back."""
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
    "A1": "S3", "A2": "S3", "A3": "S2", "A4": "S2", "A5": "S3", "A6": "S2", "A7": "S3",
    "B1": "S2", "B2": "S8", "B3": "S2",
    "C2": "S3", "C3": "S3", "C4": "S2",
    "D2": "S2", "D3": "S2", "D4": "S5", "D5": "S3", "D6": "S2",
    "E2": "S3", "E3": "S11", "E4": "S2",
    "F2": "S8", "F3": "S9", "F4": "S8", "F5": "S9",
    "G2": "S3", "G3": "S2", "G4": "S3", "G5": "S3", "G6": "S3",
    "H1": "S2", "H2": "S3", "H3": "S5", "H4": "S3", "H5": "S6", "H6": "S8",
    "I4": "S8", "I5": "S2",
    "J1": "S4", "J2": "S8", "J5": "S2", "J6": "S3",
    "K1": "S2", "K2": "S2", "K3": "S2", "K4": "S2", "K5": "S2",
}


def ex(sid, title, n, task, work, hints, sol_code, sol_note, revisits=None, raises=False):
    md(f"### {sid} · {title}  {badge(n, revisits or REVISITS.get(sid))}\n\n{task}")
    code(work, raises=raises)
    _hints_solution(hints, sol_code, sol_note)


def ex_fix(sid, title, n, task, demo, work, hints, sol_code, sol_note, raises=True, revisits=None):
    """The two-cell pattern: a cell that shows the mistake, then a blank cell for the fix."""
    md(f"### {sid} · {title}  {badge(n, revisits or REVISITS.get(sid))}\n\n{task}")
    code(demo, raises=raises)
    code(work)
    _hints_solution(hints, sol_code, sol_note)


def section(header):
    md(header)


def _words(x):
    """A factor rounded to a whole number, in words when it is small."""
    n = int(round(x))
    return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][n] if n <= 10 else str(n)


def rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ======================================================================
# The real numbers, computed here so every solution note is exact.
# ======================================================================
TABLE = pd.read_csv(ROOT / "data" / "market_features.csv", parse_dates=["date"]).set_index("date")
COLUMNS = list(TABLE.columns[:-1])
PAIR = ["vol_20d", "ret_20d"]
TRAIN, TEST = TABLE.loc[:"2022-12-31"], TABLE.loc["2023-01-01":]
FIT, STOP = TRAIN.loc[:"2021-12-31"], TRAIN.loc["2022-01-01":]
Y, YT = TRAIN["vol_next"], TEST["vol_next"]
TS5 = TimeSeriesSplit(n_splits=5)
N_TRAIN, N_TEST = len(TRAIN), len(TEST)

CREDIT = pd.read_csv(ROOT / "data" / "credit.csv")
C_COLS = ["limit", "age", "late_now", "months_late", "bill", "paid", "utilisation"]
C_TRAIN, C_TEST = train_test_split(CREDIT, test_size=0.3, random_state=0, stratify=CREDIT["default"])
CY, CYT = C_TRAIN["default"], C_TEST["default"]

PRICES = pd.read_csv(ROOT / "data" / "prices.csv", parse_dates=["date"])
RETS = PRICES.pivot(index="date", columns="ticker", values="close").pct_change().dropna() * 100

# ---- A: the six days ----
X6 = np.array([0.4, 0.6, 0.8, 1.2, 1.6, 2.4])
Y6 = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])
SIX = pd.DataFrame({"vol_20d": X6})
A1_RES = Y6 - Y6.mean()
A1_RSS = round(float((A1_RES ** 2).sum()), 3)
assert round(float(Y6.mean()), 3) == 0.95 and A1_RSS == 0.815 and abs(A1_RES.sum()) < 1e-12
_left6 = X6 <= 0.8
A2_L, A2_R = float(A1_RES[_left6].mean()), float(A1_RES[~_left6].mean())
A2_F = Y6.mean() + 0.5 * np.where(_left6, A2_L, A2_R)
A2_RSS = round(float(((Y6 - A2_F) ** 2).sum()), 3)
assert (round(A2_L, 2), round(A2_R, 2), A2_RSS) == (-0.35, 0.35, 0.264)
assert np.allclose(np.unique(A2_F), [0.775, 1.125])


def _boost_step(x, y, forecast, nu):
    stump = DecisionTreeRegressor(max_depth=1, random_state=0).fit(x, y - forecast)
    return forecast + nu * stump.predict(x), stump


_f = np.full(6, Y6.mean())
A4_PATH = [A1_RSS]
_stumps6 = []
for _m in range(50):
    _f, _s = _boost_step(SIX, Y6, _f, 0.5)
    _stumps6.append(_s)
    if _m < 6:
        A4_PATH.append(round(float(((Y6 - _f) ** 2).sum()), 3))
assert A4_PATH == [0.815, 0.264, 0.126, 0.083, 0.059, 0.049, 0.038]
assert (A4_PATH[0] - A4_PATH[1]) / A4_PATH[0] > 2 / 3                        # "more than two thirds"
A6_RSS = float(((Y6 - _f) ** 2).sum())
A6_GAP = float(np.abs(Y6 - _f).max())
assert round(A6_RSS, 4) == 0.0001 and A6_GAP < 0.006
_new = pd.DataFrame({"vol_20d": [1.0, 1.4]})
_tot = np.zeros(2)
for _s in _stumps6:
    _tot = _tot + _s.predict(_new)
A6_NEW = Y6.mean() + 0.5 * _tot
assert np.allclose(GradientBoostingRegressor(n_estimators=50, learning_rate=0.5, max_depth=1, random_state=0)
                   .fit(SIX, Y6).predict(_new), A6_NEW)
assert {round(float(s.tree_.threshold[0]), 2) for s in _stumps6} <= {0.5, 0.7, 1.0, 1.4, 2.0}
A7_CUTS = (X6[:-1] + X6[1:]) / 2
A7_STEPS = np.diff(Y6)
_exact = Y6[0] + sum(st * (X6 > c) for st, c in zip(A7_STEPS, A7_CUTS))
assert np.allclose(_exact, Y6) and np.allclose(A7_STEPS, [0.1, 0.1, 0.5, 0.3, -0.3])

# ---- B ----
_h = 0.000001
_slope = (0.5 * (Y6 - (Y6.mean() + _h)) ** 2 - 0.5 * (Y6 - (Y6.mean() - _h)) ** 2) / (2 * _h)
assert np.allclose(-_slope, A1_RES)
YB = np.array([1, 0, 0, 1])
FB = np.array([0.5, -1.0, 0.2, 2.0])


def _logloss(y, f):
    return -(y * np.log(sigmoid(f)) + (1 - y) * np.log(1 - sigmoid(f)))


_sl = (_logloss(YB, FB + _h) - _logloss(YB, FB - _h)) / (2 * _h)
B2_GAP = YB - sigmoid(FB)
assert np.allclose(-_sl, B2_GAP)
assert np.round(B2_GAP, 3).tolist() == [0.378, -0.269, -0.55, 0.119]
B3 = {}
for _nu in [0.5, 0.1]:
    _fv, _steps = 0.0, 0
    while abs(Y6.mean() - _fv) >= 0.001 and _steps < 1000:
        _steps += 1
        _fv = _fv + _nu * (Y6 - _fv).mean()
    B3[_nu] = (_steps, float(np.log(0.001 / Y6.mean()) / np.log(1 - _nu)))
assert B3[0.5][0] == 10 and B3[0.1][0] == 66

# ---- C ----
_c1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0).fit(TRAIN[PAIR], Y)
C1_TR, C1_TE = rmse(Y, _c1.predict(TRAIN[PAIR])), rmse(YT, _c1.predict(TEST[PAIR]))
_lect = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0).fit(TRAIN[COLUMNS], Y)
LECT_TE = rmse(YT, _lect.predict(TEST[COLUMNS]))
assert round(LECT_TE, 3) == 0.222 and round(C1_TE, 3) == 0.244
_c2 = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=1, random_state=0).fit(TRAIN[COLUMNS], Y)
C2_PATH = [rmse(Y, _p) for _p in _c2.staged_predict(TRAIN[COLUMNS])]
assert bool((np.diff(C2_PATH) < 0).all())


def _best_stage(model, rows):
    scores = [rmse(rows["vol_next"], _p) for _p in model.staged_predict(rows[COLUMNS])]
    return int(np.argmin(scores)) + 1, float(min(scores))


C4 = {}
for _nu in [0.3, 0.1, 0.03]:
    _g = GradientBoostingRegressor(n_estimators=1000, learning_rate=_nu, max_depth=1, random_state=0).fit(FIT[COLUMNS], FIT["vol_next"])
    C4[_nu] = _best_stage(_g, STOP)
assert {k: v[0] for k, v in C4.items()} == {0.3: 47, 0.1: 219, 0.03: 785}
C5 = {1: C4[0.1]}
for _d in [2, 3]:
    _g = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=_d, random_state=0).fit(FIT[COLUMNS], FIT["vol_next"])
    C5[_d] = _best_stage(_g, STOP)
assert {k: v[0] for k, v in C5.items()} == {1: 219, 2: 69, 3: 41}
assert C5[1][1] < C5[2][1] < C5[3][1]

# ---- D ----
_x600 = XGBRegressor(n_estimators=600, learning_rate=0.1, max_depth=1, random_state=0)
_x600.fit(FIT[COLUMNS], FIT["vol_next"], eval_set=[(STOP[COLUMNS], STOP["vol_next"])], verbose=False)
D1_SCORES = _x600.evals_result()["validation_0"]["rmse"]
D1_BEST = int(np.argmin(D1_SCORES))


def _stop_round(scores, patience=50):
    best_score, best_round = scores[0], 0
    for i, s in enumerate(scores):
        if s < best_score:
            best_score, best_round = s, i
        if i - best_round >= patience:
            return best_round, i
    return best_round, len(scores) - 1


D2 = _stop_round(D1_SCORES, 50)
_xs = XGBRegressor(n_estimators=3000, learning_rate=0.1, max_depth=1, early_stopping_rounds=50, random_state=0)
_xs.fit(FIT[COLUMNS], FIT["vol_next"], eval_set=[(STOP[COLUMNS], STOP["vol_next"])], verbose=False)
XS_BEST = int(_xs.best_iteration)
assert D1_BEST == 243 and D2 == (243, 293) and XS_BEST == 243
D3 = {_p: _stop_round(D1_SCORES, _p) for _p in [5, 20, 50, 200]}
assert D3[5][0] == D3[20][0] == 50 and D3[50][0] == D3[200][0] == 243
D4 = {}
for _n in [10, 25, 50, 100, 200, 400]:
    D4[_n] = float(-cross_val_score(GradientBoostingRegressor(n_estimators=_n, learning_rate=0.1, max_depth=1, random_state=0),
                                    TRAIN[COLUMNS], Y, cv=TS5, scoring="neg_root_mean_squared_error").mean())
D4_BEST = min(D4, key=D4.get)
_ridge = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1000))])
D4_RIDGE = float(-cross_val_score(_ridge, TRAIN[COLUMNS], Y, cv=TS5, scoring="neg_root_mean_squared_error").mean())
assert D4_BEST == 25 and round(D4_RIDGE, 3) == 0.560 and min(D4.values()) > D4_RIDGE
assert all(np.diff([D4[k] for k in [25, 50, 100, 200, 400]]) > 0)               # "rises steadily after that"
D6 = []
for _seed in range(5):
    _e = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.1, max_depth=1, validation_fraction=0.1,
                                   n_iter_no_change=10, random_state=_seed).fit(TRAIN[COLUMNS], Y)
    D6.append(int(_e.n_estimators_))
assert max(D6) / min(D6) > 2.5

# ---- E ----
_e1 = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0).fit(TRAIN[PAIR], Y)
E1_TE = rmse(YT, _e1.predict(TEST[PAIR]))
assert round(E1_TE, 3) == 0.242
E2 = {}
for _lam in [0, 1, 10]:
    _one = XGBRegressor(n_estimators=1, max_depth=1, learning_rate=1, base_score=float(Y6.mean()), reg_lambda=_lam,
                        min_child_weight=0, random_state=0).fit(SIX, Y6)
    _wl = A1_RES[_left6].sum() / (_left6.sum() + _lam)
    _wr = A1_RES[~_left6].sum() / ((~_left6).sum() + _lam)
    assert np.allclose(_one.predict(SIX), Y6.mean() + np.where(_left6, _wl, _wr), atol=0.00001)
    E2[_lam] = (float(_wl), float(_wr))
assert [round(E2[k][1], 4) for k in E2] == [0.35, 0.2625, 0.0808]
E3 = {}
for _name, _m in [("scikit-learn", GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=1, random_state=0)),
                  ("XGBoost", XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=1, random_state=0, n_jobs=1)),
                  ("LightGBM", LGBMRegressor(n_estimators=300, learning_rate=0.1, num_leaves=2, random_state=0, n_jobs=1, verbose=-1))]:
    _t0 = time.perf_counter()
    _m.fit(TRAIN[COLUMNS], Y)
    E3[_name] = (time.perf_counter() - _t0, rmse(YT, _m.predict(TEST[COLUMNS])))
E4 = {}
for _nl in [2, 4, 8, 16]:
    _l = LGBMRegressor(n_estimators=3000, learning_rate=0.1, num_leaves=_nl, random_state=0, verbose=-1)
    _l.fit(FIT[COLUMNS], FIT["vol_next"], eval_set=[(STOP[COLUMNS], STOP["vol_next"])],
           callbacks=[early_stopping(50, verbose=False)])
    E4[_nl] = (int(_l.best_iteration_), float(_l.best_score_["valid_0"]["l2"] ** 0.5))
assert [E4[k][0] for k in E4] == [48, 15, 13, 13] and E4[2][1] < E4[4][1] < E4[8][1] < E4[16][1]
E5 = {}
for _ss, _cs in [(1.0, 1.0), (0.5, 1.0), (1.0, 0.5), (0.5, 0.5)]:
    _x = XGBRegressor(n_estimators=3000, learning_rate=0.1, max_depth=1, subsample=_ss, colsample_bytree=_cs,
                      early_stopping_rounds=50, random_state=0)
    _x.fit(FIT[COLUMNS], FIT["vol_next"], eval_set=[(STOP[COLUMNS], STOP["vol_next"])], verbose=False)
    E5[(_ss, _cs)] = (int(_x.best_iteration) + 1, float(_x.best_score))
assert min(E5, key=lambda k: E5[k][1]) == (1.0, 1.0)

# ---- F ----
_f1 = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=0).fit(C_TRAIN[C_COLS], CY)
F1_AUC = float(roc_auc_score(CYT, _f1.predict_proba(C_TEST[C_COLS])[:, 1]))
XGB3 = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(C_TRAIN[C_COLS], CY)
P_XGB = XGB3.predict_proba(C_TEST[C_COLS])[:, 1]
XGB3_AUC = float(roc_auc_score(CYT, P_XGB))
assert round(XGB3_AUC, 3) == 0.777 and round(F1_AUC, 3) == 0.775
_margin = XGB3.predict(C_TEST[C_COLS], output_margin=True)
assert np.allclose(sigmoid(_margin), P_XGB, atol=0.000001)
F2_MIN, F2_MAX = float(_margin.min()), float(_margin.max())
_logit = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression())]).fit(C_TRAIN[C_COLS], CY)
P_LOGIT = _logit.predict_proba(C_TEST[C_COLS])[:, 1]
LOGIT_AUC = float(roc_auc_score(CYT, P_LOGIT))
assert round(LOGIT_AUC, 3) == 0.748
F3 = {}
for _name, _p in [("XGBoost", P_XGB), ("logistic regression", P_LOGIT)]:
    _top = pd.Series(_p, index=C_TEST.index).nlargest(500).index
    F3[_name] = float(CYT.loc[_top].mean())
F3_BASE = float(CYT.mean())
assert F3["XGBoost"] > F3["logistic regression"] > 3 * F3_BASE
FIT_C, STOP_C = train_test_split(C_TRAIN, test_size=0.2, random_state=0, stratify=C_TRAIN["default"])
F5 = {}
for _metric in ["logloss", "auc"]:
    _x = XGBClassifier(n_estimators=3000, learning_rate=0.1, max_depth=3, early_stopping_rounds=50, eval_metric=_metric,
                       random_state=0)
    _x.fit(FIT_C[C_COLS], FIT_C["default"], eval_set=[(STOP_C[C_COLS], STOP_C["default"])], verbose=False)
    F5[_metric] = (int(_x.best_iteration) + 1, float(roc_auc_score(CYT, _x.predict_proba(C_TEST[C_COLS])[:, 1])))
assert F5["logloss"][0] == 107 and F5["auc"][0] == 91 and abs(F5["logloss"][1] - F5["auc"][1]) < 0.001

# ---- G ----
G1_GAIN = pd.Series(XGB3.feature_importances_, index=C_COLS).sort_values(ascending=False)
_w = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0, importance_type="weight").fit(C_TRAIN[C_COLS], CY)
G1_WEIGHT = pd.Series(_w.feature_importances_, index=C_COLS).sort_values(ascending=False)
assert G1_GAIN.index[0] == "late_now" and G1_WEIGHT.index[0] == "bill" and G1_WEIGHT.index[-1] == "late_now"
G1_NUNIQUE = C_TRAIN[C_COLS].nunique()
_noisy = C_TRAIN.copy()
_rng = np.random.default_rng(0)
_noisy["noise_num"] = _rng.normal(size=len(_noisy))
_noisy["noise_coin"] = _rng.integers(0, 2, size=len(_noisy))
_nc = C_COLS + ["noise_num", "noise_coin"]
G2 = pd.Series(LGBMClassifier(random_state=0, verbose=-1).fit(_noisy[_nc], _noisy["default"]).feature_importances_,
               index=_nc).sort_values(ascending=False)
assert G2.index[0] == "noise_num" and G2.index[-1] == "noise_coin"
_drops = []
for _k in range(10):
    _sh = C_TEST.copy()
    _sh["months_late"] = np.random.default_rng(_k).permutation(_sh["months_late"].values)
    _drops.append(XGB3_AUC - roc_auc_score(CYT, XGB3.predict_proba(_sh[C_COLS])[:, 1]))
G3_HAND, G3_SD = float(np.mean(_drops)), float(np.std(_drops))
_pi = permutation_importance(XGB3, C_TEST[C_COLS], CYT, scoring="roc_auc", n_repeats=10, random_state=0)
PERM = pd.Series(_pi.importances_mean, index=C_COLS).sort_values(ascending=False)
assert abs(G3_HAND - PERM["months_late"]) < 2 * G3_SD
G4 = {}
for _c in C_COLS:
    _rest = [k for k in C_COLS if k != _c]
    _m = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(C_TRAIN[_rest], CY)
    G4[_c] = XGB3_AUC - float(roc_auc_score(CYT, _m.predict_proba(C_TEST[_rest])[:, 1]))
G4 = pd.Series(G4).sort_values(ascending=False)
assert list(G4.index[:2]) == ["months_late", "late_now"] and list(PERM.index[:2]) == ["months_late", "late_now"]
assert G4.index[-1] == "age" and PERM.index[-1] == "age" and G4.iloc[2] < 0.005
G6 = {}
for _name, _group in [("payment history", ["late_now", "months_late"]),
                      ("amounts", ["limit", "bill", "paid", "utilisation"]), ("age", ["age"])]:
    _ds = []
    for _k in range(10):
        _sh = C_TEST.copy()
        _order = np.random.default_rng(_k).permutation(len(_sh))
        _sh[_group] = C_TEST[_group].values[_order]
        _ds.append(XGB3_AUC - roc_auc_score(CYT, XGB3.predict_proba(_sh[C_COLS])[:, 1]))
    G6[_name] = float(np.mean(_ds))
assert G6["payment history"] > PERM["months_late"] + PERM["late_now"]

# ---- H ----
H1_MODELS = {
    "logistic regression": Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression())]),
    "k-nearest neighbours": Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=25))]),
    "tree": DecisionTreeClassifier(max_depth=5, random_state=0),
    "random forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=100, random_state=0),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=0),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0, n_jobs=1),
    "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=8, random_state=0, n_jobs=1, verbose=-1),
}
H1 = {}
for _name, _m in H1_MODELS.items():
    _t0 = time.perf_counter()
    _m.fit(C_TRAIN[C_COLS], CY)
    H1[_name] = (time.perf_counter() - _t0, float(roc_auc_score(CYT, _m.predict_proba(C_TEST[C_COLS])[:, 1])))
assert round(H1["k-nearest neighbours"][1], 3) == 0.749 and round(H1["random forest"][1], 3) == 0.777
_boost_aucs = [H1[k][1] for k in ["GradientBoosting", "HistGradientBoosting", "XGBoost", "LightGBM"]]
assert 0.775 <= min(_boost_aucs) and max(_boost_aucs) < 0.778
_binned = max(H1[k][0] for k in ["HistGradientBoosting", "XGBoost", "LightGBM"])
_slow = min(H1[k][0] for k in ["random forest", "GradientBoosting"])
FIT_C3, VAL_C3 = FIT_C, STOP_C              # the same stratified fifth as F5
H3 = {}
for _n in [250, 500, 1000, 2000, 4000, 8000, 16800]:
    _rows = FIT_C3.iloc[:_n]
    _lo = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression())]).fit(_rows[C_COLS], _rows["default"])
    _xg = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(_rows[C_COLS], _rows["default"])
    H3[_n] = (float(roc_auc_score(VAL_C3["default"], _lo.predict_proba(VAL_C3[C_COLS])[:, 1])),
              float(roc_auc_score(VAL_C3["default"], _xg.predict_proba(VAL_C3[C_COLS])[:, 1])))
assert len(FIT_C3) == 16800 and H3[2000][0] > H3[2000][1] and H3[4000][0] < H3[4000][1]
assert H3[250][0] > H3[250][1] and max(v[0] for v in H3.values()) < 0.762
H5_RIDGE = -cross_val_score(_ridge, TRAIN[COLUMNS], Y, cv=TS5, scoring="neg_root_mean_squared_error")
H5_BOOST = -cross_val_score(GradientBoostingRegressor(n_estimators=25, learning_rate=0.1, max_depth=1, random_state=0),
                            TRAIN[COLUMNS], Y, cv=TS5, scoring="neg_root_mean_squared_error")
H5_WINS = sum(1 for _r, _b in zip(H5_RIDGE, H5_BOOST) if _b < _r)
assert H5_WINS == 0
_med = TRAIN["vol_next"].median()
_busy_tr, _busy_te = (TRAIN["vol_next"] > _med).astype(int), (TEST["vol_next"] > _med).astype(int)
_reg = GradientBoostingRegressor(n_estimators=219, learning_rate=0.1, max_depth=1, random_state=0).fit(TRAIN[COLUMNS], Y)
_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0).fit(TRAIN[COLUMNS], _busy_tr)
H6 = {"forecast": float(roc_auc_score(_busy_te, _reg.predict(TEST[COLUMNS]))),
      "classifier": float(roc_auc_score(_busy_te, _clf.predict_proba(TEST[COLUMNS])[:, 1])),
      "vol_20d": float(roc_auc_score(_busy_te, TEST["vol_20d"]))}
H6_SHARE_TE = float(_busy_te.mean())
assert H6["forecast"] > H6["classifier"] > H6["vol_20d"]

# ---- I ----
_r1 = XGBRegressor(n_estimators=XS_BEST, learning_rate=0.1, max_depth=1, random_state=0).fit(FIT[COLUMNS], FIT["vol_next"])
_r2 = XGBRegressor(n_estimators=XS_BEST + 1, learning_rate=0.1, max_depth=1, random_state=0).fit(FIT[COLUMNS], FIT["vol_next"])
assert not np.allclose(_r1.predict(TEST[COLUMNS]), _xs.predict(TEST[COLUMNS]))
assert np.allclose(_r2.predict(TEST[COLUMNS]), _xs.predict(TEST[COLUMNS]))
I2_DIFF = float(np.abs(_r1.predict(TEST[COLUMNS]) - _xs.predict(TEST[COLUMNS])).max())
I2_SHORT, I2_RIGHT = rmse(YT, _r1.predict(TEST[COLUMNS])), rmse(YT, _xs.predict(TEST[COLUMNS]))
_hist = HistGradientBoostingRegressor(max_iter=200, max_depth=2, random_state=0).fit(TRAIN[COLUMNS], Y)
I3_TE = rmse(YT, _hist.predict(TEST[COLUMNS]))
I3_TOP = pd.Series(permutation_importance(_hist, TEST[COLUMNS], YT, scoring="neg_root_mean_squared_error", n_repeats=10,
                                          random_state=0).importances_mean, index=COLUMNS).nlargest(3)
_jump_tr = (TRAIN["vol_next"] > 1.5 * TRAIN["vol_20d"]).astype(int)
_jump_te = (TEST["vol_next"] > 1.5 * TEST["vol_20d"]).astype(int)
_xj = XGBClassifier(n_estimators=100, max_depth=1, random_state=0).fit(TRAIN[COLUMNS], _jump_tr)
I4_AUC = float(roc_auc_score(_jump_te, _xj.predict_proba(TEST[COLUMNS])[:, 1]))
I4_SHARE_TR, I4_N_TE = float(_jump_tr.mean()), int(_jump_te.sum())
I5_RMSE = rmse(YT, list(_lect.staged_predict(TEST[COLUMNS]))[49])

# ---- J ----
_spy = RETS["SPY"]
DIRECTION = pd.DataFrame()
for _k in range(5):
    DIRECTION["lag_" + str(_k)] = _spy.shift(_k)
DIRECTION["up_next"] = (_spy.shift(-1) > 0).astype(int)
DIRECTION = DIRECTION.iloc[5:-1]
LAGS = ["lag_" + str(_k) for _k in range(5)]
D_TRAIN, D_TEST = DIRECTION.loc[:"2022-12-31"], DIRECTION.loc["2023-01-01":]
D_FIT, D_STOP = D_TRAIN.loc[:"2021-12-31"], D_TRAIN.loc["2022-01-01":]
J1_SHARES = (float(D_TRAIN["up_next"].mean()), float(D_TEST["up_next"].mean()))
_jl = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression())]).fit(D_TRAIN[LAGS], D_TRAIN["up_next"])
J2 = (float(roc_auc_score(D_TRAIN["up_next"], _jl.predict_proba(D_TRAIN[LAGS])[:, 1])),
      float(roc_auc_score(D_TEST["up_next"], _jl.predict_proba(D_TEST[LAGS])[:, 1])))
_j3 = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, random_state=0).fit(D_TRAIN[LAGS], D_TRAIN["up_next"])
J3 = (float(roc_auc_score(D_TRAIN["up_next"], _j3.predict_proba(D_TRAIN[LAGS])[:, 1])),
      float(roc_auc_score(D_TEST["up_next"], _j3.predict_proba(D_TEST[LAGS])[:, 1])))
_j4 = XGBClassifier(n_estimators=1000, learning_rate=0.1, max_depth=4, early_stopping_rounds=50, eval_metric="auc",
                    random_state=0)
_j4.fit(D_FIT[LAGS], D_FIT["up_next"], eval_set=[(D_STOP[LAGS], D_STOP["up_next"])], verbose=False)
J4 = (int(_j4.best_iteration), float(_j4.best_score),
      float(roc_auc_score(D_TEST["up_next"], _j4.predict_proba(D_TEST[LAGS])[:, 1])))
_rng = np.random.default_rng(0)
J4_CHANCE_SD = float(np.std([roc_auc_score(D_TEST["up_next"], _rng.random(len(D_TEST))) for _ in range(1000)]))
assert J4[0] == 3 and (J4[2] - 0.5) < 2 * J4_CHANCE_SD
_shuf = D_TRAIN.copy()
_shuf["up_next"] = np.random.default_rng(0).permutation(_shuf["up_next"].values)
_j5 = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, random_state=0).fit(_shuf[LAGS], _shuf["up_next"])
J5 = (float(roc_auc_score(_shuf["up_next"], _j5.predict_proba(_shuf[LAGS])[:, 1])),
      float(roc_auc_score(D_TEST["up_next"], _j5.predict_proba(D_TEST[LAGS])[:, 1])))
_j6 = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, eval_metric="auc", random_state=0)
_j6.fit(D_FIT[LAGS], D_FIT["up_next"], eval_set=[(D_FIT[LAGS], D_FIT["up_next"]), (D_STOP[LAGS], D_STOP["up_next"])],
        verbose=False)
_ev = _j6.evals_result()
J6_FIT = _ev["validation_0"]["auc"]
J6_STOP = _ev["validation_1"]["auc"]
assert J6_FIT[-1] > 0.98 and J6_STOP[-1] < J6_STOP[0]

# ---- K ----
_f = np.full(6, Y6.mean())
K1_STUMPS = []
for _m in range(3):
    _st = DecisionTreeRegressor(max_depth=1, random_state=0).fit(SIX, Y6 - _f)
    _v = _st.tree_.value.ravel()
    K1_STUMPS.append((round(float(_st.tree_.threshold[0]), 2), round(float(_v[1]), 4), round(float(_v[2]), 4)))
    _f = _f + 0.5 * _st.predict(SIX)


def _k1_predict(x):
    total = 0.95
    for cut, left, right in K1_STUMPS:
        total = total + 0.5 * (left if x <= cut else right)
    return total


K1_OUT = [round(_k1_predict(x), 4) for x in X6]
assert np.allclose(K1_OUT, _f, atol=0.0001)
_vals = TRAIN["vol_20d"]
K2_EDGES = np.quantile(_vals, [0.25, 0.5, 0.75])


def _bin_of(value, edges):
    if value <= edges[0]:
        return 0
    elif value <= edges[1]:
        return 1
    elif value <= edges[2]:
        return 2
    return 3


_bins = [_bin_of(v, K2_EDGES) for v in _vals]
K2_COUNTS = pd.Series(_bins).value_counts().sort_index()
K2_MEANS = TRAIN["vol_next"].groupby(np.array(_bins)).mean()
assert K2_MEANS.is_monotonic_increasing and K2_COUNTS.max() - K2_COUNTS.min() <= 1
_g6 = Y6.mean() - Y6          # the gradient of half the squared error, f - y, at the mean
K3 = {}
for _lam in [1, 0]:
    K3[_lam] = {}
    for _c in [0.4, 0.6, 0.8, 1.2, 1.6]:
        _L = X6 <= _c
        _GL, _GR, _HL, _HR = _g6[_L].sum(), _g6[~_L].sum(), float(_L.sum()), float((~_L).sum())
        K3[_lam][_c] = float(_GL ** 2 / (_HL + _lam) + _GR ** 2 / (_HR + _lam) - (_GL + _GR) ** 2 / (_HL + _HR + _lam))
assert max(K3[1], key=K3[1].get) == 0.8 == max(K3[0], key=K3[0].get)
assert round(K3[0][0.8], 3) == round(0.815 - 0.08, 3)
K4 = {lam: [round(n / (n + lam), 3) for n in [1, 5, 25, 100, 500]] for lam in [1, 10, 100]}
K5_SETTINGS = [{"n_estimators": 200, "learning_rate": 0.1, "max_depth": 1},
               {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 2},
               {"n_estimators": 50, "learning_rate": 0.3, "max_depth": 1}]
K5 = []
for _s in K5_SETTINGS:
    _m = XGBRegressor(random_state=0, **_s).fit(FIT[COLUMNS], FIT["vol_next"])
    K5.append(rmse(STOP["vol_next"], _m.predict(STOP[COLUMNS])))
assert int(np.argmin(K5)) == 0

# ======================================================================
# Notebook
# ======================================================================
md(
"# \U0001f9ea Session 12 exercises\n"
"### Boosting, and reading a model\n\n"
"The lecture built boosting from one step: fit a small tree to what the "
"forecast still gets wrong, and add a fraction of it. Then it chose the number "
"of trees on a later block, met XGBoost and LightGBM, boosted a label, and read "
"the models through their importance. These exercises do each of those steps "
"yourself, starting with the six days by hand.\n\n"
"Three tables, all from earlier sessions. The **index table** from Session 6, "
"with the volatility over the next 20 days as the target. The **credit table** "
"from Session 9, with a default as the label. And the **daily prices** of the "
"eleven instruments, for a target of your own in section J."
)

md(
"## How to use this notebook\n\n"
"- Run the **setup cell** below first. It loads the three tables, makes the "
"splits, and imports scikit-learn, XGBoost and LightGBM.\n"
"- Each exercise has a **task**, then a **code cell** for your work. Cells with "
"`...` are blanks to fill in. Replace them with real code.\n"
"- Stuck? Open the **\U0001f4a1 Hint**, but only after a genuine attempt. Open the "
"**✅ Solution** to *check* yourself, not to skip the thinking.\n"
"- Every cell runs cleanly even with the blanks still in place, so pressing "
"**Run all** never floods you with errors.\n"
"- Most exercises stand alone. A few short runs build on each other (A4 to A6, "
"C2 to C3, D1 to D3, D4 to D5, G4 to G5, H1 to H2, H3 to H4); the task says "
"which earlier exercise it continues from. Section I shows five mistakes and "
"asks you to fix them. Section J uses the table from J1, and **section K is five "
"small cases that each start from scratch**. Eight of them ask you to draw "
"something: A5, C3, D5, F4, G5, H2, H4 and J6.\n"
"- A few cells fit hundreds of trees or several models and take several "
"seconds. The task says so where it matters.\n\n"
"**You are not expected to finish all of these.** Do what you can, and come back "
"to the rest when you revise. Short on time? Read the hint, then the solution. A "
"worked solution you genuinely understand is real learning too.\n\n"
"**Units.** The index table and the daily returns are in percent, as in the "
"lecture. The credit table is in New Taiwan dollars, as it comes."
)

md(
"### Difficulty\n\n"
"| badge | what to expect |\n|:--|:--|\n"
"| ★☆☆☆☆ | One step, straight from the lecture. You are checking that you can type it. |\n"
"| ★★☆☆☆ | The same idea on new data, or two steps in a row. Nothing to decide. |\n"
"| ★★★☆☆ | Combine two ideas, or adapt a pattern rather than copy it. |\n"
"| ★★★★☆ | You choose the approach. Several steps, and something has to be worked out before you type. |\n"
"| ★★★★★ | A genuine puzzle: an insight, or a constraint that rules out the obvious route. Always solvable with what you have. |\n\n"
"The stars rate the work against **this** session. A three-star task here "
"assumes everything from Sessions 1 to 11, so it is a bigger piece of work "
"than a three-star task in an earlier notebook.\n\n"
"Some exercises also carry a **revisits** tag. Those need something from an "
"earlier session as well as today's material, and they are there on purpose: "
"the skills are meant to accumulate."
)

md("---")

md(
"## \U0001f9f0 Toolkit\n\n"
"New this session. Hover a name for what it does.\n\n"
'<span title="Boosted trees for a number. n_estimators is the number of trees M, learning_rate is nu, max_depth=1 makes each tree a stump.">`GradientBoostingRegressor`</span> · '
'<span title="The same for a label: the trees are added on the log-odds scale, each fitted to y minus p.">`GradientBoostingClassifier`</span> · '
'<span title="Yields the forecast after 1, 2, ..., M trees, one at a time. It is a generator: wrap it in list() to index it.">`.staged_predict()`</span> · '
'<span title="Stops on its own once n_iter_no_change trees have not helped on a random validation_fraction of the training rows; n_estimators_ is the number kept.">`n_iter_no_change` `validation_fraction` `n_estimators_`</span> · '
'<span title="scikit-learn&#39;s binned boosting: fast, and it has no feature_importances_.">`HistGradientBoostingRegressor`</span> · '
'<span title="XGBoost&#39;s boosted trees, with scikit-learn&#39;s create, fit, predict pattern. reg_lambda is the penalty on the leaf values, gamma the price per leaf.">`XGBRegressor` `XGBClassifier`</span> · '
'<span title="fit(..., eval_set=[(X, y)], verbose=False) scores every round on those rows; early_stopping_rounds=50 in the constructor stops 50 rounds after the best.">`eval_set` `early_stopping_rounds`</span> · '
'<span title="The best round, counted from 0, so the model keeps best_iteration + 1 trees; best_score is its score on the eval_set.">`.best_iteration` `.best_score`</span> · '
'<span title="XGBoost: the score of every round on every eval_set entry, as lists, under validation_0, validation_1, ...">`.evals_result()`</span> · '
'<span title="A random share of the rows (subsample) or of the columns (colsample_bytree) for each tree.">`subsample` `colsample_bytree`</span> · '
'<span title="LightGBM&#39;s boosted trees, grown leaf by leaf. num_leaves=2 makes a stump; verbose=-1 silences it.">`LGBMRegressor` `LGBMClassifier`</span> · '
'<span title="LightGBM&#39;s early stopping: fit(..., eval_set=[...], callbacks=[early_stopping(50, verbose=False)]); then best_iteration_ trees are used.">`early_stopping` `.best_iteration_`</span> · '
'<span title="Importance from the cuts on the training rows. XGBoost: gain by default, importance_type=&#39;weight&#39; counts cuts; LightGBM: counts by default, importance_type=&#39;gain&#39; sums the fall.">`.feature_importances_` `importance_type`</span> · '
'<span title="The score lost when one column is shuffled, n_repeats times, on rows the model did not fit. importances_mean holds one number per column.">`permutation_importance`</span>\n\n'
"**Formulas**\n\n"
"- the boosted forecast after M trees: $\\hat f_M(x) = \\bar y + \\nu \\sum_{m=1}^{M} h_m(x)$\n"
"- each tree $h_m$ is fitted to the residuals $y_i - \\hat f_{m-1}(x_i)$, the negative gradient of "
"$\\tfrac{1}{2}(y - f)^2$; for a label, to $y_i - \\hat p_i$\n"
"- XGBoost's leaf value for squared error: $w = \\sum r_i / (n + \\lambda)$, the leaf's residuals summed, over its days plus $\\lambda$\n"
"- XGBoost's gain of a cut: $G_L^2/(H_L+\\lambda) + G_R^2/(H_R+\\lambda) - (G_L+G_R)^2/(H_L+H_R+\\lambda)$, "
"with $G$ the sum of the gradients $f - y$ in a box and $H$, for squared error, its number of rows\n"
"- permutation importance: $I_j = s - \\frac{1}{K}\\sum_{k=1}^{K} s_{k,j}$"
)

md("---")

md("## ⚙️ Setup · run me first")

code(
XGB_GUARD + '''
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor, XGBClassifier             # the setup guide's install line includes both
from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping

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
    """Root mean squared error, as a plain number."""
    return float(np.sqrt(mean_squared_error(actual, predicted)))


# ---- the index table from Session 6: one row per trading day, percent ----
table = pd.read_csv(data_path("market_features.csv"), parse_dates=["date"]).set_index("date")
columns = list(table.columns[:-1])      # the 19 feature columns
pair = ["vol_20d", "ret_20d"]
train = table.loc[:"2022-12-31"]
test = table.loc["2023-01-01":]
fit_rows = train.loc[:"2021-12-31"]      # 2015 to 2021, to fit on
stop_rows = train.loc["2022-01-01":]     # 2022, to choose the number of trees on
folds = TimeSeriesSplit(n_splits=5)

# ---- the credit table from Session 9: one row per borrower, no time order ----
credit = pd.read_csv(data_path("credit.csv"))
c_columns = ["limit", "age", "late_now", "months_late", "bill", "paid", "utilisation"]
c_train, c_test = train_test_split(credit, test_size=0.3, random_state=0,
                                   stratify=credit["default"])

# ---- the eleven instruments' daily returns, in percent, for section J ----
prices = pd.read_csv(data_path("prices.csv"), parse_dates=["date"])
rets = prices.pivot(index="date", columns="ticker", values="close").pct_change().dropna() * 100

print("index :", table.shape, "  train", len(train), "(fit", len(fit_rows), "+ stop", len(stop_rows), ")  test", len(test))
print("credit:", credit.shape, "  default share", round(credit["default"].mean(), 4))
print("rets  :", rets.shape)'''
)

md("---")

# ====================================================== A
section(
"## A · Boosting the six days by hand\n\n"
"The lecture's six days, boosted one step at a time with NumPy, lists and loops. "
"A4 to A6 run together."
)

ex("A1", "Start from the mean", 1,
   "The six days of the lecture are below. Boosting starts from one number for "
   "every day, the mean of `vol_next`. Compute that first forecast as an array of "
   "six equal values, the residuals `vol_next` minus the forecast, and the RSS, the "
   "sum of the squared residuals.",
   "vol_20d = np.array([0.4, 0.6, 0.8, 1.2, 1.6, 2.4])\nvol_next = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\n\n"
   "forecast = ...\nresiduals = ...\nrss = ...\n\nprint(forecast)\nprint(residuals)\nprint(rss)",
   "`np.full(6, vol_next.mean())` repeats the mean six times. Subtract, square "
   "and `.sum()`; wrap the RSS in `round(float(...), 3)`.",
   "vol_20d = np.array([0.4, 0.6, 0.8, 1.2, 1.6, 2.4])\nvol_next = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\n\n"
   "forecast = np.full(6, vol_next.mean())\nresiduals = vol_next - forecast\n"
   "rss = round(float((residuals ** 2).sum()), 3)\n\nprint(forecast)\nprint(residuals)\nprint(rss)",
   f"Every day starts at 0.95. The residuals run from {A1_RES.min():.2f} to "
   f"{A1_RES.max():.2f} and sum to zero, as residuals from a mean always do. The "
   f"RSS is {A1_RSS}: the error the first tree has to reduce.")

ex("A2", "One stump on the residuals", 2,
   "The first stump cuts the six days after a `vol_20d` of 0.8. A stump forecasts "
   "the mean of the residuals on each side of its cut. With a mask, compute the "
   "two means, the stump's forecast `h` for every day, the forecast after one step "
   "at a learning rate of 0.5, and the new RSS.",
   "vol_20d = np.array([0.4, 0.6, 0.8, 1.2, 1.6, 2.4])\nvol_next = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\n\n"
   "residuals = ...\nleft = ...\nh = ...\nforecast = ...\nnew_rss = ...\n\n"
   "print(h)\nprint(forecast)\nprint(new_rss)",
   ["`left = vol_20d <= 0.8` is True for the three calm days. `np.where(left, a, b)` "
    "takes `a` where `left` is True and `b` elsewhere.",
    "`residuals[left].mean()` and `residuals[~left].mean()` are the stump's two "
    "values. The new forecast is `vol_next.mean() + 0.5 * h`."],
   "vol_20d = np.array([0.4, 0.6, 0.8, 1.2, 1.6, 2.4])\nvol_next = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\n\n"
   "residuals = vol_next - vol_next.mean()\nleft = vol_20d <= 0.8\n"
   "h = np.where(left, residuals[left].mean(), residuals[~left].mean())\n"
   "forecast = vol_next.mean() + 0.5 * h\nnew_rss = round(float(((vol_next - forecast) ** 2).sum()), 3)\n\n"
   "print(h)\nprint(forecast)\nprint(new_rss)",
   f"The stump says {A2_L:.2f} for the three calm days and {A2_R:.2f} for the three "
   "busy ones. Half of that moves the forecast to 0.775 and 1.125, and the RSS falls "
   f"from {A1_RSS} to {A2_RSS}. This is the first step of the lecture's animation.")

ex("A3", "The same, with lists", 2,
   "Redo A1 without NumPy: the mean with `sum` and `len`, the residuals with a loop "
   "that appends to a list, and the RSS with a second loop. Then check the result "
   "against NumPy with `np.isclose`.",
   "vol_next = [0.5, 0.6, 0.7, 1.2, 1.5, 1.2]\n\nmean = ...\nresiduals = []\nfor v in vol_next:\n    ...\n"
   "rss = 0\nfor r in residuals:\n    ...\n\nprint(mean, residuals, rss)\nprint(...)",
   ["`residuals.append(v - mean)` inside the first loop, and `rss = rss + r ** 2` "
    "inside the second.",
    "`np.isclose(rss, ((np.array(vol_next) - np.mean(vol_next)) ** 2).sum())` "
    "compares the two."],
   "vol_next = [0.5, 0.6, 0.7, 1.2, 1.5, 1.2]\n\nmean = sum(vol_next) / len(vol_next)\nresiduals = []\n"
   "for v in vol_next:\n    residuals.append(v - mean)\nrss = 0\nfor r in residuals:\n    rss = rss + r ** 2\n\n"
   "print(mean, residuals, rss)\nprint(np.isclose(rss, ((np.array(vol_next) - np.mean(vol_next)) ** 2).sum()))",
   "0.95, the six residuals and 0.815 again, so `True`. The mean prints as "
   "`0.9500000000000001` and the residuals carry similar tails, because 0.95 has no "
   "exact binary form. That is why the check uses `np.isclose` and not `==`.")

ex("A4", "One step as a function", 3,
   "Write `boost_step(x, y, forecast, nu)`: it fits a stump, "
   "`DecisionTreeRegressor(max_depth=1, random_state=0)`, to the residuals "
   "`y - forecast` and returns the new forecast and the stump. Then start from the "
   "mean of the six days and take six steps at a learning rate of 0.5, collecting "
   "the RSS before the first step and after each one in a list `rss_path`, rounded "
   "to three decimals.",
   "six = pd.DataFrame({'vol_20d': [0.4, 0.6, 0.8, 1.2, 1.6, 2.4]})\n"
   "y6 = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\n\n\n"
   "def boost_step(x, y, forecast, nu):\n    \"\"\"...\"\"\"\n    ...\n\n\n"
   "forecast = np.full(6, y6.mean())\nrss_path = [...]\nfor step in range(6):\n    ...\n\nprint(rss_path)",
   ["Inside the function: fit the stump on `x` and `y - forecast`, then "
    "`return forecast + nu * stump.predict(x), stump`.",
    "Start the list with the RSS of the mean. In the loop: `forecast, stump = "
    "boost_step(six, y6, forecast, 0.5)`, then append "
    "`round(float(((y6 - forecast) ** 2).sum()), 3)`."],
   "six = pd.DataFrame({'vol_20d': [0.4, 0.6, 0.8, 1.2, 1.6, 2.4]})\n"
   "y6 = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\n\n\n"
   "def boost_step(x, y, forecast, nu):\n"
   "    \"\"\"One step of boosting: a stump fitted to the residuals, added at the rate nu.\"\"\"\n"
   "    stump = DecisionTreeRegressor(max_depth=1, random_state=0)\n    stump.fit(x, y - forecast)\n"
   "    return forecast + nu * stump.predict(x), stump\n\n\n"
   "forecast = np.full(6, y6.mean())\nrss_path = [round(float(((y6 - forecast) ** 2).sum()), 3)]\n"
   "for step in range(6):\n    forecast, stump = boost_step(six, y6, forecast, 0.5)\n"
   "    rss_path.append(round(float(((y6 - forecast) ** 2).sum()), 3))\n\nprint(rss_path)",
   f"`{A4_PATH}`: the RSS of the lecture's animation, step by step. Each stump "
   "takes the cut that lowers the RSS most at that step, and the function is the "
   "whole procedure; the loop only repeats it. Keep `boost_step`, `six` and `y6`: "
   "A5 and A6 continue from here.")

ex("A5", "Draw the path", 2,
   "Draw `rss_path` from A4 against the step, 0 to 6, with a marker on every "
   "point, a label on each axis and a title.",
   "fig, ax = plt.subplots(figsize=(7, 3))\n...\n...\nplt.show()",
   ["`ax.plot(range(7), rss_path, marker='o')`.",
    "`ax.set_xlabel('step')`, `ax.set_ylabel('RSS')` and `ax.set_title(..., loc='left')`."],
   "fig, ax = plt.subplots(figsize=(7, 3))\nax.plot(range(7), rss_path, marker='o', color='#1c5cab')\n"
   "ax.set_xlabel('step')\nax.set_ylabel('RSS')\n"
   "ax.set_title('Boosting the six days at a learning rate of 0.5', loc='left')\nplt.show()",
   "The first step removes more than two thirds of the RSS and every later step "
   "less, because each stump can only fix what is left, and only half of it.")

ex("A6", "A new day through fifty stumps", 4,
   "Using `boost_step` from A4, take 50 steps from the mean at a learning rate of "
   "0.5 and keep every stump in a list. A new day is forecast with the same sum: "
   "the mean plus 0.5 times each stump's forecast for that day. Forecast two new "
   "days, with a `vol_20d` of 1.0 and of 1.4, from your list, and check the result "
   "against `GradientBoostingRegressor` with the same settings.",
   "forecast = np.full(6, y6.mean())\nstumps = []\nfor step in range(50):\n    ...\n\n"
   "new_days = pd.DataFrame({'vol_20d': [1.0, 1.4]})\nby_hand = ...\nlibrary = ...\n\n"
   "print(by_hand)\nprint(library)",
   ["In the loop: `forecast, stump = boost_step(six, y6, forecast, 0.5)` and "
    "`stumps.append(stump)`.",
    "Start `total = np.zeros(2)` and add `stump.predict(new_days)` for every stump; "
    "then `by_hand = y6.mean() + 0.5 * total`. The library model is "
    "`GradientBoostingRegressor(n_estimators=50, learning_rate=0.5, max_depth=1, "
    "random_state=0)`, fitted on `six` and `y6`."],
   "forecast = np.full(6, y6.mean())\nstumps = []\nfor step in range(50):\n"
   "    forecast, stump = boost_step(six, y6, forecast, 0.5)\n    stumps.append(stump)\n\n"
   "new_days = pd.DataFrame({'vol_20d': [1.0, 1.4]})\ntotal = np.zeros(2)\nfor stump in stumps:\n"
   "    total = total + stump.predict(new_days)\nby_hand = y6.mean() + 0.5 * total\n\n"
   "gbr = GradientBoostingRegressor(n_estimators=50, learning_rate=0.5, max_depth=1, random_state=0)\n"
   "gbr.fit(six, y6)\nlibrary = gbr.predict(new_days)\n\nprint(by_hand)\nprint(library)",
   f"{A6_NEW[0]:.3f} and {A6_NEW[1]:.3f}, and the library agrees. After 50 steps the "
   f"six days are fitted to within {A6_GAP:.3f}, and every cut sits halfway between "
   "two of them. A day at 1.0 is on a cut, and `<=` sends it to the side of 0.8, "
   "whose `vol_next` was 0.7; a day at 1.4 lands with 1.2, whose `vol_next` was 1.2. "
   "A boosted model is a list of trees and a sum, and between the training days its "
   "forecast jumps.")

ex("A7", "Six days in five stumps", 5,
   "Boosting needed 50 steps to fit the six days to within 0.006. Five stumps can "
   "fit them exactly. Each stump below adds its value to every day above its cut, "
   "and the model starts from the first day's 0.5. Find the five values without "
   "fitting anything, so that 0.5 plus the five stumps reproduces `vol_next`, and "
   "check with `np.allclose`.",
   "vol_20d = np.array([0.4, 0.6, 0.8, 1.2, 1.6, 2.4])\nvol_next = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\n"
   "cuts = [0.5, 0.7, 1.0, 1.4, 2.0]\n\nvalues = ...\nfit = 0.5\n...\n\n"
   "print(values)\nprint(fit)\nprint(...)",
   ["Which days does the stump at 1.0 change, and by how much must it lift them so "
    "that the day at 1.2 is right once the first two stumps have done their part?",
    "Each value is the jump between two neighbouring days: `np.diff(vol_next)`. Then "
    "`for cut, value in zip(cuts, values):` with `fit = fit + value * (vol_20d > cut)` "
    "adds each stump to every day at once."],
   "vol_20d = np.array([0.4, 0.6, 0.8, 1.2, 1.6, 2.4])\nvol_next = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\n"
   "cuts = [0.5, 0.7, 1.0, 1.4, 2.0]\n\nvalues = np.diff(vol_next)\nfit = 0.5\nfor cut, value in zip(cuts, values):\n"
   "    fit = fit + value * (vol_20d > cut)\n\nprint(values)\nprint(fit)\nprint(np.allclose(fit, vol_next))",
   "The values are 0.1, 0.1, 0.5, 0.3 and -0.3, the jumps between neighbouring "
   "days, and the check prints `True`. Any six numbers in a row can be written as a "
   "start and five steps, so five stumps memorise six days. Boosting took 50 steps "
   "and still missed by 0.006 because it is greedy and adds only half of each "
   "stump: it is built to approach the data slowly, not to reach it.")

md("---")

# ====================================================== B
section(
"## B · What each tree is fitted to\n\n"
"The residuals are the negative gradient of the squared error, and for a label "
"the gaps y minus p play the same part. B1 and B2 check both numerically, and B3 "
"runs the learning rate on a single number."
)

ex("B1", "The slope of the squared error, numerically", 2,
   "For one day the squared error of a forecast $f$ is $L(f) = \\tfrac{1}{2}(y - f)^2$. "
   "Write `loss(y, f)` returning it for arrays, then estimate its slope at the mean "
   "forecast of the six days with a central difference, "
   "$\\big(L(f + h) - L(f - h)\\big) / 2h$ with $h = 0.000001$. Check that minus the "
   "slope equals the residuals.",
   "y = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\nf = np.full(6, y.mean())\nh = 0.000001\n\n\n"
   "def loss(y, f):\n    \"\"\"...\"\"\"\n    ...\n\n\nslope = ...\n\nprint(slope)\nprint(...)",
   ["`return 0.5 * (y - f) ** 2`.",
    "`slope = (loss(y, f + h) - loss(y, f - h)) / (2 * h)`, and "
    "`np.allclose(-slope, y - f)` is the check."],
   "y = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\nf = np.full(6, y.mean())\nh = 0.000001\n\n\n"
   "def loss(y, f):\n    \"\"\"Half the squared error of each forecast.\"\"\"\n    return 0.5 * (y - f) ** 2\n\n\n"
   "slope = (loss(y, f + h) - loss(y, f - h)) / (2 * h)\n\nprint(slope)\nprint(np.allclose(-slope, y - f))",
   "The slopes are 0.45, 0.35, 0.25, -0.25, -0.55 and -0.25, the residuals of A1 "
   "with the sign turned, and the check prints `True`. A tree fitted to the "
   "residuals is fitted to the direction in which the loss falls fastest, which is "
   "why the method is called gradient boosting.")

ex("B2", "The slope of the log-loss", 3,
   "For a label $y$ of 0 or 1 and a log-odds $f$, the log-loss is "
   "$L(f) = -\\big[y \\log \\sigma(f) + (1 - y) \\log(1 - \\sigma(f))\\big]$, with $\\sigma$ "
   "the sigmoid. For the four borrowers below, estimate the slope at their log-odds "
   "with a central difference, as in B1, and check that minus the slope equals "
   "$y - \\sigma(f)$: the label minus the probability.",
   "y = np.array([1, 0, 0, 1])\nf = np.array([0.5, -1.0, 0.2, 2.0])\nh = 0.000001\n\n\n"
   "def sigmoid(z):\n    \"\"\"...\"\"\"\n    ...\n\n\n"
   "def log_loss(y, f):\n    \"\"\"...\"\"\"\n    ...\n\n\nslope = ...\ngap = ...\n\nprint(gap)\nprint(...)",
   ["`sigmoid` returns `1 / (1 + np.exp(-z))`, and `log_loss` uses it twice, "
    "with `np.log`.",
    "`gap = y - sigmoid(f)`; `np.allclose(-slope, gap)` is the check."],
   "y = np.array([1, 0, 0, 1])\nf = np.array([0.5, -1.0, 0.2, 2.0])\nh = 0.000001\n\n\n"
   "def sigmoid(z):\n    \"\"\"The probability for a log-odds z.\"\"\"\n    return 1 / (1 + np.exp(-z))\n\n\n"
   "def log_loss(y, f):\n    \"\"\"The log-loss of each label for a log-odds f.\"\"\"\n"
   "    return -(y * np.log(sigmoid(f)) + (1 - y) * np.log(1 - sigmoid(f)))\n\n\n"
   "slope = (log_loss(y, f + h) - log_loss(y, f - h)) / (2 * h)\ngap = y - sigmoid(f)\n\n"
   "print(gap)\nprint(np.allclose(-slope, gap))",
   f"The gaps are {B2_GAP[0]:.3f}, {B2_GAP[1]:.3f}, {B2_GAP[2]:.3f} and {B2_GAP[3]:.3f}, "
   "and the check prints `True`. A boosted classifier fits each tree to these gaps: "
   f"the borrower who defaulted at a probability of {sigmoid(0.5):.2f} pulls the next "
   f"tree up by {B2_GAP[0]:.2f}, and the one who did not default at {sigmoid(0.2):.2f} "
   f"pulls it down by {-B2_GAP[2]:.2f}.")

ex("B3", "A learning rate on one number", 3,
   "Boosting with no columns at all moves one number. Start a forecast at 0 for the "
   "six days and repeat $f \\leftarrow f + \\nu \\cdot \\text{mean}(y - f)$ in a `while` "
   "loop until $f$ is within 0.001 of the mean, counting the steps, for $\\nu = 0.5$ "
   "and for $\\nu = 0.1$. Compare each count with the formula "
   "$\\log(0.001 / \\bar y) / \\log(1 - \\nu)$. The loop is bounded at 1,000 steps.",
   "y = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\n\nfor nu in [0.5, 0.1]:\n    f = 0.0\n    steps = 0\n"
   "    while abs(y.mean() - f) >= 0.001 and steps < 1000:\n        steps = steps + 1\n        ...\n"
   "    formula = ...\n    print(nu, steps, formula)",
   ["Inside the loop, `f = f + nu * (y - f).mean()`.",
    "`np.log(0.001 / y.mean()) / np.log(1 - nu)`; the loop needs the next whole "
    "number above it."],
   "y = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\n\nfor nu in [0.5, 0.1]:\n    f = 0.0\n    steps = 0\n"
   "    while abs(y.mean() - f) >= 0.001 and steps < 1000:\n        steps = steps + 1\n"
   "        f = f + nu * (y - f).mean()\n    formula = np.log(0.001 / y.mean()) / np.log(1 - nu)\n"
   "    print(nu, steps, round(formula, 2))",
   f"{B3[0.5][0]} steps at ν = 0.5 and {B3[0.1][0]} at ν = 0.1, against "
   f"{B3[0.5][1]:.1f} and {B3[0.1][1]:.1f} from the formula. Each step closes the "
   "share ν of the gap that is left, so the gap after m steps is (1 − ν) to the "
   "power m of the first one. A smaller learning rate needs more steps for the same "
   "distance, roughly in proportion to 1/ν when ν is small.")

md("---")

# ====================================================== C
_c4_txt = ", ".join(f"{C4[k][0]} at {k}" for k in C4)
_c4_prod = ", ".join(f"{C4[k][0] * k:.0f}" for k in C4)
section(
"## C · Boosting in scikit-learn\n\n"
"`GradientBoostingRegressor` on the index table: two columns, then all nineteen, "
"the training error stump by stump, and the learning rate and the depth chosen "
"on 2022. C2 and C3 run together."
)

ex("C1", "Two columns, a hundred stumps", 1,
   "Fit `GradientBoostingRegressor` with 100 stumps at a learning rate of 0.1 on "
   "`pair` alone, and print its training and test RMSE.",
   "boost_pair = ...\n...\n\nprint(...)\nprint(...)",
   "`GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, "
   "random_state=0)`, fitted on `train[pair]` and `train['vol_next']`.",
   "boost_pair = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)\n"
   "boost_pair.fit(train[pair], train['vol_next'])\n\n"
   "print(rmse(train['vol_next'], boost_pair.predict(train[pair])))\n"
   "print(rmse(test['vol_next'], boost_pair.predict(test[pair])))",
   f"{C1_TR:.3f} on the training days and {C1_TE:.3f} on the test days. On all "
   f"nineteen columns the lecture's 100 stumps scored {LECT_TE:.3f}: the other "
   f"seventeen columns are worth about {C1_TE - LECT_TE:.2f} here.")

ex("C2", "The training error, stump by stump", 2,
   "Fit 300 stumps at a learning rate of 0.1 on all of `columns`. `staged_predict` "
   "gives the forecast after each stump; compute the training RMSE after each one "
   "into a list `train_path`, and check with `np.diff` that it falls at every "
   "step.",
   "boost = ...\n...\ntrain_path = ...\nfalls_every_step = ...\n\nprint(falls_every_step)",
   ["`[rmse(train['vol_next'], f) for f in boost.staged_predict(train[columns])]` "
    "builds the whole list in one line.",
    "`np.diff(train_path)` holds the 299 changes, and `(np.diff(train_path) < 0).all()` "
    "is True when every one of them is negative."],
   "boost = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=1, random_state=0)\n"
   "boost.fit(train[columns], train['vol_next'])\n"
   "train_path = [rmse(train['vol_next'], f) for f in boost.staged_predict(train[columns])]\n"
   "falls_every_step = bool((np.diff(train_path) < 0).all())\n\n"
   "print(round(train_path[0], 4), round(train_path[-1], 4))\nprint(falls_every_step)",
   f"{C2_PATH[0]:.3f} after one stump and {C2_PATH[-1]:.3f} after 300, and `True`: "
   "on the rows it is fitted on, every stump lowers the error, because each one is "
   "chosen to lower it. That is why the training error cannot say when to stop. "
   "Keep `train_path`: C3 draws it.")

ex("C3", "Draw it on a log axis", 2,
   "Draw `train_path` from C2 against the number of stumps, 1 to 300, with the "
   "x-axis on a log scale, a label on each axis and a title.",
   "fig, ax = plt.subplots(figsize=(8, 3))\n...\n...\nplt.show()",
   ["`ax.plot(range(1, 301), train_path)` and `ax.set_xscale('log')`.",
    "Label the axes, and `ax.set_title(..., loc='left')`."],
   "fig, ax = plt.subplots(figsize=(8, 3))\nax.plot(range(1, 301), train_path, color='#1c5cab')\n"
   "ax.set_xscale('log')\nax.set_xlabel('stumps')\nax.set_ylabel('training RMSE')\n"
   "ax.set_title('Training error of boosted stumps, learning rate 0.1', loc='left')\nplt.show()",
   f"On a log axis the first ten stumps take as much width as the next ninety. The "
   f"error is {C2_PATH[9]:.3f} after 10 stumps, {C2_PATH[99]:.3f} after 100 and "
   f"{C2_PATH[-1]:.3f} after 300, still falling with no floor in sight.")

ex("C4", "The learning rate and the number of trees", 3,
   "Fit 1,000 stumps on `fit_rows`, 2015 to 2021, at each learning rate in "
   "`[0.3, 0.1, 0.03]`. For each, score every stage on `stop_rows`, 2022, find the "
   "number of stumps with the lowest 2022 RMSE, and store it in a dictionary "
   "`best_m` keyed by the rate. Print the dictionary and, for each rate, the number "
   "of stumps times the rate. This cell takes about 20 seconds.",
   "best_m = {}\nfor nu in [0.3, 0.1, 0.03]:\n    ...\n\nprint(best_m)\nfor nu in best_m:\n    print(nu, ...)",
   ["Inside the loop: fit the model, then `scores = [rmse(stop_rows['vol_next'], f) "
    "for f in model.staged_predict(stop_rows[columns])]`.",
    "`int(np.argmin(scores)) + 1` is the number of stumps, since the first stage "
    "is one stump."],
   "best_m = {}\nfor nu in [0.3, 0.1, 0.03]:\n"
   "    model = GradientBoostingRegressor(n_estimators=1000, learning_rate=nu, max_depth=1, random_state=0)\n"
   "    model.fit(fit_rows[columns], fit_rows['vol_next'])\n"
   "    scores = [rmse(stop_rows['vol_next'], f) for f in model.staged_predict(stop_rows[columns])]\n"
   "    best_m[nu] = int(np.argmin(scores)) + 1\n\nprint(best_m)\nfor nu in best_m:\n"
   "    print(nu, round(best_m[nu] * nu, 1))",
   f"{_c4_txt}; the products are {_c4_prod}. A smaller rate needs more trees, roughly "
   "in proportion, as B3 predicted. The lowest 2022 RMSEs are "
   f"{C4[0.3][1]:.3f}, {C4[0.1][1]:.3f} and {C4[0.03][1]:.3f}: on this table the "
   "slower rates did not end lower, they only took longer.")

ex("C5", "Deeper trees stop sooner", 2,
   "Repeat C4's search at a learning rate of 0.1 with trees of depth 1, 2 and 3, "
   "1,000 trees each. Print the best number of trees and the best 2022 RMSE for "
   "each depth. This takes about 20 seconds.",
   "for depth in [1, 2, 3]:\n    ...",
   "The same loop as C4, with `max_depth=depth` and `learning_rate=0.1`; print "
   "`int(np.argmin(scores)) + 1` and `round(min(scores), 4)`.",
   "for depth in [1, 2, 3]:\n"
   "    model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=depth, random_state=0)\n"
   "    model.fit(fit_rows[columns], fit_rows['vol_next'])\n"
   "    scores = [rmse(stop_rows['vol_next'], f) for f in model.staged_predict(stop_rows[columns])]\n"
   "    print(depth, int(np.argmin(scores)) + 1, round(min(scores), 4))",
   f"{C5[1][0]} stumps, {C5[2][0]} trees of depth 2 and {C5[3][0]} of depth 3, with "
   f"2022 RMSEs of {C5[1][1]:.3f}, {C5[2][1]:.3f} and {C5[3][1]:.3f}. Deeper trees fit "
   "more per step, so they reach their best sooner, and on 2022 they never reach "
   "the stumps' level.")

md("---")

# ====================================================== D
assert D3[5][0] == D3[20][0] and D3[50][0] == D3[200][0]
section(
"## D · How many trees\n\n"
"Early stopping is a choice made on validation rows. These rebuild XGBoost's rule "
"by hand, and compare the 2022 block with the folds. D1 to D3, and D4 to D5, run "
"together."
)

ex("D1", "Every round, scored on 2022", 2,
   "Fit `XGBRegressor` with 600 stumps at a learning rate of 0.1 on `fit_rows`, "
   "with `stop_rows` as `eval_set` and no early stopping. `evals_result()` returns "
   "the 2022 RMSE after every round under `['validation_0']['rmse']`. Store that "
   "list as `scores`, and print its length, the round with the lowest score, "
   "counted from 0, and that score.",
   "xgb600 = ...\n...\nscores = ...\n\nprint(...)",
   ["`xgb600.fit(fit_rows[columns], fit_rows['vol_next'], eval_set=[(stop_rows[columns], "
    "stop_rows['vol_next'])], verbose=False)`.",
    "`int(np.argmin(scores))` is the round, and `min(scores)` the score."],
   "xgb600 = XGBRegressor(n_estimators=600, learning_rate=0.1, max_depth=1, random_state=0)\n"
   "xgb600.fit(fit_rows[columns], fit_rows['vol_next'],\n"
   "           eval_set=[(stop_rows[columns], stop_rows['vol_next'])], verbose=False)\n"
   "scores = xgb600.evals_result()['validation_0']['rmse']\n\n"
   "print(len(scores), int(np.argmin(scores)), round(min(scores), 4))",
   f"600 scores, and the lowest is {min(D1_SCORES):.3f}, at round {D1_BEST}; the first "
   f"round scores {D1_SCORES[0]:.3f}. Keep `scores`: D2 and D3 stop on it by hand.")

ex("D2", "Early stopping, by hand", 4,
   "XGBoost's `early_stopping_rounds=50` stops once 50 rounds have passed without "
   "a new lowest score, and keeps the best round. Write that rule with a `for` loop "
   "over `scores` from D1: keep the best score so far and its round, and `break` as "
   "soon as the current round is 50 past the best. Then fit the stopped model as "
   "the lecture did, and check that its `best_iteration` equals your best round.",
   "best_score = ...\nbest_round = ...\n...\nstopped_at = ...\n\n"
   "print('best round', best_round, ' stopped at', stopped_at)\n\nxgb_stop = ...\n...\nprint(...)",
   ["Start from `best_score = scores[0]` and `best_round = 0`, and loop with "
    "`for i, s in enumerate(scores):`. Inside, `if s < best_score:` updates both, and "
    "`if i - best_round >= 50: break` stops. After the loop, `stopped_at = i`.",
    "`XGBRegressor(n_estimators=3000, learning_rate=0.1, max_depth=1, "
    "early_stopping_rounds=50, random_state=0)`, fitted with the same `eval_set`; "
    "compare `xgb_stop.best_iteration == best_round`."],
   "best_score = scores[0]\nbest_round = 0\nfor i, s in enumerate(scores):\n    if s < best_score:\n"
   "        best_score = s\n        best_round = i\n    if i - best_round >= 50:\n        break\nstopped_at = i\n\n"
   "print('best round', best_round, ' stopped at', stopped_at)\n\n"
   "xgb_stop = XGBRegressor(n_estimators=3000, learning_rate=0.1, max_depth=1,\n"
   "                        early_stopping_rounds=50, random_state=0)\n"
   "xgb_stop.fit(fit_rows[columns], fit_rows['vol_next'],\n"
   "             eval_set=[(stop_rows[columns], stop_rows['vol_next'])], verbose=False)\n"
   "print(xgb_stop.best_iteration == best_round)",
   f"The loop stops at round {D2[1]} with the best at {D2[0]}, and XGBoost agrees: "
   "`True`. Early stopping fits 50 rounds it will not keep, and `predict` then uses "
   f"the first {D2[0] + 1} trees, rounds 0 to {D2[0]}.")

ex("D3", "Patience as an argument", 3,
   "Turn D2's loop into a function `stop_round(scores, patience=50)` that returns "
   "the best round and the round it stopped at. Call it on `scores` from D1 with a "
   "patience of 5, 20, 50 and 200, and print the best round and its 2022 score for "
   "each.",
   "def stop_round(scores, patience=50):\n    \"\"\"...\"\"\"\n    ...\n\n\nfor patience in [5, 20, 50, 200]:\n    ...",
   ["Inside the function, D2's loop with `patience` in place of 50, returning "
    "`best_round, i` at the `break`. After the loop, `return best_round, len(scores) - 1` "
    "covers a run that never stops.",
    "In the loop below: `best, stopped = stop_round(scores, patience)` and "
    "`print(patience, best, stopped, round(scores[best], 4))`."],
   "def stop_round(scores, patience=50):\n"
   "    \"\"\"The best round, and the round at which patience ran out.\"\"\"\n"
   "    best_score = scores[0]\n    best_round = 0\n    for i, s in enumerate(scores):\n"
   "        if s < best_score:\n            best_score = s\n            best_round = i\n"
   "        if i - best_round >= patience:\n            return best_round, i\n"
   "    return best_round, len(scores) - 1\n\n\n"
   "for patience in [5, 20, 50, 200]:\n    best, stopped = stop_round(scores, patience)\n"
   "    print(patience, best, stopped, round(scores[best], 4))",
   f"With a patience of 5 or 20 the best round is {D3[5][0]}, at {D1_SCORES[D3[5][0]]:.3f}; "
   f"with 50 or 200 it is {D3[50][0]}, at {min(D1_SCORES):.3f}. The scores rise after "
   f"round {D3[5][0]} and fall below it only much later, so too little patience stops "
   "in the first dip. The default argument makes 50 the value you get when you do "
   "not say.")

ex("D4", "The folds choose fewer trees", 3,
   "Score boosted stumps at a learning rate of 0.1 on the time-series `folds` for "
   "`n_estimators` in `[10, 25, 50, 100, 200, 400]`, collect the mean fold RMSE in a "
   "dictionary `fold_rmse`, and print the number of trees the folds prefer. The "
   "2022 block preferred 219 in C4. This takes about 20 seconds.",
   "fold_rmse = {}\nfor n in [10, 25, 50, 100, 200, 400]:\n    ...\n\nprint(fold_rmse)\nprint(...)",
   ["`-cross_val_score(GradientBoostingRegressor(n_estimators=n, learning_rate=0.1, "
    "max_depth=1, random_state=0), train[columns], train['vol_next'], cv=folds, "
    "scoring='neg_root_mean_squared_error').mean()`.",
    "`min(fold_rmse, key=fold_rmse.get)` is the key with the smallest value."],
   "fold_rmse = {}\nfor n in [10, 25, 50, 100, 200, 400]:\n"
   "    model = GradientBoostingRegressor(n_estimators=n, learning_rate=0.1, max_depth=1, random_state=0)\n"
   "    fold_rmse[n] = float(-cross_val_score(model, train[columns], train['vol_next'], cv=folds,\n"
   "                                          scoring='neg_root_mean_squared_error').mean())\n\n"
   "print({n: round(v, 4) for n, v in fold_rmse.items()})\nprint(min(fold_rmse, key=fold_rmse.get))",
   f"The folds prefer {D4_BEST} stumps, at {D4[D4_BEST]:.3f}, and the error rises "
   f"steadily after that, to {D4[400]:.3f} at 400. Ridge with an alpha of 1000 scores "
   f"{D4_RIDGE:.3f} on the same folds. On the early folds the fitting rows are few, "
   "so more trees overfit them quickly. Two validation choices that both respect "
   f"time disagree here by a factor of about {_words(219 / D4_BEST)}. Keep `fold_rmse`: "
   "D5 draws it.")

ex("D5", "Draw the fold curve", 2,
   "Draw `fold_rmse` from D4 against the number of trees on a log x-axis, mark the "
   f"lowest point, and add a dashed horizontal line at ridge's fold RMSE, "
   f"{D4_RIDGE:.3f}, with a legend.",
   "fig, ax = plt.subplots(figsize=(7, 3))\n...\n...\n...\nplt.show()",
   ["`ax.plot(list(fold_rmse.keys()), list(fold_rmse.values()), marker='o', "
    "label='boosted stumps')` and `ax.set_xscale('log')`.",
    f"`ax.axhline({D4_RIDGE:.3f}, linestyle='--', color='grey', label='ridge')` for "
    "ridge, `ax.scatter` for the lowest point, then `ax.legend()`."],
   "best = min(fold_rmse, key=fold_rmse.get)\n\nfig, ax = plt.subplots(figsize=(7, 3))\n"
   "ax.plot(list(fold_rmse.keys()), list(fold_rmse.values()), marker='o', color='#1c5cab', label='boosted stumps')\n"
   "ax.scatter([best], [fold_rmse[best]], s=80, color='#b3402f', zorder=3)\n"
   f"ax.axhline({D4_RIDGE:.3f}, linestyle='--', color='grey', label='ridge')\n"
   "ax.set_xscale('log')\nax.set_xlabel('trees')\nax.set_ylabel('mean fold RMSE')\nax.legend()\n"
   "ax.set_title('Boosted stumps on the time-series folds', loc='left')\nplt.show()",
   "The curve sits above ridge's line everywhere. On these folds boosted stumps "
   "are never better than ridge, and they get worse with every tree after 25.")

ex("D6", "Stopping on random days, five times", 2,
   "`GradientBoostingRegressor` stops on its own with `n_iter_no_change=10` and "
   "`validation_fraction=0.1`, holding out a random tenth of the training days. Fit "
   "it with up to 5,000 stumps at a learning rate of 0.1 for `random_state` 0 to 4, "
   "and print the number of trees each keeps, `n_estimators_`. This takes about 15 "
   "seconds.",
   "for seed in range(5):\n    ...",
   "`GradientBoostingRegressor(n_estimators=5000, learning_rate=0.1, max_depth=1, "
   "validation_fraction=0.1, n_iter_no_change=10, random_state=seed)`, fitted on "
   "`train[columns]`.",
   "for seed in range(5):\n"
   "    model = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.1, max_depth=1,\n"
   "                                      validation_fraction=0.1, n_iter_no_change=10, random_state=seed)\n"
   "    model.fit(train[columns], train['vol_next'])\n    print(seed, model.n_estimators_)",
   f"{', '.join(str(v) for v in D6[:-1])} and {D6[-1]} trees: the number depends on "
   f"which tenth was drawn, by a factor of {_words(max(D6) / min(D6))}. The held-out days "
   "have neighbours among the fitting days, so every run stops late and none agrees "
   "with another, which is why the lecture stopped on a later block instead.")

md("---")

# ====================================================== E
_e3_ratio = (E3["scikit-learn"][0] / E3["XGBoost"][0], E3["scikit-learn"][0] / E3["LightGBM"][0])
section(
"## E · XGBoost and LightGBM\n\n"
"The two libraries on the same table: their numbers against scikit-learn's, the "
"penalty inside XGBoost's leaf values, their speed, and settings the lecture "
"listed as sometimes worth changing."
)

ex("E1", "The same stumps in XGBoost", 1,
   "Fit `XGBRegressor` with 100 stumps at a learning rate of 0.1 on `pair`, and "
   "print its test RMSE.",
   "xgb_pair = ...\n...\n\nprint(...)",
   "`XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)`, "
   "fitted on `train[pair]`.",
   "xgb_pair = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)\n"
   "xgb_pair.fit(train[pair], train['vol_next'])\n\nprint(rmse(test['vol_next'], xgb_pair.predict(test[pair])))",
   f"{E1_TE:.3f}, against {C1_TE:.3f} for scikit-learn's 100 stumps on the same two "
   "columns (C1). XGBoost searches 256 bins per column and puts a small penalty on "
   "each leaf, so the two are close and not identical.")

ex("E2", "The leaf values from the formula", 4,
   "For squared error, XGBoost gives a leaf the value $\\sum r_i / (n + \\lambda)$: the "
   "sum of its residuals over its number of days plus $\\lambda$, which is "
   "`reg_lambda`. Fit one stump on the six "
   "days with `learning_rate=1`, `base_score` set to the mean of `y6` and "
   "`reg_lambda` of 0, 1 and 10 in turn; its cut falls after 0.8 each time. Compute "
   "the two leaf values from the formula and check the model's forecasts against "
   "them.",
   "six = pd.DataFrame({'vol_20d': [0.4, 0.6, 0.8, 1.2, 1.6, 2.4]})\n"
   "y6 = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\n\nleft = ...\nresiduals = ...\n\n"
   "for lam in [0, 1, 10]:\n    one_tree = ...\n    ...\n    w_left = ...\n    w_right = ...\n"
   "    print(lam, w_left, w_right, ...)",
   ["`XGBRegressor(n_estimators=1, max_depth=1, learning_rate=1, base_score=y6.mean(), "
    "reg_lambda=lam, random_state=0)`, fitted on `six` and `y6`.",
    "`left = six['vol_20d'].values <= 0.8` and `w_left = residuals[left].sum() / "
    "(left.sum() + lam)`. The model's forecasts are the mean plus the leaf value on "
    "each side: `np.allclose(one_tree.predict(six), y6.mean() + np.where(left, w_left, "
    "w_right), atol=0.00001)`."],
   "six = pd.DataFrame({'vol_20d': [0.4, 0.6, 0.8, 1.2, 1.6, 2.4]})\n"
   "y6 = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\n\nleft = six['vol_20d'].values <= 0.8\n"
   "residuals = y6 - y6.mean()\n\nfor lam in [0, 1, 10]:\n"
   "    one_tree = XGBRegressor(n_estimators=1, max_depth=1, learning_rate=1, base_score=y6.mean(),\n"
   "                            reg_lambda=lam, random_state=0)\n    one_tree.fit(six, y6)\n"
   "    w_left = residuals[left].sum() / (left.sum() + lam)\n"
   "    w_right = residuals[~left].sum() / ((~left).sum() + lam)\n"
   "    same = np.allclose(one_tree.predict(six), y6.mean() + np.where(left, w_left, w_right), atol=0.00001)\n"
   "    print(lam, round(w_left, 4), round(w_right, 4), same)",
   f"±{E2[0][1]:.2f} with λ = 0, the plain means of A2; ±{E2[1][1]:.4f} with λ = 1; "
   f"±{E2[10][1]:.3f} with λ = 10. The model agrees each time. λ shrinks a leaf of n "
   "days to n/(n + λ) of its mean: a leaf of three days keeps three quarters at "
   "λ = 1, a leaf of 300 days almost all of it. The penalty reins in small leaves.")

ex("E3", "Three libraries, one core", 2,
   "Time 300 stumps at a learning rate of 0.1 on all of `columns` in "
   "`GradientBoostingRegressor`, `XGBRegressor` and `LGBMRegressor`, with "
   "`n_jobs=1` for the last two so that each uses one core. Store the seconds and "
   "the test RMSE of each in a dictionary of tuples.",
   "models = {\n    'scikit-learn': ...,\n    'XGBoost': ...,\n    'LightGBM': ...,\n}\nresults = {}\n"
   "for name in models:\n    ...\n\nprint(results)",
   ["LightGBM makes a stump with `num_leaves=2`, and wants `verbose=-1`.",
    "In the loop: `start = time.perf_counter()`, fit `models[name]`, then "
    "`results[name] = (round(time.perf_counter() - start, 3), "
    "round(rmse(test['vol_next'], models[name].predict(test[columns])), 4))`."],
   "models = {\n"
   "    'scikit-learn': GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=1, random_state=0),\n"
   "    'XGBoost': XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=1, random_state=0, n_jobs=1),\n"
   "    'LightGBM': LGBMRegressor(n_estimators=300, learning_rate=0.1, num_leaves=2, random_state=0,\n"
   "                              n_jobs=1, verbose=-1),\n}\nresults = {}\nfor name in models:\n"
   "    start = time.perf_counter()\n    models[name].fit(train[columns], train['vol_next'])\n"
   "    seconds = time.perf_counter() - start\n"
   "    results[name] = (round(seconds, 3), round(rmse(test['vol_next'], models[name].predict(test[columns])), 4))\n\n"
   "print(results)",
   f"Here scikit-learn took {E3['scikit-learn'][0]:.2f} seconds, XGBoost "
   f"{E3['XGBoost'][0]:.2f} and LightGBM {E3['LightGBM'][0]:.2f}, with test RMSEs of "
   f"{E3['scikit-learn'][1]:.3f}, {E3['XGBoost'][1]:.3f} and {E3['LightGBM'][1]:.3f}. "
   f"The two libraries were about {_e3_ratio[0]:.0f} and {_e3_ratio[1]:.0f} times "
   "faster at the same job. The seconds depend on the machine; the ratios much "
   "less.")

ex("E4", "LightGBM's leaves, stopped on 2022", 3,
   "Fit `LGBMRegressor` at a learning rate of 0.1 with up to 3,000 trees and "
   "`num_leaves` of 2, 4, 8 and 16, each stopped on `stop_rows` with the "
   "`early_stopping(50, verbose=False)` callback. Collect the number of trees, "
   "`best_iteration_`, and the best 2022 RMSE in a dictionary keyed by the number "
   "of leaves.",
   "by_leaves = {}\nfor leaves in [2, 4, 8, 16]:\n    ...\n\nprint(by_leaves)",
   ["`model.fit(fit_rows[columns], fit_rows['vol_next'], eval_set=[(stop_rows[columns], "
    "stop_rows['vol_next'])], callbacks=[early_stopping(50, verbose=False)])`.",
    "LightGBM records the squared error: `model.best_score_['valid_0']['l2'] ** 0.5` "
    "is the RMSE."],
   "by_leaves = {}\nfor leaves in [2, 4, 8, 16]:\n"
   "    model = LGBMRegressor(n_estimators=3000, learning_rate=0.1, num_leaves=leaves, random_state=0, verbose=-1)\n"
   "    model.fit(fit_rows[columns], fit_rows['vol_next'],\n"
   "              eval_set=[(stop_rows[columns], stop_rows['vol_next'])],\n"
   "              callbacks=[early_stopping(50, verbose=False)])\n"
   "    by_leaves[leaves] = (model.best_iteration_, round(float(model.best_score_['valid_0']['l2'] ** 0.5), 4))\n\n"
   "print(by_leaves)",
   f"{E4[2][0]} stumps at {E4[2][1]:.3f}; {E4[4][0]} trees of 4 leaves at "
   f"{E4[4][1]:.3f}; {E4[8][0]} of 8 at {E4[8][1]:.3f}; {E4[16][0]} of 16 at "
   f"{E4[16][1]:.3f}. Bigger trees stop sooner and score worse on 2022, the pattern "
   "of C5 in another library. On this table the stump is the right size.")

_e5 = {k: v for k, v in E5.items()}
ex("E5", "Random rows and random columns", 3,
   "`subsample` fits each tree on a random share of the rows, and "
   "`colsample_bytree` gives it a random share of the columns, as a forest does. "
   "Fit XGBoost stumps at a learning rate of 0.1, stopped on `stop_rows` 50 rounds "
   "after the best, with (subsample, colsample_bytree) of (1, 1), (0.5, 1), "
   "(1, 0.5) and (0.5, 0.5). Print the number of trees kept and the best 2022 RMSE "
   "for each pair.",
   "for rows_share, cols_share in [(1.0, 1.0), (0.5, 1.0), (1.0, 0.5), (0.5, 0.5)]:\n    ...",
   ["`XGBRegressor(n_estimators=3000, learning_rate=0.1, max_depth=1, "
    "subsample=rows_share, colsample_bytree=cols_share, early_stopping_rounds=50, "
    "random_state=0)`.",
    "Print `model.best_iteration + 1` and `round(model.best_score, 4)`."],
   "for rows_share, cols_share in [(1.0, 1.0), (0.5, 1.0), (1.0, 0.5), (0.5, 0.5)]:\n"
   "    model = XGBRegressor(n_estimators=3000, learning_rate=0.1, max_depth=1, subsample=rows_share,\n"
   "                         colsample_bytree=cols_share, early_stopping_rounds=50, random_state=0)\n"
   "    model.fit(fit_rows[columns], fit_rows['vol_next'],\n"
   "              eval_set=[(stop_rows[columns], stop_rows['vol_next'])], verbose=False)\n"
   "    print(rows_share, cols_share, model.best_iteration + 1, round(model.best_score, 4))",
   f"Without either, {E5[(1.0, 1.0)][0]} trees and {E5[(1.0, 1.0)][1]:.3f}. Half the "
   f"rows: {E5[(0.5, 1.0)][0]} trees and {E5[(0.5, 1.0)][1]:.3f}; half the columns: "
   f"{E5[(1.0, 0.5)][0]} and {E5[(1.0, 0.5)][1]:.3f}; both: {E5[(0.5, 0.5)][0]} and "
   f"{E5[(0.5, 0.5)][1]:.3f}. Here the randomness made each tree noisier without "
   "making the sum better on 2022, so both settings stay at their defaults.")

md("---")

# ====================================================== F
section(
"## F · Boosting a label\n\n"
"The credit table, as in the lecture: the depth, the log-odds behind a "
"probability, the riskiest borrowers, two rankings drawn, and two ways to stop."
)

ex("F1", "Trees of depth 2", 1,
   "Fit `XGBClassifier` with 100 trees of depth 2 at a learning rate of 0.1 on the "
   f"credit table, and print its test AUC. The lecture's trees of depth 3 scored "
   f"{XGB3_AUC:.3f}.",
   "xgb_c2 = ...\n...\n\nprint(...)",
   "`roc_auc_score(c_test['default'], xgb_c2.predict_proba(c_test[c_columns])[:, 1])`.",
   "xgb_c2 = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=0)\n"
   "xgb_c2.fit(c_train[c_columns], c_train['default'])\n\n"
   "print(roc_auc_score(c_test['default'], xgb_c2.predict_proba(c_test[c_columns])[:, 1]))",
   f"{F1_AUC:.3f}. One level less costs {XGB3_AUC - F1_AUC:.3f} of AUC: most of what "
   "trees of depth 3 find, trees of depth 2 find too.")

ex("F2", "The log-odds behind a probability", 3,
   "A boosted classifier adds its trees on the log-odds scale. Fit XGBoost with 100 "
   "trees of depth 3 at a learning rate of 0.1, get the summed log-odds of the test "
   "borrowers with `predict(..., output_margin=True)`, turn them into probabilities "
   "with the sigmoid, and check them against `predict_proba`. Print the smallest "
   "and largest log-odds.",
   "xgb_c = ...\n...\nmargin = ...\nprobability = ...\n\nprint(...)\nprint(...)",
   ["`margin = xgb_c.predict(c_test[c_columns], output_margin=True)`.",
    "`probability = 1 / (1 + np.exp(-margin))`, then `np.allclose(probability, "
    "xgb_c.predict_proba(c_test[c_columns])[:, 1], atol=0.000001)`."],
   "xgb_c = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)\n"
   "xgb_c.fit(c_train[c_columns], c_train['default'])\n"
   "margin = xgb_c.predict(c_test[c_columns], output_margin=True)\nprobability = 1 / (1 + np.exp(-margin))\n\n"
   "print(np.allclose(probability, xgb_c.predict_proba(c_test[c_columns])[:, 1], atol=0.000001))\n"
   "print(round(float(margin.min()), 2), round(float(margin.max()), 2))",
   f"`True`. The log-odds run from {F2_MIN:.2f} to {F2_MAX:.2f}, which the sigmoid "
   f"turns into probabilities from {sigmoid(F2_MIN):.3f} to {sigmoid(F2_MAX):.2f}. The "
   "trees never see a probability: they move the log-odds, and the sigmoid is "
   "applied once, at the end.")

ex("F3", "The 500 borrowers ranked riskiest", 3,
   "Rank the test borrowers by the probability of a default from XGBoost (100 "
   "trees of depth 3) and from a scaled logistic regression. For each model, take "
   "the 500 borrowers with the highest probability and print the share of them who "
   "defaulted, beside the share among all test borrowers.",
   "xgb_c = ...\n...\nlogit = ...\n...\n\nfor name, model in [('XGBoost', xgb_c), ('logistic regression', logit)]:\n"
   "    ...\n\nprint('all test borrowers:', ...)",
   ["`p = pd.Series(model.predict_proba(c_test[c_columns])[:, 1], index=c_test.index)` "
    "and `top = p.nlargest(500).index`.",
    "`c_test.loc[top, 'default'].mean()` is the share who defaulted among them."],
   "xgb_c = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)\n"
   "xgb_c.fit(c_train[c_columns], c_train['default'])\n"
   "logit = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "logit.fit(c_train[c_columns], c_train['default'])\n\n"
   "for name, model in [('XGBoost', xgb_c), ('logistic regression', logit)]:\n"
   "    p = pd.Series(model.predict_proba(c_test[c_columns])[:, 1], index=c_test.index)\n"
   "    top = p.nlargest(500).index\n    print(name, round(c_test.loc[top, 'default'].mean(), 3))\n\n"
   "print('all test borrowers:', round(c_test['default'].mean(), 3))",
   f"{100 * F3['XGBoost']:.1f} percent of XGBoost's 500 defaulted and "
   f"{100 * F3['logistic regression']:.1f} percent of logistic regression's, against "
   f"{100 * F3_BASE:.1f} percent overall. At the very top the two models differ by "
   f"{100 * (F3['XGBoost'] - F3['logistic regression']):.0f} percentage points; the "
   f"AUC gap, {XGB3_AUC:.3f} against {LOGIT_AUC:.3f}, is spread over the whole "
   "ranking.")

ex("F4", "Two ROC curves on one axis", 2,
   "Draw the ROC curves of a scaled logistic regression and of XGBoost (100 trees "
   "of depth 3) on the credit test rows, on one axis, with the diagonal of a "
   "ranking that knows nothing, a legend and a title.",
   "p_logit = ...\np_xgb = ...\n\nfig, ax = plt.subplots(figsize=(5, 5))\n...\n...\n...\nplt.show()",
   ["`fpr, tpr, _ = roc_curve(c_test['default'], p_logit)` and "
    "`ax.plot(fpr, tpr, label='logistic regression')`; the same for `p_xgb`.",
    "`ax.plot([0, 1], [0, 1], linestyle='--', color='grey')` is the diagonal."],
   "logit = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "logit.fit(c_train[c_columns], c_train['default'])\n"
   "xgb_c = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)\n"
   "xgb_c.fit(c_train[c_columns], c_train['default'])\n"
   "p_logit = logit.predict_proba(c_test[c_columns])[:, 1]\np_xgb = xgb_c.predict_proba(c_test[c_columns])[:, 1]\n\n"
   "fig, ax = plt.subplots(figsize=(5, 5))\n"
   "for name, p in [('logistic regression', p_logit), ('XGBoost', p_xgb)]:\n"
   "    fpr, tpr, _ = roc_curve(c_test['default'], p)\n    ax.plot(fpr, tpr, label=name)\n"
   "ax.plot([0, 1], [0, 1], linestyle='--', color='grey')\n"
   "ax.set_xlabel('false positive rate')\nax.set_ylabel('true positive rate')\nax.legend()\n"
   "ax.set_title('Credit test rows: two rankings', loc='left')\nplt.show()",
   "XGBoost's curve lies above logistic regression's over most of the range, and "
   f"the difference in the area under them is small: {XGB3_AUC - LOGIT_AUC:.3f}. Both "
   "rise steeply at first, because the borrowers each model ranks riskiest mostly "
   "did default, as F3 found.")

ex("F5", "Stop on the log-loss or on the AUC", 3,
   "Hold back a stratified fifth of the credit training rows, as the lecture did, "
   "and stop XGBoost (trees of depth 3, learning rate 0.1) on it twice: once on the "
   "log-loss, its default, and once with `eval_metric='auc'`. Print the number of "
   "trees each keeps and its test AUC.",
   "fit_c = ...\nstop_c = ...\n\nfor metric in ['logloss', 'auc']:\n    ...",
   ["`fit_c, stop_c = train_test_split(c_train, test_size=0.2, random_state=0, "
    "stratify=c_train['default'])` fills both lines at once.",
    "`XGBClassifier(n_estimators=3000, learning_rate=0.1, max_depth=3, "
    "early_stopping_rounds=50, eval_metric=metric, random_state=0)`, fitted with "
    "`eval_set=[(stop_c[c_columns], stop_c['default'])]` and `verbose=False`."],
   "fit_c, stop_c = train_test_split(c_train, test_size=0.2, random_state=0, stratify=c_train['default'])\n\n"
   "for metric in ['logloss', 'auc']:\n"
   "    model = XGBClassifier(n_estimators=3000, learning_rate=0.1, max_depth=3,\n"
   "                          early_stopping_rounds=50, eval_metric=metric, random_state=0)\n"
   "    model.fit(fit_c[c_columns], fit_c['default'],\n"
   "              eval_set=[(stop_c[c_columns], stop_c['default'])], verbose=False)\n"
   "    auc = roc_auc_score(c_test['default'], model.predict_proba(c_test[c_columns])[:, 1])\n"
   "    print(metric, model.best_iteration + 1, round(auc, 4))",
   f"{F5['logloss'][0]} trees on the log-loss and {F5['auc'][0]} on the AUC, with "
   f"test AUCs of {F5['logloss'][1]:.4f} and {F5['auc'][1]:.4f}. The two yardsticks "
   "stop at different points and end with the same model, near enough: between "
   f"{F5['auc'][0]} and {F5['logloss'][0]} trees the AUC hardly moves.")

md("---")

# ====================================================== G
section(
"## G · Reading a model\n\n"
"Importance on the credit table, measured four ways: from the cuts, by count and "
"by gain; by shuffling a column; and by refitting without it. G4 and G5 run "
"together."
)

ex("G1", "Gain or count", 2,
   "Fit XGBoost (100 trees of depth 3, learning rate 0.1) twice, once as it comes "
   "and once with `importance_type='weight'`, and print both `feature_importances_` "
   "as Series sorted from the largest. `'weight'` counts the cuts on each column; "
   "the default, `'gain'`, averages how much they lowered the loss.",
   "by_gain = ...\n...\nby_count = ...\n...\n\nprint(...)\nprint(...)",
   ["`XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0, "
    "importance_type='weight')`.",
    "`pd.Series(model.feature_importances_, index=c_columns).sort_values(ascending=False)`."],
   "by_gain = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)\n"
   "by_gain.fit(c_train[c_columns], c_train['default'])\n"
   "by_count = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0,\n"
   "                         importance_type='weight')\nby_count.fit(c_train[c_columns], c_train['default'])\n\n"
   "print(pd.Series(by_gain.feature_importances_, index=c_columns).sort_values(ascending=False).round(3))\n"
   "print(pd.Series(by_count.feature_importances_, index=c_columns).sort_values(ascending=False).round(3))",
   f"By gain, `late_now` comes first with {G1_GAIN['late_now']:.3f} and `months_late` "
   f"second; by count, `bill` comes first with {G1_WEIGHT['bill']:.3f} and `late_now` "
   f"last with {G1_WEIGHT['late_now']:.3f}. The same 100 trees, two opposite rankings: "
   f"`bill` has {G1_NUNIQUE['bill']:,} different values in the training rows to cut "
   f"at, and `late_now` has {G1_NUNIQUE['late_now']}.")

ex("G2", "Two columns of noise", 3,
   "Add two columns of noise to a copy of the credit training rows: `noise_num`, "
   "drawn from a normal distribution, and `noise_coin`, 0 or 1 at random, both from "
   "`np.random.default_rng(0)`, in that order. Fit `LGBMClassifier(random_state=0, "
   "verbose=-1)` on the seven columns and the two noise columns, and print its "
   "`feature_importances_`, the counts of cuts, sorted.",
   "noisy = ...\nrng = ...\n...\n...\nnoisy_columns = ...\n\nlgbm = ...\n...\n\nprint(...)",
   ["`rng = np.random.default_rng(0)`, then `rng.normal(size=len(noisy))` and "
    "`rng.integers(0, 2, size=len(noisy))`.",
    "Fit on `c_columns + ['noise_num', 'noise_coin']`."],
   "noisy = c_train.copy()\nrng = np.random.default_rng(0)\nnoisy['noise_num'] = rng.normal(size=len(noisy))\n"
   "noisy['noise_coin'] = rng.integers(0, 2, size=len(noisy))\n"
   "noisy_columns = c_columns + ['noise_num', 'noise_coin']\n\n"
   "lgbm = LGBMClassifier(random_state=0, verbose=-1)\nlgbm.fit(noisy[noisy_columns], noisy['default'])\n\n"
   "print(pd.Series(lgbm.feature_importances_, index=noisy_columns).sort_values(ascending=False))",
   f"`noise_num` is cut {G2['noise_num']} times, more than any real column, and "
   f"`noise_coin` {G2['noise_coin']} times, fewer than any. Both carry nothing. A "
   "column with thousands of values offers thousands of cuts, and some of them fit "
   "noise; a column with two values offers one. Counts of cuts measure opportunity "
   "as much as information.")

ex("G3", "Permutation importance, by hand", 4,
   "Fit XGBoost (100 trees of depth 3) and compute its test AUC. Then shuffle "
   "`months_late` in a copy of the test rows ten times, with "
   "`np.random.default_rng(k).permutation` for k from 0 to 9, and record the AUC "
   "each shuffle loses. Print the mean loss beside what `permutation_importance` "
   "reports for the same column with `n_repeats=10`.",
   "xgb_c = ...\n...\nbase_auc = ...\n\nlosses = []\nfor k in range(10):\n    ...\n\n"
   "result = ...\n\nprint('by hand :', ...)\nprint('function:', ...)",
   ["In the loop: `shuffled = c_test.copy()`, `shuffled['months_late'] = "
    "np.random.default_rng(k).permutation(shuffled['months_late'].values)`, then the "
    "AUC on `shuffled[c_columns]` and `losses.append(base_auc - ...)`.",
    "`permutation_importance(xgb_c, c_test[c_columns], c_test['default'], "
    "scoring='roc_auc', n_repeats=10, random_state=0)`; "
    "`result.importances_mean[c_columns.index('months_late')]` is the column's entry."],
   "xgb_c = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)\n"
   "xgb_c.fit(c_train[c_columns], c_train['default'])\n"
   "base_auc = roc_auc_score(c_test['default'], xgb_c.predict_proba(c_test[c_columns])[:, 1])\n\n"
   "losses = []\nfor k in range(10):\n    shuffled = c_test.copy()\n"
   "    shuffled['months_late'] = np.random.default_rng(k).permutation(shuffled['months_late'].values)\n"
   "    auc = roc_auc_score(c_test['default'], xgb_c.predict_proba(shuffled[c_columns])[:, 1])\n"
   "    losses.append(base_auc - auc)\n\n"
   "result = permutation_importance(xgb_c, c_test[c_columns], c_test['default'], scoring='roc_auc',\n"
   "                                n_repeats=10, random_state=0)\n\n"
   "print('by hand :', round(np.mean(losses), 4), ' sd', round(np.std(losses), 4))\n"
   "print('function:', round(result.importances_mean[c_columns.index('months_late')], 4))",
   f"{G3_HAND:.3f} by hand and {PERM['months_late']:.3f} from the function. The two "
   f"use different shuffles, and the ten losses by hand vary with a standard "
   f"deviation of {G3_SD:.3f}, so a difference of that size is noise. The function "
   "runs the same loop for every column.")

ex("G4", "Refit without each column", 3,
   "Drop-column importance refits the model without a column and measures what it "
   "loses. Fit XGBoost (100 trees of depth 3) on all seven columns as `full`. Then, "
   "for each column, refit on the other six and store the test AUC lost in a "
   "dictionary `drop_loss`. Print it as a Series sorted from the largest.",
   "full = ...\n...\nfull_auc = ...\n\ndrop_loss = {}\nfor column in c_columns:\n    ...\n\nprint(...)",
   ["`rest = [c for c in c_columns if c != column]` holds the other six.",
    "`pd.Series(drop_loss).sort_values(ascending=False)` sorts it."],
   "full = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)\n"
   "full.fit(c_train[c_columns], c_train['default'])\n"
   "full_auc = roc_auc_score(c_test['default'], full.predict_proba(c_test[c_columns])[:, 1])\n\n"
   "drop_loss = {}\nfor column in c_columns:\n    rest = [c for c in c_columns if c != column]\n"
   "    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)\n"
   "    model.fit(c_train[rest], c_train['default'])\n"
   "    drop_loss[column] = full_auc - roc_auc_score(c_test['default'], model.predict_proba(c_test[rest])[:, 1])\n\n"
   "print(pd.Series(drop_loss).sort_values(ascending=False).round(4))",
   f"`months_late` {G4['months_late']:.3f} and `late_now` {G4['late_now']:.3f} lead; "
   f"the other five lose under {G4.iloc[2] + 0.0005:.3f}. Every loss is far below the "
   f"shuffle of G3 ({PERM['months_late']:.3f} for `months_late`), because a refit "
   "lets the other lateness column take over. Keep `full` and `drop_loss`: G5 draws "
   "them.")

ex("G5", "Two importances, side by side", 2,
   "Compute the permutation importance of `full` from G4 on the test rows, ten "
   "repeats, as a Series. Draw it and `drop_loss` from G4 as two horizontal bar "
   "charts side by side, each sorted, with a title on each.",
   "perm = ...\n\nfig, axes = plt.subplots(1, 2, figsize=(10, 3.5))\n...\n...\nplt.show()",
   ["`perm = pd.Series(permutation_importance(full, c_test[c_columns], c_test['default'], "
    "scoring='roc_auc', n_repeats=10, random_state=0).importances_mean, index=c_columns)`.",
    "`s = perm.sort_values()` and `axes[0].barh(s.index, s.values)` draws one chart "
    "with the largest at the top; the same for `pd.Series(drop_loss)` on `axes[1]`."],
   "perm = pd.Series(permutation_importance(full, c_test[c_columns], c_test['default'], scoring='roc_auc',\n"
   "                                        n_repeats=10, random_state=0).importances_mean, index=c_columns)\n\n"
   "fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))\n"
   "for ax, s, title in [(axes[0], perm, 'AUC lost when shuffled'),\n"
   "                     (axes[1], pd.Series(drop_loss), 'AUC lost by a refit without it')]:\n"
   "    s = s.sort_values()\n    ax.barh(s.index, s.values, color='#1c5cab')\n    ax.set_title(title, loc='left')\n"
   "fig.tight_layout()\nplt.show()",
   "Both rank `months_late` first and `late_now` second, and both put `age` last. "
   f"The scales differ by a factor of about {_words(PERM['months_late'] / G4['months_late'])}: "
   "shuffling breaks a column for a model that relies on it, and a refit lets the "
   "model rely on something else.")

ex("G6", "Groups of columns", 3,
   "Shuffle groups of columns together, with one permutation of the rows for the "
   "whole group, ten times each: the payment history (`late_now`, `months_late`), "
   "the amounts (`limit`, `bill`, `paid`, `utilisation`) and `age`. Print the mean AUC "
   "lost by XGBoost (100 trees of depth 3) for each group.",
   "xgb_c = ...\n...\nbase_auc = ...\ngroups = {\n    'payment history': ['late_now', 'months_late'],\n"
   "    'amounts': ['limit', 'bill', 'paid', 'utilisation'],\n    'age': ['age'],\n}\n\n"
   "group_loss = {}\nfor name in groups:\n    ...\n\nprint(group_loss)",
   ["`order = np.random.default_rng(k).permutation(len(c_test))`, then "
    "`shuffled[groups[name]] = c_test[groups[name]].values[order]` moves every column "
    "of the group with the same order.",
    "An inner loop over k from 0 to 9 collects the ten losses; store their mean in "
    "`group_loss[name]`."],
   "xgb_c = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)\n"
   "xgb_c.fit(c_train[c_columns], c_train['default'])\n"
   "base_auc = roc_auc_score(c_test['default'], xgb_c.predict_proba(c_test[c_columns])[:, 1])\n"
   "groups = {\n    'payment history': ['late_now', 'months_late'],\n"
   "    'amounts': ['limit', 'bill', 'paid', 'utilisation'],\n    'age': ['age'],\n}\n\n"
   "group_loss = {}\nfor name in groups:\n    losses = []\n    for k in range(10):\n"
   "        shuffled = c_test.copy()\n        order = np.random.default_rng(k).permutation(len(c_test))\n"
   "        shuffled[groups[name]] = c_test[groups[name]].values[order]\n"
   "        auc = roc_auc_score(c_test['default'], xgb_c.predict_proba(shuffled[c_columns])[:, 1])\n"
   "        losses.append(base_auc - auc)\n    group_loss[name] = round(float(np.mean(losses)), 4)\n\n"
   "print(group_loss)",
   f"The payment history together costs {G6['payment history']:.3f}, the amounts "
   f"{G6['amounts']:.3f} and `age` {G6['age']:.3f}. Shuffled one at a time, the two "
   f"lateness columns cost {PERM['months_late']:.3f} and {PERM['late_now']:.3f}, less "
   "than the pair together: with one shuffled, the other still carries much of the "
   "same information. Shuffled together, nothing is left of it.")

md("---")

# ====================================================== H
_h1_order = sorted(H1, key=lambda k: H1[k][1])
section(
"## H · Comparing models\n\n"
"Every classifier of the course on the credit table, timed; how many rows "
"boosting needs; and two comparisons on the index table. H1 and H2, and H3 and "
"H4, run together."
)

ex("H1", "Eight classifiers, one function", 3,
   "Write `evaluate(model)`: it fits the model on the credit training rows, times "
   "the fit, and returns the seconds and the test AUC. Run it on the eight models "
   "below and collect the results in a DataFrame `board`, one row per model, sorted "
   "by AUC. The forest and scikit-learn's `GradientBoosting` take a few seconds each.",
   "models = {\n"
   "    'logistic regression': Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())]),\n"
   "    'k-nearest neighbours': Pipeline([('scale', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=25))]),\n"
   "    'tree': DecisionTreeClassifier(max_depth=5, random_state=0),\n"
   "    'random forest': RandomForestClassifier(n_estimators=100, min_samples_leaf=100, random_state=0),\n"
   "    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0),\n"
   "    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=0),\n"
   "    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0, n_jobs=1),\n"
   "    'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=8, random_state=0,\n"
   "                               n_jobs=1, verbose=-1),\n}\n\n\n"
   "def evaluate(model):\n    \"\"\"...\"\"\"\n    ...\n\n\nrows = []\nfor name in models:\n    ...\n\nboard = ...\nprint(board)",
   ["Inside `evaluate`: `start = time.perf_counter()`, fit, `seconds = "
    "time.perf_counter() - start`, then `return seconds, roc_auc_score(...)`.",
    "In the loop: `seconds, auc = evaluate(models[name])` and `rows.append({'model': "
    "name, 'seconds': round(seconds, 3), 'auc': round(auc, 4)})`. Then "
    "`pd.DataFrame(rows).set_index('model').sort_values('auc')`."],
   "models = {\n"
   "    'logistic regression': Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())]),\n"
   "    'k-nearest neighbours': Pipeline([('scale', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=25))]),\n"
   "    'tree': DecisionTreeClassifier(max_depth=5, random_state=0),\n"
   "    'random forest': RandomForestClassifier(n_estimators=100, min_samples_leaf=100, random_state=0),\n"
   "    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0),\n"
   "    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=0),\n"
   "    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0, n_jobs=1),\n"
   "    'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=8, random_state=0,\n"
   "                               n_jobs=1, verbose=-1),\n}\n\n\n"
   "def evaluate(model):\n    \"\"\"Fit on the credit training rows; return the seconds and the test AUC.\"\"\"\n"
   "    start = time.perf_counter()\n    model.fit(c_train[c_columns], c_train['default'])\n"
   "    seconds = time.perf_counter() - start\n"
   "    return seconds, roc_auc_score(c_test['default'], model.predict_proba(c_test[c_columns])[:, 1])\n\n\n"
   "rows = []\nfor name in models:\n    seconds, auc = evaluate(models[name])\n"
   "    rows.append({'model': name, 'seconds': round(seconds, 3), 'auc': round(auc, 4)})\n\n"
   "board = pd.DataFrame(rows).set_index('model').sort_values('auc')\nprint(board)",
   f"Logistic regression {H1['logistic regression'][1]:.3f} and k-nearest neighbours "
   f"{H1['k-nearest neighbours'][1]:.3f}, the scores of Session 9; a tree of depth 5 "
   f"{H1['tree'][1]:.3f}; the forest {H1['random forest'][1]:.3f}; the four boosting "
   f"implementations {min(_boost_aucs):.3f} to {max(_boost_aucs):.3f}. Here the forest "
   f"took {H1['random forest'][0]:.1f} seconds and `GradientBoosting` "
   f"{H1['GradientBoosting'][0]:.1f}, while the three binned implementations took "
   f"{_binned:.2f} seconds or less for the same score. Keep `board`: H2 draws it.")

ex("H2", "AUC against time", 2,
   "Draw `board` from H1 as a scatter of AUC against seconds, with the x-axis on a "
   "log scale and each point labelled with its model's name.",
   "fig, ax = plt.subplots(figsize=(8, 4))\n...\n...\nplt.show()",
   ["`ax.scatter(board['seconds'], board['auc'])` and `ax.set_xscale('log')`.",
    "`for name in board.index: ax.annotate(name, (board.loc[name, 'seconds'], "
    "board.loc[name, 'auc']))` writes the labels."],
   "fig, ax = plt.subplots(figsize=(8, 4))\nax.scatter(board['seconds'], board['auc'], color='#1c5cab')\n"
   "for name in board.index:\n    ax.annotate(name, (board.loc[name, 'seconds'], board.loc[name, 'auc']),\n"
   "                textcoords='offset points', xytext=(5, 4), fontsize=9)\n"
   "ax.set_xscale('log')\nax.set_xlabel('seconds to fit, one core')\nax.set_ylabel('test AUC')\n"
   "ax.set_title('Eight classifiers on the credit table', loc='left')\nplt.show()",
   "The top left is the place to be, and the binned boosting libraries sit there: as "
   "accurate as the forest in a fraction of its time. Their labels overlap, which is "
   "the point. Logistic regression is the fastest of all and about 0.03 behind.")

ex("H3", "How many rows boosting needs", 3,
   "Split a stratified fifth off the credit training rows as validation rows. Fit "
   "logistic regression (scaled) and XGBoost (100 trees of depth 3) on the first n "
   "of the remaining rows, for n in `[250, 500, 1000, 2000, 4000, 8000, 16800]`, and "
   "store both validation AUCs for each n in a DataFrame `curve` with columns "
   "`logistic` and `xgboost`.",
   "fit_c = ...\nval_c = ...\n\nrows = []\nfor n in [250, 500, 1000, 2000, 4000, 8000, 16800]:\n    ...\n\ncurve = ...\nprint(curve)",
   ["`fit_c, val_c = train_test_split(c_train, test_size=0.2, random_state=0, "
    "stratify=c_train['default'])`, then `fit_c.iloc[:n]` for the first n rows.",
    "Append `{'n': n, 'logistic': ..., 'xgboost': ...}` for each n, and "
    "`pd.DataFrame(rows).set_index('n')`."],
   "fit_c, val_c = train_test_split(c_train, test_size=0.2, random_state=0, stratify=c_train['default'])\n\n"
   "rows = []\nfor n in [250, 500, 1000, 2000, 4000, 8000, 16800]:\n    part = fit_c.iloc[:n]\n"
   "    logit = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "    logit.fit(part[c_columns], part['default'])\n"
   "    xgb_c = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)\n"
   "    xgb_c.fit(part[c_columns], part['default'])\n"
   "    rows.append({'n': n,\n                 'logistic': roc_auc_score(val_c['default'], logit.predict_proba(val_c[c_columns])[:, 1]),\n"
   "                 'xgboost': roc_auc_score(val_c['default'], xgb_c.predict_proba(val_c[c_columns])[:, 1])})\n\n"
   "curve = pd.DataFrame(rows).set_index('n')\nprint(curve.round(4))",
   f"Logistic regression is at {H3[250][0]:.3f} with 250 borrowers and never passes "
   f"{max(v[0] for v in H3.values()):.3f}. XGBoost starts below it, at {H3[250][1]:.3f}, "
   f"passes it between 2,000 and 4,000 borrowers, and reaches {H3[16800][1]:.3f} on "
   "all 16,800. A flexible model needs rows to pay for its flexibility; with a few "
   "hundred, logistic regression is the better choice. Keep `curve`: H4 draws it.")

ex("H4", "Draw the two curves", 2,
   "Draw both columns of `curve` from H3 against n on a log x-axis, with markers, "
   "axis labels and a legend.",
   "fig, ax = plt.subplots(figsize=(7, 3.5))\n...\n...\nplt.show()",
   ["`ax.plot(curve.index, curve['logistic'], marker='o', label='logistic regression')`, "
    "and the same for `curve['xgboost']`.",
    "`ax.set_xscale('log')` and `ax.legend()`."],
   "fig, ax = plt.subplots(figsize=(7, 3.5))\n"
   "ax.plot(curve.index, curve['logistic'], marker='o', label='logistic regression')\n"
   "ax.plot(curve.index, curve['xgboost'], marker='o', label='XGBoost')\n"
   "ax.set_xscale('log')\nax.set_xlabel('training borrowers')\nax.set_ylabel('validation AUC')\nax.legend()\n"
   "ax.set_title('More rows help the flexible model more', loc='left')\nplt.show()",
   "The lines cross once. To the left of the crossing the simpler model wins; to "
   "the right, boosting does, and its line is still rising at the end.")

ex("H5", "Fold by fold", 3,
   "On the index table, compute the five fold RMSEs of ridge (a scaled pipeline "
   "with an alpha of 1000) and of 25 boosted stumps at a learning rate of 0.1, with "
   "`cross_val_score` on `folds`. Count with `zip` the folds where boosting has the "
   "lower RMSE, and print both lists.",
   "ridge_folds = ...\nboost_folds = ...\nboost_wins = ...\n\nprint(...)\nprint(...)\nprint(boost_wins)",
   ["`-cross_val_score(Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1000))]), "
    "train[columns], train['vol_next'], cv=folds, scoring='neg_root_mean_squared_error')` "
    "gives the five RMSEs.",
    "`sum(1 for r, b in zip(ridge_folds, boost_folds) if b < r)`."],
   "ridge_folds = -cross_val_score(Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1000))]),\n"
   "                               train[columns], train['vol_next'], cv=folds,\n"
   "                               scoring='neg_root_mean_squared_error')\n"
   "boost_folds = -cross_val_score(GradientBoostingRegressor(n_estimators=25, learning_rate=0.1, max_depth=1,\n"
   "                                                         random_state=0),\n"
   "                               train[columns], train['vol_next'], cv=folds,\n"
   "                               scoring='neg_root_mean_squared_error')\n"
   "boost_wins = sum(1 for r, b in zip(ridge_folds, boost_folds) if b < r)\n\n"
   "print(ridge_folds.round(3))\nprint(boost_folds.round(3))\nprint(boost_wins)",
   f"Ridge wins all five: {', '.join(f'{v:.3f}' for v in H5_RIDGE)} against "
   f"{', '.join(f'{v:.3f}' for v in H5_BOOST)}, and `boost_wins` is 0. Boosting does "
   "not lose on average because of one bad fold; it loses on every one, including "
   "fold 3, which holds 2020.")

ex("H6", "A forecast used as a ranking", 4,
   "On the index table, call a day busy when `vol_next` is above the median of the "
   "training days. Fit 219 boosted stumps at a learning rate of 0.1, the number "
   "2022 chose in C4, on `vol_next` itself, and `XGBClassifier` with 100 stumps on "
   "the busy label. Compute the test AUC of the regression's forecast as a score "
   "for the label, of the classifier's probability, and of `vol_20d` itself.",
   "median = ...\nbusy_train = ...\nbusy_test = ...\n\nregression = ...\n...\nclassifier = ...\n...\n\n"
   "print('forecast  :', ...)\nprint('classifier:', ...)\nprint('vol_20d   :', ...)",
   ["`busy_train = (train['vol_next'] > median).astype(int)`, and the same on the test "
    "days with the training median.",
    "`roc_auc_score` takes any score that ranks the days: "
    "`roc_auc_score(busy_test, regression.predict(test[columns]))`."],
   "median = train['vol_next'].median()\nbusy_train = (train['vol_next'] > median).astype(int)\n"
   "busy_test = (test['vol_next'] > median).astype(int)\n\n"
   "regression = GradientBoostingRegressor(n_estimators=219, learning_rate=0.1, max_depth=1, random_state=0)\n"
   "regression.fit(train[columns], train['vol_next'])\n"
   "classifier = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)\n"
   "classifier.fit(train[columns], busy_train)\n\n"
   "print('forecast  :', roc_auc_score(busy_test, regression.predict(test[columns])))\n"
   "print('classifier:', roc_auc_score(busy_test, classifier.predict_proba(test[columns])[:, 1]))\n"
   "print('vol_20d   :', roc_auc_score(busy_test, test['vol_20d']))",
   f"The forecast of the number scores {H6['forecast']:.3f}, the classifier "
   f"{H6['classifier']:.3f} and `vol_20d` alone {H6['vol_20d']:.3f}. The classifier "
   "only ever saw whether a day was above the median; the regression saw by how "
   "much, which is more information. Here, modelling the number and ranking by its "
   "forecast beat modelling the label made from it.")

md("---")

# ====================================================== I
section(
"## I · When it goes wrong\n\n"
"Five mistakes that come with boosting. Four stop with an error, and one runs "
"quietly with a forecast that is slightly off. In each, the first cell shows the "
"mistake and the second is for your fix."
)

ex_fix("I1", "Early stopping with nothing to watch", 1,
   "The cell below asks XGBoost to stop early and raises an error. Read its last "
   "line. Then fit the model in the second cell so that it stops on 2022, and print "
   "`best_iteration`.",
   "xgb_bad = XGBRegressor(n_estimators=3000, learning_rate=0.1, max_depth=1, early_stopping_rounds=50,\n"
   "                       random_state=0)\nxgb_bad.fit(fit_rows[columns], fit_rows['vol_next'])",
   "xgb_good = ...\n...\n\nprint(...)",
   ["Early stopping needs rows to score each round on.",
    "`eval_set=[(stop_rows[columns], stop_rows['vol_next'])]` in `fit`, with "
    "`verbose=False` to keep the output short."],
   "xgb_good = XGBRegressor(n_estimators=3000, learning_rate=0.1, max_depth=1, early_stopping_rounds=50,\n"
   "                        random_state=0)\n"
   "xgb_good.fit(fit_rows[columns], fit_rows['vol_next'],\n"
   "             eval_set=[(stop_rows[columns], stop_rows['vol_next'])], verbose=False)\n\n"
   "print(xgb_good.best_iteration)",
   f"`ValueError: Must have at least 1 validation dataset for early stopping`, then "
   f"{XS_BEST}. Without an `eval_set` there is nothing to stop on.")

ex_fix("I2", "One tree short", 2,
   "The cell below rebuilds the stopped model with a fixed number of trees, and its "
   "`assert` fails. Find the mistake and fix it in the second cell, with the "
   "`assert` passing.",
   "xgb_stop = XGBRegressor(n_estimators=3000, learning_rate=0.1, max_depth=1, early_stopping_rounds=50,\n"
   "                        random_state=0)\n"
   "xgb_stop.fit(fit_rows[columns], fit_rows['vol_next'],\n"
   "             eval_set=[(stop_rows[columns], stop_rows['vol_next'])], verbose=False)\n"
   "refit = XGBRegressor(n_estimators=xgb_stop.best_iteration, learning_rate=0.1, max_depth=1, random_state=0)\n"
   "refit.fit(fit_rows[columns], fit_rows['vol_next'])\n"
   "assert np.allclose(refit.predict(test[columns]), xgb_stop.predict(test[columns]))",
   "refit = ...\n...\n...\n\nprint(...)",
   "`best_iteration` counts rounds from 0. How many trees does round 243 mean?",
   "refit = XGBRegressor(n_estimators=xgb_stop.best_iteration + 1, learning_rate=0.1, max_depth=1,\n"
   "                     random_state=0)\nrefit.fit(fit_rows[columns], fit_rows['vol_next'])\n"
   "assert np.allclose(refit.predict(test[columns]), xgb_stop.predict(test[columns]))\n\n"
   "print(rmse(test['vol_next'], refit.predict(test[columns])))",
   f"`AssertionError`. Round {XS_BEST} is the {XS_BEST + 1}th tree, so the refit "
   f"needs `best_iteration + 1` trees. One tree short, the forecasts differ by up to "
   f"{I2_DIFF:.4f} and the test RMSE is {I2_SHORT:.5f} instead of {I2_RIGHT:.5f}. "
   "Nothing warned; only the `assert` noticed.")

ex_fix("I3", "No importances to read", 2,
   "`HistGradientBoostingRegressor` is scikit-learn's binned boosting. The cell "
   "below asks it for importances and raises an error. In the second cell, measure "
   "its importance on the test days with `permutation_importance` instead, and print "
   "the three columns it loses most without.",
   "hist = HistGradientBoostingRegressor(max_iter=200, max_depth=2, random_state=0)\n"
   "hist.fit(train[columns], train['vol_next'])\nprint(hist.feature_importances_)",
   "hist = ...\n...\nresult = ...\nimportance = ...\n\nprint(...)",
   ["Refit `hist` as in the cell above. Then `permutation_importance(hist, test[columns], "
    "test['vol_next'], scoring='neg_root_mean_squared_error', n_repeats=10, random_state=0)`.",
    "`pd.Series(result.importances_mean, index=columns).nlargest(3)`."],
   "hist = HistGradientBoostingRegressor(max_iter=200, max_depth=2, random_state=0)\n"
   "hist.fit(train[columns], train['vol_next'])\n"
   "result = permutation_importance(hist, test[columns], test['vol_next'],\n"
   "                                scoring='neg_root_mean_squared_error', n_repeats=10, random_state=0)\n"
   "importance = pd.Series(result.importances_mean, index=columns)\n\nprint(importance.nlargest(3).round(4))",
   "`AttributeError`: the binned models of scikit-learn have no "
   f"`feature_importances_`. Shuffled on the test days, the model loses most without "
   f"`{I3_TOP.index[0]}` ({I3_TOP.iloc[0]:.4f} of RMSE), then `{I3_TOP.index[1]}` and "
   f"`{I3_TOP.index[2]}`. Permutation importance works for any model that predicts, "
   "which is one more reason to prefer it.")

ex_fix("I4", "A label written as words", 2,
   "The cell below fits XGBoost on a label of words, `'jump'` when `vol_next` is "
   "more than 1.5 times `vol_20d` and `'quiet'` otherwise, and raises an error. Read "
   "what it expected. Then fit the same model in the second cell on the same label "
   "as 0 and 1, with 1 for a jump, and print its test AUC.",
   "words = np.where(train['vol_next'] > 1.5 * train['vol_20d'], 'jump', 'quiet')\n"
   "xgb_words = XGBClassifier(n_estimators=100, max_depth=1, random_state=0)\n"
   "xgb_words.fit(train[columns], words)",
   "jump_train = ...\njump_test = ...\nxgb_jump = ...\n...\n\nprint(...)",
   ["`(train['vol_next'] > 1.5 * train['vol_20d']).astype(int)` is 1 for a jump; the "
    "same on `test`."],
   "jump_train = (train['vol_next'] > 1.5 * train['vol_20d']).astype(int)\n"
   "jump_test = (test['vol_next'] > 1.5 * test['vol_20d']).astype(int)\n"
   "xgb_jump = XGBClassifier(n_estimators=100, max_depth=1, random_state=0)\nxgb_jump.fit(train[columns], jump_train)\n\n"
   "print(roc_auc_score(jump_test, xgb_jump.predict_proba(test[columns])[:, 1]))",
   "`ValueError: Invalid classes inferred from unique values of y. Expected: [0 1], "
   "got ['jump' 'quiet']`. scikit-learn's own classifiers accept words; XGBoost wants "
   f"the classes numbered from 0. With numbers, the test AUC is {I4_AUC:.3f}, on "
   f"{I4_N_TE} jumps among {N_TEST} test days.")

ex_fix("I5", "A generator is not a list", 2,
   "The cell below wants the test forecast after 50 stumps and raises an error. "
   "Fix it in the second cell, and print the test RMSE after 50 stumps.",
   "boost = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)\n"
   "boost.fit(train[columns], train['vol_next'])\nafter_50 = boost.staged_predict(test[columns])[49]",
   "boost = ...\n...\nafter_50 = ...\n\nprint(...)",
   "`staged_predict` hands out one forecast at a time, in order. `list(...)` "
   "collects them all, and then `[49]` is the fiftieth.",
   "boost = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)\n"
   "boost.fit(train[columns], train['vol_next'])\nafter_50 = list(boost.staged_predict(test[columns]))[49]\n\n"
   "print(rmse(test['vol_next'], after_50))",
   "`TypeError: 'generator' object is not subscriptable`. A generator produces each "
   "stage only when asked, which saves memory on long runs; `list()` stores them "
   f"all. After 50 stumps the test RMSE is {I5_RMSE:.3f}.")

md("---")

# ====================================================== J
section(
"## J · A target with nothing in it\n\n"
"Whether the index rises tomorrow, from its last five daily returns. Daily "
"returns are close to unpredictable, which makes this a clean test of what a "
"flexible model does when there is nothing to find. J2 to J6 use the table from "
"J1."
)

ex("J1", "Five past returns and tomorrow's direction", 3,
   "From `rets['SPY']`, the index's daily returns in percent, build a DataFrame "
   "`direction` with `lag_0` (today's return) to `lag_4` (the return four days "
   "earlier), made with `shift` in a loop, and `up_next`: 1 when tomorrow's return is "
   "positive. Drop the first five rows and the last one, which are incomplete, and "
   "split at the end of 2022 into `d_train` and `d_test`. Print the share of up days "
   "in each.",
   "spy = rets['SPY']\ndirection = pd.DataFrame()\nfor k in range(5):\n    ...\n...\n...\n\n"
   "d_train = ...\nd_test = ...\nlags = ['lag_0', 'lag_1', 'lag_2', 'lag_3', 'lag_4']\n\nprint(...)",
   ["`direction['lag_' + str(k)] = spy.shift(k)` in the loop. Tomorrow is "
    "`spy.shift(-1)`, so `direction['up_next'] = (spy.shift(-1) > 0).astype(int)`.",
    "`direction = direction.iloc[5:-1]` drops the incomplete rows; then "
    "`.loc[:'2022-12-31']` and `.loc['2023-01-01':]`."],
   "spy = rets['SPY']\ndirection = pd.DataFrame()\nfor k in range(5):\n    direction['lag_' + str(k)] = spy.shift(k)\n"
   "direction['up_next'] = (spy.shift(-1) > 0).astype(int)\ndirection = direction.iloc[5:-1]\n\n"
   "d_train = direction.loc[:'2022-12-31']\nd_test = direction.loc['2023-01-01':]\n"
   "lags = ['lag_0', 'lag_1', 'lag_2', 'lag_3', 'lag_4']\n\n"
   "print(len(d_train), len(d_test), round(d_train['up_next'].mean(), 3), round(d_test['up_next'].mean(), 3))",
   f"{len(D_TRAIN):,} training days and {len(D_TEST)} test days; "
   f"{100 * J1_SHARES[0]:.1f} percent of the training days and "
   f"{100 * J1_SHARES[1]:.1f} percent of the test days were followed by a rise. "
   "Every column is known at the close, and the label is the next day's move: "
   "nothing in the table looks ahead.")

ex("J2", "Logistic regression first", 2,
   "Fit a scaled logistic regression of `up_next` on the five lags, and print its "
   "AUC on the training days and on the test days.",
   "logit = ...\n...\n\nprint('train:', ...)\nprint('test :', ...)",
   "`Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])`, fitted "
   "on `d_train[lags]` and `d_train['up_next']`.",
   "logit = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "logit.fit(d_train[lags], d_train['up_next'])\n\n"
   "print('train:', roc_auc_score(d_train['up_next'], logit.predict_proba(d_train[lags])[:, 1]))\n"
   "print('test :', roc_auc_score(d_test['up_next'], logit.predict_proba(d_test[lags])[:, 1]))",
   f"{J2[0]:.3f} on the training days and {J2[1]:.3f} on the test days, below the 0.5 "
   "of a coin. With five columns a straight line cannot fit much noise, and there "
   "was little else to fit.")

ex("J3", "Three hundred deeper trees", 2,
   "Fit `XGBClassifier` with 300 trees of depth 4 at a learning rate of 0.1 on the "
   "same columns, and print both AUCs.",
   "xgb_dir = ...\n...\n\nprint('train:', ...)\nprint('test :', ...)",
   "`XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, random_state=0)`.",
   "xgb_dir = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, random_state=0)\n"
   "xgb_dir.fit(d_train[lags], d_train['up_next'])\n\n"
   "print('train:', roc_auc_score(d_train['up_next'], xgb_dir.predict_proba(d_train[lags])[:, 1]))\n"
   "print('test :', roc_auc_score(d_test['up_next'], xgb_dir.predict_proba(d_test[lags])[:, 1]))",
   f"{J3[0]:.3f} on the training days and {J3[1]:.3f} on the test days. The trees "
   f"found patterns in {len(D_TRAIN):,} days that rank them almost perfectly, and "
   "none of it carried over.")

ex("J4", "What early stopping says", 3,
   "Fit the same model on the days up to the end of 2021 and stop it on 2022, with "
   "`eval_metric='auc'` and `early_stopping_rounds=50`, from up to 1,000 trees. Print "
   "`best_iteration`, the best 2022 AUC and the test AUC.",
   "d_fit = ...\nd_stop = ...\n\nxgb_dir = ...\n...\n\nprint(...)",
   ["`d_train.loc[:'2021-12-31']` and `d_train.loc['2022-01-01':]`.",
    "`XGBClassifier(n_estimators=1000, learning_rate=0.1, max_depth=4, "
    "early_stopping_rounds=50, eval_metric='auc', random_state=0)`, with "
    "`eval_set=[(d_stop[lags], d_stop['up_next'])]`."],
   "d_fit = d_train.loc[:'2021-12-31']\nd_stop = d_train.loc['2022-01-01':]\n\n"
   "xgb_dir = XGBClassifier(n_estimators=1000, learning_rate=0.1, max_depth=4,\n"
   "                        early_stopping_rounds=50, eval_metric='auc', random_state=0)\n"
   "xgb_dir.fit(d_fit[lags], d_fit['up_next'], eval_set=[(d_stop[lags], d_stop['up_next'])], verbose=False)\n\n"
   "print(xgb_dir.best_iteration, round(xgb_dir.best_score, 4),\n"
   "      round(roc_auc_score(d_test['up_next'], xgb_dir.predict_proba(d_test[lags])[:, 1]), 4))",
   f"`best_iteration` is {J4[0]}, so the model keeps {J4[0] + 1} trees, with a 2022 AUC "
   f"of {J4[1]:.3f} and a test AUC of {J4[2]:.3f}. Early stopping cannot turn noise "
   f"into signal, but it stops the fitting of it: {J4[0] + 1} trees instead of 300. "
   f"Random scores on these {len(D_TEST)} test days give AUCs with a standard "
   f"deviation of {J4_CHANCE_SD:.3f}, so {J4[2]:.3f} is less than two standard "
   "deviations from a coin's 0.5.")

ex("J5", "A label shuffled on purpose", 4,
   "Shuffle `up_next` in a copy of `d_train` with `np.random.default_rng(0).permutation`, "
   "so that no column can carry any information about it. Fit J3's model on the "
   "shuffled label, and print its AUC on the shuffled training days and on the real "
   "test days.",
   "shuffled = ...\n...\nxgb_noise = ...\n...\n\nprint('train:', ...)\nprint('test :', ...)",
   ["`shuffled = d_train.copy()`, then `shuffled['up_next'] = "
    "np.random.default_rng(0).permutation(shuffled['up_next'].values)`.",
    "Score the training AUC against the shuffled label, and the test AUC against "
    "`d_test['up_next']`."],
   "shuffled = d_train.copy()\nshuffled['up_next'] = np.random.default_rng(0).permutation(shuffled['up_next'].values)\n"
   "xgb_noise = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, random_state=0)\n"
   "xgb_noise.fit(shuffled[lags], shuffled['up_next'])\n\n"
   "print('train:', roc_auc_score(shuffled['up_next'], xgb_noise.predict_proba(shuffled[lags])[:, 1]))\n"
   "print('test :', roc_auc_score(d_test['up_next'], xgb_noise.predict_proba(d_test[lags])[:, 1]))",
   f"{J5[0]:.3f} on the training days, against {J3[0]:.3f} with the real label, and "
   f"{J5[1]:.3f} on the test days. A model this flexible fits a label it cannot "
   "possibly predict almost as well as the real one. A high training score shows "
   "capacity, not signal.")

ex("J6", "Both AUCs, tree by tree", 2,
   "Fit 300 trees of depth 4 on `d_fit` from J4, with `eval_metric='auc'` and two "
   "`eval_set` entries, the fitting days first and 2022 second, and no early "
   "stopping. Draw both AUC paths from `evals_result()` against the number of trees, "
   "with a dashed line at 0.5 and a legend.",
   "xgb_paths = ...\n...\npaths = ...\n\nfig, ax = plt.subplots(figsize=(8, 3))\n...\n...\nplt.show()",
   ["`evals_result()['validation_0']['auc']` belongs to the first entry of `eval_set`, "
    "the fitting days, and `['validation_1']['auc']` to the second.",
    "`ax.axhline(0.5, linestyle='--', color='grey')`."],
   "xgb_paths = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, eval_metric='auc', random_state=0)\n"
   "xgb_paths.fit(d_fit[lags], d_fit['up_next'],\n"
   "              eval_set=[(d_fit[lags], d_fit['up_next']), (d_stop[lags], d_stop['up_next'])], verbose=False)\n"
   "paths = xgb_paths.evals_result()\n\nfig, ax = plt.subplots(figsize=(8, 3))\n"
   "ax.plot(range(1, 301), paths['validation_0']['auc'], label='fitting days, 2015 to 2021')\n"
   "ax.plot(range(1, 301), paths['validation_1']['auc'], label='2022')\n"
   "ax.axhline(0.5, linestyle='--', color='grey')\nax.set_xlabel('trees')\nax.set_ylabel('AUC')\nax.legend()\n"
   "ax.set_title('Tomorrow up or down: what the trees learn', loc='left')\nplt.show()",
   f"The fitting days climb from {J6_FIT[0]:.3f} to {J6_FIT[-1]:.3f} while 2022 drifts "
   f"down from {J6_STOP[0]:.3f} to {J6_STOP[-1]:.3f}. When the two lines move apart "
   "from the first trees on, the trees are fitting noise.")

md("---")

# ====================================================== K
_k2_means = ", ".join(f"{v:.3f}" for v in K2_MEANS.values)
_k4_df = pd.DataFrame(K4, index=[1, 5, 25, 100, 500])
section(
"## K · Five small cases\n\n"
"Each of these stands on its own and needs nothing from earlier in the notebook "
"beyond the setup cell. They keep functions, loops, dictionaries and NumPy in "
"working order, in this session's setting."
)

ex("K1", "A boosted model as a sum", 2,
   "The first three stumps that boosting fits to the six days at a learning rate of "
   "0.5 are below, each as (cut, value at or below the cut, value above it). The "
   "model starts from 0.95. Write `predict(x)`, which adds 0.5 times each stump's "
   "value to 0.95 for one day, and print the forecast for each of the six days.",
   f"stumps = {K1_STUMPS}\ndays = [0.4, 0.6, 0.8, 1.2, 1.6, 2.4]\n\n\n"
   "def predict(x):\n    \"\"\"...\"\"\"\n    ...\n\n\nfor x in days:\n    print(x, ...)",
   ["Start `total = 0.95`, loop `for cut, left, right in stumps:`, and add "
    "`0.5 * left` if `x <= cut`, otherwise `0.5 * right`.",
    "Return `total`, and print `round(predict(x), 4)`."],
   f"stumps = {K1_STUMPS}\ndays = [0.4, 0.6, 0.8, 1.2, 1.6, 2.4]\n\n\n"
   "def predict(x):\n    \"\"\"The forecast of three boosted stumps for one day.\"\"\"\n    total = 0.95\n"
   "    for cut, left, right in stumps:\n        if x <= cut:\n            total = total + 0.5 * left\n"
   "        else:\n            total = total + 0.5 * right\n    return total\n\n\n"
   "for x in days:\n    print(x, round(predict(x), 4))",
   f"{K1_OUT[0]} for the three calm days, {K1_OUT[3]} for 1.2 and {K1_OUT[4]} for 1.6 "
   "and 2.4: the forecast after A4's third step. The third stump cut at 1.4, so it "
   "was the first to split the busy side.")

ex("K2", "Bins, by hand", 3,
   "XGBoost and LightGBM cut a column only at bin edges set by its quantiles. Make "
   "four bins of `vol_20d` on the training days of the index table by hand: compute "
   "the 25th, 50th and 75th percentiles with `np.quantile`, then write "
   "`bin_of(value, edges)` that returns 0, 1, 2 or 3 with `if` and `elif`. Count the "
   "days in each bin, and print the mean `vol_next` of each bin.",
   "values = train['vol_20d']\nedges = ...\n\n\ndef bin_of(value, edges):\n    \"\"\"...\"\"\"\n    ...\n\n\n"
   "bins = []\nfor v in values:\n    ...\n\ncounts = ...\nmeans = ...\n\nprint(edges)\nprint(counts)\nprint(means)",
   ["`edges = np.quantile(values, [0.25, 0.5, 0.75])`. In `bin_of`: `if value <= "
    "edges[0]: return 0`, then `elif value <= edges[1]: return 1`, and so on.",
    "`pd.Series(bins).value_counts().sort_index()` counts them, and "
    "`train['vol_next'].groupby(np.array(bins)).mean()` gives the means."],
   "values = train['vol_20d']\nedges = np.quantile(values, [0.25, 0.5, 0.75])\n\n\n"
   "def bin_of(value, edges):\n    \"\"\"The bin, 0 to 3, that a value falls in.\"\"\"\n"
   "    if value <= edges[0]:\n        return 0\n    elif value <= edges[1]:\n        return 1\n"
   "    elif value <= edges[2]:\n        return 2\n    return 3\n\n\n"
   "bins = []\nfor v in values:\n    bins.append(bin_of(v, edges))\n\n"
   "counts = pd.Series(bins).value_counts().sort_index()\nmeans = train['vol_next'].groupby(np.array(bins)).mean()\n\n"
   "print(edges.round(3))\nprint(counts)\nprint(means.round(3))",
   f"Edges at {K2_EDGES[0]:.3f}, {K2_EDGES[1]:.3f} and {K2_EDGES[2]:.3f} percent, "
   f"{K2_COUNTS.min()} or {K2_COUNTS.max()} days in each bin, and a mean `vol_next` "
   f"of {_k2_means}: next month rises with this month at every step. XGBoost does "
   "the same with 256 bins per column, which is why it only ever tries 255 cuts per "
   "column, however many days there are.")

ex("K3", "XGBoost's gain of a cut", 4,
   "XGBoost scores a cut with "
   "$G_L^2/(H_L+\\lambda) + G_R^2/(H_R+\\lambda) - (G_L+G_R)^2/(H_L+H_R+\\lambda)$, where "
   "$G$ is the sum of the gradients $f - y$ in a box and $H$, for squared error, its "
   "number of days. (The library also halves it, which changes no ranking.) On the "
   "six days, with the forecast at their mean and $\\lambda = 1$, compute the gain of "
   "each cut after 0.4, 0.6, 0.8, 1.2 and 1.6 with a loop, store it in a dictionary, "
   "and print the best cut. Then do the same with $\\lambda = 0$ and compare the best "
   "gain with the fall in RSS that cut gives, from 0.815 to 0.080.",
   "x = np.array([0.4, 0.6, 0.8, 1.2, 1.6, 2.4])\ny = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\ng = ...\n\n"
   "for lam in [1, 0]:\n    gains = {}\n    for cut in [0.4, 0.6, 0.8, 1.2, 1.6]:\n        ...\n"
   "    print(lam, gains)\n    print('best:', ...)",
   ["`g = y.mean() - y`, the gradient of ½(y − f)² at the mean. For a cut, "
    "`left = x <= cut`, `GL = g[left].sum()`, `HL = left.sum()`, and the same with "
    "`~left`.",
    "`gains[cut] = round(GL**2 / (HL + lam) + GR**2 / (HR + lam) - (GL + GR)**2 / "
    "(HL + HR + lam), 4)`, and `max(gains, key=gains.get)` is the best cut."],
   "x = np.array([0.4, 0.6, 0.8, 1.2, 1.6, 2.4])\ny = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\ng = y.mean() - y\n\n"
   "for lam in [1, 0]:\n    gains = {}\n    for cut in [0.4, 0.6, 0.8, 1.2, 1.6]:\n        left = x <= cut\n"
   "        GL, GR = g[left].sum(), g[~left].sum()\n        HL, HR = left.sum(), (~left).sum()\n"
   "        gains[cut] = round(float(GL**2 / (HL + lam) + GR**2 / (HR + lam) - (GL + GR)**2 / (HL + HR + lam)), 4)\n"
   "    print(lam, gains)\n    print('best:', max(gains, key=gains.get))",
   f"With λ = 1 the cut after 0.8 wins with {K3[1][0.8]:.3f}, ahead of 0.6 and 1.2 at "
   f"{K3[1][0.6]:.3f} each. With λ = 0 its gain is {K3[0][0.8]:.3f}, exactly the fall in "
   "RSS from 0.815 to 0.080: without the penalty, XGBoost's rule is the tree rule of "
   f"Session 11. The penalty lowers every gain and small boxes most: the cut after "
   f"0.4, which leaves one day alone, falls from {K3[0][0.4]:.3f} to {K3[1][0.4]:.3f}.")

ex("K4", "How much a leaf keeps", 2,
   "With the penalty $\\lambda$, a leaf of $n$ days keeps $n/(n + \\lambda)$ of its "
   "plain mean residual (E2). Build a DataFrame with one row per $n$ in "
   "`[1, 5, 25, 100, 500]` and one column per $\\lambda$ in `[1, 10, 100]`, filled with "
   "that share to three decimals.",
   "sizes = [1, 5, 25, 100, 500]\nshares = {}\nfor lam in [1, 10, 100]:\n    ...\n\nkept = ...\nprint(kept)",
   ["For each λ, build a list of `round(n / (n + lam), 3)` over the sizes, and store "
    "it as `shares[lam]`.",
    "`pd.DataFrame(shares, index=sizes)` makes the table."],
   "sizes = [1, 5, 25, 100, 500]\nshares = {}\nfor lam in [1, 10, 100]:\n"
   "    shares[lam] = [round(n / (n + lam), 3) for n in sizes]\n\nkept = pd.DataFrame(shares, index=sizes)\nprint(kept)",
   f"At λ = 1 every leaf of 25 days or more keeps over 96 percent; at λ = 100 a leaf "
   f"of 5 days keeps {100 * _k4_df.loc[5, 100]:.1f} percent and a leaf of 500 keeps "
   f"{100 * _k4_df.loc[500, 100]:.1f} percent. On a table of {N_TRAIN:,} days with "
   "stumps, λ = 1 changes almost nothing; the small gap between XGBoost's stumps and "
   "scikit-learn's came from the bins.")

ex("K5", "Settings as dictionaries", 3,
   "Three settings for XGBoost are stored below as dictionaries. "
   "`XGBRegressor(random_state=0, **settings)` passes a dictionary's entries as "
   "arguments. Fit each on `fit_rows`, score it on `stop_rows`, and print the "
   "settings with the lowest 2022 RMSE.",
   "candidates = [\n    {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 1},\n"
   "    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 2},\n"
   "    {'n_estimators': 50, 'learning_rate': 0.3, 'max_depth': 1},\n]\nrmse_2022 = []\n"
   "for settings in candidates:\n    ...\n\nbest = ...\nprint(rmse_2022)\nprint(best)",
   ["In the loop: `model = XGBRegressor(random_state=0, **settings)`, fit on "
    "`fit_rows[columns]`, and append the RMSE on `stop_rows`.",
    "`candidates[int(np.argmin(rmse_2022))]` is the best dictionary."],
   "candidates = [\n    {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 1},\n"
   "    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 2},\n"
   "    {'n_estimators': 50, 'learning_rate': 0.3, 'max_depth': 1},\n]\nrmse_2022 = []\n"
   "for settings in candidates:\n    model = XGBRegressor(random_state=0, **settings)\n"
   "    model.fit(fit_rows[columns], fit_rows['vol_next'])\n"
   "    rmse_2022.append(round(rmse(stop_rows['vol_next'], model.predict(stop_rows[columns])), 4))\n\n"
   "best = candidates[int(np.argmin(rmse_2022))]\nprint(rmse_2022)\nprint(best)",
   f"{K5[0]:.3f}, {K5[1]:.3f} and {K5[2]:.3f}: 200 stumps at a rate of 0.1 win. A grid "
   "search does this for every combination in a dictionary of lists, and `**` is how "
   "a chosen dictionary of settings becomes a model.")

# ---------------------------------------------------------------- closing
md(
"## \U0001f3c1 Done\n\n"
"You boosted the six days by hand and as a function, showed that five stumps can "
"memorise them, checked that the trees are fitted to the negative gradient for a "
"number and for a label, watched the training error fall at every stump, chose "
"the number of trees on 2022 and on the folds, rebuilt XGBoost's early stopping "
"and its leaf values, timed three libraries, boosted the credit label, measured "
"importance four ways, compared every classifier of the course, and gave a "
"flexible model a target with nothing in it.\n\n"
"The case is the last part of the risk report: boosting for its volatility "
"forecast, held to everything the report has fitted since Part 4, and then every "
"model of the course side by side, ending with the report as it would go out."
)

# ---------------------------------------------------------------- write
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
for _i, _c in enumerate(nb.cells):
    _c["id"] = f"c{_i:04d}"

OUT.write_text(nbf.writes(nb), encoding="utf-8")

n_ex = sum(1 for c in cells if c.cell_type == "markdown" and c.source[:4] == "### " and c.source[4].isupper() and c.source[5].isdigit())
print("wrote", OUT, " (", len(cells), "cells,", n_ex, "exercises )")
