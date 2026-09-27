# -*- coding: utf-8 -*-
"""Build session_11_exercises.ipynb.

Same conventions as Sessions 1 to 9: pleasant intro, 1-5 star badges, toolkit
card with title= hover docs, task -> work cell (blank-safe `...`) -> 1-2 folded
hints -> folded solution, no em-dashes, plain explanatory tone.

Session 11 is trees and forests. The exercises use the lecture's two tables:
the index table from Session 6 (returns in percent), where the target is the
volatility over the next 20 days, and the credit table from Session 9, where the
label is a default. They compute the error of a box by hand, search for the best
cut with loops, fit and draw trees, watch depth overfit, choose settings on the
time-series folds, draw bootstrap samples, bag trees by hand and with
scikit-learn, grow random forests, count and time grids, and grow a tree and a
forest for a label.

Sections A to H work through the session. Section J forecasts the eleven
instruments one or two columns at a time. Section K is five standalone small
cases that deliberately reach back to loops, `while`, dictionaries, f-strings and
functions, in this session's context.

Only tools taught by the end of Session 11. New this session:
DecisionTreeRegressor, DecisionTreeClassifier, plot_tree (feature_names, filled,
impurity, proportion, label, precision, fontsize), max_depth, min_samples_leaf,
ccp_alpha, get_depth, get_n_leaves, tree_.feature, BaggingRegressor,
RandomForestRegressor, RandomForestClassifier, max_features, n_estimators,
oob_score, oob_prediction_, oob_decision_function_, estimators_,
time.perf_counter, RandomizedSearchCV (named in the lecture), n_jobs. Named in
the task where used, because the lecture did not show them: tree_.threshold,
.apply(), class_names.

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
import warnings
import numpy as np
import pandas as pd
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import (train_test_split, cross_val_score, TimeSeriesSplit,
                                     GridSearchCV, RandomizedSearchCV)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "session_11" / "session_11_exercises.ipynb"

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
    "A1": "S3", "A2": "S3", "A3": "S2", "A4": "S2", "A5": "S3",
    "B1": "S3", "B2": "S2", "B4": "S2",
    "C2": "S3", "C4": "S3", "C6": "S3",
    "D2": "S3", "D3": "S6", "D4": "S3",
    "E2": "S2", "E3": "S3", "E4": "S5",
    "F3": "S3", "F5": "S5", "F6": "S2",
    "G1": "S2", "G5": "S2",
    "H2": "S9", "H5": "S8", "H6": "S3",
    "I1": "S3", "I3": "S5", "I4": "S6",
    "J1": "S2", "J2": "S5", "J5": "S3",
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


def rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))


# ======================================================================
# The real numbers, computed here so every solution note is exact.
# ======================================================================
TABLE = pd.read_csv(ROOT / "data" / "market_features.csv", parse_dates=["date"]).set_index("date")
COLUMNS = list(TABLE.columns[:-1])
PAIR = ["vol_20d", "ret_20d"]
TRAIN, TEST = TABLE.loc[:"2022-12-31"], TABLE.loc["2023-01-01":]
Y, YT = TRAIN["vol_next"], TEST["vol_next"]
TS5 = TimeSeriesSplit(n_splits=5)
N_TRAIN, N_TEST = len(TRAIN), len(TEST)

CREDIT = pd.read_csv(ROOT / "data" / "credit.csv")
C_COLS = ["limit", "age", "late_now", "months_late", "bill", "paid", "utilisation"]
C_TRAIN, C_TEST = train_test_split(CREDIT, test_size=0.3, random_state=0, stratify=CREDIT["default"])

PRICES = pd.read_csv(ROOT / "data" / "prices.csv", parse_dates=["date"])
TICKERS = sorted(PRICES["ticker"].unique())
RETS = PRICES.pivot(index="date", columns="ticker", values="close").pct_change().dropna() * 100

# ---- A ----
SIX_X = [0.4, 0.6, 0.8, 1.2, 1.6, 2.4]
SIX_Y = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])
A1_MEAN = round(float(SIX_Y.mean()), 3)
A1_RSS = round(float(((SIX_Y - SIX_Y.mean()) ** 2).sum()), 3)
_const = np.arange(0.5, 1.5, 0.01)
A2_BEST = round(float(_const[np.argmin([((SIX_Y - c) ** 2).sum() for c in _const])]), 2)


def _rss(v):
    v = np.asarray(v, dtype=float)
    return round(float(((v - v.mean()) ** 2).sum()), 3)


A3_TRAIN = _rss(Y)
_six = pd.DataFrame({"vol_20d": SIX_X, "vol_next": SIX_Y})
A4 = {}
for _c in [0.4, 0.6, 0.8, 1.2, 1.6]:
    A4[_c] = round(_rss(_six[_six["vol_20d"] <= _c]["vol_next"]) + _rss(_six[_six["vol_20d"] > _c]["vol_next"]), 3)
A4_BEST = min(A4, key=A4.get)
_cuts = np.quantile(TRAIN["vol_20d"], np.linspace(0.02, 0.98, 60))
_tot = []
for _c in _cuts:
    _l, _r = Y[TRAIN["vol_20d"] <= _c], Y[TRAIN["vol_20d"] > _c]
    _tot.append(float(((_l - _l.mean()) ** 2).sum() + ((_r - _r.mean()) ** 2).sum()))
A5_CUT = round(float(_cuts[int(np.argmin(_tot))]), 3)
A5_MIN = round(min(_tot), 1)
A5_NONE = round(float(((Y - Y.mean()) ** 2).sum()), 1)
_flat = _cuts[np.array(_tot) <= min(_tot) + 5]
A5_FLAT = (round(float(_flat.min()), 2), round(float(_flat.max()), 2))

# ---- B ----
_calm, _busy = TRAIN[TRAIN["vol_20d"] <= 1.0], TRAIN[TRAIN["vol_20d"] > 1.0]
B1_COUNTS = (len(_calm), len(_busy))
B1_MEANS = (round(float(_calm["vol_next"].mean()), 3), round(float(_busy["vol_next"].mean()), 3))


def _best_split(rows, col):
    y = rows["vol_next"]
    best, best_total = None, np.inf
    for cut in np.unique(rows[col])[:-1]:
        left, right = y[rows[col] <= cut], y[rows[col] > cut]
        total = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
        if total < best_total:
            best, best_total = cut, total
    return round(float(best), 4), round(float(best_total), 1)


B2 = _best_split(TRAIN, "vol_20d")
_st = DecisionTreeRegressor(max_depth=1, random_state=0).fit(TRAIN[["vol_20d"]], Y)
B3_CUT = round(float(_st.tree_.threshold[0]), 4)
_above = float(np.unique(TRAIN["vol_20d"])[np.unique(TRAIN["vol_20d"]) > B2[0]][0])
B3_NEXT = round(_above, 4)
B4_ERR = {}
for _c in COLUMNS:
    _m = DecisionTreeRegressor(max_depth=1, random_state=0).fit(TRAIN[[_c]], Y)
    B4_ERR[_c] = round(rmse(Y, _m.predict(TRAIN[[_c]])), 4)
B4_BEST = min(B4_ERR, key=B4_ERR.get)
B4_SECOND = sorted(B4_ERR, key=B4_ERR.get)[1]
_tog = DecisionTreeRegressor(max_depth=1, random_state=0).fit(TRAIN[COLUMNS], Y)
B4_FIRST = COLUMNS[_tog.tree_.feature[0]]
B4_CUT = round(float(_tog.tree_.threshold[0]), 3)
B4_N_RIGHT = int(_tog.tree_.n_node_samples[2])
B5 = {}
for _name, _rows in [("calm", TRAIN[TRAIN["vol_20d"] <= 0.986]), ("busy", TRAIN[TRAIN["vol_20d"] > 0.986])]:
    _s = DecisionTreeRegressor(max_depth=1, random_state=0).fit(_rows[PAIR], _rows["vol_next"])
    B5[_name] = (PAIR[_s.tree_.feature[0]], round(float(_s.tree_.threshold[0]), 3))

# ---- C ----
SMALL = DecisionTreeRegressor(max_depth=2, random_state=0).fit(TRAIN[PAIR], Y)
C2 = pd.Series(SMALL.predict(TEST[PAIR]), index=TEST.index).round(3).value_counts()
C3 = {}
for _d in range(1, 7):
    C3[_d] = int(DecisionTreeRegressor(max_depth=_d, random_state=0).fit(TRAIN[PAIR], Y).get_n_leaves())
C4_A = round(float(SMALL.predict(pd.DataFrame({"vol_20d": [1.5], "ret_20d": [-0.6]}))[0]), 3)
C4_B = round(float(SMALL.predict(pd.DataFrame({"vol_20d": [1.5], "ret_20d": [0.1]}))[0]), 3)
_lin_pair = LinearRegression().fit(TRAIN[PAIR], Y)
C4_LIN = _lin_pair.predict(pd.DataFrame({"vol_20d": [1.5, 1.5], "ret_20d": [-0.6, 0.1]})).round(3).tolist()
TREE3 = DecisionTreeRegressor(max_depth=3, random_state=0).fit(TRAIN[COLUMNS], Y)
_t3 = TREE3.tree_
C5_ROOT = COLUMNS[_t3.feature[0]]
C5_SECOND = {COLUMNS[_t3.feature[_t3.children_left[0]]], COLUMNS[_t3.feature[_t3.children_right[0]]]}
assert len(C5_SECOND) == 1, C5_SECOND           # the note says both second questions ask the same column
C5_SECOND = C5_SECOND.pop()
C5_SMALL = sorted(int(_t3.n_node_samples[i]) for i in range(_t3.node_count) if _t3.children_left[i] == -1)[:2]
C5_TEST = round(rmse(YT, TREE3.predict(TEST[COLUMNS])), 3)
_ids3 = TREE3.apply(TEST[COLUMNS])
_leaf = pd.Series(_ids3).value_counts()
C6_N_LEAVES_USED = int(len(_leaf))
C6_MAX = int(_leaf.max())
C6_PCT = round(100 * C6_MAX / N_TEST, 1)
C6_FORECAST = round(float(_t3.value[_leaf.idxmax()][0][0]), 3)
C6_ACTUAL = round(float(YT[_ids3 == _leaf.idxmax()].mean()), 2)

# ---- D ----
FULL = DecisionTreeRegressor(random_state=0).fit(TRAIN[PAIR], Y)
D1 = (FULL.get_depth(), FULL.get_n_leaves(), round(rmse(Y, FULL.predict(TRAIN[PAIR])), 4),
      round(rmse(YT, FULL.predict(TEST[PAIR])), 4))
D2_TRAIN, D2_TEST = [], []
for _d in range(1, 16):
    _m = DecisionTreeRegressor(max_depth=_d, random_state=0).fit(TRAIN[PAIR], Y)
    D2_TRAIN.append(rmse(Y, _m.predict(TRAIN[PAIR])))
    D2_TEST.append(rmse(YT, _m.predict(TEST[PAIR])))
assert all(np.diff(D2_TRAIN) < 0)                # the note says the training error falls every step
assert round(D2_TEST[0], 3) == round(D2_TEST[1], 3) == min(round(v, 3) for v in D2_TEST)
assert min(D2_TEST[9:]) > 0.43                    # "above 0.43 from depth 10 on"
D3 = GridSearchCV(DecisionTreeRegressor(random_state=0), {"max_depth": [1, 2, 3, 4, 6, 8]}, cv=TS5,
                  scoring="neg_root_mean_squared_error").fit(TRAIN[PAIR], Y)
D3_BEST = D3.best_params_["max_depth"]
D3_CV = round(float(-D3.best_score_), 4)
D3_TEST = round(rmse(YT, D3.predict(TEST[PAIR])), 4)
_folds = [-D3.cv_results_[f"split{k}_test_score"][D3.best_index_] for k in range(5)]
D3_FOLD_2020 = round(float(_folds[2]), 2)
assert max(_folds) == _folds[2] and max(_folds[:2] + _folds[3:]) < 0.6
D4 = pd.Series(-D3.cv_results_["mean_test_score"], index=list(D3.cv_results_["param_max_depth"])).round(4).sort_values()
D5 = GridSearchCV(DecisionTreeRegressor(random_state=0), {"min_samples_leaf": [20, 50, 100, 200, 400]}, cv=TS5,
                  scoring="neg_root_mean_squared_error").fit(TRAIN[PAIR], Y)
D5_BEST = D5.best_params_["min_samples_leaf"]
D5_DEPTH = D5.best_estimator_.get_depth()
D5_LEAVES = D5.best_estimator_.get_n_leaves()
D5_TEST = round(rmse(YT, D5.predict(TEST[PAIR])), 4)
D6 = {}
for _a in [0, 0.005, 0.01, 0.02, 0.05]:
    _m = DecisionTreeRegressor(ccp_alpha=_a, random_state=0).fit(TRAIN[PAIR], Y)
    D6[_a] = (int(_m.get_n_leaves()), round(rmse(YT, _m.predict(TEST[PAIR])), 4))
# at 0.05 two leaves merge that no test day lands in, so the score does not move
assert D6[0.05][0] == D6[0.02][0] - 1 and D6[0.05][1] == D6[0.02][1]

# ---- E ----
_s1 = TRAIN.sample(n=N_TRAIN, replace=True, random_state=1)
E1_SHARE = round(_s1.index.nunique() / N_TRAIN, 3)
E1_FORMULA = round(1 - (1 - 1 / N_TRAIN) ** N_TRAIN, 3)
E2 = [round(TRAIN.sample(n=N_TRAIN, replace=True, random_state=s).index.nunique() / N_TRAIN, 3) for s in range(20)]
E2_MEAN, E2_MIN, E2_MAX = round(float(np.mean(E2)), 3), min(E2), max(E2)
_s0 = TRAIN.sample(n=N_TRAIN, replace=True, random_state=0)
_draws = _s0.index.value_counts().reindex(TRAIN.index, fill_value=0)
E3 = _draws.value_counts().sort_index()
E3_ZERO = int(E3.loc[0])
E3_MAX = int(E3.index.max())
_ha = TRAIN.sample(frac=0.5, random_state=1)
_hb = TRAIN.drop(_ha.index)
_ta = DecisionTreeRegressor(random_state=0).fit(_ha[PAIR], _ha["vol_next"])
_tb = DecisionTreeRegressor(random_state=0).fit(_hb[PAIR], _hb["vol_next"])
_pa, _pb = _ta.predict(TEST[PAIR]), _tb.predict(TEST[PAIR])
E4_DIFF = round(float(np.abs(_pa - _pb).mean()), 3)
E4_RA, E4_RB, E4_AVG = round(rmse(YT, _pa), 3), round(rmse(YT, _pb), 3), round(rmse(YT, (_pa + _pb) / 2), 3)
_trees = []
for _b in range(25):
    _smp = TRAIN.sample(n=N_TRAIN, replace=True, random_state=_b)
    _trees.append(DecisionTreeRegressor(random_state=0).fit(_smp[COLUMNS], _smp["vol_next"]))
_all = np.array([t.predict(TEST[COLUMNS]) for t in _trees])
E5 = round(rmse(YT, np.mean(_all, axis=0)), 4)
E5_ONE = round(rmse(YT, _all[0]), 4)
_bag = BaggingRegressor(DecisionTreeRegressor(), n_estimators=25, random_state=0).fit(TRAIN[COLUMNS], Y)
E6 = pd.Series([COLUMNS[t.tree_.feature[0]] for t in _bag.estimators_]).value_counts()
E6_TEST = round(rmse(YT, _bag.predict(TEST[COLUMNS])), 4)
E6_VOL = int(sum(v for k, v in E6.items() if "vol" in k))
_total, _count = np.zeros(N_TRAIN), np.zeros(N_TRAIN)
for _b in range(25):
    _smp = TRAIN.sample(n=N_TRAIN, replace=True, random_state=_b)
    _out = ~TRAIN.index.isin(_smp.index)
    _total[_out] += _trees[_b].predict(TRAIN[COLUMNS][_out])
    _count[_out] += 1
assert _count.min() > 0                           # the hint promises no division by zero
E7_OOB = round(rmse(Y, _total / _count), 4)
E7_TREES = round(float(_count.mean()), 1)

# ---- F ----
F1_FOREST = RandomForestRegressor(n_estimators=100, max_features=0.33, random_state=0).fit(TRAIN[COLUMNS], Y)
F1 = round(rmse(YT, F1_FOREST.predict(TEST[COLUMNS])), 4)
F1_EACH = max(1, int(0.33 * len(COLUMNS)))
F2 = {}
for _mf in [1, 4, 19]:
    _f = RandomForestRegressor(n_estimators=100, max_features=_mf, random_state=0).fit(TRAIN[COLUMNS], Y)
    F2[_mf] = int(pd.Series([COLUMNS[t.tree_.feature[0]] for t in _f.estimators_]).nunique())
_f200 = RandomForestRegressor(n_estimators=200, max_features="sqrt", random_state=0).fit(TRAIN[COLUMNS], Y)
_pt = np.array([e.predict(TEST[COLUMNS].to_numpy()) for e in _f200.estimators_])
_cum = _pt.cumsum(axis=0) / np.arange(1, 201)[:, None]
F3 = {n: round(rmse(YT, _cum[n - 1]), 4) for n in [1, 10, 50, 100, 200]}
LINE_ONE = round(rmse(YT, LinearRegression().fit(TRAIN[["vol_20d"]], Y).predict(TEST[["vol_20d"]])), 4)
_f4 = RandomForestRegressor(n_estimators=100, max_features="sqrt", min_samples_leaf=50, oob_score=True,
                            random_state=0).fit(TRAIN[COLUMNS], Y)
F4_OOB = round(rmse(Y, _f4.oob_prediction_), 4)
F4_FOLDS = round(float(-cross_val_score(_f4, TRAIN[COLUMNS], Y, cv=TS5,
                                        scoring="neg_root_mean_squared_error").mean()), 4)
F4_TEST = round(rmse(YT, _f4.predict(TEST[COLUMNS])), 4)
F5 = {}
for _mf in [1, 2, 4, 8, 19]:
    _f = RandomForestRegressor(n_estimators=100, max_features=_mf, min_samples_leaf=50, random_state=0)
    F5[_mf] = round(float(-cross_val_score(_f, TRAIN[COLUMNS], Y, cv=TS5,
                                           scoring="neg_root_mean_squared_error").mean()), 4)
F5_BEST = min(F5, key=F5.get)
assert F5_BEST == 1 and max(F5, key=F5.get) == 19 and list(F5.values()) == sorted(F5.values())
F6 = {}
for _mf in [1, 4]:
    for _leaf in [20, 50, 100]:
        _f = RandomForestRegressor(n_estimators=100, max_features=_mf, min_samples_leaf=_leaf, random_state=0)
        F6[(_mf, _leaf)] = round(float(-cross_val_score(_f, TRAIN[COLUMNS], Y, cv=TS5,
                                                        scoring="neg_root_mean_squared_error").mean()), 4)
F6_BEST = min(F6, key=F6.get)

# ---- G ----
G1_COMBOS = 5 * 6 * 3 * 4
G3 = GridSearchCV(RandomForestRegressor(n_estimators=50, random_state=0),
                  {"max_features": [1, 4, 19], "min_samples_leaf": [5, 20, 50, 100, 200]}, cv=TS5,
                  scoring="neg_root_mean_squared_error").fit(TRAIN[COLUMNS], Y)
G3_BEST = G3.best_params_
G3_CV = round(float(-G3.best_score_), 4)
_g3 = pd.DataFrame(G3.cv_results_).pivot(index="param_min_samples_leaf", columns="param_max_features",
                                          values="mean_test_score").mul(-1)
G3_WORST = round(float(_g3.values.max()), 4)
G4 = RandomizedSearchCV(RandomForestRegressor(n_estimators=50, random_state=0),
                        {"max_features": [1, 2, 4, 8, 19], "min_samples_leaf": [5, 20, 50, 100, 200, 400],
                         "max_depth": [3, 5, 10, None]},
                        n_iter=10, cv=TS5, scoring="neg_root_mean_squared_error", random_state=0).fit(TRAIN[COLUMNS], Y)
G4_BEST = G4.best_params_
assert G4_BEST == {"min_samples_leaf": 50, "max_features": 2, "max_depth": 3}   # the lecture's 14-minute choice
G4_CV = round(float(-G4.best_score_), 4)
G4_TEST = round(rmse(YT, G4.predict(TEST[COLUMNS])), 4)
assert (_g3[19] > _g3[4]).all() and (_g3[19] > _g3[1]).all()
assert all(_g3[m].idxmin() in (50, 100) for m in _g3.columns)
_curve = [rmse(YT, row) for row in _cum]
assert min(_curve) > LINE_ONE
G5 = {}
for _n in [25, 50, 100, 200, 400]:
    _f = RandomForestRegressor(n_estimators=_n, max_features="sqrt", min_samples_leaf=50, random_state=0).fit(TRAIN[COLUMNS], Y)
    G5[_n] = round(rmse(YT, _f.predict(TEST[COLUMNS])), 4)

# ---- H ----
_yc = C_TRAIN["default"]
H1_N, H1_K = len(_yc), int(_yc.sum())
H1_DIRECT = round(float(((_yc - _yc.mean()) ** 2).sum()), 2)
H1_FORMULA = round(H1_K * (H1_N - H1_K) / H1_N, 2)
H1_GINI = round(2 * (H1_K / H1_N) * (1 - H1_K / H1_N), 4)
H1_TWICE = round(2 * H1_DIRECT / H1_N, 4)
assert H1_DIRECT == H1_FORMULA and H1_GINI == H1_TWICE
assert f"{2 * (H1_K / H1_N) * (1 - H1_K / H1_N):.3f}" == "0.345"      # what plot_tree prints at the root
CTREE = DecisionTreeClassifier(max_depth=2, random_state=0).fit(C_TRAIN[C_COLS], C_TRAIN["default"])
H1_AUC = round(float(roc_auc_score(C_TEST["default"], CTREE.predict_proba(C_TEST[C_COLS])[:, 1])), 4)
H1_PROBS = np.unique(CTREE.predict_proba(C_TEST[C_COLS])[:, 1]).round(3).tolist()
_pr = float(C_TRAIN["default"].mean())
_left = C_TRAIN[C_TRAIN["late_now"] <= 1]["default"]
_right = C_TRAIN[C_TRAIN["late_now"] > 1]["default"]
H3_ROOT = round(2 * _pr * (1 - _pr), 4)
H3_LEFT = round(2 * _left.mean() * (1 - _left.mean()), 4)
H3_RIGHT = round(2 * _right.mean() * (1 - _right.mean()), 4)
_g = lambda v: 2 * v.mean() * (1 - v.mean())
H3_WEIGHTED = round(float((len(_left) * _g(_left) + len(_right) * _g(_right)) / len(C_TRAIN)), 4)
H3_N_RIGHT = len(_right)
H3_SHARE_RIGHT = round(100 * len(_right) / len(C_TRAIN), 1)
_ct = CTREE.tree_
assert [C_COLS[_ct.feature[i]] for i in (0, 1, 4)] == ["late_now", "months_late", "utilisation"]
H2_SMALL_N = int(_ct.n_node_samples[5])
H2_SMALL_P = round(float(_ct.value[5][0][1] / _ct.value[5][0].sum()), 2)
H2_LAST_P = round(float(_ct.value[6][0][1] / _ct.value[6][0].sum()), 2)
_rf_default = RandomForestRegressor(n_estimators=100, random_state=0).fit(TRAIN[COLUMNS], Y)
_rf_sqrt = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=0).fit(TRAIN[COLUMNS], Y)
I4_DEFAULT_MF = _rf_default.get_params()["max_features"]
I4_CLASSIFIER_MF = RandomForestClassifier().get_params()["max_features"]
I4_DEFAULT, I4_FIXED = round(rmse(YT, _rf_default.predict(TEST[COLUMNS])), 4), round(rmse(YT, _rf_sqrt.predict(TEST[COLUMNS])), 4)
I4_OPEN = (int(pd.Series([COLUMNS[t.tree_.feature[0]] for t in _rf_default.estimators_]).nunique()),
           int(pd.Series([COLUMNS[t.tree_.feature[0]] for t in _rf_sqrt.estimators_]).nunique()))
assert I4_DEFAULT_MF == 1.0 and I4_CLASSIFIER_MF == "sqrt" and I4_FIXED < I4_DEFAULT
_cf = RandomForestClassifier(n_estimators=100, min_samples_leaf=50, oob_score=True, random_state=0).fit(C_TRAIN[C_COLS], C_TRAIN["default"])
H5_OOB = round(float(roc_auc_score(C_TRAIN["default"], _cf.oob_decision_function_[:, 1])), 4)
H5_TEST = round(float(roc_auc_score(C_TEST["default"], _cf.predict_proba(C_TEST[C_COLS])[:, 1])), 4)
_lg = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression())]).fit(C_TRAIN[C_COLS], C_TRAIN["default"])
H5_LOGIT = round(float(roc_auc_score(C_TEST["default"], _lg.predict_proba(C_TEST[C_COLS])[:, 1])), 4)
_leaves = pd.Series(CTREE.apply(C_TRAIN[C_COLS]), index=C_TRAIN.index)
H6 = C_TRAIN["default"].groupby(_leaves).mean()
assert f"{H6.iloc[-1]:.2f}" == f"{H2_LAST_P:.2f}"
H6_SHARE = round(float(C_TRAIN["default"].mean()), 3)


# ---- J ----
def _vol_table(ticker):
    r = RETS[ticker]
    frame = pd.DataFrame({"vol_20d": r.rolling(20).std(), "ret_20d": r.rolling(20).mean()})
    frame["vol_next"] = r.rolling(20).std().shift(-20)
    return frame.dropna()


J1_SHAPE = _vol_table("AAPL").shape
J2_TREE, J2_LINE, J3_CUT, J3_ABOVE, J3_WHEN, J4_TREE, J4_LINE = {}, {}, {}, {}, {}, {}, {}
for _t in TICKERS:
    _f = _vol_table(_t)
    _a, _b = _f.loc[:"2022-12-31"], _f.loc["2023-01-01":]
    _s = DecisionTreeRegressor(max_depth=1, random_state=0).fit(_a[["vol_20d"]], _a["vol_next"])
    _l = LinearRegression().fit(_a[["vol_20d"]], _a["vol_next"])
    J2_TREE[_t] = round(rmse(_b["vol_next"], _s.predict(_b[["vol_20d"]])), 4)
    J2_LINE[_t] = round(rmse(_b["vol_next"], _l.predict(_b[["vol_20d"]])), 4)
    J3_CUT[_t] = round(float(_s.tree_.threshold[0]), 3)
    _busy = _a[_a["vol_20d"] > _s.tree_.threshold[0]]
    J3_ABOVE[_t] = len(_busy)
    J3_WHEN[_t] = (_busy.index.min(), _busy.index.max(), len(_a))
    _d2 = DecisionTreeRegressor(max_depth=2, random_state=0).fit(_a[PAIR], _a["vol_next"])
    _l2 = LinearRegression().fit(_a[PAIR], _a["vol_next"])
    J4_TREE[_t] = round(rmse(_b["vol_next"], _d2.predict(_b[PAIR])), 4)
    J4_LINE[_t] = round(rmse(_b["vol_next"], _l2.predict(_b[PAIR])), 4)
J2_WINS = [t for t in TICKERS if J2_TREE[t] < J2_LINE[t]]
J3_SORTED = sorted(J3_CUT, key=J3_CUT.get)
# stumps whose busy box is only the crash of spring 2020
J3_CRASH = [t for t in TICKERS if J3_WHEN[t][0] >= pd.Timestamp("2020-02-01") and J3_WHEN[t][1] <= pd.Timestamp("2020-07-31")]
J3_CRASH_N = sorted(J3_ABOVE[t] for t in J3_CRASH)
_low = J3_SORTED[0]
J3_LOW_CALM = J3_WHEN[_low][2] - J3_ABOVE[_low]
J4_WINS = sorted([t for t in TICKERS if J4_TREE[t] < J4_LINE[t]],
                 key=lambda t: (J4_LINE[t] - J4_TREE[t]) / J4_LINE[t], reverse=True)
J4_PCT = {t: round(100 * (J4_LINE[t] - J4_TREE[t]) / J4_LINE[t], 1) for t in J4_WINS}
J5_RATIO = {t: round(J2_TREE[t] / J2_LINE[t], 3) for t in TICKERS}
J5_WORST = max(J5_RATIO, key=J5_RATIO.get)


# ---- K ----
def _k1(vol, ret):
    if vol <= 0.99:
        return 0.70 if vol <= 0.82 else 0.99
    return 2.84 if ret <= -0.45 else 1.25


K1_DAYS = [(0.55, 0.10), (0.90, -0.20), (1.40, -0.70), (2.10, 0.05)]
K1_OUT = [_k1(v, r) for v, r in K1_DAYS]
# K2: the arrays of a real tree of depth 2 offered three columns, cuts to three decimals
K2_NAMES = ["vol_20d", "ret_20d", "vol_5d"]
_k2 = DecisionTreeRegressor(max_depth=2, random_state=0).fit(TRAIN[K2_NAMES], Y)
K2_FEATURE = [int(v) for v in _k2.tree_.feature]
K2_THRESHOLD = [round(float(v), 3) for v in _k2.tree_.threshold]
K2_QUESTIONS = [f"{K2_NAMES[f]} <= {t}" for f, t in zip(K2_FEATURE, K2_THRESHOLD) if f >= 0]
K2_ASKED = {K2_NAMES[f] for f in K2_FEATURE if f >= 0}
assert K2_FEATURE == [2, 0, -2, -2, 0, -2, -2] and K2_THRESHOLD == [3.251, 0.841, -2.0, -2.0, 4.587, -2.0, -2.0]
assert len(K2_QUESTIONS) == 3 and K2_ASKED == {"vol_5d", "vol_20d"}
_k2_busy = TRAIN[TRAIN["vol_5d"] > _k2.tree_.threshold[0]]
K2_BUSY = len(_k2_busy)
K2_BUSY_2020 = int((_k2_busy.index.year == 2020).sum())
assert set(_k2_busy.index[_k2_busy.index.year == 2020].month) <= {3, 4}   # "in March and April 2020"
_left = np.zeros(N_TRAIN, dtype=bool)
K3_SAMPLES = 0
K3_AFTER_10 = None
while K3_SAMPLES < 100:
    K3_SAMPLES += 1
    _smp = TRAIN.sample(n=N_TRAIN, replace=True, random_state=K3_SAMPLES)
    _left = _left | ~TRAIN.index.isin(_smp.index)
    if K3_SAMPLES == 10:
        K3_AFTER_10 = int((~_left).sum())
    if _left.all():
        break
_p_in = (1 - 1 / N_TRAIN) ** N_TRAIN            # the chance a day is left out of one sample
K3_FORMULA = int(np.ceil(np.log(1 / N_TRAIN) / np.log(1 - _p_in)))
assert K3_SAMPLES < 100 and K3_AFTER_10 > 0
# K4: G3's grid of fold RMSEs, rows max_features [1, 4, 19], columns min_samples_leaf [5 ... 200]
K4_SCORES = _g3.T.round(3).values.tolist()
assert _g3.columns.tolist() == [1, 4, 19] and _g3.index.tolist() == [5, 20, 50, 100, 200]
assert K4_SCORES == [[0.639, 0.62, 0.593, 0.601, 0.615], [0.715, 0.662, 0.62, 0.613, 0.628],
                     [0.804, 0.741, 0.696, 0.682, 0.705]]
_k4 = np.array(K4_SCORES)
K4_POS = int(_k4.argmin())
K4_ROW, K4_COL = K4_POS // _k4.shape[1], K4_POS % _k4.shape[1]
K4_PAIR = ([1, 4, 19][K4_ROW], [5, 20, 50, 100, 200][K4_COL])
assert K4_PAIR == (G3_BEST["max_features"], G3_BEST["min_samples_leaf"]) == (1, 50)
# K5: ten training days, one every 180, the leaf of the lecture's depth-2 tree each landed in
_k5 = TRAIN.iloc[list(range(0, N_TRAIN, 180))[:10]]
K5_LANDED = [int(v) for v in SMALL.apply(_k5[PAIR])]
K5_NEXT = [round(float(v), 2) for v in _k5["vol_next"]]
_groups = {}
for _l, _v in zip(K5_LANDED, K5_NEXT):
    _groups.setdefault(_l, []).append(_v)
K5_MEANS = {k: round(sum(v) / len(v), 3) for k, v in _groups.items()}
K5_SIZES = {k: len(v) for k, v in _groups.items()}
_leaf_all = pd.Series(SMALL.apply(TRAIN[PAIR])).value_counts()
K5_TREE = {k: (round(float(SMALL.tree_.value[k].ravel()[0]), 3), int(_leaf_all[k])) for k in K5_MEANS}
assert K5_LANDED == [2, 3, 2, 2, 2, 6, 2, 6, 6, 3] and K5_MEANS == {2: 0.594, 3: 0.87, 6: 0.747}
assert pd.Series(K5_NEXT).groupby(K5_LANDED).mean().round(3).to_dict() == K5_MEANS

# ======================================================================
# Notebook
# ======================================================================
md(
"# \U0001f9ea Session 11 exercises\n"
"### Trees and forests\n\n"
"The lecture built a tree from its parts: the error of a box, the best cut, one "
"question at a time, and a formula for the boxes at the end. Then it grew trees "
"too deep, averaged many of them, and tuned a forest on the time-series folds. "
"These exercises do each of those steps yourself, with loops and functions "
"first and scikit-learn second.\n\n"
"Two tables, the same two as the lecture. The **index table** from Session 6, "
"with the volatility over the next 20 days as the target. The **credit table** "
"from Session 9, with a default as the label."
)

md(
"## How to use this notebook\n\n"
"- Run the **setup cell** below first. It loads both tables and the eleven "
"instruments, makes the splits, and imports the scikit-learn pieces.\n"
"- Each exercise has a **task**, then a **code cell** for your work. Cells with "
"`...` are blanks to fill in. Replace them with real code.\n"
"- Stuck? Open the **\U0001f4a1 Hint**, but only after a genuine attempt. Open the "
"**✅ Solution** to *check* yourself, not to skip the thinking.\n"
"- Every cell runs cleanly even with the blanks still in place, so pressing "
"**Run all** never floods you with errors.\n"
"- Most exercises stand alone. A few short runs build on each other (A3 to A4, "
"B2 to B3, D3 to D4); the task says which earlier exercise it continues from. "
"Section I shows five mistakes and asks you to fix them. Section J uses the "
"function from J1, and **section K is five small cases that each start from "
"scratch**. Ten of them ask you to draw something: A5, C1, C5, D2, E3, F3, G3, "
"H3, H6 and J5.\n"
"- A few cells fit forests or search grids and take several seconds. The task "
"says so where it matters.\n\n"
"**You are not expected to finish all of these.** Do what you can, and come back "
"to the rest when you revise. Short on time? Read the hint, then the solution. A "
"worked solution you genuinely understand is real learning too.\n\n"
"**Units.** The index table is in percent, as in the lecture. The credit table "
"is in New Taiwan dollars, as it comes."
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
"assumes everything from Sessions 1 to 10, so it is a bigger piece of work "
"than a three-star task in an earlier notebook.\n\n"
"Some exercises also carry a **revisits** tag. Those need something from an "
"earlier session as well as today's material, and they are there on purpose: "
"the skills are meant to accumulate."
)

md("---")

md(
"## \U0001f9f0 Toolkit\n\n"
"New this session. Hover a name for what it does.\n\n"
'<span title="A tree for a number. Each leaf forecasts the mean of its training rows. max_depth, min_samples_leaf and ccp_alpha limit its size.">`DecisionTreeRegressor`</span> · '
'<span title="A tree for a label. Each leaf holds the share of each class, and predict_proba returns it.">`DecisionTreeClassifier`</span> · '
'<span title="Draws a fitted tree with matplotlib. feature_names names the columns; filled shades the boxes; impurity=False drops the error line; label=root names the fields in the top box only; precision and fontsize set the text.">`plot_tree`</span> · '
'<span title="The number of questions on the longest path, and the number of boxes at the end.">`.get_depth()` `.get_n_leaves()`</span> · '
'<span title="Inside a fitted tree: tree_.feature[0] is the column number of the first question and tree_.threshold[0] its cut.">`tree_`</span> · '
'<span title="Grows each tree on a bootstrap sample and averages them. The trees are kept in estimators_.">`BaggingRegressor`</span> · '
'<span title="Bagging where each question chooses among max_features columns drawn at random. oob_score=True scores each row with the trees that did not see it.">`RandomForestRegressor`</span> · '
'<span title="The same for a label. oob_decision_function_ holds the out-of-bag probabilities.">`RandomForestClassifier`</span> · '
'<span title="Reads a clock in seconds. The difference of two readings is the time between them.">`time.perf_counter`</span> · '
'<span title="Tries n_iter combinations drawn at random from a grid, instead of all of them.">`RandomizedSearchCV`</span> · '
'<span title="n_jobs=-1 fits the trees of a forest, or the folds of a search, on every core.">`n_jobs`</span>\n\n'
"**Formulas**\n\n"
"- the forecast of a box = the mean of its training rows\n"
"- the error of a box: RSS = sum of (y minus the box mean) squared; for labels of 0 and 1, k(n minus k)/n with k ones among n rows\n"
"- a cut is chosen to make RSS(left) + RSS(right) as small as possible\n"
"- Gini for two classes = 2p(1 minus p), with p the share of ones\n"
"- the share of distinct rows in a bootstrap sample of n rows is about 1 minus (1 minus 1/n) to the power n, which is about 0.632\n"
"- combinations in a grid = the product of the lengths of its lists; fits = combinations x folds"
)

md("---")

md("## ⚙️ Setup · run me first")

code(
'''import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import (train_test_split, cross_val_score, TimeSeriesSplit,
                                     GridSearchCV, RandomizedSearchCV)

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
    """Root mean squared error, as in Sessions 5 and 6."""
    return np.sqrt(mean_squared_error(actual, predicted))


# ---- the index table from Session 6: one row per trading day, percent ----
table = pd.read_csv(data_path("market_features.csv"), parse_dates=["date"]).set_index("date")
columns = list(table.columns[:-1])      # the 19 feature columns
pair = ["vol_20d", "ret_20d"]
train = table.loc[:"2022-12-31"]
test = table.loc["2023-01-01":]
folds = TimeSeriesSplit(n_splits=5)

# ---- the credit table from Session 9: one row per borrower, no time order ----
credit = pd.read_csv(data_path("credit.csv"))
c_columns = ["limit", "age", "late_now", "months_late", "bill", "paid", "utilisation"]
c_train, c_test = train_test_split(credit, test_size=0.3, random_state=0,
                                   stratify=credit["default"])

# ---- the eleven instruments, for section J. Returns in percent. ----
prices = pd.read_csv(data_path("prices.csv"), parse_dates=["date"])
tickers = sorted(prices["ticker"].unique())
rets = prices.pivot(index="date", columns="ticker", values="close").pct_change().dropna() * 100

print("index :", table.shape, "  train", len(train), " test", len(test))
print("credit:", credit.shape, "  default share", round(credit["default"].mean(), 4))
print("prices:", prices.shape[0], "rows,", len(tickers), "instruments")'''
)

md("---")

# ====================================================== A
section(
"## A · A box and its error\n\n"
"A tree forecasts with boxes, and every box forecasts one number. These "
"exercises find that number and the error it leaves, with NumPy and loops "
"first. A3 and A4 run together."
)

ex("A1", "The forecast of one box", 1,
   "The six days of the lecture are below. A box forecasts the mean of its days. "
   "Compute the mean of `vol_next` and the box's RSS, the sum of squared "
   "distances from that mean, with NumPy.",
   "vol_next = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\n\nmean = ...\nbox_rss = ...\n\nprint(mean)\nprint(box_rss)",
   "`vol_next.mean()` is the mean. Subtract it from the array, square, and "
   "`.sum()`. Wrap each in `round(float(...), 3)` for a plain number.",
   "vol_next = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\n\n"
   "mean = round(float(vol_next.mean()), 3)\n"
   "box_rss = round(float(((vol_next - vol_next.mean()) ** 2).sum()), 3)\n\n"
   "print(mean)\nprint(box_rss)",
   f"The box forecasts {A1_MEAN} for all six days and leaves an RSS of {A1_RSS}. "
   "Subtracting a number from an array subtracts it from every element, which "
   "is what makes this one line instead of a loop.")

ex("A2", "No constant beats the mean", 2,
   "Try every constant from 0.5 to 1.5 in steps of 0.01 as the forecast for the "
   "same six days. Compute the RSS for each, and print the constant with the "
   "smallest RSS.",
   "vol_next = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\nconstants = np.arange(0.5, 1.5, 0.01)\n"
   "totals = []\nfor c in constants:\n    ...\n\nbest = ...\n\nprint(best)",
   ["Inside the loop: `totals.append(((vol_next - c) ** 2).sum())`.",
    "`np.argmin(totals)` is the POSITION of the smallest total, so the constant "
    "is `constants[np.argmin(totals)]`."],
   "vol_next = np.array([0.5, 0.6, 0.7, 1.2, 1.5, 1.2])\nconstants = np.arange(0.5, 1.5, 0.01)\n"
   "totals = []\nfor c in constants:\n    totals.append(((vol_next - c) ** 2).sum())\n\n"
   "best = round(float(constants[np.argmin(totals)]), 2)\n\nprint(best)",
   f"{A2_BEST}, the mean from A1. The lecture proved it with a derivative; here "
   "a hundred tries find the same answer. The mean is the best single number "
   "under squared error, which is why every leaf of a regression tree forecasts "
   "a mean.")

ex("A3", "The error of a box, as a function", 2,
   "Write `rss(values)` returning the sum of squared distances from the mean of "
   "`values`, rounded to three decimals, with a docstring. It should work on a "
   "list, a NumPy array and a pandas Series. Test it on the six days and on the "
   "`vol_next` of all training days.",
   "def rss(values):\n    \"\"\"...\"\"\"\n    ...\n\n\n"
   "print(rss([0.5, 0.6, 0.7, 1.2, 1.5, 1.2]))\nprint(rss(train['vol_next']))",
   "`values = np.asarray(values, dtype=float)` turns any of the three into an "
   "array first. Then the same line as A1.",
   "def rss(values):\n"
   "    \"\"\"The sum of squared distances from the mean: the error of a box.\"\"\"\n"
   "    values = np.asarray(values, dtype=float)\n"
   "    return round(float(((values - values.mean()) ** 2).sum()), 3)\n\n\n"
   "print(rss([0.5, 0.6, 0.7, 1.2, 1.5, 1.2]))\nprint(rss(train['vol_next']))",
   f"{A1_RSS} and {A3_TRAIN:,}. The second is the error of a tree with no "
   "question at all: every training day forecast by the one mean. Keep `rss`: "
   "A4 continues from here.")

ex("A4", "Every cut on the six", 2,
   "Using `rss` from A3, loop over the cuts after 0.4, 0.6, 0.8, 1.2 and 1.6 on "
   "the six days. Store each cut's total RSS, left box plus right box, in a "
   "dictionary, and print the dictionary and the cut with the smallest total.",
   "six = pd.DataFrame({'vol_20d': [0.4, 0.6, 0.8, 1.2, 1.6, 2.4],\n"
   "                    'vol_next': [0.5, 0.6, 0.7, 1.2, 1.5, 1.2]})\n"
   "by_cut = {}\nfor cut in [0.4, 0.6, 0.8, 1.2, 1.6]:\n    ...\n\nbest_cut = ...\n\n"
   "print(by_cut)\nprint(best_cut)",
   ["The left box is `six[six['vol_20d'] <= cut]['vol_next']`, the right box "
    "the same with `>`.",
    "`min(by_cut, key=by_cut.get)` gives the KEY with the smallest value."],
   "six = pd.DataFrame({'vol_20d': [0.4, 0.6, 0.8, 1.2, 1.6, 2.4],\n"
   "                    'vol_next': [0.5, 0.6, 0.7, 1.2, 1.5, 1.2]})\n"
   "by_cut = {}\nfor cut in [0.4, 0.6, 0.8, 1.2, 1.6]:\n"
   "    left = six[six['vol_20d'] <= cut]['vol_next']\n"
   "    right = six[six['vol_20d'] > cut]['vol_next']\n"
   "    by_cut[cut] = round(rss(left) + rss(right), 3)\n\nbest_cut = min(by_cut, key=by_cut.get)\n\n"
   "print(by_cut)\nprint(best_cut)",
   f"`{A4}`, and the cut after {A4_BEST} wins with {A4[A4_BEST]}. It puts the "
   "three calm days in one box and the three busy days in the other, and each "
   "box is then close to its own mean.")

ex("A5", "Draw the RSS of every cut", 2,
   "For 60 cuts on `vol_20d`, the percentiles from 2 to 98 of the training days, "
   "compute the total RSS each cut leaves on `vol_next`. Draw the total against "
   "the cut and mark the lowest point.",
   "cuts = np.quantile(train['vol_20d'], np.linspace(0.02, 0.98, 60))\ntotals = []\n"
   "for cut in cuts:\n    left = train[train['vol_20d'] <= cut]['vol_next']\n"
   "    right = train[train['vol_20d'] > cut]['vol_next']\n    ...\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3))\n...\n...\n"
   "ax.set_xlabel('cut on vol_20d')\nax.set_ylabel('total RSS')\nplt.show()",
   ["Inside the loop: `totals.append(((left - left.mean()) ** 2).sum() + "
    "((right - right.mean()) ** 2).sum())`.",
    "`ax.plot(cuts, totals)` for the curve. The lowest point is at "
    "`cuts[np.argmin(totals)]` and `min(totals)`; mark it with `ax.scatter`."],
   "cuts = np.quantile(train['vol_20d'], np.linspace(0.02, 0.98, 60))\ntotals = []\n"
   "for cut in cuts:\n    left = train[train['vol_20d'] <= cut]['vol_next']\n"
   "    right = train[train['vol_20d'] > cut]['vol_next']\n"
   "    totals.append(((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum())\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3))\nax.plot(cuts, totals, color='#1c5cab')\n"
   "ax.scatter([cuts[np.argmin(totals)]], [min(totals)], color='#b3402f', zorder=3)\n"
   "ax.set_xlabel('cut on vol_20d')\nax.set_ylabel('total RSS')\n"
   "ax.set_title('Total RSS left by each cut, training days', loc='left')\nplt.show()",
   f"With no cut the total is {A5_NONE}. On this grid the curve is lowest at a "
   f"cut of {A5_CUT}, with {A5_MIN}, and within 5 of that from {A5_FLAT[0]} to "
   f"{A5_FLAT[1]}. The exact best cut, {B2[0]:.3f} with {B2[1]}, sits in the same "
   "flat stretch, so a tree grown on slightly different days can cut anywhere "
   "along it.")

md("---")

# ====================================================== B
section(
"## B · The first question\n\n"
"A cut is a mask, and the best cut is the one that leaves the smallest RSS. "
"These search for it with loops, then check the answer against scikit-learn. B2 "
"and B3 run together."
)

ex("B1", "A cut as a mask", 1,
   "Split the training days at a `vol_20d` of 1.0. Count the days on each side "
   "and print the mean `vol_next` of each side.",
   "calm = ...\nbusy = ...\ncounts = ...\nmeans = ...\n\nprint(counts)\nprint(means)",
   "`train[train['vol_20d'] <= 1.0]` keeps the calm side. Store the counts as a "
   "tuple `(len(calm), len(busy))`.",
   "calm = train[train['vol_20d'] <= 1.0]\nbusy = train[train['vol_20d'] > 1.0]\n"
   "counts = (len(calm), len(busy))\n"
   "means = (round(float(calm['vol_next'].mean()), 3), round(float(busy['vol_next'].mean()), 3))\n\n"
   "print(counts)\nprint(means)",
   f"{B1_COUNTS[0]:,} calm days forecast {B1_MEANS[0]} and {B1_COUNTS[1]} busy days "
   f"forecast {B1_MEANS[1]}. That is a stump written by hand: one question, two "
   "boxes, two means.")

ex("B2", "The best cut, by a loop", 4,
   "Write `best_split(rows, col)` that tries every distinct value of `rows[col]` "
   "as a cut, except the largest, and returns the cut with the smallest total "
   "RSS of `vol_next` and that total. Run it on the training days with "
   "`'vol_20d'`. It takes a second or two.",
   "def best_split(rows, col):\n    \"\"\"...\"\"\"\n    ...\n\n\nprint(best_split(train, 'vol_20d'))",
   ["`np.unique(rows[col])[:-1]` gives the sorted distinct values without the "
    "largest. Loop over them, build the two boxes of `rows['vol_next']` with "
    "masks, and add up their RSS.",
    "Keep the best so far in two variables that start as `None` and `np.inf`, "
    "and replace them with an `if total < best_total:`."],
   "def best_split(rows, col):\n"
   "    \"\"\"The cut on col with the smallest total RSS of vol_next, and that total.\"\"\"\n"
   "    y = rows['vol_next']\n    best, best_total = None, np.inf\n"
   "    for cut in np.unique(rows[col])[:-1]:\n"
   "        left, right = y[rows[col] <= cut], y[rows[col] > cut]\n"
   "        total = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()\n"
   "        if total < best_total:\n            best, best_total = cut, total\n"
   "    return round(float(best), 4), round(float(best_total), 1)\n\n\n"
   "print(best_split(train, 'vol_20d'))",
   f"`{B2}`: the best cut leaves {B2[1]}, down from {A5_NONE} with no cut. "
   "scikit-learn tries every value too. Keeping the best so far in a variable, "
   "and replacing it when something beats it, is the pattern behind every "
   "search by loop.")

ex("B3", "The same cut from a stump", 2,
   "Continuing from B2. Fit a `DecisionTreeRegressor` with `max_depth=1` on "
   "`vol_20d` alone. Its cut is `stump.tree_.threshold[0]`. Print it beside the "
   "cut from B2, and explain the difference in a comment.",
   "stump = ...\n...\ncut = ...\n\nprint(cut)\nprint(best_split(train, 'vol_20d'))\n\n# why do the two cuts differ?",
   "Fit on `train[['vol_20d']]` and `train['vol_next']`, then "
   "`round(float(stump.tree_.threshold[0]), 4)`.",
   "stump = DecisionTreeRegressor(max_depth=1, random_state=0)\n"
   "stump.fit(train[['vol_20d']], train['vol_next'])\n"
   "cut = round(float(stump.tree_.threshold[0]), 4)\n\n"
   "print(cut)\nprint(best_split(train, 'vol_20d'))\n\n"
   "# best_split cuts AT a training value; scikit-learn cuts halfway between the\n"
   "# last value on the left and the first on the right. The boxes are the same.",
   f"{B3_CUT} against {B2[0]}. The next training value above {B2[0]} is "
   f"{B3_NEXT}, and scikit-learn puts its cut halfway between the two. Every "
   "training day lands in the same box either way, so the RSS is the same.")

ex("B4", "Which column asks first", 3,
   "Fit a stump on each of the 19 columns alone and record its training RMSE in "
   "a dictionary. Which column makes the best first question? Then fit one "
   "stump on all 19 columns and check it picks the same one: its column number "
   "is `tree_.feature[0]`.",
   "errors = {}\nfor col in columns:\n    ...\n\nbest_col = ...\n"
   "together = ...\n...\nfirst = ...\n\nprint(best_col)\nprint(first)",
   ["Inside the loop, fit `DecisionTreeRegressor(max_depth=1, random_state=0)` "
    "on `train[[col]]` and store `rmse(train['vol_next'], s.predict(train[[col]]))`.",
    "`first = columns[together.tree_.feature[0]]`: the number indexes the list "
    "of names."],
   "errors = {}\nfor col in columns:\n"
   "    s = DecisionTreeRegressor(max_depth=1, random_state=0).fit(train[[col]], train['vol_next'])\n"
   "    errors[col] = round(float(rmse(train['vol_next'], s.predict(train[[col]]))), 4)\n\n"
   "best_col = min(errors, key=errors.get)\n"
   "together = DecisionTreeRegressor(max_depth=1, random_state=0)\n"
   "together.fit(train[columns], train['vol_next'])\nfirst = columns[together.tree_.feature[0]]\n\n"
   "print(best_col)\nprint(first)",
   f"Both are `{B4_BEST}`, with `{B4_SECOND}` second. The smallest training RMSE "
   "is the smallest RSS, because every stump is scored on the same days, so "
   "the loop and scikit-learn use one criterion. The stump on all 19 cuts at "
   f"{B4_CUT} and sends {B4_N_RIGHT} days to the busy side: the first question "
   "sets aside the most volatile weeks.")

ex("B5", "One question inside each box", 3,
   "Split the training days at 0.986 on `vol_20d`. Fit a stump on `pair` inside "
   "each half, and store in a dictionary which column each stump asks about and "
   "where it cuts.",
   "calm = train[train['vol_20d'] <= 0.986]\nbusy = train[train['vol_20d'] > 0.986]\n"
   "questions = {}\nfor name, rows in [('calm', calm), ('busy', busy)]:\n    ...\n\nprint(questions)",
   "Inside the loop: fit the stump on `rows[pair]` and `rows['vol_next']`, then "
   "`questions[name] = (pair[s.tree_.feature[0]], round(float(s.tree_.threshold[0]), 3))`.",
   "calm = train[train['vol_20d'] <= 0.986]\nbusy = train[train['vol_20d'] > 0.986]\n"
   "questions = {}\nfor name, rows in [('calm', calm), ('busy', busy)]:\n"
   "    s = DecisionTreeRegressor(max_depth=1, random_state=0).fit(rows[pair], rows['vol_next'])\n"
   "    questions[name] = (pair[s.tree_.feature[0]], round(float(s.tree_.threshold[0]), 3))\n\n"
   "print(questions)",
   f"`{B5}`. The calm box asks about the level again, the busy box about the "
   "direction. Two stumps grown in two boxes are exactly a tree of depth 2, and "
   "the different second questions are the interaction.")

md("---")

# ====================================================== C
section(
"## C · Fitting, drawing and reading trees\n\n"
"`plot_tree` shows a fitted tree, and its forecasts can be counted and traced. "
"Every exercise here fits its own tree."
)

ex("C1", "Draw a tree of depth 2", 1,
   "Fit a tree of depth 2 on `pair` and draw it with `plot_tree`, with the "
   "column names and filled boxes.",
   "small = ...\n...\n\nplt.figure(figsize=(10, 3))\n...\nplt.show()",
   "`plot_tree(small, feature_names=pair, filled=True, fontsize=10)`.",
   "small = DecisionTreeRegressor(max_depth=2, random_state=0)\n"
   "small.fit(train[pair], train['vol_next'])\n\n"
   "plt.figure(figsize=(10, 3))\nplot_tree(small, feature_names=pair, filled=True, fontsize=10)\nplt.show()",
   "Four leaves: 0.698 for 969 days, 0.994 for 185, 2.84 for 47 and 1.252 for "
   "693. The darkest box is the small one where volatility was high after a "
   "fall.")

ex("C2", "The forecasts a tree can give", 2,
   "Fit a tree of depth 2 on `pair`, forecast the test days, and count how many "
   "test days receive each distinct forecast.",
   "small = ...\n...\n\nforecasts = ...\ncounts = ...\n\nprint(counts)",
   "`pd.Series(small.predict(test[pair]), index=test.index).round(3)`, then "
   "`.value_counts()`.",
   "small = DecisionTreeRegressor(max_depth=2, random_state=0)\nsmall.fit(train[pair], train['vol_next'])\n\n"
   "forecasts = pd.Series(small.predict(test[pair]), index=test.index).round(3)\n"
   "counts = forecasts.value_counts()\n\nprint(counts)",
   f"Three forecasts for {N_TEST} days: {int(C2.iloc[0])} days get {C2.index[0]}, "
   f"{int(C2.iloc[1])} get {C2.index[1]} and {int(C2.iloc[2])} get {C2.index[2]}. "
   "The fourth leaf is never used, because no test day was volatile after a "
   "steep fall.")

ex("C3", "Depth and leaves", 1,
   "For depths 1 to 6, fit a tree on `pair` and store its number of leaves in a "
   "dictionary keyed by the depth.",
   "leaves = {}\nfor depth in range(1, 7):\n    ...\n\nprint(leaves)",
   "Inside the loop, fit the tree and store `t.get_n_leaves()`.",
   "leaves = {}\nfor depth in range(1, 7):\n"
   "    t = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(train[pair], train['vol_next'])\n"
   "    leaves[depth] = int(t.get_n_leaves())\n\nprint(leaves)",
   f"`{C3}`. Each level can at most double the leaves, so depth d allows 2 to "
   f"the power d. At depth 6 there are {C3[6]} rather than 64, because some "
   "boxes were already down to a single day.")

ex("C4", "One day through the tree", 2,
   "Fit a tree of depth 2 on `pair`. Make a one-row DataFrame for a day with "
   "`vol_20d` of 1.5 and `ret_20d` of -0.6, and forecast it. Then do the same "
   "with a `ret_20d` of 0.1.",
   "small = ...\n...\n\nday = ...\nother = ...\n\nprint(...)\nprint(...)",
   "`pd.DataFrame({'vol_20d': [1.5], 'ret_20d': [-0.6]})`. The columns must "
   "have the names the tree was fitted on.",
   "small = DecisionTreeRegressor(max_depth=2, random_state=0)\nsmall.fit(train[pair], train['vol_next'])\n\n"
   "day = pd.DataFrame({'vol_20d': [1.5], 'ret_20d': [-0.6]})\n"
   "other = pd.DataFrame({'vol_20d': [1.5], 'ret_20d': [0.1]})\n\n"
   "print(small.predict(day))\nprint(small.predict(other))",
   f"{C4_A} and {C4_B}: the same level of volatility gets a forecast more than "
   "twice as high after a steep fall. Linear regression on the same two columns "
   f"gives {C4_LIN[0]} and {C4_LIN[1]}. It also forecasts more after a fall, but "
   "by the same amount at every level of volatility; the tree asks about the "
   "fall only in busy weeks.")

ex("C5", "A deeper tree, drawn so it fits", 2,
   "Fit a tree of depth 3 on all 19 columns, and draw it so that its eight "
   "leaves fit on one screen: field names in the top box only, two decimals, "
   "no error line, and small text.",
   "tree = ...\n...\n\nplt.figure(figsize=(14, 4))\n...\nplt.show()",
   "`plot_tree(tree, feature_names=columns, filled=True, impurity=False, "
   "label='root', precision=2, fontsize=8)`.",
   "tree = DecisionTreeRegressor(max_depth=3, random_state=0)\ntree.fit(train[columns], train['vol_next'])\n\n"
   "plt.figure(figsize=(14, 4))\nplot_tree(tree, feature_names=columns, filled=True, impurity=False,\n"
   "          label='root', precision=2, fontsize=8)\nplt.show()",
   f"The first question is about `{C5_ROOT}`, and both second questions about "
   f"`{C5_SECOND}`, Disney's volatility rather than the index's own. Two of the "
   f"eight leaves hold {C5_SMALL[0]} and {C5_SMALL[1]} days. A tree this size "
   "spends its questions on a handful of extreme days, and it scores "
   f"{C5_TEST:.3f} on the test days.")

ex("C6", "Which leaf each day lands in", 3,
   "`.apply(X)` returns, for every row of `X`, the number of the leaf it lands "
   "in. Fit a tree of depth 3 on all 19 columns, apply it to the test days, "
   "count the days in each leaf, and print the share of test days in the "
   "busiest leaf.",
   "tree = ...\n...\n\nleaf_ids = ...\nper_leaf = ...\nshare_busiest = ...\n\nprint(per_leaf)\nprint(share_busiest)",
   "`pd.Series(tree.apply(test[columns])).value_counts()`, then its `.max()` "
   "divided by `len(test)`.",
   "tree = DecisionTreeRegressor(max_depth=3, random_state=0)\ntree.fit(train[columns], train['vol_next'])\n\n"
   "leaf_ids = tree.apply(test[columns])\nper_leaf = pd.Series(leaf_ids).value_counts()\n"
   "share_busiest = round(per_leaf.max() / len(test), 3)\n\nprint(per_leaf)\nprint(share_busiest)",
   f"The {N_TEST} test days use only {C6_N_LEAVES_USED} of the eight leaves, and "
   f"{C6_MAX} of them, {C6_PCT} percent, land in one. Those days all get "
   f"{C6_FORECAST}, while their volatility over the next 20 days averaged "
   f"{C6_ACTUAL:.2f}. A leaf learned in busier years forecasts a busy month for "
   "calm ones.")

md("---")

# ====================================================== D
section(
"## D · Depth and overfitting\n\n"
"Deeper trees fit the training days better and the test days worse. D3 and D4 "
"run together."
)

ex("D1", "A tree without a limit", 1,
   "Fit a tree with no depth limit on `pair`. Print its depth and number of "
   "leaves, then its training and test RMSE.",
   "full = ...\n...\nsize = ...\nerrors = ...\n\nprint(size)\nprint(errors)",
   "`size = (full.get_depth(), int(full.get_n_leaves()))` and the errors from `rmse` "
   "on the training and the test days.",
   "full = DecisionTreeRegressor(random_state=0)\nfull.fit(train[pair], train['vol_next'])\n"
   "size = (full.get_depth(), int(full.get_n_leaves()))\n"
   "errors = (round(float(rmse(train['vol_next'], full.predict(train[pair]))), 4),\n"
   "          round(float(rmse(test['vol_next'], full.predict(test[pair]))), 4))\n\n"
   "print(size)\nprint(errors)",
   f"Depth {D1[0]}, {D1[1]:,} leaves, a training RMSE of {D1[2]} and a test RMSE "
   f"of {D1[3]}. One leaf per training day reproduces every training day and "
   "forecasts the test days from single past days.")

ex("D2", "Training and test error against depth", 2,
   "For depths 1 to 15, fit a tree on `pair`, record the training and the test "
   "RMSE, and draw both against the depth with a legend.",
   "depths = range(1, 16)\ntrain_err = []\ntest_err = []\nfor depth in depths:\n    ...\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3))\n...\n...\n...\n"
   "ax.set_xlabel('max_depth')\nax.set_ylabel('RMSE')\nplt.show()",
   ["Inside the loop: fit, then append `rmse(train['vol_next'], t.predict(train[pair]))` "
    "and the same on the test days.",
    "`ax.plot(depths, train_err, label='training days')` and the same for the "
    "test days, so `ax.legend()` has names to show."],
   "depths = range(1, 16)\ntrain_err = []\ntest_err = []\nfor depth in depths:\n"
   "    t = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(train[pair], train['vol_next'])\n"
   "    train_err.append(rmse(train['vol_next'], t.predict(train[pair])))\n"
   "    test_err.append(rmse(test['vol_next'], t.predict(test[pair])))\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3))\n"
   "ax.plot(depths, train_err, marker='o', label='training days')\n"
   "ax.plot(depths, test_err, marker='o', label='test days')\n"
   "ax.set_xlabel('max_depth')\nax.set_ylabel('RMSE')\nax.legend()\nplt.show()",
   f"The training error falls at every step, from {D2_TRAIN[0]:.3f} to "
   f"{D2_TRAIN[-1]:.3f}. The test error is lowest at depths 1 and 2, "
   f"{min(D2_TEST):.3f}, and above 0.43 from depth 10 on. The gap between the "
   "two lines is the variance a deep tree carries.")

ex("D3", "Depth on the time-series folds", 2,
   "Search `max_depth` over 1, 2, 3, 4, 6 and 8 with `GridSearchCV`, `folds` "
   "from the setup cell, and RMSE as the score. Print the best depth, its mean "
   "fold RMSE and its test RMSE.",
   "search = ...\n...\n\nprint(...)\nprint(...)\nprint(...)",
   ["`GridSearchCV(DecisionTreeRegressor(random_state=0), {'max_depth': [...]}, "
    "cv=folds, scoring='neg_root_mean_squared_error')`, fitted on `train[pair]`.",
    "The score is negated, so the fold RMSE is `-search.best_score_`."],
   "search = GridSearchCV(DecisionTreeRegressor(random_state=0), {'max_depth': [1, 2, 3, 4, 6, 8]},\n"
   "                      cv=folds, scoring='neg_root_mean_squared_error')\n"
   "search.fit(train[pair], train['vol_next'])\n\n"
   "print(search.best_params_)\nprint(round(-search.best_score_, 4))\n"
   "print(round(float(rmse(test['vol_next'], search.predict(test[pair]))), 4))",
   f"Depth {D3_BEST}, with a mean fold RMSE of {D3_CV} and a test RMSE of "
   f"{D3_TEST}. The fold mean is much larger because the third fold, April 2019 "
   f"to July 2020, holds the crash of 2020 and scores {D3_FOLD_2020} on its own. "
   "Keep `search`: D4 continues from here.")

ex("D4", "The fold errors, as a table", 3,
   "Continuing from D3. Make a pandas Series of the mean fold RMSE for every "
   "depth, indexed by the depth and sorted from best to worst. The numbers are "
   "in `search.cv_results_`.",
   "by_depth = ...\n\nprint(by_depth)",
   ["`search.cv_results_['mean_test_score']` holds the negated scores, and "
    "`search.cv_results_['param_max_depth']` the depths in the same order.",
    "`pd.Series(-scores, index=list(depths)).round(4).sort_values()`."],
   "by_depth = pd.Series(-search.cv_results_['mean_test_score'],\n"
   "                     index=list(search.cv_results_['param_max_depth'])).round(4).sort_values()\n\n"
   "print(by_depth)",
   f"Depth {D4.index[0]} leads with {D4.iloc[0]}, then {D4.index[1]} with "
   f"{D4.iloc[1]}, and depth {D4.index[-1]} is last at {D4.iloc[-1]}. The order "
   "is not smooth: depth 3 does worse than 4 on these folds, which is why the "
   "search tries each value rather than assuming a shape.")

ex("D5", "A minimum leaf size", 2,
   "Search `min_samples_leaf` over 20, 50, 100, 200 and 400 on the folds, with "
   "no depth limit. Print the best value, the depth and number of leaves of the "
   "tree it chose, and its test RMSE.",
   "search_leaf = ...\n...\n\nprint(...)\nprint(...)\nprint(...)",
   "The chosen tree is `search_leaf.best_estimator_`, with `.get_depth()` and "
   "`.get_n_leaves()`.",
   "search_leaf = GridSearchCV(DecisionTreeRegressor(random_state=0),\n"
   "                           {'min_samples_leaf': [20, 50, 100, 200, 400]},\n"
   "                           cv=folds, scoring='neg_root_mean_squared_error')\n"
   "search_leaf.fit(train[pair], train['vol_next'])\n\n"
   "print(search_leaf.best_params_)\n"
   "print(search_leaf.best_estimator_.get_depth(), search_leaf.best_estimator_.get_n_leaves())\n"
   "print(round(float(rmse(test['vol_next'], search_leaf.predict(test[pair]))), 4))",
   f"Leaves of at least {D5_BEST} days, a tree of depth {D5_DEPTH} with "
   f"{D5_LEAVES} leaves, and a test RMSE of {D5_TEST}. A leaf size lets the tree "
   "go deep where days are plentiful and stop where they are not.")

ex("D6", "Pruning with ccp_alpha", 3,
   "Loop over `ccp_alpha` of 0, 0.005, 0.01, 0.02 and 0.05 on `pair`, with no "
   "depth limit. Store the number of leaves and the test RMSE of each pruned tree "
   "in a dictionary keyed by alpha.",
   "pruned = {}\nfor alpha in [0, 0.005, 0.01, 0.02, 0.05]:\n    ...\n\nprint(pruned)",
   "`DecisionTreeRegressor(ccp_alpha=alpha, random_state=0)`, then store a tuple "
   "of `int(t.get_n_leaves())` and the rounded test RMSE.",
   "pruned = {}\nfor alpha in [0, 0.005, 0.01, 0.02, 0.05]:\n"
   "    t = DecisionTreeRegressor(ccp_alpha=alpha, random_state=0).fit(train[pair], train['vol_next'])\n"
   "    pruned[alpha] = (int(t.get_n_leaves()), round(float(rmse(test['vol_next'], t.predict(test[pair]))), 4))\n\n"
   "print(pruned)",
   f"`{D6}`. A price of 0.02 per leaf cuts {D6[0][0]:,} leaves down to "
   f"{D6[0.02][0]}, and the test RMSE falls from {D6[0][1]} to {D6[0.02][1]}. At "
   "0.05 two more leaves merge and the score does not move, because no test day "
   "lands in either of them. Like ridge's alpha, a larger value gives a simpler "
   "model.")

md("---")

# ====================================================== E
section(
"## E · Averaging, and bagging\n\n"
"A deep tree changes with its sample of days; an average of many does not. "
"These draw the samples and build the average by hand before using "
"scikit-learn."
)

ex("E1", "One bootstrap sample", 2,
   "Draw a bootstrap sample of the training days with `random_state=1`: as many "
   "rows as the table, drawn with replacement. Print the share of distinct days "
   "in it, and the share the formula 1 minus (1 minus 1/n) to the power n "
   "gives.",
   "n = len(train)\nsample = ...\nshare = ...\nformula = ...\n\nprint(share)\nprint(formula)",
   "`train.sample(n=n, replace=True, random_state=1)`. The distinct days are "
   "`sample.index.nunique()`.",
   "n = len(train)\nsample = train.sample(n=n, replace=True, random_state=1)\n"
   "share = round(sample.index.nunique() / n, 3)\nformula = round(1 - (1 - 1 / n) ** n, 3)\n\n"
   "print(share)\nprint(formula)",
   f"{E1_SHARE} against {E1_FORMULA}. About a third of the days are missing from "
   "any one sample, and those days are the out-of-bag days a forest can score "
   "itself on.")

ex("E2", "Twenty samples", 3,
   "Repeat E1 for `random_state` 0 to 19. Collect the twenty shares in a list, "
   "and print their mean, the smallest and the largest.",
   "shares = []\nfor seed in range(20):\n    ...\n\nprint(...)\nprint(...)\nprint(...)",
   "Inside the loop, draw the sample with `random_state=seed` and append the "
   "rounded share. Then `np.mean(shares)`, `min(shares)` and `max(shares)`.",
   "shares = []\nfor seed in range(20):\n"
   "    s = train.sample(n=len(train), replace=True, random_state=seed)\n"
   "    shares.append(round(s.index.nunique() / len(train), 3))\n\n"
   "print(round(float(np.mean(shares)), 3))\nprint(min(shares))\nprint(max(shares))",
   f"A mean of {E2_MEAN}, from {E2_MIN} to {E2_MAX}. The share hardly moves from "
   "sample to sample, which is why 0.632 is quoted as a property of the "
   "bootstrap rather than of one draw.")

ex("E3", "How often each day is drawn", 3,
   "In one bootstrap sample, count how many times each training day was drawn, "
   "including the days drawn zero times. Then draw how many days were drawn 0, "
   "1, 2, 3 and more times, as a bar chart.",
   "sample = train.sample(n=len(train), replace=True, random_state=0)\n"
   "draws = ...\nhow_often = ...\n\nfig, ax = plt.subplots(figsize=(7, 3))\n...\n"
   "ax.set_xlabel('times a day was drawn')\nax.set_ylabel('days')\nplt.show()",
   ["`sample.index.value_counts()` counts the days that were drawn. "
    "`.reindex(train.index, fill_value=0)` adds the days that were not.",
    "`draws.value_counts().sort_index()` then counts how many days were drawn "
    "each number of times."],
   "sample = train.sample(n=len(train), replace=True, random_state=0)\n"
   "draws = sample.index.value_counts().reindex(train.index, fill_value=0)\n"
   "how_often = draws.value_counts().sort_index()\n\n"
   "fig, ax = plt.subplots(figsize=(7, 3))\nax.bar(how_often.index, how_often.values, color='#1c5cab')\n"
   "ax.set_xlabel('times a day was drawn')\nax.set_ylabel('days')\nplt.show()",
   f"{E3_ZERO} days were not drawn at all, and one day was drawn {E3_MAX} times. "
   "Each tree of a bagged model sees some days twice or more and a third of "
   "them not at all, which is where the differences between the trees come "
   "from.")

ex("E4", "Two trees, two halves", 3,
   "Split the training days into two random halves with `random_state=1`. Grow "
   "a full tree on `pair` in each half, then print the mean absolute difference "
   "between their test forecasts, the test RMSE of each tree, and the test RMSE "
   "of their average.",
   "half_a = ...\nhalf_b = ...\ntree_a = ...\ntree_b = ...\n...\n\nprint(...)\nprint(...)\nprint(...)",
   ["`half_a = train.sample(frac=0.5, random_state=1)` and "
    "`half_b = train.drop(half_a.index)`.",
    "With `p_a` and `p_b` the two forecasts, the average is `(p_a + p_b) / 2`."],
   "half_a = train.sample(frac=0.5, random_state=1)\nhalf_b = train.drop(half_a.index)\n"
   "tree_a = DecisionTreeRegressor(random_state=0).fit(half_a[pair], half_a['vol_next'])\n"
   "tree_b = DecisionTreeRegressor(random_state=0).fit(half_b[pair], half_b['vol_next'])\n"
   "p_a, p_b = tree_a.predict(test[pair]), tree_b.predict(test[pair])\n\n"
   "print(round(float(np.abs(p_a - p_b).mean()), 3))\n"
   "print(round(float(rmse(test['vol_next'], p_a)), 3), round(float(rmse(test['vol_next'], p_b)), 3))\n"
   "print(round(float(rmse(test['vol_next'], (p_a + p_b) / 2)), 3))",
   f"The two trees differ by {E4_DIFF} on the average test day. Alone they score "
   f"{E4_RA} and {E4_RB}; their average scores {E4_AVG}. Two noisy forecasts "
   "that make different mistakes average out to a better one.")

ex("E5", "Bagging with a list of trees", 3,
   "Grow 25 full trees on the 19 columns, each on its own bootstrap sample with "
   "`random_state` 0 to 24, and keep them in a list. Stack their test forecasts "
   "into one array and average them with `np.mean(..., axis=0)`. Print the "
   "RMSE of the average and of the first tree alone. It takes a few seconds.",
   "trees = []\nfor b in range(25):\n    ...\n\nforecasts = ...\naverage = ...\n\nprint(...)\nprint(...)",
   ["Inside the loop: draw the sample, fit a `DecisionTreeRegressor(random_state=0)`, "
    "and `trees.append(...)` it.",
    "`forecasts = np.array([t.predict(test[columns]) for t in trees])` has one "
    "row per tree, so `axis=0` averages down the columns, day by day."],
   "trees = []\nfor b in range(25):\n"
   "    s = train.sample(n=len(train), replace=True, random_state=b)\n"
   "    trees.append(DecisionTreeRegressor(random_state=0).fit(s[columns], s['vol_next']))\n\n"
   "forecasts = np.array([t.predict(test[columns]) for t in trees])\n"
   "average = np.mean(forecasts, axis=0)\n\n"
   "print(round(float(rmse(test['vol_next'], average)), 4))\n"
   "print(round(float(rmse(test['vol_next'], forecasts[0])), 4))",
   f"{E5} for the average against {E5_ONE} for the first tree alone. The array "
   "has one row per tree and one column per test day, and `axis=0` is the "
   "direction that averages trees rather than days.")

ex("E6", "BaggingRegressor and its first questions", 2,
   "Fit `BaggingRegressor` with 25 full trees and `random_state=0` on the 19 "
   "columns. Print its test RMSE, and count which column each tree asks about "
   "first. The fitted trees are in `bag.estimators_`.",
   "bag = ...\n...\nfirst = ...\n\nprint(...)\nprint(...)",
   ["`BaggingRegressor(DecisionTreeRegressor(), n_estimators=25, random_state=0)`.",
    "Loop over `bag.estimators_` and append `columns[t.tree_.feature[0]]` to a "
    "list, then `pd.Series(first).value_counts()`."],
   "bag = BaggingRegressor(DecisionTreeRegressor(), n_estimators=25, random_state=0)\n"
   "bag.fit(train[columns], train['vol_next'])\nfirst = []\nfor t in bag.estimators_:\n"
   "    first.append(columns[t.tree_.feature[0]])\n\n"
   "print(round(float(rmse(test['vol_next'], bag.predict(test[columns]))), 4))\n"
   "print(pd.Series(first).value_counts())",
   f"A test RMSE of {E6_TEST}. The first questions are `{E6.to_dict()}`: "
   f"{E6_VOL} of the 25 trees open with a volatility column, and "
   f"{int(E6.iloc[0])} with `{E6.index[0]}`. Trees that ask alike make mistakes "
   "alike, and averaging cancels less of them. That is what a forest's "
   "`max_features` sets out to change.")

ex("E7", "Out-of-bag, by hand", 5,
   "Grow 25 full trees on the 19 columns, each on a bootstrap sample drawn "
   "with `random_state` 0 to 24, as in E5. This time, for every training day, "
   "add up the forecasts of only the trees whose sample left that day out, and "
   "count those trees. Divide to get each day's out-of-bag forecast, and print "
   "its RMSE on the training days. It takes a few seconds.",
   "n = len(train)\ntotal = np.zeros(n)\ncount = np.zeros(n)\nfor b in range(25):\n    ...\n\n"
   "oob = ...\n\nprint(...)",
   ["`left_out = ~train.index.isin(sample.index)` is True for the days the "
    "sample missed. `total[left_out] += tree.predict(train[columns][left_out])` "
    "adds the forecasts in the right places.",
    "`count[left_out] += 1` counts the trees, and `oob = total / count` divides "
    "day by day. Every day was left out by at least two trees here, so nothing "
    "is divided by zero."],
   "n = len(train)\ntotal = np.zeros(n)\ncount = np.zeros(n)\nfor b in range(25):\n"
   "    sample = train.sample(n=n, replace=True, random_state=b)\n"
   "    tree = DecisionTreeRegressor(random_state=0).fit(sample[columns], sample['vol_next'])\n"
   "    left_out = ~train.index.isin(sample.index)\n"
   "    total[left_out] += tree.predict(train[columns][left_out])\n"
   "    count[left_out] += 1\n\n"
   "oob = total / count\n\nprint(round(float(rmse(train['vol_next'], oob)), 4))",
   f"{E7_OOB:.4f}, with each day left out by {E7_TREES} trees on average. That is "
   f"better than the {E5} the same trees score on the test days, although the "
   "training days include 2020 and the test years were calm. A day a tree left "
   "out still had its neighbours in the sample, and neighbouring days share "
   "most of a 20-day window. F4 puts this number beside the folds.")

md("---")

# ====================================================== F
section(
"## F · Random forests\n\n"
"A forest is bagging where each question sees only some of the columns. These "
"count what that changes, and check the out-of-bag score against the folds."
)

ex("F1", "A forest with a third of the columns", 1,
   "Fit a random forest of 100 trees with `max_features=0.33`, a third of the "
   "columns at each question, and `random_state=0`. Print its test RMSE and the "
   "number of columns each question chooses among.",
   "forest = ...\n...\n\nprint(...)\nprint(...)",
   "A float `max_features` is a share: each question sees "
   "`int(0.33 * len(columns))` columns.",
   "forest = RandomForestRegressor(n_estimators=100, max_features=0.33, random_state=0)\n"
   "forest.fit(train[columns], train['vol_next'])\n\n"
   "print(round(float(rmse(test['vol_next'], forest.predict(test[columns]))), 4))\n"
   "print(int(0.33 * len(columns)))",
   f"{F1} with {F1_EACH} columns per question, against 0.2835 for the lecture's 4 "
   "columns. `max_features` is one setting among several, and the next "
   "exercises count what it does.")

ex("F2", "How many columns open the trees", 3,
   "For `max_features` of 1, 4 and 19, fit a forest of 100 trees and count how "
   "many DIFFERENT columns ask the first question across its trees. Store the "
   "counts in a dictionary.",
   "variety = {}\nfor m in [1, 4, 19]:\n    ...\n\nprint(variety)",
   "Inside the loop, fit the forest, collect `columns[t.tree_.feature[0]]` for "
   "every tree in `f.estimators_`, and store `pd.Series(first).nunique()`.",
   "variety = {}\nfor m in [1, 4, 19]:\n"
   "    f = RandomForestRegressor(n_estimators=100, max_features=m, random_state=0)\n"
   "    f.fit(train[columns], train['vol_next'])\n"
   "    first = [columns[t.tree_.feature[0]] for t in f.estimators_]\n"
   "    variety[m] = int(pd.Series(first).nunique())\n\nprint(variety)",
   f"`{F2}`. With all 19 columns to choose from, the 100 trees open with only "
   f"{F2[19]} different columns. With one column drawn at random for each "
   f"question, {F2[1]} of the 19 open some tree. Less choice at each question "
   "is what makes the trees differ.")

ex("F3", "The error as trees are added", 4,
   "Fit a forest of 200 trees with `max_features='sqrt'`. Take each tree's test "
   "forecasts from `forest.estimators_`, and compute the RMSE of the average of "
   "the first 1, 2, ..., 200 trees. Draw it on a log x-axis, with a dashed line "
   "at linear regression's test RMSE on `vol_20d`.",
   "forest = RandomForestRegressor(n_estimators=200, max_features='sqrt', random_state=0)\n"
   "forest.fit(train[columns], train['vol_next'])\n\n"
   "each = ...\nrunning = ...\nerrors = ...\nline = ...\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3))\n...\n...\n"
   "ax.set_xscale('log')\nax.set_xlabel('trees averaged')\nax.set_ylabel('test RMSE')\nplt.show()",
   ["`each = np.array([e.predict(test[columns].to_numpy()) for e in forest.estimators_])` "
    "has one row per tree. `each.cumsum(axis=0)` adds the trees up in order; "
    "divide row n by n + 1 for the running average.",
    "`running = each.cumsum(axis=0) / np.arange(1, 201)[:, None]`, then one RMSE "
    "per row. The dashed line is `ax.axhline(line, linestyle='--')`."],
   "forest = RandomForestRegressor(n_estimators=200, max_features='sqrt', random_state=0)\n"
   "forest.fit(train[columns], train['vol_next'])\n\n"
   "each = np.array([e.predict(test[columns].to_numpy()) for e in forest.estimators_])\n"
   "running = each.cumsum(axis=0) / np.arange(1, 201)[:, None]\n"
   "errors = [rmse(test['vol_next'], row) for row in running]\n"
   "line = rmse(test['vol_next'], LinearRegression().fit(train[['vol_20d']], train['vol_next'])\n"
   "            .predict(test[['vol_20d']]))\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3))\n"
   "ax.plot(range(1, 201), errors, color='#1c5cab', label='random forest')\n"
   "ax.axhline(line, color='grey', linestyle='--', label='linear regression on vol_20d')\n"
   "ax.set_xscale('log')\nax.set_xlabel('trees averaged')\nax.set_ylabel('test RMSE')\nax.legend()\nplt.show()",
   f"The error falls fast over the first ten trees, from {F3[1]} to {F3[10]}, "
   f"then drifts down with small wiggles: {F3[100]} at 100 and {F3[200]} at 200. "
   f"It stays above the dashed line at {LINE_ONE} throughout. Adding trees does "
   "not overfit; it only stops helping, and every tree still costs time.")

ex("F4", "Out-of-bag on days", 3,
   "Fit a forest of 100 trees with `max_features='sqrt'`, `min_samples_leaf=50` "
   "and `oob_score=True`. Print its out-of-bag RMSE on the training days, its "
   "mean RMSE on the time-series folds, and its test RMSE. The folds take a few "
   "seconds.",
   "forest = ...\n...\noob = ...\non_folds = ...\n\nprint(oob)\nprint(on_folds)\nprint(...)",
   ["`forest.oob_prediction_` holds a forecast for every training day from the "
    "trees that did not see it: `rmse(train['vol_next'], forest.oob_prediction_)`.",
    "`cross_val_score(forest, train[columns], train['vol_next'], cv=folds, "
    "scoring='neg_root_mean_squared_error')`, negated and averaged."],
   "forest = RandomForestRegressor(n_estimators=100, max_features='sqrt', min_samples_leaf=50,\n"
   "                               oob_score=True, random_state=0)\n"
   "forest.fit(train[columns], train['vol_next'])\n"
   "oob = round(float(rmse(train['vol_next'], forest.oob_prediction_)), 4)\n"
   "on_folds = round(float(-cross_val_score(forest, train[columns], train['vol_next'], cv=folds,\n"
   "                                        scoring='neg_root_mean_squared_error').mean()), 4)\n\n"
   "print(oob)\nprint(on_folds)\nprint(round(float(rmse(test['vol_next'], forest.predict(test[columns]))), 4))",
   f"{F4_OOB} out-of-bag against {F4_FOLDS} on folds that score only later days, "
   f"and {F4_TEST} on the test days. Larger leaves narrow the gap the lecture "
   "showed for full trees, but out-of-bag still flatters: a day left out of a "
   "tree has its neighbours in the bag.")

ex("F5", "max_features on the folds", 3,
   "Score forests of 100 trees with `min_samples_leaf=50` and `max_features` of "
   "1, 2, 4, 8 and 19 on the time-series folds. Store each one's mean fold RMSE "
   "in a dictionary and print the best setting. It takes a few seconds.",
   "by_features = {}\nfor m in [1, 2, 4, 8, 19]:\n    ...\n\nbest = ...\n\nprint(by_features)\nprint(best)",
   ["Inside the loop, build the forest and score it with `cross_val_score(f, "
    "train[columns], train['vol_next'], cv=folds, "
    "scoring='neg_root_mean_squared_error')`, negated and averaged.",
    "`min(by_features, key=by_features.get)` gives the key with the smallest "
    "value."],
   "by_features = {}\nfor m in [1, 2, 4, 8, 19]:\n"
   "    f = RandomForestRegressor(n_estimators=100, max_features=m, min_samples_leaf=50, random_state=0)\n"
   "    scores = cross_val_score(f, train[columns], train['vol_next'], cv=folds,\n"
   "                             scoring='neg_root_mean_squared_error')\n"
   "    by_features[m] = round(float(-scores.mean()), 4)\n\n"
   "best = min(by_features, key=by_features.get)\n\nprint(by_features)\nprint(best)",
   f"`{F5}`, best at {F5_BEST}. The error rises with every column added to the "
   "choice, and at 19 the forest is bagging with leaves of 50, the worst of the "
   "five. The folds choose the setting, so the test days stay unopened for the "
   "model you keep. F6 adds a second setting to the same loop.")

ex("F6", "Two settings in one loop", 4,
   "Score forests of 100 trees on the time-series folds for every pair of "
   "`max_features` in [1, 4] and `min_samples_leaf` in [20, 50, 100], with two "
   "nested loops. Store the mean fold RMSE in a dictionary keyed by the pair, "
   "and print the best pair. It takes about ten seconds.",
   "scores = {}\nfor m in [1, 4]:\n    for leaf in [20, 50, 100]:\n        ...\n\nbest_pair = ...\n\n"
   "print(scores)\nprint(best_pair)",
   "Key the dictionary with a tuple, `scores[(m, leaf)] = ...`, and score with "
   "`cross_val_score(..., cv=folds, scoring='neg_root_mean_squared_error')`.",
   "scores = {}\nfor m in [1, 4]:\n    for leaf in [20, 50, 100]:\n"
   "        f = RandomForestRegressor(n_estimators=100, max_features=m, min_samples_leaf=leaf,\n"
   "                                  random_state=0)\n"
   "        scores[(m, leaf)] = round(float(-cross_val_score(f, train[columns], train['vol_next'],\n"
   "                                  cv=folds, scoring='neg_root_mean_squared_error').mean()), 4)\n\n"
   "best_pair = min(scores, key=scores.get)\n\nprint(scores)\nprint(best_pair)",
   f"The best pair is `{F6_BEST}` with {F6[F6_BEST]}. Two nested loops over a "
   "dictionary of results is exactly what `GridSearchCV` does for you, which "
   "section G uses next.")

md("---")

# ====================================================== G
section(
"## G · Tuning, and what it costs\n\n"
"A grid multiplies: every setting you add multiplies the fits. These count, "
"time and search grids, and check whether a bigger forest is a better one."
)

ex("G1", "How big is a grid", 1,
   "A grid is a dictionary of lists. With a loop, compute how many combinations "
   "the grid below has, and how many fits that means on five folds.",
   "grid = {'max_features': [1, 2, 4, 8, 19],\n        'min_samples_leaf': [5, 20, 50, 100, 200, 400],\n"
   "        'n_estimators': [100, 300, 500],\n        'max_depth': [3, 5, 10, None]}\n\n"
   "combinations = 1\nfor values in grid.values():\n    ...\n\nfits = ...\n\nprint(combinations)\nprint(fits)",
   "Inside the loop: `combinations = combinations * len(values)`. The fits are "
   "the combinations times 5.",
   "grid = {'max_features': [1, 2, 4, 8, 19],\n        'min_samples_leaf': [5, 20, 50, 100, 200, 400],\n"
   "        'n_estimators': [100, 300, 500],\n        'max_depth': [3, 5, 10, None]}\n\n"
   "combinations = 1\nfor values in grid.values():\n    combinations = combinations * len(values)\n\n"
   "fits = combinations * 5\n\nprint(combinations)\nprint(fits)",
   f"{G1_COMBOS} combinations and {G1_COMBOS * 5:,} fits, each a forest of up to "
   "500 trees. This is the grid that took 14 minutes on one core in the "
   "lecture; adding one more list of four values would make it an hour.")

ex("G2", "Timing one fit", 2,
   "Time one fit of a forest of 100 trees with `max_features='sqrt'`, using "
   "`time.perf_counter()` before and after. Then estimate, in minutes, how long "
   "the 1,800 fits of G1 would take if every fit took as long.",
   "forest = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=0)\n\n"
   "start = ...\n...\nseconds = ...\nminutes = ...\n\nprint(seconds)\nprint(minutes)",
   "`start = time.perf_counter()`, then the fit, then "
   "`seconds = time.perf_counter() - start`.",
   "forest = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=0)\n\n"
   "start = time.perf_counter()\nforest.fit(train[columns], train['vol_next'])\n"
   "seconds = round(time.perf_counter() - start, 2)\nminutes = round(seconds * 1800 / 60, 1)\n\n"
   "print(seconds)\nprint(minutes)",
   "The lecture's machine took 0.57 seconds, which makes about 17 minutes for "
   "the grid; your number depends on your machine. The estimate is rough, "
   "because 300 or 500 trees take longer and shallow trees take less, but it "
   "tells you before you start whether to wait or to shrink the grid.")

ex("G3", "A grid as three lines", 4,
   "Search `max_features` in [1, 4, 19] and `min_samples_leaf` in [5, 20, 50, "
   "100, 200] for a forest of 50 trees on the folds. Put the mean fold RMSE into "
   "a table with one column per `max_features`, and draw one line per column "
   "against the leaf size on a log axis. It takes up to half a minute.",
   "search = ...\n...\nresults = ...\n\nfig, ax = plt.subplots(figsize=(8, 3))\n...\n"
   "ax.set_xscale('log')\nax.set_xlabel('min_samples_leaf')\nax.set_ylabel('mean fold RMSE')\nplt.show()",
   ["`pd.DataFrame(search.cv_results_).pivot(index='param_min_samples_leaf', "
    "columns='param_max_features', values='mean_test_score')` gives the table; "
    "multiply by -1.",
    "`for m in results.columns: ax.plot(results.index, results[m], marker='o', "
    "label=f'max_features={m}')`, then `ax.legend()`."],
   "search = GridSearchCV(RandomForestRegressor(n_estimators=50, random_state=0),\n"
   "                      {'max_features': [1, 4, 19], 'min_samples_leaf': [5, 20, 50, 100, 200]},\n"
   "                      cv=folds, scoring='neg_root_mean_squared_error')\n"
   "search.fit(train[columns], train['vol_next'])\n"
   "results = pd.DataFrame(search.cv_results_).pivot(index='param_min_samples_leaf',\n"
   "                                                 columns='param_max_features',\n"
   "                                                 values='mean_test_score') * -1\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3))\nfor m in results.columns:\n"
   "    ax.plot(results.index, results[m], marker='o', label=f'max_features={m}')\n"
   "ax.set_xscale('log')\nax.set_xlabel('min_samples_leaf')\nax.set_ylabel('mean fold RMSE')\n"
   "ax.legend()\nplt.show()",
   f"The best pair is `{G3_BEST}` with {G3_CV}, and the worst cell is "
   f"{G3_WORST}. The line for all 19 columns sits highest at every leaf size, "
   "and each line is lowest at a leaf size of 50 or 100, not at either end. "
   "Both settings matter, and they matter together.")

ex("G4", "A random sample of a grid", 3,
   "`RandomizedSearchCV` tries `n_iter` combinations drawn at random instead of "
   "all of them. Search the grid below with `n_iter=10`, `random_state=0` and a "
   "forest of 50 trees on the folds. Print the best settings, the mean fold "
   "RMSE and the test RMSE. It takes about ten seconds.",
   "grid = {'max_features': [1, 2, 4, 8, 19],\n        'min_samples_leaf': [5, 20, 50, 100, 200, 400],\n"
   "        'max_depth': [3, 5, 10, None]}\n\nsampled = ...\n...\n\nprint(...)\nprint(...)\nprint(...)",
   "`RandomizedSearchCV(RandomForestRegressor(n_estimators=50, random_state=0), "
   "grid, n_iter=10, cv=folds, scoring='neg_root_mean_squared_error', "
   "random_state=0)`. It is read like a `GridSearchCV`.",
   "grid = {'max_features': [1, 2, 4, 8, 19],\n        'min_samples_leaf': [5, 20, 50, 100, 200, 400],\n"
   "        'max_depth': [3, 5, 10, None]}\n\n"
   "sampled = RandomizedSearchCV(RandomForestRegressor(n_estimators=50, random_state=0), grid,\n"
   "                             n_iter=10, cv=folds, scoring='neg_root_mean_squared_error',\n"
   "                             random_state=0)\nsampled.fit(train[columns], train['vol_next'])\n\n"
   "print(sampled.best_params_)\nprint(round(-sampled.best_score_, 4))\n"
   "print(round(float(rmse(test['vol_next'], sampled.predict(test[columns]))), 4))",
   f"`{G4_BEST}`, with {G4_CV} on the folds and {G4_TEST} on the test days. "
   "These are the settings the lecture's 14-minute search chose, found here "
   "from 10 of the 120 combinations. A random sample will not always land on "
   "the best corner, but it usually lands near it, at a fraction of the cost.")

ex("G5", "More trees", 2,
   "Fit forests with `max_features='sqrt'`, `min_samples_leaf=50` and 25, 50, "
   "100, 200 and 400 trees. Print one line per forest with the number of trees "
   "and the test RMSE, lined up with field widths.",
   "for n in [25, 50, 100, 200, 400]:\n    ...",
   "Fit inside the loop, then `print(f\"{n:>5}{error:>9.4f}\")`.",
   "for n in [25, 50, 100, 200, 400]:\n"
   "    f = RandomForestRegressor(n_estimators=n, max_features='sqrt', min_samples_leaf=50, random_state=0)\n"
   "    f.fit(train[columns], train['vol_next'])\n"
   "    error = rmse(test['vol_next'], f.predict(test[columns]))\n    print(f\"{n:>5}{error:>9.4f}\")",
   f"From {G5[25]} with 25 trees to {G5[400]} with 400, and {G5[100]} at 100. "
   "Sixteen times the trees buy almost nothing, and the fit takes sixteen times "
   "as long.")

ex("G6", "Every core", 2,
   "Time the same forest of 200 trees twice: once with `n_jobs=1` and once with "
   "`n_jobs=-1`, which uses every core. Print both times and how many times "
   "faster the second was.",
   "times = {}\nfor jobs in [1, -1]:\n    ...\n\nprint(times)\nprint(...)",
   "Inside the loop, build the forest with `n_jobs=jobs`, time the fit with "
   "`time.perf_counter()`, and store the seconds in `times[jobs]`.",
   "times = {}\nfor jobs in [1, -1]:\n"
   "    f = RandomForestRegressor(n_estimators=200, max_features='sqrt', random_state=0, n_jobs=jobs)\n"
   "    start = time.perf_counter()\n    f.fit(train[columns], train['vol_next'])\n"
   "    times[jobs] = round(time.perf_counter() - start, 2)\n\n"
   "print(times)\nprint(round(times[1] / times[-1], 1))",
   "The speed-up depends on how many cores your machine has, and it is always "
   "less than that number, because handing out the work takes time too. The "
   "trees of a forest are independent, so they can be grown side by side, and "
   "the forecasts are identical either way.")

md("---")

# ====================================================== H
section(
"## H · Trees for a label\n\n"
"The same cuts, with the share of each class in every leaf. These use the "
"credit table, split as in Session 9, and start from the error of a box of "
"0s and 1s."
)

ex("H1", "The error of a box of labels", 3,
   "For labels of 0 and 1, a box with n rows and k ones has an RSS of "
   "k(n minus k)/n, and a Gini of 2p(1 minus p), where p = k/n is the share of "
   "ones. Check both on the credit training rows: compute the RSS of `default` "
   "directly and from the formula, then Gini, and show that Gini is twice the "
   "RSS per row.",
   "y = c_train['default']\n\nn = ...\nk = ...\ndirect = ...\nformula = ...\n"
   "gini = ...\ntwice_per_row = ...\n\nprint(direct, formula)\nprint(gini, twice_per_row)",
   ["`n = len(y)` and `k = int(y.sum())`. The direct RSS is "
    "`((y - y.mean()) ** 2).sum()`, the same line as A1 with `y` in place of "
    "the array.",
    "With `p = k / n`, Gini is `2 * p * (1 - p)`, and twice the RSS per row is "
    "`2 * direct / n`. Round each so the pairs can be compared."],
   "y = c_train['default']\n\nn = len(y)\nk = int(y.sum())\n"
   "direct = round(float(((y - y.mean()) ** 2).sum()), 2)\nformula = round(k * (n - k) / n, 2)\n"
   "gini = round(2 * (k / n) * (1 - k / n), 4)\ntwice_per_row = round(2 * direct / n, 4)\n\n"
   "print(direct, formula)\nprint(gini, twice_per_row)",
   f"Both RSS values are {H1_DIRECT:,}, from {H1_K:,} defaults among {H1_N:,} "
   f"borrowers, and both of the others are {H1_GINI}: the 0.345 that `plot_tree` "
   "prints in the top box of a credit tree. A box of labels needs only its two "
   "counts to know its error. Gini is the squared error per row, doubled, so a "
   "classification tree chooses the same cuts a regression tree would on the "
   "0s and 1s.")

ex("H2", "A classification tree", 1,
   "Fit a `DecisionTreeClassifier` of depth 2 on the seven credit columns. Print "
   "its test AUC and the distinct probabilities of default it gives.",
   "c_tree = ...\n...\np_tree = ...\n\nprint(...)\nprint(...)",
   "`p_tree = c_tree.predict_proba(c_test[c_columns])[:, 1]`, then "
   "`roc_auc_score(c_test['default'], p_tree)` and `np.unique(p_tree).round(3)`.",
   "c_tree = DecisionTreeClassifier(max_depth=2, random_state=0)\n"
   "c_tree.fit(c_train[c_columns], c_train['default'])\np_tree = c_tree.predict_proba(c_test[c_columns])[:, 1]\n\n"
   "print(round(roc_auc_score(c_test['default'], p_tree), 4))\nprint(np.unique(p_tree).round(3))",
   f"An AUC of {H1_AUC} from four probabilities, {H1_PROBS}. Four distinct "
   "numbers can rank 9,000 borrowers only in four groups, which caps how high "
   "a shallow tree's AUC can go.")

ex("H3", "Draw the classification tree", 1,
   "Fit a classification tree of depth 2 on the seven credit columns, and draw "
   "it with shares instead of counts, filled boxes, no Gini line, and the "
   "classes named `'repaid'` and `'defaulted'` with `class_names`.",
   "c_tree = ...\n...\n\nplt.figure(figsize=(11, 3))\n...\nplt.show()",
   "`plot_tree(c_tree, feature_names=c_columns, filled=True, proportion=True, "
   "impurity=False, class_names=['repaid', 'defaulted'], fontsize=9)`.",
   "c_tree = DecisionTreeClassifier(max_depth=2, random_state=0)\nc_tree.fit(c_train[c_columns], c_train['default'])\n\n"
   "plt.figure(figsize=(11, 3))\n"
   "plot_tree(c_tree, feature_names=c_columns, filled=True, proportion=True, impurity=False,\n"
   "          class_names=['repaid', 'defaulted'], fontsize=9)\nplt.show()",
   "The first question is `late_now <= 1.5`. Borrowers up to one month behind "
   "are then asked about `months_late`, and those two or more months behind "
   f"about `utilisation`: the {H2_SMALL_N} of them who used almost none of their "
   f"limit defaulted at {H2_SMALL_P}, the rest at {H2_LAST_P}. Only that last "
   "leaf has `defaulted` as its larger class.")

ex("H4", "How much the first question lowers Gini", 3,
   "Compute Gini, 2p(1 minus p), for all credit training rows, and for the two "
   "boxes of the first question, `late_now` of 1 or less and above 1. Then "
   "compute the average of the two boxes' Gini weighted by their sizes.",
   "def gini(labels):\n    \"\"\"...\"\"\"\n    ...\n\n\n"
   "left = c_train[c_train['late_now'] <= 1]['default']\nright = c_train[c_train['late_now'] > 1]['default']\n\n"
   "root = ...\nweighted = ...\n\nprint(root)\nprint(weighted)",
   ["`gini` returns `2 * p * (1 - p)` with `p = labels.mean()`.",
    "The weighted average is `(len(left) * gini(left) + len(right) * gini(right)) / len(c_train)`."],
   "def gini(labels):\n    \"\"\"Gini of a box of 0/1 labels: 2p(1 - p).\"\"\"\n"
   "    p = labels.mean()\n    return 2 * p * (1 - p)\n\n\n"
   "left = c_train[c_train['late_now'] <= 1]['default']\nright = c_train[c_train['late_now'] > 1]['default']\n\n"
   "root = round(float(gini(c_train['default'])), 4)\n"
   "weighted = round(float((len(left) * gini(left) + len(right) * gini(right)) / len(c_train)), 4)\n\n"
   "print(root)\nprint(weighted)",
   f"From {H3_ROOT} to {H3_WEIGHTED}: the left box has a Gini of {H3_LEFT} and "
   f"the right box {H3_RIGHT}. The right box is less pure than the whole table, "
   f"but it holds only {H3_N_RIGHT:,} borrowers, {H3_SHARE_RIGHT} percent, while "
   "the left box, with everyone else, became purer. That fall in the weighted "
   "average is what the tree made as large as it could when it chose this "
   "question.")

ex("H5", "A forest for the label", 3,
   "Fit a random forest of 100 trees with `min_samples_leaf=50` and "
   "`oob_score=True` on the credit table. Print its out-of-bag AUC, its test "
   "AUC, and the test AUC of a scaled logistic regression beside them.",
   "c_forest = ...\n...\nlogistic = ...\n...\n\nprint(...)\nprint(...)\nprint(...)",
   ["The out-of-bag probabilities are `c_forest.oob_decision_function_[:, 1]`, "
    "scored against `c_train['default']`.",
    "`Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])` "
    "for the comparison."],
   "c_forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=50, oob_score=True,\n"
   "                                  random_state=0)\nc_forest.fit(c_train[c_columns], c_train['default'])\n"
   "logistic = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "logistic.fit(c_train[c_columns], c_train['default'])\n\n"
   "print(round(roc_auc_score(c_train['default'], c_forest.oob_decision_function_[:, 1]), 4))\n"
   "print(round(roc_auc_score(c_test['default'], c_forest.predict_proba(c_test[c_columns])[:, 1]), 4))\n"
   "print(round(roc_auc_score(c_test['default'], logistic.predict_proba(c_test[c_columns])[:, 1]), 4))",
   f"{H5_OOB:.3f} out-of-bag and {H5_TEST:.3f} on the test borrowers, against "
   f"{H5_LOGIT:.3f} for logistic regression. Here the out-of-bag score is close to the test "
   "score, because the rows are separate borrowers and a left-out borrower has "
   "no neighbour in the bag.")

ex("H6", "The default rate in each leaf", 3,
   "Fit the depth-2 classification tree, find the leaf of every training "
   "borrower with `.apply()`, and draw the share that defaulted in each leaf as "
   "bars, with a dashed line at the share for everyone.",
   "c_tree = ...\n...\n\n"
   "leaf = ...\nby_leaf = ...\n\nfig, ax = plt.subplots(figsize=(7, 3))\n...\n...\n"
   "ax.set_xlabel('leaf')\nax.set_ylabel('share that defaulted')\nplt.show()",
   ["`leaf = pd.Series(c_tree.apply(c_train[c_columns]), index=c_train.index)`, then "
    "`c_train['default'].groupby(leaf).mean()`.",
    "Draw the bars against the leaf numbers as text, `by_leaf.index.astype(str)`, "
    "so they sit side by side."],
   "c_tree = DecisionTreeClassifier(max_depth=2, random_state=0)\nc_tree.fit(c_train[c_columns], c_train['default'])\n\n"
   "leaf = pd.Series(c_tree.apply(c_train[c_columns]), index=c_train.index)\n"
   "by_leaf = c_train['default'].groupby(leaf).mean()\n\n"
   "fig, ax = plt.subplots(figsize=(7, 3))\nax.bar(by_leaf.index.astype(str), by_leaf.values, color='#1c5cab')\n"
   "ax.axhline(c_train['default'].mean(), color='grey', linestyle='--')\n"
   "ax.set_xlabel('leaf')\nax.set_ylabel('share that defaulted')\nplt.show()",
   f"The four leaves default at {', '.join(f'{v:.2f}' for v in H6.iloc[:-1])} and "
   f"{H6.iloc[-1]:.2f}, around the dashed line at {H6_SHARE}. Each bar is exactly "
   "the probability the tree gives every borrower in that leaf.")

md("---")

# ====================================================== I
I2_RMSE = round(rmse(YT, SMALL.predict(TEST[PAIR])), 4)

section(
"## I · When it goes wrong\n\n"
"Five mistakes that are easy to make with trees. Two of them stop with an "
"error. Three run quietly and print a wrong number, which is worse. In each, "
"the first cell shows the mistake and the second is for your fix."
)

ex_fix("I1", "One bracket too few", 1,
   "The cell below fits a stump on one column and raises an error. Run it and "
   "read the last line of the message. Then fit the stump correctly in the "
   "second cell and print its cut.",
   "stump = DecisionTreeRegressor(max_depth=1, random_state=0)\n"
   "stump.fit(train['vol_20d'], train['vol_next'])",
   "stump = ...\n...\n\nprint(...)",
   ["`train['vol_20d']` is a Series: one column of numbers. scikit-learn wants "
    "a table of columns, even when there is only one.",
    "`train[['vol_20d']]`, with a list inside the brackets, is a DataFrame with "
    "one column."],
   "stump = DecisionTreeRegressor(max_depth=1, random_state=0)\n"
   "stump.fit(train[['vol_20d']], train['vol_next'])\n\n"
   "print(round(float(stump.tree_.threshold[0]), 4))",
   f"A `ValueError` asking for a 2-dimensional input, then {B3_CUT}. Single "
   "brackets give a Series and double brackets a DataFrame. The target may be "
   "a Series; the columns the model asks about may not.")

ex_fix("I2", "The columns in another order", 2,
   "The tree below was fitted on `vol_20d` and `ret_20d`, and its last line "
   "forecasts the test days with the same two columns in the other order. Run "
   "it and read the message. Then refit the tree in the second cell, forecast "
   "the test days correctly, and print the test RMSE.",
   "small = DecisionTreeRegressor(max_depth=2, random_state=0)\n"
   "small.fit(train[['vol_20d', 'ret_20d']], train['vol_next'])\n"
   "small.predict(test[['ret_20d', 'vol_20d']])",
   "small = ...\n...\nforecasts = ...\n\nprint(...)",
   "Give `predict` the columns in the order the tree was fitted on: `test[pair]`, "
   "or the same list written out.",
   "small = DecisionTreeRegressor(max_depth=2, random_state=0)\n"
   "small.fit(train[pair], train['vol_next'])\nforecasts = small.predict(test[pair])\n\n"
   "print(round(float(rmse(test['vol_next'], forecasts)), 4))",
   f"A `ValueError`: the feature names must be in the same order as in the fit. "
   f"Then {I2_RMSE}. A tree stores its questions by column position, so swapped "
   "columns would ask the `vol_20d` question of `ret_20d`, and scikit-learn "
   "checks the names to stop that. Keeping the column list in one variable, "
   "like `pair`, avoids the mistake.")

ex_fix("I3", "A perfect score", 2,
   "The cell below reports a test RMSE of zero. Nothing raised, and the number "
   "is wrong. Find the mistake, and fix it in the second cell.",
   "full = DecisionTreeRegressor(random_state=0)\n"
   "full.fit(table[pair], table['vol_next'])\n"
   "print(round(float(rmse(test['vol_next'], full.predict(test[pair]))), 4))",
   "full = ...\n...\n\nprint(...)",
   "Which rows was the tree fitted on, and which rows is it scored on?",
   "full = DecisionTreeRegressor(random_state=0)\n"
   "full.fit(train[pair], train['vol_next'])\n\n"
   "print(round(float(rmse(test['vol_next'], full.predict(test[pair]))), 4))",
   f"0.0 becomes {D1[3]}. The tree was fitted on `table`, which holds the test "
   "days too, and a tree with no limit reproduces every day it was fitted on. "
   "A score that looks too good is a reason to check the split before anything "
   "else.", raises=False)

ex_fix("I4", "A random forest that is not random", 2,
   "The cell below fits what its author thought was a random forest, and it "
   "scores worse than the lecture's. Nothing raised. Read the forest's settings "
   "with `get_params()` to find out why, then fit a real random forest in the "
   "second cell and print the test RMSE of both.",
   "forest = RandomForestRegressor(n_estimators=100, random_state=0)\n"
   "forest.fit(train[columns], train['vol_next'])\n"
   "print(rmse(test['vol_next'], forest.predict(test[columns])))",
   "print(...)\n\nreal = ...\n...\nprint(...)",
   ["`forest.get_params()['max_features']` is how many columns each question "
    "may choose among. A float is a share of the columns.",
    "`RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=0)` "
    "is the forest the lecture used."],
   "forest = RandomForestRegressor(n_estimators=100, random_state=0)\n"
   "forest.fit(train[columns], train['vol_next'])\n"
   "print(forest.get_params()['max_features'])\n\n"
   "real = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=0)\n"
   "real.fit(train[columns], train['vol_next'])\n"
   "print(rmse(test['vol_next'], forest.predict(test[columns])), rmse(test['vol_next'], real.predict(test[columns])))",
   f"`max_features` is {I4_DEFAULT_MF} by default for a forest on a number: every "
   "question sees every column, which is bagging, not a random forest. It "
   f"scores {I4_DEFAULT} against {I4_FIXED}, and its 100 trees open with "
   f"{I4_OPEN[0]} different columns against {I4_OPEN[1]}. A forest for a label "
   f"defaults to `'{I4_CLASSIFIER_MF}'`, which is what makes the mistake easy: "
   "the same name, a different default. Read the settings before trusting a "
   "name.",
   raises=False)

ex_fix("I5", "A forest that changes every run", 2,
   "The cell below fits the same forest twice and prints two different test "
   "RMSEs. Run it again and watch them move. In the second cell, fit both "
   "forests so that they agree, and write an `assert` that checks it.",
   "first = RandomForestRegressor(n_estimators=50, max_features='sqrt')\n"
   "second = RandomForestRegressor(n_estimators=50, max_features='sqrt')\n"
   "first.fit(train[columns], train['vol_next'])\n"
   "second.fit(train[columns], train['vol_next'])\n"
   "print(rmse(test['vol_next'], first.predict(test[columns])))\n"
   "print(rmse(test['vol_next'], second.predict(test[columns])))",
   "first = ...\nsecond = ...\n...\nscore_first = ...\nscore_second = ...\n...\n\n"
   "print(score_first)\nprint(score_second)",
   ["Each forest draws its bootstrap samples and its columns at random. "
    "`random_state` fixes the draws.",
    "`assert score_first == score_second` stops the cell with an "
    "`AssertionError` if the two differ."],
   "first = RandomForestRegressor(n_estimators=50, max_features='sqrt', random_state=0)\n"
   "second = RandomForestRegressor(n_estimators=50, max_features='sqrt', random_state=0)\n"
   "first.fit(train[columns], train['vol_next'])\nsecond.fit(train[columns], train['vol_next'])\n"
   "score_first = rmse(test['vol_next'], first.predict(test[columns]))\n"
   "score_second = rmse(test['vol_next'], second.predict(test[columns]))\n"
   "assert score_first == score_second\n\nprint(score_first)\nprint(score_second)",
   "Without a seed the two numbers differ, often in the second decimal, and "
   "differently on every run. With `random_state=0` both forests draw the same "
   "samples and the same columns, so they are the same forest. Fix the seed "
   "whenever a number goes into a report, so that it can be reproduced.",
   raises=False)

md("---")

# ====================================================== J
section(
"## J · The desk, one column at a time\n\n"
"The eleven instruments, forecast with one or two columns by a tree and by a "
"line. J2 to J5 use the function from J1. Returns are in percent, from `rets` "
"in the setup cell."
)

ex("J1", "A table per instrument", 3,
   "Write `vol_table(ticker)` returning a DataFrame with the columns `vol_20d`, "
   "`ret_20d` and `vol_next` for one instrument, with incomplete rows dropped. "
   "Test it on `'AAPL'`.",
   "def vol_table(ticker):\n    \"\"\"...\"\"\"\n    ...\n\n\nprint(vol_table('AAPL'))",
   ["`r = rets[ticker]`, then `r.rolling(20).std()` and `r.rolling(20).mean()`.",
    "The target looks forward: `r.rolling(20).std().shift(-20)`. End with "
    "`.dropna()`."],
   "def vol_table(ticker):\n    \"\"\"vol_20d, ret_20d and vol_next for one instrument, in percent.\"\"\"\n"
   "    r = rets[ticker]\n"
   "    frame = pd.DataFrame({'vol_20d': r.rolling(20).std(), 'ret_20d': r.rolling(20).mean()})\n"
   "    frame['vol_next'] = r.rolling(20).std().shift(-20)\n    return frame.dropna()\n\n\n"
   "print(vol_table('AAPL'))",
   f"{J1_SHAPE[0]:,} rows and {J1_SHAPE[1]} columns for Apple. One function, "
   "eleven instruments: writing the table once is what keeps the next four "
   "exercises short.")

ex("J2", "A stump against a line", 3,
   "Using `vol_table` from J1, fit a stump and a linear regression on `vol_20d` "
   "for every instrument, split at the end of 2022. Store both test RMSEs in "
   "dictionaries, and print the instruments where the stump wins.",
   "stump_err = {}\nline_err = {}\nfor ticker in tickers:\n    ...\n\nwins = ...\n\nprint(wins)",
   "Inside the loop: `frame = vol_table(ticker)`, split with `.loc`, fit both "
   "models on `early[['vol_20d']]`, and store the rounded test RMSEs.",
   "stump_err = {}\nline_err = {}\nfor ticker in tickers:\n"
   "    frame = vol_table(ticker)\n    early, late = frame.loc[:'2022-12-31'], frame.loc['2023-01-01':]\n"
   "    s = DecisionTreeRegressor(max_depth=1, random_state=0).fit(early[['vol_20d']], early['vol_next'])\n"
   "    l = LinearRegression().fit(early[['vol_20d']], early['vol_next'])\n"
   "    stump_err[ticker] = round(float(rmse(late['vol_next'], s.predict(late[['vol_20d']]))), 4)\n"
   "    line_err[ticker] = round(float(rmse(late['vol_next'], l.predict(late[['vol_20d']]))), 4)\n\n"
   "wins = [t for t in tickers if stump_err[t] < line_err[t]]\n\nprint(wins)",
   f"The stump wins on {len(J2_WINS)} of the 11, {' and '.join(J2_WINS)}. One "
   "question gives two forecasts, and on most instruments two numbers are not "
   "enough to beat a slope.")

ex("J3", "Where each stump cuts", 3,
   "Using `vol_table` from J1, fit a stump on `vol_20d` for every instrument's "
   "training days. Store its cut and the number of training days above the cut "
   "as a tuple in a dictionary, and print one line per instrument, sorted from "
   "the lowest cut to the highest.",
   "stumps = {}\nfor ticker in tickers:\n    ...\n\nfor ticker in sorted(stumps, key=stumps.get):\n    ...",
   ["Inside the loop: `cut = s.tree_.threshold[0]`, then "
    "`stumps[ticker] = (round(float(cut), 3), int((early['vol_20d'] > cut).sum()))`.",
    "Tuples sort by their first element, so `sorted(stumps, key=stumps.get)` "
    "orders the instruments by the cut."],
   "stumps = {}\nfor ticker in tickers:\n"
   "    early = vol_table(ticker).loc[:'2022-12-31']\n"
   "    s = DecisionTreeRegressor(max_depth=1, random_state=0).fit(early[['vol_20d']], early['vol_next'])\n"
   "    cut = s.tree_.threshold[0]\n"
   "    stumps[ticker] = (round(float(cut), 3), int((early['vol_20d'] > cut).sum()))\n\n"
   "for ticker in sorted(stumps, key=stumps.get):\n    print(ticker, stumps[ticker])",
   f"From {J3_SORTED[0]} at {J3_CUT[J3_SORTED[0]]} to {J3_SORTED[-1]} at "
   f"{J3_CUT[J3_SORTED[-1]]}. For {', '.join(J3_CRASH[:-1])} and {J3_CRASH[-1]} the "
   f"busy box holds only {J3_CRASH_N[0]} to {J3_CRASH_N[-1]} days, all in the "
   "spring of 2020: their one question is spent on the crash. "
   f"{J3_SORTED[0]}'s stump does the opposite and sets aside its {J3_LOW_CALM} "
   "calmest days. The same question means different things on different "
   "instruments.")

ex("J4", "Two questions against a line", 5,
   "Using `vol_table` from J1, fit a tree of depth 2 and a linear regression on "
   "`vol_20d` and `ret_20d` for every instrument. Print the instruments where "
   "the tree wins, sorted by the size of the win as a percentage of the line's "
   "RMSE, each with that percentage.",
   "tree_err = {}\nline_err = {}\nfor ticker in tickers:\n    ...\n\n"
   "wins = ...\n...",
   ["The same loop as J2 with `pair` as the columns and `max_depth=2`.",
    "The win for one instrument is `100 * (line_err[t] - tree_err[t]) / line_err[t]`. "
    "Sort the winners with `sorted(..., key=..., reverse=True)`."],
   "tree_err = {}\nline_err = {}\nfor ticker in tickers:\n"
   "    frame = vol_table(ticker)\n    early, late = frame.loc[:'2022-12-31'], frame.loc['2023-01-01':]\n"
   "    t = DecisionTreeRegressor(max_depth=2, random_state=0).fit(early[pair], early['vol_next'])\n"
   "    l = LinearRegression().fit(early[pair], early['vol_next'])\n"
   "    tree_err[ticker] = round(float(rmse(late['vol_next'], t.predict(late[pair]))), 4)\n"
   "    line_err[ticker] = round(float(rmse(late['vol_next'], l.predict(late[pair]))), 4)\n\n"
   "wins = sorted([t for t in tickers if tree_err[t] < line_err[t]],\n"
   "              key=lambda t: (line_err[t] - tree_err[t]) / line_err[t], reverse=True)\n"
   "for ticker in wins:\n    print(ticker, round(100 * (line_err[ticker] - tree_err[ticker]) / line_err[ticker], 1))",
   f"The tree wins on {len(J4_WINS)} of the 11: "
   f"{', '.join(f'{t} by {J4_PCT[t]} percent' for t in J4_WINS[:-1])} and "
   f"{J4_WINS[-1]} by {J4_PCT[J4_WINS[-1]]} percent. The wins are small. Four "
   "leaves rarely beat a slope on a target that moves as smoothly as "
   "volatility, even with the direction of the market to ask about.")

ex("J5", "Draw the desk", 2,
   "Using `stump_err` and `line_err` from J2, or rebuilding them, draw the "
   "ratio of the stump's RMSE to the line's for every instrument as bars, with "
   "a dashed line at 1. A bar below the line is a win for the stump.",
   "stump_err = {}\nline_err = {}\nfor ticker in tickers:\n    ...\n\nratio = ...\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3))\n...\n...\n"
   "ax.set_ylabel('stump RMSE / line RMSE')\nplt.show()",
   ["Rebuild the two dictionaries with the loop from J2, then "
    "`ratio = pd.Series({t: stump_err[t] / line_err[t] for t in tickers})`.",
    "`ax.bar(ratio.index, ratio.values)` and `ax.axhline(1, linestyle='--', color='grey')`."],
   "stump_err = {}\nline_err = {}\nfor ticker in tickers:\n"
   "    frame = vol_table(ticker)\n    early, late = frame.loc[:'2022-12-31'], frame.loc['2023-01-01':]\n"
   "    s = DecisionTreeRegressor(max_depth=1, random_state=0).fit(early[['vol_20d']], early['vol_next'])\n"
   "    l = LinearRegression().fit(early[['vol_20d']], early['vol_next'])\n"
   "    stump_err[ticker] = rmse(late['vol_next'], s.predict(late[['vol_20d']]))\n"
   "    line_err[ticker] = rmse(late['vol_next'], l.predict(late[['vol_20d']]))\n\n"
   "ratio = pd.Series({t: stump_err[t] / line_err[t] for t in tickers})\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3))\nax.bar(ratio.index, ratio.values, color='#1c5cab')\n"
   "ax.axhline(1, color='grey', linestyle='--')\nax.set_ylabel('stump RMSE / line RMSE')\nplt.show()",
   f"Most bars stand above 1, and {J5_WORST} furthest, at {J5_RATIO[J5_WORST]}. A "
   "ratio puts eleven instruments with very different volatility on one scale, "
   "which the raw RMSEs cannot do.")

md("---")

# ====================================================== K
section(
"## K · Five small cases\n\n"
"Each of these stands completely on its own and needs no model from earlier "
"in the notebook. They are here to keep loops, `while`, dictionaries, f-strings "
"and functions in working order, in this session's setting."
)

ex("K1", "A tree as if-statements", 2,
   "The lecture's tree of depth 2 is written out below as rules. Write "
   "`forecast(vol, ret)` that follows them, and print the forecast for each of "
   "the four days in `days`.\n\n"
   "- `vol` 0.99 or less: forecast 0.70 if `vol` is 0.82 or less, otherwise 0.99\n"
   "- `vol` above 0.99: forecast 2.84 if `ret` is -0.45 or less, otherwise 1.25",
   "def forecast(vol, ret):\n    \"\"\"...\"\"\"\n    ...\n\n\n"
   "days = [(0.55, 0.10), (0.90, -0.20), (1.40, -0.70), (2.10, 0.05)]\nfor vol, ret in days:\n    ...",
   ["An outer `if vol <= 0.99:` and an `else:`, each with its own `if`.",
    "In the loop: `print(vol, ret, forecast(vol, ret))`."],
   "def forecast(vol, ret):\n    \"\"\"The depth-2 tree of the lecture, as nested if-statements.\"\"\"\n"
   "    if vol <= 0.99:\n        if vol <= 0.82:\n            return 0.70\n        return 0.99\n"
   "    if ret <= -0.45:\n        return 2.84\n    return 1.25\n\n\n"
   "days = [(0.55, 0.10), (0.90, -0.20), (1.40, -0.70), (2.10, 0.05)]\n"
   "for vol, ret in days:\n    print(vol, ret, forecast(vol, ret))",
   f"The four forecasts are {K1_OUT}. A fitted tree is nothing more than this: "
   "nested `if` statements whose cuts and numbers were chosen from the data.")

_k2_feature = ", ".join(str(v) for v in K2_FEATURE)
_k2_threshold = ", ".join(str(v) for v in K2_THRESHOLD)
ex("K2", "A tree's questions, as text", 2,
   "Inside a fitted tree, `tree_.feature` holds the column number of each "
   "box's question, with -2 for a leaf, and `tree_.threshold` holds its cut. "
   "Below are the two arrays of a tree of depth 2 fitted on three columns of "
   "the index table, with the cuts rounded to three decimals. Loop over the "
   "boxes with `zip` and, for every box that asks a question, add the text of "
   "the question, such as `'vol_20d <= 0.841'`, to `questions` with an "
   "f-string. Then count the different columns the tree asks about with a set.",
   f"names = {K2_NAMES}\nfeature = [{_k2_feature}]\nthreshold = [{_k2_threshold}]\n\n"
   "questions = []\nfor f, t in zip(feature, threshold):\n    ...\n\nasked = ...\n\n"
   "print(questions)\nprint(len(questions), 'questions about', ..., 'of', len(names), 'columns')",
   ["A box asks a question when `f >= 0`, and `names[f]` is its column: "
    "`questions.append(f'{names[f]} <= {t}')`.",
    "`set(names[f] for f in feature if f >= 0)` keeps each column once, and "
    "`len(asked)` counts them."],
   f"names = {K2_NAMES}\nfeature = [{_k2_feature}]\nthreshold = [{_k2_threshold}]\n\n"
   "questions = []\nfor f, t in zip(feature, threshold):\n    if f >= 0:\n"
   "        questions.append(f'{names[f]} <= {t}')\n\n"
   "asked = set(names[f] for f in feature if f >= 0)\n\n"
   "print(questions)\nprint(len(questions), 'questions about', len(asked), 'of', len(names), 'columns')",
   f"`{K2_QUESTIONS}`: three questions about two of the three columns. The "
   f"first asks about `vol_5d`, and its cut at 3.251 percent sets apart "
   f"{K2_BUSY} of the {N_TRAIN:,} training days, {K2_BUSY_2020} of them in March "
   "and April 2020; both boxes below it then ask about `vol_20d`. Offered "
   "`ret_20d`, this tree never used it. The boxes are numbered depth first, "
   "which is why the second question sits at position 1 and the third at "
   "position 4.")

ex("K3", "Until every day has been left out once", 4,
   "A day gets an out-of-bag forecast only once some tree's sample has left it "
   "out. Draw bootstrap samples of the training days one at a time, with "
   "`random_state` 1, 2, 3 and so on, and after each one mark the days it left "
   "out. Stop with a `break` as soon as every day has been left out at least "
   "once, and print how many samples that took. The `while` is bounded at 100.",
   "left_out_once = np.zeros(len(train), dtype=bool)\nsamples = 0\nwhile samples < 100:\n"
   "    samples += 1\n    ...\n\nprint(samples)",
   ["`~train.index.isin(sample.index)` is True for the days one sample left "
    "out, and `left_out_once | ...` adds them to the days left out before.",
    "`left_out_once.all()` is True once every day has been left out; `break` "
    "there."],
   "left_out_once = np.zeros(len(train), dtype=bool)\nsamples = 0\nwhile samples < 100:\n"
   "    samples += 1\n"
   "    sample = train.sample(n=len(train), replace=True, random_state=samples)\n"
   "    left_out_once = left_out_once | ~train.index.isin(sample.index)\n"
   "    if left_out_once.all():\n        break\n\nprint(samples)",
   f"{K3_SAMPLES} samples. After 10 there were still {K3_AFTER_10} days that "
   "every sample had drawn. A day is left out of one sample with a chance of "
   "about 0.368, so it is in all of k samples with a chance of 0.632 to the "
   f"power k, and the expected number of days never left out, {N_TRAIN:,} times "
   f"that, falls below one at k = {K3_FORMULA}. A forest of 100 trees gives every "
   "day plenty of trees that never saw it.")

_k4_rows = ",\n          ".join("[" + ", ".join(f"{v:.3f}" for v in row) + "]" for row in K4_SCORES)
ex("K4", "The best cell of a grid, by hand", 3,
   "Below are the mean fold RMSEs of a grid of forests: one row per "
   "`max_features`, one column per `min_samples_leaf`. Find the best cell with "
   "two nested loops over the positions, keeping the best score so far and its "
   "pair of settings. Then find the same cell with NumPy: `argmin` on the array "
   "gives one position, counted along the first row, then the second, and `//` "
   "and `%` by the number of columns turn it into a row and a column.",
   "max_features = [1, 4, 19]\nleaf_sizes = [5, 20, 50, 100, 200]\n"
   f"scores = [{_k4_rows}]\n\n"
   "best_score = scores[0][0]\nbest_pair = (max_features[0], leaf_sizes[0])\n"
   "for i in range(len(max_features)):\n    for j in range(len(leaf_sizes)):\n        ...\n\n"
   "print(best_pair, best_score)\n\n"
   "grid = np.array(scores)\nposition = ...\nrow = ...\ncol = ...\nsame = ...\n\n"
   "print(position, row, col)\nprint(same)",
   ["Inside the inner loop: `if scores[i][j] < best_score:`, then update both "
    "`best_score` and `best_pair = (max_features[i], leaf_sizes[j])`.",
    "`position = int(grid.argmin())`, `row = position // grid.shape[1]` and "
    "`col = position % grid.shape[1]`. `same` compares "
    "`(max_features[row], leaf_sizes[col])` with `best_pair`."],
   "max_features = [1, 4, 19]\nleaf_sizes = [5, 20, 50, 100, 200]\n"
   f"scores = [{_k4_rows}]\n\n"
   "best_score = scores[0][0]\nbest_pair = (max_features[0], leaf_sizes[0])\n"
   "for i in range(len(max_features)):\n    for j in range(len(leaf_sizes)):\n"
   "        if scores[i][j] < best_score:\n            best_score = scores[i][j]\n"
   "            best_pair = (max_features[i], leaf_sizes[j])\n\n"
   "print(best_pair, best_score)\n\n"
   "grid = np.array(scores)\nposition = int(grid.argmin())\nrow = position // grid.shape[1]\n"
   "col = position % grid.shape[1]\nsame = (max_features[row], leaf_sizes[col]) == best_pair\n\n"
   "print(position, row, col)\nprint(same)",
   f"`{K4_PAIR}` at {min(min(r) for r in K4_SCORES):.3f}, and `True`: position "
   f"{K4_POS} is row {K4_ROW}, column {K4_COL}. These are the fold scores of "
   "G3's grid, rounded, and the two loops do what `GridSearchCV` does once "
   "every cell is scored: keep the best and store its settings in "
   "`best_params_`. `//` counts the whole rows passed and `%` what is left "
   "over, the same arithmetic as turning minutes into hours and minutes.")

_k5_parts = [f"{K5_TREE[k][0]} from {K5_TREE[k][1]:,}" for k in K5_MEANS]      # three leaves, asserted above
_k5_tree = f"{_k5_parts[0]} days, {_k5_parts[1]} and {_k5_parts[2]}"
ex("K5", "Leaf forecasts, by hand", 3,
   "Below are ten training days of the index table, one every 180 trading days: "
   "the leaf of the lecture's tree of depth 2 each one landed in, and what "
   "`vol_next` turned out to be. Collect the `vol_next` values of each leaf in a "
   "dictionary of lists with a loop, then build `means`, the mean of each leaf "
   "to three decimals. Check the means with pandas `groupby` in one line.",
   f"landed = {K5_LANDED}\nvol_next = [{', '.join(f'{v:.2f}' for v in K5_NEXT)}]\n\n"
   "by_leaf = {}\nfor leaf, value in zip(landed, vol_next):\n    ...\n\n"
   "means = {}\nfor leaf in by_leaf:\n    ...\n\ncheck = ...\n\nprint(means)\nprint(check)",
   ["Inside the first loop: `if leaf not in by_leaf:` start an empty list, "
    "then `by_leaf[leaf].append(value)`.",
    "`means[leaf] = round(sum(by_leaf[leaf]) / len(by_leaf[leaf]), 3)`, and "
    "`pd.Series(vol_next).groupby(landed).mean().round(3)` is the one line."],
   f"landed = {K5_LANDED}\nvol_next = [{', '.join(f'{v:.2f}' for v in K5_NEXT)}]\n\n"
   "by_leaf = {}\nfor leaf, value in zip(landed, vol_next):\n"
   "    if leaf not in by_leaf:\n        by_leaf[leaf] = []\n    by_leaf[leaf].append(value)\n\n"
   "means = {}\nfor leaf in by_leaf:\n"
   "    means[leaf] = round(sum(by_leaf[leaf]) / len(by_leaf[leaf]), 3)\n\n"
   "check = pd.Series(vol_next).groupby(landed).mean().round(3)\n\nprint(means)\nprint(check)",
   f"`{K5_MEANS}`, and `groupby` gives the same three numbers. The tree's own "
   f"forecasts for these leaves are the same calculation on all their training "
   f"days: {_k5_tree}. Ten days give only a rough version of them, which is "
   "why a leaf needs many days before its mean can be trusted.")

# ---------------------------------------------------------------- closing
md(
"## \U0001f3c1 Done\n\n"
"You found the best number for a box and the best cut with loops and NumPy, "
"checked both against scikit-learn, drew trees and traced days through them, "
"watched a deep tree fit the training days and miss the test days, chose the "
"size of a tree on the time-series folds, drew bootstrap samples and bagged "
"trees by hand, counted what `max_features` changes, found out-of-bag scores "
"too kind on days and fair on borrowers, counted, timed and sampled grids, and "
"fixed five mistakes that trees make easy.\n\n"
"The case gives the risk report's volatility forecast a tree and a forest, "
"held to its one-column model and to ridge, on Apple and across the desk, "
"and ends with one question on Part 9's jump label."
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
