# -*- coding: utf-8 -*-
"""Build session_12_case.ipynb  (The Analyst's Notebook, Part 12, the last part).

Conventions (approved for Sessions 1 to 11): no star badges, one cumulative
investigation where later questions reuse what earlier ones stored, folded
hints and solutions, stated formulas, plain explanatory tone, no em-dashes.
Part 11 is the last part before this one (Sessions 7 and 10 have no case part).
The QUICK LOAD restores what the risk report holds after Part 11, as the numbers
those parts printed: the volatility forecast (Part 5's target; Part 6's one
column, ridge, lasso and elastic net, their settings and the desk; Part 11's
stump, its tuned forest, its settings and its desk) and the jump warning (Part
9's label, its one-column, twenty-column and k-NN models on the test block and
the folds, and its threshold; Part 11's forest on that label). Nothing is
refitted to get them back; Q1 refits the one-column forecast once and checks it
with True.

Part 12 follows its session's topic first: boosting for the report's forecast,
held to the restored numbers. On Apple, 46 boosted stumps on vol_20d, stopped on
2022, score 0.00375 on the test block, the lowest number the report has
printed; the linear regression fitted on the same seven years scores 0.00390,
refitted on all eight years the stumps score 0.00398 against 0.00412, and on the
folds they lose, 0.00780 against 0.00755. On twenty columns XGBoost (stumps at
0.3, 13 trees, chosen on 2022) is behind ridge and the forest on both yardsticks.
Read through importance, it leans on Disney's volatility, which permutation shows
to hurt on the test days; the desk's ten columns, shuffled together, make both
XGBoost and ridge better there. One light question boosts Part 9's jump label
(0.838 and 0.705, below the one column's 0.889 and 0.792).

Because this is the last part, the second half sets the report side by side:
every forecast it has fitted since Part 4, sixteen of them, on the same rows,
the test block and the folds; a figure; how much of the average's error each
removes; every family across the desk; the days the forecast missed most and the
warning on them; the forecast for the month after the data ends; and the final
report. The closing sections go through the case part by part.

Returns stay in plain decimals here, as in Parts 1 to 11.

BLANK-SAFE, and this one needs care because the case is cumulative: no
pre-written line may CALL anything on a variable an earlier question produced.
Every such dependency sits inside the student's own blank.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import nbformat as nbf
from _shared import XGB_GUARD
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "session_12" / "session_12_case.ipynb"

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


def scaled(model):
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def fold_rmse(model, X, y):
    return float(-cross_val_score(model, X, y, cv=TS5, scoring="neg_root_mean_squared_error").mean())


# ---- the real numbers, so every note is exact ------------------------------
PX = pd.read_csv(ROOT / "data" / "prices.csv", parse_dates=["date"])
W = PX.pivot(index="date", columns="ticker", values="close")
R = W.pct_change()
TICKERS = sorted(W.columns)
TS5 = TimeSeriesSplit(n_splits=5)
GRID = [1, 10, 100, 1000, 10000]
SWEEP = np.arange(0.05, 0.60, 0.01)


def wide_table(ticker, keep_unknown=False):
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
    if keep_unknown:
        return frame.dropna(subset=list(frame.columns[:-1]))
    return frame.dropna()


def ridge_search():
    return GridSearchCV(Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]), {"ridge__alpha": GRID},
                        cv=TS5, scoring="neg_root_mean_squared_error")


def cost_at(y, p, threshold):
    tn, fp, fn, tp = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1]).ravel()
    return int(5 * fn + fp)


TBL = wide_table("AAPL")
LATEST = wide_table("AAPL", keep_unknown=True)
COLS = list(TBL.columns[:-1])
TRAIN, TEST = TBL.loc[:"2022-12-31"], TBL.loc["2023-01-01":]
FIT, STOP = TRAIN.loc[:"2021-12-31"], TRAIN.loc["2022-01-01":]
Y, YT = TRAIN["vol_next"], TEST["vol_next"]
J = (TRAIN["vol_next"] > 1.5 * TRAIN["vol_20d"]).astype(int)
JT = (TEST["vol_next"] > 1.5 * TEST["vol_20d"]).astype(int)
N_TBL, N_TRAIN, N_TEST = len(TBL), len(TRAIN), len(TEST)
UNKNOWN = LATEST[LATEST["vol_next"].isna()]
assert len(LATEST) - N_TBL == len(UNKNOWN) == 20

# ---- the earlier parts, restated (Parts 1 to 5, for the closing overview) ---
_a24, _k24 = W.loc["2024", "AAPL"], W.loc["2024", "KO"]
P1 = {"AAPL": ((_a24.max() - _a24.min()) / _a24.iloc[0], _a24.iloc[-1] / _a24.iloc[0] - 1),
      "KO": ((_k24.max() - _k24.min()) / _k24.iloc[0], _k24.iloc[-1] / _k24.iloc[0] - 1)}
assert round(100 * P1["AAPL"][0], 2) == 51.22 and round(100 * P1["KO"][0], 2) == 26.15
P3 = (R.loc["2024"].std() * np.sqrt(252)).sort_values(ascending=False)
assert P3.index[0] == "NVDA" and P3.index[-1] == "SPY" and round(P3["NVDA"], 3) == 0.525


def _part4_table(ticker):
    r = R[ticker]
    frame = pd.DataFrame({"vol_20d": r.rolling(20).std(), "ret_20d": r.rolling(20).mean(),
                          "up_20d": (r > 0).rolling(20).mean()})
    frame["vol_next"] = r.rolling(20).std().shift(-20)
    return frame.dropna()


_t4 = _part4_table("AAPL")
_tr4, _te4 = _t4.loc[:"2022-12-31"], _t4.loc["2023-01-01":]
P4_BASE = rmse(_te4["vol_next"], np.full(len(_te4), _tr4["vol_next"].mean()))
P4_PERS = rmse(_te4["vol_next"], _te4["vol_20d"])
P5_RMSE = rmse(_te4["vol_next"], LinearRegression().fit(_tr4[["vol_20d"]], _tr4["vol_next"]).predict(_te4[["vol_20d"]]))
assert (round(P4_BASE, 5), round(P4_PERS, 5), round(P5_RMSE, 5)) == (0.00514, 0.00417, 0.00402)

# ---- what the report holds after Part 11: the forecast ----------------------
ONE = LinearRegression().fit(TRAIN[["vol_20d"]], Y)
ONE_TE = rmse(YT, ONE.predict(TEST[["vol_20d"]]))
ONE_CV = fold_rmse(LinearRegression(), TRAIN[["vol_20d"]], Y)
SEARCH = ridge_search().fit(TRAIN[COLS], Y)
RIDGE_TE = rmse(YT, SEARCH.predict(TEST[COLS]))
RIDGE_ALPHA, RIDGE_CV = SEARCH.best_params_["ridge__alpha"], float(-SEARCH.best_score_)
assert (round(ONE_TE, 5), round(ONE_CV, 5), round(RIDGE_TE, 5), round(RIDGE_CV, 5)) == (0.00412, 0.00755, 0.0046, 0.00761)
LSEARCH = GridSearchCV(Pipeline([("scale", StandardScaler()), ("lasso", Lasso(max_iter=20000))]),
                       {"lasso__alpha": [0.00001, 0.0001, 0.001, 0.01]}, cv=TS5,
                       scoring="neg_root_mean_squared_error").fit(TRAIN[COLS], Y)
LASSO_ALPHA = LSEARCH.best_params_["lasso__alpha"]
ESEARCH = GridSearchCV(Pipeline([("scale", StandardScaler()), ("enet", ElasticNet(max_iter=20000))]),
                       {"enet__alpha": [0.0001, 0.001, 0.01], "enet__l1_ratio": [0.1, 0.5, 0.9]}, cv=TS5,
                       scoring="neg_root_mean_squared_error").fit(TRAIN[COLS], Y)
ENET = {"alpha": ESEARCH.best_params_["enet__alpha"], "l1_ratio": ESEARCH.best_params_["enet__l1_ratio"]}
assert LASSO_ALPHA == 0.001 and ENET == {"alpha": 0.001, "l1_ratio": 0.9}
DESK = {}
for _t in TICKERS:
    _tb = wide_table(_t)
    _tr, _te = _tb.loc[:"2022-12-31"], _tb.loc["2023-01-01":]
    _cc = list(_tb.columns[:-1])
    _g = ridge_search().fit(_tr[_cc], _tr["vol_next"])
    _o = LinearRegression().fit(_tr[["vol_20d"]], _tr["vol_next"])
    DESK[_t] = (round(rmse(_te["vol_next"], _g.predict(_te[_cc])), 5),
                round(rmse(_te["vol_next"], _o.predict(_te[["vol_20d"]])), 5))
assert sum(1 for t in TICKERS if DESK[t][0] < DESK[t][1]) == 7
STUMP = DecisionTreeRegressor(max_depth=1, random_state=0).fit(TRAIN[["vol_20d"]], Y)
STUMP_TE = rmse(YT, STUMP.predict(TEST[["vol_20d"]]))
STUMP_CUT = float(STUMP.tree_.threshold[0])
FOREST_SET = {"max_features": 5, "min_samples_leaf": 100}            # what Part 11's search chose


def tuned_forest():
    return RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1, **FOREST_SET)


FOREST_TE = rmse(YT, tuned_forest().fit(TRAIN[COLS], Y).predict(TEST[COLS]))
FOREST_CV = fold_rmse(tuned_forest(), TRAIN[COLS], Y)
assert (round(STUMP_TE, 5), round(FOREST_TE, 5), round(FOREST_CV, 5)) == (0.00416, 0.00462, 0.00757)
DESK_FOREST = {}
for _t in TICKERS:
    _tb = wide_table(_t)
    _tr, _te = _tb.loc[:"2022-12-31"], _tb.loc["2023-01-01":]
    _cc = list(_tb.columns[:-1])
    DESK_FOREST[_t] = round(rmse(_te["vol_next"], tuned_forest().fit(_tr[_cc], _tr["vol_next"]).predict(_te[_cc])), 5)
assert sum(1 for t in TICKERS if DESK_FOREST[t] < DESK[t][0]) == 5

# ---- and the warning --------------------------------------------------------
JUMP = logit_pipe().fit(TRAIN[["vol_20d"]], J)
P_JUMP_TR = JUMP.predict_proba(TRAIN[["vol_20d"]])[:, 1]
P_JUMP_TE = JUMP.predict_proba(TEST[["vol_20d"]])[:, 1]
P9_AUC = float(roc_auc_score(JT, P_JUMP_TE))
P9_CHOSEN = round(float(SWEEP[int(np.argmin([cost_at(J, P_JUMP_TR, t) for t in SWEEP]))]), 2)
P9_FOLDS = float(cross_val_score(logit_pipe(), TRAIN[["vol_20d"]], J, cv=TS5, scoring="roc_auc").mean())
_wide9 = GridSearchCV(logit_pipe(max_iter=1000), {"logit__C": [0.0001, 0.001, 0.01, 0.1, 1, 10]}, cv=TS5,
                      scoring="roc_auc").fit(TRAIN[COLS], J)
P9_WIDE_AUC = float(roc_auc_score(JT, _wide9.predict_proba(TEST[COLS])[:, 1]))
P9_WIDE_FOLDS = float(_wide9.best_score_)
_knn = GridSearchCV(Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier())]),
                    {"knn__n_neighbors": [1, 5, 15, 51, 101, 151, 201, 301]}, cv=TS5,
                    scoring="roc_auc").fit(TRAIN[["vol_20d"]], J)
P9_KNN_AUC = float(roc_auc_score(JT, _knn.predict_proba(TEST[["vol_20d"]])[:, 1]))
P9_KNN_FOLDS = float(_knn.best_score_)
P9_COST, P9_COST_HALF = cost_at(JT, P_JUMP_TE, P9_CHOSEN), cost_at(JT, P_JUMP_TE, 0.5)
assert P9_CHOSEN == 0.25 and round(P9_AUC, 4) == 0.8892 and round(P9_FOLDS, 4) == 0.7923
assert round(P9_WIDE_AUC, 4) == 0.7458 and round(P9_WIDE_FOLDS, 4) == 0.7547
assert (round(P9_KNN_AUC, 4), round(P9_KNN_FOLDS, 4)) == (0.8378, 0.7586) and (P9_COST, P9_COST_HALF) == (178, 210)
_jf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1, **FOREST_SET).fit(TRAIN[COLS], J)
P11_JF_AUC = float(roc_auc_score(JT, _jf.predict_proba(TEST[COLS])[:, 1]))
P11_JF_FOLDS = float(cross_val_score(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1, **FOREST_SET),
                                     TRAIN[COLS], J, cv=TS5, scoring="roc_auc").mean())
assert (round(P11_JF_AUC, 3), round(P11_JF_FOLDS, 3)) == (0.869, 0.682)
_rise_tr = (TRAIN["vol_next"] > TRAIN["vol_20d"]).astype(int)
_rise_te = (TEST["vol_next"] > TEST["vol_20d"]).astype(int)
P8_AUC = float(roc_auc_score(_rise_te, logit_pipe().fit(TRAIN[["vol_20d"]], _rise_tr).predict_proba(TEST[["vol_20d"]])[:, 1]))
assert round(P8_AUC, 4) == 0.7971

# ---- Part 12 ----------------------------------------------------------------
ONE_COL = ["vol_20d"]
# Q2 the first stump is Part 11's
B100 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0).fit(TRAIN[ONE_COL], Y)
_stages = list(B100.staged_predict(TRAIN[ONE_COL]))
assert np.allclose(_stages[0], Y.mean() + 0.1 * (STUMP.predict(TRAIN[ONE_COL]) - Y.mean()))
Q2_TRAIN = [rmse(Y, _stages[m - 1]) for m in [1, 10, 100]]
assert Q2_TRAIN[0] > Q2_TRAIN[1] > Q2_TRAIN[2]

# Q3 stop on 2022
BSTOP = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=1, random_state=0).fit(FIT[ONE_COL], FIT["vol_next"])
STOP_SCORES = np.array([rmse(STOP["vol_next"], _p) for _p in BSTOP.staged_predict(STOP[ONE_COL])])
BEST_M = int(np.argmin(STOP_SCORES)) + 1
STOPPED_TE = rmse(YT, list(BSTOP.staged_predict(TEST[ONE_COL]))[BEST_M - 1])
_flat = np.where(STOP_SCORES <= STOP_SCORES.min() + 0.00001)[0] + 1
assert BEST_M == 46 and round(STOPPED_TE, 5) == 0.00375 and (_flat.min(), _flat.max()) == (45, 89)

# Q4 a fair comparison
LINE_FIT_TE = rmse(YT, LinearRegression().fit(FIT[ONE_COL], FIT["vol_next"]).predict(TEST[ONE_COL]))
BOOST_ALL = GradientBoostingRegressor(n_estimators=BEST_M, learning_rate=0.1, max_depth=1, random_state=0)
BOOST_TE = rmse(YT, BOOST_ALL.fit(TRAIN[ONE_COL], Y).predict(TEST[ONE_COL]))
BOOST_CV = fold_rmse(GradientBoostingRegressor(n_estimators=BEST_M, learning_rate=0.1, max_depth=1, random_state=0),
                     TRAIN[ONE_COL], Y)
assert STOPPED_TE < LINE_FIT_TE < ONE_TE and BOOST_TE < ONE_TE and BOOST_CV > ONE_CV
assert (round(LINE_FIT_TE, 5), round(BOOST_TE, 5), round(BOOST_CV, 5)) == (0.0039, 0.00398, 0.0078)

# Q5 twenty columns in XGBoost, chosen on 2022, refitted on every training year
GRID_SCORE, GRID_TREES = {}, {}
for _d in [1, 2, 3]:
    for _r in [0.3, 0.1, 0.03]:
        _x = XGBRegressor(n_estimators=3000, learning_rate=_r, max_depth=_d, early_stopping_rounds=50, random_state=0)
        _x.fit(FIT[COLS], FIT["vol_next"], eval_set=[(STOP[COLS], STOP["vol_next"])], verbose=False)
        GRID_SCORE[(_d, _r)] = float(_x.best_score)
        GRID_TREES[(_d, _r)] = int(_x.best_iteration) + 1
BEST_KEY = min(GRID_SCORE, key=GRID_SCORE.get)
WIDE_SET = {"n_estimators": GRID_TREES[BEST_KEY], "learning_rate": BEST_KEY[1], "max_depth": BEST_KEY[0]}
assert BEST_KEY == (1, 0.3) and WIDE_SET["n_estimators"] == 13
XGB_WIDE = XGBRegressor(random_state=0, **WIDE_SET).fit(TRAIN[COLS], Y)
XGB_TE = rmse(YT, XGB_WIDE.predict(TEST[COLS]))
XGB_CV = fold_rmse(XGBRegressor(random_state=0, **WIDE_SET), TRAIN[COLS], Y)
assert XGB_TE > max(RIDGE_TE, FOREST_TE) and XGB_CV > max(RIDGE_CV, FOREST_CV)

# Q6 what it leans on
GAIN = pd.Series(XGB_WIDE.feature_importances_, index=COLS).sort_values(ascending=False)
NEVER = int((GAIN == 0).sum())
_pi = permutation_importance(XGB_WIDE, TEST[COLS], YT, scoring="neg_root_mean_squared_error", n_repeats=10, random_state=0)
PERM = pd.Series(_pi.importances_mean, index=COLS).sort_values(ascending=False)
assert GAIN.index[0] == "DIS_vol" and GAIN.index[1] == "vol_20d" and NEVER == 10
assert PERM.index[0] == "vol_20d" and PERM.index[-1] == "DIS_vol" and PERM["DIS_vol"] < 0

# Q7 three kinds of column
GROUPS = {"own volatility": [c for c in COLS if c.startswith("vol_")],
          "own returns": ["ret_5d", "ret_20d", "ret_60d", "up_20d"],
          "the desk": [c for c in COLS if c.endswith("_vol")]}
assert [len(v) for v in GROUPS.values()] == [6, 4, 10]
RIDGE = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))]).fit(TRAIN[COLS], Y)
assert round(rmse(YT, RIDGE.predict(TEST[COLS])), 5) == round(RIDGE_TE, 5)
CHANGE = {}
for _name, _group in GROUPS.items():
    _dx, _dr = [], []
    for _k in range(10):
        _order = np.random.default_rng(_k).permutation(N_TEST)
        _sh = TEST.copy()
        _sh[_group] = TEST[_group].values[_order]
        _dx.append(rmse(YT, XGB_WIDE.predict(_sh[COLS])) - XGB_TE)
        _dr.append(rmse(YT, RIDGE.predict(_sh[COLS])) - RIDGE_TE)
    CHANGE[_name] = (float(np.mean(_dx)), float(np.mean(_dr)))
assert CHANGE["own volatility"][0] > 0 and CHANGE["own volatility"][1] > 0
assert CHANGE["the desk"][0] < 0 and CHANGE["the desk"][1] < 0
assert max(CHANGE, key=lambda k: CHANGE[k][1]) == "own volatility"

# Q8 the warning, once
JF_, JS_ = J.loc[:"2021-12-31"], J.loc["2022-01-01":]
_xj = XGBClassifier(n_estimators=3000, learning_rate=0.1, max_depth=1, early_stopping_rounds=50, random_state=0)
_xj.fit(FIT[COLS], JF_, eval_set=[(STOP[COLS], JS_)], verbose=False)
JUMP_TREES = int(_xj.best_iteration) + 1
_xja = XGBClassifier(n_estimators=JUMP_TREES, learning_rate=0.1, max_depth=1, random_state=0).fit(TRAIN[COLS], J)
XJ_AUC = float(roc_auc_score(JT, _xja.predict_proba(TEST[COLS])[:, 1]))
XJ_CV = float(cross_val_score(XGBClassifier(n_estimators=JUMP_TREES, learning_rate=0.1, max_depth=1, random_state=0),
                              TRAIN[COLS], J, cv=TS5, scoring="roc_auc").mean())
assert JUMP_TREES == 116 and XJ_AUC < P9_AUC and XJ_CV < P9_FOLDS
WARN_ROWS = {"logistic regression, vol_20d (Part 9)": (P9_AUC, P9_FOLDS),
             "logistic regression, twenty columns (Part 9)": (P9_WIDE_AUC, P9_WIDE_FOLDS),
             "k-nearest neighbours, vol_20d (Part 9)": (P9_KNN_AUC, P9_KNN_FOLDS),
             "forest, twenty columns (Part 11)": (P11_JF_AUC, P11_JF_FOLDS),
             "XGBoost, twenty columns (Part 12)": (XJ_AUC, XJ_CV)}
assert max(WARN_ROWS, key=lambda k: WARN_ROWS[k][0]) == max(WARN_ROWS, key=lambda k: WARN_ROWS[k][1]) \
    == "logistic regression, vol_20d (Part 9)"

# Q9 every forecast, on the same rows
ONE_, THREE_, PAIR_ = ["vol_20d"], ["vol_20d", "ret_20d", "up_20d"], ["vol_20d", "ret_20d"]
CANDIDATES = {
    "linear regression, vol_20d": (5, ONE_, LinearRegression()),
    "linear regression, three columns": (5, THREE_, LinearRegression()),
    "OLS, twenty columns": (6, COLS, LinearRegression()),
    "ridge": (6, COLS, scaled(Ridge(alpha=RIDGE_ALPHA))),
    "lasso": (6, COLS, scaled(Lasso(alpha=LASSO_ALPHA, max_iter=20000))),
    "elastic net": (6, COLS, scaled(ElasticNet(alpha=ENET["alpha"], l1_ratio=ENET["l1_ratio"], max_iter=20000))),
    "stump, vol_20d": (11, ONE_, DecisionTreeRegressor(max_depth=1, random_state=0)),
    "stump, twenty columns": (11, COLS, DecisionTreeRegressor(max_depth=1, random_state=0)),
    "tree of depth 2, two columns": (11, PAIR_, DecisionTreeRegressor(max_depth=2, random_state=0)),
    "bagging": (11, COLS, RandomForestRegressor(n_estimators=100, max_features=1.0, random_state=0, n_jobs=-1)),
    "forest": (11, COLS, RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=0, n_jobs=-1)),
    "tuned forest": (11, COLS, tuned_forest()),
    "boosted stumps, vol_20d": (12, ONE_, GradientBoostingRegressor(n_estimators=BEST_M, learning_rate=0.1, max_depth=1,
                                                                    random_state=0)),
    "XGBoost, twenty columns": (12, COLS, XGBRegressor(random_state=0, **WIDE_SET)),
}
_rows = []
_preds = {}
for _name, (_part, _cols, _m) in CANDIDATES.items():
    _m.fit(TRAIN[_cols], Y)
    _preds[_name] = _m.predict(TEST[_cols])
    _rows.append({"model": _name, "part": _part, "test": rmse(YT, _preds[_name]), "folds": fold_rmse(_m, TRAIN[_cols], Y)})
_avg_f, _pers_f = [], []
for _fi, _vi in TS5.split(TRAIN):
    _avg_f.append(rmse(Y.iloc[_vi], np.full(len(_vi), Y.iloc[_fi].mean())))
    _pers_f.append(rmse(Y.iloc[_vi], TRAIN["vol_20d"].iloc[_vi]))
_preds["training average"] = np.full(N_TEST, Y.mean())
_preds["persistence"] = TEST["vol_20d"].values
_rows.append({"model": "training average", "part": 4, "test": rmse(YT, _preds["training average"]),
              "folds": float(np.mean(_avg_f))})
_rows.append({"model": "persistence", "part": 4, "test": rmse(YT, _preds["persistence"]), "folds": float(np.mean(_pers_f))})
BOARD = pd.DataFrame(_rows).set_index("model").sort_values("test")
assert all(_preds[n].mean() > YT.mean() for n in CANDIDATES)      # "every forecast fitted on 2015 to 2022 ran high"
assert len(BOARD) == 16
assert abs(BOARD.loc["ridge", "test"] - 0.00460) < 0.000005 and abs(BOARD.loc["tuned forest", "test"] - 0.00462) < 0.000005
assert abs(BOARD.loc["linear regression, vol_20d", "test"] - 0.00412) < 0.000005
assert abs(BOARD.loc["stump, vol_20d", "test"] - 0.00416) < 0.000005
assert BOARD.index[0] == "boosted stumps, vol_20d" and BOARD["folds"].idxmin() == "linear regression, vol_20d"
_ahead = BOARD.index[:list(BOARD.index).index("linear regression, vol_20d")]
assert all(BOARD.loc[m, "folds"] > BOARD.loc["linear regression, vol_20d", "folds"] for m in _ahead)
BY_FOLDS = BOARD.sort_values("folds")
assert list(BY_FOLDS.index[:3]) == ["linear regression, vol_20d", "tuned forest", "ridge"]
GAP_TOP = BOARD.loc["linear regression, vol_20d", "test"] - BOARD["test"].min()

# Q11 how far the report came
AVG_TE = BOARD.loc["training average", "test"]
SKILL = (1 - (BOARD["test"] / AVG_TE) ** 2)
BEST_BY_PART = BOARD.groupby("part")[["test", "folds"]].min()
assert BEST_BY_PART["folds"].idxmin() == 5
assert round(SKILL["persistence"], 3) == 0.386 and round(SKILL["linear regression, vol_20d"], 3) == 0.402

# Q12 every family across the desk
DESK_BOOST, DESK_TREES = {}, {}
for _t in TICKERS:
    _tb = wide_table(_t)
    _tr, _te = _tb.loc[:"2022-12-31"], _tb.loc["2023-01-01":]
    _fr, _sr = _tr.loc[:"2021-12-31"], _tr.loc["2022-01-01":]
    _cc = list(_tb.columns[:-1])
    _x = XGBRegressor(n_estimators=3000, learning_rate=WIDE_SET["learning_rate"], max_depth=WIDE_SET["max_depth"],
                      early_stopping_rounds=50, random_state=0)
    _x.fit(_fr[_cc], _fr["vol_next"], eval_set=[(_sr[_cc], _sr["vol_next"])], verbose=False)
    DESK_TREES[_t] = int(_x.best_iteration) + 1
    _xa = XGBRegressor(n_estimators=DESK_TREES[_t], learning_rate=WIDE_SET["learning_rate"],
                       max_depth=WIDE_SET["max_depth"], random_state=0).fit(_tr[_cc], _tr["vol_next"])
    DESK_BOOST[_t] = rmse(_te["vol_next"], _xa.predict(_te[_cc]))
DESK_ALL = pd.DataFrame({"one column": {t: DESK[t][1] for t in TICKERS}, "ridge": {t: DESK[t][0] for t in TICKERS},
                         "forest": DESK_FOREST, "boosting": DESK_BOOST})
WINNERS = DESK_ALL.idxmin(axis=1)
WIN_COUNTS = WINNERS.value_counts()
_sorted = np.sort(DESK_ALL.values, axis=1)
assert (_sorted[:, 1] - _sorted[:, 0]).min() > 0.000005             # no winner decided by the rounding
assert WIN_COUNTS.to_dict() == {"boosting": 4, "one column": 3, "ridge": 2, "forest": 2}
assert WINNERS["AAPL"] == "one column"
BOOST_WINS = sorted(WINNERS[WINNERS == "boosting"].index)
ONE_WINS = sorted(WINNERS[WINNERS == "one column"].index)
WORST_T = max(TICKERS, key=lambda t: DESK_BOOST[t] / DESK[t][1])

# Q13 the days the forecast missed most
ERR = YT - ONE.predict(TEST[ONE_COL])
DAYS = pd.DataFrame({"vol_20d": TEST["vol_20d"], "vol_next": YT, "error": ERR, "probability": P_JUMP_TE})
DAYS["warning"] = DAYS["probability"] >= P9_CHOSEN
SHORT = DAYS.nlargest(5, "error")
OVER = DAYS.nsmallest(5, "error")
assert all(SHORT.index.strftime("%Y-%m") == "2024-06") and SHORT["warning"].all()
assert (TEST.loc[SHORT.index, "vol_next"] > 1.5 * TEST.loc[SHORT.index, "vol_20d"]).all()
assert all(OVER.index.strftime("%Y-%m") == "2024-05") and not OVER["warning"].any()
N_WARN_TEST = int(DAYS["warning"].sum())

# Q14 the forecast for the month after the data ends
LAST = LATEST.iloc[[-1]]
assert str(LAST.index[0].date()) == "2024-12-31"
FC_NEXT = float(ONE.predict(LAST[ONE_COL])[0])
ANN_NEXT, ANN_NOW = 100 * FC_NEXT * np.sqrt(252), 100 * float(LAST["vol_20d"].iloc[0]) * np.sqrt(252)
P_NOW = float(JUMP.predict_proba(LAST[ONE_COL])[0, 1])
WARN_UNKNOWN = int((JUMP.predict_proba(UNKNOWN[ONE_COL])[:, 1] >= P9_CHOSEN).sum())
assert P_NOW >= P9_CHOSEN and WARN_UNKNOWN == 20 and ANN_NEXT > ANN_NOW


def _span(a, b):
    """3 to 7 June 2024, or 28 May to 3 June 2024."""
    if (a.year, a.month) == (b.year, b.month):
        return f"{a.day} to {b.day} {b.strftime('%B %Y')}"
    return f"{a.day} {a.strftime('%B')} to {b.day} {b.strftime('%B %Y')}"


# ---------------------------------------------------------------- top matter
md(
"# \U0001f4bc The Analyst's Notebook · Part 12\n"
"### Boosting, and the report as a whole\n\n"
"After Part 11 the risk report holds two results, each with a flexible challenger "
"behind it. The volatility forecast is still the one-column linear regression "
f"chosen in Part 5: {ONE_TE:.5f} on the test block and {ONE_CV:.5f} on the folds, "
f"where ridge on twenty columns scored {RIDGE_TE:.5f} and {RIDGE_CV:.5f} and a forest "
f"tuned on the folds {FOREST_TE:.5f} and {FOREST_CV:.5f}. The jump warning is still "
f"Part 9's one-column logistic regression, with an AUC of {P9_AUC:.3f}, run at a "
f"threshold of {P9_CHOSEN}.\n\n"
"This part does two things. It first gives boosting the question the report was "
"built on, with the number of trees chosen on 2022, and reads the boosted model "
"through its importance. Then, because this is the last part of the case, it "
"sets every model the report has fitted since Part 4 side by side: on the same "
"rows and both yardsticks, across the desk, and on the days that went wrong. It "
"ends with the report as it would go out on the last day of the data."
)

md(
"## How to work through this\n\n"
"- Run the **quick load** cell first. It brings back what the risk report holds "
"after Part 11 and loads the price table.\n"
"- Each question builds on the last, so keep them in order and keep your "
"variables. Later questions use the names earlier ones created.\n"
"- Cells with `...` are blanks. The notebook runs cleanly even before you fill "
"them in, so **Run all** is always safe.\n"
"- Hints and solutions are folded under each question. Work first, then check.\n"
"- Q9 refits sixteen models on the training block and the folds, and takes about "
"a minute on Colab.\n\n"
"**A note on units.** The lecture worked in percent, on the index table. This "
"notebook keeps the plain decimals of Parts 1 to 11 and stays on Apple, so every "
"number compares directly with the report's. Boosted trees, like single trees, do "
"not mind the units.\n\n"
"*Stuck for more than 15 minutes? Ask a friend, ask an AI for a hint (not the "
"answer), or email me at `jobo@econ.au.dk`.*"
)

md("---")

# ------------------------------------------------------------- quick load
md(
"## ⚙️ Quick load\n\n"
"The packages, the price table, and what the risk report holds after Part 11. "
"Run it and read what it prints."
)

_desk_lines = ",\n".join(f"    '{t}': ({DESK[t][0]:.5f}, {DESK[t][1]:.5f})" for t in TICKERS)
_forest_lines = ",\n".join(f"    '{t}': {DESK_FOREST[t]:.5f}" for t in TICKERS)

code(
XGB_GUARD + '''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor, XGBClassifier             # the setup guide's install line includes it

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

# --- What the risk report holds after Part 11, on Apple ---
part11_split = "by date: train to 2022-12-31, test from 2023-01-01"
part5_target = "sd of daily returns over the next 20 trading days"

# the forecast: Part 6's linear models and Part 11's trees, on the test block and the folds
part6_one_rmse = ''' + f"{ONE_TE:.5f}" + '''        # linear regression on vol_20d
part6_one_folds = ''' + f"{ONE_CV:.5f}" + '''
part6_ridge_rmse = ''' + f"{RIDGE_TE:.5f}" + '''      # ridge on all twenty columns
part6_ridge_folds = ''' + f"{RIDGE_CV:.5f}" + '''
part6_ridge_alpha = ''' + f"{RIDGE_ALPHA}" + '''         # the settings the folds chose in Part 6
part6_lasso_alpha = ''' + f"{LASSO_ALPHA}" + '''
part6_enet = ''' + f"{{'alpha': {ENET['alpha']}, 'l1_ratio': {ENET['l1_ratio']}}}" + '''
part6_desk = {                  # (ridge, one column) on each instrument's test block
''' + _desk_lines + '''
}
part11_stump_rmse = ''' + f"{STUMP_TE:.5f}" + '''     # a stump on vol_20d
part11_forest_settings = ''' + f"{FOREST_SET}" + '''   # chosen on the folds
part11_forest_rmse = ''' + f"{FOREST_TE:.5f}" + '''    # the tuned forest on all twenty columns
part11_forest_folds = ''' + f"{FOREST_CV:.5f}" + '''
part11_desk_forest = {          # the same forest on each instrument's test block
''' + _forest_lines + '''
}

# the warning: Part 9's label and models, and Part 11's forest on it (AUC)
part9_label = "jump: vol_next more than 1.5 times vol_20d"
part9_auc = ''' + f"{P9_AUC:.4f}" + '''              # logistic regression on vol_20d, the test block
part9_folds_auc = ''' + f"{P9_FOLDS:.4f}" + '''        # the same model on the folds
part9_wide_auc = ''' + f"{P9_WIDE_AUC:.4f}" + '''         # twenty columns, C chosen on the folds
part9_wide_folds_auc = ''' + f"{P9_WIDE_FOLDS:.4f}" + '''
part9_knn_auc = ''' + f"{P9_KNN_AUC:.4f}" + '''          # k-nearest neighbours on vol_20d, k chosen on the folds
part9_knn_folds_auc = ''' + f"{P9_KNN_FOLDS:.4f}" + '''
part9_threshold = ''' + f"{P9_CHOSEN}" + '''          # chosen from the desk's costs: a miss 5, a false alarm 1
part11_jump_forest_auc = ''' + f"{P11_JF_AUC:.4f}" + '''        # Part 11's tuned forest on the jump label
part11_jump_forest_folds_auc = ''' + f"{P11_JF_FOLDS:.4f}" + '''  # the same forest on the folds

print("Loaded prices:", prices.shape[0], "rows")
print("Instruments  :", ", ".join(TICKERS))
print()
print("After Part 11 the risk report holds, on Apple:")
print(f"  a forecast: one column, RMSE {part6_one_rmse:.5f} on the test block and {part6_one_folds:.5f} on the folds")
print(f"              (ridge {part6_ridge_rmse:.5f} and {part6_ridge_folds:.5f}; the tuned forest "
      f"{part11_forest_rmse:.5f} and {part11_forest_folds:.5f})")
print(f"  a warning : one column, AUC {part9_auc:.3f} on the test block and {part9_folds_auc:.3f} on the folds,")
print(f"              run at a threshold of {part9_threshold}")
print()
print("This is the last part. Boosting first, then every model side by side.")'''
)

md("---")

# ==================================================================== Q1
q("Q1", "Where the report stands",
  "Write `wide_table(ticker, keep_unknown=False)`. With the default it returns the "
  "table Parts 6 to 11 worked on, as you wrote it in Part 11: the six volatility "
  "windows, the three return windows, `up_20d`, the 20-day volatility of every "
  "other instrument, the target `vol_next`, and incomplete rows dropped. With "
  "`keep_unknown=True` it also keeps the last days of the data, whose features are "
  "known and whose `vol_next` is not yet, by dropping only rows with a missing "
  "feature. Build Apple's table both ways, as `table` and `latest`, store the "
  "twenty feature names in `columns`, and split `table` at the end of 2022. Then "
  "refit the report's forecast as `one_model` and check its test RMSE, `one_rmse`, "
  "against `part6_one_rmse`.",
  "def wide_table(ticker, keep_unknown=False):\n    \"\"\"...\"\"\"\n    ...\n\n\n"
  "table = ...\nlatest = ...\ncolumns = ...\ntrain = ...\ntest = ...\n\n"
  "one_model = ...\n...\none_rmse = ...\n\n"
  "print('days whose target is not known yet:', ...)\nprint(one_rmse)\nprint('matches the report:', ...)",
  ["Build `frame` as in Part 11. Before the last line, `if keep_unknown:` returns "
   "`frame.dropna(subset=list(frame.columns[:-1]))`, which only looks at the feature "
   "columns; otherwise `frame.dropna()` as before.",
   "`len(latest) - len(table)` counts the extra days, and the check is "
   "`abs(one_rmse - part6_one_rmse) < 0.00001`."],
  "def wide_table(ticker, keep_unknown=False):\n"
  "    \"\"\"The report's twenty columns and the target for one instrument, in plain decimals.\n"
  "    keep_unknown=True keeps the last days, whose target is not known yet.\"\"\"\n"
  "    frame = pd.DataFrame()\n"
  "    for w in [5, 10, 20, 40, 60, 120]:\n        frame['vol_' + str(w) + 'd'] = rets[ticker].rolling(w).std()\n"
  "    for w in [5, 20, 60]:\n        frame['ret_' + str(w) + 'd'] = rets[ticker].rolling(w).mean()\n"
  "    frame['up_20d'] = (rets[ticker] > 0).rolling(20).mean()\n"
  "    for t in TICKERS:\n        if t != ticker:\n            frame[t + '_vol'] = rets[t].rolling(20).std()\n"
  "    frame['vol_next'] = rets[ticker].rolling(20).std().shift(-20)\n"
  "    if keep_unknown:\n        return frame.dropna(subset=list(frame.columns[:-1]))\n"
  "    return frame.dropna()\n\n\n"
  "table = wide_table('AAPL')\nlatest = wide_table('AAPL', keep_unknown=True)\ncolumns = list(table.columns[:-1])\n"
  "train = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\n\n"
  "one_model = LinearRegression()\none_model.fit(train[['vol_20d']], train['vol_next'])\n"
  "one_rmse = rmse(test['vol_next'], one_model.predict(test[['vol_20d']]))\n\n"
  "print('days whose target is not known yet:', len(latest) - len(table))\nprint(one_rmse)\n"
  "print('matches the report:', abs(one_rmse - part6_one_rmse) < 0.00001)",
  f"20 days, {_span(UNKNOWN.index[0], UNKNOWN.index[-1])}: the data ends before "
  f"their next twenty trading days do. Then {ONE_TE:.5f} and `True`, on "
  f"{N_TRAIN:,} training days and {N_TEST} test days. Every model in this part is "
  "held to the forecast's two scores, the test block and the folds. Q14 returns to "
  "`latest` for the forecast those last 20 days are waiting for.")

# ==================================================================== Q2
q("Q2", "The first stump is Part 11's",
  f"Part 11's stump on `vol_20d` scored {STUMP_TE:.5f} with two numbers. Boosting "
  "starts from the training average and adds a tenth of a stump fitted to what is "
  "left. Refit Part 11's stump as `stump`, and fit `GradientBoostingRegressor` with "
  "100 stumps at a learning rate of 0.1 on `vol_20d` as `boost_one`. Check that "
  "its first stage is the average plus a tenth of the stump's distance from it, "
  "$\\hat f_1 = \\bar y + 0.1\\,(\\text{stump} - \\bar y)$. Then print the training RMSE after 1, 10 and 100 "
  "stumps.",
  "stump = ...\n...\nboost_one = ...\n...\n\naverage = ...\nstages = ...\ncheck = ...\n\n"
  "print('first stage = average + 0.1 x (stump - average):', check)\nprint(...)",
  ["`stages = list(boost_one.staged_predict(train[['vol_20d']]))`, so `stages[0]` is "
   "the first stage. The check is `np.allclose(stages[0], average + 0.1 * "
   "(stump.predict(train[['vol_20d']]) - average))`.",
   "`[rmse(train['vol_next'], stages[m - 1]) for m in [1, 10, 100]]`."],
  "stump = DecisionTreeRegressor(max_depth=1, random_state=0)\nstump.fit(train[['vol_20d']], train['vol_next'])\n"
  "boost_one = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)\n"
  "boost_one.fit(train[['vol_20d']], train['vol_next'])\n\n"
  "average = train['vol_next'].mean()\nstages = list(boost_one.staged_predict(train[['vol_20d']]))\n"
  "check = np.allclose(stages[0], average + 0.1 * (stump.predict(train[['vol_20d']]) - average))\n\n"
  "print('first stage = average + 0.1 x (stump - average):', check)\n"
  "print([round(rmse(train['vol_next'], stages[m - 1]), 5) for m in [1, 10, 100]])",
  f"`True`, and {Q2_TRAIN[0]:.5f}, {Q2_TRAIN[1]:.5f} and {Q2_TRAIN[2]:.5f}. A stump fitted "
  f"to y − ȳ cuts where a stump fitted to y cuts, at {STUMP_CUT:.4f}, and forecasts "
  "each side's mean minus ȳ, so boosting's first step is Part 11's stump, a tenth of "
  "the way. Every later stump is fitted to what the earlier ones left, and the "
  "training error falls with each. How many to keep is a setting, and Q3 chooses "
  "it without the test block.")

# ==================================================================== Q3
q("Q3", "How many stumps, chosen on 2022",
  "Split the training days into `fit_rows`, 2015 to 2021, and `stop_rows`, 2022. "
  "Fit 1,000 stumps at a learning rate of 0.1 on `vol_20d` in `fit_rows` as "
  "`boost_stop`, score every stage on 2022 into a list `stop_scores`, and keep the "
  "number of stumps with the lowest 2022 RMSE as `best_m`. Draw the 2022 RMSE "
  "against the number of stumps with `best_m` marked. Then open the test block "
  "once, for the model with `best_m` stumps, as `stopped_rmse`.",
  "fit_rows = ...\nstop_rows = ...\nboost_stop = ...\n...\nstop_scores = ...\nbest_m = ...\n\n"
  "fig, ax = plt.subplots(figsize=(9, 3))\n...\n...\nplt.show()\n\n"
  "stopped_rmse = ...\nprint(best_m, stopped_rmse, ' one column:', one_rmse)",
  ["`stop_scores = [rmse(stop_rows['vol_next'], f) for f in "
   "boost_stop.staged_predict(stop_rows[['vol_20d']])]` and `best_m = "
   "int(np.argmin(stop_scores)) + 1`.",
   "`ax.plot(range(1, 1001), stop_scores)` and `ax.scatter([best_m], [min(stop_scores)])`. "
   "The test forecast with `best_m` stumps is "
   "`list(boost_stop.staged_predict(test[['vol_20d']]))[best_m - 1]`."],
  "fit_rows = train.loc[:'2021-12-31']\nstop_rows = train.loc['2022-01-01':]\n"
  "boost_stop = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=1, random_state=0)\n"
  "boost_stop.fit(fit_rows[['vol_20d']], fit_rows['vol_next'])\n"
  "stop_scores = [rmse(stop_rows['vol_next'], f) for f in boost_stop.staged_predict(stop_rows[['vol_20d']])]\n"
  "best_m = int(np.argmin(stop_scores)) + 1\n\n"
  "fig, ax = plt.subplots(figsize=(9, 3))\nax.plot(range(1, 1001), stop_scores, color='#1c5cab')\n"
  "ax.scatter([best_m], [min(stop_scores)], s=60, color='#b3402f', zorder=3)\n"
  "ax.set_xlabel('stumps, fitted on 2015 to 2021')\nax.set_ylabel('RMSE on 2022')\n"
  "ax.set_title(f'Boosted stumps on vol_20d: the lowest 2022 RMSE at {best_m}', loc='left')\nplt.show()\n\n"
  "stopped_rmse = rmse(test['vol_next'], list(boost_stop.staged_predict(test[['vol_20d']]))[best_m - 1])\n"
  "print(best_m, stopped_rmse, ' one column:', one_rmse)",
  f"{BEST_M} stumps, at {STOP_SCORES[BEST_M - 1]:.5f} on 2022, against "
  f"{STOP_SCORES[0]:.5f} after one stump and {STOP_SCORES[-1]:.5f} after 1,000; anything "
  f"from {_flat.min()} to {_flat.max()} stumps comes within 0.00001 of the best. On the "
  f"test block the {BEST_M} stumps score {STOPPED_TE:.5f}, the lowest number the report "
  f"has printed, {ONE_TE - STOPPED_TE:.5f} below the one-column model. Q4 asks what "
  "that comparison is worth.")

# ==================================================================== Q4
q("Q4", "A fair comparison",
  "Q3's model was fitted on seven years and the report's forecast on eight. First "
  "fit the linear regression on `fit_rows` alone and score it on the test block, "
  "as `line_fit_rmse`. Then refit `best_m` stumps on all of `train`, as the report "
  "refits every model once its settings are chosen, and score it on the test "
  "block as `boost_rmse` and on the folds as `boost_folds`. Print the three "
  "comparisons.",
  "line_fit = ...\n...\nline_fit_rmse = ...\n\nboost_all = ...\n...\nboost_rmse = ...\nboost_folds = ...\n\n"
  "print('the same seven years:', stopped_rmse, ' against', line_fit_rmse)\n"
  "print('all eight years     :', boost_rmse, ' against', one_rmse)\n"
  "print('the folds           :', boost_folds, ' against', part6_one_folds)",
  ["`LinearRegression()`, fitted on `fit_rows[['vol_20d']]` and `fit_rows['vol_next']`.",
   "`GradientBoostingRegressor(n_estimators=best_m, learning_rate=0.1, max_depth=1, "
   "random_state=0)`; on the folds, `-cross_val_score(..., train[['vol_20d']], "
   "train['vol_next'], cv=folds, scoring='neg_root_mean_squared_error').mean()`."],
  "line_fit = LinearRegression()\nline_fit.fit(fit_rows[['vol_20d']], fit_rows['vol_next'])\n"
  "line_fit_rmse = rmse(test['vol_next'], line_fit.predict(test[['vol_20d']]))\n\n"
  "boost_all = GradientBoostingRegressor(n_estimators=best_m, learning_rate=0.1, max_depth=1, random_state=0)\n"
  "boost_all.fit(train[['vol_20d']], train['vol_next'])\n"
  "boost_rmse = rmse(test['vol_next'], boost_all.predict(test[['vol_20d']]))\n"
  "boost_folds = -cross_val_score(boost_all, train[['vol_20d']], train['vol_next'], cv=folds,\n"
  "                               scoring='neg_root_mean_squared_error').mean()\n\n"
  "print('the same seven years:', stopped_rmse, ' against', line_fit_rmse)\n"
  "print('all eight years     :', boost_rmse, ' against', one_rmse)\n"
  "print('the folds           :', boost_folds, ' against', part6_one_folds)",
  f"On the same seven years the linear regression scores {LINE_FIT_TE:.5f}, so more "
  f"than half of Q3's gap, {ONE_TE - LINE_FIT_TE:.5f} of {ONE_TE - STOPPED_TE:.5f}, came "
  "from leaving out 2022, the busiest year of the training block, and the rest from "
  "the stumps. Refitted on all eight years, the "
  f"stumps score {BOOST_TE:.5f} against {ONE_TE:.5f}. On the folds they lose, "
  f"{BOOST_CV:.5f} against {ONE_CV:.5f}. By the rule the report has used since Part "
  "11, a model that wins on the test block and loses on the folds has not won: the "
  "forecast stays as it is.")

# ==================================================================== Q5
_g = GRID_SCORE
q("Q5", "Twenty columns in XGBoost",
  "Give XGBoost all twenty columns. For every depth in `[1, 2, 3]` and learning "
  "rate in `[0.3, 0.1, 0.03]`, fit up to 3,000 trees on `fit_rows`, stopped on "
  "`stop_rows` 50 rounds after the best, and store the best 2022 RMSE in a "
  "dictionary `grid_2022` and the number of trees in `trees_2022`, both keyed by "
  "`(depth, rate)`. Keep the best combination as `wide_settings`, a dictionary of "
  "`n_estimators`, `learning_rate` and `max_depth`, refit it on all of `train` as "
  "`xgb_wide`, and print its test RMSE, `xgb_rmse`, and its mean fold RMSE, "
  "`xgb_folds`, beside ridge and the tuned forest.",
  "grid_2022 = {}\ntrees_2022 = {}\nfor depth in [1, 2, 3]:\n    for rate in [0.3, 0.1, 0.03]:\n        ...\n\n"
  "best = ...\nwide_settings = ...\nxgb_wide = ...\n...\nxgb_rmse = ...\nxgb_folds = ...\n\n"
  "print(best, wide_settings)\n"
  "print('test :', xgb_rmse, ' ridge:', part6_ridge_rmse, ' forest:', part11_forest_rmse)\n"
  "print('folds:', xgb_folds, ' ridge:', part6_ridge_folds, ' forest:', part11_forest_folds)",
  ["Inside the loops: `model = XGBRegressor(n_estimators=3000, learning_rate=rate, "
   "max_depth=depth, early_stopping_rounds=50, random_state=0)`, fitted with "
   "`eval_set=[(stop_rows[columns], stop_rows['vol_next'])]` and `verbose=False`; "
   "then `grid_2022[(depth, rate)] = model.best_score` and `trees_2022[(depth, rate)] "
   "= model.best_iteration + 1`.",
   "`best = min(grid_2022, key=grid_2022.get)`, `wide_settings = {'n_estimators': "
   "trees_2022[best], 'learning_rate': best[1], 'max_depth': best[0]}`, and "
   "`XGBRegressor(random_state=0, **wide_settings)` passes the dictionary as "
   "arguments."],
  "grid_2022 = {}\ntrees_2022 = {}\nfor depth in [1, 2, 3]:\n    for rate in [0.3, 0.1, 0.03]:\n"
  "        model = XGBRegressor(n_estimators=3000, learning_rate=rate, max_depth=depth,\n"
  "                             early_stopping_rounds=50, random_state=0)\n"
  "        model.fit(fit_rows[columns], fit_rows['vol_next'],\n"
  "                  eval_set=[(stop_rows[columns], stop_rows['vol_next'])], verbose=False)\n"
  "        grid_2022[(depth, rate)] = model.best_score\n        trees_2022[(depth, rate)] = model.best_iteration + 1\n\n"
  "best = min(grid_2022, key=grid_2022.get)\n"
  "wide_settings = {'n_estimators': trees_2022[best], 'learning_rate': best[1], 'max_depth': best[0]}\n"
  "xgb_wide = XGBRegressor(random_state=0, **wide_settings)\nxgb_wide.fit(train[columns], train['vol_next'])\n"
  "xgb_rmse = rmse(test['vol_next'], xgb_wide.predict(test[columns]))\n"
  "xgb_folds = -cross_val_score(XGBRegressor(random_state=0, **wide_settings), train[columns], train['vol_next'],\n"
  "                              cv=folds, scoring='neg_root_mean_squared_error').mean()\n\n"
  "print(best, wide_settings)\n"
  "print('test :', xgb_rmse, ' ridge:', part6_ridge_rmse, ' forest:', part11_forest_rmse)\n"
  "print('folds:', xgb_folds, ' ridge:', part6_ridge_folds, ' forest:', part11_forest_folds)",
  f"Stumps at a rate of 0.3 have the lowest 2022 RMSE, {_g[(1, 0.3)]:.5f}, stopped after "
  f"{WIDE_SET['n_estimators']} trees; every depth-1 setting beats every deeper one on "
  f"2022. Refitted on all eight years they score {XGB_TE:.5f} on the test block and "
  f"{XGB_CV:.5f} on the folds: behind ridge ({RIDGE_TE:.5f} and {RIDGE_CV:.5f}) and the "
  f"tuned forest ({FOREST_TE:.5f} and {FOREST_CV:.5f}) on both. On Apple, with twenty "
  "columns, boosting is the weakest of the three flexible forecasts.")
assert max(_g[(1, r)] for r in [0.3, 0.1, 0.03]) < min(_g[(d, r)] for d in [2, 3] for r in [0.3, 0.1, 0.03])

# ==================================================================== Q6
q("Q6", "What XGBoost leans on",
  "Read `xgb_wide` two ways. Print its five largest `feature_importances_` and the "
  "number of columns it never cut on. Then compute its permutation importance on "
  "the test block with `scoring='neg_root_mean_squared_error'` and ten repeats, and "
  "print the three columns it loses most without and the three it does best "
  "without.",
  "gain = ...\nnever_cut = ...\nresult = ...\nshuffled = ...\n\n"
  "print(...)\nprint('never cut on:', never_cut)\nprint(...)\nprint(...)",
  ["`gain = pd.Series(xgb_wide.feature_importances_, index=columns).sort_values(ascending=False)`, "
   "and `(gain == 0).sum()` counts the columns never used.",
   "`permutation_importance(xgb_wide, test[columns], test['vol_next'], "
   "scoring='neg_root_mean_squared_error', n_repeats=10, random_state=0)`, then a "
   "sorted Series of `result.importances_mean`; `.head(3)` and `.tail(3)`. With an "
   "error as the score, a positive number means the error rose when the column was "
   "shuffled."],
  "gain = pd.Series(xgb_wide.feature_importances_, index=columns).sort_values(ascending=False)\n"
  "never_cut = int((gain == 0).sum())\n"
  "result = permutation_importance(xgb_wide, test[columns], test['vol_next'],\n"
  "                                scoring='neg_root_mean_squared_error', n_repeats=10, random_state=0)\n"
  "shuffled = pd.Series(result.importances_mean, index=columns).sort_values(ascending=False)\n\n"
  "print(gain.head(5).round(3))\nprint('never cut on:', never_cut)\n"
  "print(shuffled.head(3).round(6))\nprint(shuffled.tail(3).round(6))",
  f"By gain, `DIS_vol` comes first with {GAIN['DIS_vol']:.3f}, ahead of `vol_20d` with "
  f"{GAIN['vol_20d']:.3f}, and {NEVER} of the twenty columns are never cut. Shuffled on "
  f"the test block, `vol_20d` matters most and `DIS_vol` least: shuffling Disney's "
  f"volatility lowers the test RMSE by {-PERM['DIS_vol']:.5f}. The trees found a "
  "pattern in Disney that held from 2015 to 2022 and not after; Part 11's stump on "
  "twenty columns asked about the same column first.")

# ==================================================================== Q7
_ch = CHANGE
q("Q7", "Three kinds of column",
  "Shuffle groups of columns together on the test block, one permutation of the "
  "rows for the whole group, ten times each: Apple's own volatility (the six "
  "`vol_` columns), its own returns (`ret_5d`, `ret_20d`, `ret_60d` and `up_20d`), and "
  "the desk (the ten `_vol` columns of the other instruments). Do it for `xgb_wide` "
  "and for Part 6's ridge, refitted as `ridge`, and store the mean change in test "
  "RMSE of each model in a dictionary `change`, keyed by the group.",
  "groups = {\n    'own volatility': ...,\n    'own returns': ...,\n    'the desk': ...,\n}\nridge = ...\n...\n\n"
  "change = {}\nfor name in groups:\n    ...\n\nprint(change)",
  ["`[c for c in columns if c.startswith('vol_')]` gives the six, and "
   "`[c for c in columns if c.endswith('_vol')]` the desk. `ridge = "
   "Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=part6_ridge_alpha))])`, "
   "fitted on `train[columns]`.",
   "Inside the loop, an inner loop over k from 0 to 9: `order = "
   "np.random.default_rng(k).permutation(len(test))`, `shuffled = test.copy()`, "
   "`shuffled[groups[name]] = test[groups[name]].values[order]`, then each model's "
   "RMSE on `shuffled[columns]` minus its RMSE on `test[columns]`. Store the two means "
   "as a tuple."],
  "groups = {\n    'own volatility': [c for c in columns if c.startswith('vol_')],\n"
  "    'own returns': ['ret_5d', 'ret_20d', 'ret_60d', 'up_20d'],\n"
  "    'the desk': [c for c in columns if c.endswith('_vol')],\n}\n"
  "ridge = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=part6_ridge_alpha))])\n"
  "ridge.fit(train[columns], train['vol_next'])\n\n"
  "base_xgb = rmse(test['vol_next'], xgb_wide.predict(test[columns]))\n"
  "base_ridge = rmse(test['vol_next'], ridge.predict(test[columns]))\n"
  "change = {}\nfor name in groups:\n    up_xgb, up_ridge = [], []\n    for k in range(10):\n"
  "        order = np.random.default_rng(k).permutation(len(test))\n        shuffled = test.copy()\n"
  "        shuffled[groups[name]] = test[groups[name]].values[order]\n"
  "        up_xgb.append(rmse(test['vol_next'], xgb_wide.predict(shuffled[columns])) - base_xgb)\n"
  "        up_ridge.append(rmse(test['vol_next'], ridge.predict(shuffled[columns])) - base_ridge)\n"
  "    change[name] = (np.mean(up_xgb), np.mean(up_ridge))\n\n"
  "print(pd.DataFrame(change, index=['XGBoost', 'ridge']).T)",
  f"Shuffling Apple's own volatility raises XGBoost's test RMSE by "
  f"{_ch['own volatility'][0]:.6f} and ridge's by {_ch['own volatility'][1]:.6f}. "
  f"Shuffling its own returns raises XGBoost's by {_ch['own returns'][0]:.6f} and leaves "
  f"ridge's almost where it was ({_ch['own returns'][1]:.6f}). Shuffling the desk's ten "
  f"columns lowers both, by {-_ch['the desk'][0]:.6f} and {-_ch['the desk'][1]:.6f}: the "
  "other instruments' volatility carried patterns from 2015 to 2022 that did not "
  "hold in 2023 and 2024. That is the report's finding since Part 6, measured "
  "directly: on Apple the extra columns cost more than they bring.")
assert abs(_ch["own returns"][1]) < 0.00001

# ==================================================================== Q8
_wb = WARN_ROWS
q("Q8", "The warning, once",
  "Make the jump label for the training and test days as `train_jump` and "
  "`test_jump`. Refit Part 9's warning, a scaled logistic regression on `vol_20d`, "
  "as `warning_model`, and check its test AUC against `part9_auc`. Then boost the "
  "label: XGBoost stumps at a learning rate of 0.1, the number of trees chosen on "
  "2022 by early stopping and then refitted on all training days, as `xgb_jump`. "
  "Collect the test and fold AUCs of the five warnings the report has fitted in a "
  "DataFrame `warning_board`.",
  "train_jump = ...\ntest_jump = ...\nwarning_model = ...\n...\nprint('Part 9 refitted:', ...)\n\n"
  "jump_stop = ...\n...\nxgb_jump = ...\n...\nxgb_jump_auc = ...\nxgb_jump_folds = ...\n\n"
  "warning_board = ...\nprint(warning_board)",
  ["The label is `(train['vol_next'] > 1.5 * train['vol_20d']).astype(int)`. For the "
   "stopping, fit on `fit_rows[columns]` with `train_jump.loc[:'2021-12-31']` and stop "
   "on `stop_rows[columns]` with `train_jump.loc['2022-01-01':]`.",
   "`warning_board = pd.DataFrame({'test': [...], 'folds': [...]}, index=[...])`, with "
   "the restored `part9_` and `part11_` numbers and your two; `.sort_values('folds')`."],
  "train_jump = (train['vol_next'] > 1.5 * train['vol_20d']).astype(int)\n"
  "test_jump = (test['vol_next'] > 1.5 * test['vol_20d']).astype(int)\n"
  "warning_model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
  "warning_model.fit(train[['vol_20d']], train_jump)\n"
  "print('Part 9 refitted:', abs(roc_auc_score(test_jump, warning_model.predict_proba(test[['vol_20d']])[:, 1])\n"
  "                              - part9_auc) < 0.0001)\n\n"
  "jump_stop = XGBClassifier(n_estimators=3000, learning_rate=0.1, max_depth=1, early_stopping_rounds=50,\n"
  "                          random_state=0)\n"
  "jump_stop.fit(fit_rows[columns], train_jump.loc[:'2021-12-31'],\n"
  "              eval_set=[(stop_rows[columns], train_jump.loc['2022-01-01':])], verbose=False)\n"
  "xgb_jump = XGBClassifier(n_estimators=jump_stop.best_iteration + 1, learning_rate=0.1, max_depth=1,\n"
  "                         random_state=0)\nxgb_jump.fit(train[columns], train_jump)\n"
  "xgb_jump_auc = roc_auc_score(test_jump, xgb_jump.predict_proba(test[columns])[:, 1])\n"
  "xgb_jump_folds = cross_val_score(xgb_jump, train[columns], train_jump, cv=folds, scoring='roc_auc').mean()\n\n"
  "warning_board = pd.DataFrame({\n"
  "    'test': [part9_auc, part9_wide_auc, part9_knn_auc, part11_jump_forest_auc, xgb_jump_auc],\n"
  "    'folds': [part9_folds_auc, part9_wide_folds_auc, part9_knn_folds_auc, part11_jump_forest_folds_auc,\n"
  "              xgb_jump_folds],\n"
  "}, index=['logistic regression, vol_20d (Part 9)', 'logistic regression, twenty columns (Part 9)',\n"
  "          'k-nearest neighbours, vol_20d (Part 9)', 'forest, twenty columns (Part 11)',\n"
  "          'XGBoost, twenty columns (Part 12)'])\n"
  "print(warning_board.sort_values('folds', ascending=False).round(3))",
  f"The refit matches Part 9: `True`. Stopped on 2022 after {JUMP_TREES} stumps and "
  f"refitted, boosting scores {XJ_AUC:.3f} on the test block and {XJ_CV:.3f} on the "
  "folds. Of the five warnings the report has fitted, the one-column logistic "
  f"regression is first on both yardsticks, {P9_AUC:.3f} and {P9_FOLDS:.3f}, as it has "
  f"been since Part 9; k-nearest neighbours is second on the folds, {P9_KNN_FOLDS:.3f}. "
  "The warning stays, threshold and all.")

# --------------------------------------------------------- the report as a whole
md(
"## The report as a whole\n\n"
"The forecast and the warning have now met every family of model in the course. "
"The rest of this part sets them side by side: every forecast the report has "
"fitted since Part 4 on the same rows and both yardsticks, the desk, the days "
"that went wrong, and the report as it would go out."
)

md("---")

# ==================================================================== Q9
_bf = BOARD
q("Q9", "Every forecast, on the same rows",
  "Refit every forecast the report has fitted since Part 5 on `train`, with the "
  "settings the report chose for it, and score each on the test block and on the "
  "folds. The dictionary below names them, with the part each comes from and its "
  "columns; fill in the models. Then add Part 4's two rules: the training average, "
  "and persistence, which repeats `vol_20d`. On the folds, the average is the mean "
  "of each fold's fitting rows. Collect everything in a DataFrame `board`, sorted "
  "by the test RMSE, and check that it reproduces the report's numbers for ridge "
  "and the tuned forest. This takes about a minute on Colab.",
  "one, three, pair = ['vol_20d'], ['vol_20d', 'ret_20d', 'up_20d'], ['vol_20d', 'ret_20d']\n"
  "candidates = {\n"
  "    'linear regression, vol_20d': (5, one, ...),\n"
  "    'linear regression, three columns': (5, three, ...),\n"
  "    'OLS, twenty columns': (6, columns, ...),\n"
  "    'ridge': (6, columns, ...),\n"
  "    'lasso': (6, columns, ...),\n"
  "    'elastic net': (6, columns, ...),\n"
  "    'stump, vol_20d': (11, one, ...),\n"
  "    'stump, twenty columns': (11, columns, ...),\n"
  "    'tree of depth 2, two columns': (11, pair, ...),\n"
  "    'bagging': (11, columns, ...),\n"
  "    'forest': (11, columns, ...),\n"
  "    'tuned forest': (11, columns, ...),\n"
  "    'boosted stumps, vol_20d': (12, one, ...),\n"
  "    'XGBoost, twenty columns': (12, columns, ...),\n"
  "}\n\nrows = []\nfor name in candidates:\n    ...\n\n...\n\nboard = ...\nprint(board)\nprint(...)",
  ["Ridge, the lasso and the elastic net sit in a pipeline with a `StandardScaler`, "
   "with `part6_ridge_alpha`, `part6_lasso_alpha` and `part6_enet`; the tuned forest "
   "is `RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1, "
   "**part11_forest_settings)`; bagging is a forest with `max_features=1.0`, the plain "
   "forest `'sqrt'`; the boosted models use `best_m` and `wide_settings`.",
   "In the loop: `part, cols, model = candidates[name]`, fit, then append "
   "`{'model': name, 'part': part, 'test': ..., 'folds': ...}`. For the two rules, "
   "loop `for fit_idx, val_idx in folds.split(train):` and use "
   "`train['vol_next'].iloc[fit_idx].mean()` and `train['vol_20d'].iloc[val_idx]`. "
   "The check: `abs(board.loc['ridge', 'test'] - part6_ridge_rmse) < 0.00001`, and the "
   "same for the tuned forest."],
  "one, three, pair = ['vol_20d'], ['vol_20d', 'ret_20d', 'up_20d'], ['vol_20d', 'ret_20d']\n"
  "candidates = {\n"
  "    'linear regression, vol_20d': (5, one, LinearRegression()),\n"
  "    'linear regression, three columns': (5, three, LinearRegression()),\n"
  "    'OLS, twenty columns': (6, columns, LinearRegression()),\n"
  "    'ridge': (6, columns, Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=part6_ridge_alpha))])),\n"
  "    'lasso': (6, columns, Pipeline([('scale', StandardScaler()),\n"
  "                                    ('lasso', Lasso(alpha=part6_lasso_alpha, max_iter=20000))])),\n"
  "    'elastic net': (6, columns, Pipeline([('scale', StandardScaler()),\n"
  "                                          ('enet', ElasticNet(alpha=part6_enet['alpha'],\n"
  "                                                              l1_ratio=part6_enet['l1_ratio'], max_iter=20000))])),\n"
  "    'stump, vol_20d': (11, one, DecisionTreeRegressor(max_depth=1, random_state=0)),\n"
  "    'stump, twenty columns': (11, columns, DecisionTreeRegressor(max_depth=1, random_state=0)),\n"
  "    'tree of depth 2, two columns': (11, pair, DecisionTreeRegressor(max_depth=2, random_state=0)),\n"
  "    'bagging': (11, columns, RandomForestRegressor(n_estimators=100, max_features=1.0, random_state=0, n_jobs=-1)),\n"
  "    'forest': (11, columns, RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=0, n_jobs=-1)),\n"
  "    'tuned forest': (11, columns, RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1,\n"
  "                                                        **part11_forest_settings)),\n"
  "    'boosted stumps, vol_20d': (12, one, GradientBoostingRegressor(n_estimators=best_m, learning_rate=0.1,\n"
  "                                                                   max_depth=1, random_state=0)),\n"
  "    'XGBoost, twenty columns': (12, columns, XGBRegressor(random_state=0, **wide_settings)),\n"
  "}\n\nrows = []\nfor name in candidates:\n    part, cols, model = candidates[name]\n"
  "    model.fit(train[cols], train['vol_next'])\n"
  "    folds_rmse = -cross_val_score(model, train[cols], train['vol_next'], cv=folds,\n"
  "                                  scoring='neg_root_mean_squared_error').mean()\n"
  "    rows.append({'model': name, 'part': part, 'test': rmse(test['vol_next'], model.predict(test[cols])),\n"
  "                 'folds': folds_rmse})\n\n"
  "average_folds, persistence_folds = [], []\nfor fit_idx, val_idx in folds.split(train):\n"
  "    actual = train['vol_next'].iloc[val_idx]\n"
  "    average_folds.append(rmse(actual, np.full(len(val_idx), train['vol_next'].iloc[fit_idx].mean())))\n"
  "    persistence_folds.append(rmse(actual, train['vol_20d'].iloc[val_idx]))\n"
  "rows.append({'model': 'training average', 'part': 4, 'folds': np.mean(average_folds),\n"
  "             'test': rmse(test['vol_next'], np.full(len(test), train['vol_next'].mean()))})\n"
  "rows.append({'model': 'persistence', 'part': 4, 'folds': np.mean(persistence_folds),\n"
  "             'test': rmse(test['vol_next'], test['vol_20d'])})\n\n"
  "board = pd.DataFrame(rows).set_index('model').sort_values('test')\nprint(board.round(5))\n"
  "print('reproduces the report:', abs(board.loc['ridge', 'test'] - part6_ridge_rmse) < 0.00001,\n"
  "      abs(board.loc['tuned forest', 'test'] - part11_forest_rmse) < 0.00001)",
  f"Sixteen forecasts, and `True True`: one loop reproduces numbers the report "
  f"printed in Parts 6 and 11. On the test block the boosted stumps lead with "
  f"{_bf['test'].iloc[0]:.5f}, then the tree of depth 2 with "
  f"{_bf.loc['tree of depth 2, two columns', 'test']:.5f}, then the report's linear "
  f"regression with {ONE_TE:.5f}. On the folds the report's linear regression leads "
  f"with {ONE_CV:.5f}, ahead of the tuned forest ({BY_FOLDS['folds'].iloc[1]:.5f}) and "
  f"ridge ({BY_FOLDS['folds'].iloc[2]:.5f}). Both models ahead of it on the test block "
  f"are behind it on the folds, by {_bf.loc['boosted stumps, vol_20d', 'folds'] - ONE_CV:.5f} "
  f"and {_bf.loc['tree of depth 2, two columns', 'folds'] - ONE_CV:.5f}. Keep `board`: "
  "Q10 and Q11 read it.")

# ==================================================================== Q10
_worst3 = list(BOARD.sort_values("test").index[-3:])
q("Q10", "Draw the scoreboard",
  "Draw `board` from Q9 as two panels of horizontal bars side by side, the test "
  "block on the left and the folds on the right, with the models in the same order "
  "in both and the report's forecast in a second colour. Let the bars start at zero.",
  "fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)\n...\n...\n...\nplt.show()",
  ["`colours = ['#b3402f' if name == 'linear regression, vol_20d' else '#1c5cab' for "
   "name in board.index]`.",
   "`axes[0].barh(board.index, board['test'], color=colours)` and the same with "
   "`board['folds']` on `axes[1]`; `axes[0].invert_yaxis()` puts the best test score "
   "at the top."],
  "colours = ['#b3402f' if name == 'linear regression, vol_20d' else '#1c5cab' for name in board.index]\n\n"
  "fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)\n"
  "axes[0].barh(board.index, board['test'], color=colours)\naxes[1].barh(board.index, board['folds'], color=colours)\n"
  "axes[0].invert_yaxis()\naxes[0].set_title('RMSE on the test block, 2023 and 2024', loc='left')\n"
  "axes[1].set_title('mean RMSE on the five folds', loc='left')\nfig.tight_layout()\nplt.show()",
  "On bars that start at zero most of the sixteen look alike, and they are: the "
  f"best test score is {GAP_TOP:.5f} below the report's forecast, whose error is "
  f"{ONE_TE:.5f}. The bars that stand out belong to the models that broke something: "
  "OLS on twenty columns, bagging, and the stump on twenty columns. Every bar on "
  "the right is longer than its partner on the left, because the folds include "
  "2020 and the test years were calm.")
assert set(_worst3) == {"OLS, twenty columns", "bagging", "stump, twenty columns"}
assert (BOARD["folds"] > BOARD["test"]).all()

# ==================================================================== Q11
_bp = BEST_BY_PART
q("Q11", "How far the report came",
  "Measure each forecast against the training average, the rule Part 4 started "
  "from: its skill is the share of the average's squared error it removes on the "
  "test block, $1 - (\\text{RMSE} / \\text{RMSE}_{\\text{average}})^2$. Store it as a Series `skill`, "
  "sorted. Then use `groupby` on `part` to print the best test RMSE and the best "
  "fold RMSE of each part.",
  "average_rmse = ...\nskill = ...\nbest_by_part = ...\n\nprint(skill)\nprint(best_by_part)",
  ["`average_rmse = board.loc['training average', 'test']` and "
   "`skill = (1 - (board['test'] / average_rmse) ** 2).sort_values(ascending=False)`.",
   "`board.groupby('part')[['test', 'folds']].min()`."],
  "average_rmse = board.loc['training average', 'test']\n"
  "skill = (1 - (board['test'] / average_rmse) ** 2).sort_values(ascending=False)\n"
  "best_by_part = board.groupby('part')[['test', 'folds']].min()\n\n"
  "print(skill.round(3))\nprint(best_by_part.round(5))",
  f"Persistence, the rule from Part 4 that repeats this month's volatility, "
  f"removes {SKILL['persistence']:.3f} of the average's squared error; the report's "
  f"linear regression {SKILL['linear regression, vol_20d']:.3f}; the best model of the "
  f"course {SKILL.max():.3f}. Most of the distance was covered before any model was "
  f"fitted. By part, the best test RMSE went from {_bp.loc[4, 'test']:.5f} (Part 4) to "
  f"{_bp.loc[5, 'test']:.5f} (5), up to {_bp.loc[6, 'test']:.5f} with the wide models of "
  f"Part 6, and back to {_bp.loc[11, 'test']:.5f} (11) and {_bp.loc[12, 'test']:.5f} (12). "
  f"On the folds the best is still Part 5's {_bp.loc[5, 'folds']:.5f}; no later part "
  "beat it.")

# ==================================================================== Q12
_wc = WIN_COUNTS
q("Q12", "Every family, across the desk",
  "Fit XGBoost on every instrument with the depth and learning rate of "
  "`wide_settings`, the number of trees chosen on the instrument's own 2022 and "
  "refitted on its training block, and store the test RMSE in `desk_boost`. Put it "
  "in a DataFrame `desk` beside the one-column model and ridge from `part6_desk` "
  "and the forest from `part11_desk_forest`, find the family with the lowest RMSE "
  "for each instrument as `winners`, and count the wins.",
  "desk_boost = {}\nfor ticker in TICKERS:\n    ...\n\ndesk = ...\nwinners = ...\n\nprint(desk)\nprint(winners)\nprint(...)",
  ["Inside the loop: `frame = wide_table(ticker)`, split it at the end of 2022 and "
   "the training block again at the end of 2021, early-stop an `XGBRegressor` with "
   "`max_depth=wide_settings['max_depth']` and "
   "`learning_rate=wide_settings['learning_rate']`, then refit with "
   "`best_iteration + 1` trees on the whole training block.",
   "`desk = pd.DataFrame({'one column': {t: part6_desk[t][1] for t in TICKERS}, "
   "'ridge': {t: part6_desk[t][0] for t in TICKERS}, 'forest': part11_desk_forest, "
   "'boosting': desk_boost})`; then `winners = desk.idxmin(axis=1)` and "
   "`winners.value_counts()`."],
  "desk_boost = {}\nfor ticker in TICKERS:\n    frame = wide_table(ticker)\n"
  "    tr, te = frame.loc[:'2022-12-31'], frame.loc['2023-01-01':]\n"
  "    fr, sr = tr.loc[:'2021-12-31'], tr.loc['2022-01-01':]\n    cols = list(frame.columns[:-1])\n"
  "    stopper = XGBRegressor(n_estimators=3000, learning_rate=wide_settings['learning_rate'],\n"
  "                           max_depth=wide_settings['max_depth'], early_stopping_rounds=50, random_state=0)\n"
  "    stopper.fit(fr[cols], fr['vol_next'], eval_set=[(sr[cols], sr['vol_next'])], verbose=False)\n"
  "    model = XGBRegressor(n_estimators=stopper.best_iteration + 1, learning_rate=wide_settings['learning_rate'],\n"
  "                         max_depth=wide_settings['max_depth'], random_state=0)\n"
  "    model.fit(tr[cols], tr['vol_next'])\n    desk_boost[ticker] = rmse(te['vol_next'], model.predict(te[cols]))\n\n"
  "desk = pd.DataFrame({'one column': {t: part6_desk[t][1] for t in TICKERS},\n"
  "                     'ridge': {t: part6_desk[t][0] for t in TICKERS},\n"
  "                     'forest': part11_desk_forest, 'boosting': desk_boost})\nwinners = desk.idxmin(axis=1)\n\n"
  "print(desk.round(5))\nprint(winners)\nprint(winners.value_counts())",
  f"Boosting has the lowest RMSE on {_wc['boosting']} of the 11 ({', '.join(BOOST_WINS)}), "
  f"the one-column model on {_wc['one column']} ({', '.join(ONE_WINS)}), ridge and the "
  f"forest on {_wc['ridge']} each. No family wins the desk, and Apple is one of the "
  f"instruments where the simplest one does. The spread is wide: on {WORST_T} boosting's "
  f"error is twice the one column's, {DESK_BOOST[WORST_T]:.5f} against {DESK[WORST_T][1]:.5f}.")
assert _wc["ridge"] == _wc["forest"] == 2 and 1.95 < DESK_BOOST[WORST_T] / DESK[WORST_T][1] < 2.05

# ==================================================================== Q13
q("Q13", "The days the forecast missed most",
  "Compute the report's forecast error on every test day, what happened minus "
  "`one_model`'s forecast, and the warning's probability from `warning_model`. Put "
  "them in a DataFrame `days` with `vol_20d` and `vol_next`, mark whether the "
  "warning was on at `part9_threshold`, and print the five days the forecast fell "
  "furthest short and the five it overshot most.",
  "errors = ...\nprobability = ...\ndays = ...\n...\n\nprint(...)\nprint(...)",
  ["`errors = test['vol_next'] - one_model.predict(test[['vol_20d']])` and "
   "`probability = warning_model.predict_proba(test[['vol_20d']])[:, 1]`.",
   "`days = pd.DataFrame({'vol_20d': test['vol_20d'], 'vol_next': test['vol_next'], "
   "'error': errors, 'probability': probability})`, then `days['warning'] = "
   "days['probability'] >= part9_threshold`. `days.nlargest(5, 'error')` and "
   "`days.nsmallest(5, 'error')`."],
  "errors = test['vol_next'] - one_model.predict(test[['vol_20d']])\n"
  "probability = warning_model.predict_proba(test[['vol_20d']])[:, 1]\n"
  "days = pd.DataFrame({'vol_20d': test['vol_20d'], 'vol_next': test['vol_next'],\n"
  "                     'error': errors, 'probability': probability})\n"
  "days['warning'] = days['probability'] >= part9_threshold\n\n"
  "print(days.nlargest(5, 'error').round(4))\nprint(days.nsmallest(5, 'error').round(4))",
  f"The five largest shortfalls are {_span(SHORT.index.min(), SHORT.index.max())}: "
  f"`vol_20d` was about {SHORT['vol_20d'].mean():.4f} and the next twenty days ran at "
  f"{SHORT['vol_next'].mean():.4f}. All five were jumps, and the warning was on for all "
  f"five, at probabilities near {SHORT['probability'].mean():.2f}. The five largest "
  f"overshoots are {_span(OVER.index.min(), OVER.index.max())}, after a busy month: "
  f"`vol_20d` was about {OVER['vol_20d'].mean():.4f} and {OVER['vol_next'].mean():.4f} "
  "came, and the warning was off. The forecast misses at the turns. The turns it "
  "misses upwards are the ones the warning was built for, and there it fired.")

# ==================================================================== Q14
q("Q14", "The forecast for the month after the data ends",
  "The last day in `latest` is 31 December 2024, where the data ends. Forecast its "
  "next twenty trading days with `one_model`, and put the forecast and the day's "
  "own `vol_20d` into annualised percent, as Part 3 did: "
  "$\\sigma_{\\text{year}} = \\sigma_{\\text{day}} \\times \\sqrt{252}$. "
  "Compute the warning's probability for that day, and count how many of the 20 "
  "days whose target is not known yet had the warning on.",
  "last_day = ...\nforecast_next = ...\nannual_next = ...\nannual_now = ...\nprobability_now = ...\n"
  "unknown = ...\nwarning_days = ...\n\n"
  "print('this month:', annual_now, ' next month:', annual_next)\n"
  "print('warning probability:', probability_now, ' on for', warning_days, 'of the last 20 days')",
  ["`last_day = latest.iloc[[-1]]` keeps it as a one-row table, so that "
   "`one_model.predict(last_day[['vol_20d']])[0]` works. Multiply by `np.sqrt(252)` "
   "and by 100.",
   "`unknown = latest[latest['vol_next'].isna()]`, and "
   "`(warning_model.predict_proba(unknown[['vol_20d']])[:, 1] >= part9_threshold).sum()`."],
  "last_day = latest.iloc[[-1]]\nforecast_next = one_model.predict(last_day[['vol_20d']])[0]\n"
  "annual_next = round(100 * forecast_next * np.sqrt(252), 1)\n"
  "annual_now = round(100 * last_day['vol_20d'].iloc[0] * np.sqrt(252), 1)\n"
  "probability_now = round(warning_model.predict_proba(last_day[['vol_20d']])[0, 1], 3)\n"
  "unknown = latest[latest['vol_next'].isna()]\n"
  "warning_days = int((warning_model.predict_proba(unknown[['vol_20d']])[:, 1] >= part9_threshold).sum())\n\n"
  "print('this month:', annual_now, ' next month:', annual_next)\n"
  "print('warning probability:', probability_now, ' on for', warning_days, 'of the last 20 days')",
  f"December 2024 ran at {ANN_NOW:.1f} percent a year, and the forecast for the next "
  f"twenty trading days is {ANN_NEXT:.1f} percent: the report expects volatility to "
  f"rise back towards its average. The warning's probability is {P_NOW:.3f}, above the "
  f"threshold of {P9_CHOSEN}, and it was on for all {WARN_UNKNOWN} of the last days. The "
  "data ends here, so this is the one number in the report that cannot be scored "
  "yet.")

# ==================================================================== Q15
q("Q15", "Write down what you would defend",
  "Collect what the report would publish in a dictionary `report`, and write "
  "`summarise(report)`: it prints one line per entry and ends with three verdicts, "
  "on the forecast, on the warning, and on next month.",
  "report = {\n    'target': ...,\n    'split': ...,\n    'forecast_model': ...,\n    'forecast_rmse_test': ...,\n"
  "    'forecast_rmse_folds': ...,\n    'forecasts_compared': ...,\n    'best_on_test': ...,\n"
  "    'best_on_folds': ...,\n    'desk_wins': ...,\n    'warning_model': ...,\n    'warning_auc_test': ...,\n"
  "    'warning_auc_folds': ...,\n    'warning_threshold': ...,\n    'next_month_percent': ...,\n"
  "    'warning_now': ...,\n}\n\n\ndef summarise(r):\n    \"\"\"...\"\"\"\n    ...\n\n\nsummarise(report)",
  ["Every value is something you already have: `part5_target`, `part11_split`, the "
   "name `'linear regression, vol_20d'`, `one_rmse`, `part6_one_folds`, `len(board)`, "
   "`board['test'].idxmin()`, `board['folds'].idxmin()`, `winners.value_counts().to_dict()`, "
   "Part 9's numbers, `annual_next`, and `'on'` or `'off'` from `probability_now`.",
   "Inside the function, `for key, value in r.items():` with an f-string, then an "
   "`if` on whether `r['best_on_folds']` is the report's forecast."],
  "report = {\n    'target': part5_target,\n    'split': part11_split,\n"
  "    'forecast_model': 'linear regression, vol_20d',\n    'forecast_rmse_test': round(one_rmse, 5),\n"
  "    'forecast_rmse_folds': part6_one_folds,\n    'forecasts_compared': len(board),\n"
  "    'best_on_test': board['test'].idxmin(),\n    'best_on_folds': board['folds'].idxmin(),\n"
  "    'desk_wins': winners.value_counts().to_dict(),\n"
  "    'warning_model': 'logistic regression, vol_20d',\n    'warning_auc_test': part9_auc,\n"
  "    'warning_auc_folds': part9_folds_auc,\n    'warning_threshold': part9_threshold,\n"
  "    'next_month_percent': annual_next,\n"
  "    'warning_now': 'on' if probability_now >= part9_threshold else 'off',\n}\n\n\n"
  "def summarise(r):\n    \"\"\"Print the report and the three verdicts it supports.\"\"\"\n"
  "    for key, value in r.items():\n        print(f'{key:<22}{value}')\n    print()\n"
  "    if r['best_on_folds'] == r['forecast_model']:\n"
  "        print(f\"Forecast  : {r['forecast_model']} stays. It is first on the folds, and the models\")\n"
  "        print(f\"            ahead of it on the test block, led by {r['best_on_test']}, are behind it there.\")\n"
  "    else:\n"
  "        print(f\"Forecast  : {r['best_on_folds']} is first on the folds; the report's model needs a second look.\")\n"
  "    print(f\"Warning   : {r['warning_model']}, run at a threshold of {r['warning_threshold']}.\")\n"
  "    print(f\"Next month: {r['next_month_percent']} percent a year, and the warning is {r['warning_now']}.\")\n\n\n"
  "summarise(report)",
  "After twelve parts the report publishes the two models it had by Part 9, and "
  "now it can say why. Sixteen forecasts and five warnings were fitted to the same "
  "question, from two rules to boosted trees; the forecast it keeps has two "
  "coefficients and is first on the folds, and the warning has two coefficients "
  "and a threshold set from the desk's costs. The flexible models were not wasted: "
  "they are why the simple ones can be defended, and on four instruments of the "
  "desk boosting is the better forecast.")

# ---------------------------------------------------------------- closing
md(
"## \U0001f4e6 What you now have\n\n"
"| what | where |\n|:--|:--|\n"
"| the report's table, with the days still waiting for their target | `wide_table`, `table`, `latest`, `columns`, `train`, `test`, `one_model`, `one_rmse` |\n"
"| boosting on the report's column, stopped on 2022 and refitted | `stump`, `boost_one`, `fit_rows`, `stop_rows`, `stop_scores`, `best_m`, `stopped_rmse`, `line_fit_rmse`, `boost_rmse`, `boost_folds` |\n"
"| boosting on twenty columns, and what it leans on | `grid_2022`, `wide_settings`, `xgb_wide`, `xgb_rmse`, `xgb_folds`, `gain`, `shuffled`, `change` |\n"
"| the warning, refitted and boosted | `train_jump`, `test_jump`, `warning_model`, `xgb_jump`, `warning_board` |\n"
"| every forecast on the same rows | `board`, `skill`, `best_by_part` |\n"
"| the desk, every family | `desk_boost`, `desk`, `winners` |\n"
"| the misses, and next month | `days`, `annual_next`, `probability_now`, `warning_days` |\n"
"| the thing you would defend | `report` |"
)

md(
"## What changed since Part 11\n\n"
f"- **Boosting came closest on the report's own column.** Stopped on 2022, {BEST_M} "
f"stumps on `vol_20d` scored {STOPPED_TE:.5f}, the lowest test number in the report. "
f"Fitted on the same seven years, the linear regression scores {LINE_FIT_TE:.5f}; "
f"refitted on all eight, the stumps score {BOOST_TE:.5f} against {ONE_TE:.5f}, and "
f"lose on the folds, {BOOST_CV:.5f} against {ONE_CV:.5f} (Q3 and Q4).\n"
f"- **On twenty columns boosting came last of the three.** XGBoost stumps at a rate "
f"of 0.3, {WIDE_SET['n_estimators']} trees chosen on 2022: {XGB_TE:.5f} and {XGB_CV:.5f}, "
f"behind ridge and the tuned forest on both yardsticks (Q5).\n"
"- **The importance explains the wide models' trouble.** XGBoost leaned hardest on "
"Disney's volatility, which permutation shows to hurt on the test days, and "
"shuffling the desk's ten columns improves both XGBoost and ridge there (Q6 and "
"Q7).\n"
f"- **The warning is unchanged.** Boosted on Part 9's label it scores {XJ_AUC:.3f} and "
f"{XJ_CV:.3f}, below the one-column logistic regression on both (Q8).\n"
f"- **Everything side by side.** Of sixteen forecasts, the report's is first on the "
f"folds and third on the test block (Q9), and it removes {SKILL['linear regression, vol_20d']:.3f} "
f"of the average's squared error against {SKILL['persistence']:.3f} for Part 4's rule (Q11). "
f"Across the desk, boosting wins on {_wc['boosting']} instruments, the one column on "
f"{_wc['one column']}, ridge and the forest on 2 each (Q12).\n"
"- **The two halves of the report cover each other.** The forecast's five worst "
"shortfalls were jumps in June 2024, and the warning was on for all five (Q13)."
)

md(
"## The report, part by part\n\n"
"| part | the question | what it added | the number |\n|:--|:--|:--|:--|\n"
f"| 1 | Which of Apple and Coca-Cola moved more in 2024? | lists, slicing and a first gauge, the price range | Apple {100 * P1['AAPL'][0]:.2f} percent, Coca-Cola {100 * P1['KO'][0]:.2f} |\n"
"| 2 | Volatility from every daily return | loops, functions, a dictionary of five stocks | Nvidia 0.0330 a day, Coca-Cola 0.0080 |\n"
f"| 3 | All eleven instruments | pandas: returns, annualised volatility, rankings, figures | Nvidia {100 * P3['NVDA']:.0f} percent a year, the S&P 500 fund {100 * P3['SPY']:.0f} |\n"
f"| 4 | Which names will be volatile next month? | a target, features from the past, a split by date, two baselines | the average {P4_BASE:.5f}, persistence {P4_PERS:.5f} |\n"
f"| 5 | The first model | linear regression on `vol_20d`, chosen on time-ordered folds | {P5_RMSE:.5f}, ahead of persistence on 11 of 11 |\n"
f"| 6 | Every column the desk has | OLS, ridge, the lasso, pipelines, `GridSearchCV` | ridge {RIDGE_TE:.5f} against {ONE_TE:.5f}; ahead on 7 of 11 |\n"
f"| 8 | Will next month be busier? | logistic regression, the threshold, the AUC | AUC {P8_AUC:.3f} |\n"
f"| 9 | A jump, priced | a rare label, costs, a threshold of {P9_CHOSEN}, calibration, k-NN | AUC {P9_AUC:.3f}; a cost of {P9_COST} against {P9_COST_HALF} |\n"
f"| 11 | Trees and forests | a tree, bagging, a forest tuned on the folds, out-of-bag | the forest {FOREST_TE:.5f}, level with ridge |\n"
f"| 12 | Boosting, and the report as a whole | boosting, XGBoost, importance, every model side by side | boosted stumps {BOOST_TE:.5f} and {BOOST_CV:.5f}; the forecast stays |\n\n"
"Parts 1 to 3 described a year that had happened. Part 4 turned the report into "
"a question about the next month and set the rules every later part kept: a "
"target that cannot see the future, a split by date, and baselines to beat. "
"Parts 5 to 12 then brought one family of models after another to that question, "
"and each was held to the same test block and the same folds."
)

md(
"## Where this leaves the risk report\n\n"
"The report ends where Part 5 left its forecast and Part 9 its warning, and that "
"is a result, not a failure to improve. On Apple, the volatility of the last "
"twenty days, used with two coefficients, carries almost everything that the "
"desk's twenty columns and every family of model could find. Across the desk the "
"answer differs by instrument, and the report says so.\n\n"
"Part 4 named three problems with its own setup. Two have answers. One split is "
"one experiment, so every model was held to five time-ordered folds as well. The "
"rows are not independent, which is why the folds keep time in order and why "
"out-of-bag flattered the forest. The third, that the regimes differ, has no "
"answer: every forecast fitted on 2015 to 2022 ran high, on average, in the calm "
"years that followed, and the jump warning is there for the turns that no "
"forecast sees coming.\n\n"
"**This is the last part of the case.** The report is complete: a forecast, a "
"warning, the evidence for both, and a forecast for the month after the data "
"ends."
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
