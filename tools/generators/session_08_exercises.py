# -*- coding: utf-8 -*-
"""Build session_08_exercises.ipynb.

Same conventions as Sessions 1 to 6: pleasant intro, 1-5 star badges, toolkit
card with title= hover docs, task -> work cell (blank-safe `...`) -> 1-2 folded
hints -> folded solution, no em-dashes, plain explanatory tone.

Session 8 is classification with logistic regression. The exercises work on the
lecture's table (the index, 19 columns, percent) and its label, "rising":
whether the next 20 days were more volatile than the last 20. They make
labels, watch a straight line fail on one, fit and read the logistic curve,
turn probabilities into predictions, score with the four counts and AUC, cross-
validate, choose C, use the l1 penalty, and finally run the workflow on every
instrument.

Only tools taught by the end of Session 8. New this session:
LogisticRegression (C, penalty, solver, max_iter, coef_, intercept_, classes_,
n_iter_), predict_proba, np.exp, log_loss, accuracy_score, confusion_matrix,
precision_score, recall_score, roc_curve, roc_auc_score, scoring='roc_auc' and
'accuracy'. NOT taught, so never required: classification_report,
DummyClassifier, class_weight, precision_recall_curve, LogisticRegressionCV,
RocCurveDisplay, pd.cut.

Returns are in PERCENT here, exactly as in the lecture.

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
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                             roc_auc_score, roc_curve, log_loss)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "session_08" / "session_08_exercises.ipynb"

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
    "A1": "S3", "A3": "S3", "A5": "S2", "A6": "S2",
    "B2": "S3", "B3": "S2", "B6": "S2", "B7": "S1", "B8": "S3",
    "C2": "S3", "C4": "S1",
    "D3": "S3", "D4": "S6",
    "E1": "S3", "E2": "S4", "E4": "S4", "E5": "S2", "E6": "S2", "E7": "S4",
    "F2": "S3", "F5": "S3", "F6": "S5", "F7": "S3",
    "G2": "S5", "G4": "S2", "G6": "S6", "G8": "S6", "G9": "S6", "G10": "S5",
    "H2": "S2", "H4": "S1",
    "I1": "S1", "I2": "S2", "I3": "S2", "I4": "S5", "I5": "S3",
    "J1": "S6", "J3": "S3", "J4": "S2",
    "K1": "S2", "K2": "S3", "K3": "S2", "K4": "S2", "K5": "S3",
}


def ex(sid, title, n, task, work, hints, sol_code, sol_note, revisits=None, raises=False):
    md(f"### {sid} · {title}  {badge(n, revisits or REVISITS.get(sid))}\n\n{task}")
    code(work, raises=raises)
    _hints_solution(hints, sol_code, sol_note)


def section(header):
    md(header)


# ---- real numbers, computed here so every solution note is exact -----------
PX = pd.read_csv(ROOT / "data" / "prices.csv", parse_dates=["date"])
WIDE = PX.pivot(index="date", columns="ticker", values="close")
RET = WIDE.pct_change().dropna() * 100
TBL = pd.read_csv(ROOT / "data" / "market_features.csv", parse_dates=["date"]).set_index("date")
COLS = list(TBL.columns[:-1])
TBL["rising"] = (TBL["vol_next"] > TBL["vol_20d"]).astype(int)
TRAIN, TEST = TBL.loc[:"2022-12-31"], TBL.loc["2023-01-01":]
N_TBL, N_TRAIN, N_TEST = len(TBL), len(TRAIN), len(TEST)
TS5 = TimeSeriesSplit(n_splits=5)
Y_TR, Y_TE = TRAIN["rising"], TEST["rising"]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cv_auc(model, cols, frame=TRAIN):
    return cross_val_score(model, frame[cols], frame["rising"], cv=TS5, scoring="roc_auc")


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
    frame = frame.dropna()
    frame["rising"] = (frame["vol_next"] > frame["vol_20d"]).astype(int)
    return frame


# A: labels
SHARE_TR, SHARE_TE = float(Y_TR.mean()), float(Y_TE.mean())
MAJ_TE = float(max(SHARE_TE, 1 - SHARE_TE))
N_RISE_TE = int(Y_TE.sum())
BUSY = (TBL["vol_next"] > 1.0).astype(int)
BUSY_TR, BUSY_TE = float(BUSY.loc[:"2022-12-31"].mean()), float(BUSY.loc["2023-01-01":].mean())
BUSY_MAJ_TE = float(max(BUSY_TE, 1 - BUSY_TE))
RISE20 = (TBL["vol_next"] > 1.2 * TBL["vol_20d"]).astype(int)
RISE20_TR = float(RISE20.loc[:"2022-12-31"].mean())
MARGINS = [1.0, 1.1, 1.2, 1.5, 2.0]
SHARE_BY_MARGIN = {m: round(float(((TRAIN["vol_next"] > m * TRAIN["vol_20d"]).astype(int)).mean()), 3) for m in MARGINS}

# B: line and curve
LINE = LinearRegression().fit(TRAIN[["vol_20d"]], Y_TR)
LINE_B0, LINE_B1 = float(LINE.intercept_), float(LINE.coef_[0])
_fit = LINE.predict(TRAIN[["vol_20d"]])
N_OUTSIDE = int(((_fit < 0) | (_fit > 1)).sum())
FIT_MIN, FIT_MAX = float(_fit.min()), float(_fit.max())
MODEL = LogisticRegression().fit(TRAIN[["vol_20d"]], Y_TR)
B0, B1 = float(MODEL.intercept_[0]), float(MODEL.coef_[0, 0])
CROSS = -B0 / B1
ODDS_RATIO = float(np.exp(B1))
P_TEST = MODEL.predict_proba(TEST[["vol_20d"]])[:, 1]
PRED = MODEL.predict(TEST[["vol_20d"]])
HAND_VOLS = [0.5, 1.0, 2.0]
HAND_P = [float(sigmoid(B0 + B1 * v)) for v in HAND_VOLS]
BANDS = [(0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.3), (1.3, 1.7), (1.7, 2.5), (2.5, 6)]
BAND_SHARE = []
for lo, hi in BANDS:
    m = (TRAIN["vol_20d"] >= lo) & (TRAIN["vol_20d"] < hi)
    BAND_SHARE.append((float(TRAIN["vol_20d"][m].mean()), float(Y_TR[m].mean()), int(m.sum())))

# C: probabilities and predictions
SHARE_PRED_1 = float(PRED.mean())
N_PRED_60 = int((P_TEST >= 0.6).sum())
AUC_WRONG_COL = float(roc_auc_score(Y_TE, MODEL.predict_proba(TEST[["vol_20d"]])[:, 0]))
AUC_TE = float(roc_auc_score(Y_TE, P_TEST))

# D: fitting
LL_MODEL = float(log_loss(Y_TR, MODEL.predict_proba(TRAIN[["vol_20d"]])[:, 1]))
LL_HALF = float(log_loss(Y_TR, np.full(N_TRAIN, 0.5)))
LL_SHARE = float(log_loss(Y_TR, np.full(N_TRAIN, SHARE_TR)))
STEPS = []
for n in [1, 2, 3, 10]:
    m = LogisticRegression(max_iter=n).fit(TRAIN[["vol_20d"]], Y_TR)
    STEPS.append((n, int(m.n_iter_[0]), float(m.coef_[0, 0]), float(log_loss(Y_TR, m.predict_proba(TRAIN[["vol_20d"]])[:, 1]))))
_p_tr = MODEL.predict_proba(TRAIN[["vol_20d"]])[:, 1]
LL_HAND = float(-(Y_TR * np.log(_p_tr) + (1 - Y_TR) * np.log(1 - _p_tr)).mean())
_few = LogisticRegression(max_iter=5).fit(TRAIN[COLS], Y_TR)
_full = LogisticRegression(max_iter=1000).fit(TRAIN[COLS], Y_TR)
N_ITER_FULL = int(_full.n_iter_[0])
AUC_FEW, AUC_FULL = float(roc_auc_score(Y_TE, _few.predict_proba(TEST[COLS])[:, 1])), float(roc_auc_score(Y_TE, _full.predict_proba(TEST[COLS])[:, 1]))
_sc = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression())]).fit(TRAIN[COLS], Y_TR)
N_ITER_SCALED = int(_sc.named_steps["logit"].n_iter_[0])

# E: scoring
ACC_TE = float(accuracy_score(Y_TE, PRED))
CM = confusion_matrix(Y_TE, PRED)
TN, FP, FN, TP = int(CM[0, 0]), int(CM[0, 1]), int(CM[1, 0]), int(CM[1, 1])
PREC, REC = float(precision_score(Y_TE, PRED)), float(recall_score(Y_TE, PRED))
ACC_BY_THR = {t: round(float(accuracy_score(Y_TE, (P_TEST >= t).astype(int))), 3) for t in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]}
ACC_BEST_THR = max(ACC_BY_THR, key=ACC_BY_THR.get)
CM_MAJ = confusion_matrix(Y_TE, np.zeros(N_TEST, dtype=int))

# F: thresholds and ROC
ROSE = Y_TE.values == 1
def rates(t):
    pred = P_TEST >= t
    return float((pred & ROSE).sum() / ROSE.sum()), float((pred & ~ROSE).sum() / (~ROSE).sum())
THREE = {t: rates(t) for t in [0.4, 0.5, 0.6]}
_fpr, _tpr, _thr = roc_curve(Y_TE, P_TEST)
N_THR = len(_thr)
AUC_MAJ = float(roc_auc_score(Y_TE, np.zeros(N_TEST)))
_p_train = MODEL.predict_proba(TRAIN[["vol_20d"]])[:, 1]
_grid_t = np.linspace(0.3, 0.7, 41)
_acc_tr = [float(accuracy_score(Y_TR, (_p_train >= t).astype(int))) for t in _grid_t]
BEST_T_TRAIN = float(_grid_t[int(np.argmax(_acc_tr))])
BEST_T_TRAIN_ACC = max(_acc_tr)
ACC_AT_BEST_T = float(accuracy_score(Y_TE, (P_TEST >= BEST_T_TRAIN).astype(int)))
_pr, _pc = P_TEST[ROSE], P_TEST[~ROSE]
_wins = 0
for _v in _pr:
    _wins += int((_v > _pc).sum())
AUC_PAIRS = _wins / (len(_pr) * len(_pc))
N_PAIRS = len(_pr) * len(_pc)

# G: cross-validation and C
CV_ONE = cv_auc(LogisticRegression(), ["vol_20d"])
CV_ONE_ACC = cross_val_score(LogisticRegression(), TRAIN[["vol_20d"]], Y_TR, cv=TS5, scoring="accuracy")
def pipe(C=1.0, **kw):
    return Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(C=C, max_iter=1000, **kw))])
CV_WIDE = cv_auc(pipe(), COLS)
CGRID = [0.0001, 0.001, 0.01, 0.1, 1, 10]
CV_BY_C = {c: round(float(cv_auc(pipe(c), COLS).mean()), 4) for c in CGRID}
CV_BEST_C = max(CV_BY_C, key=CV_BY_C.get)
GS = GridSearchCV(pipe(), {"logit__C": CGRID}, cv=TS5, scoring="roc_auc").fit(TRAIN[COLS], Y_TR)
GS_C, GS_SCORE = GS.best_params_["logit__C"], float(GS.best_score_)
GS_TE = float(roc_auc_score(Y_TE, GS.predict_proba(TEST[COLS])[:, 1]))
GS_ACC = GridSearchCV(pipe(), {"logit__C": CGRID}, cv=TS5, scoring="accuracy").fit(TRAIN[COLS], Y_TR)
GS_ACC_C, GS_ACC_SCORE = GS_ACC.best_params_["logit__C"], float(GS_ACC.best_score_)
FINE = np.logspace(-5, 2, 15)
CV_FINE = {float(c): float(cv_auc(pipe(c), COLS).mean()) for c in FINE}
SUMSQ = {c: float((pipe(c).fit(TRAIN[COLS], Y_TR).named_steps["logit"].coef_ ** 2).sum()) for c in [100, 1, 0.01, 0.0001]}
TEST_BY_C = {c: round(float(roc_auc_score(Y_TE, pipe(c).fit(TRAIN[COLS], Y_TR).predict_proba(TEST[COLS])[:, 1])), 4) for c in CGRID}
TEST_BEST_C = max(TEST_BY_C, key=TEST_BY_C.get)
_C, _prev, WHILE_TRACE = 100.0, None, []
while _C >= 0.000001:
    _a = float(cv_auc(pipe(_C), COLS).mean())
    WHILE_TRACE.append((_C, round(_a, 4)))
    if _prev is not None and _a < _prev - 0.002:
        break
    _prev = _a
    _C = _C / 10
WHILE_BEST = max(WHILE_TRACE, key=lambda x: x[1])[0]

# H: l1 and arguments
def l1pipe(C):
    return Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(penalty="l1", solver="liblinear", C=C))])
L1 = l1pipe(0.01).fit(TRAIN[COLS], Y_TR)
L1_KEPT = [c for c, b in zip(COLS, L1.named_steps["logit"].coef_[0]) if b != 0]
L1_TE = float(roc_auc_score(Y_TE, L1.predict_proba(TEST[COLS])[:, 1]))
SURV = {c: int((l1pipe(c).fit(TRAIN[COLS], Y_TR).named_steps["logit"].coef_[0] != 0).sum()) for c in [0.001, 0.003, 0.01, 0.03, 0.1, 1]}
DEFAULTS = LogisticRegression().get_params()

# I: reading the result
SD_COEF = float(_sc.named_steps["logit"].coef_[0, COLS.index("vol_20d")])
_one_sc = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression())]).fit(TRAIN[["vol_20d"]], Y_TR)
ONE_SD_COEF = float(_one_sc.named_steps["logit"].coef_[0, 0])
ONE_SD_OR = float(np.exp(ONE_SD_COEF))
RANKED = {"one column": AUC_TE, "19 columns, C = 1": TEST_BY_C[1], "19 columns, C chosen": GS_TE, "19 columns, l1": L1_TE}
RANKED_ORDER = sorted(RANKED, key=RANKED.get, reverse=True)
CV_ONE_PIPE = float(cv_auc(Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression())]), ["vol_20d"]).mean())
_lowest = int(np.argmin(np.where(ROSE, P_TEST, 9)))
WORST_DAY, WORST_P, WORST_VOL = TEST.index[_lowest].date(), float(P_TEST[_lowest]), float(TEST["vol_20d"].iloc[_lowest])
_highest = int(np.argmax(np.where(~ROSE, P_TEST, -1)))
CONF_DAY, CONF_P = TEST.index[_highest].date(), float(P_TEST[_highest])

# J: across the desk
_nv = build_table("NVDA")
_nv_tr, _nv_te = _nv.loc[:"2022-12-31"], _nv.loc["2023-01-01":]
_nv_m = LogisticRegression().fit(_nv_tr[["vol_20d"]], _nv_tr["rising"])
NVDA_AUC = float(roc_auc_score(_nv_te["rising"], _nv_m.predict_proba(_nv_te[["vol_20d"]])[:, 1]))
NVDA_ACC = float(accuracy_score(_nv_te["rising"], _nv_m.predict(_nv_te[["vol_20d"]])))
NVDA_MAJ = float(max(_nv_te["rising"].mean(), 1 - _nv_te["rising"].mean()))
NVDA_CROSS = float(-_nv_m.intercept_[0] / _nv_m.coef_[0, 0])
DESK_AUC, DESK_C = {}, {}
for _t in RET.columns:
    _tb = build_table(_t)
    _tr, _te = _tb.loc[:"2022-12-31"], _tb.loc["2023-01-01":]
    _cc = list(_tb.columns[:-2])
    _m = LogisticRegression().fit(_tr[["vol_20d"]], _tr["rising"])
    DESK_AUC[_t] = round(float(roc_auc_score(_te["rising"], _m.predict_proba(_te[["vol_20d"]])[:, 1])), 3)
    _g = GridSearchCV(pipe(), {"logit__C": CGRID}, cv=TS5, scoring="roc_auc").fit(_tr[_cc], _tr["rising"])
    DESK_C[_t] = _g.best_params_["logit__C"]
DESK_BEST, DESK_WORST = max(DESK_AUC, key=DESK_AUC.get), min(DESK_AUC, key=DESK_AUC.get)
N_PICK_SMALLEST = sum(1 for t in DESK_C if DESK_C[t] == 0.0001)
C_OTHERS = {t: c for t, c in DESK_C.items() if c != 0.0001}


# K: small standalone cases
_last = 20
_p_last = P_TEST[-_last:]
_y_last = Y_TE.values[-_last:]
K1_HITS = int(((_p_last >= 0.5).astype(int) == _y_last).sum())
K2 = {}
for _t in RET.columns:
    _now = RET[_t].rolling(20).std()
    _nxt = _now.shift(-20)
    _both = pd.DataFrame({"now": _now, "next": _nxt}).dropna()
    K2[_t] = round(float((_both["next"] > _both["now"]).loc["2024-01-01":"2024-12-31"].mean()), 3)
K2_TOP = sorted(K2, key=K2.get, reverse=True)[:3]
def _report(ticker):
    _now = RET[ticker].rolling(20).std()
    _nxt = _now.shift(-20)
    _frame = pd.DataFrame({"vol_20d": _now, "vol_next": _nxt}).dropna()
    _frame["rising"] = (_frame["vol_next"] > _frame["vol_20d"]).astype(int)
    _tr, _te = _frame.loc[:"2022-12-31"], _frame.loc["2023-01-01":]
    _m = LogisticRegression().fit(_tr[["vol_20d"]], _tr["rising"])
    _auc = roc_auc_score(_te["rising"], _m.predict_proba(_te[["vol_20d"]])[:, 1])
    _acc = accuracy_score(_te["rising"], _m.predict(_te[["vol_20d"]]))
    _maj = max(_te["rising"].mean(), 1 - _te["rising"].mean())
    return float(_auc), float(_acc), float(_maj)
K3 = {t: _report(t) for t in ["KO", "JPM", "DIS"]}
_streak, K4_LONGEST, K4_END = 0, 0, None
for _d, _v in zip(TEST.index, PRED):
    if _v == 1:
        _streak += 1
        if _streak > K4_LONGEST:
            K4_LONGEST, K4_END = _streak, _d.date()
    else:
        _streak = 0
_right = pd.Series(PRED == Y_TE.values, index=TEST.index)
K5 = _right.groupby(_right.index.year).mean().round(3).to_dict()

# ---------------------------------------------------------------- top matter
md(
"# \U0001f4d3 Session 8 · Exercises\n"
"### Classification with logistic regression\n\n"
"The lecture gave the 19-column table a label as its target: 1 when the next "
"20 days were more volatile than the last 20, 0 otherwise. A straight line "
"fitted to that label gave probabilities below zero, the logistic curve did "
"not, and the curve's probabilities were turned into predictions, scored with "
"the four counts and the AUC, and improved with a penalty whose strength is "
"called `C`.\n\n"
"These exercises do the same on the same table, one piece at a time, and then "
"on every instrument in the data."
)

md(
"## How to use this notebook\n\n"
"- Run the **setup cell** below first. It loads the price data and the "
"lecture's table, makes the label, and imports the scikit-learn pieces.\n"
"- Each exercise has a **task**, then a **code cell** for your work. Cells with "
"`...` are blanks to fill in. Replace them with real code.\n"
"- Stuck? Open the **\U0001f4a1 Hint**, but only after a genuine attempt. Open the "
"**✅ Solution** to *check* yourself, not to skip the thinking.\n"
"- Every cell runs cleanly even with the blanks still in place, so pressing "
"**Run all** never floods you with errors.\n"
"- Most exercises stand alone. A few short runs build on each other (B4 to "
"B7, C1 to C3, E2 to E4, F1 to F2, G5 to G6); the task says which earlier "
"exercise it continues from. If one defeats you, open its solution, run it, "
"and carry on. Section J uses the function from J1, and section K is five "
"small cases that each start from scratch.\n\n"
"**You are not expected to finish all of these.** Do what you can, and come back "
"to the rest when you revise. Short on time? Read the hint, then the solution. A "
"worked solution you genuinely understand is real learning too.\n\n"
"**Returns are in percent here**, exactly as in the lecture. The label has no "
"units at all: it is 1 or 0."
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
"assumes everything from Sessions 1 to 7, so it is a bigger piece of work than a "
"three-star task in an earlier notebook.\n\n"
"Some exercises also carry a **revisits** tag. Those need something from an "
"earlier session as well as today's material, and they are there on purpose: "
"the skills are meant to accumulate."
)

md(
"## \U0001f9f0 Your toolkit for today\n\n"
"Everything from Sessions 1 to 7 still applies. This card holds what Session 8 "
"added.\n\n"
"> Names in brackets (`frame`, `columns`, `model`, ...) are **placeholders**: put "
"your own variable there. **Hover any tool** to see what it does."
)

md(
'<p style="line-height:2.1"><strong>The label and the model</strong><br>\n'
'<code style="cursor:help" title="A label from a comparison: True and False become 1 and 0.">(frame[\'a\'] > frame[\'b\']).astype(int)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Fits the log-odds of the label as a straight line in the columns. Create, then .fit(X, y), as for LinearRegression. A ridge penalty of strength C=1 is on by default.">LogisticRegression()</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="intercept_ holds one number in an array; coef_ has one ROW per class boundary, so coef_[0, 0] is the first slope.">model.intercept_[0]  ·  model.coef_[0, 0]</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The classes in the order predict_proba uses for its columns: 0 then 1.">model.classes_</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="A coefficient adds to the log-odds; its exponential multiplies the odds.">np.exp(model.coef_)</code></p>\n\n'
'<p style="line-height:2.1"><strong>Probabilities and predictions</strong><br>\n'
'<code style="cursor:help" title="One row per observation, one column per class, each row summing to one. Column 1 is the probability of a 1.">model.predict_proba(X)[:, 1]</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The 0/1 predictions: 1 where the probability of a 1 is at least one half.">model.predict(X)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Predictions at any other threshold: a comparison on the probability column.">(p >= 0.6).astype(int)</code></p>\n\n'
'<p style="line-height:2.1"><strong>How it is fitted</strong><br>\n'
'<code style="cursor:help" title="The objective, averaged over the rows: minus the log of the probability given to what happened. Takes the true labels and the probabilities.">log_loss(y, p)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="How many steps the solver may take (default 100) and how many it took.">LogisticRegression(max_iter=1000)  ·  model.n_iter_</code></p>\n\n'
'<p style="line-height:2.1"><strong>Scoring a classifier</strong><br>\n'
'<code style="cursor:help" title="The share of predictions that match the label. True labels first, predictions second, like every metric.">accuracy_score(y, predicted)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="A 2 by 2 array: rows are what happened, columns are the prediction, both in the order 0 then 1.">confusion_matrix(y, predicted)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Precision: the share of predicted 1s that were 1. Recall: the share of real 1s that were predicted.">precision_score(y, predicted)  ·  recall_score(y, predicted)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The false positive rate and recall at every threshold, plus the thresholds. Pass the PROBABILITIES.">roc_curve(y, p)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The area under the ROC curve: 0.5 for a random ranking, 1 for a perfect one. Pass the PROBABILITIES, never the 0/1 predictions.">roc_auc_score(y, p)</code></p>\n\n'
'<p style="line-height:2.1"><strong>Choosing, and the penalty</strong><br>\n'
'<code style="cursor:help" title="The same folds as for regression with a classification score. Larger is better, so no minus sign. Also \'accuracy\'.">cross_val_score(model, X, y, cv=folds, scoring=\'roc_auc\')</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The strength of the penalty, upside down: C = 1/alpha, so a small C is a strong penalty. Standardise in a pipeline and search a grid in factors of ten.">LogisticRegression(C=0.01)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The lasso\'s charge on absolute values, which sets coefficients to exactly zero. Needs a solver that can handle it.">LogisticRegression(penalty=\'l1\', solver=\'liblinear\', C=0.01)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The grid key is the step name, two underscores, the argument.">{\'logit__C\': [0.0001, 0.001, 0.01, 0.1, 1, 10]}</code></p>'
)

md(
"**Formulas you will reach for**\n\n"
r"| what | formula |" "\n"
r"|:--|:--|" "\n"
r"| The sigmoid | $$\sigma(z)=\dfrac{1}{1+e^{-z}}$$ |" "\n"
r"| Logistic regression | $$P(y=1\mid x)=\sigma(\beta_0+\beta_1 x)$$ |" "\n"
r"| Log-odds | $$\log\dfrac{P}{1-P}=\beta_0+\beta_1 x$$ |" "\n"
r"| Log-loss | $$-\dfrac{1}{n}\sum_i \big[y_i\log \hat p_i+(1-y_i)\log(1-\hat p_i)\big]$$ |" "\n"
r"| Accuracy | $$\dfrac{TP+TN}{TP+TN+FP+FN}$$ |" "\n"
r"| Precision, recall | $$\dfrac{TP}{TP+FP},\qquad \dfrac{TP}{TP+FN}$$ |" "\n"
r"| False positive rate | $$\dfrac{FP}{FP+TN}$$ |" "\n"
r"| The objective with the penalty | $$C\cdot\text{log-loss}+\tfrac{1}{2}\sum_j\beta_j^2$$ |" "\n"
)

md("---")

# ---------------------------------------------------------------- setup cell
md(
"## ⚙️ Setup: run this first\n\n"
"This loads the price data and the lecture's 19-column table, makes the label, "
"splits by date, and imports the scikit-learn pieces. If you are in Google "
"Colab it downloads the data by itself."
)

code(
'''import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                             roc_curve, roc_auc_score, log_loss)
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


# Eleven instruments, 2015 to 2024. Returns in PERCENT, as in the lecture.
prices = load_csv("prices.csv", parse_dates=["date"])
wide = prices.pivot(index="date", columns="ticker", values="close")
rets = wide.pct_change().dropna() * 100

# The lecture's table: 19 columns looking back, vol_next looking forward
table = load_csv("market_features.csv", parse_dates=["date"]).set_index("date")
columns = list(table.columns[:-1])

# The lecture's label: 1 when the next 20 days were more volatile than the last 20
table["rising"] = (table["vol_next"] > table["vol_20d"]).astype(int)

train = table.loc[:"2022-12-31"]
test = table.loc["2023-01-01":]

folds = TimeSeriesSplit(n_splits=5)

print("table:", table.shape, "rows x columns")
print("train:", len(train), " test:", len(test), " feature columns:", len(columns))
print("share of days with a rise, train:", round(train["rising"].mean(), 3), " test:", round(test["rising"].mean(), 3))'''
)

md("---")

# ============================================================ A
section(
"## \U0001f3f7️ A · Making labels\n\n"
"A label is a column of ones and zeros that you make from a comparison. "
"These exercises make the lecture's label and a few others, and find the "
"share a model has to beat for each."
)

ex("A1", "The lecture's label, by hand", 1,
   "Make the label yourself as `own`: 1 where `vol_next` is larger than "
   "`vol_20d`, 0 otherwise. Then check it agrees with the setup's `table['rising']` "
   "on every row.",
   "own = ...\nprint(own)\nprint('same on every row:', ...)",
   ["A comparison of two columns gives `True` and `False`; `.astype(int)` turns "
    "them into 1 and 0.",
    "`(own == table['rising']).all()` is `True` only when every row agrees."],
   "own = (table['vol_next'] > table['vol_20d']).astype(int)\nprint(own.tail(3))\n"
   "print('same on every row:', (own == table['rising']).all())",
   "`True`. The comparison is asked of every row at once, and `.all()` is the "
   "vectorised way to ask whether a whole column of checks passed.")

ex("A2", "The share to beat", 2,
   "Print the share of days with a rise in `train` and in `test`. Then print the "
   "accuracy of predicting 0 on every test day.",
   "print('train:', ...)\nprint('test :', ...)\nprint('predict 0 everywhere:', ...)",
   ["The mean of a column of ones and zeros is the share of ones.",
    "Predicting 0 is right on every day whose label is 0, so its accuracy is "
    "`1 - test['rising'].mean()`."],
   "print('train:', round(train['rising'].mean(), 4))\nprint('test :', round(test['rising'].mean(), 4))\n"
   "print('predict 0 everywhere:', round(1 - test['rising'].mean(), 4))",
   f"{SHARE_TR:.3f} and {SHARE_TE:.3f}, so predicting 0 on every test day is "
   f"right {MAJ_TE:.1%} of the time. That is the majority rule, and every model "
   "in this notebook is measured against it.")

ex("A3", "A label with a fixed cut", 2,
   "Make a second label, `busy`: 1 where `vol_next` is above 1.0 (a daily "
   "volatility of one percent). Print its share in the training and test rows, "
   "and the majority rule's accuracy on the test rows.",
   "busy = ...\nbusy_train = ...\nbusy_test = ...\n\nprint(..., ...)\nprint('majority rule:', ...)",
   ["Compare a column with a number this time: `table['vol_next'] > 1.0`.",
    "Slice the label by date, `busy.loc[:'2022-12-31']`, exactly as the setup "
    "sliced the table. The majority rule's accuracy is the larger of the share "
    "and one minus the share."],
   "busy = (table['vol_next'] > 1.0).astype(int)\nbusy_train = busy.loc[:'2022-12-31']\n"
   "busy_test = busy.loc['2023-01-01':]\n\n"
   "print(round(busy_train.mean(), 3), round(busy_test.mean(), 3))\n"
   "print('majority rule:', round(max(busy_test.mean(), 1 - busy_test.mean()), 3))",
   f"{BUSY_TR:.3f} in the training rows and {BUSY_TE:.3f} in the test rows: "
   f"2023 and 2024 were calm, so the majority rule scores {BUSY_MAJ_TE:.1%} on "
   "this label with no model at all. A fixed cut makes an unbalanced label, and "
   "the share to beat depends on which years are being scored.")

ex("A4", "A rise with a margin", 2,
   "Make `rising_20`: 1 where `vol_next` is more than 1.2 times `vol_20d`, a "
   "rise of at least 20 percent. Print its share in the training rows next to "
   "the share of the plain `rising` label.",
   "rising_20 = ...\nprint(..., ...)",
   "Multiply the column before comparing: `table['vol_next'] > 1.2 * table['vol_20d']`.",
   "rising_20 = (table['vol_next'] > 1.2 * table['vol_20d']).astype(int)\n"
   "print(round(rising_20.loc[:'2022-12-31'].mean(), 3), round(train['rising'].mean(), 3))",
   f"{RISE20_TR:.3f} against {SHARE_TR:.3f}. Demanding a margin makes a rise "
   "rarer. The same table can carry many labels, and each is a different "
   "question.")

ex("A5", "A function that makes the label", 3,
   "Write `make_label(frame, margin)`: it returns a Series of ones and zeros that "
   "is 1 where `vol_next` is more than `margin` times `vol_20d`. Check that "
   "`make_label(table, 1.0)` agrees with `table['rising']` everywhere.",
   "def make_label(frame, margin):\n    ...\n\ncheck = make_label(table, 1.0)\nprint('agrees:', ...)",
   ["The body is A4's line with `margin` in place of `1.2` and `frame` in place "
    "of `table`, then `return`.",
    "`(check == table['rising']).all()`."],
   "def make_label(frame, margin):\n"
   "    return (frame['vol_next'] > margin * frame['vol_20d']).astype(int)\n\n"
   "check = make_label(table, 1.0)\nprint('agrees:', (check == table['rising']).all())",
   "`True`. A margin of 1.0 is the plain comparison. The function is one line, "
   "and it is what the next exercise loops over.")

ex("A6", "How the share falls with the margin", 3,
   "Using `make_label` from A5, loop over the margins `[1.0, 1.1, 1.2, 1.5, 2.0]` "
   "and store the share of ones in the **training** rows in a dictionary "
   "`share_by_margin`. Print it.",
   "share_by_margin = {}\n\nfor margin in [1.0, 1.1, 1.2, 1.5, 2.0]:\n    ...\n\nprint(share_by_margin)",
   "Inside the loop: `share_by_margin[margin] = round(float(make_label(train, margin).mean()), 3)`. "
   "Calling the function on `train` gives the training share directly; `float()` "
   "makes it print as a plain number.",
   "share_by_margin = {}\n\nfor margin in [1.0, 1.1, 1.2, 1.5, 2.0]:\n"
   "    share_by_margin[margin] = round(float(make_label(train, margin).mean()), 3)\n\nprint(share_by_margin)",
   f"From {SHARE_BY_MARGIN[1.0]} at a margin of 1.0 down to {SHARE_BY_MARGIN[2.0]} "
   "at 2.0: a doubling of volatility in a month happens on one training day in "
   f"{round(1 / SHARE_BY_MARGIN[2.0])}. A loop over a function into a dictionary "
   "is the shape of every search in this course.")

# ============================================================ B
section(
"## \U0001f4c8 B · The straight line, and the curve\n\n"
"Why a regression cannot give a probability, and the curve that can. B4 to "
"B7 share the model B4 fits."
)

ex("B1", "A regression on the label", 1,
   "Fit `LinearRegression` on `vol_20d` with `rising` as the target, and print "
   "the intercept and the slope.",
   "line = LinearRegression()\n...\nprint(..., ...)",
   "`.fit(train[['vol_20d']], train['rising'])`. The target can be a column of "
   "ones and zeros; nothing stops it.",
   "line = LinearRegression()\nline.fit(train[['vol_20d']], train['rising'])\n"
   "print(line.intercept_, line.coef_)",
   f"An intercept of {LINE_B0:.3f} and a slope of {LINE_B1:.3f}. The higher this "
   "month's volatility, the lower the fitted value, which is the direction the "
   "data has: high volatility is followed by lower volatility more often than not.")

ex("B2", "Where the line leaves the band", 2,
   "Compute the fitted values of a regression on `vol_20d` for the training "
   "rows. Print the smallest and the largest, and count how many are below 0 "
   "or above 1.",
   "line = LinearRegression()\nline.fit(train[['vol_20d']], train['rising'])\n"
   "fitted = ...\n\nprint(..., ...)\nprint('outside [0, 1]:', ...)",
   ["`line.predict(train[['vol_20d']])` is an array of fitted values.",
    "Two masks joined with `|`: `((fitted < 0) | (fitted > 1)).sum()`."],
   "line = LinearRegression()\nline.fit(train[['vol_20d']], train['rising'])\n"
   "fitted = line.predict(train[['vol_20d']])\n\n"
   "print(fitted.min(), fitted.max())\nprint('outside [0, 1]:', ((fitted < 0) | (fitted > 1)).sum())",
   f"From {FIT_MIN:.2f} to {FIT_MAX:.2f}, with {N_OUTSIDE} training days below "
   "zero. Read as a probability, a negative number means nothing, and a straight "
   "line has no way to stop at the edge of the band.",
   revisits="S3")

ex("B3", "The sigmoid, as a function", 2,
   "Write `sigmoid(z)` from the formula and print its value at −2, 0 and 2.\n\n"
   "$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$",
   "def sigmoid(z):\n    ...\n\nprint(sigmoid(-2), sigmoid(0), sigmoid(2))",
   "`np.exp(-z)` is $e^{-z}$. The function works on a single number and on a "
   "whole array alike.",
   "def sigmoid(z):\n    return 1 / (1 + np.exp(-z))\n\nprint(sigmoid(-2), sigmoid(0), sigmoid(2))",
   "0.119, 0.5 and 0.881. Any number in, a number between 0 and 1 out, and "
   "one half exactly at zero. Keep this function: B6 uses it.",
   revisits="S2")

ex("B4", "Logistic regression in three lines", 1,
   "Fit `LogisticRegression` on `vol_20d` with `rising` as the target, and print "
   "the intercept and the coefficient.",
   "model = LogisticRegression()\n...\nprint(..., ...)",
   "Create it, then `.fit(train[['vol_20d']], train['rising'])`. `coef_` has "
   "two pairs of brackets: one row of coefficients per boundary between classes.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "print(model.intercept_, model.coef_)",
   f"An intercept of {B0:.3f} and a slope of {B1:.3f}, on the log-odds scale. "
   "The slope is negative, as the regression's was, but these two numbers now "
   "describe a curve that stays between 0 and 1.")

ex("B5", "Where the curve crosses one half", 2,
   "Using `model` from B4, compute the volatility at which the probability of a "
   "rise is exactly one half, and print it. Then check by predicting the "
   "probability at that volatility.\n\n"
   "$$\\beta_0 + \\beta_1 x = 0 \\quad\\Longrightarrow\\quad x = -\\beta_0 / \\beta_1$$",
   "crossing = ...\nprint(crossing)\n\ncheck = ...\nprint(check)",
   ["`-model.intercept_[0] / model.coef_[0, 0]`: `[0]` takes the number out of "
    "the intercept array, `[0, 0]` takes the first entry of the first row of `coef_`.",
    "`predict_proba` wants a table with the same column name, so build a "
    "one-row DataFrame around the number: "
    "`model.predict_proba(pd.DataFrame({'vol_20d': [crossing]}))`."],
   "crossing = -model.intercept_[0] / model.coef_[0, 0]\nprint(crossing)\n\n"
   "check = model.predict_proba(pd.DataFrame({'vol_20d': [crossing]}))\nprint(check)",
   f"{CROSS:.3f} percent, and the check returns `[[0.5, 0.5]]`. Below that "
   "volatility the model predicts a rise, above it no rise.")

ex("B6", "The curve by hand, and by the model", 3,
   "For the volatilities 0.5, 1.0 and 2.0, compute the probability of a rise "
   "two ways: with your `sigmoid` from B3 applied to the score "
   "`intercept + coefficient * x`, and with `model.predict_proba` from B4. Print "
   "both, and check they agree to six decimals.",
   "vols = np.array([0.5, 1.0, 2.0])\n\nby_hand = ...\nby_model = ...\n\n"
   "print(by_hand)\nprint(by_model)\nprint('agree:', ...)",
   ["The score is `model.intercept_[0] + model.coef_[0, 0] * vols`, an array; "
    "`sigmoid` of it is an array of probabilities.",
    "`model.predict_proba(pd.DataFrame({'vol_20d': vols}))[:, 1]` is the "
    "second column. `np.abs(by_hand - by_model).max() < 1e-6` is the check."],
   "vols = np.array([0.5, 1.0, 2.0])\n\n"
   "by_hand = sigmoid(model.intercept_[0] + model.coef_[0, 0] * vols)\n"
   "by_model = model.predict_proba(pd.DataFrame({'vol_20d': vols}))[:, 1]\n\n"
   "print(by_hand)\nprint(by_model)\nprint('agree:', np.abs(by_hand - by_model).max() < 1e-6)",
   f"{HAND_P[0]:.3f}, {HAND_P[1]:.3f} and {HAND_P[2]:.3f}, and `True`. "
   "`predict_proba` is the sigmoid of the linear score and nothing more; the "
   "two numbers from B4 are the whole model.",
   revisits="S2")

ex("B7", "The odds, in a sentence", 2,
   "Compute the factor by which one percentage point more volatility multiplies "
   "the odds of a rise, from `model` in B4, and print one sentence with an "
   "f-string that states it to two decimals.",
   "factor = ...\nsentence = ...\nprint(sentence)",
   "`np.exp(model.coef_[0, 0])`. In the f-string, `{factor:.2f}` rounds.",
   "factor = np.exp(model.coef_[0, 0])\n"
   "sentence = f'One percentage point more volatility this month multiplies the odds of a rise next month by {factor:.2f}.'\n"
   "print(sentence)",
   f"The factor is {ODDS_RATIO:.2f}: the odds fall to about a quarter. A "
   "coefficient adds on the log-odds scale, so its exponential multiplies the "
   "odds, and that factor is the number to quote.",
   revisits="S1")

ex("B8", "The curve over the data", 4,
   "Draw the fitted curve from B4 over the data it was fitted to. For each band "
   "of `vol_20d` in `bands`, compute the share of training days with a rise and "
   "the band's mean volatility with a mask; plot those as dots, and the curve "
   "as a line over `np.linspace(0.05, 4, 200)`.",
   "bands = [(0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.3), (1.3, 1.7), (1.7, 2.5), (2.5, 6)]\n"
   "xs, shares = [], []\nfor low, high in bands:\n    ...\n\n"
   "grid = np.linspace(0.05, 4, 200)\ncurve = ...\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3.2))\n...\nplt.show()",
   ["The mask is `(train['vol_20d'] >= low) & (train['vol_20d'] < high)`; append "
    "`train['vol_20d'][mask].mean()` to `xs` and `train['rising'][mask].mean()` "
    "to `shares`.",
    "`curve = model.predict_proba(pd.DataFrame({'vol_20d': grid}))[:, 1]`, then "
    "`ax.scatter(xs, shares)` and `ax.plot(grid, curve)`."],
   "bands = [(0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.3), (1.3, 1.7), (1.7, 2.5), (2.5, 6)]\n"
   "xs, shares = [], []\nfor low, high in bands:\n"
   "    mask = (train['vol_20d'] >= low) & (train['vol_20d'] < high)\n"
   "    xs.append(train['vol_20d'][mask].mean())\n    shares.append(train['rising'][mask].mean())\n\n"
   "grid = np.linspace(0.05, 4, 200)\ncurve = model.predict_proba(pd.DataFrame({'vol_20d': grid}))[:, 1]\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3.2))\n"
   "ax.scatter(xs, shares, label='share of days with a rise, per band')\n"
   "ax.plot(grid, curve, label='logistic regression')\n"
   "ax.set_xlabel('volatility over the last 20 days (%)')\nax.set_ylabel('probability of a rise')\n"
   "ax.legend()\nplt.show()",
   f"The dots fall from {BAND_SHARE[0][1]:.2f} in the calmest band to "
   f"{BAND_SHARE[-2][1]:.2f} above 1.7 percent, and the curve runs through them. "
   "The band shares are the data's own answer to the question the model is "
   "asked, which is why they are the right thing to draw the curve against.",
   revisits="S3")

# ============================================================ C
section(
"## \U0001f3b2 C · Probabilities and predictions\n\n"
"What `predict_proba` returns, how `predict` reads it, and the mistake that "
"costs the most. C1 to C3 share the model C1 fits."
)

ex("C1", "Two columns per day", 1,
   "Fit the one-column logistic regression as `model` and call `predict_proba` on "
   "the test rows. Print `classes_`, the shape of the result, and its first "
   "three rows.",
   "model = LogisticRegression()\n...\nprobs = ...\n\nprint(...)\nprint(...)\nprint(...)",
   "`probs = model.predict_proba(test[['vol_20d']])`. `.shape` is a pair of "
   "numbers; `probs[:3]` is the first three rows.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "probs = model.predict_proba(test[['vol_20d']])\n\n"
   "print(model.classes_)\nprint(probs.shape)\nprint(probs[:3])",
   f"`[0 1]`, `({N_TEST}, 2)`, and three rows that each add up to one. The "
   "columns come in the order of `classes_`, so the second column is the "
   "probability of a rise.")

ex("C2", "predict, written out", 2,
   "Take the second column of `probs` from C1 as `p`. Build the 0/1 predictions "
   "two ways, with `model.predict` and with a comparison against 0.5, and write "
   "an assertion that the two arrays are equal on every day.",
   "p = ...\npredicted = ...\nby_hand = ...\n\nassert ...\nprint('share predicted as a rise:', ...)",
   ["`probs[:, 1]` is every row of the second column.",
    "`(p >= 0.5).astype(int)` for the comparison; `(predicted == by_hand).all()` "
    "in the assertion; `predicted.mean()` is the share of ones."],
   "p = probs[:, 1]\npredicted = model.predict(test[['vol_20d']])\nby_hand = (p >= 0.5).astype(int)\n\n"
   "assert (predicted == by_hand).all()\nprint('share predicted as a rise:', predicted.mean())",
   f"The assertion passes, and the model predicts a rise on {SHARE_PRED_1:.1%} of "
   "the test days. `.predict()` is a comparison with one half, and once that is "
   "written out the threshold is yours to change.",
   revisits="S3")

ex("C3", "A stricter threshold", 2,
   "Using `p` from C2, count the test days on which the probability of a rise "
   "is at least 0.6, and the days on which it is at least 0.7.",
   "print('at least 0.6:', ...)\nprint('at least 0.7:', ...)",
   "`(p >= 0.6).sum()` counts the days that pass a comparison.",
   "print('at least 0.6:', (p >= 0.6).sum())\nprint('at least 0.7:', (p >= 0.7).sum())",
   f"{N_PRED_60} days at 0.6, and none at 0.7: the model never gives a test day "
   "a probability above 0.67. A threshold above that predicts no rise at all.")

ex("C4", "Fix the bug: the wrong column", 3,
   "The cell below scores the model with `roc_auc_score` and gets a number far "
   "below one half, which would mean a model worse than a random ranking. "
   "Find the mistake, fix it, and print the right AUC.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n\n"
   "wrong = roc_auc_score(test['rising'], model.predict_proba(test[['vol_20d']])[:, 0])\nprint('as written:', wrong)\n\n"
   "right = ...\nprint('fixed     :', right)",
   ["Column 0 of `predict_proba` is the probability of **no** rise. Scoring the "
    "label 1 with the probability of 0 ranks every day backwards.",
    "`[:, 1]` is the column that goes with the label."],
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n\n"
   "wrong = roc_auc_score(test['rising'], model.predict_proba(test[['vol_20d']])[:, 0])\nprint('as written:', wrong)\n\n"
   "right = roc_auc_score(test['rising'], model.predict_proba(test[['vol_20d']])[:, 1])\nprint('fixed     :', right)",
   f"{AUC_WRONG_COL:.3f} as written and {AUC_TE:.3f} fixed, and the two add up "
   "to one: the wrong column is the right ranking turned upside down. An AUC "
   "well below 0.5 is almost always this bug, never a model that is worse than "
   "random.",
   revisits="S1")

# ============================================================ D
section(
"## \U0001f6b6 D · How the curve is fitted\n\n"
"The objective, the steps, and the warning."
)

ex("D1", "Log-loss, of the model and of a guess", 2,
   "Fit the one-column model, and print its log-loss on the training rows next "
   "to the log-loss of giving every training day a probability of one half.",
   "model = LogisticRegression()\n...\np_train = ...\n\n"
   "print('the model  :', ...)\nprint('one half   :', ...)",
   ["`log_loss(train['rising'], p_train)` with the probability column.",
    "A probability of one half for every day is `np.full(len(train), 0.5)`."],
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "p_train = model.predict_proba(train[['vol_20d']])[:, 1]\n\n"
   "print('the model  :', log_loss(train['rising'], p_train))\n"
   "print('one half   :', log_loss(train['rising'], np.full(len(train), 0.5)))",
   f"{LL_MODEL:.4f} against {LL_HALF:.4f}, which is $\\log 2$: the cost of "
   "saying one half about everything. The fitted curve is the pair of numbers "
   "with the lowest log-loss on these rows, and this is how much lower it got.")

ex("D2", "Watch the steps", 2,
   "Fit the one-column model with `max_iter` set to 1, 2, 3 and 10 in a loop. For "
   "each, print the limit, `n_iter_`, the coefficient and the training log-loss.",
   "for n in [1, 2, 3, 10]:\n    steps = LogisticRegression(max_iter=n)\n    ...",
   "After fitting: `steps.n_iter_`, `steps.coef_.round(3)`, and `log_loss` of "
   "`steps.predict_proba(train[['vol_20d']])[:, 1]`.",
   "for n in [1, 2, 3, 10]:\n    steps = LogisticRegression(max_iter=n)\n"
   "    steps.fit(train[['vol_20d']], train['rising'])\n"
   "    p_n = steps.predict_proba(train[['vol_20d']])[:, 1]\n"
   "    print(n, steps.n_iter_, steps.coef_.round(3), round(log_loss(train['rising'], p_n), 5))",
   f"The log-loss falls from {STEPS[0][3]:.4f} after one step to {STEPS[3][3]:.4f}, "
   f"and the run allowed 10 steps stops after {STEPS[3][1]}, because the "
   "coefficients had stopped changing. Each fit with a small limit shows the "
   "solver part of the way down the hill.")

ex("D3", "Log-loss from the formula", 3,
   "Compute the training log-loss of the one-column model yourself, from the "
   "formula, without `log_loss`, and check it against the function.\n\n"
   "$$-\\frac{1}{n}\\sum_i \\big[y_i \\log \\hat p_i + (1 - y_i)\\log(1 - \\hat p_i)\\big]$$",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "p_train = model.predict_proba(train[['vol_20d']])[:, 1]\ny = train['rising'].values\n\n"
   "by_hand = ...\nprint(by_hand)\nprint(log_loss(y, p_train))",
   ["With `y` as ones and zeros, `y * np.log(p_train)` keeps the log of the "
    "probability on the days with a rise and `(1 - y) * np.log(1 - p_train)` on "
    "the others. Add them, take the mean, put a minus in front.",
    "The whole thing is one line of array arithmetic: no loop over days."],
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "p_train = model.predict_proba(train[['vol_20d']])[:, 1]\ny = train['rising'].values\n\n"
   "by_hand = -(y * np.log(p_train) + (1 - y) * np.log(1 - p_train)).mean()\nprint(by_hand)\n"
   "print(log_loss(y, p_train))",
   f"{LL_HAND:.5f} both ways. Multiplying by `y` and by `1 - y` is the switch in "
   "the formula written as arithmetic, and it is vectorised over all "
   f"{N_TRAIN:,} days at once.",
   revisits="S3")

ex("D4", "Provoke the warning, then cure it", 3,
   "Fit `LogisticRegression(max_iter=5)` on all 19 **raw** columns and watch the "
   "`ConvergenceWarning` appear. Then fit again with enough steps for it to go "
   "away, print both test AUCs, and print how many steps the second fit took.",
   "few = LogisticRegression(max_iter=5)\nfew.fit(train[columns], train['rising'])\n"
   "print('5 steps   :', roc_auc_score(test['rising'], few.predict_proba(test[columns])[:, 1]))\n\n"
   "enough = ...\n...\nprint('more steps:', ...)\nprint('steps taken:', ...)",
   "`LogisticRegression(max_iter=1000)` is enough; `enough.n_iter_` says how "
   "many it used.",
   "few = LogisticRegression(max_iter=5)\nfew.fit(train[columns], train['rising'])\n"
   "print('5 steps   :', roc_auc_score(test['rising'], few.predict_proba(test[columns])[:, 1]))\n\n"
   "enough = LogisticRegression(max_iter=1000)\nenough.fit(train[columns], train['rising'])\n"
   "print('more steps:', roc_auc_score(test['rising'], enough.predict_proba(test[columns])[:, 1]))\n"
   "print('steps taken:', enough.n_iter_)",
   f"The first fit prints a warning and scores {AUC_FEW:.3f}; the second is "
   f"silent, takes {N_ITER_FULL} steps and scores {AUC_FULL:.3f}. On raw "
   f"columns the solver needs {N_ITER_FULL} steps; on standardised ones it needs "
   f"{N_ITER_SCALED}, which is the other cure the warning names.",
   revisits="S6")

# ============================================================ E
section(
"## \U0001f9ee E · Scoring with the four counts\n\n"
"Accuracy, the confusion matrix, precision and recall, each first by hand and "
"then with the function. E2 to E4 share the masks E2 builds."
)

ex("E1", "Accuracy, two ways", 2,
   "Fit the one-column model, predict the test rows, and compute the accuracy "
   "as the share of days where the prediction equals the label, then with "
   "`accuracy_score`. Print both next to the majority rule from A2.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\npredicted = ...\n\n"
   "print('by hand   :', ...)\nprint('function  :', ...)\nprint('majority  :', ...)",
   "`(predicted == test['rising']).mean()` and `accuracy_score(test['rising'], predicted)`.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "predicted = model.predict(test[['vol_20d']])\n\n"
   "print('by hand   :', (predicted == test['rising']).mean())\n"
   "print('function  :', accuracy_score(test['rising'], predicted))\n"
   "print('majority  :', 1 - test['rising'].mean())",
   f"{ACC_TE:.4f} both ways, against {MAJ_TE:.4f} for predicting 0 on every day. "
   "Accuracy only means something next to that share.",
   revisits="S3")

ex("E2", "The four counts, with masks", 2,
   "Build two masks, `rose` (the label is 1) and `pred_rise` (the prediction is "
   "1), and count the four kinds of test day: predicted rise and it rose, "
   "predicted rise and it did not, predicted no rise but it rose, predicted no "
   "rise and none came.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\npredicted = model.predict(test[['vol_20d']])\n\n"
   "rose = ...\npred_rise = ...\n\ntp = ...\nfp = ...\nfn = ...\ntn = ...\nprint(tp, fp, fn, tn)",
   ["`rose = test['rising'].values == 1` and `pred_rise = predicted == 1` are "
    "boolean arrays.",
    "`(pred_rise & rose).sum()`, `(pred_rise & ~rose).sum()`, "
    "`(~pred_rise & rose).sum()`, `(~pred_rise & ~rose).sum()`."],
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "predicted = model.predict(test[['vol_20d']])\n\n"
   "rose = test['rising'].values == 1\npred_rise = predicted == 1\n\n"
   "tp = (pred_rise & rose).sum()\nfp = (pred_rise & ~rose).sum()\n"
   "fn = (~pred_rise & rose).sum()\ntn = (~pred_rise & ~rose).sum()\nprint(tp, fp, fn, tn)",
   f"{TP}, {FP}, {FN} and {TN}, which add up to {N_TEST}. The four counts are "
   "the whole story of a classifier at one threshold; every score in this "
   "section is a ratio of some of them.",
   revisits="S4")

ex("E3", "confusion_matrix", 2,
   "Print `confusion_matrix` for the same predictions, and check that its four "
   "entries are the four counts from E2 in the layout the lecture described: "
   "`tn` top left, `tp` bottom right.",
   "cm = ...\nprint(cm)\nprint('matches:', ...)",
   "`confusion_matrix(test['rising'], predicted)`. `cm[0, 0]` is top left and "
   "`cm[1, 1]` bottom right; compare them with `tn` and `tp`.",
   "cm = confusion_matrix(test['rising'], predicted)\nprint(cm)\n"
   "print('matches:', cm[0, 0] == tn and cm[1, 1] == tp and cm[0, 1] == fp and cm[1, 0] == fn)",
   "`True`. Rows are what happened and columns are the prediction, both with 0 "
   "first. Getting that layout wrong swaps precision and recall silently, "
   "which is why the check is worth a line.")

ex("E4", "Precision and recall, two ways", 3,
   "Compute precision and recall from the counts `tp`, `fp` and `fn` of E2, then "
   "with `precision_score` and `recall_score`, and print all four numbers.",
   "print('precision by hand:', ...)\nprint('precision function:', ...)\n"
   "print('recall by hand   :', ...)\nprint('recall function   :', ...)",
   ["Precision is `tp / (tp + fp)`, the share of predicted rises that happened. "
    "Recall is `tp / (tp + fn)`, the share of the rises that were predicted.",
    "The functions take the true labels first and the predictions second."],
   "print('precision by hand:', tp / (tp + fp))\n"
   "print('precision function:', precision_score(test['rising'], predicted))\n"
   "print('recall by hand   :', tp / (tp + fn))\n"
   "print('recall function   :', recall_score(test['rising'], predicted))",
   f"Precision {PREC:.3f} and recall {REC:.3f}, both ways. The model predicts "
   f"a rise on {TP + FP} days and is right on {TP} of them, and it predicts "
   f"{TP} of the {TP + FN} rises that came.",
   revisits="S4")

ex("E5", "A function for the four counts", 3,
   "Write `four_counts(y, predicted)`: it returns a dictionary with the keys "
   "`'tp'`, `'fp'`, `'fn'` and `'tn'`. Run it on the test labels and the "
   "predictions of a one-column model, and print the dictionary.",
   "def four_counts(y, predicted):\n    ...\n\nmodel = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "counts = four_counts(test['rising'].values, model.predict(test[['vol_20d']]))\nprint(counts)",
   ["Inside, build the two masks from the arguments, `y == 1` and "
    "`predicted == 1`, then the four sums as in E2.",
    "Wrap each sum in `int()` so the dictionary prints plain numbers."],
   "def four_counts(y, predicted):\n    rose = y == 1\n    pred_rise = predicted == 1\n"
   "    return {'tp': int((pred_rise & rose).sum()), 'fp': int((pred_rise & ~rose).sum()),\n"
   "            'fn': int((~pred_rise & rose).sum()), 'tn': int((~pred_rise & ~rose).sum())}\n\n"
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "counts = four_counts(test['rising'].values, model.predict(test[['vol_20d']]))\nprint(counts)",
   f"`{{'tp': {TP}, 'fp': {FP}, 'fn': {FN}, 'tn': {TN}}}`. Four lines that any "
   "classifier at any threshold can be handed, and the next exercise hands "
   "them several.",
   revisits="S2")

ex("E6", "Accuracy at six thresholds", 3,
   "Using `four_counts` from E5, loop over the thresholds `[0.4, 0.45, 0.5, "
   "0.55, 0.6, 0.65]`: turn the test probabilities into predictions at each, "
   "compute the accuracy from the four counts, and store it in a dictionary. "
   "Print the dictionary and the best threshold.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\np = model.predict_proba(test[['vol_20d']])[:, 1]\n\n"
   "acc_by_threshold = {}\nfor threshold in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:\n    ...\n\n"
   "print(acc_by_threshold)\nprint('best:', ...)",
   ["`predicted = (p >= threshold).astype(int)`, then `c = four_counts(...)` and "
    "accuracy is `(c['tp'] + c['tn']) / len(test)`.",
    "`max(acc_by_threshold, key=acc_by_threshold.get)` is the key with the "
    "largest value."],
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "p = model.predict_proba(test[['vol_20d']])[:, 1]\n\n"
   "acc_by_threshold = {}\nfor threshold in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:\n"
   "    predicted = (p >= threshold).astype(int)\n    c = four_counts(test['rising'].values, predicted)\n"
   "    acc_by_threshold[threshold] = round((c['tp'] + c['tn']) / len(test), 3)\n\n"
   "print(acc_by_threshold)\nprint('best:', max(acc_by_threshold, key=acc_by_threshold.get))",
   f"Best at {ACC_BEST_THR}, with {ACC_BY_THR[ACC_BEST_THR]:.3f} against "
   f"{ACC_BY_THR[0.5]:.3f} at one half. A threshold is a setting, and here it "
   "was read off the test rows, which is not allowed for choosing; F6 does it "
   "on the training rows.",
   revisits="S2")

ex("E7", "The majority rule's four counts", 3,
   "Score the rule that predicts 0 on every test day with `confusion_matrix`, "
   "`accuracy_score` and `recall_score`. What does its confusion matrix look "
   "like, and why is its recall what it is?",
   "never = np.zeros(len(test), dtype=int)\n\nprint(...)\nprint('accuracy:', ...)\nprint('recall  :', ...)",
   "`np.zeros(len(test), dtype=int)` is a prediction of 0 on every day. The "
   "three functions take it like any other prediction.",
   "never = np.zeros(len(test), dtype=int)\n\nprint(confusion_matrix(test['rising'], never))\n"
   "print('accuracy:', accuracy_score(test['rising'], never))\nprint('recall  :', recall_score(test['rising'], never))",
   f"The right-hand column of the matrix is all zeros: {CM_MAJ[0, 0]} true "
   f"negatives, {CM_MAJ[1, 0]} misses, nothing predicted. Accuracy {MAJ_TE:.3f} "
   "and recall 0: a rule that never predicts a rise cannot catch one. On this "
   "balanced label the model beats it on both; on a rare label accuracy alone "
   "would not show the difference.",
   revisits="S4")

# ============================================================ F
section(
"## \U0001f4c9 F · Thresholds and the ROC curve\n\n"
"Every threshold is one point; all of them are the curve. F1 and F2 share the "
"probabilities F1 computes."
)

ex("F1", "Three thresholds, two shares", 2,
   "Fit the one-column model and take the test probabilities as `p`. For the "
   "thresholds 0.4, 0.5 and 0.6, print the recall and the false positive rate: "
   "the share of days **without** a rise on which a rise was predicted.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "p = model.predict_proba(test[['vol_20d']])[:, 1]\nrose = test['rising'].values == 1\n\n"
   "for threshold in [0.4, 0.5, 0.6]:\n    ...",
   ["`pred = p >= threshold` is a mask. Recall is `(pred & rose).sum() / rose.sum()`.",
    "The false positive rate is `(pred & ~rose).sum() / (~rose).sum()`."],
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "p = model.predict_proba(test[['vol_20d']])[:, 1]\nrose = test['rising'].values == 1\n\n"
   "for threshold in [0.4, 0.5, 0.6]:\n    pred = p >= threshold\n"
   "    recall = (pred & rose).sum() / rose.sum()\n"
   "    false_alarm_rate = (pred & ~rose).sum() / (~rose).sum()\n"
   "    print(threshold, round(recall, 3), round(false_alarm_rate, 3))",
   f"At 0.4, recall {THREE[0.4][0]:.3f} and a false positive rate of "
   f"{THREE[0.4][1]:.3f}; at 0.6, {THREE[0.6][0]:.3f} and {THREE[0.6][1]:.3f}. "
   "Raising the threshold lowers both. Each pair is one point of the ROC curve.")

ex("F2", "The ROC curve, by hand and by the function", 4,
   "Repeat F1 over `np.linspace(0, 1, 101)` and collect the two shares in two "
   "lists. Then call `roc_curve` on `p` and draw both on one figure: your points "
   "as dots, the function's curve as a line, false positive rate across and "
   "recall up.",
   "recalls, false_alarm_rates = [], []\nfor threshold in np.linspace(0, 1, 101):\n    ...\n\n"
   "fpr, tpr, thresholds = ..., ..., ...\n\nfig, ax = plt.subplots(figsize=(5, 4))\n...\nplt.show()",
   ["The loop body is F1's with `.append` in place of `print`.",
    "`fpr, tpr, thresholds = roc_curve(test['rising'], p)` returns three arrays. "
    "`ax.scatter(false_alarm_rates, recalls)` and `ax.plot(fpr, tpr)`."],
   "recalls, false_alarm_rates = [], []\nfor threshold in np.linspace(0, 1, 101):\n"
   "    pred = p >= threshold\n    recalls.append((pred & rose).sum() / rose.sum())\n"
   "    false_alarm_rates.append((pred & ~rose).sum() / (~rose).sum())\n\n"
   "fpr, tpr, thresholds = roc_curve(test['rising'], p)\n\n"
   "fig, ax = plt.subplots(figsize=(5, 4))\n"
   "ax.scatter(false_alarm_rates, recalls, s=12, label='101 thresholds by hand')\n"
   "ax.plot(fpr, tpr, label='roc_curve')\nax.plot([0, 1], [0, 1], linestyle='--', color='grey')\n"
   "ax.set_xlabel('false positive rate')\nax.set_ylabel('recall')\nax.legend()\nplt.show()",
   f"The dots sit on the line. `roc_curve` used {N_THR} thresholds, one at "
   "every probability where a point moves, and the loop used 101 evenly spaced "
   "ones; both draw the same curve, because the curve only depends on how the "
   "probabilities rank the days.",
   revisits="S3")

ex("F3", "The area", 1,
   "Fit the one-column model and print its AUC on the test rows.",
   "model = LogisticRegression()\n...\nprint(...)",
   "`roc_auc_score(test['rising'], model.predict_proba(test[['vol_20d']])[:, 1])`.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "print(roc_auc_score(test['rising'], model.predict_proba(test[['vol_20d']])[:, 1]))",
   f"{AUC_TE:.4f}. The second argument is the probability column, not the "
   "0/1 predictions.")

ex("F4", "The AUC of a rule with no ranking", 2,
   "Compute the AUC of a prediction that gives every test day the same "
   "probability, 0.5, and of one that gives every day a probability of 0. "
   "Explain the two numbers in one sentence.",
   "same = np.full(len(test), 0.5)\nzeros = ...\n\nprint(roc_auc_score(test['rising'], same))\nprint(...)",
   "`np.zeros(len(test))`. When every day has the same score there is no "
   "ranking, and the area is one half whatever the score is.",
   "same = np.full(len(test), 0.5)\nzeros = np.zeros(len(test))\n\n"
   "print(roc_auc_score(test['rising'], same))\nprint(roc_auc_score(test['rising'], zeros))",
   f"{AUC_MAJ:.1f} both times. The AUC scores a ranking, and a rule that ranks "
   "nothing scores one half, which is why one half, not zero, is the floor.")

ex("F5", "Precision and recall against the threshold", 3,
   "For thresholds from 0.3 to 0.7 in steps of 0.01, collect the precision and "
   "the recall of the one-column model's test predictions in two lists, and draw "
   "both against the threshold on one axis with a legend.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\np = model.predict_proba(test[['vol_20d']])[:, 1]\n\n"
   "thresholds = np.arange(0.3, 0.7, 0.01)\nprecisions, recalls = [], []\nfor threshold in thresholds:\n    ...\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3.2))\n...\nplt.show()",
   ["`predicted = (p >= threshold).astype(int)`, then `precision_score(test['rising'], predicted)` "
    "and `recall_score(...)`, each appended.",
    "`ax.plot(thresholds, precisions, label='precision')` and the same for recall."],
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "p = model.predict_proba(test[['vol_20d']])[:, 1]\n\n"
   "thresholds = np.arange(0.3, 0.7, 0.01)\nprecisions, recalls = [], []\nfor threshold in thresholds:\n"
   "    predicted = (p >= threshold).astype(int)\n"
   "    precisions.append(precision_score(test['rising'], predicted))\n"
   "    recalls.append(recall_score(test['rising'], predicted))\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3.2))\nax.plot(thresholds, precisions, label='precision')\n"
   "ax.plot(thresholds, recalls, label='recall')\nax.set_xlabel('threshold')\nax.legend()\nplt.show()",
   "Recall falls from 1 to 0 as the threshold rises, and precision climbs. The "
   "two lines cross near 0.55; where a threshold should sit depends on which "
   "of the two errors costs more, which the next session takes up.",
   revisits="S3")

ex("F6", "Choose the threshold on the training rows", 4,
   "A threshold is a setting, so it is chosen without the test rows. Compute the "
   "one-column model's probabilities on the **training** rows, find the "
   "threshold in `np.linspace(0.3, 0.7, 41)` with the highest training accuracy, "
   "and then print the test accuracy at that threshold.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "p_train = ...\np_test = ...\n\nbest_threshold = None\nbest_accuracy = 0\n"
   "for threshold in np.linspace(0.3, 0.7, 41):\n    ...\n\n"
   "print('chosen on train:', best_threshold, round(best_accuracy, 3))\nprint('test accuracy  :', ...)",
   ["Inside the loop, compute the training accuracy at the threshold and, if it "
    "beats `best_accuracy`, store both.",
    "Round the threshold, `round(threshold, 2)`, when you store it; "
    "`np.linspace` produces numbers like 0.4100000001."],
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "p_train = model.predict_proba(train[['vol_20d']])[:, 1]\np_test = model.predict_proba(test[['vol_20d']])[:, 1]\n\n"
   "best_threshold = None\nbest_accuracy = 0\nfor threshold in np.linspace(0.3, 0.7, 41):\n"
   "    accuracy = accuracy_score(train['rising'], (p_train >= threshold).astype(int))\n"
   "    if accuracy > best_accuracy:\n        best_accuracy = accuracy\n        best_threshold = round(threshold, 2)\n\n"
   "print('chosen on train:', best_threshold, round(best_accuracy, 3))\n"
   "print('test accuracy  :', accuracy_score(test['rising'], (p_test >= best_threshold).astype(int)))",
   f"The training rows choose {BEST_T_TRAIN:.2f}, at a training accuracy of "
   f"{BEST_T_TRAIN_ACC:.3f}, and that threshold scores {ACC_AT_BEST_T:.3f} on "
   f"the test rows. E6 found {ACC_BEST_THR} by looking at the test rows, which "
   "is the difference between choosing and reporting.",
   revisits="S5")

ex("F7", "The AUC as a share of pairs", 5,
   "The lecture read the AUC as a probability: take a day with a rise and a day "
   "without at random, and the model ranks them correctly that often. Compute it "
   "that way. Split the test probabilities into those of the days with a rise "
   "and those without, count the pairs in which the rise has the higher "
   "probability, and divide by the number of pairs. Compare with "
   "`roc_auc_score`.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "p = model.predict_proba(test[['vol_20d']])[:, 1]\nrose = test['rising'].values == 1\n\n"
   "p_rise = ...\np_calm = ...\n\nwins = 0\n...\n\n"
   "print('by pairs:', ...)\nprint('function:', ...)",
   ["`p[rose]` and `p[~rose]` split the probabilities.",
    "Loop `for value in p_rise:`; inside, `(value > p_calm).sum()` counts the "
    "no-rise days that one rise day beats. Add that up over every `value`, then "
    "divide by `len(p_rise) * len(p_calm)`."],
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "p = model.predict_proba(test[['vol_20d']])[:, 1]\nrose = test['rising'].values == 1\n\n"
   "p_rise = p[rose]\np_calm = p[~rose]\n\nwins = 0\nfor value in p_rise:\n"
   "    wins = wins + (value > p_calm).sum()\n\n"
   "print('by pairs:', wins / (len(p_rise) * len(p_calm)))\n"
   "print('function:', roc_auc_score(test['rising'], p))",
   f"{AUC_PAIRS:.4f} by pairs and {AUC_TE:.4f} from the function, over "
   f"{N_PAIRS:,} pairs; the tiny gap is pairs with exactly equal probabilities, "
   "which the function counts as half a win. One loop over the rise days and a "
   "vectorised comparison inside it is enough. The definition and the area are "
   "the same number.",
   revisits="S3")

# ============================================================ G
section(
"## \U0001f504 G · Cross-validation, and C\n\n"
"The folds with a classification score, all 19 columns, and the strength of "
"the penalty. G5 and G6 share the search G5 fits."
)

ex("G1", "AUC on the folds", 1,
   "Cross-validate the one-column `LogisticRegression()` with `folds` and "
   "`scoring='roc_auc'`. Print the five scores and their mean.",
   "scores = ...\nprint(...)\nprint(...)",
   "`cross_val_score(LogisticRegression(), train[['vol_20d']], train['rising'], cv=folds, scoring='roc_auc')`. "
   "Larger is better, so no minus sign.",
   "scores = cross_val_score(LogisticRegression(), train[['vol_20d']], train['rising'],\n"
   "                         cv=folds, scoring='roc_auc')\nprint(scores.round(3))\nprint(scores.mean())",
   f"A mean of {CV_ONE.mean():.3f}, with the folds from {CV_ONE.min():.2f} to "
   f"{CV_ONE.max():.2f}. The same five folds as for regression, read with a "
   "score that has no minus sign to strip.")

ex("G2", "Accuracy on the folds", 2,
   "Run the same cross-validation with `scoring='accuracy'` and print the mean "
   "next to the AUC's from G1. Why can one of them be far below the other?",
   "auc = cross_val_score(LogisticRegression(), train[['vol_20d']], train['rising'], cv=folds, scoring='roc_auc')\n"
   "acc = ...\nprint('accuracy:', ...)\nprint('AUC     :', ...)",
   "Only the scoring string changes. Accuracy needs a threshold, which is one "
   "half inside `cross_val_score`; the AUC does not.",
   "auc = cross_val_score(LogisticRegression(), train[['vol_20d']], train['rising'], cv=folds, scoring='roc_auc')\n"
   "acc = cross_val_score(LogisticRegression(), train[['vol_20d']], train['rising'],\n"
   "                      cv=folds, scoring='accuracy')\nprint('accuracy:', acc.mean())\nprint('AUC     :', auc.mean())",
   f"{CV_ONE_ACC.mean():.3f} against {CV_ONE.mean():.3f}. The AUC scores the "
   "ranking of every fold's days; accuracy scores the predictions at one half, "
   f"and on the first fold it is {CV_ONE_ACC[0]:.3f}, since that fold's calm "
   "years sit almost entirely on one side of the threshold.",
   revisits="S5")

ex("G3", "All 19 columns, in a pipeline", 2,
   "Build a pipeline with a `StandardScaler` step `'scale'` and a "
   "`LogisticRegression(max_iter=1000)` step `'logit'`, cross-validate it on all "
   "`columns` with the AUC, and print the mean next to the one-column mean, "
   "which the first line computes.",
   "one_scores = cross_val_score(LogisticRegression(), train[['vol_20d']], train['rising'], cv=folds, scoring='roc_auc')\n\n"
   "wide_pipe = Pipeline([...])\nwide_scores = ...\n\nprint('19 columns:', ...)\nprint('one column:', ...)",
   "`Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(max_iter=1000))])`, "
   "then `cross_val_score` on `train[columns]`.",
   "one_scores = cross_val_score(LogisticRegression(), train[['vol_20d']], train['rising'], cv=folds, scoring='roc_auc')\n\n"
   "wide_pipe = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(max_iter=1000))])\n"
   "wide_scores = cross_val_score(wide_pipe, train[columns], train['rising'], cv=folds, scoring='roc_auc')\n\n"
   "print('19 columns:', wide_scores.mean())\nprint('one column:', one_scores.mean())",
   f"{CV_WIDE.mean():.3f} against {CV_ONE.mean():.3f}: on the folds, 19 columns "
   "rank the days worse than one. The same overfitting as in regression, with "
   "a different score reporting it.")

ex("G4", "A loop over C", 3,
   "For each `C` in `[0.0001, 0.001, 0.01, 0.1, 1, 10]`, build the pipeline "
   "with that `C`, cross-validate it with the AUC, and store the mean in a "
   "dictionary `cv_by_C`. Print it and the best key.",
   "cv_by_C = {}\n\nfor C in [0.0001, 0.001, 0.01, 0.1, 1, 10]:\n    ...\n\n"
   "print(cv_by_C)\nprint('best:', ...)",
   ["`LogisticRegression(C=C, max_iter=1000)` inside the pipeline; give the "
    "pipeline its own name so `wide_pipe` is kept.",
    "`max(cv_by_C, key=cv_by_C.get)`: the AUC is larger-is-better, so `max`."],
   "cv_by_C = {}\n\nfor C in [0.0001, 0.001, 0.01, 0.1, 1, 10]:\n"
   "    candidate = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(C=C, max_iter=1000))])\n"
   "    candidate_scores = cross_val_score(candidate, train[columns], train['rising'], cv=folds, scoring='roc_auc')\n"
   "    cv_by_C[C] = round(float(candidate_scores.mean()), 4)\n\n"
   "print(cv_by_C)\nprint('best:', max(cv_by_C, key=cv_by_C.get))",
   f"Best at {CV_BEST_C}, with the AUC rising from {CV_BY_C[10]:.3f} at C = 10 "
   f"to {CV_BY_C[CV_BEST_C]:.3f}. A small C is a strong penalty, so the folds "
   "prefer the coefficients pulled hard towards zero. Note `max`, where the "
   "alpha loop used `min`.",
   revisits="S2")

ex("G5", "GridSearchCV over C", 2,
   "Let `GridSearchCV` run G4: the pipeline with `LogisticRegression(max_iter=1000)` "
   "and no `C`, the grid `{'logit__C': [0.0001, 0.001, 0.01, 0.1, 1, 10]}`, "
   "`folds`, and `scoring='roc_auc'`. Fit it as `search` and print the best C "
   "and score.",
   "base = ...\ngrid = ...\n\nsearch = ...\n...\n\nprint(...)\nprint(...)",
   "`GridSearchCV(base, grid, cv=folds, scoring='roc_auc')`, then `.fit`. The "
   "best score is positive this time.",
   "base = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(max_iter=1000))])\n"
   "grid = {'logit__C': [0.0001, 0.001, 0.01, 0.1, 1, 10]}\n\n"
   "search = GridSearchCV(base, grid, cv=folds, scoring='roc_auc')\n"
   "search.fit(train[columns], train['rising'])\n\n"
   "print(search.best_params_)\nprint(search.best_score_)",
   f"C = {GS_C} at {GS_SCORE:.4f}, the numbers from your loop. `'logit__C'` is "
   "the step name, two underscores, then the argument.")

ex("G6", "The test rows, once", 2,
   "Use `search` from G5 to compute the test AUC, and read the chosen `C` back "
   "out of `search.best_estimator_`.",
   "print('test AUC:', ...)\nprint('C       :', ...)",
   ["`search.predict_proba(test[columns])[:, 1]` uses the best pipeline, "
    "refitted on all the training rows.",
    "`search.best_estimator_.named_steps['logit'].C`."],
   "print('test AUC:', roc_auc_score(test['rising'], search.predict_proba(test[columns])[:, 1]))\n"
   "print('C       :', search.best_estimator_.named_steps['logit'].C)",
   f"{GS_TE:.4f} with C = {GS_C}. The search chose C on the folds and refitted "
   "on every training row; the test rows are opened once, here.",
   revisits="S6")

ex("G7", "The same grid, scored by accuracy", 3,
   "Run G5's grid search again with `scoring='accuracy'` and print the best C "
   "and score. Then print the AUC-chosen and accuracy-chosen C side by side.",
   "acc_search = ...\n...\n\nprint(...)\n"
   "print('chosen by AUC     :', ...)\nprint('chosen by accuracy:', ...)",
   "Only the scoring string changes. Read the two winners from the two "
   "`best_params_` dictionaries with the key `'logit__C'`.",
   "acc_search = GridSearchCV(base, grid, cv=folds, scoring='accuracy')\n"
   "acc_search.fit(train[columns], train['rising'])\n\n"
   "print(acc_search.best_params_, acc_search.best_score_)\n"
   "print('chosen by AUC     :', search.best_params_['logit__C'])\n"
   "print('chosen by accuracy:', acc_search.best_params_['logit__C'])",
   f"Accuracy picks C = {GS_ACC_C} at {GS_ACC_SCORE:.3f}; the AUC picked "
   f"{GS_C}. The score is part of the choice, so it is fixed before the search "
   "and matched to what will be reported.")

ex("G8", "The validation curve", 3,
   "Repeat G4 on the finer grid `np.logspace(-5, 2, 15)` and draw the mean AUC "
   "against C with a logarithmic x-axis. Add a horizontal line at the "
   "one-column mean, which the first line computes.",
   "one_scores = cross_val_score(LogisticRegression(), train[['vol_20d']], train['rising'], cv=folds, scoring='roc_auc')\n\n"
   "Cs = np.logspace(-5, 2, 15)\naucs = []\n\nfor C in Cs:\n    ...\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3.2))\n...\nplt.show()",
   ["The loop body is G4's, appending the mean to `aucs`.",
    "`ax.plot(Cs, aucs, marker='o')`, `ax.set_xscale('log')`, and "
    "`ax.axhline(one_scores.mean(), linestyle='--')` for the one-column line."],
   "one_scores = cross_val_score(LogisticRegression(), train[['vol_20d']], train['rising'], cv=folds, scoring='roc_auc')\n\n"
   "Cs = np.logspace(-5, 2, 15)\naucs = []\n\nfor C in Cs:\n"
   "    candidate = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(C=C, max_iter=1000))])\n"
   "    candidate_scores = cross_val_score(candidate, train[columns], train['rising'], cv=folds, scoring='roc_auc')\n"
   "    aucs.append(candidate_scores.mean())\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3.2))\nax.plot(Cs, aucs, marker='o', label='19 columns')\n"
   "ax.axhline(one_scores.mean(), linestyle='--', color='grey', label='one column')\n"
   "ax.set_xscale('log')\nax.set_xlabel('C')\nax.set_ylabel('cross-validated AUC')\nax.legend()\nplt.show()",
   f"The curve rises from about {min(CV_FINE.values()):.3f} on the right to about "
   f"{max(CV_FINE.values()):.3f} on the left and then flattens, and it never "
   "reaches the dashed line. For ridge the curve had a bottom in the middle; "
   "here the strongest penalty is the best, and it is still not enough to beat "
   "one column.",
   revisits="S6")

ex("G9", "How much coefficient is left", 3,
   "For C in `[100, 1, 0.01, 0.0001]`, fit the pipeline on all columns and "
   "compute the sum of the squared coefficients, the quantity the penalty "
   "charges for. Print it for each C, without a loop over the coefficients.",
   "for C in [100, 1, 0.01, 0.0001]:\n    fitted_pipe = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(C=C, max_iter=1000))])\n"
   "    fitted_pipe.fit(train[columns], train['rising'])\n    size = ...\n    print(C, size)",
   "`fitted_pipe.named_steps['logit'].coef_` is an array with one row; square it "
   "and sum it: `(coef_ ** 2).sum()`.",
   "for C in [100, 1, 0.01, 0.0001]:\n"
   "    fitted_pipe = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(C=C, max_iter=1000))])\n"
   "    fitted_pipe.fit(train[columns], train['rising'])\n"
   "    size = (fitted_pipe.named_steps['logit'].coef_ ** 2).sum()\n    print(C, round(size, 6))",
   f"{SUMSQ[100]:.2f} at C = 100, {SUMSQ[1]:.2f} at 1, {SUMSQ[0.01]:.4f} at 0.01 "
   f"and {SUMSQ[0.0001]:.6f} at 0.0001. The penalty term shrinks by a factor of "
   "about a hundred for every factor of a hundred in C, and you can watch it "
   "go, as you did for alpha.",
   revisits="S6")

ex("G10", "The test rows would choose differently", 4,
   "For each C in `[0.0001, 0.001, 0.01, 0.1, 1, 10]`, cross-validate the "
   "pipeline on the training rows **and** fit it and score it on the test rows, "
   "storing the two AUCs in two dictionaries. Print the C the folds pick and the "
   "C the test rows would pick. Which one is allowed to choose?",
   "cv_by_C = {}\ntest_by_C = {}\n\nfor C in [0.0001, 0.001, 0.01, 0.1, 1, 10]:\n    ...\n\n"
   "print(test_by_C)\nprint('the folds pick          :', ...)\nprint('the test rows would pick:', ...)",
   ["Inside the loop, `cross_val_score(...)` on the training rows for the first "
    "dictionary, then `candidate.fit(train[columns], train['rising'])` and "
    "`roc_auc_score` on `candidate.predict_proba(test[columns])[:, 1]` for the second.",
    "`max(cv_by_C, key=cv_by_C.get)` and `max(test_by_C, key=test_by_C.get)`."],
   "cv_by_C = {}\ntest_by_C = {}\n\nfor C in [0.0001, 0.001, 0.01, 0.1, 1, 10]:\n"
   "    candidate = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(C=C, max_iter=1000))])\n"
   "    candidate_scores = cross_val_score(candidate, train[columns], train['rising'], cv=folds, scoring='roc_auc')\n"
   "    cv_by_C[C] = round(float(candidate_scores.mean()), 4)\n"
   "    candidate.fit(train[columns], train['rising'])\n"
   "    test_by_C[C] = round(float(roc_auc_score(test['rising'], candidate.predict_proba(test[columns])[:, 1])), 4)\n\n"
   "print(test_by_C)\nprint('the folds pick          :', max(cv_by_C, key=cv_by_C.get))\n"
   "print('the test rows would pick:', max(test_by_C, key=test_by_C.get))",
   f"The test rows would pick C = {TEST_BEST_C} at {TEST_BY_C[TEST_BEST_C]:.3f}, "
   f"and the folds picked {CV_BEST_C}, which scores {TEST_BY_C[CV_BEST_C]:.3f} on "
   "the test rows. The two disagree, and only the folds are allowed to choose: "
   "a C picked on the test rows leaves nothing to report it with. The test "
   "AUCs are within 0.04 of each other, which is the size of the disagreement.",
   revisits="S5")

# ============================================================ H
section(
"## ✂️ H · The l1 penalty, and the arguments\n\n"
"Exact zeros, the settings, and the error you will meet. H3 and H4 share "
"nothing but the lesson."
)

ex("H1", "The lasso's penalty on a classifier", 2,
   "Build and fit a pipeline with `LogisticRegression(penalty='l1', "
   "solver='liblinear', C=0.01)` on all columns, and print the names of the "
   "columns whose coefficient is not zero, with the coefficient.",
   "sparse = Pipeline([...])\n...\n\n...",
   ["Name the step `'logit'`. The coefficients are "
    "`sparse.named_steps['logit'].coef_[0]`, one row.",
    "`for name, b in zip(columns, coefs):` with `if b != 0: print(name, round(b, 3))`."],
   "sparse = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(penalty='l1', solver='liblinear', C=0.01))])\n"
   "sparse.fit(train[columns], train['rising'])\n\n"
   "for name, b in zip(columns, sparse.named_steps['logit'].coef_[0]):\n    if b != 0:\n        print(name, round(b, 3))",
   f"{' and '.join('`' + c + '`' for c in L1_KEPT)}, and nothing else: "
   f"{19 - len(L1_KEPT)} of the 19 coefficients are exactly zero. The one "
   "column the lecture started with is one of the two the penalty keeps.")

ex("H2", "How many survive as C grows", 3,
   "For C in `[0.001, 0.003, 0.01, 0.03, 0.1, 1]`, fit the l1 pipeline and store "
   "the number of non-zero coefficients in a dictionary. Print it.",
   "survivors = {}\n\nfor C in [0.001, 0.003, 0.01, 0.03, 0.1, 1]:\n    ...\n\nprint(survivors)",
   "`int((candidate.named_steps['logit'].coef_[0] != 0).sum())` counts the "
   "survivors as a plain number.",
   "survivors = {}\n\nfor C in [0.001, 0.003, 0.01, 0.03, 0.1, 1]:\n"
   "    candidate = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(penalty='l1', solver='liblinear', C=C))])\n"
   "    candidate.fit(train[columns], train['rising'])\n"
   "    survivors[C] = int((candidate.named_steps['logit'].coef_[0] != 0).sum())\n\nprint(survivors)",
   f"{SURV[0.001]} at C = 0.001, then {SURV[0.003]}, {SURV[0.01]}, {SURV[0.03]}, "
   f"{SURV[0.1]} and {SURV[1]} at C = 1. With the l1 penalty a small C removes "
   "columns rather than merely shrinking them, and at C = 0.001 the model is the "
   "intercept alone.",
   revisits="S2")

ex("H3", "What the defaults are", 1,
   "Print the `penalty`, `C`, `solver` and `max_iter` of a `LogisticRegression()` "
   "from `get_params()`.",
   "settings = ...\nprint(..., ..., ..., ...)",
   "`get_params()` returns a dictionary; read four keys from it.",
   "settings = LogisticRegression().get_params()\n"
   "print(settings['penalty'], settings['C'], settings['solver'], settings['max_iter'])",
   f"`{DEFAULTS['penalty']}`, `{DEFAULTS['C']}`, `{DEFAULTS['solver']}` and "
   f"`{DEFAULTS['max_iter']}`. Every logistic regression in this notebook that "
   "did not say otherwise had a ridge penalty of strength 1 on, solved by lbfgs "
   "in at most 100 steps.")

md(f"### H4 · The error you will meet  {badge(3, REVISITS['H4'])}\n\n"
   "The cell below asks the default solver for the l1 penalty, and it raises. "
   "Run it and read the message, then fix the call in the second cell, fit it, "
   "and print the number of non-zero coefficients.")
code("broken = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(penalty='l1', C=0.01))])\n"
     "broken.fit(train[columns], train['rising'])", raises=True)
code("fixed = ...\n...\nprint(...)")
_hints_solution(
   ["The message names the solvers that can handle `l1`.",
    "`solver='liblinear'` is the one the lecture used."],
   "fixed = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(penalty='l1', solver='liblinear', C=0.01))])\n"
   "fixed.fit(train[columns], train['rising'])\nprint((fixed.named_steps['logit'].coef_[0] != 0).sum())",
   f"A `ValueError` saying that lbfgs supports only the l2 penalty, then "
   f"{len(L1_KEPT)} once the solver is changed. The last line of a traceback "
   "names the problem, and here it also names the cure.")

# ============================================================ I
section(
"## \U0001f4d6 I · Reading the result\n\n"
"What the numbers mean, and which model to report. Each of these fits what it "
"needs."
)

ex("I1", "A coefficient per standard deviation, in a sentence", 2,
   "Fit a scaling pipeline with `LogisticRegression()` on `vol_20d` alone, take "
   "the coefficient, and print one sentence with an f-string: the factor by "
   "which one standard deviation more volatility multiplies the odds of a rise, "
   "to two decimals.",
   "one_pipe = ...\n...\nb = ...\n\nsentence = ...\nprint(sentence)",
   "`one_pipe.named_steps['logit'].coef_[0, 0]` is the coefficient per standard "
   "deviation; `np.exp(b)` is the factor.",
   "one_pipe = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "one_pipe.fit(train[['vol_20d']], train['rising'])\nb = one_pipe.named_steps['logit'].coef_[0, 0]\n\n"
   "sentence = f'One standard deviation more volatility multiplies the odds of a rise by {np.exp(b):.2f}.'\n"
   "print(sentence)",
   f"The coefficient is {ONE_SD_COEF:.3f} per standard deviation and the factor "
   f"is {ONE_SD_OR:.2f}. Per percentage point the factor was {ODDS_RATIO:.2f}; "
   "the two describe the same curve in different units of the column.",
   revisits="S1")

ex("I2", "Four test AUCs, ranked", 3,
   "Put the test AUC of four models into one dictionary: one column; 19 "
   "columns in a scaling pipeline with C = 1; 19 columns with C = 0.0001; and 19 "
   "columns with the l1 penalty at C = 0.01. Print them from highest to lowest.",
   "four = {}\n\n...\n\nfor name in sorted(four, key=four.get, reverse=True):\n    print(f'{name:22} {four[name]:.4f}')",
   ["Each entry is one model fitted on the training rows and scored with "
    "`roc_auc_score` on the test rows.",
    "`sorted(four, key=four.get, reverse=True)` orders the keys from the largest "
    "value down."],
   "four = {}\n\n"
   "one = LogisticRegression()\none.fit(train[['vol_20d']], train['rising'])\n"
   "four['one column'] = roc_auc_score(test['rising'], one.predict_proba(test[['vol_20d']])[:, 1])\n\n"
   "wide_1 = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(C=1, max_iter=1000))])\n"
   "wide_1.fit(train[columns], train['rising'])\n"
   "four['19 columns, C = 1'] = roc_auc_score(test['rising'], wide_1.predict_proba(test[columns])[:, 1])\n\n"
   "wide_small = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(C=0.0001, max_iter=1000))])\n"
   "wide_small.fit(train[columns], train['rising'])\n"
   "four['19 columns, C chosen'] = roc_auc_score(test['rising'], wide_small.predict_proba(test[columns])[:, 1])\n\n"
   "wide_l1 = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(penalty='l1', solver='liblinear', C=0.01))])\n"
   "wide_l1.fit(train[columns], train['rising'])\n"
   "four['19 columns, l1'] = roc_auc_score(test['rising'], wide_l1.predict_proba(test[columns])[:, 1])\n\n"
   "for name in sorted(four, key=four.get, reverse=True):\n    print(f'{name:22} {four[name]:.4f}')",
   f"Highest is \"{RANKED_ORDER[0]}\" at {RANKED[RANKED_ORDER[0]]:.4f} and lowest "
   f"\"{RANKED_ORDER[-1]}\" at {RANKED[RANKED_ORDER[-1]]:.4f}, all four within "
   "0.02. On the folds the one-column model was ahead by a wide margin; on the "
   "test years the four are level. That is the fold spread from the lecture "
   "showing up as a reminder that two years of test rows are two years.",
   revisits="S2")

ex("I3", "One function for any classifier", 4,
   "Write `evaluate(p)`: it cross-validates the model or pipeline `p` on the "
   "training rows with `folds` and the AUC, refits it on all of them, and returns "
   "the pair `(cv_auc, test_auc)`. Run it on the one-column model with "
   "`train[['vol_20d']]` and `test[['vol_20d']]`, so give the function the two "
   "feature tables as arguments too.",
   "def evaluate(p, X_train, X_test):\n    ...\n\n"
   "print('one column:', evaluate(LogisticRegression(), train[['vol_20d']], test[['vol_20d']]))\n"
   "print('19 columns:', evaluate(Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(max_iter=1000))]), train[columns], test[columns]))",
   ["Inside: `cross_val_score(p, X_train, train['rising'], cv=folds, scoring='roc_auc')` "
    "for the first number, then `p.fit(X_train, train['rising'])` and "
    "`roc_auc_score` on `p.predict_proba(X_test)[:, 1]` for the second.",
    "`return round(float(cv.mean()), 4), round(test_auc, 4)` returns a pair."],
   "def evaluate(p, X_train, X_test):\n"
   "    cv = cross_val_score(p, X_train, train['rising'], cv=folds, scoring='roc_auc')\n"
   "    p.fit(X_train, train['rising'])\n"
   "    test_auc = roc_auc_score(test['rising'], p.predict_proba(X_test)[:, 1])\n"
   "    return round(float(cv.mean()), 4), round(float(test_auc), 4)\n\n"
   "print('one column:', evaluate(LogisticRegression(), train[['vol_20d']], test[['vol_20d']]))\n"
   "print('19 columns:', evaluate(Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(max_iter=1000))]), train[columns], test[columns]))",
   f"One column ({CV_ONE.mean():.4f}, {AUC_TE:.4f}) and 19 columns "
   f"({CV_WIDE.mean():.4f}, {TEST_BY_C[1]:.4f}). Because every scikit-learn "
   "classifier fits and gives probabilities the same way, the function never "
   "needs to know what is inside it.",
   revisits="S2")

ex("I4", "The rise the model was surest would not come", 3,
   "Fit the one-column model, take the test probabilities, and among the days "
   "**with** a rise find the one given the lowest probability. Print its date, "
   "its probability and its volatility.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\np = model.predict_proba(test[['vol_20d']])[:, 1]\n"
   "rose = test['rising'].values == 1\n\np_rise = ...\nworst = ...\nprint(..., ..., ...)",
   ["`p_rise = np.where(rose, p, 9)` keeps the probability on the days with a "
    "rise and puts a 9 everywhere else, so `p_rise.argmin()` is the position "
    "of the rise with the lowest probability.",
    "`test.index[worst].date()`, `p[worst]` and `test['vol_20d'].iloc[worst]`."],
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "p = model.predict_proba(test[['vol_20d']])[:, 1]\nrose = test['rising'].values == 1\n\n"
   "p_rise = np.where(rose, p, 9)\nworst = p_rise.argmin()\n"
   "print(test.index[worst].date(), round(p[worst], 3), round(test['vol_20d'].iloc[worst], 3))",
   f"{WORST_DAY}, given a probability of {WORST_P:.3f} at a volatility of "
   f"{WORST_VOL:.2f} percent. Look up what the market did in the 20 days after "
   "that date. A model that only sees the last 20 days cannot see the event "
   "coming, and the rise it was surest would not come is always the one before "
   "an event.",
   revisits="S5")

ex("I5", "Where the two kinds of day sit", 3,
   "Draw two histograms on one axis: the test probabilities of the days with a "
   "rise and of the days without, with a legend, and a vertical line at 0.5.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\np = model.predict_proba(test[['vol_20d']])[:, 1]\n"
   "rose = test['rising'].values == 1\n\nfig, ax = plt.subplots(figsize=(8, 3.2))\n...\nplt.show()",
   ["`ax.hist(p[rose], bins=30, alpha=0.6, label='a rise came')` and the same "
    "for `p[~rose]`.",
    "`ax.axvline(0.5, color='black')` and `ax.legend()`."],
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "p = model.predict_proba(test[['vol_20d']])[:, 1]\nrose = test['rising'].values == 1\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3.2))\n"
   "ax.hist(p[rose], bins=30, alpha=0.6, label='a rise came')\n"
   "ax.hist(p[~rose], bins=30, alpha=0.6, label='no rise')\n"
   "ax.axvline(0.5, color='black')\nax.set_xlabel('probability of a rise')\nax.set_ylabel('test days')\n"
   "ax.legend()\nplt.show()",
   "The two piles overlap across the whole range, and the days with a rise sit "
   "further right. Every threshold cuts through both piles, which is the "
   "trade-off between the two kinds of error drawn as data.",
   revisits="S3")

# ============================================================ J
section(
"## \U0001f501 J · Across the desk\n\n"
"The whole workflow, once per instrument. J2 to J4 use the function from J1; "
"if you skipped J1, copy its solution into the first cell."
)

ex("J1", "A table with a label, for any ticker", 4,
   "Write `build_table(ticker)`: the six volatility windows `[5, 10, 20, 40, 60, "
   "120]` and the three return windows `[5, 20, 60]` of that ticker, the 20-day "
   "volatility of every **other** instrument as `<ticker>_vol`, the target "
   "`vol_next`, incomplete rows dropped, and then the label `rising`. Return the "
   "table, and check its shape on `'SPY'`.",
   "def build_table(ticker):\n    ...\n\nspy_table = build_table('SPY')\nprint(spy_table)",
   ["Three loops, as when the table was built for ridge: `'vol_' + str(w) + 'd'` "
    "with `.rolling(w).std()`, `'ret_' + str(w) + 'd'` with `.rolling(w).mean()`, "
    "and `if t != ticker:` inside the loop over `rets.columns`.",
    "Add `vol_next` as `.rolling(20).std().shift(-20)`, `dropna()`, and only "
    "then the label, so the comparison sees complete rows."],
   "def build_table(ticker):\n    frame = pd.DataFrame()\n"
   "    for w in [5, 10, 20, 40, 60, 120]:\n"
   "        frame['vol_' + str(w) + 'd'] = rets[ticker].rolling(w).std()\n"
   "    for w in [5, 20, 60]:\n"
   "        frame['ret_' + str(w) + 'd'] = rets[ticker].rolling(w).mean()\n"
   "    for t in rets.columns:\n        if t != ticker:\n"
   "            frame[t + '_vol'] = rets[t].rolling(20).std()\n"
   "    frame['vol_next'] = rets[ticker].rolling(20).std().shift(-20)\n"
   "    frame = frame.dropna()\n"
   "    frame['rising'] = (frame['vol_next'] > frame['vol_20d']).astype(int)\n"
   "    return frame\n\nprint(build_table('SPY').shape)",
   f"{N_TBL:,} rows and 21 columns: 19 features, the target and the label, "
   "which for SPY is the setup's table. The label goes on last, after "
   "`dropna()`, so it never compares an incomplete row.",
   revisits="S6")

ex("J2", "Nvidia", 3,
   "Build the table for `'NVDA'`, split at the end of 2022, fit the one-column "
   "logistic regression on `vol_20d`, and print its test AUC, its test accuracy, "
   "and the majority rule's accuracy on the same rows.",
   "nvda = build_table('NVDA')\ntr = ...\nte = ...\n\nmodel_n = LogisticRegression()\n...\n\n"
   "print('AUC     :', ...)\nprint('accuracy:', ...)\nprint('majority:', ...)",
   "The three fits and scores from earlier sections with `tr` and `te` in "
   "place of `train` and `test`. The majority rule is the larger of the share "
   "of rises and one minus it.",
   "nvda = build_table('NVDA')\ntr = nvda.loc[:'2022-12-31']\nte = nvda.loc['2023-01-01':]\n\n"
   "model_n = LogisticRegression()\nmodel_n.fit(tr[['vol_20d']], tr['rising'])\n\n"
   "print('AUC     :', roc_auc_score(te['rising'], model_n.predict_proba(te[['vol_20d']])[:, 1]))\n"
   "print('accuracy:', accuracy_score(te['rising'], model_n.predict(te[['vol_20d']])))\n"
   "print('majority:', max(te['rising'].mean(), 1 - te['rising'].mean()))",
   f"An AUC of {NVDA_AUC:.3f} and an accuracy of {NVDA_ACC:.3f} against "
   f"{NVDA_MAJ:.3f} for the majority rule. Nvidia's curve crosses one half at "
   f"{NVDA_CROSS:.2f} percent, three times the index's, because its volatility "
   "is three times the index's; the label has no units, but the crossing point "
   "does.")

ex("J3", "Every instrument's AUC", 4,
   "For every ticker in `rets`, build its table, fit the one-column model on the "
   "training rows and store the test AUC in a dictionary `desk_auc`. Draw the "
   "AUCs as a bar chart, sorted from highest to lowest, with a line at 0.5.",
   "desk_auc = {}\nfor ticker in rets.columns:\n    ...\n\n"
   "ranked = ...\n\nfig, ax = plt.subplots(figsize=(8, 3.2))\n...\nplt.show()",
   ["The loop body is J2 with `ticker` in place of `'NVDA'`, ending in "
    "`desk_auc[ticker] = roc_auc_score(...)`.",
    "`ranked = pd.Series(desk_auc).sort_values(ascending=False)`, then "
    "`ax.bar(ranked.index, ranked.values)` and `ax.axhline(0.5)`."],
   "desk_auc = {}\nfor ticker in rets.columns:\n"
   "    frame = build_table(ticker)\n    tr = frame.loc[:'2022-12-31']\n    te = frame.loc['2023-01-01':]\n"
   "    model_t = LogisticRegression()\n    model_t.fit(tr[['vol_20d']], tr['rising'])\n"
   "    desk_auc[ticker] = roc_auc_score(te['rising'], model_t.predict_proba(te[['vol_20d']])[:, 1])\n\n"
   "ranked = pd.Series(desk_auc).sort_values(ascending=False)\n\n"
   "fig, ax = plt.subplots(figsize=(8, 3.2))\nax.bar(ranked.index, ranked.values)\n"
   "ax.axhline(0.5, color='grey', linestyle='--')\nax.set_ylabel('test AUC')\n"
   "ax.set_title('One column, logistic regression: test AUC per instrument', loc='left')\nplt.show()",
   f"From {DESK_AUC[DESK_BEST]:.3f} on {DESK_BEST} down to "
   f"{DESK_AUC[DESK_WORST]:.3f} on {DESK_WORST}, every one of the 11 well above "
   "one half. Whether next month is more volatile than this one is predictable "
   "from this month alone on every instrument, because volatility mean-reverts "
   "everywhere.",
   revisits="S3")

ex("J4", "Which C does each instrument pick", 5,
   "For every ticker, build its table, run the grid search from G5 on its 19 "
   "columns and training rows, and store the winning C in a dictionary `best_C`. "
   "Print it, and count how many instruments pick the smallest value in the grid.",
   "best_C = {}\n\nfor ticker in rets.columns:\n    ...\n\n"
   "print(best_C)\nprint('pick 0.0001:', ...)",
   ["The feature columns of a built table are `list(frame.columns[:-2])`: every "
    "column but the target and the label.",
    "`sum(1 for t in best_C if best_C[t] == 0.0001)` counts. Eleven searches "
    "take a little while."],
   "best_C = {}\n\nfor ticker in rets.columns:\n"
   "    frame = build_table(ticker)\n    tr = frame.loc[:'2022-12-31']\n"
   "    cols = list(frame.columns[:-2])\n"
   "    search_t = GridSearchCV(Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(max_iter=1000))]),\n"
   "                            {'logit__C': [0.0001, 0.001, 0.01, 0.1, 1, 10]},\n"
   "                            cv=folds, scoring='roc_auc')\n"
   "    search_t.fit(tr[cols], tr['rising'])\n"
   "    best_C[ticker] = search_t.best_params_['logit__C']\n\n"
   "print(best_C)\nprint('pick 0.0001:', sum(1 for t in best_C if best_C[t] == 0.0001))",
   f"{N_PICK_SMALLEST} of {len(DESK_C)} pick 0.0001" +
   (f"; {', '.join(f'{t} picks {c}' for t, c in C_OTHERS.items())}." if C_OTHERS else ".") +
   " For ridge, every instrument agreed on alpha; here the chosen C runs across "
   "the whole grid. On a label the folds are noisier than on a number, because "
   "a 0 or a 1 carries less information than a volatility, so the choice of C "
   "moves with the instrument. The test rows were never touched.",
   revisits="S2")


# ============================================================ K
section(
"## \U0001f9e9 K · Small cases\n\n"
"Five short investigations that each start from scratch: no variable from an "
"earlier section, and no scaffold beyond the first line. Each one needs "
"today's tools and a few older ones."
)

ex("K1", "A month of predictions, checked one by one", 3,
   "Fit the one-column model and take the test probabilities. For the **last 20** "
   "test days, print one line each: the date, the probability to two decimals, "
   "the word `rise` or `no rise` depending on whether the probability is at "
   "least one half, and `right` or `wrong` depending on the label. Count the "
   "days the model got right.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "p = model.predict_proba(test[['vol_20d']])[:, 1]\n\nhits = 0\n...\n\nprint('right on', hits, 'of 20 days')",
   ["`zip(test.index[-20:], p[-20:], test['rising'].values[-20:])` walks the "
    "three together. `date.date()` prints a date without the time.",
    "Inside the loop, an `if` sets `call` to 1 or 0 and the word; `right` is "
    "`call == label`; add one to `hits` when it is."],
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "p = model.predict_proba(test[['vol_20d']])[:, 1]\n\nhits = 0\n"
   "for date, prob, label in zip(test.index[-20:], p[-20:], test['rising'].values[-20:]):\n"
   "    if prob >= 0.5:\n        call, word = 1, 'rise'\n    else:\n        call, word = 0, 'no rise'\n"
   "    if call == label:\n        verdict = 'right'\n        hits = hits + 1\n    else:\n        verdict = 'wrong'\n"
   "    print(f'{date.date()}  {prob:.2f}  {word:8}  {verdict}')\n\nprint('right on', hits, 'of 20 days')",
   f"Right on {K1_HITS} of the last 20 test days. Twenty lines of a loop with an "
   "`if` inside, an f-string with a width for the word so the columns line up, "
   "and a counter: the pieces of the first two sessions, doing the reading "
   "that the confusion matrix does in one line.")

ex("K2", "Which instrument rose most often in 2024", 3,
   "For every ticker in `rets`, compute the share of 2024 days after which "
   "volatility rose, straight from the returns: the 20-day volatility, the same "
   "series shifted 20 days up, a table of the two with incomplete rows dropped, "
   "and the comparison. Store the shares in a dictionary and print the three "
   "largest.",
   "share_2024 = {}\n\nfor ticker in rets.columns:\n    ...\n\n"
   "for ticker in sorted(share_2024, key=share_2024.get, reverse=True)[:3]:\n    print(ticker, share_2024[ticker])",
   ["`now = rets[ticker].rolling(20).std()`, `after = now.shift(-20)`, then "
    "`both = pd.DataFrame({'now': now, 'after': after}).dropna()`.",
    "`(both['after'] > both['now']).loc['2024-01-01':'2024-12-31'].mean()` is the "
    "share; wrap it in `round(float(...), 3)`."],
   "share_2024 = {}\n\nfor ticker in rets.columns:\n"
   "    now = rets[ticker].rolling(20).std()\n    after = now.shift(-20)\n"
   "    both = pd.DataFrame({'now': now, 'after': after}).dropna()\n"
   "    share_2024[ticker] = round(float((both['after'] > both['now']).loc['2024-01-01':'2024-12-31'].mean()), 3)\n\n"
   "for ticker in sorted(share_2024, key=share_2024.get, reverse=True)[:3]:\n    print(ticker, share_2024[ticker])",
   f"{K2_TOP[0]} at {K2[K2_TOP[0]]}, then {K2_TOP[1]} at {K2[K2_TOP[1]]} and "
   f"{K2_TOP[2]} at {K2[K2_TOP[2]]}. The `dropna()` matters: the last 20 days of "
   "2024 have no next month, and a comparison with a missing value quietly "
   "counts as 0.")

ex("K3", "One sentence per instrument", 4,
   "Write `report(ticker)`: from the returns alone, build a two-column table of "
   "`vol_20d` and `vol_next` for that ticker, add the label, split at the end of "
   "2022, fit the one-column logistic regression, and **return** one sentence "
   "with the test AUC to two decimals and the test accuracy next to the "
   "majority rule, both as percentages. Print it for `KO`, `JPM` and `DIS`.",
   "def report(ticker):\n    ...\n\nfor ticker in ['KO', 'JPM', 'DIS']:\n    print(report(ticker))",
   ["Inside: `now = rets[ticker].rolling(20).std()`, the table "
    "`pd.DataFrame({'vol_20d': now, 'vol_next': now.shift(-20)}).dropna()`, the "
    "label as a comparison, then the split and the fit as in section E.",
    "`f'{ticker}: AUC {auc:.2f}, accuracy {acc:.1%} against {majority:.1%} for the majority rule.'`"],
   "def report(ticker):\n"
   "    now = rets[ticker].rolling(20).std()\n"
   "    frame = pd.DataFrame({'vol_20d': now, 'vol_next': now.shift(-20)}).dropna()\n"
   "    frame['rising'] = (frame['vol_next'] > frame['vol_20d']).astype(int)\n"
   "    tr = frame.loc[:'2022-12-31']\n    te = frame.loc['2023-01-01':]\n"
   "    m = LogisticRegression()\n    m.fit(tr[['vol_20d']], tr['rising'])\n"
   "    auc = roc_auc_score(te['rising'], m.predict_proba(te[['vol_20d']])[:, 1])\n"
   "    acc = accuracy_score(te['rising'], m.predict(te[['vol_20d']]))\n"
   "    majority = max(te['rising'].mean(), 1 - te['rising'].mean())\n"
   "    return f'{ticker}: AUC {auc:.2f}, accuracy {acc:.1%} against {majority:.1%} for the majority rule.'\n\n"
   "for ticker in ['KO', 'JPM', 'DIS']:\n    print(report(ticker))",
   f"Coca-Cola scores an AUC of {K3['KO'][0]:.2f}, JPMorgan {K3['JPM'][0]:.2f} and "
   f"Disney {K3['DIS'][0]:.2f}, each with an accuracy above its majority rule. "
   "The whole workflow, from raw returns to a scored classifier, is one function "
   "of twelve lines, and the sentence it returns is the one a report would carry.")

ex("K4", "The longest run of predicted rises", 4,
   "Fit the one-column model and predict the test rows. Walk through the "
   "predictions in order with a loop and find the longest run of consecutive "
   "days predicted as a rise, and the date on which that run ended.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\npredicted = model.predict(test[['vol_20d']])\n\n"
   "streak = 0\nlongest = 0\nend_date = None\n...\n\nprint(longest, 'days, ending', end_date)",
   ["`for date, value in zip(test.index, predicted):` and, inside, add one to "
    "`streak` when `value == 1` and set it back to 0 otherwise.",
    "Whenever `streak` passes `longest`, store both `streak` and `date.date()`."],
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "predicted = model.predict(test[['vol_20d']])\n\n"
   "streak = 0\nlongest = 0\nend_date = None\n"
   "for date, value in zip(test.index, predicted):\n"
   "    if value == 1:\n        streak = streak + 1\n"
   "        if streak > longest:\n            longest = streak\n            end_date = date.date()\n"
   "    else:\n        streak = 0\n\nprint(longest, 'days, ending', end_date)",
   f"{K4_LONGEST} trading days, ending {K4_END}. A run that long is the model "
   "saying \"volatility will rise\" through a whole stretch of calm, because "
   "volatility stayed below the crossing point the entire time. A counter that "
   "resets is the loop shape for any \"longest run\" question.")

ex("K5", "Accuracy, year by year", 3,
   "Fit the one-column model, predict the test rows, and compute the accuracy "
   "separately for 2023 and 2024 with `groupby`: make a Series of `True` and "
   "`False` for whether each prediction was right, indexed by date, and group it "
   "by year.",
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\npredicted = model.predict(test[['vol_20d']])\n\n"
   "right = ...\nby_year = ...\nprint(by_year)",
   ["`right = pd.Series(predicted == test['rising'].values, index=test.index)`.",
    "`right.groupby(right.index.year).mean()`: the mean of a column of `True` "
    "and `False` is the share of `True`."],
   "model = LogisticRegression()\nmodel.fit(train[['vol_20d']], train['rising'])\n"
   "predicted = model.predict(test[['vol_20d']])\n\n"
   "right = pd.Series(predicted == test['rising'].values, index=test.index)\n"
   "by_year = right.groupby(right.index.year).mean()\nprint(by_year)",
   f"{K5[2023]:.3f} in 2023 and {K5[2024]:.3f} in 2024, " +
   ("close to each other: one accuracy over two years hid nothing this time, and a `groupby` on the year is the one-line way to check that."
    if abs(K5[2023] - K5[2024]) < 0.03 else
    "far enough apart that one accuracy over two years hides a real difference; a `groupby` on the year is the one-line way to see it."))

# ---------------------------------------------------------------- closing
md(
"## \U0001f3c1 Done\n\n"
"You made labels, watched a straight line give probabilities below zero, fitted "
"the curve that cannot, and read its two numbers as a crossing point and an "
"odds factor. You turned probabilities into predictions at a threshold of your "
"choosing, scored them with the four counts and the AUC, chose C on the folds, "
"and ran the whole workflow on every instrument.\n\n"
"The case takes the same tools back to the risk report, where the data is in "
"decimals, and where the default penalty does something to a raw column that "
"the label's lack of units does not protect against."
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
