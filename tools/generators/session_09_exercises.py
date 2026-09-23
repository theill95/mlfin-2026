# -*- coding: utf-8 -*-
"""Build session_09_exercises.ipynb.

Same conventions as Sessions 1 to 8: pleasant intro, 1-5 star badges, toolkit
card with title= hover docs, task -> work cell (blank-safe `...`) -> 1-2 folded
hints -> folded solution, no em-dashes, plain explanatory tone.

Session 9 is classification in practice. The exercises use both of the
lecture's tables: the credit table, where one borrower in five defaults, and
the index table from Session 6, where the label is a rare jump in volatility
or a three-way move. They make rare labels and majority rules, split a
cross-section at random, read the four counts, choose a threshold from costs,
weight a rare class, check calibration, fit a three-class model and read its
scores, and put k-nearest neighbours against logistic regression on the same
folds.

Sections A to J work through the session. Section K is five standalone small
cases that deliberately reach back to loops, `if`, dictionaries, f-strings and
functions, in this session's context.

Only tools taught by the end of Session 9. New this session: train_test_split
(test_size, random_state, stratify), class_weight="balanced", pd.cut,
brier_score_loss, CalibratedClassifierCV, average_precision_score,
classification_report, f1_score(average=), KNeighborsClassifier(n_neighbors),
decision_function, confusion_matrix(labels=), .ravel(). NOT taught, so never
required unless the task names it: precision_recall_curve (named in E4),
calibration_curve, StratifiedKFold, DummyClassifier, RocCurveDisplay,
LogisticRegressionCV, SMOTE.

Market returns are in PERCENT here, exactly as in the lecture. The credit
table is in New Taiwan dollars, as it comes.

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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                             roc_auc_score, average_precision_score, brier_score_loss,
                             f1_score, precision_recall_curve)
from sklearn.model_selection import (cross_val_score, TimeSeriesSplit, GridSearchCV,
                                     train_test_split)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "session_09" / "session_09_exercises.ipynb"

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
    "A1": "S1", "A3": "S3", "A5": "S2", "A6": "S2", "A7": "S3",
    "B3": "S5", "B5": "S4",
    "C2": "S4", "C3": "S2", "C4": "S2", "C5": "S2", "C6": "S2",
    "D2": "S3", "D5": "S2", "D6": "S1", "D7": "S3",
    "E3": "S4", "E5": "S3",
    "F1": "S3", "F2": "S3", "F6": "S3",
    "G2": "S3", "G5": "S5", "G6": "S3", "G9": "S3",
    "H2": "S6", "H3": "S2", "H4": "S6", "H6": "S5", "H7": "S2", "H8": "S6",
    "I2": "S6", "I4": "S3", "I5": "S8",
    "J1": "S2", "J2": "S2", "J3": "S5", "J4": "S2", "J5": "S5",
    "K1": "S2", "K2": "S2", "K3": "S2", "K4": "S2", "K5": "S3",
}


def ex(sid, title, n, task, work, hints, sol_code, sol_note, revisits=None, raises=False):
    md(f"### {sid} · {title}  {badge(n, revisits or REVISITS.get(sid))}\n\n{task}")
    code(work, raises=raises)
    _hints_solution(hints, sol_code, sol_note)


def section(header):
    md(header)


# ======================================================================
# The real numbers, computed here so every solution note is exact.
# ======================================================================
CREDIT = pd.read_csv(ROOT / "data" / "credit.csv")
CCOLS = ["limit", "age", "late_now", "months_late", "bill", "paid", "utilisation"]
N_CREDIT = len(CREDIT)
N_DEFAULT = int(CREDIT["default"].sum())
SHARE_DEFAULT = float(CREDIT["default"].mean())
BY_LATE = CREDIT.groupby("months_late")["default"].agg(["size", "mean"])

C_TRAIN, C_TEST = train_test_split(CREDIT, test_size=0.3, random_state=0,
                                   stratify=CREDIT["default"])
N_CTRAIN, N_CTEST = len(C_TRAIN), len(C_TEST)
SHARE_TE = float(C_TEST["default"].mean())
MAJORITY_C = 1 - SHARE_TE
# stratify on a SMALL test set, where the drift is easy to see
STRAT_SHARES, PLAIN_SHARES = [], []
for _seed in [0, 1, 2, 3, 4]:
    _a, _b = train_test_split(CREDIT, test_size=0.02, random_state=_seed,
                              stratify=CREDIT["default"])
    STRAT_SHARES.append(round(float(_b["default"].mean()), 4))
    _a, _b = train_test_split(CREDIT, test_size=0.02, random_state=_seed)
    PLAIN_SHARES.append(round(float(_b["default"].mean()), 4))
SPREAD_STRAT = max(STRAT_SHARES) - min(STRAT_SHARES)
SPREAD_PLAIN = max(PLAIN_SHARES) - min(PLAIN_SHARES)


def cpipe(**kw):
    return Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(**kw))])


CMODEL = cpipe().fit(C_TRAIN[CCOLS], C_TRAIN["default"])
PC = CMODEL.predict_proba(C_TEST[CCOLS])[:, 1]
PC_TRAIN = CMODEL.predict_proba(C_TRAIN[CCOLS])[:, 1]
PRED_C = CMODEL.predict(C_TEST[CCOLS])
YC = C_TEST["default"]
ACC_C = float(accuracy_score(YC, PRED_C))
AUC_C = float(roc_auc_score(YC, PC))
AP_C = float(average_precision_score(YC, PC))
CM_C = confusion_matrix(YC, PRED_C)
TN_C, FP_C, FN_C, TP_C = (int(CM_C[0, 0]), int(CM_C[0, 1]), int(CM_C[1, 0]), int(CM_C[1, 1]))
PREC_C = float(precision_score(YC, PRED_C))
REC_C = float(recall_score(YC, PRED_C))
BRIER_C = float(brier_score_loss(YC, PC))

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
BY_THR = {}
for _t in THRESHOLDS:
    _pr = (PC >= _t).astype(int)
    _tn, _fp, _fn, _tp = confusion_matrix(YC, _pr).ravel()
    BY_THR[_t] = dict(precision=float(precision_score(YC, _pr, zero_division=0)),
                      recall=float(recall_score(YC, _pr)),
                      cost=int(5 * _fn + _fp), fn=int(_fn), fp=int(_fp))
# walking DOWN from 0.5 in steps of 0.05, the first threshold to reach recall 0.5
WALK = [round(0.5 - 0.05 * i, 2) for i in range(9)]
FIRST_REC_HALF = None
for _t in WALK:
    if recall_score(YC, (PC >= _t).astype(int)) >= 0.5:
        FIRST_REC_HALF = _t
        break
REC_AT_FIRST = float(recall_score(YC, (PC >= FIRST_REC_HALF).astype(int)))

COST_GRID = np.arange(0.05, 0.85, 0.01)
COST_TEST = np.array([5 * confusion_matrix(YC, (PC >= t).astype(int)).ravel()[2]
                      + confusion_matrix(YC, (PC >= t).astype(int)).ravel()[1] for t in COST_GRID])
BEST_THR = float(COST_GRID[COST_TEST.argmin()])
BEST_COST = int(COST_TEST.min())
COST_AT_HALF = int(COST_TEST[int(np.argmin(np.abs(COST_GRID - 0.5)))])
COST_TRAIN = np.array([5 * confusion_matrix(C_TRAIN["default"], (PC_TRAIN >= t).astype(int)).ravel()[2]
                       + confusion_matrix(C_TRAIN["default"], (PC_TRAIN >= t).astype(int)).ravel()[1]
                       for t in COST_GRID])
TRAIN_THR = float(COST_GRID[COST_TRAIN.argmin()])
COST_AT_TRAIN_THR = int(COST_TEST[COST_TRAIN.argmin()])
FORMULA_THR = 1 / 6

CBAL = cpipe(class_weight="balanced").fit(C_TRAIN[CCOLS], C_TRAIN["default"])
PB = CBAL.predict_proba(C_TEST[CCOLS])[:, 1]
PRED_B = CBAL.predict(C_TEST[CCOLS])
ACC_B = float(accuracy_score(YC, PRED_B))
REC_B = float(recall_score(YC, PRED_B))
PREC_B = float(precision_score(YC, PRED_B))
AUC_B = float(roc_auc_score(YC, PB))
BRIER_B = float(brier_score_loss(YC, PB))
MEAN_PB = float(PB.mean())
MEAN_PC = float(PC.mean())

_prec, _rec, _thr = precision_recall_curve(YC, PC)
_i60 = int(np.argmin(np.abs(_rec - 0.6)))
PREC_AT_REC60 = float(_prec[_i60])
THR_AT_REC60 = float(_thr[min(_i60, len(_thr) - 1)])

CUTS = [0, 0.2, 0.4, 0.6, 1.0]
_bk = pd.cut(PC, CUTS)
BUCKETS = pd.DataFrame({"p": PC, "y": YC.values}).groupby(_bk, observed=True)["y"].agg(["size", "mean"])
_bkb = pd.cut(PB, CUTS)
BUCKETS_B = pd.DataFrame({"p": PB, "y": YC.values}).groupby(_bkb, observed=True)["y"].agg(["size", "mean"])
CAL = CalibratedClassifierCV(cpipe(class_weight="balanced"), method="sigmoid", cv=5)
CAL.fit(C_TRAIN[CCOLS], C_TRAIN["default"])
P_CAL = CAL.predict_proba(C_TEST[CCOLS])[:, 1]
MEAN_CAL = float(P_CAL.mean())
BRIER_CAL = float(brier_score_loss(YC, P_CAL))
COST_ONE_SIXTH = {}
for _name, _p in [("plain", PC), ("balanced", PB), ("calibrated", P_CAL)]:
    _tn, _fp, _fn, _tp = confusion_matrix(YC, (_p >= 1 / 6).astype(int)).ravel()
    COST_ONE_SIXTH[_name] = int(5 * _fn + _fp)

# ---- the index table, in percent, as in the lecture ------------------------
TBL = pd.read_csv(ROOT / "data" / "market_features.csv", parse_dates=["date"]).set_index("date")
COLUMNS = list(TBL.columns[:19])
RATIO = TBL["vol_next"] / TBL["vol_20d"]
TBL["jump"] = (RATIO > 1.5).astype(int)
TBL["move"] = pd.cut(RATIO, [-np.inf, 0.85, 1.25, np.inf], labels=["falls", "stays", "rises"])
TRAIN, TEST = TBL.loc[:"2022-12-31"], TBL.loc["2023-01-01":]
N_TRAIN, N_TEST = len(TRAIN), len(TEST)
TS5 = TimeSeriesSplit(n_splits=5)

SHARE_JUMP_TR = float(TRAIN["jump"].mean())
SHARE_JUMP_TE = float(TEST["jump"].mean())
N_JUMP_TE = int(TEST["jump"].sum())
MAJORITY_J = 1 - SHARE_JUMP_TE
JMODEL = Pipeline([("scale", StandardScaler()),
                   ("logit", LogisticRegression())]).fit(TRAIN[["vol_20d"]], TRAIN["jump"])
PJ = JMODEL.predict_proba(TEST[["vol_20d"]])[:, 1]
PRED_J = JMODEL.predict(TEST[["vol_20d"]])
ACC_J = float(accuracy_score(TEST["jump"], PRED_J))
AUC_J = float(roc_auc_score(TEST["jump"], PJ))
N_PRED_J = int(PRED_J.sum())
PJ_MAX = float(PJ.max())

MOVE_TR = TRAIN["move"].value_counts(normalize=True)
MOVE_TE = TEST["move"].value_counts(normalize=True)
MAJ_MOVE = str(TRAIN["move"].value_counts().idxmax())
MAJ_MOVE_ACC = float((TEST["move"] == MAJ_MOVE).mean())
MC = Pipeline([("scale", StandardScaler()),
               ("logit", LogisticRegression(max_iter=1000))]).fit(TRAIN[["vol_20d", "ret_20d"]],
                                                                 TRAIN["move"])
MC_LOGIT = MC.named_steps["logit"]
PRED_MOVE = MC.predict(TEST[["vol_20d", "ret_20d"]])
ACC_MOVE = float(accuracy_score(TEST["move"], PRED_MOVE))
ORDER = ["falls", "stays", "rises"]
CM_MOVE = confusion_matrix(TEST["move"], PRED_MOVE, labels=ORDER)
F1_MACRO = float(f1_score(TEST["move"], PRED_MOVE, average="macro", zero_division=0))
F1_WEIGHTED = float(f1_score(TEST["move"], PRED_MOVE, average="weighted", zero_division=0))
F1_EACH = f1_score(TEST["move"], PRED_MOVE, average=None, labels=ORDER, zero_division=0)
REC_EACH = recall_score(TEST["move"], PRED_MOVE, average=None, labels=ORDER, zero_division=0)
DAY = TEST[["vol_20d", "ret_20d"]].head(1)
SCORES = MC.decision_function(DAY)[0]
EXPS = np.exp(SCORES)
PROBS = EXPS / EXPS.sum()
TWO_STEP = int(CM_MOVE[0, 2] + CM_MOVE[2, 0])

TBL["rising"] = (TBL["vol_next"] > TBL["vol_20d"]).astype(int)
TRAIN, TEST = TBL.loc[:"2022-12-31"], TBL.loc["2023-01-01":]
TWO = ["vol_20d", "ret_20d"]
KNN_AUC = {}
for _k in [1, 15, 51, 151, 301]:
    _p = Pipeline([("scale", StandardScaler()),
                   ("knn", KNeighborsClassifier(n_neighbors=_k))]).fit(TRAIN[TWO], TRAIN["rising"])
    KNN_AUC[_k] = (float(roc_auc_score(TRAIN["rising"], _p.predict_proba(TRAIN[TWO])[:, 1])),
                   float(roc_auc_score(TEST["rising"], _p.predict_proba(TEST[TWO])[:, 1])))
KGRID = [1, 5, 15, 51, 101, 151, 201, 301]
KSEARCH = GridSearchCV(Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier())]),
                       {"knn__n_neighbors": KGRID}, cv=TS5, scoring="roc_auc").fit(TRAIN[TWO], TRAIN["rising"])
BEST_K = KSEARCH.best_params_["knn__n_neighbors"]
BEST_K_CV = float(KSEARCH.best_score_)
KNN_TEST_AUC = float(roc_auc_score(TEST["rising"], KSEARCH.best_estimator_.predict_proba(TEST[TWO])[:, 1]))
LOG_CV = cross_val_score(Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression())]),
                         TRAIN[TWO], TRAIN["rising"], cv=TS5, scoring="roc_auc")
LOG_CV_MEAN = float(LOG_CV.mean())
_lm = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression())]).fit(TRAIN[TWO], TRAIN["rising"])
LOG_TEST_AUC = float(roc_auc_score(TEST["rising"], _lm.predict_proba(TEST[TWO])[:, 1]))
FOLD_SIZES = [(len(a), len(b)) for a, b in TS5.split(TRAIN)]
SMALLEST_FOLD = FOLD_SIZES[0][0]

_bare = KNeighborsClassifier(n_neighbors=25).fit(C_TRAIN[CCOLS], C_TRAIN["default"])
AUC_KNN_BARE = float(roc_auc_score(YC, _bare.predict_proba(C_TEST[CCOLS])[:, 1]))
_scaled = Pipeline([("scale", StandardScaler()),
                    ("knn", KNeighborsClassifier(n_neighbors=25))]).fit(C_TRAIN[CCOLS], C_TRAIN["default"])
AUC_KNN_SCALED = float(roc_auc_score(YC, _scaled.predict_proba(C_TEST[CCOLS])[:, 1]))
SD_LIMIT = float(C_TRAIN["limit"].std())
SD_LATE = float(C_TRAIN["late_now"].std())
_k5 = Pipeline([("scale", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=5))]).fit(TRAIN[TWO], TRAIN["rising"])
P_K5 = sorted(set(np.round(_k5.predict_proba(TEST[TWO])[:, 1], 3)))

# ---- the desk, for section J ----------------------------------------------
PX = pd.read_csv(ROOT / "data" / "prices.csv", parse_dates=["date"])
WIDE = PX.pivot(index="date", columns="ticker", values="close")
RET = WIDE.pct_change().dropna() * 100
TICKERS = sorted(WIDE.columns)


def jump_table(ticker):
    r = RET[ticker]
    frame = pd.DataFrame({"vol_20d": r.rolling(20).std()})
    frame["vol_next"] = r.rolling(20).std().shift(-20)
    frame = frame.dropna()
    frame["jump"] = (frame["vol_next"] > 1.5 * frame["vol_20d"]).astype(int)
    return frame


DESK_SHARE, DESK_AUC, DESK_PRED1 = {}, {}, {}
for _t in TICKERS:
    _f = jump_table(_t)
    _tr, _te = _f.loc[:"2022-12-31"], _f.loc["2023-01-01":]
    DESK_SHARE[_t] = round(float(_te["jump"].mean()), 3)
    _m = Pipeline([("scale", StandardScaler()),
                   ("logit", LogisticRegression())]).fit(_tr[["vol_20d"]], _tr["jump"])
    DESK_AUC[_t] = round(float(roc_auc_score(_te["jump"], _m.predict_proba(_te[["vol_20d"]])[:, 1])), 3)
    DESK_PRED1[_t] = int(_m.predict(_te[["vol_20d"]]).sum())
N_SILENT = sum(1 for t in TICKERS if DESK_PRED1[t] == 0)
BEST_DESK = max(TICKERS, key=lambda t: DESK_AUC[t])
WORST_DESK = min(TICKERS, key=lambda t: DESK_AUC[t])

# ---- section K numbers -----------------------------------------------------
K_PROBS = [0.04, 0.19, 0.33, 0.51, 0.08, 0.62, 0.27]
K1_WORDS = ["low" if x < 0.2 else ("watch" if x < 0.5 else "act") for x in K_PROBS]
K2_SAID = [1, 0, 1, 1, 0, 0, 1, 0]
K2_WAS = [1, 0, 0, 1, 1, 0, 1, 1]
K2_COUNTS = {"tp": sum(1 for s, w in zip(K2_SAID, K2_WAS) if s == 1 and w == 1),
             "fp": sum(1 for s, w in zip(K2_SAID, K2_WAS) if s == 1 and w == 0),
             "fn": sum(1 for s, w in zip(K2_SAID, K2_WAS) if s == 0 and w == 1),
             "tn": sum(1 for s, w in zip(K2_SAID, K2_WAS) if s == 0 and w == 0)}
_called = (PJ >= 0.3).astype(int)
_run = _best = 0
for _v in _called:
    _run = _run + 1 if _v == 1 else 0
    _best = max(_best, _run)
K4_RUN = _best
K4_CALLED = int(_called.sum())
_right = pd.Series(PRED_J == TEST["jump"].values, index=TEST.index)
K5 = _right.groupby(_right.index.year).mean().round(3).to_dict()

# extra numbers for exercises that introduce their own label
CALM = (RATIO < 0.7).astype(int)
SHARE_CALM = float(CALM.mean())
N_CALM = int(CALM.sum())
SHARE_RISING_TE = float(TEST["rising"].mean())

# ======================================================================
# Notebook
# ======================================================================
md(
"# \U0001f9ea Session 9 exercises\n"
"### Classification in practice\n\n"
"Session 8 fitted a classifier and scored it. This session is about the "
"decisions around it: what happens when the thing you predict is rare, where "
"the threshold should sit once each mistake has a price, whether the "
"probabilities mean what they say, what changes with three classes, and how a "
"second kind of classifier compares.\n\n"
"Two tables, the same two as the lecture. The **credit table** has 30,000 "
"borrowers and one column saying whether each defaulted. The **index table** "
"is the one from Session 6, with returns in percent."
)

md(
"## How to use this notebook\n\n"
"- Run the **setup cell** below first. It loads both tables, makes the splits "
"and the labels, and imports the scikit-learn pieces.\n"
"- Each exercise has a **task**, then a **code cell** for your work. Cells with "
"`...` are blanks to fill in. Replace them with real code.\n"
"- Stuck? Open the **\U0001f4a1 Hint**, but only after a genuine attempt. Open the "
"**✅ Solution** to *check* yourself, not to skip the thinking.\n"
"- Every cell runs cleanly even with the blanks still in place, so pressing "
"**Run all** never floods you with errors.\n"
"- Most exercises stand alone. A few short runs build on each other (B4 to "
"B5, D3 to D4, F3 to F5, G3 to G7, H3 to H4); the task says which earlier "
"exercise it continues from. Section J uses the function from J1, and "
"**section K is five small cases that each start from scratch**. Six of "
"them ask you to draw something: A7, D7, E5, F6, G9 and H8.\n\n"
"**You are not expected to finish all of these.** Do what you can, and come back "
"to the rest when you revise. Short on time? Read the hint, then the solution. A "
"worked solution you genuinely understand is real learning too.\n\n"
"**Units.** The index table is in percent, as in the lecture. The credit table "
"is in New Taiwan dollars, as it comes. Labels have no units at all."
)

md("---")

md(
"## \U0001f9f0 Toolkit\n\n"
"New this session. Hover a name for what it does.\n\n"
'<span title="Shuffle the rows and cut them in two. test_size is the share held back, random_state fixes which rows, stratify keeps a label share equal in both halves.">`train_test_split`</span> · '
'<span title="An argument of LogisticRegression. balanced counts each class in inverse proportion to its size while fitting.">`class_weight`</span> · '
'<span title="Sort every value into one of the ranges given. Returns the range each row fell in, or a name if labels= is passed.">`pd.cut`</span> · '
'<span title="Mean squared error of the probabilities, with what happened as 1 or 0. Smaller is better.">`brier_score_loss`</span> · '
'<span title="Wraps a model and learns a small second model mapping its numbers onto probabilities, using folds.">`CalibratedClassifierCV`</span> · '
'<span title="Summarises the whole precision-recall curve in one number. Compare it with the share of ones.">`average_precision_score`</span> · '
'<span title="Prints precision, recall and F1 for every class, plus the macro and weighted averages.">`classification_report`</span> · '
'<span title="average=None gives one score per class, macro averages them equally, weighted averages them by class size.">`f1_score(average=)`</span> · '
'<span title="Classifies a row by a vote among its k nearest rows. Needs a scaler in front of it.">`KNeighborsClassifier`</span> · '
'<span title="The raw score per class, before the exponential and the division that turn scores into probabilities.">`decision_function`</span> · '
'<span title="Flattens a 2 by 2 confusion matrix into tn, fp, fn, tp in reading order.">`.ravel()`</span>\n\n'
"**Formulas**\n\n"
"- cost of a set of decisions = (cost of a miss) x misses + (cost of a false alarm) x false alarms\n"
"- the cheapest threshold = cost of a false alarm / (cost of a false alarm + cost of a miss)\n"
"- softmax: probability of class *j* = exp(score *j*) divided by the sum of exp over all classes\n"
"- a model is calibrated when, among the rows it gave about *p*, about *p* of them happened"
)

md("---")

md("## ⚙️ Setup · run me first")

code(
'''import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                             roc_auc_score, average_precision_score, brier_score_loss,
                             f1_score, classification_report)
from sklearn.model_selection import (train_test_split, cross_val_score, TimeSeriesSplit,
                                     GridSearchCV)

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


# ---- the credit table: one row per borrower, no time order ----
credit = pd.read_csv(data_path("credit.csv"))
c_cols = ["limit", "age", "late_now", "months_late", "bill", "paid", "utilisation"]
c_train, c_test = train_test_split(credit, test_size=0.3, random_state=0,
                                   stratify=credit["default"])

# ---- the eleven instruments, for section J. Returns in percent. ----
prices = pd.read_csv(data_path("prices.csv"), parse_dates=["date"])
tickers = sorted(prices["ticker"].unique())
rets = prices.pivot(index="date", columns="ticker", values="close").pct_change().dropna() * 100

# ---- the index table from Session 6: one row per trading day, percent ----
table = pd.read_csv(data_path("market_features.csv"), parse_dates=["date"]).set_index("date")
columns = list(table.columns[:19])
ratio = table["vol_next"] / table["vol_20d"]

table["rising"] = (table["vol_next"] > table["vol_20d"]).astype(int)   # Session 8's label
table["jump"]   = (ratio > 1.5).astype(int)                            # a RARE label
table["move"]   = pd.cut(ratio, [-np.inf, 0.85, 1.25, np.inf],
                         labels=["falls", "stays", "rises"])           # three classes

train = table.loc[:"2022-12-31"]
test = table.loc["2023-01-01":]
folds = TimeSeriesSplit(n_splits=5)

print("credit:", credit.shape, "  default share", round(credit["default"].mean(), 4))
print("prices:", prices.shape[0], "rows,", len(tickers), "instruments")
print("index :", table.shape, "  train", len(train), " test", len(test))
print("jump share: train", round(train["jump"].mean(), 3), " test", round(test["jump"].mean(), 3))
print("move counts in the training days:")
print(train["move"].value_counts())'''
)

md("---")

# ====================================================== A
section(
"## A · Rare labels, and the rule to beat\n\n"
"A label is only as interesting as the baseline it has to beat. These build "
"the baselines."
)

ex("A1", "The share that defaulted", 1,
   "Print the number of borrowers who defaulted and the share, the share "
   "formatted as a percentage with one decimal.",
   'n_defaults = ...\nshare_text = ...\n\nprint(n_defaults)\nprint(share_text)',
   "The label is a column of ones and zeros, so `.sum()` counts them and "
   "`.mean()` is the share. Format it with `f\"{value:.1%}\"`.",
   'n_defaults = credit[\'default\'].sum()\nshare_text = f"{credit[\'default\'].mean():.1%}"\n\nprint(n_defaults)\nprint(share_text)',
   f"{N_DEFAULT:,} of {N_CREDIT:,} borrowers defaulted, which is "
   f"{SHARE_DEFAULT:.1%}. One borrower in five is not rare in the way fraud is, "
   "but it is far enough from half that accuracy stops being useful on its own.")

ex("A2", "The rule to beat", 2,
   "Without fitting anything, work out the accuracy of predicting that nobody "
   "defaults, on the test borrowers. Build the prediction with `np.zeros` and "
   "score it with `accuracy_score` rather than reasoning it out.",
   'nobody = ...\nbaseline = ...\n\nprint(baseline)',
   ["`np.zeros(len(c_test), dtype=int)` is a prediction of 0 on every row.",
    "`accuracy_score(c_test['default'], nobody)`."],
   "nobody = np.zeros(len(c_test), dtype=int)\nbaseline = round(accuracy_score(c_test['default'], nobody), 4)\n\nprint(baseline)",
   f"{MAJORITY_C:.4f}. A model that scores 0.79 here has done nothing at all. "
   "Computing the baseline rather than assuming it is worth the two lines, "
   "because it is the number every later score is read against.")

ex("A3", "Default rate by months late", 2,
   "Group the borrowers by `months_late` and report, for each value, how many "
   "borrowers there are and what share of them defaulted, rounded to three "
   "decimals.",
   "by_late = ...\n\nprint(by_late)",
   "`credit.groupby('months_late')['default'].agg(['size', 'mean'])`, then "
   "`.round(3)`.",
   "by_late = credit.groupby('months_late')['default'].agg(['size', 'mean']).round(3)\n\n"
   "print(by_late)",
   f"The default rate climbs from {BY_LATE['mean'].iloc[0]:.3f} for borrowers who "
   f"were never late to {BY_LATE['mean'].iloc[-1]:.3f} for those late in all six "
   "months. There is a signal in the table, which is worth confirming before "
   "fitting anything to it.")

ex("A4", "A rare label of your own", 2,
   "On the index table, make a label for a calm month: `ratio` below 0.7, "
   "meaning next month's volatility is at least 30 percent lower than this "
   "month's. Report how many such days there are and their share. `ratio` "
   "comes from the setup cell.",
   'calm = ...\nn_calm = ...\nshare_calm = ...\n\nprint(n_calm)\nprint(share_calm)',
   "The same shape as the `jump` label in the setup cell, with `<` instead of "
   "`>` and a different number.",
   "table['calm'] = (ratio < 0.7).astype(int)\ncalm = table['calm']\nn_calm = int(calm.sum())\nshare_calm = round(float(calm.mean()), 4)\n\nprint(n_calm)\nprint(share_calm)",
   f"{N_CALM} days, or {SHARE_CALM:.1%} of them. Any comparison makes a label, "
   "and where you put the cut decides how rare it is. A cut that leaves too "
   "few ones cannot be scored at all.")

ex("A5", "The same share, counted by hand", 2,
   "Count the jumps among the test days with a `for` loop and an `if`, then "
   "check your count against the vectorised `.sum()`. Print both.",
   "count = 0\nfor value in test['jump']:\n    ...\n\n"
   "print(count)\nprint(test['jump'].sum())",
   "Inside the loop: `if value == 1:` and then `count += 1`.",
   "count = 0\nfor value in test['jump']:\n    if value == 1:\n        count += 1\n\n"
   "print(count)\nprint(test['jump'].sum())",
   f"Both print {N_JUMP_TE}. The loop is the definition and `.sum()` is the "
   "fast way to say the same thing. When a vectorised line surprises you, "
   "writing the loop once is how you find out which of you is wrong.")

ex("A6", "Shares into a dictionary", 2,
   "Build a dictionary with one entry per label, holding the share of test "
   "days on which it is 1: the keys `'rising'` and `'jump'`, the values their "
   "means. Then print the key with the smaller share.",
   "shares = {}\nfor name in ['rising', 'jump']:\n    ...\n\nsmaller = ...\n\nprint(shares)\nprint(smaller)",
   ["Inside the loop: `shares[name] = test[name].mean()`.",
    "`min(dictionary, key=dictionary.get)` gives the KEY with the smallest "
    "value, from Session 5."],
   "shares = {}\nfor name in ['rising', 'jump']:\n    shares[name] = round(float(test[name].mean()), 3)\n\nsmaller = min(shares, key=shares.get)\n\nprint(shares)\nprint(smaller)",
   f"`rising` is 1 on {SHARE_RISING_TE:.3f} of the test days and `jump` on "
   f"{SHARE_JUMP_TE:.3f}, so `jump` is the smaller. Both labels come from the "
   "same ratio, and only the cut differs.")

ex("A7", "Draw the default rate", 1,
   "Draw the default rate by `months_late` as a bar chart, with a dashed "
   "horizontal line at the rate for everyone. Label both axes.",
   "by_late = credit.groupby('months_late')['default'].mean()\n\n"
   "fig, ax = plt.subplots(figsize=(7, 3))\n...\n...\n"
   "ax.set_xlabel('months of the last six that were late')\n"
   "ax.set_ylabel('share who defaulted')\nplt.show()",
   "`ax.bar(by_late.index, by_late.values)` and "
   "`ax.axhline(credit['default'].mean(), linestyle='--', color='grey')`.",
   "by_late = credit.groupby('months_late')['default'].mean()\n\n"
   "fig, ax = plt.subplots(figsize=(7, 3))\n"
   "ax.bar(by_late.index, by_late.values)\n"
   "ax.axhline(credit['default'].mean(), linestyle='--', color='grey')\n"
   "ax.set_xlabel('months of the last six that were late')\n"
   "ax.set_ylabel('share who defaulted')\n"
   "ax.set_title('Default rate by months late', loc='left')\nplt.show()",
   f"Every bar from one month late upwards sits above the dashed line at "
   f"{SHARE_DEFAULT:.2f}, and the climb runs from {BY_LATE['mean'].iloc[0]:.2f} "
   f"to {BY_LATE['mean'].iloc[-1]:.2f}. A3 gave you these numbers; the picture "
   "is what makes the shape of them obvious at a glance.")

md("---")

# ====================================================== B
section(
"## B · Splitting rows that are not a time series\n\n"
"The credit rows are borrowers, not days, so they can be shuffled. B4 and B5 "
"run together."
)

ex("B1", "A stratified split", 2,
   "Split the credit table into 70 percent training rows and 30 percent test "
   "rows, with `random_state=0`, keeping the share of defaults equal in both "
   "halves. Print the two sizes and the two shares.",
   '...\nsizes = ...\nshares = ...\n\nprint(sizes)\nprint(shares)',
   "`train_test_split(credit, test_size=0.3, random_state=0, stratify=credit['default'])`.",
   "a, b = train_test_split(credit, test_size=0.3, random_state=0,\n                        stratify=credit['default'])\nsizes = (len(a), len(b))\nshares = (round(float(a['default'].mean()), 4), round(float(b['default'].mean()), 4))\n\nprint(sizes)\nprint(shares)",
   f"{N_CTRAIN:,} and {N_CTEST:,} rows, both with a default share of "
   f"{SHARE_TE:.4f}. This is the split the setup cell already made, so `a` and "
   "`b` hold the same borrowers as `c_train` and `c_test`.")

ex("B2", "What stratify is for", 3,
   "Take a deliberately small test set, 2 percent of the borrowers, five "
   "times with `random_state` 0 to 4, once with `stratify` and once without. "
   "Collect the default share of the test set each time and print both lists.",
   "with_strat = []\nwithout = []\nfor seed in range(5):\n"
   "    _, s = train_test_split(credit, test_size=0.02, random_state=seed,\n"
   "                            stratify=credit['default'])\n"
   "    _, u = train_test_split(credit, test_size=0.02, random_state=seed)\n"
   "    ...\n    ...\n\nprint(with_strat)\nprint(without)",
   "`with_strat.append(round(s['default'].mean(), 4))`, and the same for `u` "
   "into `without`.",
   "with_strat = []\nwithout = []\nfor seed in range(5):\n"
   "    _, s = train_test_split(credit, test_size=0.02, random_state=seed,\n"
   "                            stratify=credit['default'])\n"
   "    _, u = train_test_split(credit, test_size=0.02, random_state=seed)\n"
   "    with_strat.append(round(s['default'].mean(), 4))\n"
   "    without.append(round(u['default'].mean(), 4))\n\n"
   "print(with_strat)\nprint(without)",
   f"With `stratify` every share is {STRAT_SHARES[0]}. Without it they run from "
   f"{min(PLAIN_SHARES)} to {max(PLAIN_SHARES)}, a spread of {SPREAD_PLAIN:.4f}. "
   "On 9,000 test rows that drift is small, but on a small sample of a rare "
   "class it is large enough to move every score you report.")

ex("B3", "The split that would be wrong here", 3,
   "The index table is a time series, so it is split by date instead. Make "
   "that split, then print the two sizes and, beside them, the last training "
   "date and the first test date.",
   'early = ...\nlate = ...\nsizes = ...\nends = ...\n\nprint(sizes)\nprint(ends)',
   "`table.loc[:'2022-12-31']` and `table.loc['2023-01-01':]`, as in Session 5.",
   "early = table.loc[:'2022-12-31']\nlate = table.loc['2023-01-01':]\nsizes = (len(early), len(late))\nends = (early.index.max().date(), late.index.min().date())\n\nprint(sizes)\nprint(ends)",
   f"{N_TRAIN:,} training days and {N_TEST} test days, with every test day "
   "after every training day. A shuffled split here would let the model fit "
   "on 2024 and be scored on 2020.")

ex("B4", "Fit the credit classifier", 2,
   "Fit a scaled logistic regression on the credit training rows, using all "
   "seven columns, and store the probability of default for the test rows in "
   "`p`. Print the accuracy of `.predict()` and the baseline from A2.",
   "model = ...\n...\np = ...\naccuracy = ...\n\nprint(accuracy)\nprint(round(1 - c_test['default'].mean(), 4))",
   ["`Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])`.",
    "`model.predict_proba(c_test[c_cols])[:, 1]` is the probability of a 1."],
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\np = model.predict_proba(c_test[c_cols])[:, 1]\naccuracy = round(accuracy_score(c_test['default'], model.predict(c_test[c_cols])), 4)\n\nprint(accuracy)\nprint(round(1 - c_test['default'].mean(), 4))",
   f"{ACC_C:.4f} against {MAJORITY_C:.4f}, so the model is "
   f"{100 * (ACC_C - MAJORITY_C):.1f} percentage points better than predicting "
   "that nobody defaults. Keep `model` and `p`: B5 continues from here.")

ex("B5", "The four counts", 2,
   "Continuing from B4. Print the confusion matrix, then unpack it with "
   "`.ravel()` into the four counts and print them with labels.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\npredicted = model.predict(c_test[c_cols])\n\ncm = ...\ncounts = ...\n\nprint(cm)\nprint(counts)",
   "`confusion_matrix(c_test['default'], predicted)`, then "
   "`tn, fp, fn, tp = cm.ravel()`, and collect the four into a dictionary with "
   "`int()` around each so they print as plain numbers.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\npredicted = model.predict(c_test[c_cols])\n\ncm = confusion_matrix(c_test['default'], predicted)\ntn, fp, fn, tp = cm.ravel()\ncounts = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}\n\nprint(cm)\nprint(counts)",
   f"tn {TN_C:,}, fp {FP_C}, fn {FN_C:,}, tp {TP_C}. The model called "
   f"{TP_C + FP_C} defaults and was right about {TP_C} of them, while "
   f"{FN_C:,} borrowers defaulted without being called. `.ravel()` gives the "
   "four counts in reading order, which is the order every formula below uses.")

md("---")

# ====================================================== C
section(
"## C · Accuracy, precision and recall\n\n"
"Four counts, and the numbers built from them. Several of these rebuild by "
"hand what scikit-learn does in one call."
)

ex("C1", "Precision and recall", 1,
   "Fit the credit model, predict the test rows, and print the precision and "
   "the recall, each to three decimals.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\npredicted = model.predict(c_test[c_cols])\n\nprecision = ...\nrecall = ...\n\nprint(precision)\nprint(recall)",
   "`precision_score(c_test['default'], predicted)` and `recall_score(...)`, "
   "true labels first.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\npredicted = model.predict(c_test[c_cols])\n\nprecision = round(precision_score(c_test['default'], predicted), 3)\nrecall = round(recall_score(c_test['default'], predicted), 3)\n\nprint(precision)\nprint(recall)",
   f"Precision {PREC_C:.3f}, recall {REC_C:.3f}. Of the borrowers it called, "
   "two thirds defaulted, but it found under a third of the defaults. On a "
   "rare class those two numbers are the report, not the accuracy.")

ex("C2", "The same two, from masks", 3,
   "Compute the same precision and recall from boolean masks, with no metric "
   "function: build `said` and `was` as columns of `True` and `False`, count "
   "the three combinations you need, and divide.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\npredicted = model.predict(c_test[c_cols])\n\nsaid = predicted == 1\nwas = c_test['default'].values == 1\n\ntp = ...\nfp = ...\nfn = ...\nprecision = ...\nrecall = ...\n\nprint(precision)\nprint(recall)",
   ["`(said & was).sum()` counts the rows where both are True.",
    "A false alarm is `said & ~was`, and a miss is `~said & was`."],
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\npredicted = model.predict(c_test[c_cols])\n\nsaid = predicted == 1\nwas = c_test['default'].values == 1\n\ntp = (said & was).sum()\nfp = (said & ~was).sum()\nfn = (~said & was).sum()\nprecision = round(tp / (tp + fp), 3)\nrecall = round(tp / (tp + fn), 3)\n\nprint(precision)\nprint(recall)",
   f"The same {PREC_C:.3f} and {REC_C:.3f} as C1. Session 4 counted these by "
   "hand before any model existed. The metric functions are a shorthand for "
   "exactly this, and knowing that is what lets you debug one that surprises "
   "you.")

ex("C3", "A function for the four counts", 4,
   "Write `counts(y_true, y_pred)` returning a dictionary with the keys "
   "`'tp'`, `'fp'`, `'fn'` and `'tn'`. Give it a docstring, and test it on the "
   "credit predictions.",
   "def counts(y_true, y_pred):\n    \"\"\"...\"\"\"\n    ...\n\n\n"
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "predicted = model.predict(c_test[c_cols])\n\n"
   "print(counts(c_test['default'], predicted))",
   ["Inside the function: `tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()`.",
    "Return `{'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}`. "
    "`int()` keeps the printed dictionary readable."],
   "def counts(y_true, y_pred):\n"
   "    \"\"\"The four counts of a 0/1 prediction, as a dictionary.\"\"\"\n"
   "    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()\n"
   "    return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}\n\n\n"
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "predicted = model.predict(c_test[c_cols])\n\n"
   "print(counts(c_test['default'], predicted))",
   f"`{{'tp': {TP_C}, 'fp': {FP_C}, 'fn': {FN_C}, 'tn': {TN_C}}}`. A function "
   "that returns a dictionary is the tidy way to hand several numbers back at "
   "once, and it makes the exercises below one line each.")

ex("C4", "Recall at several thresholds", 2,
   "Loop over the thresholds 0.1, 0.2, 0.3, 0.4 and 0.5 and print each one "
   "with the recall it gives, rounded to three decimals.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "p = model.predict_proba(c_test[c_cols])[:, 1]\n\n"
   "for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:\n    called = ...\n    ...",
   "`called = (p >= threshold).astype(int)`, then "
   "`print(threshold, round(recall_score(c_test['default'], called), 3))`.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "p = model.predict_proba(c_test[c_cols])[:, 1]\n\n"
   "for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:\n"
   "    called = (p >= threshold).astype(int)\n"
   "    print(threshold, round(recall_score(c_test['default'], called), 3))",
   f"Recall falls from {BY_THR[0.1]['recall']:.3f} at 0.1 to "
   f"{BY_THR[0.5]['recall']:.3f} at 0.5. The model never changed; only the "
   "number its probabilities are compared against did.")

ex("C5", "A table that lines up", 3,
   "Print one line per threshold with the threshold, the precision and the "
   "recall, using field widths so the columns line up under the header.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "p = model.predict_proba(c_test[c_cols])[:, 1]\n\n"
   "print(f\"{'thr':>5}{'prec':>8}{'recall':>8}\")\n"
   "for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:\n"
   "    called = (p >= threshold).astype(int)\n"
   "    prec = precision_score(c_test['default'], called, zero_division=0)\n"
   "    rec = recall_score(c_test['default'], called)\n    ...",
   "`print(f\"{threshold:>5}{prec:>8.3f}{rec:>8.3f}\")`. The number after the "
   "colon is the width, and `>` right-aligns.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "p = model.predict_proba(c_test[c_cols])[:, 1]\n\n"
   "print(f\"{'thr':>5}{'prec':>8}{'recall':>8}\")\n"
   "for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:\n"
   "    called = (p >= threshold).astype(int)\n"
   "    prec = precision_score(c_test['default'], called, zero_division=0)\n"
   "    rec = recall_score(c_test['default'], called)\n"
   "    print(f\"{threshold:>5}{prec:>8.3f}{rec:>8.3f}\")",
   "Precision climbs as recall falls, and a column that lines up is the "
   "difference between seeing that and not. Field widths came up in Session 2 "
   "and are worth the extra characters every time you print inside a loop.")

ex("C6", "Walking down to a recall of one half", 4,
   "Start at a threshold of 0.5 and step down by 0.05 at a time. Stop at the "
   "first threshold whose recall reaches 0.5, and print it with its recall. "
   "Use `while` and `break`.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "p = model.predict_proba(c_test[c_cols])[:, 1]\n\n"
   "threshold = 0.5\nrec = 0.0\nwhile threshold > 0.05:\n"
   "    rec = recall_score(c_test['default'], (p >= threshold).astype(int))\n"
   "    if ...:\n        break\n    threshold = round(threshold - 0.05, 2)\n\n"
   "print(threshold, round(rec, 3))",
   "The condition is `rec >= 0.5`. The `while` is bounded, so the loop always "
   "stops even if the condition is never met.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "p = model.predict_proba(c_test[c_cols])[:, 1]\n\n"
   "threshold = 0.5\nrec = 0.0\nwhile threshold > 0.05:\n"
   "    rec = recall_score(c_test['default'], (p >= threshold).astype(int))\n"
   "    if rec >= 0.5:\n        break\n    threshold = round(threshold - 0.05, 2)\n\n"
   "print(threshold, round(rec, 3))",
   f"It stops at {FIRST_REC_HALF}, where the recall is {REC_AT_FIRST:.3f}. A "
   "`while` with a `break` is the right shape when you are looking for the "
   "first value that satisfies something, and the bound on the `while` is what "
   "stops it running forever when nothing does.")

md("---")

# ====================================================== D
section(
"## D · Thresholds and costs\n\n"
"A missed default and a false alarm are not worth the same, so the threshold "
"that makes the fewest mistakes is not the cheapest one. D3 and D4 run "
"together. Throughout: a miss costs 5, a false alarm costs 1."
)

ex("D1", "The cost of a threshold", 2,
   "Write the cost of the test decisions at a threshold of 0.5, counting 5 "
   "for every missed default and 1 for every false alarm.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "p = model.predict_proba(c_test[c_cols])[:, 1]\n\n"
   "called = (p >= 0.5).astype(int)\ntn, fp, fn, tp = confusion_matrix(c_test['default'], called).ravel()\n"
   "cost = ...\n\nprint(fn, fp, cost)",
   "A miss is `fn` and a false alarm is `fp`, so the cost is `5 * fn + fp`.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "p = model.predict_proba(c_test[c_cols])[:, 1]\n\n"
   "called = (p >= 0.5).astype(int)\n"
   "tn, fp, fn, tp = confusion_matrix(c_test['default'], called).ravel()\n"
   "cost = 5 * fn + fp\n\nprint(fn, fp, cost)",
   f"{BY_THR[0.5]['fn']:,} misses and {BY_THR[0.5]['fp']} false alarms, costing "
   f"{BY_THR[0.5]['cost']:,}. Almost all of the cost is misses, which is the "
   "clue that one half is the wrong threshold here.")

ex("D2", "Sweeping the threshold", 3,
   "Sweep the threshold from 0.05 to 0.85 in steps of 0.01 with `np.arange`, "
   "collect the cost at each one in a list, and print the cheapest threshold "
   "and its cost. Use `np.array` and `.argmin()`.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\np = model.predict_proba(c_test[c_cols])[:, 1]\n\ngrid = np.arange(0.05, 0.85, 0.01)\ncosts = []\nfor threshold in grid:\n    tn, fp, fn, tp = confusion_matrix(c_test['default'], (p >= threshold).astype(int)).ravel()\n    ...\n\ncosts = np.array(costs)\nbest = ...\ncheapest = ...\n\nprint(best)\nprint(cheapest)",
   ["Inside the loop: `costs.append(5 * fn + fp)`.",
    "`.argmin()` gives the POSITION of the smallest cost, so "
    "`best = grid[costs.argmin()]`."],
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\np = model.predict_proba(c_test[c_cols])[:, 1]\n\ngrid = np.arange(0.05, 0.85, 0.01)\ncosts = []\nfor threshold in grid:\n    tn, fp, fn, tp = confusion_matrix(c_test['default'], (p >= threshold).astype(int)).ravel()\n    costs.append(5 * fn + fp)\n\ncosts = np.array(costs)\nbest = round(float(grid[costs.argmin()]), 2)\ncheapest = int(costs.min())\n\nprint(best)\nprint(cheapest)",
   f"{BEST_THR:.2f}, costing {BEST_COST:,} against {COST_AT_HALF:,} at one "
   f"half. Moving one number saved {100 * (COST_AT_HALF - BEST_COST) / COST_AT_HALF:.0f} "
   "percent of the cost, with no change to the model at all.")

ex("D3", "The threshold from the formula", 2,
   "The cheapest threshold can be written down without a sweep: the cost of a "
   "false alarm divided by the two costs added together. Compute it for a "
   "miss of 5 and a false alarm of 1, and print the cost it gives.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\np = model.predict_proba(c_test[c_cols])[:, 1]\ncost_miss = 5\ncost_false = 1\n\nthreshold = ...\ncost_here = ...\n\nprint(threshold)\nprint(cost_here)",
   "`cost_false / (cost_false + cost_miss)`, which is one sixth.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\np = model.predict_proba(c_test[c_cols])[:, 1]\ncost_miss = 5\ncost_false = 1\n\nthreshold = cost_false / (cost_false + cost_miss)\ntn, fp, fn, tp = confusion_matrix(c_test['default'], (p >= threshold).astype(int)).ravel()\ncost_here = 5 * fn + fp\n\nprint(round(threshold, 3))\nprint(cost_here)",
   f"{FORMULA_THR:.3f}, costing {COST_ONE_SIXTH['plain']:,} against the "
   f"{BEST_COST:,} the sweep found. The formula lands within half a percent of "
   "the best available, and it needs no test rows to find it. D4 continues "
   "from here.")

ex("D4", "Choosing it on the training rows", 4,
   "Continuing from D3. The test rows are not available when the threshold is "
   "chosen, so sweep on the **training** rows instead, then report what that "
   "threshold costs on the test rows.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\np_train = model.predict_proba(c_train[c_cols])[:, 1]\np = model.predict_proba(c_test[c_cols])[:, 1]\ngrid = np.arange(0.05, 0.85, 0.01)\n\nchosen = ...\ncost_on_test = ...\n\nprint(chosen)\nprint(cost_on_test)",
   "`p_train = model.predict_proba(c_train[c_cols])[:, 1]`. Everything else "
   "is the sweep from D2 with the training label.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\np_train = model.predict_proba(c_train[c_cols])[:, 1]\np = model.predict_proba(c_test[c_cols])[:, 1]\ngrid = np.arange(0.05, 0.85, 0.01)\n\ncosts = []\nfor threshold in grid:\n    tn, fp, fn, tp = confusion_matrix(c_train['default'], (p_train >= threshold).astype(int)).ravel()\n    costs.append(5 * fn + fp)\n\nchosen = round(float(grid[np.array(costs).argmin()]), 2)\ntn, fp, fn, tp = confusion_matrix(c_test['default'], (p >= chosen).astype(int)).ravel()\ncost_on_test = 5 * fn + fp\n\nprint(chosen)\nprint(cost_on_test)",
   f"The training rows choose {TRAIN_THR:.2f}, which costs "
   f"{COST_AT_TRAIN_THR:,} on the test rows against the {BEST_COST:,} the test "
   "rows would have chosen for themselves. The threshold is a setting like "
   "`C`: chosen where the model may look, reported where it may not.")

ex("D5", "A cost function with a default", 4,
   "Write `total_cost(y_true, p, threshold, cost_miss=5, cost_false=1)` "
   "returning the cost as a plain integer. Call it twice: once with the "
   "defaults, once with a miss costing 20.",
   "def total_cost(y_true, p, threshold, cost_miss=5, cost_false=1):\n"
   "    \"\"\"...\"\"\"\n    ...\n\n\n"
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "p = model.predict_proba(c_test[c_cols])[:, 1]\n\n"
   "print(total_cost(c_test['default'], p, 0.5))\n"
   "print(total_cost(c_test['default'], p, 0.5, cost_miss=20))",
   ["Inside: build `called`, unpack the four counts with `.ravel()`, and "
    "return `int(cost_miss * fn + cost_false * fp)`.",
    "A parameter with an `=` in the `def` line is a default: the caller may "
    "leave it out."],
   "def total_cost(y_true, p, threshold, cost_miss=5, cost_false=1):\n"
   "    \"\"\"What a set of 0/1 decisions at this threshold costs.\"\"\"\n"
   "    called = (p >= threshold).astype(int)\n"
   "    tn, fp, fn, tp = confusion_matrix(y_true, called).ravel()\n"
   "    return int(cost_miss * fn + cost_false * fp)\n\n\n"
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "p = model.predict_proba(c_test[c_cols])[:, 1]\n\n"
   "print(total_cost(c_test['default'], p, 0.5))\n"
   "print(total_cost(c_test['default'], p, 0.5, cost_miss=20))",
   f"{BY_THR[0.5]['cost']:,} with the defaults and "
   f"{20 * BY_THR[0.5]['fn'] + BY_THR[0.5]['fp']:,} when a miss costs 20. "
   "Defaults let the common case stay short while the unusual one is still "
   "reachable, which is exactly how scikit-learn's own arguments work.")

ex("D6", "Which costs make one half right", 2,
   "The formula says the cheapest threshold is the cost of a false alarm over "
   "the two costs added together. Work out on paper what the two costs must "
   "be for that to give 0.5, then check your answer in code with any pair of "
   "numbers that fits.",
   'cost_miss = ...\ncost_false = ...\nthreshold = ...\n\nprint(threshold)',
   "For the fraction to be one half, the numerator has to be half the "
   "denominator, so the two costs must be equal.",
   'cost_miss = 1\ncost_false = 1\nthreshold = cost_false / (cost_false + cost_miss)\n\nprint(threshold)',
   "The two costs have to be equal, and any equal pair works. The default "
   "threshold of one half is not a neutral choice: it is the claim that a "
   "missed default and a wrongly refused customer cost the bank the same, "
   "which almost nobody believes.")

ex("D7", "Draw the cost curve", 2,
   "Draw the cost of the test decisions against the threshold, with a vertical "
   "line at the cheapest threshold. The sweep is the one from D2.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "p = model.predict_proba(c_test[c_cols])[:, 1]\n\n"
   "grid = np.arange(0.05, 0.85, 0.01)\ncosts = []\nfor threshold in grid:\n"
   "    tn, fp, fn, tp = confusion_matrix(c_test['default'], (p >= threshold).astype(int)).ravel()\n"
   "    costs.append(5 * fn + fp)\n\n"
   "fig, ax = plt.subplots(figsize=(7, 3))\n...\n...\n"
   "ax.set_xlabel('threshold')\nax.set_ylabel('cost')\nplt.show()",
   "`ax.plot(grid, costs)`, then `ax.axvline(grid[np.array(costs).argmin()], "
   "linestyle='--', color='grey')`.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "p = model.predict_proba(c_test[c_cols])[:, 1]\n\n"
   "grid = np.arange(0.05, 0.85, 0.01)\ncosts = []\nfor threshold in grid:\n"
   "    tn, fp, fn, tp = confusion_matrix(c_test['default'], (p >= threshold).astype(int)).ravel()\n"
   "    costs.append(5 * fn + fp)\n\n"
   "fig, ax = plt.subplots(figsize=(7, 3))\nax.plot(grid, costs)\n"
   "ax.axvline(grid[np.array(costs).argmin()], linestyle='--', color='grey')\n"
   "ax.set_xlabel('threshold')\nax.set_ylabel('cost')\n"
   "ax.set_title('Cost of the 9,000 test decisions', loc='left')\nplt.show()",
   f"The curve falls steeply, turns at {BEST_THR:.2f} and climbs slowly after "
   "it. The shape matters as much as the minimum: anywhere between about 0.15 "
   "and 0.3 costs nearly the same, so the exact threshold does not need "
   "defending to two decimals.")

md("---")

# ====================================================== E
section(
"## E · Weighting the rare class, and the PR curve\n\n"
"A second way to move the cut, and the curve that reads a rare class properly."
)

ex("E1", "class_weight balanced", 2,
   "Fit the same pipeline with `class_weight='balanced'` on the classifier, "
   "then add it to the list in the loop so both models are reported.",
   "plain = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nplain.fit(c_train[c_cols], c_train['default'])\n\nweighted = ...\n...\n\nfor name, m in [('plain', plain)]:\n    pred = m.predict(c_test[c_cols])\n    print(name, round(accuracy_score(c_test['default'], pred), 4),\n          round(recall_score(c_test['default'], pred), 3))",
   "`Pipeline([('scale', StandardScaler()), ('logit', "
   "LogisticRegression(class_weight='balanced'))])`.",
   "plain = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nplain.fit(c_train[c_cols], c_train['default'])\n\nweighted = Pipeline([('scale', StandardScaler()),\n                     ('logit', LogisticRegression(class_weight='balanced'))])\nweighted.fit(c_train[c_cols], c_train['default'])\n\nfor name, m in [('plain', plain), ('weighted', weighted)]:\n    pred = m.predict(c_test[c_cols])\n    print(name, round(accuracy_score(c_test['default'], pred), 4),\n          round(recall_score(c_test['default'], pred), 3))",
   f"Accuracy falls from {ACC_C:.4f} to {ACC_B:.4f}, which is below the "
   f"{MAJORITY_C:.4f} of predicting that nobody defaults, while recall rises "
   f"from {REC_C:.3f} to {REC_B:.3f}. The model got worse by one number and "
   "more useful by another.")

ex("E2", "What the weights did not change", 2,
   "Print the AUC of both models. Before running it, decide whether you "
   "expect them to differ.",
   "plain = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nplain.fit(c_train[c_cols], c_train['default'])\nweighted = Pipeline([('scale', StandardScaler()),\n                     ('logit', LogisticRegression(class_weight='balanced'))])\nweighted.fit(c_train[c_cols], c_train['default'])\n\nauc_plain = ...\nauc_weighted = ...\n\nprint(auc_plain)\nprint(auc_weighted)",
   "`roc_auc_score(c_test['default'], plain.predict_proba(c_test[c_cols])[:, 1])`, "
   "and the same for `weighted`.",
   "plain = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nplain.fit(c_train[c_cols], c_train['default'])\nweighted = Pipeline([('scale', StandardScaler()),\n                     ('logit', LogisticRegression(class_weight='balanced'))])\nweighted.fit(c_train[c_cols], c_train['default'])\n\nauc_plain = round(roc_auc_score(c_test['default'],\n                                plain.predict_proba(c_test[c_cols])[:, 1]), 4)\nauc_weighted = round(roc_auc_score(c_test['default'],\n                                   weighted.predict_proba(c_test[c_cols])[:, 1]), 4)\n\nprint(auc_plain)\nprint(auc_weighted)",
   f"{AUC_C:.4f} and {AUC_B:.4f}, which are the same to three decimals. AUC "
   "reads the order of the borrowers, and the weights did not reorder anyone. "
   "They moved where the cut falls, which is something a threshold can do "
   "too, and more precisely.")

ex("E3", "Average precision against the base rate", 2,
   "Print the average precision of the plain model's probabilities and, "
   "beside it, the share of test borrowers who defaulted. The second is what "
   "guessing at random would score.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\np = model.predict_proba(c_test[c_cols])[:, 1]\n\naverage_precision = ...\n\nprint(average_precision)\nprint(round(c_test['default'].mean(), 4))",
   "`average_precision_score(c_test['default'], p)`, with the probabilities "
   "and not the predictions.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\np = model.predict_proba(c_test[c_cols])[:, 1]\n\naverage_precision = round(average_precision_score(c_test['default'], p), 4)\n\nprint(average_precision)\nprint(round(c_test['default'].mean(), 4))",
   f"{AP_C:.4f} against {SHARE_TE:.4f}. Where an AUC of 0.5 is the "
   "no-information line, the no-information line for average precision is the "
   "share of ones, which moves with how rare the class is. Always report the "
   "two together.")

ex("E4", "Precision at a recall of 0.6", 3,
   "`precision_recall_curve(y, p)` returns three arrays: the precision, the "
   "recall, and the thresholds. Use it to find the precision at the point "
   "where recall is closest to 0.6. Import it yourself.",
   "from sklearn.metrics import precision_recall_curve\n\nmodel = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\np = model.predict_proba(c_test[c_cols])[:, 1]\n\nprec, rec, thresholds = precision_recall_curve(c_test['default'], p)\nspot = ...\nrecall_here = ...\nprecision_here = ...\n\nprint(recall_here)\nprint(precision_here)",
   ["`np.abs(rec - 0.6)` is how far each recall is from 0.6.",
    "`.argmin()` gives the position of the smallest distance, so "
    "`spot = np.abs(rec - 0.6).argmin()`."],
   "from sklearn.metrics import precision_recall_curve\n\nmodel = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\np = model.predict_proba(c_test[c_cols])[:, 1]\n\nprec, rec, thresholds = precision_recall_curve(c_test['default'], p)\nspot = np.abs(rec - 0.6).argmin()\nrecall_here = round(float(rec[spot]), 3)\nprecision_here = round(float(prec[spot]), 3)\n\nprint(recall_here)\nprint(precision_here)",
   f"At a recall of about 0.6 the precision is {PREC_AT_REC60:.3f}: catching "
   "three defaults in five means being wrong about rather more than half the "
   "borrowers you call. That trade is the decision, and the curve is how you "
   "see all of it at once.")

ex("E5", "Draw the curve", 3,
   "Plot precision against recall from E4, with axis labels and a horizontal "
   "line at the share of ones. Three steps, as always: make the axes, draw, "
   "then say what the reader is looking at.",
   "from sklearn.metrics import precision_recall_curve\n\n"
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "p = model.predict_proba(c_test[c_cols])[:, 1]\n"
   "prec, rec, thresholds = precision_recall_curve(c_test['default'], p)\n\n"
   "fig, ax = plt.subplots(figsize=(6, 3.4))\n...\n...\n"
   "ax.set_xlabel('recall')\nax.set_ylabel('precision')\nax.set_ylim(0, 1)\nplt.show()",
   "`ax.plot(rec, prec)` and `ax.axhline(c_test['default'].mean(), "
   "linestyle='--', color='grey')`.",
   "from sklearn.metrics import precision_recall_curve\n\n"
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "p = model.predict_proba(c_test[c_cols])[:, 1]\n"
   "prec, rec, thresholds = precision_recall_curve(c_test['default'], p)\n\n"
   "fig, ax = plt.subplots(figsize=(6, 3.4))\nax.plot(rec, prec)\n"
   "ax.axhline(c_test['default'].mean(), linestyle='--', color='grey')\n"
   "ax.set_xlabel('recall')\nax.set_ylabel('precision')\nax.set_ylim(0, 1)\n"
   "ax.set_title('Precision against recall', loc='left')\nplt.show()",
   "The curve starts high and falls away as recall rises, ending at the "
   "dashed line, which is where calling every borrower a default would put "
   "you. The area under it is the average precision from E3.")

md("---")

# ====================================================== F
section(
"## F · Calibration\n\n"
"Whether a probability of 0.3 means 0.3. F3 to F5 run together."
)

ex("F1", "Bucket the probabilities", 2,
   "Put the test borrowers into four buckets by their predicted probability, "
   "with the edges 0, 0.2, 0.4, 0.6 and 1.0, and report how many borrowers "
   "are in each and what share of them defaulted.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "checked = c_test.copy()\nchecked['p'] = model.predict_proba(c_test[c_cols])[:, 1]\n\n"
   "checked['bucket'] = ...\nby_bucket = ...\n\nprint(by_bucket)",
   ["`pd.cut(checked['p'], [0, 0.2, 0.4, 0.6, 1.0])`.",
    "`checked.groupby('bucket', observed=True)['default'].agg(['size', 'mean']).round(3)`."],
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "checked = c_test.copy()\nchecked['p'] = model.predict_proba(c_test[c_cols])[:, 1]\n\n"
   "checked['bucket'] = pd.cut(checked['p'], [0, 0.2, 0.4, 0.6, 1.0])\n"
   "by_bucket = checked.groupby('bucket', observed=True)['default'].agg(['size', 'mean']).round(3)\n\n"
   "print(by_bucket)",
   "Each bucket's share falls inside its own range: " +
   ", ".join(f"{BUCKETS['mean'].iloc[i]:.3f}" for i in range(len(BUCKETS))) +
   ". These numbers can be read as probabilities, which is what calibrated "
   "means.")

ex("F2", "The average against what happened", 1,
   "Print the average probability the model gave the test borrowers and the "
   "share of them that actually defaulted.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\np = model.predict_proba(c_test[c_cols])[:, 1]\n\nsaid = ...\nhappened = ...\n\nprint(said)\nprint(happened)",
   "`p.mean()` and `c_test['default'].mean()`.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nmodel.fit(c_train[c_cols], c_train['default'])\np = model.predict_proba(c_test[c_cols])[:, 1]\n\nsaid = round(p.mean(), 4)\nhappened = round(c_test['default'].mean(), 4)\n\nprint(said)\nprint(happened)",
   f"{MEAN_PC:.4f} against {SHARE_TE:.4f}. This is the cheapest calibration "
   "check there is, and it is the first thing to run when a probability is "
   "about to be multiplied by an amount of money.")

ex("F3", "The Brier score, both models", 2,
   "Print the Brier score of the plain model and of the weighted one. Keep "
   "both fitted models: F4 and F5 continue from here.",
   "plain = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nplain.fit(c_train[c_cols], c_train['default'])\nweighted = Pipeline([('scale', StandardScaler()),\n                     ('logit', LogisticRegression(class_weight='balanced'))])\nweighted.fit(c_train[c_cols], c_train['default'])\n\nbrier_plain = ...\nbrier_weighted = ...\n\nprint(brier_plain)\nprint(brier_weighted)",
   "`brier_score_loss(c_test['default'], plain.predict_proba(c_test[c_cols])[:, 1])`.",
   "plain = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nplain.fit(c_train[c_cols], c_train['default'])\nweighted = Pipeline([('scale', StandardScaler()),\n                     ('logit', LogisticRegression(class_weight='balanced'))])\nweighted.fit(c_train[c_cols], c_train['default'])\n\nbrier_plain = round(brier_score_loss(c_test['default'],\n                                     plain.predict_proba(c_test[c_cols])[:, 1]), 4)\nbrier_weighted = round(brier_score_loss(c_test['default'],\n                                        weighted.predict_proba(c_test[c_cols])[:, 1]), 4)\n\nprint(brier_plain)\nprint(brier_weighted)",
   f"{BRIER_C:.4f} for the plain model and {BRIER_B:.4f} for the weighted one. "
   "The AUC could not tell them apart, and this can: the weighted model says "
   f"{MEAN_PB:.2f} on average where {SHARE_TE:.2f} happens.")

ex("F4", "Put the level back", 4,
   "Continuing from F3. Wrap the weighted pipeline in `CalibratedClassifierCV` "
   "with `method='sigmoid'` and `cv=5`, fit it on the training rows, and print "
   "the average probability and the Brier score.",
   "weighted = Pipeline([('scale', StandardScaler()),\n                     ('logit', LogisticRegression(class_weight='balanced'))])\n\nfixed = ...\n...\np_fixed = ...\n\nprint(p_fixed)",
   "`CalibratedClassifierCV(weighted, method='sigmoid', cv=5)`. Pass the "
   "unfitted pipeline: it refits inside on the folds.",
   "weighted = Pipeline([('scale', StandardScaler()),\n                     ('logit', LogisticRegression(class_weight='balanced'))])\n\nfixed = CalibratedClassifierCV(weighted, method='sigmoid', cv=5)\nfixed.fit(c_train[c_cols], c_train['default'])\np_fixed = fixed.predict_proba(c_test[c_cols])[:, 1]\n\nprint(round(p_fixed.mean(), 4))\nprint(round(brier_score_loss(c_test['default'], p_fixed), 4))",
   f"The average comes back to {MEAN_CAL:.4f} and the Brier score to "
   f"{BRIER_CAL:.4f}, which is the plain model's. The ranking was never the "
   "problem, so calibrating did not need to fix it.")

ex("F5", "What calibration is worth", 5,
   "Continuing from F4. Apply the cost threshold of one sixth to all three "
   "sets of probabilities, and print the cost of each.",
   "plain = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "plain.fit(c_train[c_cols], c_train['default'])\n"
   "weighted = Pipeline([('scale', StandardScaler()),\n"
   "                     ('logit', LogisticRegression(class_weight='balanced'))])\n"
   "weighted.fit(c_train[c_cols], c_train['default'])\n"
   "fixed = CalibratedClassifierCV(Pipeline([('scale', StandardScaler()),\n"
   "    ('logit', LogisticRegression(class_weight='balanced'))]), method='sigmoid', cv=5)\n"
   "fixed.fit(c_train[c_cols], c_train['default'])\n\n"
   "for name, m in [('plain', plain), ('weighted', weighted), ('fixed', fixed)]:\n"
   "    prob = m.predict_proba(c_test[c_cols])[:, 1]\n"
   "    tn, fp, fn, tp = confusion_matrix(c_test['default'], (prob >= 1 / 6).astype(int)).ravel()\n"
   "    ...",
   "`print(name, 5 * fn + fp)` inside the loop.",
   "plain = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "plain.fit(c_train[c_cols], c_train['default'])\n"
   "weighted = Pipeline([('scale', StandardScaler()),\n"
   "                     ('logit', LogisticRegression(class_weight='balanced'))])\n"
   "weighted.fit(c_train[c_cols], c_train['default'])\n"
   "fixed = CalibratedClassifierCV(Pipeline([('scale', StandardScaler()),\n"
   "    ('logit', LogisticRegression(class_weight='balanced'))]), method='sigmoid', cv=5)\n"
   "fixed.fit(c_train[c_cols], c_train['default'])\n\n"
   "for name, m in [('plain', plain), ('weighted', weighted), ('fixed', fixed)]:\n"
   "    prob = m.predict_proba(c_test[c_cols])[:, 1]\n"
   "    tn, fp, fn, tp = confusion_matrix(c_test['default'], (prob >= 1 / 6).astype(int)).ravel()\n"
   "    print(name, 5 * fn + fp)",
   f"{COST_ONE_SIXTH['plain']:,} for the plain model, "
   f"{COST_ONE_SIXTH['balanced']:,} for the weighted one and "
   f"{COST_ONE_SIXTH['calibrated']:,} once it is calibrated. The threshold "
   "from the cost formula is a statement about probabilities, so it only "
   "works on probabilities that mean what they say.")

ex("F6", "Said against happened", 3,
   "Draw the calibration check: the average probability in each bucket on the "
   "horizontal axis, the share that defaulted on the vertical, and a dashed "
   "diagonal from (0, 0) to (1, 1) to compare against.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "checked = c_test.copy()\nchecked['p'] = model.predict_proba(c_test[c_cols])[:, 1]\n"
   "checked['bucket'] = pd.cut(checked['p'], [0, 0.2, 0.4, 0.6, 1.0])\n"
   "said = checked.groupby('bucket', observed=True)['p'].mean()\n"
   "happened = checked.groupby('bucket', observed=True)['default'].mean()\n\n"
   "fig, ax = plt.subplots(figsize=(4.2, 4))\n...\n...\n"
   "ax.set_xlabel('the model said')\nax.set_ylabel('what happened')\nplt.show()",
   "`ax.plot([0, 1], [0, 1], linestyle='--', color='grey')` for the diagonal, "
   "then `ax.plot(said, happened, 'o-')` for the buckets.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "checked = c_test.copy()\nchecked['p'] = model.predict_proba(c_test[c_cols])[:, 1]\n"
   "checked['bucket'] = pd.cut(checked['p'], [0, 0.2, 0.4, 0.6, 1.0])\n"
   "said = checked.groupby('bucket', observed=True)['p'].mean()\n"
   "happened = checked.groupby('bucket', observed=True)['default'].mean()\n\n"
   "fig, ax = plt.subplots(figsize=(4.2, 4))\n"
   "ax.plot([0, 1], [0, 1], linestyle='--', color='grey')\n"
   "ax.plot(said, happened, 'o-')\n"
   "ax.set_xlabel('the model said')\nax.set_ylabel('what happened')\n"
   "ax.set_title('Four buckets of test borrowers', loc='left')\nplt.show()",
   "The four points sit close to the diagonal. Run the same plot on the "
   "weighted model from F3 and every point drops well below it, which is what "
   "a model that overstates its probabilities looks like.")

md("---")

# ====================================================== G
section(
"## G · Three classes\n\n"
"Back to the index table, with `move`: `falls`, `stays` or `rises`. G3 to G7 "
"run together on the same fitted model."
)

ex("G1", "A three-way label of your own", 2,
   "Make a second three-way label with tighter cuts, 0.9 and 1.1, called "
   "`move2`, and print how many training days fall in each class. `ratio` is "
   "in the setup cell.",
   "table['move2'] = ...\n\nprint(table.loc[:'2022-12-31', 'move2'].value_counts())",
   "`pd.cut(ratio, [-np.inf, 0.9, 1.1, np.inf], labels=['falls', 'stays', 'rises'])`.",
   "table['move2'] = pd.cut(ratio, [-np.inf, 0.9, 1.1, np.inf],\n"
   "                        labels=['falls', 'stays', 'rises'])\n\n"
   "print(table.loc[:'2022-12-31', 'move2'].value_counts())",
   "The middle class is much smaller than with the lecture's cuts of 0.85 and "
   "1.25, because the band is narrower. Where the cuts go decides how big "
   "each class is, and a class that is too small cannot be learned or scored.")

ex("G2", "The rule to beat, with three classes", 2,
   "Find the most common `move` among the training days, then report the "
   "share of TEST days that carry that same label. That is the baseline.",
   'most_common = ...\nbaseline = ...\n\nprint(most_common)\nprint(baseline)',
   ["`train['move'].value_counts().idxmax()` gives the most common label.",
    "`(test['move'] == most_common).mean()` is the share of test days it gets right."],
   "most_common = train['move'].value_counts().idxmax()\nbaseline = round((test['move'] == most_common).mean(), 4)\n\nprint(most_common)\nprint(baseline)",
   f"`{MAJ_MOVE}` is the most common training label, and it is right on "
   f"{MAJ_MOVE_ACC:.4f} of the test days. With three classes the baseline is "
   "lower than with two, because guessing is harder, so a model has more room "
   "to look good without being good.")

ex("G3", "Fit three classes", 2,
   "Fit a scaled logistic regression on `vol_20d` and `ret_20d` predicting "
   "`move`, with `max_iter=1000`, and print its accuracy on the test days. "
   "Keep the model: G4 to G7 continue from here.",
   'three = ...\n...\naccuracy = ...\n\nprint(accuracy)',
   "The pipeline is the usual one; only the label changed. "
   "`LogisticRegression(max_iter=1000)` inside it.",
   "three = Pipeline([('scale', StandardScaler()),\n                  ('logit', LogisticRegression(max_iter=1000))])\nthree.fit(train[['vol_20d', 'ret_20d']], train['move'])\naccuracy = round(accuracy_score(test['move'],\n                                three.predict(test[['vol_20d', 'ret_20d']])), 4)\n\nprint(accuracy)",
   f"{ACC_MOVE:.4f}, against {MAJ_MOVE_ACC:.4f} for the baseline in G2. "
   "Nothing in the call changed: `LogisticRegression` saw three values in the "
   "label and fitted three classes.")

ex("G4", "What the fitted model grew", 2,
   "Continuing from G3. Print `classes_`, the shape of `coef_` and the shape "
   "of `intercept_`. Reach inside the pipeline with `named_steps`.",
   "three = Pipeline([('scale', StandardScaler()),\n                  ('logit', LogisticRegression(max_iter=1000))])\nthree.fit(train[['vol_20d', 'ret_20d']], train['move'])\n\nclasses = ...\ncoef_shape = ...\nintercept_shape = ...\n\nprint(classes)\nprint(coef_shape)\nprint(intercept_shape)",
   "`three.named_steps['logit']`, as in Session 6.",
   "three = Pipeline([('scale', StandardScaler()),\n                  ('logit', LogisticRegression(max_iter=1000))])\nthree.fit(train[['vol_20d', 'ret_20d']], train['move'])\n\nlogit = three.named_steps['logit']\nclasses = logit.classes_\ncoef_shape = logit.coef_.shape\nintercept_shape = logit.intercept_.shape\n\nprint(classes)\nprint(coef_shape)\nprint(intercept_shape)",
   f"`{list(MC_LOGIT.classes_)}`, `coef_` of shape "
   f"{MC_LOGIT.coef_.shape} and `intercept_` of shape "
   f"{MC_LOGIT.intercept_.shape}. The classes come back in alphabetical "
   "order, not the order the cuts were written in, and `coef_` has one row per "
   "class rather than one row in total.")

ex("G5", "Three scores for one day", 3,
   "Continuing from G3. Take the first test day and print the three raw "
   "scores from `decision_function`, rounded to three decimals, beside "
   "`classes_` so you can tell which is which.",
   "three = Pipeline([('scale', StandardScaler()),\n                  ('logit', LogisticRegression(max_iter=1000))])\nthree.fit(train[['vol_20d', 'ret_20d']], train['move'])\nday = test[['vol_20d', 'ret_20d']].head(1)\n\nscores = ...\n\nprint(three.named_steps['logit'].classes_)\nprint(scores)",
   "`three.decision_function(day)[0]`. The `[0]` takes the single row out of "
   "the result.",
   "three = Pipeline([('scale', StandardScaler()),\n                  ('logit', LogisticRegression(max_iter=1000))])\nthree.fit(train[['vol_20d', 'ret_20d']], train['move'])\nday = test[['vol_20d', 'ret_20d']].head(1)\n\nscores = three.decision_function(day)[0].round(3)\n\nprint(three.named_steps['logit'].classes_)\nprint(scores)",
   f"`{np.round(SCORES, 3).tolist()}`. These are not probabilities: one is "
   "negative, and they do not add to one. G6 turns them into probabilities.")

ex("G6", "The softmax, by hand", 4,
   "Continuing from G5. Turn the three scores into three probabilities with "
   "the exponential and a division, then check your answer against "
   "`predict_proba`.",
   "three = Pipeline([('scale', StandardScaler()),\n                  ('logit', LogisticRegression(max_iter=1000))])\nthree.fit(train[['vol_20d', 'ret_20d']], train['move'])\nday = test[['vol_20d', 'ret_20d']].head(1)\nscores = three.decision_function(day)[0]\n\nraised = ...\nprobabilities = ...\n\nprint(probabilities)\nprint(three.predict_proba(day).round(3))",
   ["`np.exp(scores)` makes every number positive.",
    "Divide by their total: `raised / raised.sum()`."],
   "three = Pipeline([('scale', StandardScaler()),\n                  ('logit', LogisticRegression(max_iter=1000))])\nthree.fit(train[['vol_20d', 'ret_20d']], train['move'])\nday = test[['vol_20d', 'ret_20d']].head(1)\nscores = three.decision_function(day)[0]\n\nraised = np.exp(scores)\nprobabilities = (raised / raised.sum()).round(3)\n\nprint(probabilities)\nprint(three.predict_proba(day).round(3))",
   f"Both print `{np.round(PROBS, 3).tolist()}`. The exponential turns "
   f"{np.round(SCORES, 3).tolist()} into {np.round(EXPS, 3).tolist()}, which "
   f"total {EXPS.sum():.3f}, and dividing by that total makes them add to one. "
   "With two classes the same two steps are the sigmoid.")

ex("G7", "The confusion matrix, in a readable order", 4,
   "Continuing from G3. Print the confusion matrix with the rows and columns "
   "in the order `falls`, `stays`, `rises` rather than alphabetically, and "
   "count how many days were wrong by two steps.",
   "three = Pipeline([('scale', StandardScaler()),\n"
   "                  ('logit', LogisticRegression(max_iter=1000))])\n"
   "three.fit(train[['vol_20d', 'ret_20d']], train['move'])\n"
   "predicted = three.predict(test[['vol_20d', 'ret_20d']])\norder = ['falls', 'stays', 'rises']\n\n"
   "cm = ...\ntwo_steps = ...\n\nprint(cm)\nprint(two_steps)",
   ["`confusion_matrix(test['move'], predicted, labels=order)`.",
    "A two-step error is a fall called a rise or a rise called a fall: "
    "`cm[0, 2] + cm[2, 0]`."],
   "three = Pipeline([('scale', StandardScaler()),\n"
   "                  ('logit', LogisticRegression(max_iter=1000))])\n"
   "three.fit(train[['vol_20d', 'ret_20d']], train['move'])\n"
   "predicted = three.predict(test[['vol_20d', 'ret_20d']])\n"
   "order = ['falls', 'stays', 'rises']\n\n"
   "cm = confusion_matrix(test['move'], predicted, labels=order)\n"
   "two_steps = cm[0, 2] + cm[2, 0]\n\nprint(cm)\nprint(two_steps)",
   f"{TWO_STEP} of the {N_TEST} test days are wrong by two steps. Without "
   "`labels=` the matrix comes back alphabetically, with `rises` in the "
   "middle, and the diagonal then means something quite different from what "
   "you expect.")

ex("G8", "One score per class", 2,
   "Print `classification_report` for the three-class model, with the same "
   "order and three decimals, then print the macro and weighted F1 separately.",
   "three = Pipeline([('scale', StandardScaler()),\n                  ('logit', LogisticRegression(max_iter=1000))])\nthree.fit(train[['vol_20d', 'ret_20d']], train['move'])\npredicted = three.predict(test[['vol_20d', 'ret_20d']])\norder = ['falls', 'stays', 'rises']\n\nprint(classification_report(test['move'], predicted, labels=order, digits=3))\n\nmacro = ...\nweighted = ...\n\nprint(macro)\nprint(weighted)",
   "`f1_score(test['move'], predicted, average='macro')` and the same with "
   "`average='weighted'`.",
   "three = Pipeline([('scale', StandardScaler()),\n                  ('logit', LogisticRegression(max_iter=1000))])\nthree.fit(train[['vol_20d', 'ret_20d']], train['move'])\npredicted = three.predict(test[['vol_20d', 'ret_20d']])\norder = ['falls', 'stays', 'rises']\n\nprint(classification_report(test['move'], predicted, labels=order, digits=3))\n\nmacro = round(f1_score(test['move'], predicted, average='macro'), 3)\nweighted = round(f1_score(test['move'], predicted, average='weighted'), 3)\n\nprint(macro)\nprint(weighted)",
   f"Macro {F1_MACRO:.3f}, weighted {F1_WEIGHTED:.3f}. Recall is "
   f"{REC_EACH[0]:.3f} for `falls` and {REC_EACH[2]:.3f} for `rises`, but only "
   f"{REC_EACH[1]:.3f} for `stays`: the model finds the two ends and struggles "
   "in the middle. One accuracy would have hidden that completely.")

ex("G9", "Draw the recall per class", 2,
   "Draw the recall of each of the three classes as a bar chart, in the order "
   "`falls`, `stays`, `rises`, with a dashed line at the accuracy of the whole "
   "model.",
   "three = Pipeline([('scale', StandardScaler()),\n"
   "                  ('logit', LogisticRegression(max_iter=1000))])\n"
   "three.fit(train[['vol_20d', 'ret_20d']], train['move'])\n"
   "predicted = three.predict(test[['vol_20d', 'ret_20d']])\n"
   "order = ['falls', 'stays', 'rises']\n\n"
   "from sklearn.metrics import recall_score\n"
   "recalls = recall_score(test['move'], predicted, average=None, labels=order)\n\n"
   "fig, ax = plt.subplots(figsize=(5, 3))\n...\n...\n"
   "ax.set_ylabel('recall')\nax.set_ylim(0, 1)\nplt.show()",
   "`ax.bar(order, recalls)` and `ax.axhline(accuracy_score(test['move'], "
   "predicted), linestyle='--', color='grey')`.",
   "three = Pipeline([('scale', StandardScaler()),\n"
   "                  ('logit', LogisticRegression(max_iter=1000))])\n"
   "three.fit(train[['vol_20d', 'ret_20d']], train['move'])\n"
   "predicted = three.predict(test[['vol_20d', 'ret_20d']])\n"
   "order = ['falls', 'stays', 'rises']\n\n"
   "from sklearn.metrics import recall_score\n"
   "recalls = recall_score(test['move'], predicted, average=None, labels=order)\n\n"
   "fig, ax = plt.subplots(figsize=(5, 3))\nax.bar(order, recalls)\n"
   "ax.axhline(accuracy_score(test['move'], predicted), linestyle='--', color='grey')\n"
   "ax.set_ylabel('recall')\nax.set_ylim(0, 1)\n"
   "ax.set_title('Recall by class', loc='left')\nplt.show()",
   f"Two bars above the line and one well below it: {REC_EACH[0]:.2f} for "
   f"`falls`, {REC_EACH[2]:.2f} for `rises` and {REC_EACH[1]:.2f} for `stays`. "
   "The single accuracy is the dashed line, and it describes none of the three.")

md("---")

# ====================================================== H
section(
"## H · k-nearest neighbours\n\n"
"A classifier with no coefficients. H3 and H4 run together."
)

ex("H1", "The four lines, with a different model", 2,
   "Fit `KNeighborsClassifier` with 15 neighbours inside a scaled pipeline, "
   "on `vol_20d` and `ret_20d` predicting `rising`, and print the test AUC.",
   'neighbours = ...\n...\np_knn = ...\nauc = ...\n\nprint(auc)',
   "`Pipeline([('scale', StandardScaler()), ('knn', "
   "KNeighborsClassifier(n_neighbors=15))])`.",
   "neighbours = Pipeline([('scale', StandardScaler()),\n                       ('knn', KNeighborsClassifier(n_neighbors=15))])\nneighbours.fit(train[['vol_20d', 'ret_20d']], train['rising'])\np_knn = neighbours.predict_proba(test[['vol_20d', 'ret_20d']])[:, 1]\nauc = round(roc_auc_score(test['rising'], p_knn), 3)\n\nprint(auc)",
   f"{KNN_AUC[15][1]:.3f}. The four lines are the ones every model in this "
   "course takes, with a different class on the first. Nothing about the "
   "scoring changes either.")

ex("H2", "Why the scaler is not optional", 3,
   "On the credit table, fit k-nearest neighbours with 25 neighbours twice, "
   "once with a scaler in front and once without, and print both test AUCs. "
   "Then print the standard deviation of `limit` and of `late_now`.",
   "bare = ...\nscaled = ...\naucs = ...\n\nprint(aucs)\nprint(round(c_train['limit'].std(), 0), round(c_train['late_now'].std(), 2))",
   ["`bare = KNeighborsClassifier(n_neighbors=25)` with no pipeline at all.",
    "`scaled = Pipeline([('scale', StandardScaler()), ('knn', "
    "KNeighborsClassifier(n_neighbors=25))])`."],
   "bare = KNeighborsClassifier(n_neighbors=25)\nscaled = Pipeline([('scale', StandardScaler()),\n                   ('knn', KNeighborsClassifier(n_neighbors=25))])\naucs = []\nfor model in [bare, scaled]:\n    model.fit(c_train[c_cols], c_train['default'])\n    aucs.append(round(float(roc_auc_score(c_test['default'],\n                model.predict_proba(c_test[c_cols])[:, 1])), 3))\n\nprint(aucs)\nprint(round(c_train['limit'].std(), 0), round(c_train['late_now'].std(), 2))",
   f"{AUC_KNN_BARE:.3f} without the scaler and {AUC_KNN_SCALED:.3f} with it. "
   f"`limit` varies by about {SD_LIMIT:,.0f} and `late_now` by {SD_LATE:.2f}, "
   "so an unscaled distance between two borrowers is the difference in their "
   "credit limits and nothing else. Ridge needed the scaler because of its "
   "penalty; this needs it because of the distance.")

ex("H3", "k as the dial", 3,
   "Loop over k values 1, 15, 51, 151 and 301 and print each one with the "
   "TRAINING AUC and the TEST AUC. Keep the loop: H4 continues the idea.",
   "for k in [1, 15, 51, 151, 301]:\n    model = Pipeline([('scale', StandardScaler()),\n                      ('knn', KNeighborsClassifier(n_neighbors=k))])\n    model.fit(train[['vol_20d', 'ret_20d']], train['rising'])\n    train_auc = ...\n    test_auc = ...\n    print(k, train_auc, test_auc)",
   "Build the pipeline inside the loop with `KNeighborsClassifier(n_neighbors=k)`.",
   "for k in [1, 15, 51, 151, 301]:\n    model = Pipeline([('scale', StandardScaler()),\n                      ('knn', KNeighborsClassifier(n_neighbors=k))])\n    model.fit(train[['vol_20d', 'ret_20d']], train['rising'])\n    train_auc = round(roc_auc_score(train['rising'],\n                      model.predict_proba(train[['vol_20d', 'ret_20d']])[:, 1]), 3)\n    test_auc = round(roc_auc_score(test['rising'],\n                     model.predict_proba(test[['vol_20d', 'ret_20d']])[:, 1]), 3)\n    print(k, train_auc, test_auc)",
   f"At k of 1 the training AUC is {KNN_AUC[1][0]:.3f} and the test AUC is "
   f"{KNN_AUC[1][1]:.3f}: every training day is its own nearest neighbour, so "
   "the model reproduces the rows it learned from and knows nothing else. By "
   f"k of 301 they are {KNN_AUC[301][0]:.3f} and {KNN_AUC[301][1]:.3f}. A "
   "large k is a simple model, in the same way that a small `C` was.")

ex("H4", "Choosing k on the folds", 4,
   "Continuing from H3. Search k over 1, 5, 15, 51, 101, 151, 201 and 301 "
   "with `GridSearchCV`, the time folds and `scoring='roc_auc'`, and print the "
   "winner with its mean score.",
   "neighbours = Pipeline([('scale', StandardScaler()), ('knn', KNeighborsClassifier())])\n\ngrid = ...\nsearch = ...\n...\nbest = ...\n\nprint(best)",
   ["The grid key is the step name, two underscores, the argument: "
    "`{'knn__n_neighbors': [1, 5, 15, 51, 101, 151, 201, 301]}`.",
    "`GridSearchCV(neighbours, grid, cv=folds, scoring='roc_auc')`."],
   "neighbours = Pipeline([('scale', StandardScaler()), ('knn', KNeighborsClassifier())])\n\ngrid = {'knn__n_neighbors': [1, 5, 15, 51, 101, 151, 201, 301]}\nsearch = GridSearchCV(neighbours, grid, cv=folds, scoring='roc_auc')\nsearch.fit(train[['vol_20d', 'ret_20d']], train['rising'])\nbest = (search.best_params_, round(float(search.best_score_), 3))\n\nprint(best)",
   f"k of {BEST_K}, with a mean AUC of {BEST_K_CV:.3f} over the five folds. "
   "The search is the one used for `alpha` and for `C`, with a different "
   "setting named in the grid.")

ex("H5", "The k the folds cannot take", 1,
   "Print the number of fitting rows in each of the five time folds. Then say "
   "why a grid containing k of 401 would return `nan` for that value.",
   "for fit_rows, score_rows in folds.split(train):\n    ...",
   "`print(len(fit_rows), len(score_rows))`.",
   "for fit_rows, score_rows in folds.split(train):\n    print(len(fit_rows), len(score_rows))",
   f"The first fold fits on only {SMALLEST_FOLD} days, so a k of 401 asks for "
   "more neighbours than there are rows to find them in. `GridSearchCV` turns "
   "the failure into `nan` and a warning rather than stopping, so the value "
   "silently drops out of the search. The limit is the smallest fold, not the "
   "size of the training rows.")

ex("H6", "k-NN against logistic regression", 3,
   "Cross-validate a scaled logistic regression on the same two columns and "
   "the same folds, and print its mean AUC. Compare it with the "
   f"{BEST_K_CV:.3f} the best k reached in H4.",
   'logistic = ...\nscores = ...\nmean_auc = ...\n\nprint(scores)\nprint(mean_auc)',
   "`Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])`.",
   "logistic = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\nscores = cross_val_score(logistic, train[['vol_20d', 'ret_20d']],\n                         train['rising'], cv=folds, scoring='roc_auc').round(3)\nmean_auc = round(float(scores.mean()), 3)\n\nprint(scores)\nprint(mean_auc)",
   f"{LOG_CV_MEAN:.3f} against {BEST_K_CV:.3f}, so the folds choose logistic "
   f"regression. On the test days the gap is wider: {LOG_TEST_AUC:.3f} against "
   f"{KNN_TEST_AUC:.3f}. A flexible model is not automatically a better one.")

ex("H7", "The probabilities a vote can give", 3,
   "Fit k-nearest neighbours with 5 neighbours on the two columns and print "
   "the distinct probabilities it produces on the test days, in order. Use "
   "`sorted` and `set`.",
   "model = Pipeline([('scale', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])\n"
   "model.fit(train[['vol_20d', 'ret_20d']], train['rising'])\n"
   "p_knn = model.predict_proba(test[['vol_20d', 'ret_20d']])[:, 1]\n\n"
   "distinct = ...\n\nprint(distinct)",
   "`p_knn.round(3).tolist()` turns the array into a plain list, then "
   "`sorted(set(...))` drops the duplicates and puts the rest in order.",
   "model = Pipeline([('scale', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])\n"
   "model.fit(train[['vol_20d', 'ret_20d']], train['rising'])\n"
   "p_knn = model.predict_proba(test[['vol_20d', 'ret_20d']])[:, 1]\n\n"
   "distinct = sorted(set(p_knn.round(3).tolist()))\n\nprint(distinct)",
   f"Six values: {P_K5}. A vote among 5 neighbours can only ever give a "
   "multiple of one fifth, so the probabilities are coarse. A larger k gives "
   "finer ones, which is a second reason beyond smoothness to prefer it.")

ex("H8", "Draw the two AUCs against k", 2,
   "Draw the training AUC and the test AUC against k as two lines, with k on a "
   "logarithmic axis and a legend naming them. The numbers are the ones from "
   "H3, over a few more values of k.",
   "ks = [1, 3, 9, 25, 51, 101, 201, 301]\ntrain_aucs = []\ntest_aucs = []\n"
   "for k in ks:\n"
   "    model = Pipeline([('scale', StandardScaler()),\n"
   "                      ('knn', KNeighborsClassifier(n_neighbors=k))])\n"
   "    model.fit(train[['vol_20d', 'ret_20d']], train['rising'])\n"
   "    train_aucs.append(roc_auc_score(train['rising'],\n"
   "                      model.predict_proba(train[['vol_20d', 'ret_20d']])[:, 1]))\n"
   "    test_aucs.append(roc_auc_score(test['rising'],\n"
   "                     model.predict_proba(test[['vol_20d', 'ret_20d']])[:, 1]))\n\n"
   "fig, ax = plt.subplots(figsize=(7, 3))\n...\n...\n...\n...\n"
   "ax.set_xlabel('k')\nax.set_ylabel('AUC')\nplt.show()",
   ["Two `ax.plot(ks, ..., label=...)` calls, then `ax.set_xscale('log')` and "
    "`ax.legend()`.",
    "A logarithmic axis is right here because the k values are spaced by "
    "factors rather than by steps, exactly as the alphas were in Session 6."],
   "ks = [1, 3, 9, 25, 51, 101, 201, 301]\ntrain_aucs = []\ntest_aucs = []\n"
   "for k in ks:\n"
   "    model = Pipeline([('scale', StandardScaler()),\n"
   "                      ('knn', KNeighborsClassifier(n_neighbors=k))])\n"
   "    model.fit(train[['vol_20d', 'ret_20d']], train['rising'])\n"
   "    train_aucs.append(roc_auc_score(train['rising'],\n"
   "                      model.predict_proba(train[['vol_20d', 'ret_20d']])[:, 1]))\n"
   "    test_aucs.append(roc_auc_score(test['rising'],\n"
   "                     model.predict_proba(test[['vol_20d', 'ret_20d']])[:, 1]))\n\n"
   "fig, ax = plt.subplots(figsize=(7, 3))\n"
   "ax.plot(ks, train_aucs, marker='o', label='training rows')\n"
   "ax.plot(ks, test_aucs, marker='o', label='test rows')\n"
   "ax.set_xscale('log')\nax.legend()\n"
   "ax.set_xlabel('k')\nax.set_ylabel('AUC')\n"
   "ax.set_title('A small k fits the training rows and nothing else', loc='left')\n"
   "plt.show()",
   f"The two lines start far apart, at {KNN_AUC[1][0]:.2f} and "
   f"{KNN_AUC[1][1]:.2f}, and close as k grows. This is the same picture the "
   "validation curve drew for alpha in Session 6 and for C in Session 8, with "
   "a different setting on the horizontal axis.")

md("---")

# ====================================================== I
section(
"## I · When it goes wrong\n\n"
"Five mistakes that are easy to make and quiet when they happen."
)

ex("I1", "AUC on the wrong argument", 1,
   "This cell passes the 0/1 predictions to `roc_auc_score` instead of the "
   "probabilities. Run it, read the number, then fix it in the cell below.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "predicted = model.predict(c_test[c_cols])\n\n"
   "print(round(roc_auc_score(c_test['default'], predicted), 4))",
   "Nothing raises here. That is the problem: the number is wrong but the "
   "code runs.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(c_train[c_cols], c_train['default'])\n"
   "p = model.predict_proba(c_test[c_cols])[:, 1]\n\n"
   "print(round(roc_auc_score(c_test['default'], p), 4))",
   f"With the predictions it reports about "
   f"{roc_auc_score(YC, PRED_C):.4f}; with the probabilities, {AUC_C:.4f}. "
   "Passing 0s and 1s throws away the ranking and leaves the AUC reading a "
   "single threshold. No error is raised, so this one is only caught by "
   "knowing what the function wants.")

ex("I2", "A pipeline that scales nothing", 2,
   "Write the fix: the cell below fits k-nearest neighbours directly on the "
   "credit columns, with no scaler. Rewrite it as a pipeline that scales "
   "first, and print both AUCs.",
   "wrong = KNeighborsClassifier(n_neighbors=25)\nwrong.fit(c_train[c_cols], c_train['default'])\nprint(round(roc_auc_score(c_test['default'], wrong.predict_proba(c_test[c_cols])[:, 1]), 3))\n\nright = ...\n...\nauc_right = ...\n\nprint(auc_right)",
   "`right = Pipeline([('scale', StandardScaler()), ('knn', "
   "KNeighborsClassifier(n_neighbors=25))])`, then `right.fit(...)` on the "
   "next line.",
   "wrong = KNeighborsClassifier(n_neighbors=25)\nwrong.fit(c_train[c_cols], c_train['default'])\nprint(round(roc_auc_score(c_test['default'], wrong.predict_proba(c_test[c_cols])[:, 1]), 3))\n\nright = Pipeline([('scale', StandardScaler()),\n                  ('knn', KNeighborsClassifier(n_neighbors=25))])\nright.fit(c_train[c_cols], c_train['default'])\nauc_right = round(roc_auc_score(c_test['default'],\n                                right.predict_proba(c_test[c_cols])[:, 1]), 3)\n\nprint(auc_right)",
   f"{AUC_KNN_BARE:.3f} becomes {AUC_KNN_SCALED:.3f}. Nothing warned you. The "
   "pipeline exists so the scaler cannot be forgotten and cannot be fitted on "
   "the test rows by accident.")

ex("I3", "A shuffled split on a time series", 2,
   "Run this cell, which splits the index table at random, and print the "
   "first and last date on each side. Then say in a comment why the score "
   "that follows would be meaningless.",
   "early, late = train_test_split(table, test_size=0.3, random_state=0)\n\n"
   "print(early.index.min().date(), early.index.max().date())\n"
   "print(late.index.min().date(), late.index.max().date())\n\n"
   "# why is this wrong here?\n...",
   "Look at the date ranges. Do they overlap?",
   "early, late = train_test_split(table, test_size=0.3, random_state=0)\n\n"
   "print(early.index.min().date(), early.index.max().date())\n"
   "print(late.index.min().date(), late.index.max().date())\n\n"
   "# Both halves run from 2015 to 2024, so the model would be fitted on days\n"
   "# that come after the days it is scored on. Neighbouring days also share\n"
   "# most of a 20-day window, so a shuffled split puts nearly the same row on\n"
   "# both sides.",
   "Both halves cover the whole period. `train_test_split` is right for the "
   "credit table, where rows are separate borrowers, and wrong here, where "
   "rows are days in order and overlapping windows make neighbouring rows "
   "nearly copies of each other.")

ex("I4", "A confusion matrix in the wrong order", 2,
   "Print the three-class confusion matrix twice, once without `labels=` and "
   "once with `labels=['falls', 'stays', 'rises']`, and print the model's "
   "`classes_` between them.",
   "three = Pipeline([('scale', StandardScaler()),\n"
   "                  ('logit', LogisticRegression(max_iter=1000))])\n"
   "three.fit(train[['vol_20d', 'ret_20d']], train['move'])\n"
   "predicted = three.predict(test[['vol_20d', 'ret_20d']])\n\n"
   "print(confusion_matrix(test['move'], predicted))\n"
   "print(three.named_steps['logit'].classes_)\nprint(...)",
   "`confusion_matrix(test['move'], predicted, labels=['falls', 'stays', 'rises'])`.",
   "three = Pipeline([('scale', StandardScaler()),\n"
   "                  ('logit', LogisticRegression(max_iter=1000))])\n"
   "three.fit(train[['vol_20d', 'ret_20d']], train['move'])\n"
   "predicted = three.predict(test[['vol_20d', 'ret_20d']])\n\n"
   "print(confusion_matrix(test['move'], predicted))\n"
   "print(three.named_steps['logit'].classes_)\n"
   "print(confusion_matrix(test['move'], predicted, labels=['falls', 'stays', 'rises']))",
   "The first matrix is in alphabetical order, so its middle row and column "
   "are `rises`, not `stays`. Both matrices are correct; only one of them "
   "means what you would assume at a glance. Read `classes_` before reading "
   "any matrix.")

ex("I5", "Two columns or three", 2,
   "Print the shape of `predict_proba` for the two-class model on `rising` "
   "and for the three-class model on `move`, and the sum of the first row of "
   "each.",
   "two = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "two.fit(train[['vol_20d', 'ret_20d']], train['rising'])\n"
   "three = Pipeline([('scale', StandardScaler()),\n"
   "                  ('logit', LogisticRegression(max_iter=1000))])\n"
   "three.fit(train[['vol_20d', 'ret_20d']], train['move'])\n\n"
   "for model in [two, three]:\n"
   "    probabilities = model.predict_proba(test[['vol_20d', 'ret_20d']])\n    ...",
   "`print(probabilities.shape, round(float(probabilities[0].sum()), 6))`.",
   "two = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "two.fit(train[['vol_20d', 'ret_20d']], train['rising'])\n"
   "three = Pipeline([('scale', StandardScaler()),\n"
   "                  ('logit', LogisticRegression(max_iter=1000))])\n"
   "three.fit(train[['vol_20d', 'ret_20d']], train['move'])\n\n"
   "for model in [two, three]:\n"
   "    probabilities = model.predict_proba(test[['vol_20d', 'ret_20d']])\n"
   "    print(probabilities.shape, round(float(probabilities[0].sum()), 6))",
   f"({N_TEST}, 2) and ({N_TEST}, 3), both summing to 1 on every row. The "
   "habit of writing `[:, 1]` comes from the two-class case, where the second "
   "column is the probability of a 1. With three classes there is no single "
   "column to take, and `[:, 1]` silently gives you the second class in "
   "alphabetical order.")

md("---")

# ====================================================== J
section(
"## J · The rare label across the desk\n\n"
"J1 writes a function; J2 to J5 all use it. The label is the rare one: "
"volatility over the next 20 days more than 1.5 times this month's."
)

ex("J1", "A function that builds the table", 4,
   "Write `jump_table(ticker)` returning a table with `vol_20d`, `vol_next` "
   "and the label `jump`, for one instrument, with the incomplete rows "
   "dropped. `rets` from the setup cell holds the returns in percent.",
   'def jump_table(ticker):\n    """..."""\n    ...\n\n\nprint(jump_table(\'AAPL\'))',
   ["The target looks forward: `r.rolling(20).std().shift(-20)`.",
    "`(frame['vol_next'] > 1.5 * frame['vol_20d']).astype(int)`, made after "
    "the dropna so the two columns line up."],
   'def jump_table(ticker):\n    """vol_20d, vol_next and a rare 0/1 jump label for one instrument."""\n    r = rets[ticker]\n    frame = pd.DataFrame({\'vol_20d\': r.rolling(20).std()})\n    frame[\'vol_next\'] = r.rolling(20).std().shift(-20)\n    frame = frame.dropna()\n    frame[\'jump\'] = (frame[\'vol_next\'] > 1.5 * frame[\'vol_20d\']).astype(int)\n    return frame\n\n\nprint(jump_table(\'AAPL\'))',
   "One function, eleven instruments. Writing the table-building code once "
   "and calling it in a loop is what makes the next four exercises short, and "
   "it is the same move as `build_table` in Session 8.",
   revisits="S2")

ex("J2", "The share of jumps per instrument", 2,
   "Using `jump_table` from J1, build a dictionary from ticker to the share "
   "of TEST days that were jumps, and print it.",
   'shares = {}\nfor ticker in tickers:\n    ...\n\nprint(shares)',
   "`shares[ticker] = round(float(late['jump'].mean()), 3)`. `float()` keeps "
   "the printed dictionary readable.",
   "shares = {}\nfor ticker in tickers:\n    late = jump_table(ticker).loc['2023-01-01':]\n    shares[ticker] = round(float(late['jump'].mean()), 3)\n\nprint(shares)",
   f"From {min(DESK_SHARE.values()):.3f} to {max(DESK_SHARE.values()):.3f}. "
   "The same definition of a jump gives a class of quite different rarity on "
   "each instrument, and the baseline to beat moves with it.")

ex("J3", "The AUC per instrument", 3,
   "Fit the one-column classifier on each instrument's training days and "
   "collect the test AUC in a dictionary. Print it sorted, largest first.",
   'aucs = {}\nfor ticker in tickers:\n    ...\n\nfor ticker in sorted(aucs, key=aucs.get, reverse=True):\n    print(ticker, aucs[ticker])',
   "`aucs[ticker] = round(float(roc_auc_score(late['jump'], "
   "model.predict_proba(late[['vol_20d']])[:, 1])), 3)`.",
   "aucs = {}\nfor ticker in tickers:\n    frame = jump_table(ticker)\n    early = frame.loc[:'2022-12-31']\n    late = frame.loc['2023-01-01':]\n    model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n    model.fit(early[['vol_20d']], early['jump'])\n    aucs[ticker] = round(float(roc_auc_score(late['jump'],\n                        model.predict_proba(late[['vol_20d']])[:, 1])), 3)\n\nfor ticker in sorted(aucs, key=aucs.get, reverse=True):\n    print(ticker, aucs[ticker])",
   f"{BEST_DESK} ranks best at {DESK_AUC[BEST_DESK]:.3f} and {WORST_DESK} "
   f"worst at {DESK_AUC[WORST_DESK]:.3f}. Eight of the eleven are above 0.74, "
   "so this month's volatility really does say something about whether next "
   "month jumps.")

ex("J4", "How often the model says nothing", 3,
   "Count how many of the eleven instruments get NO predicted jump at all at "
   "a threshold of one half, and print the count with the list of tickers.",
   'silent = []\nfor ticker in tickers:\n    ...\n\nprint(len(silent))\nprint(silent)',
   "`if predicted.sum() == 0:` is the condition for a model that never "
   "predicts a jump.",
   "silent = []\nfor ticker in tickers:\n    frame = jump_table(ticker)\n    early = frame.loc[:'2022-12-31']\n    late = frame.loc['2023-01-01':]\n    model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n    model.fit(early[['vol_20d']], early['jump'])\n    predicted = model.predict(late[['vol_20d']])\n    if predicted.sum() == 0:\n        silent.append(ticker)\n\nprint(len(silent))\nprint(silent)",
   f"{N_SILENT} of the 11. On those instruments the model scores exactly the "
   "majority rule, catches nothing, and looks respectable while doing it. "
   "This is the single most useful thing to check on a rare class, and it "
   "takes one line.")

ex("J5", "The report line", 5,
   "Write a loop that prints one line per instrument with the ticker, the "
   "share of jumps and the AUC, lined up with field widths and sorted by AUC. "
   "Reuse `shares` from J2 and `aucs` from J3, or rebuild them.",
   'shares = {}\naucs = {}\nfor ticker in tickers:\n    ...\n\nprint(f"{\'ticker\':<8}{\'share\':>8}{\'AUC\':>8}")\nfor ticker in sorted(aucs, key=aucs.get, reverse=True):\n    print(f"{ticker:<8}{shares[ticker]:>8.3f}{aucs[ticker]:>8.3f}")',
   "Loop over `sorted(aucs, key=aucs.get, reverse=True)` and print "
   "`f\"{ticker:<8}{shares[ticker]:>8.3f}{aucs[ticker]:>8.3f}\"`.",
   'shares = {}\naucs = {}\nfor ticker in tickers:\n    frame = jump_table(ticker)\n    early = frame.loc[:\'2022-12-31\']\n    late = frame.loc[\'2023-01-01\':]\n    model = Pipeline([(\'scale\', StandardScaler()), (\'logit\', LogisticRegression())])\n    model.fit(early[[\'vol_20d\']], early[\'jump\'])\n    shares[ticker] = late[\'jump\'].mean()\n    aucs[ticker] = roc_auc_score(late[\'jump\'], model.predict_proba(late[[\'vol_20d\']])[:, 1])\n\nprint(f"{\'ticker\':<8}{\'share\':>8}{\'AUC\':>8}")\nfor ticker in sorted(aucs, key=aucs.get, reverse=True):\n    print(f"{ticker:<8}{shares[ticker]:>8.3f}{aucs[ticker]:>8.3f}")',
   "A dictionary, a `sorted` with `key=`, and three field widths. That is the "
   "whole of a desk report, and every piece of it came from Sessions 2 and 5.")

md("---")

# ====================================================== K
section(
"## K · Five small cases\n\n"
"Each of these stands completely on its own and needs no model from earlier "
"in the notebook. They are here to keep loops, `if`, dictionaries, f-strings "
"and functions in working order, in this session's setting."
)

ex("K1", "Turning probabilities into advice", 2,
   "A model has given seven days these probabilities of a jump. Loop over "
   "them and print one line each: `low` below 0.2, `act` at 0.5 or above, and "
   "`watch` in between. Print the probability as a percentage with no "
   "decimals, beside the word.",
   "probabilities = [0.04, 0.19, 0.33, 0.51, 0.08, 0.62, 0.27]\n\n"
   "for value in probabilities:\n    ...",
   ["An `if`, an `elif` and an `else`, in that order, inside the loop.",
    "`print(f\"{value:.0%}\", word)` once the word is decided."],
   "probabilities = [0.04, 0.19, 0.33, 0.51, 0.08, 0.62, 0.27]\n\n"
   "for value in probabilities:\n    if value < 0.2:\n        word = 'low'\n"
   "    elif value >= 0.5:\n        word = 'act'\n    else:\n        word = 'watch'\n"
   "    print(f\"{value:.0%}\", word)",
   f"The seven come out as {', '.join(K1_WORDS)}. Two thresholds and three "
   "words: a classifier's output becomes a decision only once someone writes "
   "down rules like these, and they are ordinary `if` statements.")

ex("K2", "A confusion matrix with no library", 3,
   "Two lists hold what a model said and what happened, for eight days. "
   "Count the four combinations into a dictionary with a single loop over "
   "`zip`, and print it.",
   "said = [1, 0, 1, 1, 0, 0, 1, 0]\nwas  = [1, 0, 0, 1, 1, 0, 1, 1]\n\n"
   "counts = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}\nfor s, w in zip(said, was):\n    ...\n\n"
   "print(counts)",
   ["Four cases: `if s == 1 and w == 1:` is a true positive.",
    "`counts['tp'] += 1` adds one to that entry."],
   "said = [1, 0, 1, 1, 0, 0, 1, 0]\nwas  = [1, 0, 0, 1, 1, 0, 1, 1]\n\n"
   "counts = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}\nfor s, w in zip(said, was):\n"
   "    if s == 1 and w == 1:\n        counts['tp'] += 1\n"
   "    elif s == 1 and w == 0:\n        counts['fp'] += 1\n"
   "    elif s == 0 and w == 1:\n        counts['fn'] += 1\n"
   "    else:\n        counts['tn'] += 1\n\nprint(counts)",
   f"`{K2_COUNTS}`. `confusion_matrix` does exactly this and nothing more. "
   "`zip` walks the two lists in step, which is the right tool whenever two "
   "sequences line up row by row.")

ex("K3", "A sentence about one instrument", 2,
   "Write `verdict(ticker, auc, share)` returning a single sentence, and call "
   "it twice. The sentence should name the instrument, give the AUC to two "
   "decimals and the share as a percentage, and end with `worth using` when "
   "the AUC is at least 0.75 and `not yet` otherwise.",
   "def verdict(ticker, auc, share):\n    \"\"\"...\"\"\"\n    ...\n\n\n"
   "print(verdict('AAPL', 0.889, 0.087))\nprint(verdict('JPM', 0.610, 0.160))",
   ["Decide the ending with an `if` and store it in a variable.",
    "Return an f-string: "
    "`f\"{ticker}: AUC {auc:.2f} on {share:.0%} jumps, {ending}\"`."],
   "def verdict(ticker, auc, share):\n"
   "    \"\"\"One sentence about one instrument's jump model.\"\"\"\n"
   "    if auc >= 0.75:\n        ending = 'worth using'\n    else:\n        ending = 'not yet'\n"
   "    return f\"{ticker}: AUC {auc:.2f} on {share:.0%} jumps, {ending}\"\n\n\n"
   "print(verdict('AAPL', 0.889, 0.087))\nprint(verdict('JPM', 0.610, 0.160))",
   "`AAPL: AUC 0.89 on 9% jumps, worth using` and `JPM: AUC 0.61 on 16% "
   "jumps, not yet`. A function that returns a string rather than printing it "
   "can be used in a loop, put in a list, or written to a file. Printing "
   "inside the function would close all three doors.")

ex("K4", "The longest run of warnings", 4,
   "Fit the one-column jump model on the index table, call a jump whenever "
   "the probability reaches 0.3, and find the longest run of consecutive "
   "called days. Use a counter and a `for` loop.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(train[['vol_20d']], train['jump'])\n"
   "p = model.predict_proba(test[['vol_20d']])[:, 1]\ncalled = (p >= 0.3).astype(int)\n\n"
   "run = 0\nlongest = 0\nfor value in called:\n    ...\n\n"
   "print(called.sum())\nprint(longest)",
   ["Inside the loop: if the value is 1, add one to `run`; otherwise reset "
    "`run` to 0.",
    "After updating `run`, keep the best: `longest = max(longest, run)`."],
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(train[['vol_20d']], train['jump'])\n"
   "p = model.predict_proba(test[['vol_20d']])[:, 1]\ncalled = (p >= 0.3).astype(int)\n\n"
   "run = 0\nlongest = 0\nfor value in called:\n"
   "    if value == 1:\n        run += 1\n    else:\n        run = 0\n"
   "    longest = max(longest, run)\n\nprint(called.sum())\nprint(longest)",
   f"{K4_CALLED} days called in total, and the longest unbroken run is "
   f"{K4_RUN}. The warnings arrive in clusters rather than spread out, which "
   "is what you would expect from a column that moves slowly. A counter that "
   "resets is the standard shape for the longest run of anything.")

ex("K5", "Accuracy, year by year", 3,
   "Fit the one-column jump model, predict the test days, and compute the "
   "accuracy separately for 2023 and 2024 with `groupby`: make a Series of "
   "`True` and `False` indexed by date, and group it by year.",
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(train[['vol_20d']], train['jump'])\n"
   "predicted = model.predict(test[['vol_20d']])\n\n"
   "right = ...\nby_year = ...\n\nprint(by_year)",
   ["`right = pd.Series(predicted == test['jump'].values, index=test.index)`.",
    "`right.groupby(right.index.year).mean()`: the mean of a column of `True` "
    "and `False` is the share of `True`."],
   "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
   "model.fit(train[['vol_20d']], train['jump'])\n"
   "predicted = model.predict(test[['vol_20d']])\n\n"
   "right = pd.Series(predicted == test['jump'].values, index=test.index)\n"
   "by_year = right.groupby(right.index.year).mean()\n\nprint(by_year)",
   f"{K5[2023]:.3f} in 2023 and {K5[2024]:.3f} in 2024. The model predicts no "
   "jump on any day, so these two numbers are just the share of days with no "
   "jump in each year: 2024 had more jumps, and the accuracy fell without the "
   "model changing at all. One number over two years would have hidden that.")

# ---------------------------------------------------------------- closing
md(
"## \U0001f3c1 Done\n\n"
"You built rare labels and the baselines they have to beat, split a "
"cross-section at random and saw why a time series cannot be, read the four "
"counts by hand and from the library, priced the two kinds of mistake and "
"found the threshold that follows from those prices, weighted a rare class "
"and watched the probabilities stop meaning anything, put them right again, "
"fitted three classes and turned three scores into three probabilities, and "
"lost a fair fight between k-nearest neighbours and logistic regression.\n\n"
"The case takes the threshold and the rare label back to the risk report, "
"where Apple's jump model scores 91 percent accuracy and predicts nothing at "
"all."
)

# ---------------------------------------------------------------- write
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
for _i, _c in enumerate(nb.cells):
    _c["id"] = f"c{_i:04d}"

OUT.write_text(nbf.writes(nb), encoding="utf-8")

n_ex = sum(1 for c in cells if c.cell_type == "markdown" and c.source.startswith("### "))
print("wrote", OUT, " (", len(cells), "cells,", n_ex, "exercises )")
