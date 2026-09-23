# -*- coding: utf-8 -*-
"""Build session_09_case.ipynb  (The Analyst's Notebook, Part 9).

Conventions (approved for Sessions 1 to 8): no star badges, one cumulative
investigation where later questions reuse what earlier ones stored, folded
hints and solutions, stated formulas, plain explanatory tone, no em-dashes.
Opens with a QUICK LOAD restoring Part 8's findings.

Part 8 ended with a classifier on Apple that ranks well (AUC about 0.80) and a
threshold of one half that buys protection almost every month. It closed by
promising two things: the threshold as a decision with the cost of each kind of
error, chosen on the training rows, and a label that is rare rather than
balanced. Part 9 delivers both on the same instrument.

The rare label is a JUMP: volatility over the next 20 days more than 1.5 times
volatility over the last 20. On Apple that is 18.5 percent of the training days
and 8.7 percent of the test days. The one-column classifier scores exactly the
majority rule on it and predicts no jump at all, while its AUC is 0.889: the
ranking is good and the threshold throws it away. That is the spine of this
part.

Returns stay in plain decimals here, as in Parts 1 to 8.

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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                             roc_auc_score, average_precision_score, brier_score_loss,
                             f1_score)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "session_09" / "session_09_case.ipynb"

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


def one_pipe(**kw):
    return Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(**kw))])


TBL = wide_table("AAPL")
COLS = list(TBL.columns[:-1])
N_TBL = len(TBL)

# ---- Part 8, restated (the same code as Part 8's generator) ---------------
TBL["rising"] = (TBL["vol_next"] > TBL["vol_20d"]).astype(int)
TRAIN, TEST = TBL.loc[:"2022-12-31"], TBL.loc["2023-01-01":]
N_TRAIN, N_TEST = len(TRAIN), len(TEST)
P8_MODEL = one_pipe().fit(TRAIN[["vol_20d"]], TRAIN["rising"])
P8_P = P8_MODEL.predict_proba(TEST[["vol_20d"]])[:, 1]
P8_AUC = float(roc_auc_score(TEST["rising"], P8_P))
P8_ACC = float(accuracy_score(TEST["rising"], P8_MODEL.predict(TEST[["vol_20d"]])))
P8_MAJORITY = float(max(TEST["rising"].mean(), 1 - TEST["rising"].mean()))
P8_CALLED = int(P8_MODEL.predict(TEST[["vol_20d"]]).sum())
P8_SHARE_CALLED = P8_CALLED / N_TEST

# ---- Part 9: the rare label ----------------------------------------------
RATIO = TBL["vol_next"] / TBL["vol_20d"]
CUT = 1.5
TBL["jump"] = (RATIO > CUT).astype(int)
TRAIN, TEST = TBL.loc[:"2022-12-31"], TBL.loc["2023-01-01":]
Y_TR, Y_TE = TRAIN["jump"], TEST["jump"]
SHARE_TR, SHARE_TE = float(Y_TR.mean()), float(Y_TE.mean())
N_JUMP_TR, N_JUMP_TE = int(Y_TR.sum()), int(Y_TE.sum())
MAJORITY = 1 - SHARE_TE

JUMP = one_pipe().fit(TRAIN[["vol_20d"]], Y_TR)
P_TR = JUMP.predict_proba(TRAIN[["vol_20d"]])[:, 1]
P_TE = JUMP.predict_proba(TEST[["vol_20d"]])[:, 1]
PRED = JUMP.predict(TEST[["vol_20d"]])
ACC = float(accuracy_score(Y_TE, PRED))
AUC = float(roc_auc_score(Y_TE, P_TE))
AP = float(average_precision_score(Y_TE, P_TE))
N_PRED_1 = int(PRED.sum())
P_MIN, P_MAX = float(P_TE.min()), float(P_TE.max())
CM = confusion_matrix(Y_TE, PRED, labels=[0, 1])
TN, FP, FN, TP = int(CM[0, 0]), int(CM[0, 1]), int(CM[1, 0]), int(CM[1, 1])
REC = float(recall_score(Y_TE, PRED))
AUC_FOLDS = cross_val_score(one_pipe(), TRAIN[["vol_20d"]], Y_TR, cv=TS5, scoring="roc_auc")
AUC_FOLDS_MEAN = float(AUC_FOLDS.mean())

# the cost sweep: a missed jump costs 5, a false alarm costs 1
COST_MISS, COST_FALSE = 5, 1
GRID = np.arange(0.05, 0.60, 0.01)


def sweep(y, prob):
    out = []
    for t in GRID:
        tn, fp, fn, tp = confusion_matrix(y, (prob >= t).astype(int), labels=[0, 1]).ravel()
        out.append(COST_MISS * fn + COST_FALSE * fp)
    return np.array(out)


COST_TR = sweep(Y_TR, P_TR)
COST_TE = sweep(Y_TE, P_TE)
CHOSEN = float(GRID[COST_TR.argmin()])
COST_AT_CHOSEN = int(COST_TE[COST_TR.argmin()])
COST_AT_HALF = int(COST_TE[int(np.argmin(np.abs(GRID - 0.5)))])
BEST_POSSIBLE = int(COST_TE.min())
BEST_POSSIBLE_THR = float(GRID[COST_TE.argmin()])
FORMULA_THR = COST_FALSE / (COST_FALSE + COST_MISS)
PRED_CHOSEN = (P_TE >= CHOSEN).astype(int)
CM_CHOSEN = confusion_matrix(Y_TE, PRED_CHOSEN, labels=[0, 1])
REC_CHOSEN = float(recall_score(Y_TE, PRED_CHOSEN))
PREC_CHOSEN = float(precision_score(Y_TE, PRED_CHOSEN, zero_division=0))
ACC_CHOSEN = float(accuracy_score(Y_TE, PRED_CHOSEN))
N_CALLED_CHOSEN = int(PRED_CHOSEN.sum())

BAL = one_pipe(class_weight="balanced").fit(TRAIN[["vol_20d"]], Y_TR)
P_BAL = BAL.predict_proba(TEST[["vol_20d"]])[:, 1]
PRED_BAL = BAL.predict(TEST[["vol_20d"]])
ACC_BAL = float(accuracy_score(Y_TE, PRED_BAL))
REC_BAL = float(recall_score(Y_TE, PRED_BAL))
N_CALLED_BAL = int(PRED_BAL.sum())
CM_BAL = confusion_matrix(Y_TE, PRED_BAL, labels=[0, 1])
AUC_BAL = float(roc_auc_score(Y_TE, P_BAL))
MEAN_BAL = float(P_BAL.mean())

# calibration
EDGES = [0, 0.1, 0.2, 0.3, 0.5]
_bk = pd.cut(P_TE, EDGES)
BUCKETS = pd.DataFrame({"p": P_TE, "y": Y_TE.values}).groupby(_bk, observed=True)["y"].agg(["size", "mean"])
MEAN_P = float(P_TE.mean())
BRIER = float(brier_score_loss(Y_TE, P_TE))
BRIER_BAL = float(brier_score_loss(Y_TE, P_BAL))
CAL = CalibratedClassifierCV(one_pipe(), method="sigmoid", cv=5).fit(TRAIN[["vol_20d"]], Y_TR)
P_CAL = CAL.predict_proba(TEST[["vol_20d"]])[:, 1]
MEAN_CAL = float(P_CAL.mean())
BRIER_CAL = float(brier_score_loss(Y_TE, P_CAL))

# twenty columns
CGRID = [0.0001, 0.001, 0.01, 0.1, 1, 10]
SEARCH = GridSearchCV(one_pipe(max_iter=1000), {"logit__C": CGRID},
                      cv=TS5, scoring="roc_auc").fit(TRAIN[COLS], Y_TR)
BEST_C = SEARCH.best_params_["logit__C"]
BEST_CV = float(SEARCH.best_score_)
AUC_WIDE = float(roc_auc_score(Y_TE, SEARCH.predict_proba(TEST[COLS])[:, 1]))

# three classes
TBL["move"] = pd.cut(RATIO, [-np.inf, 0.85, 1.25, np.inf], labels=["falls", "stays", "rises"])
TRAIN, TEST = TBL.loc[:"2022-12-31"], TBL.loc["2023-01-01":]
MOVE_TR = TRAIN["move"].value_counts(normalize=True)
MOVE_TE = TEST["move"].value_counts(normalize=True)
MAJ_MOVE = str(TRAIN["move"].value_counts().idxmax())
MAJ_MOVE_ACC = float((TEST["move"] == MAJ_MOVE).mean())
MC = one_pipe(max_iter=1000).fit(TRAIN[["vol_20d"]], TRAIN["move"])
PRED_MOVE = MC.predict(TEST[["vol_20d"]])
ACC_MOVE = float(accuracy_score(TEST["move"], PRED_MOVE))
F1_MOVE = float(f1_score(TEST["move"], PRED_MOVE, average="macro", zero_division=0))
ORDER = ["falls", "stays", "rises"]
CM_MOVE = confusion_matrix(TEST["move"], PRED_MOVE, labels=ORDER)
REC_MOVE = recall_score(TEST["move"], PRED_MOVE, average=None, labels=ORDER, zero_division=0)

# k-NN
KGRID = [1, 5, 15, 51, 101, 151, 201, 301]
KSEARCH = GridSearchCV(Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier())]),
                       {"knn__n_neighbors": KGRID}, cv=TS5,
                       scoring="roc_auc").fit(TRAIN[["vol_20d"]], Y_TR)
BEST_K = KSEARCH.best_params_["knn__n_neighbors"]
BEST_K_CV = float(KSEARCH.best_score_)
AUC_KNN = float(roc_auc_score(Y_TE, KSEARCH.best_estimator_.predict_proba(TEST[["vol_20d"]])[:, 1]))

# the desk
DESK_SHARE, DESK_AUC, DESK_SILENT = {}, {}, []
for _t in TICKERS:
    _f = wide_table(_t)
    _r = _f["vol_next"] / _f["vol_20d"]
    _f["jump"] = (_r > CUT).astype(int)
    _tr, _te = _f.loc[:"2022-12-31"], _f.loc["2023-01-01":]
    _m = one_pipe().fit(_tr[["vol_20d"]], _tr["jump"])
    DESK_SHARE[_t] = round(float(_te["jump"].mean()), 3)
    DESK_AUC[_t] = round(float(roc_auc_score(_te["jump"], _m.predict_proba(_te[["vol_20d"]])[:, 1])), 3)
    if _m.predict(_te[["vol_20d"]]).sum() == 0:
        DESK_SILENT.append(_t)
N_SILENT = len(DESK_SILENT)
DESK_BEST = max(TICKERS, key=lambda t: DESK_AUC[t])
DESK_WORST = min(TICKERS, key=lambda t: DESK_AUC[t])
N_ABOVE = sum(1 for t in TICKERS if DESK_AUC[t] >= 0.75)

# ---------------------------------------------------------------- top matter
md(
"# \U0001f4bc The Analyst's Notebook · Part 9\n"
"### A threshold, and a label that is rare\n\n"
"Part 8 gave the risk report its first classifier. On Apple it ranked the test "
f"days well, with an AUC of {P8_AUC:.3f}, and at a threshold of one half it "
f"called a rise on {P8_CALLED} of the {N_TEST} test days, which is "
f"{P8_SHARE_CALLED:.0%} of them. Buying protection {P8_SHARE_CALLED:.0%} of the "
"time is not a policy anyone would sign off, and Part 8 left the question of "
"where the threshold should sit open.\n\n"
"Part 9 answers it, and changes the label while it is there. The desk does not "
"care about every rise; it cares about a **jump**, a month at least 1.5 times "
"as volatile as the one before. That label is rare, and a rare label breaks "
"accuracy in a way that is worth seeing once on your own data."
)

md(
"## How to work through this\n\n"
"- Run the **quick load** cell first. It brings back what Part 8 established and "
"loads the price table.\n"
"- Each question builds on the last, so keep them in order and keep your "
"variables. Later questions use the names earlier ones created.\n"
"- Cells with `...` are blanks. The notebook runs cleanly even before you fill "
"them in, so **Run all** is always safe.\n"
"- Hints and solutions are folded under each question. Work first, then check.\n\n"
"**A note on units.** The lecture worked in percent and on two different "
"tables. This notebook keeps the plain decimals of Parts 1 to 8 and stays on "
"Apple, so every number here can be compared with Part 8's directly.\n\n"
"*Stuck for more than 15 minutes? Ask a friend, ask an AI for a hint (not the "
"answer), or email me at `jobo@econ.au.dk`.*"
)

md("---")

# ------------------------------------------------------------- quick load
md(
"## ⚙️ Quick load\n\n"
"The packages, the price table, and what Part 8 left you. Run it and read what "
"it prints."
)

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


# The whole universe: eleven instruments, 2015 to 2024, returns in plain decimals
prices = pd.read_csv(data_path("prices.csv"), parse_dates=["date"])
wide = prices.pivot(index="date", columns="ticker", values="close")
rets = wide.pct_change()
TICKERS = sorted(prices["ticker"].unique())

folds = TimeSeriesSplit(n_splits=5)

# --- What Part 8 established, on Apple ---
part8_label = "rising: next 20 days more volatile than the last 20"
part8_auc = ''' + f"{P8_AUC:.4f}" + '''                  # one column, vol_20d, on the test block
part8_accuracy = ''' + f"{P8_ACC:.4f}" + '''             # at a threshold of one half
part8_majority = ''' + f"{P8_MAJORITY:.4f}" + '''             # predicting the more common label every day
part8_days_called = ''' + f"{P8_CALLED}" + '''                # test days it called a rise, of ''' + f"{N_TEST}" + '''
part8_split = "by date: train to 2022-12-31, test from 2023-01-01"

print("Loaded prices:", prices.shape[0], "rows")
print("Instruments  :", ", ".join(TICKERS))
print()
print("Part 8 left you a classifier on Apple:")
print("  label   :", part8_label)
print("  split   :", part8_split)
print(f"  test AUC {part8_auc:.3f}, accuracy {part8_accuracy:.3f} against {part8_majority:.3f} for the majority rule")
print(f"  it called a rise on {part8_days_called} of ''' + f"{N_TEST}" + ''' test days, which is {part8_days_called / ''' + f"{N_TEST}" + ''':.0%} of them")
print()
print("Open question from Part 8: where should the threshold sit?")'''
)

md("---")

# ==================================================================== Q1
q("Q1", "Where Part 8 stopped",
  "Rebuild Part 8's table for Apple as `table`: the six volatility windows "
  "`[5, 10, 20, 40, 60, 120]` as `vol_<w>d`, the three return windows "
  "`[5, 20, 60]` as `ret_<w>d`, `up_20d`, the 20-day volatility of every "
  "**other** instrument as `<ticker>_vol`, the target `vol_next`, and "
  "incomplete rows dropped. Store the twenty feature names in `columns`. Then "
  "add Part 8's `rising` label, split at the end of 2022, refit the "
  "one-column classifier in a scaled pipeline and check its test AUC matches "
  "`part8_auc`.",
  "table = pd.DataFrame()\n\nfor w in [5, 10, 20, 40, 60, 120]:\n    ...\n\n"
  "for w in [5, 20, 60]:\n    ...\n\ntable['up_20d'] = ...\n\n"
  "for t in TICKERS:\n    ...\n\ntable['vol_next'] = ...\ntable = ...\ncolumns = ...\n\n"
  "# Part 8's label and split\n...\ntrain = ...\ntest = ...\n\n"
  "check = ...\nprint(check)\nprint('matches Part 8:', ...)",
  ["Column names are text built from the number: `'vol_' + str(w) + 'd'`. "
   "Inside the last loop, `if t != 'AAPL':`. `up_20d` is "
   "`(rets['AAPL'] > 0).rolling(20).mean()`.",
   "`columns = list(table.columns[:-1])` before the label is added. The check "
   "is `abs(check - part8_auc) < 0.001`."],
  "table = pd.DataFrame()\n\nfor w in [5, 10, 20, 40, 60, 120]:\n"
  "    table['vol_' + str(w) + 'd'] = rets['AAPL'].rolling(w).std()\n\n"
  "for w in [5, 20, 60]:\n    table['ret_' + str(w) + 'd'] = rets['AAPL'].rolling(w).mean()\n\n"
  "table['up_20d'] = (rets['AAPL'] > 0).rolling(20).mean()\n\n"
  "for t in TICKERS:\n    if t != 'AAPL':\n        table[t + '_vol'] = rets[t].rolling(20).std()\n\n"
  "table['vol_next'] = rets['AAPL'].rolling(20).std().shift(-20)\ntable = table.dropna()\n"
  "columns = list(table.columns[:-1])\n\n"
  "table['rising'] = (table['vol_next'] > table['vol_20d']).astype(int)\n"
  "train = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\n\n"
  "model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
  "model.fit(train[['vol_20d']], train['rising'])\n"
  "check = round(roc_auc_score(test['rising'], model.predict_proba(test[['vol_20d']])[:, 1]), 4)\n"
  "print(check)\nprint('matches Part 8:', abs(check - part8_auc) < 0.001)",
  f"{P8_AUC:.4f}, and `True`: {N_TBL:,} rows, {N_TRAIN:,} to fit on and "
  f"{N_TEST} to check on, exactly as in Part 8. The table and the split carry "
  "forward unchanged; only the label is about to move.")

# ==================================================================== Q2
q("Q2", "A label the desk would act on",
  "A rise of one percent and a rise of 60 percent both counted as `rising`. "
  "Add a rarer label, `jump`: 1 where `vol_next` is more than **1.5 times** "
  "`vol_20d`. Split again so `train` and `test` carry it, and print the share "
  "of jumps in each half and the accuracy of predicting 0 every day, stored as "
  "`majority`.",
  "# add the jump label to table, then split again\n...\ntrain = ...\ntest = ...\n\n"
  "print('train:', ...)\nprint('test :', ...)\nmajority = ...\nprint('majority rule:', majority)",
  ["`(table['vol_next'] > 1.5 * table['vol_20d']).astype(int)`.",
   "`majority = 1 - test['jump'].mean()`, because 0 is now much the more "
   "common label."],
  "table['jump'] = (table['vol_next'] > 1.5 * table['vol_20d']).astype(int)\n"
  "train = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\n\n"
  "print('train:', round(train['jump'].mean(), 4))\n"
  "print('test :', round(test['jump'].mean(), 4))\n"
  "majority = round(1 - test['jump'].mean(), 4)\nprint('majority rule:', majority)",
  f"{SHARE_TR:.4f} of the training days and {SHARE_TE:.4f} of the test days, "
  f"which is {N_JUMP_TE} jumps in the two test years against {N_JUMP_TR} in the "
  f"eight training years. The rule to beat has moved from {P8_MAJORITY:.3f} to "
  f"{MAJORITY:.4f}. Note also that the label is rarer in the test years than in "
  "the training years, which matters later.")

# ==================================================================== Q3
q("Q3", "The same model, the new label",
  "Fit the one-column scaled pipeline on `vol_20d` predicting `jump`, store "
  "the test probabilities as `p_jump`, and print the accuracy beside "
  "`majority` from Q2. Store the accuracy as `acc_jump`.",
  "jump_model = ...\n...\np_jump = ...\nacc_jump = ...\n\n"
  "print('accuracy:', acc_jump)\nprint('majority:', majority)",
  ["The pipeline is the one from Q1 with `train['jump']` as the target.",
   "`acc_jump = round(accuracy_score(test['jump'], jump_model.predict(test[['vol_20d']])), 4)`."],
  "jump_model = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
  "jump_model.fit(train[['vol_20d']], train['jump'])\n"
  "p_jump = jump_model.predict_proba(test[['vol_20d']])[:, 1]\n"
  "acc_jump = round(accuracy_score(test['jump'], jump_model.predict(test[['vol_20d']])), 4)\n\n"
  "print('accuracy:', acc_jump)\nprint('majority:', majority)",
  f"{ACC:.4f} and {MAJORITY:.4f}. They are not close, they are **identical**, "
  "and that is not a coincidence. Q4 finds out why.")

# ==================================================================== Q4
q("Q4", "Why the two numbers are the same",
  "Print how many test days the model predicted as a jump, and the smallest "
  "and largest probability it gave. Then say, in a comment, why the accuracy "
  "had to equal the majority rule.",
  "called = ...\n\nprint('days called a jump:', ...)\n"
  "print('lowest probability :', ...)\nprint('highest probability:', ...)\n\n"
  "# why is the accuracy exactly the majority rule?",
  "`called = jump_model.predict(test[['vol_20d']])`, then `called.sum()`. "
  "Compare the largest probability with one half.",
  "called = jump_model.predict(test[['vol_20d']])\n\n"
  "print('days called a jump:', int(called.sum()))\n"
  "print('lowest probability :', round(float(p_jump.min()), 4))\n"
  "print('highest probability:', round(float(p_jump.max()), 4))\n\n"
  "# No test day reaches a probability of 0.5, so .predict() returns 0 on every\n"
  "# day. Predicting 0 everywhere IS the majority rule, so the two accuracies\n"
  "# are the same number by construction.",
  f"{N_PRED_1} days called, with probabilities running from {P_MIN:.4f} to "
  f"{P_MAX:.4f}. The highest is below one half, so `.predict()` says no on "
  "every single day. The model has not failed to learn; it has been asked a "
  "question at a threshold that makes its answer always the same.")

# ==================================================================== Q5
q("Q5", "The ranking underneath",
  "Score the same probabilities with the AUC and with the average precision, "
  "and print the share of jumps beside the second one. Store the AUC as "
  "`auc_jump`.",
  "auc_jump = ...\nap_jump = ...\n\n"
  "print('AUC              :', auc_jump)\nprint('average precision:', ap_jump)\n"
  "print('share of jumps   :', ...)",
  "`roc_auc_score(test['jump'], p_jump)` and "
  "`average_precision_score(test['jump'], p_jump)`. The no-information line "
  "for average precision is the share of ones, not 0.5.",
  "auc_jump = round(roc_auc_score(test['jump'], p_jump), 4)\n"
  "ap_jump = round(average_precision_score(test['jump'], p_jump), 4)\n\n"
  "print('AUC              :', auc_jump)\nprint('average precision:', ap_jump)\n"
  "print('share of jumps   :', round(test['jump'].mean(), 4))",
  f"An AUC of {AUC:.4f} and an average precision of {AP:.4f} against "
  f"{SHARE_TE:.4f} for guessing. The ranking is better than Part 8's "
  f"{P8_AUC:.3f} on the easier label: the model knows perfectly well which "
  "days are dangerous. Everything that went wrong in Q3 and Q4 was the "
  "threshold, and the threshold is not part of the model.")

# ==================================================================== Q6
q("Q6", "Putting a price on the two mistakes",
  "The desk prices a missed jump at 5 and a false alarm at 1. Sweep the "
  "threshold from 0.05 to 0.60 in steps of 0.01 **on the training rows**, "
  "pick the cheapest, and store it as `chosen`. Print it beside the value the "
  "cost formula gives.",
  "p_train = ...\ngrid = np.arange(0.05, 0.60, 0.01)\ncosts = []\n\n"
  "for threshold in grid:\n    ...\n\nchosen = ...\n\n"
  "print('chosen on the training rows:', chosen)\nprint('the formula says       :', round(1 / (1 + 5), 3))",
  ["`p_train = jump_model.predict_proba(train[['vol_20d']])[:, 1]`. Inside the "
   "loop, build the 0/1 calls, unpack `confusion_matrix(...).ravel()` into "
   "`tn, fp, fn, tp`, and append `5 * fn + fp`.",
   "`chosen = round(float(grid[np.array(costs).argmin()]), 2)`."],
  "p_train = jump_model.predict_proba(train[['vol_20d']])[:, 1]\n"
  "grid = np.arange(0.05, 0.60, 0.01)\ncosts = []\n\n"
  "for threshold in grid:\n"
  "    tn, fp, fn, tp = confusion_matrix(train['jump'], (p_train >= threshold).astype(int)).ravel()\n"
  "    costs.append(5 * fn + fp)\n\n"
  "chosen = round(float(grid[np.array(costs).argmin()]), 2)\n\n"
  "print('chosen on the training rows:', chosen)\nprint('the formula says       :', round(1 / (1 + 5), 3))",
  f"The training rows choose {CHOSEN:.2f}, against {FORMULA_THR:.3f} from the "
  "formula. The two differ because the formula assumes the probabilities are "
  "calibrated, and Q9 shows that on this label they are not. Either way, both "
  "are far below one half.")

# ==================================================================== Q7
q("Q7", "What the chosen threshold does",
  "Apply `chosen` to the test probabilities. Print the confusion matrix, the "
  "recall, the precision, the accuracy, and the cost, and compare the cost "
  "with the cost at one half.",
  "called_chosen = ...\ncalled_half = ...\nmatrix = ...\ncost_chosen = ...\ncost_half = ...\n\nprint(matrix)\nprint('recall   :', ...)\nprint('precision:', ...)\nprint('accuracy :', ...)\nprint('cost at the chosen threshold:', cost_chosen)\nprint('cost at one half            :', cost_half)",
  ["`called_chosen = (p_jump >= chosen).astype(int)` and "
   "`called_half = (p_jump >= 0.5).astype(int)`.",
   "For each cost, unpack `confusion_matrix(test['jump'], called).ravel()` into "
   "`tn, fp, fn, tp` and compute `5 * fn + fp`."],
  "called_chosen = (p_jump >= chosen).astype(int)\ncalled_half = (p_jump >= 0.5).astype(int)\nmatrix = confusion_matrix(test['jump'], called_chosen)\n\ntn, fp, fn, tp = matrix.ravel()\ntn2, fp2, fn2, tp2 = confusion_matrix(test['jump'], called_half).ravel()\ncost_chosen = 5 * fn + fp\ncost_half = 5 * fn2 + fp2\n\nprint(matrix)\nprint('recall   :', round(recall_score(test['jump'], called_chosen), 4))\nprint('precision:', round(precision_score(test['jump'], called_chosen, zero_division=0), 4))\nprint('accuracy :', round(accuracy_score(test['jump'], called_chosen), 4))\nprint('cost at the chosen threshold:', cost_chosen)\nprint('cost at one half            :', cost_half)",
  f"At {CHOSEN:.2f} the model calls {N_CALLED_CHOSEN} of the {N_TEST} test days "
  f"and catches {CM_CHOSEN[1, 1]} of the {N_JUMP_TE} jumps, so the recall is "
  f"{REC_CHOSEN:.3f} and the precision {PREC_CHOSEN:.3f}. The accuracy has "
  f"**fallen** from {ACC:.3f} to {ACC_CHOSEN:.3f}, well below the majority "
  f"rule, while the cost has fallen from {COST_AT_HALF} to {COST_AT_CHOSEN}. "
  "The model got worse by the number Part 8 reported and better by the number "
  "the desk actually pays.")

# ==================================================================== Q8
q("Q8", "The other way to move the cut",
  "Fit the same pipeline again with `class_weight='balanced'` on the "
  "classifier, and print its accuracy, recall, how many days it calls, and its "
  "AUC. Compare each with the plain model's.",
  "weighted = ...\n...\npred_w = ...\n\n"
  "print('accuracy:', ..., 'was', acc_jump)\nprint('recall  :', ...)\n"
  "print('days called:', ...)\nprint('AUC     :', ..., 'was', auc_jump)",
  "`LogisticRegression(class_weight='balanced')` inside the pipeline. The AUC "
  "still needs `predict_proba(...)[:, 1]`.",
  "weighted = Pipeline([('scale', StandardScaler()),\n"
  "                     ('logit', LogisticRegression(class_weight='balanced'))])\n"
  "weighted.fit(train[['vol_20d']], train['jump'])\npred_w = weighted.predict(test[['vol_20d']])\n\n"
  "print('accuracy:', round(accuracy_score(test['jump'], pred_w), 4), 'was', acc_jump)\n"
  "print('recall  :', round(recall_score(test['jump'], pred_w), 4))\n"
  "print('days called:', int(pred_w.sum()))\n"
  "print('AUC     :', round(roc_auc_score(test['jump'], weighted.predict_proba(test[['vol_20d']])[:, 1]), 4),\n"
  "      'was', auc_jump)",
  f"The weights catch every one of the {N_JUMP_TE} jumps, a recall of "
  f"{REC_BAL:.1f}, by calling {N_CALLED_BAL} of the {N_TEST} days, and the "
  f"accuracy falls to {ACC_BAL:.4f}. The AUC is {AUC_BAL:.4f}, the same as "
  "before: the weights moved the cut, not the ranking. A threshold does the "
  "same job and can be set to any value, rather than the one the class sizes "
  "happen to imply.")

# ==================================================================== Q9
q("Q9", "Do the probabilities mean what they say",
  "Put the test days into the buckets 0 to 0.1, 0.1 to 0.2, 0.2 to 0.3 and 0.3 "
  "to 0.5 with `pd.cut`, and report how many days are in each and what share "
  "of them jumped. Then print the average probability beside the share that "
  "jumped.",
  "checked = ...\nby_bucket = ...\n\nprint(by_bucket)\n\nprint('average probability:', ...)\nprint('share that jumped  :', ...)",
  ["`pd.cut(checked['p'], [0, 0.1, 0.2, 0.3, 0.5])`.",
   "`checked.groupby('bucket', observed=True)['jump'].agg(['size', 'mean']).round(3)`."],
  "checked = test.copy()\nchecked['p'] = p_jump\nchecked['bucket'] = pd.cut(checked['p'], [0, 0.1, 0.2, 0.3, 0.5])\nby_bucket = checked.groupby('bucket', observed=True)['jump'].agg(['size', 'mean']).round(3)\n\nprint(by_bucket)\n\nprint('average probability:', round(float(p_jump.mean()), 4))\nprint('share that jumped  :', round(float(test['jump'].mean()), 4))",
  f"The model gives {MEAN_P:.4f} on average where {SHARE_TE:.4f} of the days "
  f"jumped, so it overstates by a factor of nearly three. Of the "
  f"{int(BUCKETS['size'].iloc[-1])} days it put between 0.3 and 0.5, "
  f"{BUCKETS['mean'].iloc[-1]:.3f} jumped; of the "
  f"{int(BUCKETS['size'].iloc[1])} days between 0.1 and 0.2, none did. The "
  "ranking is right and the level is wrong.")

# ==================================================================== Q10
q("Q10", "Whether calibration can repair it",
  "Print the Brier score of the plain model and of the weighted one from Q8. "
  "Then wrap the plain pipeline in `CalibratedClassifierCV` with "
  "`method='sigmoid'` and `cv=5`, fit it on the training rows, and print its "
  "average probability and Brier score.",
  "print('plain   :', ...)\nprint('weighted:', ...)\n\n"
  "fixed = ...\n...\np_fixed = ...\n\n"
  "print('calibrated average:', ...)\nprint('calibrated Brier  :', ...)",
  ["`brier_score_loss(test['jump'], p_jump)`, and the same with the weighted "
   "model's probabilities.",
   "`CalibratedClassifierCV(Pipeline([...]), method='sigmoid', cv=5)`, fitted "
   "on `train[['vol_20d']]` and `train['jump']`."],
  "print('plain   :', round(brier_score_loss(test['jump'], p_jump), 4))\n"
  "print('weighted:', round(brier_score_loss(test['jump'],\n"
  "      weighted.predict_proba(test[['vol_20d']])[:, 1]), 4))\n\n"
  "fixed = CalibratedClassifierCV(\n"
  "    Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())]),\n"
  "    method='sigmoid', cv=5)\nfixed.fit(train[['vol_20d']], train['jump'])\n"
  "p_fixed = fixed.predict_proba(test[['vol_20d']])[:, 1]\n\n"
  "print('calibrated average:', round(float(p_fixed.mean()), 4))\n"
  "print('calibrated Brier  :', round(brier_score_loss(test['jump'], p_fixed), 4))",
  f"{BRIER:.4f} for the plain model and {BRIER_BAL:.4f} for the weighted one, "
  f"and calibrating moves the average only from {MEAN_P:.4f} to "
  f"{MEAN_CAL:.4f} and the Brier score from {BRIER:.4f} to {BRIER_CAL:.4f}. "
  "This is worth sitting with. `CalibratedClassifierCV` learns its correction "
  f"on the training years, where {SHARE_TR:.1%} of days jumped, and applies it "
  f"to test years where {SHARE_TE:.1%} did. The model is not miscalibrated "
  "because it was built wrongly; it is miscalibrated because the world got "
  "calmer, and no amount of arithmetic on the training rows can know that.")

# ==================================================================== Q11
q("Q11", "Twenty columns, one more time",
  "Search `C` over `[0.0001, 0.001, 0.01, 0.1, 1, 10]` on the twenty columns "
  "with the time folds and `scoring='roc_auc'`, and print the winning `C`, its "
  "mean fold score, and its test AUC beside the one-column model's.",
  "grid = ...\nsearch = ...\n...\n\nauc_wide = ...\n\n"
  "print('best C  :', ...)\nprint('fold AUC:', ...)\n"
  "print('test AUC:', auc_wide, 'against', auc_jump, 'for one column')",
  ["The grid key is the step name, two underscores, the argument: "
   "`{'logit__C': [...]}`. Use `max_iter=1000` on the classifier.",
   "`search.best_params_`, `search.best_score_`, and "
   "`roc_auc_score(test['jump'], search.predict_proba(test[columns])[:, 1])`."],
  "grid = {'logit__C': [0.0001, 0.001, 0.01, 0.1, 1, 10]}\n"
  "search = GridSearchCV(\n"
  "    Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression(max_iter=1000))]),\n"
  "    grid, cv=folds, scoring='roc_auc')\nsearch.fit(train[columns], train['jump'])\n\n"
  "auc_wide = round(roc_auc_score(test['jump'], search.predict_proba(test[columns])[:, 1]), 4)\n\n"
  "print('best C  :', search.best_params_)\nprint('fold AUC:', round(search.best_score_, 4))\n"
  "print('test AUC:', auc_wide, 'against', auc_jump, 'for one column')",
  f"C of {BEST_C} wins the folds with {BEST_CV:.4f}, and scores "
  f"{AUC_WIDE:.4f} on the test days against {AUC:.4f} for the single column. "
  "The twenty columns lose on the jump label exactly as they lost on the "
  "volatility forecast in Part 6 and on the rising label in Part 8. Three "
  "parts, three answers, the same answer.")

# ==================================================================== Q12
q("Q12", "Three answers instead of two",
  "Build `move` with `pd.cut` on the ratio `vol_next / vol_20d`, with cuts at "
  "0.85 and 1.25 and the names `falls`, `stays`, `rises`. Split again, fit the "
  "one-column pipeline with `max_iter=1000`, and print the accuracy, the "
  "baseline from the most common training label, and the macro F1.",
  "# add the ratio and the move label to table, then split again\n...\ntrain = ...\ntest = ...\n\nthree = ...\n...\npred_move = ...\n\nprint('accuracy:', ...)\nprint('baseline:', ...)\nprint('macro F1:', ...)",
  ["`pd.cut(ratio, [-np.inf, 0.85, 1.25, np.inf], labels=['falls', 'stays', 'rises'])`.",
   "The baseline is `(test['move'] == train['move'].value_counts().idxmax()).mean()`, "
   "and the macro F1 is `f1_score(test['move'], pred_move, average='macro')`."],
  "ratio = table['vol_next'] / table['vol_20d']\ntable['move'] = pd.cut(ratio, [-np.inf, 0.85, 1.25, np.inf],\n                       labels=['falls', 'stays', 'rises'])\ntrain = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\n\nthree = Pipeline([('scale', StandardScaler()),\n                  ('logit', LogisticRegression(max_iter=1000))])\nthree.fit(train[['vol_20d']], train['move'])\npred_move = three.predict(test[['vol_20d']])\n\nprint('accuracy:', round(accuracy_score(test['move'], pred_move), 4))\nprint('baseline:', round((test['move'] == train['move'].value_counts().idxmax()).mean(), 4))\nprint('macro F1:', round(f1_score(test['move'], pred_move, average='macro'), 4))",
  f"{ACC_MOVE:.4f} against a baseline of {MAJ_MOVE_ACC:.4f}, with a macro F1 "
  f"of {F1_MOVE:.4f}. Recall is {REC_MOVE[2]:.3f} for `rises` but only "
  f"{REC_MOVE[0]:.3f} for `falls`: on Apple the model finds the busy months "
  "and cannot pick out the calm ones, which is the opposite of what the index "
  "did in the lecture.")

# ==================================================================== Q13
q("Q13", "A second classifier on the same folds",
  "Search `k` over `[1, 5, 15, 51, 101, 151, 201, 301]` for "
  "`KNeighborsClassifier` in a scaled pipeline, on `vol_20d` predicting "
  "`jump`, with the same folds and `scoring='roc_auc'`. Print the winning `k`, "
  "its fold score and its test AUC, and put the logistic model's fold score "
  "beside it.",
  "# put the jump label back on table and split again\n...\ntrain = ...\ntest = ...\n\nknn_search = ...\n...\nlogistic_folds = ...\n\nprint('best k   :', ...)\nprint('k-NN folds:', ...)\nprint('k-NN test :', ...)\nprint('logistic folds:', ...)",
  ["`GridSearchCV(Pipeline([('scale', StandardScaler()), ('knn', "
   "KNeighborsClassifier())]), {'knn__n_neighbors': [...]}, cv=folds, "
   "scoring='roc_auc')`.",
   "`logistic_folds = cross_val_score(Pipeline([...LogisticRegression()...]), "
   "train[['vol_20d']], train['jump'], cv=folds, scoring='roc_auc').mean()`."],
  "table['jump'] = (table['vol_next'] > 1.5 * table['vol_20d']).astype(int)\ntrain = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\n\nknn_search = GridSearchCV(\n    Pipeline([('scale', StandardScaler()), ('knn', KNeighborsClassifier())]),\n    {'knn__n_neighbors': [1, 5, 15, 51, 101, 151, 201, 301]},\n    cv=folds, scoring='roc_auc')\nknn_search.fit(train[['vol_20d']], train['jump'])\n\nlogistic_folds = cross_val_score(\n    Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())]),\n    train[['vol_20d']], train['jump'], cv=folds, scoring='roc_auc').mean()\n\nprint('best k   :', knn_search.best_params_)\nprint('k-NN folds:', round(knn_search.best_score_, 4))\nprint('k-NN test :', round(roc_auc_score(test['jump'],\n      knn_search.best_estimator_.predict_proba(test[['vol_20d']])[:, 1]), 4))\nprint('logistic folds:', round(logistic_folds, 4))",
  f"k of {BEST_K} wins its own search with {BEST_K_CV:.4f} on the folds, "
  f"against {AUC_FOLDS_MEAN:.4f} for logistic regression, so the folds keep "
  f"logistic regression. On the test days it is {AUC_KNN:.4f} against "
  f"{AUC:.4f}, the same ordering. One column and a boundary that is a single "
  "number is all this problem has ever needed.")

# ==================================================================== Q14
q("Q14", "Across the desk",
  "For every instrument, build the same one-column jump model and collect "
  "three things in dictionaries: the share of test days that jumped, the test "
  "AUC, and whether the model calls no jump at all at a threshold of one half. "
  "Print the count of silent instruments and the AUCs sorted, largest first.",
  "desk_share = {}\ndesk_auc = {}\nsilent = []\n\nfor t in TICKERS:\n    ...\n\n"
  "print('silent at 0.5:', len(silent), 'of', len(TICKERS))\nprint(silent)\n"
  "for t in sorted(desk_auc, key=desk_auc.get, reverse=True):\n    print(t, desk_auc[t])",
  ["Inside the loop, rebuild the two columns for that ticker: `vol_20d` is "
   "`rets[t].rolling(20).std()` and `vol_next` is the same shifted by -20, then "
   "`dropna()` and the 1.5 comparison.",
   "`if model_t.predict(te[['vol_20d']]).sum() == 0: silent.append(t)`."],
  "desk_share = {}\ndesk_auc = {}\nsilent = []\n\nfor t in TICKERS:\n"
  "    frame = pd.DataFrame({'vol_20d': rets[t].rolling(20).std()})\n"
  "    frame['vol_next'] = rets[t].rolling(20).std().shift(-20)\n"
  "    frame = frame.dropna()\n"
  "    frame['jump'] = (frame['vol_next'] > 1.5 * frame['vol_20d']).astype(int)\n"
  "    tr = frame.loc[:'2022-12-31']\n    te = frame.loc['2023-01-01':]\n"
  "    model_t = Pipeline([('scale', StandardScaler()), ('logit', LogisticRegression())])\n"
  "    model_t.fit(tr[['vol_20d']], tr['jump'])\n"
  "    desk_share[t] = round(float(te['jump'].mean()), 3)\n"
  "    desk_auc[t] = round(float(roc_auc_score(te['jump'],\n"
  "                        model_t.predict_proba(te[['vol_20d']])[:, 1])), 3)\n"
  "    if model_t.predict(te[['vol_20d']]).sum() == 0:\n        silent.append(t)\n\n"
  "print('silent at 0.5:', len(silent), 'of', len(TICKERS))\nprint(silent)\n"
  "for t in sorted(desk_auc, key=desk_auc.get, reverse=True):\n    print(t, desk_auc[t])",
  f"{N_SILENT} of the 11 instruments call no jump at all at a threshold of one "
  f"half, and {N_ABOVE} of the 11 rank at 0.75 or better, from "
  f"{DESK_AUC[DESK_BEST]:.3f} on {DESK_BEST} down to "
  f"{DESK_AUC[DESK_WORST]:.3f} on {DESK_WORST}. Apple was not a special case: "
  "the default threshold silences most of the desk, and the ranking is "
  "usable on most of it.")

# ==================================================================== Q15
q("Q15", "Draw the decision",
  "Draw the test probabilities over time as a line, with a horizontal line at "
  "`chosen` and another at 0.5, and mark the days that actually jumped. Label "
  "both axes and give the figure a title.",
  "fig, ax = plt.subplots(figsize=(10, 3.4))\n\n...\n...\n...\n\n"
  "ax.set_xlabel('date')\nax.set_ylabel('probability of a jump')\nplt.show()",
  ["`ax.plot(test.index, p_jump)` for the line, and two `ax.axhline(...)` "
   "calls for the thresholds.",
   "For the jumps: `days = test.index[test['jump'] == 1]`, then "
   "`ax.scatter(days, p_jump[test['jump'].values == 1], s=14, color='#b3402f')`."],
  "fig, ax = plt.subplots(figsize=(10, 3.4))\n\n"
  "ax.plot(test.index, p_jump, linewidth=1.4, color='#1c5cab')\n"
  "days = test.index[test['jump'] == 1]\n"
  "ax.scatter(days, p_jump[test['jump'].values == 1], s=14, color='#b3402f', zorder=3)\n"
  "ax.axhline(chosen, color='#b8860b', linestyle='--')\n"
  "ax.axhline(0.5, color='grey', linestyle=':')\n\n"
  "ax.set_xlabel('date')\nax.set_ylabel('probability of a jump')\n"
  "ax.set_title('Apple: the probability of a jump, the chosen threshold, and the jumps',\n"
  "             loc='left')\nplt.show()",
  "The dotted line at one half sits above everything, which is Q4 as a "
  f"picture. The dashed line at {CHOSEN:.2f} cuts through the series, and most "
  "of the marked jumps sit above it. The probabilities move slowly, so the "
  "warnings come in runs rather than one day at a time.")

# ==================================================================== Q16
q("Q16", "Write down what you would defend",
  "Collect what Part 9 established into a dictionary `report`, and write "
  "`summarise(report)` printing one line per entry and ending with a verdict: "
  "whether the desk should run this model on Apple, and at what threshold.",
  "report = {\n    'label': ...,\n    'auc': ...,\n    'accuracy_at_half': ...,\n"
  "    'days_called_at_half': ...,\n    'chosen_threshold': ...,\n    'recall_at_chosen': ...,\n"
  "    'cost_at_chosen': ...,\n    'cost_at_half': ...,\n    'calibrated': ...,\n"
  "    'silent_instruments': ...,\n}\n\n\ndef summarise(r):\n    \"\"\"...\"\"\"\n    ...\n\n\n"
  "summarise(report)",
  ["Every value is a number or a short string you already printed. For "
   "`calibrated`, a short string such as `'no: says 0.23 where 0.09 happens'`.",
   "Inside the function, `for key, value in r.items():` and a `print` with an "
   "f-string, then an `if` on the AUC for the verdict."],
  "report = {\n    'label': 'jump: vol_next > 1.5 x vol_20d',\n"
  "    'auc': auc_jump,\n    'accuracy_at_half': acc_jump,\n"
  "    'days_called_at_half': 0,\n    'chosen_threshold': chosen,\n"
  f"    'recall_at_chosen': {REC_CHOSEN:.3f},\n"
  f"    'cost_at_chosen': {COST_AT_CHOSEN},\n    'cost_at_half': {COST_AT_HALF},\n"
  "    'calibrated': 'no: says 0.23 on average where 0.09 happens',\n"
  f"    'silent_instruments': '{N_SILENT} of 11 at a threshold of 0.5',\n" + "}\n\n\n"
  "def summarise(r):\n    \"\"\"Print the report and the verdict it supports.\"\"\"\n"
  "    for key, value in r.items():\n        print(f\"{key:<22}{value}\")\n    print()\n"
  "    if r['auc'] >= 0.75:\n"
  "        print(f\"Verdict: the ranking is usable. Run it at {r['chosen_threshold']},\")\n"
  "        print(\"not at one half, and treat the probabilities as an order, not a level.\")\n"
  "    else:\n        print('Verdict: the ranking is too weak to act on.')\n\n\n"
  "summarise(report)",
  "The honest report has two halves that point in opposite directions. The "
  f"model ranks Apple's dangerous months well, with an AUC of {AUC:.3f} and "
  f"{N_ABOVE} of 11 instruments above 0.75. It is also unusable as shipped: at "
  "one half it predicts nothing, its probabilities run about three times too "
  "high, and calibrating on the training years cannot fix that. What you "
  f"defend is the ranking plus a threshold of {CHOSEN:.2f} chosen from the "
  "desk's own costs, reviewed whenever the costs change.")

# ---------------------------------------------------------------- closing
md(
"## \U0001f4e6 What you now have\n\n"
"| what | where |\n|:--|:--|\n"
"| the table and split carried from Part 8 | `table`, `columns`, `train`, `test` |\n"
"| the rare label and its baseline | `table['jump']`, `majority` |\n"
"| the one-column classifier and its probabilities | `jump_model`, `p_jump`, `auc_jump`, `acc_jump` |\n"
"| the threshold chosen from costs | `chosen`, `called_chosen` |\n"
"| the weighted alternative | `weighted`, `pred_w` |\n"
"| the calibration check | `by_bucket`, `fixed`, `p_fixed` |\n"
"| twenty columns with C chosen | `search`, `auc_wide` |\n"
"| three classes | `table['move']`, `three`, `pred_move` |\n"
"| the second classifier | `knn_search`, `logistic_folds` |\n"
"| the whole desk | `desk_share`, `desk_auc`, `silent` |\n"
"| the thing you would defend | `report` |"
)

md(
"## What changed since Part 8\n\n"
f"- **The label became rare.** {SHARE_TE:.1%} of test days rather than about "
f"half, so the rule to beat rose from {P8_MAJORITY:.3f} to {MAJORITY:.4f}.\n"
f"- **Accuracy stopped working.** The model scores {ACC:.4f}, exactly the "
"majority rule, while predicting no jump on any day, because no probability "
f"reaches one half (Q3 and Q4). Its AUC is {AUC:.3f}.\n"
f"- **The threshold became a decision.** Priced at 5 for a miss and 1 for a "
f"false alarm, the training rows choose {CHOSEN:.2f}, which catches "
f"{REC_CHOSEN:.0%} of the jumps and cuts the cost from {COST_AT_HALF} to "
f"{COST_AT_CHOSEN} (Q6 and Q7).\n"
f"- **The probabilities are not calibrated, and cannot be fixed here.** The "
f"model says {MEAN_P:.2f} on average where {SHARE_TE:.2f} happens, because "
f"{SHARE_TR:.0%} of the training days jumped against {SHARE_TE:.0%} of the "
"test days (Q9 and Q10).\n"
f"- **The wide model lost again.** {AUC_WIDE:.3f} against {AUC:.3f} with C "
"chosen on the folds (Q11), as in Parts 6 and 8.\n"
f"- **The desk agrees.** {N_SILENT} of 11 instruments are silent at one half, "
f"and {N_ABOVE} of 11 rank at 0.75 or better (Q14)."
)

md(
"## Where this leaves the risk report\n\n"
"The report can now say three separate things and keep them separate: how well "
"the model ranks months by danger, where the desk has chosen to act given what "
"each mistake costs it, and how much trust the probability itself deserves. "
"Part 8 could only say the first. The third is the uncomfortable one on this "
"data, because the answer is that the level cannot be trusted on a period when "
"the base rate has moved.\n\n"
"**Next part:** the whole classification workflow on one instrument from "
"beginning to end, as a single investigation rather than a sequence of "
"questions."
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
