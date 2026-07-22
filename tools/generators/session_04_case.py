# -*- coding: utf-8 -*-
"""Build session_04_case.ipynb  (The Analyst's Notebook, Part 4).

Conventions (approved for Sessions 1 to 3): no star badges, one cumulative
investigation where later questions reuse what earlier ones stored, folded
hints and solutions, stated formulas, plain tone, no em-dashes. Opens with a
QUICK LOAD restoring Part 3's findings.

Part 4 turns the finished risk report into a prediction problem and stops
exactly before fitting anything. The target is next month's realised
volatility, which is the natural continuation of a risk report and, unlike a
return, is genuinely forecastable. Nothing here is copy-pasteable from the
lecture: the deck predicted Apple's return from the market's, same day.

BLANK-SAFE, and this one needs care because the case is cumulative: no
pre-written line may CALL anything on a variable an earlier question produced.
Every such dependency sits inside the student's own blank, so a placeholder
just propagates.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "session_04" / "session_04_case.ipynb"

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

# what Part 3 finished with
AVOL24 = (R.loc["2024"].std() * np.sqrt(252)).sort_values(ascending=False)


def build(ticker):
    frame = pd.DataFrame({
        "vol_20d": R[ticker].rolling(20).std(),
        "ret_20d": R[ticker].rolling(20).mean(),
        "up_20d": (R[ticker] > 0).rolling(20).mean(),
    })
    frame["vol_next"] = R[ticker].rolling(20).std().shift(-20)
    return frame


RAW = build("AAPL")
TBL = RAW.dropna()
N_RAW, N_TBL = len(RAW), len(TBL)
N_COLS = TBL.shape[1]
TARGET_MEAN, TARGET_MED, TARGET_MAX = (TBL["vol_next"].mean(),
                                       TBL["vol_next"].median(),
                                       TBL["vol_next"].max())
WORST_DATE = TBL["vol_next"].idxmax()
TRAIN, TEST = TBL.loc[:"2022-12-31"], TBL.loc["2023-01-01":]
N_TRAIN, N_TEST = len(TRAIN), len(TEST)


def rmse(y, p):
    return float(np.sqrt(((np.asarray(y) - np.asarray(p)) ** 2).mean()))


def mae(y, p):
    return float(np.abs(np.asarray(y) - np.asarray(p)).mean())


BASE_PRED = np.full(N_TEST, TRAIN["vol_next"].mean())
BASE_RMSE, BASE_MAE = rmse(TEST["vol_next"], BASE_PRED), mae(TEST["vol_next"], BASE_PRED)
PERS_RMSE, PERS_MAE = rmse(TEST["vol_next"], TEST["vol_20d"]), mae(TEST["vol_next"], TEST["vol_20d"])
PERS_R2 = float(1 - ((TEST["vol_next"] - TEST["vol_20d"]) ** 2).sum() /
                ((TEST["vol_next"] - TRAIN["vol_next"].mean()) ** 2).sum())

R2_BY = {}
for _t in TICKERS:
    _d = build(_t).dropna()
    _tr, _te = _d.loc[:"2022-12-31"], _d.loc["2023-01-01":]
    R2_BY[_t] = float(1 - ((_te["vol_next"] - _te["vol_20d"]) ** 2).sum() /
                      ((_te["vol_next"] - _tr["vol_next"].mean()) ** 2).sum())
R2_S = pd.Series(R2_BY).sort_values(ascending=False)
N_POSITIVE = int((R2_S > 0).sum())

THRESHOLD = float(TRAIN["vol_next"].median())
HIGH_TRUE = TEST["vol_next"] > THRESHOLD
HIGH_PRED = TEST["vol_20d"] > THRESHOLD
TP = int((HIGH_PRED & HIGH_TRUE).sum())
FP = int((HIGH_PRED & ~HIGH_TRUE).sum())
FN = int((~HIGH_PRED & HIGH_TRUE).sum())
TN = int((~HIGH_PRED & ~HIGH_TRUE).sum())
BASE_RATE = float(HIGH_TRUE.mean())
ACC = (TP + TN) / N_TEST
PREC = TP / (TP + FP)
REC = TP / (TP + FN)

# ---------------------------------------------------------------- top matter
md(
"# \U0001f4bc The Analyst's Notebook · Part 4\n"
"### From a report to a question\n\n"
"Parts 1 to 3 built a risk report: two stocks compared by hand, then a proper "
"volatility built out of loops, then eleven instruments measured, ranked, drawn "
"and labelled with pandas.\n\n"
"Every number in it describes a period that has already happened. The desk has "
"read it, and asked the obvious next question:\n\n"
"> **Which of these names will be volatile next month?**\n\n"
"That is a different kind of question, and answering it responsibly takes more "
"setup than most people expect. Part 4 is that setup, done properly and stopping "
"exactly where a model would begin."
)

md(
"## How to work through this\n\n"
"- Run the **quick load** cell first. It brings back what Part 3 found and loads "
"the price table.\n"
"- Each question builds on the last, so keep them in order and keep your "
"variables. Later questions use the names earlier ones created.\n"
"- Cells with `...` are blanks. The notebook runs cleanly even before you fill "
"them in, so **Run all** is always safe.\n"
"- Hints and solutions are folded under each question. Work first, then check.\n\n"
"**You will not fit a model here.** You will define the problem so precisely "
"that fitting one becomes the easy part, which is the honest order to do it in.\n\n"
"*Stuck for more than 15 minutes? Ask a friend, ask an AI for a hint (not the "
"answer), or email me at `jobo@econ.au.dk`.*"
)

md("---")

# ------------------------------------------------------------- quick load
md(
"## ⚙️ Quick load\n\n"
"The packages, the price table, and what Part 3 left you. Run it and read what it "
"prints."
)

# The recap has to be the numbers Part 3 actually produced, so build the
# literal from the data rather than typing it out.
PART3_DICT = "\n".join(f'    "{_t}": {_v:.3f},' for _t, _v in AVOL24.items())

code(
'''import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# --- What Part 3 found: annualised volatility in 2024 ---
part3_annual_vol = {
''' + PART3_DICT + '''
}
part3_riskiest = "''' + AVOL24.index[0] + '''"
part3_calmest = "''' + AVOL24.index[-1] + '''"

print("Loaded prices:", prices.shape[0], "rows")
print("Instruments  :", ", ".join(TICKERS))
print()
print("Part 3 recap: annualised volatility in 2024")
for ticker, value in part3_annual_vol.items():
    print(f"  {ticker:5} {value:.3f}")
print(f"  riskiest: {part3_riskiest}   calmest: {part3_calmest}")
print()
print("Part 3 also found that the ranking only partly carried from 2015-2022")
print("into 2023-2024. That gap is what Part 4 is about.")'''
)

md("---")

# ==================================================================== Q1
q("Q1", "Say exactly what you are predicting",
  "\"Will it be volatile next month?\" is not yet a question a computer can answer. "
  "Pin it down.\n\n"
  "The target is **the volatility of the next twenty trading days**: the standard "
  "deviation of the daily returns from tomorrow to twenty days from now. Build it "
  "for Apple, as a Series called `target`.\n\n"
  "$$\\text{target}_t = \\text{sd}\\big(r_{t+1},\\, r_{t+2},\\, \\ldots,\\, r_{t+20}\\big)$$",
  "target = ...\ntarget",
  ["A rolling standard deviation is `rets['AAPL'].rolling(20).std()`. That is the "
   "volatility of the twenty days **ending** today.",
   "To move it so that each row holds the *next* twenty days instead, shift it "
   "backwards: `.shift(-20)`."],
  "target = rets['AAPL'].rolling(20).std().shift(-20)\ntarget",
  "One number per day, and every one of them describes a period that had not "
  "happened yet on that date. That is exactly what makes it a target rather than a "
  "feature.\n\n"
  "*A note on why volatility and not return: returns are close to unpredictable, "
  "and a case study that ends in `R2 = -0.02` teaches you less. Volatility "
  "clusters, so there is something real to find.*")

# ==================================================================== Q2
q("Q2", "Only what you would have known",
  "Now the features. Each one must be computable **on the day itself**, using "
  "nothing from the future. Build three, each over the last twenty trading days:\n\n"
  "- `vol_20d`: the volatility of the returns up to and including today\n"
  "- `ret_20d`: the average daily return over the same window\n"
  "- `up_20d`: the share of those days that were up\n\n"
  "Collect them into a DataFrame called `features`, with exactly those three "
  "column names.",
  "features = ...\nfeatures",
  ["The first two are `rets['AAPL'].rolling(20).std()` and "
   "`rets['AAPL'].rolling(20).mean()`.",
   "For the third, `(rets['AAPL'] > 0)` gives True and False, and a rolling mean of "
   "those is the share that were True."],
  "features = pd.DataFrame({\n    'vol_20d': rets['AAPL'].rolling(20).std(),\n"
  "    'ret_20d': rets['AAPL'].rolling(20).mean(),\n"
  "    'up_20d': (rets['AAPL'] > 0).rolling(20).mean(),\n})\nfeatures",
  "Note the asymmetry: the features look **backwards** twenty days, the target "
  "looks **forwards** twenty days, and they never overlap. Draw it on paper if it "
  "helps. Getting this wrong is the most expensive mistake in applied finance, and "
  "it is invisible in the output.")

# ==================================================================== Q3
q("Q3", "The feature that knew too much",
  "A colleague built the table below while you were in the meeting. It has one "
  "feature and the same target, and they are pleased with it, because the two "
  "columns line up almost perfectly.\n\n"
  "Run it, read the two `shift` calls carefully, and work out why nobody could "
  "ever have used this table. Then repair the feature.",
  "leaky = pd.DataFrame({\n"
  "    'vol_feature': rets['AAPL'].rolling(20).std().shift(-20),\n"
  "    'vol_next': rets['AAPL'].rolling(20).std().shift(-20),\n"
  "}).dropna()\nprint('correlation:', leaky['vol_feature'].corr(leaky['vol_next']))\n\n"
  "honest = ...\nprint('correlation:', ...)",
  ["Look at the two shifts. They are identical, so the feature **is** the target: "
   "it is the volatility of the next twenty days, which nobody knows yet.",
   "A legitimate feature looks backwards. Drop the shift from the feature and "
   "leave it on the target only."],
  "leaky = pd.DataFrame({\n"
  "    'vol_feature': rets['AAPL'].rolling(20).std().shift(-20),\n"
  "    'vol_next': rets['AAPL'].rolling(20).std().shift(-20),\n"
  "}).dropna()\nprint('correlation:', leaky['vol_feature'].corr(leaky['vol_next']))\n\n"
  "honest = pd.DataFrame({\n"
  "    'vol_feature': rets['AAPL'].rolling(20).std(),\n"
  "    'vol_next': rets['AAPL'].rolling(20).std().shift(-20),\n"
  "}).dropna()\nprint('correlation:', honest['vol_feature'].corr(honest['vol_next']))",
  "A correlation of **1.000** against **0.50** once repaired.\n\n"
  "A perfect correlation between a feature and a target is never good news. It "
  "means the feature contains the answer, and the only question left is how it "
  "got there. Here it is obvious because both shifts are on the same line; in a "
  "real project the leak arrives through a join with a table someone else built, "
  "and the only symptom is a result that is too good.\n\n"
  "**Treat a suspiciously strong feature as a bug report, not a discovery.**")

# ==================================================================== Q4
q("Q4", "One table, and its shape",
  "Put the features and the target into one table called `table`, drop the rows "
  "that are not complete, and report `n` and `p`.",
  "table = ...\nn = ...\np = ...\nprint('n =', n)\nprint('p =', p)",
  ["`features['vol_next'] = target` adds the target as a fourth column, then "
   "`.dropna()`.",
   "`table.shape` gives `(rows, columns)`, and one of those columns is the target, "
   "so `p` is one less than the column count."],
  "features['vol_next'] = target\ntable = features.dropna()\n\nn = table.shape[0]\np = table.shape[1] - 1\nprint('n =', n)\nprint('p =', p)",
  f"**n = {N_TBL:,} and p = 3.** This is the object the whole of Session 4 was "
  "about: rows are observations, three columns are features, one is the target.\n\n"
  "Everything you do from here is either about improving those columns or about "
  "finding an honest way to score a rule that maps them to the last one.")

# ==================================================================== Q5
q("Q5", "What the cleaning cost",
  "Count how many rows the `dropna()` threw away, and work out where they were.",
  "n_before = ...\nn_after = ...\nprint('rows before:', n_before)\nprint('rows after :', n_after)\nprint('lost       :', ...)",
  ["`features` before dropping still has every date, so `len(features)` is the "
   "before count.",
   "The lost rows are the difference. Think about which end of the sample each kind "
   "of missing row sits at."],
  "n_before = len(features)\nn_after = len(table)\nprint('rows before:', n_before)\nprint('rows after :', n_after)\nprint('lost       :', n_before - n_after)",
  f"**{N_RAW - N_TBL} rows lost from {N_RAW:,}.** Nineteen at the start, where the "
  "twenty-day windows had not filled up yet, and twenty at the end, where the "
  "target reaches past the last date in the file.\n\n"
  "Neither loss is a data quality problem. Both are the price of asking a question "
  "about a window of time, and both are worth stating out loud rather than "
  "discovering later.")

# ==================================================================== Q6
q("Q6", "The extreme months",
  "Find the three dates with the **largest** target values, and say what happened.",
  "worst = ...\nworst",
  ["`table['vol_next'].sort_values(ascending=False)` puts the biggest first.",
   "Then take the top three with `.head(3)`."],
  "worst = table['vol_next'].sort_values(ascending=False).head(3)\nworst",
  f"All three are late February 2020, with the target reaching **{TARGET_MAX:.4f}** "
  f"on {WORST_DATE:%d %B %Y}. Standing on those dates, the next twenty trading days "
  "were the fastest crash in modern market history.\n\n"
  "So: do you drop them? **No.** A risk model that has never seen a crash is not a "
  "risk model. This is the Session 4 point about outliers in finance, arriving in "
  "your own data: the extremes are not contamination, they are the subject.")

# ==================================================================== Q7
q("Q7", "The shape of the target",
  "Compare the **mean** and the **median** of the target, and say what the gap "
  "tells you about its distribution.",
  "target_mean = ...\ntarget_median = ...\nprint('mean  :', target_mean)\nprint('median:', target_median)\nprint('ratio :', ...)",
  "Both are methods on `table['vol_next']`.",
  "target_mean = table['vol_next'].mean()\ntarget_median = table['vol_next'].median()\nprint('mean  :', target_mean)\nprint('median:', target_median)\nprint('ratio :', target_mean / target_median)",
  f"Mean **{TARGET_MEAN:.5f}**, median **{TARGET_MED:.5f}**. The mean sits above the "
  "median, so the distribution has a long right tail: most months are calm and a "
  "few are terrible.\n\n"
  "That has a direct consequence for Part 4's scoring. A squared metric will be "
  "dominated by those few months, and an absolute one will not. Neither is wrong. "
  "You just have to decide, in advance, which question you are asking.")

# ==================================================================== Q8
q("Q8", "Split, and say why it has to be a date",
  "Split `table` into `train` (everything up to the end of 2022) and `test` "
  "(2023 onwards), and report the size of each.",
  "train = ...\ntest = ...\nprint('train rows:', ...)\nprint('test rows :', ...)",
  "The index is dates, so `table.loc[:'2022-12-31']` and `table.loc['2023-01-01':]`.",
  "train = table.loc[:'2022-12-31']\ntest = table.loc['2023-01-01':]\nprint('train rows:', len(train))\nprint('test rows :', len(test))",
  f"**{N_TRAIN:,} training rows and {N_TEST} test rows.**\n\n"
  "It has to be a date rather than a random draw of rows. A randomly chosen test "
  "day would sit between two training days, so anything you built would already "
  "know roughly what the market was doing that week. The test set has to be the "
  "*future*, because that is the only situation you will ever actually be in.")

# ==================================================================== Q9
q("Q9", "The baseline nobody is allowed to skip",
  "Before any model, score the laziest rule there is: predict the **average of the "
  "training target** on every test day. Report its RMSE and MAE.\n\n"
  "$$\\text{RMSE}=\\sqrt{\\tfrac{1}{n}\\sum_i (y_i-\\hat{y}_i)^2}, \\qquad "
  "\\text{MAE}=\\tfrac{1}{n}\\sum_i |y_i-\\hat{y}_i|$$",
  "base_pred = ...\nbase_resid = ...\n\nbase_rmse = ...\nbase_mae = ...\nprint('baseline RMSE:', base_rmse)\nprint('baseline MAE :', base_mae)",
  ["The prediction is one number repeated: `train['vol_next'].mean()`.",
   "`base_resid = test['vol_next'] - base_pred`, then RMSE is "
   "`np.sqrt((base_resid ** 2).mean())` and MAE is `base_resid.abs().mean()`."],
  "base_pred = train['vol_next'].mean()\nbase_resid = test['vol_next'] - base_pred\n\n"
  "base_rmse = np.sqrt((base_resid ** 2).mean())\nbase_mae = base_resid.abs().mean()\n"
  "print('baseline RMSE:', base_rmse)\nprint('baseline MAE :', base_mae)",
  f"RMSE **{BASE_RMSE:.5f}**, MAE **{BASE_MAE:.5f}**. Remember these two numbers. "
  "Anything you build later has to beat them, and a surprising amount of published "
  "work never checks.")

# ==================================================================== Q10
q("Q10", "The rule a trader would actually use",
  "Now a real rule, and still not a model: **next month will look like this "
  "month**. Predict each test day's target with that day's `vol_20d`, and score it "
  "the same two ways.",
  "pers_pred = ...\npers_resid = ...\n\npers_rmse = ...\npers_mae = ...\nprint('persistence RMSE:', pers_rmse)\nprint('persistence MAE :', pers_mae)",
  ["The prediction is simply the `vol_20d` column of `test`.",
   "Everything else is exactly as in Q9."],
  "pers_pred = test['vol_20d']\npers_resid = test['vol_next'] - pers_pred\n\n"
  "pers_rmse = np.sqrt((pers_resid ** 2).mean())\npers_mae = pers_resid.abs().mean()\n"
  "print('persistence RMSE:', pers_rmse)\nprint('persistence MAE :', pers_mae)",
  f"RMSE **{PERS_RMSE:.5f}** against the baseline's {BASE_RMSE:.5f}, and MAE "
  f"**{PERS_MAE:.5f}** against {BASE_MAE:.5f}. A rule with no parameters, no "
  "fitting and no data science beats the average on both.\n\n"
  "That is volatility clustering, and it is why forecasting volatility is a real "
  "business while forecasting returns is mostly not.")

# ==================================================================== Q11
q("Q11", "How much better, in one number",
  "Express that improvement as an $R^2$: how much of the test variation the "
  "persistence rule accounts for, measured against the training-mean baseline.\n\n"
  "$$R^2 = 1 - \\frac{\\sum_i (y_i-\\hat{y}_i)^2}{\\sum_i (y_i-\\bar{y}_{\\text{train}})^2}$$",
  "rss = ...\ntss = ...\npers_r2 = ...\nprint('persistence R2 on the test years:', pers_r2)",
  ["`rss` is the sum of squared persistence residuals: `(pers_resid ** 2).sum()`.",
   "`tss` uses the **training** mean as the baseline: "
   "`((test['vol_next'] - train['vol_next'].mean()) ** 2).sum()`."],
  "rss = (pers_resid ** 2).sum()\ntss = ((test['vol_next'] - train['vol_next'].mean()) ** 2).sum()\n"
  "pers_r2 = 1 - rss / tss\nprint('persistence R2 on the test years:', pers_r2)",
  f"**{PERS_R2:.3f}.** On data it had never seen, a one-line rule accounts for about "
  "a third of the variation in next month's volatility.\n\n"
  "Hold on to that number too. It is now the bar. A model that takes three features "
  "and a fitting procedure and lands below it has earned nothing.")

# ==================================================================== Q12
q("Q12", "Draw the forecast against what happened",
  "Two numbers told you the persistence rule beats the baseline. A picture tells "
  "you *how*.\n\n"
  "On the test years only, plot the realised target and the persistence "
  "prediction on the same axes, and label both.",
  "fig, ax = plt.subplots(figsize=(11, 3.2))\n...\nax.legend()\nplt.show()",
  ["The two series are `test['vol_next']` and `test['vol_20d']`, both against "
   "`test.index`.",
   "`ax.plot(test.index, test['vol_next'], label='what happened')` and the same "
   "for the prediction, then a `set_ylabel` and a `set_title`."],
  "fig, ax = plt.subplots(figsize=(11, 3.2))\n"
  "ax.plot(test.index, test['vol_next'], linewidth=1.6, label='what happened')\n"
  "ax.plot(test.index, test['vol_20d'], linewidth=1.4, label='persistence forecast')\n"
  "ax.set_ylabel('20-day volatility')\n"
  "ax.set_title('Apple, 2023 to 2024: forecast against outcome', loc='left')\n"
  "ax.legend()\nplt.show()",
  "The two lines have the same shape, and the forecast is the outcome **shifted "
  "to the right**. That is what a persistence rule is: it repeats what just "
  "happened, so it is always a month late to every turn.\n\n"
  "Look at where the gaps are widest. They are at the turning points, exactly "
  "where a forecast would have been worth having. A rule that is right in the "
  "calm stretches and wrong at every turn can still post a respectable R2, which "
  "is why you look at the picture as well as the number.")

# ==================================================================== Q13
q("Q13", "Does it hold for the whole desk?",
  "Apple is one name. Repeat the whole thing for all eleven: build the table, "
  "split it at the same date, and compute the persistence $R^2$. Collect the "
  "answers in a Series, worst last.",
  "r2_by_ticker = {}\n\nfor ticker in TICKERS:\n    t = pd.DataFrame({'vol_20d': rets[ticker].rolling(20).std()})\n    t['vol_next'] = ...\n    t = t.dropna()\n\n    tr = ...\n    te = ...\n\n    r2_by_ticker[ticker] = ...\n\nresult = ...\nresult",
  ["Inside the loop, the target is the same construction as Q1 with `ticker` in "
   "place of `'AAPL'`, and the split is the same two `.loc` slices as Q8.",
   "The R2 is the Q11 expression: "
   "`1 - ((te['vol_next'] - te['vol_20d']) ** 2).sum() / ((te['vol_next'] - tr['vol_next'].mean()) ** 2).sum()`. "
   "Afterwards, `pd.Series(r2_by_ticker).sort_values(ascending=False)`."],
  "r2_by_ticker = {}\n\nfor ticker in TICKERS:\n"
  "    t = pd.DataFrame({'vol_20d': rets[ticker].rolling(20).std()})\n"
  "    t['vol_next'] = rets[ticker].rolling(20).std().shift(-20)\n"
  "    t = t.dropna()\n\n"
  "    tr = t.loc[:'2022-12-31']\n    te = t.loc['2023-01-01':]\n\n"
  "    rss = ((te['vol_next'] - te['vol_20d']) ** 2).sum()\n"
  "    tss = ((te['vol_next'] - tr['vol_next'].mean()) ** 2).sum()\n"
  "    r2_by_ticker[ticker] = 1 - rss / tss\n\n"
  "result = pd.Series(r2_by_ticker).sort_values(ascending=False)\nresult",
  f"**Positive for only {N_POSITIVE} of the eleven.** {R2_S.index[0]} leads at "
  f"{R2_S.iloc[0]:.2f}; {R2_S.index[-1]} is at {R2_S.iloc[-1]:.2f}, which means the "
  "rule is far worse there than simply predicting the training average.\n\n"
  "This is the most useful result in the notebook. A rule that looked convincing on "
  "one name falls apart across the desk, which is precisely the gap a real model is "
  "supposed to close. It also shows why you evaluate per instrument and not on a "
  "single flattering example.")

# ==================================================================== Q14
q("Q14", "Does Part 3's ranking tell you anything?",
  "The quick load restated Part 3's finding: Nvidia was the most volatile name of "
  "2024, the index fund the calmest. A natural guess is that the calm names are "
  "the predictable ones.\n\n"
  "You now have the evidence to check it. Put the 2024 volatilities beside the "
  "R2 values from Q13 and measure how strongly they move together.",
  "vol_2024 = pd.Series(part3_annual_vol)\n\n"
  "comparison = pd.DataFrame({'vol_2024': vol_2024, 'r2': ...})\n"
  "correlation = ...\n\n"
  "print(comparison.sort_values('vol_2024', ascending=False).round(3))\nprint('correlation:', correlation)",
  ["`r2_by_ticker` from Q13 is a dictionary, so `pd.Series(r2_by_ticker)` lines it "
   "up against the volatilities by ticker.",
   "`comparison['vol_2024'].corr(comparison['r2'])` measures how they move "
   "together, between -1 and +1."],
  "vol_2024 = pd.Series(part3_annual_vol)\n\n"
  "comparison = pd.DataFrame({'vol_2024': vol_2024, 'r2': pd.Series(r2_by_ticker)})\n"
  "correlation = comparison['vol_2024'].corr(comparison['r2'])\n\n"
  "print(comparison.sort_values('vol_2024', ascending=False).round(3))\nprint('correlation:', correlation)",
  "**-0.35.** Weakly negative, which says the more volatile names were, if "
  "anything, slightly harder to forecast.\n\n"
  "Now be careful, because this is where a report goes wrong. You have **eleven "
  "points**. A correlation of -0.35 on eleven points is roughly what you get from "
  "pure noise, and the two most extreme names, Nvidia and Disney, are doing most "
  "of the work. The honest sentence is *this sample cannot tell*, not *volatile "
  "names are harder to predict*.\n\n"
  "Part 3's ranking was a good description of 2024. It is not a guide to which "
  "names a model will do well on, and you now know that because you checked "
  "rather than assumed.")

# ==================================================================== Q15
q("Q15", "The same question, asked as a label",
  "The desk does not really need a number. It needs to know **which names to watch**. "
  "Turn the problem into a classification: a month counts as *high volatility* if "
  "the target is above the **median of the training target**.\n\n"
  "Compute that threshold, then build `high_true` (it really was high) and "
  "`high_pred` (this month was high, so the rule says next month will be).",
  "threshold = ...\n\nhigh_true = ...\nhigh_pred = ...\n\nprint('threshold      :', threshold)\nprint('share high, test:', ...)",
  ["The threshold is `train['vol_next'].median()`, and it must come from the "
   "training rows only.",
   "`high_true = test['vol_next'] > threshold` and `high_pred = test['vol_20d'] > threshold`."],
  "threshold = train['vol_next'].median()\n\nhigh_true = test['vol_next'] > threshold\nhigh_pred = test['vol_20d'] > threshold\n\n"
  "print('threshold      :', threshold)\nprint('share high, test:', high_true.mean())",
  f"The threshold is **{THRESHOLD:.5f}**, and only **{BASE_RATE:.1%}** of the test "
  "months clear it.\n\n"
  "Read that carefully. By construction half of the *training* months were above "
  "the training median, so the quiet market of 2023 and 2024 has left the two "
  "periods badly out of balance. A threshold learned on one regime does not "
  "transfer to another, and no amount of modelling fixes that on its own.")

# ==================================================================== Q16
q("Q16", "Score the warning system",
  "Count the four outcomes for that rule, then compute precision and recall.\n\n"
  "$$\\text{precision}=\\frac{TP}{TP+FP}, \\qquad \\text{recall}=\\frac{TP}{TP+FN}$$",
  "tp = ...\nfp = ...\nfn = ...\ntn = ...\n\nprecision = ...\nrecall = ...\nprint('TP', tp, ' FP', fp, ' FN', fn, ' TN', tn)\nprint('precision:', precision)\nprint('recall   :', recall)",
  ["Combine the two boolean Series with `&`, and flip one with `~`: "
   "`(high_pred & high_true).sum()` is TP.",
   "Precision is `tp / (tp + fp)` and recall is `tp / (tp + fn)`."],
  "tp = (high_pred & high_true).sum()\nfp = (high_pred & ~high_true).sum()\n"
  "fn = (~high_pred & high_true).sum()\ntn = (~high_pred & ~high_true).sum()\n\n"
  "precision = tp / (tp + fp)\nrecall = tp / (tp + fn)\n"
  "print('TP', tp, ' FP', fp, ' FN', fn, ' TN', tn)\nprint('precision:', precision)\nprint('recall   :', recall)",
  f"TP **{TP}**, FP **{FP}**, FN **{FN}**, TN **{TN}**. Precision **{PREC:.3f}** and "
  f"recall **{REC:.3f}**, on an accuracy of {ACC:.3f}.\n\n"
  "So the warning system fires often and is right about four times in ten, and it "
  "still misses more than half of the months that really were turbulent. Which of "
  "those two failures is worse is not a statistical question. On a risk desk a "
  "missed turbulent month costs far more than a false alarm, so you would move the "
  "threshold down and accept the extra noise.")

# ==================================================================== Q17
q("Q17", "Write the specification",
  "You now have everything a modeller would need. Write it down in one place, so "
  "somebody else could pick it up and be judged fairly.\n\n"
  "Fill in the dictionary, then write `describe(spec)`: a function that prints it "
  "line by line and ends with a one-sentence verdict on whether the bar is worth "
  "clearing.\n\n"
  "A dictionary, a loop, an f-string and an `if`. Everything in this last cell of "
  "the case comes from Sessions 1 and 2.",
  "spec = {\n    'target': ...,\n    'features': ...,\n    'n_train': ...,\n    'n_test': ...,\n    'split': ...,\n    'metric': ...,\n    'baseline_rmse': ...,\n    'bar_to_beat_r2': ...,\n}\n\n"
  "def describe(spec):\n    ...\n\ndescribe(spec)",
  ["Most of the values are already sitting in variables you created: `base_rmse` "
   "from Q9 and `pers_r2` from Q11.",
   "Inside the function, `for key in spec:` then "
   "`print(f\"{key:15} {spec[key]}\")`, exactly as in Session 2's dictionary loop.",
   "For the verdict, an `if` on `spec['bar_to_beat_r2']`: above about 0.2 there is "
   "real signal to beat, below it the naive rule is barely better than nothing."],
  "spec = {\n    'target': 'sd of daily returns over the next 20 trading days',\n"
  "    'features': list(train.columns.drop('vol_next')),\n"
  "    'n_train': len(train),\n    'n_test': len(test),\n"
  "    'split': 'by date: train to 2022-12-31, test from 2023-01-01',\n"
  "    'metric': 'RMSE, with MAE reported alongside because the target is skewed',\n"
  "    'baseline_rmse': round(base_rmse, 5),\n"
  "    'bar_to_beat_r2': round(pers_r2, 3),\n}\n\n"
  "def describe(spec):\n    \"\"\"Print a learning problem, and say whether the bar is a serious one.\"\"\"\n"
  "    for key in spec:\n        print(f\"{key:15} {spec[key]}\")\n\n"
  "    bar = spec['bar_to_beat_r2']\n    if bar > 0.2:\n        print(f\"\\nA model must beat R2 = {bar:.2f}. That is a real bar, not a formality.\")\n"
  "    else:\n        print(f\"\\nThe naive rule only reaches R2 = {bar:.2f}, so almost anything should beat it.\")\n\n"
  "describe(spec)",
  "That is a machine learning problem, fully specified, and not one model has been "
  "fitted.\n\n"
  "Look at what the last cell of this case is made of: a dictionary and a loop from "
  "Session 2, an f-string from Session 1, a function with a docstring from Session "
  "2, and numbers from Session 4. Four weeks of tools in fifteen lines, which is "
  "what the exam will ask of you.\n\n"
  "Everything in the rest of the course slots into the one gap left in it: the rule "
  "that turns those three feature columns into a prediction. Choosing that rule "
  "well is what the next block is about, and you now know exactly what it has to "
  "beat.")

# ------------------------------------------------------------------ closing
md(
"## \U0001f9ed What you have now\n\n"
"| what | where it lives |\n"
"|:--|:--|\n"
"| the question, written as a target | `target` |\n"
"| features that cannot see the future | `features` |\n"
"| the learning table, complete rows only | `table`, `n`, `p` |\n"
"| an honest split by date | `train`, `test` |\n"
"| the baseline nobody may skip | `base_rmse`, `base_mae` |\n"
"| the bar a model has to clear | `pers_rmse`, `pers_r2` |\n"
"| the same problem as a classification | `threshold`, `precision`, `recall` |\n"
"| the whole thing, written down | `spec` |\n"
)

md(
"## What is still wrong with it\n\n"
"Three things, and noticing them yourself is the point of having built it:\n\n"
"- **The rows are not independent.** Consecutive days share nineteen of their "
"twenty return observations, so 2,457 rows is nowhere near 2,457 independent "
"pieces of evidence. Any confidence interval you computed here would be far too "
"narrow.\n"
"- **One split is one experiment.** You chose the end of 2022 because it was "
"convenient. A different cut could easily give a different verdict.\n"
"- **The regimes differ.** The training years contain a crash and the test years "
"do not, which is why the classification threshold transferred so badly.\n\n"
"All three have names and all three have answers, and they are the subject of the "
"next block: model selection, cross-validation, and the discipline of not "
"believing your own first result."
)

md(
"## The end of the beginning\n\n"
"Four sessions ago you compared two stocks with a subtraction. You now have a "
"fully specified prediction problem on eleven instruments, with a target, honest "
"features, a dated split, two baselines and a metric you can defend.\n\n"
"The one thing you have never done is fit a model. That was deliberate. Fitting "
"is the easy part, it is three lines of scikit-learn, and doing it before the "
"work above is how people produce results that cannot survive contact with next "
"month.\n\n"
"**Next block:** the three lines, and everything that has to be true before you "
"are allowed to believe them."
)

# ---------------------------------------------------------------- write
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
# Deterministic cell ids: nbformat assigns a fresh random uuid to every cell,
# which would make every regeneration look like a rewrite in git.
for _i, _c in enumerate(nb.cells):
    _c["id"] = f"c{_i:04d}"

OUT.write_text(nbf.writes(nb), encoding="utf-8")

n_q = sum(1 for c in cells if c.cell_type == "markdown" and c.source.startswith("### Q"))
print("wrote", OUT, " (", len(cells), "cells,", n_q, "questions )")
