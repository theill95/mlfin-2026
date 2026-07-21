# -*- coding: utf-8 -*-
"""Build session_03_case.ipynb  (The Analyst's Notebook, Part 3).

Conventions (approved for Sessions 1 and 2): no star badges, one cumulative
investigation where later questions reuse what earlier ones stored, folded
hints and solutions, stated formulas, plain tone, no em-dashes. Opens with a
QUICK LOAD restoring Part 2's findings.

NOT COPY-PASTEABLE FROM THE LECTURE: the deck ranked the whole ten-year period
with `wide.pct_change().std()`. Here the student works on 2024 only (so the
answer can be checked against Part 2), annualises it, cross-checks the pivot
route against the groupby route, and finishes with a co-movement count the deck
never showed.

BLANK-SAFE, and this one needs care because the case is cumulative: no cell may
CALL anything on a variable an earlier question produced. Every such dependency
sits inside the student's own blank, so a placeholder just propagates. The only
pre-written code that touches data is the quick load (which really does load
`prices`) and lines that use `prices` itself.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "session_03" / "session_03_case.ipynb"

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
P24 = W.loc["2024"]
R24 = P24.pct_change()
VOL = R24.std().sort_values(ascending=False)
AVOL = VOL * np.sqrt(252)
RD = R24.dropna()
SAME = {t: int(((RD[t] * RD["SPY"]) > 0).sum()) for t in RD.columns}
N = len(RD)
ROWS_EACH = int(PX.groupby("ticker").size().iloc[0])
NVDA_TR = P24["NVDA"].iloc[-1] / P24["NVDA"].iloc[0] - 1

# ---------------------------------------------------------------- top matter
md(
"# \U0001f4bc The Analyst's Notebook · Part 3\n"
"### Eleven instruments, one afternoon\n\n"
"In Part 1 you compared two stocks with a crude price range. In Part 2 you "
"replaced it with a proper volatility, built out of every daily return of a "
"year, using loops and a function you wrote yourself.\n\n"
"The desk now wants that report for the **whole universe**: eleven instruments, "
"ten years of daily prices. You are not going to write eleven loops."
)

md(
"## How to work through this\n\n"
"- Run the **quick load** cell first. It brings back what you found in Part 2 and "
"loads the price table.\n"
"- Each question builds on the last, so keep them in order and keep your "
"variables. Later questions use the names earlier ones created.\n"
"- Cells with `...` are blanks. The notebook runs cleanly even before you fill "
"them in, so **Run all** is always safe.\n"
"- Hints and solutions are folded under each question. Work first, then check.\n\n"
"*Stuck for more than 15 minutes? Ask a friend, ask an AI for a hint (not the "
"answer), or email me at `jobo@econ.au.dk`.*"
)

md("---")

# ------------------------------------------------------------- quick load
md(
"## ⚙️ Quick load\n\n"
"The three packages, the price table, and what Part 2 left you. Run it and read "
"what it prints."
)

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


# The whole universe: eleven instruments, 2015 to 2024, one row per stock per day
prices = pd.read_csv(data_path("prices.csv"), parse_dates=["date"])
TICKERS = sorted(prices["ticker"].unique())

# --- What Part 2 found, restated so you can build on it ---
part2_vol = {          # daily volatility in 2024, computed with your own loops
    "AAPL": 0.0141,
    "KO":   0.0080,
    "NVDA": 0.0330,
    "JNJ":  0.0094,
    "JPM":  0.0148,
}
part2_riskiest = "NVDA"
part2_calmest = "KO"

print("Loaded prices:", prices.shape[0], "rows x", prices.shape[1], "columns")
print("Instruments:", ", ".join(TICKERS))
print()
print("Part 2 recap: daily volatility in 2024, five stocks")
for ticker in part2_vol:
    print(f"  {ticker:5} {part2_vol[ticker]:.4f}")
print(f"  riskiest: {part2_riskiest}   calmest: {part2_calmest}")'''
)

md("---")

# ==================================================================== Q1
q("Q1", "Look before you trust it",
  "The quick load read the file with `pd.read_csv(path, parse_dates=['date'])`. "
  "Never compute on a table you have not looked at, so find out:\n\n"
  "- what type of data each column holds\n"
  "- which tickers are in it, and how many rows each one has",
  "print(prices.dtypes)\nprint()\n\nrows_per_ticker = ...\nrows_per_ticker",
  ["`prices.dtypes` is already written for you. Read what it says about `date`.",
   "For the row counts, group by ticker and ask how big each group is: "
   "`prices.groupby('ticker').size()`."],
  "print(prices.dtypes)\nprint()\n\nrows_per_ticker = prices.groupby('ticker').size()\nrows_per_ticker",
  "Eleven tickers, {:,} rows each, and no surprises in the types: real dates, text "
  "tickers, float prices. Equal row counts matter. If one stock had fewer days you "
  "would be comparing different periods without noticing.\n\n"
  "*`parse_dates` is what made `date` a real date rather than text. Without it, "
  "none of the slicing by year in Q2 would work.*".format(ROWS_EACH))

# ==================================================================== Q2
q("Q2", "One column per stock",
  "The table is **tall**: one row per stock per day. For comparing stocks it is "
  "easier **wide**: one row per date, one column per ticker.\n\n"
  "Reshape `prices` into `wide`, holding the closing prices. Then take 2024 out of "
  "it into `p24`.",
  "wide = ...\np24 = ...\np24",
  ["`prices.pivot(index='date', columns='ticker', values='close')` does the "
   "reshaping.",
   "Because the index is dates, `wide.loc['2024']` gives you just that year."],
  "wide = prices.pivot(index='date', columns='ticker', values='close')\np24 = wide.loc['2024']\np24.head()",
  "`p24` has {} rows and {} columns: every trading day of 2024, with every "
  "instrument beside it. That is the shape you want whenever you are comparing "
  "things rather than storing them.".format(P24.shape[0], P24.shape[1]))

# ==================================================================== Q3
q("Q3", "Apple again, the short way",
  "In Part 2 you computed Apple's 2024 volatility with two loops and a function of "
  "your own. Do it again from `p24`, in two lines, and compare the answers.\n\n"
  "$$\\sigma = \\text{the standard deviation of the daily returns}$$",
  "aapl_returns = ...\naapl_vol = ...\n\nprint('Part 3:', aapl_vol)\nprint('Part 2:', part2_vol['AAPL'])",
  ["`p24['AAPL'].pct_change()` gives every daily return, with a `NaN` on the first "
   "day because there is no day before it.",
   "Then `.std()` on those returns. It skips the `NaN` for you."],
  "aapl_returns = p24['AAPL'].pct_change()\naapl_vol = aapl_returns.std()\n\nprint('Part 3:', round(aapl_vol, 4))\nprint('Part 2:', part2_vol['AAPL'])",
  "Both give **{:.4f}**. Two loops, an empty list and a function of your own, "
  "replaced by `.pct_change().std()`.\n\n"
  "*Compare more decimals and they differ very slightly. Your Part 2 function "
  "divided by $n$; pandas divides by $n-1$ by default. On 251 returns that moves "
  "the fifth decimal, and neither is wrong.*".format(VOL["AAPL"]))

# ==================================================================== Q4
q("Q4", "All eleven at once",
  "Now the point of the whole session. Compute the 2024 daily volatility of "
  "**every** instrument into `vol`, sorted riskiest first.",
  "vol = ...\nvol",
  ["`p24.pct_change()` gives a whole table of returns, one column per instrument.",
   "`.std()` on that table gives one number per column, and `.sort_values("
   "ascending=False)` puts the riskiest at the top."],
  "vol = p24.pct_change().std().sort_values(ascending=False)\nvol",
  "Eleven answers, one line. Nvidia at **{:.4f}** is more than four times as "
  "volatile as the calmest thing in the table. Your five Part 2 numbers are all in "
  "here, unchanged.".format(VOL.iloc[0]))

# ==================================================================== Q5
q("Q5", "Check it a second way",
  "That was quick enough to be worth distrusting. Compute the same eleven numbers "
  "**without** the wide table: take 2024 out of `prices`, add a return column "
  "within each ticker, and take the standard deviation of each group.\n\n"
  "Then look at whether the two routes agree.",
  "p2024 = prices[prices['date'].dt.year == 2024].copy()\n\np2024['ret'] = ...\nvol_check = ...\nvol_check",
  ["`p2024.groupby('ticker')['close'].pct_change()` computes the return **inside** "
   "each stock. Grouping first is what stops a return being taken across two "
   "different companies.",
   "Then `p2024.groupby('ticker')['ret'].std()`, sorted the same way as `vol`."],
  "p2024 = prices[prices['date'].dt.year == 2024].copy()\n\np2024['ret'] = p2024.groupby('ticker')['close'].pct_change()\nvol_check = p2024.groupby('ticker')['ret'].std().sort_values(ascending=False)\nvol_check",
  "The same eleven numbers in the same order. Two different routes agreeing is the "
  "cheapest check you will ever run, and worth doing whenever one line has replaced "
  "forty.\n\n"
  "*The `groupby` before `pct_change` is the part that matters. Without it, the "
  "first return of each stock would be computed from the last price of whichever "
  "stock sits above it in the table.*")

# ==================================================================== Q6
q("Q6", "Put it in the units the desk uses",
  "Nobody quotes a daily volatility. Annualise `vol`, then store the riskiest and "
  "the calmest name.\n\n"
  "$$\\sigma_{\\text{year}} = \\sigma_{\\text{day}} \\times \\sqrt{252}$$",
  "annual_vol = ...\nriskiest = ...\ncalmest = ...\n\nprint(annual_vol)\nprint('riskiest:', riskiest, ' calmest:', calmest)",
  ["`np.sqrt(252)` is the multiplier, and you can apply it to the whole Series at "
   "once.",
   "`vol` is already sorted riskiest first, so `annual_vol.index[0]` is the riskiest "
   "name and `annual_vol.index[-1]` is the calmest."],
  "annual_vol = (vol * np.sqrt(252)).round(3)\nriskiest = annual_vol.index[0]\ncalmest = annual_vol.index[-1]\n\nprint(annual_vol)\nprint('riskiest:', riskiest, ' calmest:', calmest)",
  "Nvidia at about **{:.0%} a year**, against **{:.0%}** for {}.\n\n"
  "Look at what came bottom: **SPY**, the fund that holds the whole S&P 500. It is "
  "calmer than every individual stock in the table, including Coca-Cola, which was "
  "the calmest thing you had found in Part 2. That is diversification, and you have "
  "just measured it.".format(AVOL.iloc[0], AVOL.iloc[-1], AVOL.index[-1]))

# ==================================================================== Q7
q("Q7", "Package it, so you can reuse it",
  "You have now written the same calculation three times. Turn it into a "
  "**function**: `volatility(prices_series, periods=252)`, which takes a column of "
  "prices and returns an annualised volatility.\n\n"
  "Give it a docstring, then check it against Apple's number from Q3.",
  "def volatility(prices_series, periods=252):\n    ...\n\n\n# Once it works, uncomment this to check it against Q6:\n# print('function:', volatility(p24['AAPL']))\n# print('Q6      :', annual_vol['AAPL'])",
  ["The body is one line: `.pct_change()`, then `.std()`, then times "
   "`np.sqrt(periods)`.",
   "`return prices_series.pct_change().std() * np.sqrt(periods)`. The default "
   "`periods=252` means you can call it with one argument most of the time."],
  "def volatility(prices_series, periods=252):\n    \"\"\"Annualised volatility of a price series.\"\"\"\n    return prices_series.pct_change().std() * np.sqrt(periods)\n\nprint('AAPL, function:', round(volatility(p24['AAPL']), 3))\nprint('AAPL, Q6      :', annual_vol['AAPL'])",
  "The same **{:.3f}**. The function is Session 2 work: `def`, a parameter, a "
  "default, a docstring and a `return`. Only the line inside it is new.\n\n"
  "*Try `volatility(p24['AAPL'], periods=21)` for a monthly figure. That is what "
  "the default argument buys you: one function, several questions.*".format(AVOL["AAPL"]))

# ==================================================================== Q8
q("Q8", "A loop, a dictionary, and a Series",
  "Now rebuild the whole ranking the Session 2 way: **loop** over `TICKERS` (the "
  "quick load made that list for you), call your function on each one, and store the "
  "answers in a **dictionary** keyed by ticker. Then turn the dictionary into a "
  "Series and check it against `annual_vol`.",
  "by_hand = {}\nfor ticker in TICKERS:\n    ...\n\nby_hand = ...\nby_hand",
  ["Inside the loop: `by_hand[ticker] = volatility(p24[ticker])`.",
   "Afterwards, `pd.Series(by_hand).sort_values(ascending=False)` turns the "
   "dictionary into the same object `annual_vol` is."],
  "by_hand = {}\nfor ticker in TICKERS:\n    by_hand[ticker] = volatility(p24[ticker])\n\nby_hand = pd.Series(by_hand).sort_values(ascending=False)\nby_hand.round(3)",
  "The same eleven numbers in the same order as Q6.\n\n"
  "Nothing in that cell is new: a loop, a function, a dictionary, all from Session "
  "2. It is worth seeing that the one-line pandas version and the eleven-line loop "
  "give the same answer, because the loop is the one you can still write when the "
  "calculation gets too odd for a single method call.")

# ==================================================================== Q9
q("Q9", "Show the desk, do not tell it",
  "Draw the ranking as horizontal bars: tickers down the side, annualised "
  "volatility across. Give it a title and an x-axis label.\n\n"
  "Sort it the other way round first, so the longest bar ends up at the top of the "
  "picture.",
  "ranked = ...\n\nfig, ax = plt.subplots(figsize=(9, 4))\n...\nplt.show()",
  ["`ranked = annual_vol.sort_values()` puts the smallest first, which draws from "
   "the bottom up.",
   "Then `ax.barh(ranked.index, ranked.values)`, `ax.set_title('...', loc='left')` "
   "and `ax.set_xlabel('...')`."],
  "ranked = annual_vol.sort_values()\n\nfig, ax = plt.subplots(figsize=(9, 4))\nax.barh(ranked.index, ranked.values)\nax.set_title('Annualised volatility, 2024', loc='left')\nax.set_xlabel('standard deviation of daily returns, annualised')\nplt.show()",
  "One picture, and the ranking is obvious without reading a number. Sorting before "
  "you plot is what makes a bar chart worth looking at: an unsorted one makes the "
  "reader do the work.")

# ==================================================================== Q10
q("Q10", "What the riskiest one actually did",
  "A volatility says how much something moved, not which way. Draw Nvidia's 2024 "
  "closing price as a line, with a title and a y-axis label, and see for yourself.",
  "nvda = ...\n\nfig, ax = plt.subplots(figsize=(9, 3))\n...\nplt.show()",
  ["`nvda = p24['NVDA']` is the column you want.",
   "Then `ax.plot(nvda.index, nvda.values)`, a title and `ax.set_ylabel('price "
   "(USD)')`."],
  "nvda = p24['NVDA']\n\nfig, ax = plt.subplots(figsize=(9, 3))\nax.plot(nvda.index, nvda.values)\nax.set_title('Nvidia (NVDA) closing price, 2024', loc='left')\nax.set_ylabel('price (USD)')\nplt.show()",
  "It nearly tripled: a total return of about **{:.0%}** over the year. So the "
  "riskiest thing in the table was also the most rewarding one, in this particular "
  "year. Risk is not a synonym for bad, which is exactly why the two questions have "
  "to be kept apart.".format(NVDA_TR))

# ==================================================================== Q11
q("Q11", "Do they move together?",
  "A portfolio's risk depends not only on how much each stock moves, but on whether "
  "they all move **at the same time**.\n\n"
  "For Apple and for Coca-Cola, count the days in 2024 when they moved in the "
  "**same direction** as the market (SPY). Two returns moved the same way when "
  "their product is positive.",
  "r24 = ...\naapl_same = ...\nko_same = ...\n\nprint('AAPL with the market:', aapl_same)\nprint('KO   with the market:', ko_same)",
  ["`r24 = p24.pct_change().dropna()` gives every daily return with the first, empty "
   "row removed.",
   "`r24['AAPL'] * r24['SPY']` is positive on a day they moved together, so "
   "`(r24['AAPL'] * r24['SPY'] > 0).sum()` counts those days. It is the up-day "
   "counter you have been writing since Session 2."],
  "r24 = p24.pct_change().dropna()\naapl_same = (r24['AAPL'] * r24['SPY'] > 0).sum()\nko_same = (r24['KO'] * r24['SPY'] > 0).sum()\n\nprint('trading days       :', len(r24))\nprint('AAPL with the market:', aapl_same)\nprint('KO   with the market:', ko_same)",
  "Out of {} days, Apple moved with the market on **{}** of them ({:.0%}) and "
  "Coca-Cola on only **{}** ({:.0%}), which is barely better than a coin toss.\n\n"
  "That gap matters more than it looks. A stock that moves with everything else "
  "adds its full risk to a portfolio. One that goes its own way partly cancels out "
  "against the others, which is why the S&P 500 fund came bottom of your ranking in "
  "Q6.".format(N, SAME["AAPL"], SAME["AAPL"] / N, SAME["KO"], SAME["KO"] / N))

# ==================================================================== Q12
q("Q12", "From a number to a category",
  "The desk does not want eleven decimals, it wants three buckets. Write "
  "`risk_label(annual_vol)` that returns `'high'` above 0.30, `'medium'` above 0.18 "
  "and `'low'` otherwise, then label every stock and collect the answers.",
  "def risk_label(vol_value):\n    ...\n\nlabels = {}\nfor ticker in TICKERS:\n    ...\n\nlabels = ...\nlabels",
  ["The function is three branches: `if annual_vol > 0.30: return 'high'`, then "
   "`elif annual_vol > 0.18: return 'medium'`, then `else: return 'low'`.",
   "In the loop, `labels[ticker] = risk_label(annual_vol[ticker])`. Afterwards, "
   "`pd.Series(labels)` makes it a column you can add to a table."],
  "def risk_label(vol_value):\n    \"\"\"Bucket an annualised volatility into high, medium or low.\"\"\"\n    if vol_value > 0.30:\n        return 'high'\n    elif vol_value > 0.18:\n        return 'medium'\n    else:\n        return 'low'\n\nlabels = {}\nfor ticker in TICKERS:\n    labels[ticker] = risk_label(annual_vol[ticker])\n\nlabels = pd.Series(labels)\nlabels",
  "Nvidia alone is `high`. Disney, JPMorgan, Apple, Microsoft and Exxon are "
  "`medium`, and the other five are `low`.\n\n"
  "You have just turned a measured number into a **category**, using an `if` chain "
  "from Session 2 and nothing else. Predicting a number is called regression; "
  "predicting a category like this one is called classification. Session 4 gives "
  "both of them their names.")

# ==================================================================== Q13
q("Q13", "The table a model would want",
  "Build the report as one row per stock, with three columns describing its 2024: "
  "annualised volatility, average daily return, and the share of days it rose.\n\n"
  "Then add your risk labels as a fourth column.",
  "vol_col = ...\nmean_col = ...\nup_col = ...\n\nreport = ...\nreport",
  ["`r24` from Q11 already holds every daily return. The three columns are "
   "`r24.std() * np.sqrt(252)`, `r24.mean()` and `(r24 > 0).mean()`.",
   "`pd.DataFrame({'vol': vol_col, 'mean_ret': mean_col, 'up_share': up_col})` lines "
   "them up on the ticker labels. Add the labels with "
   "`report['risk'] = labels`."],
  "vol_col = r24.std() * np.sqrt(252)\nmean_col = r24.mean()\nup_col = (r24 > 0).mean()\n\nreport = pd.DataFrame({'vol': vol_col, 'mean_ret': mean_col, 'up_share': up_col}).round(4)\nreport['risk'] = labels\nreport.sort_values('vol', ascending=False)",
  "Eleven rows, four columns. This is the report, and it is also a shape worth "
  "recognising: **one row per thing you are studying, one column per fact about "
  "it**. Session 4 calls the rows *observations*, the numeric columns *features*, "
  "and whichever column you are trying to predict the *target*.\n\n"
  "Everything in this course from here on arrives in that shape.")

# ==================================================================== Q14
q("Q14", "Would last year have told you?",
  "Every number above describes 2024, a year that has already happened. The useful "
  "question is whether it would have helped you in advance.\n\n"
  "Compute each stock's annualised volatility over **2015 to 2022**, and again over "
  "**2023 to 2024**, put them side by side, and sort by the early column.",
  "early = ...\nlate = ...\n\nearly_vol = ...\nlate_vol = ...\n\ncomparison = ...\ncomparison",
  ["`wide.loc['2015':'2022']` and `wide.loc['2023':'2024']` cut the history in two, "
   "because the index is dates.",
   "Each volatility is `period.pct_change().std() * np.sqrt(252)`, and "
   "`pd.DataFrame({'early': early_vol, 'late': late_vol})` puts them side by side."],
  "early = wide.loc['2015':'2022']\nlate = wide.loc['2023':'2024']\n\nearly_vol = early.pct_change().std() * np.sqrt(252)\nlate_vol = late.pct_change().std() * np.sqrt(252)\n\ncomparison = pd.DataFrame({'early': early_vol, 'late': late_vol})\ncomparison.sort_values('early', ascending=False).round(3)",
  "Partly, and only partly. Nvidia is the most volatile in both periods, and "
  "Coca-Cola and SPY are among the calmest in both. But Apple falls from second "
  "riskiest to sixth, and Johnson & Johnson climbs from last.\n\n"
  "Notice what you did there: you fitted your view on one stretch of time and "
  "checked it on a **later** one you had not looked at. Splitting by time rather "
  "than at random is not a detail. A randomly chosen test day would sit between two "
  "days you had already seen, and any method would look better than it is.\n\n"
  "That gap between the two columns is the subject of the rest of this course. A "
  "quantity that carried over perfectly would need no model; one that carried over "
  "not at all could not be predicted. Volatility sits in between, which is exactly "
  "what makes it worth forecasting.")

# ------------------------------------------------------------------ closing
md(
"## \U0001f9ed What you have now\n\n"
"The risk report is finished. In three sessions it has gone from two stocks and a "
"price range, to eleven instruments measured properly, ranked, drawn, labelled, "
"and checked three different ways.\n\n"
"Notice how little of it was new. The volatility is a function you wrote, called "
"in a loop, storing into a dictionary, with an `if` chain deciding the labels. "
"pandas did not replace any of that. It replaced the tedious parts: reading the "
"file, lining up the dates, and doing the arithmetic eleven times.\n\n"
"**Carry these into Part 4:**\n\n"
"| what | where |\n"
"|:--|:--|\n"
"| the price table, tall and wide | `prices`, `wide`, `p24` |\n"
"| 2024 volatility, daily and annualised | `vol`, `annual_vol` |\n"
"| riskiest and calmest | `riskiest`, `calmest` |\n"
"| a reusable calculation | `volatility()`, `risk_label()` |\n"
"| the risk categories | `labels` |\n"
"| one row per stock, four columns | `report` |\n"
)

md(
"## The question this cannot answer\n\n"
"Everything above is about a year that has already happened. It is "
"**description**: an honest account of what these prices did in 2024.\n\n"
"The desk does not really want that. It wants to know what next month looks like. "
"The moment you ask that, the shape of the problem changes:\n\n"
"- which past facts are you allowed to use, and which would be cheating?\n"
"- what exactly are you predicting, and how would you know if you were right?\n"
"- if a rule describes 2024 perfectly, is that good news or a warning?\n\n"
"You have already built the pieces. `report` in Q13 is a table of observations and "
"features. `labels` in Q12 is a category you could try to predict. The split in "
"Q14 is how you would find out honestly whether you could.\n\n"
"Next session is about those three questions. You will not fit a model. You will "
"learn what a model is being asked to do, and why that is harder than it sounds.\n\n"
"**Part 4** turns this report into a prediction problem: observations, features, "
"and a target."
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
