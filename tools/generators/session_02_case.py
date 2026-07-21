# -*- coding: utf-8 -*-
"""Build session_02_case.ipynb (The Analyst's Notebook, Part 2).

Conventions (instructor-approved on Session 1):
- ONE cumulative investigation: later questions reuse what earlier ones stored.
- NO star badges; difficulty rises naturally.
- Opens with a QUICK LOAD that restores Part 1's data and saved findings.
- Two-cell debug (broken cell tagged raises-exception, then a fix cell).
- Ends by framing the next question. No modelling.
- No em-dashes, plain academic tone.
- Only Session 1 + Session 2 tools (no sorted, no //, no dict .values()).

NOT COPY-PASTEABLE FROM THE LECTURE (explicit instructor rule): the deck showed
one monolithic `volatility(prices)` with everything inlined. Here the student
builds it by COMPOSING two helper functions they wrote themselves
(`daily_returns` and `mean`), so the deck's function cannot simply be pasted in.

BLANK-SAFE CHAIN: a cumulative case must survive a blank Run all. The trick is
that any function whose result is consumed downstream gets a scaffold that
returns a real (empty) list, so later loops iterate zero times instead of
crashing on None. Dicts consumed downstream are seeded as {} for the same
reason.
"""
from pathlib import Path
import pandas as pd
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "session_02" / "session_02_case.ipynb"
REPO = Path(__file__).resolve().parents[2]
TICKERS = ["AAPL", "KO", "NVDA", "JNJ", "JPM"]
NAMES = {"AAPL": "Apple", "KO": "Coca-Cola", "NVDA": "Nvidia",
         "JNJ": "Johnson & Johnson", "JPM": "JPMorgan"}

# ---- exact answers, computed from the real CSVs ----
def load(tk):
    for d in ["data", "session_02/data"]:
        p = REPO / d / f"{tk.lower()}_2024_closes.csv"
        if p.exists():
            t = pd.read_csv(p)
            return t["close"].tolist()
    raise FileNotFoundError(tk)

def rets(p):
    return [(p[i] - p[i - 1]) / p[i - 1] for i in range(1, len(p))]
def mean(x):
    return sum(x) / len(x)
def vol(p):
    r = rets(p)
    m = mean(r)
    return (sum((x - m) ** 2 for x in r) / len(r)) ** 0.5

CL = {tk: load(tk) for tk in TICKERS}
R = {tk: rets(CL[tk]) for tk in TICKERS}
V = {tk: vol(CL[tk]) for tk in TICKERS}
M = {tk: mean(R[tk]) for tk in TICKERS}

# Part 1's saved findings, recomputed so the restatement is exact
A1_RET = (CL["AAPL"][-1] - CL["AAPL"][0]) / CL["AAPL"][0]
K1_RET = (CL["KO"][-1] - CL["KO"][0]) / CL["KO"][0]
A1_RISK = (max(CL["AAPL"]) - min(CL["AAPL"])) / CL["AAPL"][0]
K1_RISK = (max(CL["KO"]) - min(CL["KO"])) / CL["KO"][0]

RISKIEST = max(TICKERS, key=lambda t: V[t])
CALMEST = min(TICKERS, key=lambda t: V[t])
N_RET = len(R["AAPL"])

cells = []
def md(t): cells.append(new_markdown_cell(t))
def code(t, raises=False):
    c = new_code_cell(t)
    if raises:
        c.metadata["tags"] = ["raises-exception"]
    cells.append(c)

def _hs(hints, sol_code, sol_note):
    if isinstance(hints, str):
        hints = [hints]
    for i, h in enumerate(hints):
        label = "Hint" if len(hints) == 1 else f"Hint {i+1}"
        md(f"<details>\n<summary>\U0001f4a1 {label}</summary>\n\n{h}\n\n</details>")
    md(f"<details>\n<summary>✅ Solution</summary>\n\n```python\n{sol_code}\n```\n\n{sol_note}\n\n</details>")
    md("---")

def q(qid, emoji, title, task, work, hints, sol_code, sol_note):
    md(f"### {emoji} {qid} · {title}\n\n{task}")
    code(work)
    _hs(hints, sol_code, sol_note)

def q_fix(qid, emoji, title, task, demo, work, hints, sol_code, sol_note):
    md(f"### {emoji} {qid} · {title}\n\n{task}")
    code(demo, raises=True)
    md("*Your fix:*")
    code(work)
    _hs(hints, sol_code, sol_note)

def checkpoint(text):
    md(f"> ✅ **Checkpoint** · {text}")
    md("---")

# ================================================================ intro
md(
"# \U0001f5c2️ The Analyst's Notebook · Part 2\n"
"### Session 2\n\n"
"This is the same investigation you started last session. You are a junior analyst "
"building a risk report on a small set of US stocks, and the report grows more "
"detailed as your tools do.\n\n"
"Part 1 ended with a complaint about its own answer. You measured risk as the gap "
"between a stock's highest and lowest close, which uses **two** days out of 252 and "
"throws the rest away. This session you replace it with **volatility**, the "
"standard deviation of every daily return, and you build the calculation yourself "
"out of loops and functions.\n\n"
"You then apply it to five stocks at once and rank them, which is where "
"dictionaries earn their place."
)

md(
"## How to work through this\n\n"
"This is one continuous investigation, not a set of separate exercises. Work "
"**top to bottom, in order**: the functions you write early on are called by the "
"questions that follow, and the numbers you store are reused in the final report.\n\n"
"- Run the **quick load** cell first. It puts last session's data back in memory "
"and restates what you found in Part 1.\n"
"- Replace each `...` with real code and run the cell.\n"
"- Folded **\U0001f4a1 Hint** and **✅ Solution** blocks sit under each question. "
"Try first, then check.\n"
"- One cell in the middle is **broken on purpose** (marked ⚠️) so you can practise "
"reading a real error. Everything else runs cleanly.\n\n"
"**Formulas you will use today**\n\n"
"**Daily return** from one close to the next:\n\n"
r"$$r_i=\dfrac{p_i-p_{i-1}}{p_{i-1}}$$"
"\n\n"
"**Average** of a list of returns:\n\n"
r"$$\bar{r}=\dfrac{1}{n}\sum_i r_i$$"
"\n\n"
"**Volatility**, the standard deviation of the daily returns:\n\n"
r"$$\sigma=\sqrt{\ \dfrac{1}{n}\sum_i \left(r_i-\bar{r}\right)^2\ }$$"
"\n\n"
"In words: take each return's distance from the average, square it, average those "
"squares, and take the square root."
)
md("---")

# ================================================================ quick load
md(
"## \U0001f504 Quick load · pick up where Part 1 left off\n\n"
"Run this cell first. It does two jobs:\n\n"
"1. Loads the 2024 daily closes for **five** stocks into a dictionary called "
"`closes`, and also hands you `aapl_closes` and `ko_closes` directly, as in Part 1.\n"
"2. Restates the four findings you saved in Part 1, so you can compare your new "
"measure against the old one.\n\n"
"You are not expected to follow the loading code. It borrows pandas, the "
"data-table library introduced in Session 3. For now, just run it."
)
code(
f'''# --- Part 1 quick load (provided; you will understand it after Session 3) ---
import os
import pandas as pd

CANDIDATE_DIRS = ["data", os.path.join("..", "data"), "."]

# If the files are not next to the notebook, as in Google Colab, they are
# downloaded from the course repository instead. Nothing to set up.
REPO_RAW_URL = "https://raw.githubusercontent.com/theill95/mlfin-2026/main/data/"

TICKERS = {TICKERS!r}


def load_closes(filename):
    """Return the close column of a two-column CSV as a plain list."""
    for folder in CANDIDATE_DIRS:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return pd.read_csv(path)["close"].tolist()
    if REPO_RAW_URL is not None:
        return pd.read_csv(REPO_RAW_URL + filename)["close"].tolist()
    raise FileNotFoundError(
        f"Could not find {{filename}}. Run this notebook from the course folder "
        f"so the data/ folder is nearby, upload the CSV into Colab, or set "
        f"REPO_RAW_URL to the course repository data URL."
    )


# Every stock's 2024 closes, stored under its ticker
closes = {{}}
for ticker in TICKERS:
    closes[ticker] = load_closes(ticker.lower() + "_2024_closes.csv")

aapl_closes = closes["AAPL"]
ko_closes = closes["KO"]

# --- What you found in Part 1, restated so you can build on it ---
aapl_return = {A1_RET:.4f}    # Apple's 2024 return
ko_return   = {K1_RET:.4f}    # Coca-Cola's 2024 return
aapl_risk   = {A1_RISK:.4f}    # Apple's relative range, Part 1's crude risk gauge
ko_risk     = {K1_RISK:.4f}    # Coca-Cola's relative range

print("Loaded", len(closes), "stocks:", TICKERS)
print("Trading days each:", len(aapl_closes))
print(f"Part 1 recap  Apple:     return {{aapl_return:.2%}}, crude risk {{aapl_risk:.2%}}")
print(f"Part 1 recap  Coca-Cola: return {{ko_return:.2%}}, crude risk {{ko_risk:.2%}}")'''
)
md("---")

# ================================================================ Q1
q("Q1", "\U0001f501", "Every daily return of the year",
r"""Part 1 computed a single return, from the first close of the year to the last. A proper risk
measure needs **every** day's return.

Fill the empty list with Apple's daily returns, one per pair of consecutive closes:

$$r_i=\dfrac{p_i-p_{i-1}}{p_{i-1}}$$

Start the loop at position **1**, because day 0 has no day before it.""",
'''aapl_returns = []
for i in range(1, len(aapl_closes)):
    ...

print("How many returns:", len(aapl_returns))
print("First three:", aapl_returns[:3])''',
["Inside the loop, today is `aapl_closes[i]` and yesterday is `aapl_closes[i - 1]`.",
 "Compute `r = (aapl_closes[i] - aapl_closes[i - 1]) / aapl_closes[i - 1]`, then `aapl_returns.append(r)`."],
'''aapl_returns = []
for i in range(1, len(aapl_closes)):
    r = (aapl_closes[i] - aapl_closes[i - 1]) / aapl_closes[i - 1]
    aapl_returns.append(r)

print("How many returns:", len(aapl_returns))
print("First three:", aapl_returns[:3])''',
f"**{N_RET} returns** from {len(CL['AAPL'])} closes, starting {', '.join(f'{x:.4f}' for x in R['AAPL'][:3])}. One fewer return than prices, always, because the first day has nothing to be compared against. Doing this by hand in Part 1 would have meant 251 lines.")

# ================================================================ Q2
q("Q2", "\U0001f6e0️", "Make it a function you can reuse",
r"""You will need those returns for every stock, not just Apple, so package the loop into a function.

Write `daily_returns(prices)` so it takes any list of prices and hands back the list of returns. The
empty list and the `return` are already in place; write the two lines inside the loop.""",
'''def daily_returns(prices):
    """Daily simple returns from a list of closing prices."""
    result = []
    for i in range(1, len(prices)):
        ...
    return result


# a tiny check you can verify in your head: 100 -> 110 is +10%, 110 -> 99 is -10%
print(daily_returns([100.0, 110.0, 99.0]))

# now the real thing, stored for the questions below
aapl_returns = daily_returns(aapl_closes)
ko_returns = daily_returns(ko_closes)''',
["It is the same two lines as Q1, with `prices` in place of `aapl_closes` and `result` in place of `aapl_returns`.",
 "`r = (prices[i] - prices[i - 1]) / prices[i - 1]`, then `result.append(r)`."],
'''def daily_returns(prices):
    """Daily simple returns from a list of closing prices."""
    result = []
    for i in range(1, len(prices)):
        r = (prices[i] - prices[i - 1]) / prices[i - 1]
        result.append(r)
    return result


print(daily_returns([100.0, 110.0, 99.0]))

aapl_returns = daily_returns(aapl_closes)
ko_returns = daily_returns(ko_closes)''',
"`[0.1, -0.1]` on the tiny list, which is the check you wanted: +10% then -10%. The same function now handles Apple's 252 closes and Coca-Cola's without a single change. Testing a function on a small input you can verify by hand, before trusting it on real data, is a habit worth keeping.")

# ================================================================ Q3
q("Q3", "\U0001f4d0", "The average daily return",
r"""Volatility measures spread **around the average**, so you need the average first.

Write `mean(values)` returning the average of a list, then use it on both stocks.

$$\bar{r}=\dfrac{1}{n}\sum_i r_i$$""",
'''def mean(values):
    """The average of a list of numbers."""
    ...


aapl_mean = mean(aapl_returns)
ko_mean = mean(ko_returns)
print("Apple mean daily return:", aapl_mean)
print("Coca-Cola mean daily return:", ko_mean)''',
["`sum(values) / len(values)` is the whole function body.",
 "Remember to `return` it, not print it, because the questions below need the number back."],
'''def mean(values):
    """The average of a list of numbers."""
    return sum(values) / len(values)


aapl_mean = mean(aapl_returns)
ko_mean = mean(ko_returns)
print("Apple mean daily return:", aapl_mean)
print("Coca-Cola mean daily return:", ko_mean)''',
f"**Apple {M['AAPL']:.4%} a day, Coca-Cola {M['KO']:.4%} a day.** Both tiny, which is normal: a year's worth of growth arrives in very small daily pieces. Note how little these averages tell you about risk. Apple's average day is barely different from Coca-Cola's, yet the two stocks do not feel remotely alike to hold. The difference is in the spread, which is what you measure next.")

checkpoint("you have turned a year of prices into a year of returns, and written two functions you will reuse.")

# ================================================================ Q4
q("Q4", "\U0001f4c9", "Apple's volatility, step by step",
r"""Now the spread. Volatility is the square root of the average squared distance from the mean:

$$\sigma=\sqrt{\ \dfrac{1}{n}\sum_i \left(r_i-\bar{r}\right)^2\ }$$

Build it in the open here, one step at a time, using `aapl_returns` and `aapl_mean` from above. Take
the square root with `** 0.5`.""",
'''squared_total = ...
for r in aapl_returns:
    ...

aapl_vol = ...
print("Apple daily volatility:", aapl_vol)''',
["Start `squared_total = 0`. Each pass adds one squared distance: `squared_total += (r - aapl_mean) ** 2`.",
 "After the loop, divide by how many returns there are and take the square root: `aapl_vol = (squared_total / len(aapl_returns)) ** 0.5`."],
'''squared_total = 0
for r in aapl_returns:
    squared_total += (r - aapl_mean) ** 2

aapl_vol = (squared_total / len(aapl_returns)) ** 0.5
print("Apple daily volatility:", aapl_vol)''',
f"**{V['AAPL']:.4f}, or about {V['AAPL']:.2%} a day.** That is the typical size of an Apple trading day in 2024. Squaring does two jobs: it removes the sign, so a fall counts as risk exactly like a rise, and it weights large moves far more heavily than small ones. The square root at the end puts the answer back into the units of a daily return, which is what makes it readable.")

# ================================================================ Q5
q("Q5", "\U0001f4e6", "Package it, using the functions you already wrote",
r"""Doing that by hand for five stocks would be miserable. Package it.

Write `volatility(prices)` that goes all the way from **prices** to a volatility. The two helpers you
wrote above do the first half of the work, so call them rather than writing their loops again. Only
the squared-distance step and the final line are left for you.""",
'''def volatility(prices):
    """Standard deviation of the daily returns of a price series."""
    r = daily_returns(prices)      # your function from Q2
    m = mean(r)                    # your function from Q3
    squared_total = 0
    for x in r:
        ...                        # add this return's squared distance from m
    return ...                     # average the squares, then square-root


ko_vol = volatility(ko_closes)
print("Coca-Cola daily volatility:", ko_vol)
print("Apple again, should match Q4:", volatility(aapl_closes))''',
["The loop line is the same as Q4 with `x` and `m` in place of `r` and `aapl_mean`: `squared_total += (x - m) ** 2`.",
 "The return line is the same as Q4's last step: `return (squared_total / len(r)) ** 0.5`."],
'''def volatility(prices):
    """Standard deviation of the daily returns of a price series."""
    r = daily_returns(prices)
    m = mean(r)
    squared_total = 0
    for x in r:
        squared_total += (x - m) ** 2
    return (squared_total / len(r)) ** 0.5


ko_vol = volatility(ko_closes)
print("Coca-Cola daily volatility:", ko_vol)
print("Apple again, should match Q4:", volatility(aapl_closes))''',
f"**Coca-Cola {V['KO']:.4f} ({V['KO']:.2%} a day)**, and Apple comes back as {V['AAPL']:.4f}, matching Q4 exactly. That match is the point of the check: the packaged version and the hand-built one are the same calculation. Notice that `volatility` is short because it delegates. Small functions that call other small functions is how real analysis code is kept readable.")

# ================================================================ Q6
q("Q6", "⚖️", "Was Part 1's crude gauge wrong?",
r"""You now have two measures of the same thing. Part 1's relative range (`aapl_risk`, `ko_risk`, both
restored by the quick load) used two days out of 252. Your volatility uses all of them.

Build one line that puts them side by side, something like:

`Apple: vol 1.41% vs range 51.22%  |  Coca-Cola: vol 0.80% vs range 26.15%`

Then read the note below, because the interesting part is what the comparison does **not** show.""",
'''comparison = ...
print(comparison)''',
["Build a single f-string holding all four saved values, each formatted with `:.2%`.",
 '`comparison = f"Apple: vol {aapl_vol:.2%} vs range {aapl_risk:.2%}  |  Coca-Cola: vol {ko_vol:.2%} vs range {ko_risk:.2%}"`'],
'''comparison = f"Apple: vol {aapl_vol:.2%} vs range {aapl_risk:.2%}  |  Coca-Cola: vol {ko_vol:.2%} vs range {ko_risk:.2%}"
print(comparison)''',
f"`Apple: vol {V['AAPL']:.2%} vs range {A1_RISK:.2%}  |  Coca-Cola: vol {V['KO']:.2%} vs range {K1_RISK:.2%}`.\n\n"
f"The two measures disagree wildly in **size** and agree on the **ranking**: Apple is riskier either way. "
f"The sizes are not comparable, and should not be: a range is a whole year's travel, a volatility is one "
f"typical day. The ratios are worth a look though. By range Apple was {A1_RISK / K1_RISK:.2f} times Coca-Cola; "
f"by volatility it is {V['AAPL'] / V['KO']:.2f} times. So Part 1's gauge was not misleading here, it was just "
f"crude, and it overstated the gap. A range is at the mercy of two single days, and one freak session can "
f"set it. Volatility is an average over 251 days, so no single day can dominate it. That robustness is why "
f"it is the standard measure.")

checkpoint("you have replaced the crude risk gauge with a proper one, and checked the two against each other.")

# ================================================================ Q7 (two-cell bug)
q_fix("Q7", "\U0001f41e", "Fix a colleague's returns loop",
r"""⚠️ **Run the cell below.** A colleague wrote their own returns loop and it crashes. Read the error,
work out which pass fails and why, then write a corrected version underneath.

Your fixed list should be exactly as long as the `aapl_returns` you built earlier.""",
'''broken_returns = []
for i in range(1, len(aapl_closes) + 1):
    broken_returns.append((aapl_closes[i] - aapl_closes[i - 1]) / aapl_closes[i - 1])
print(len(broken_returns))''',
'''fixed_returns = []
...

print("How many:", len(fixed_returns))
print("Same as Q1?", len(fixed_returns) == len(aapl_returns))''',
["There are 252 closes, so the valid positions are 0 to 251. The `+ 1` makes the last pass ask for `aapl_closes[252]`, which does not exist.",
 "Drop the `+ 1`: `for i in range(1, len(aapl_closes)):`. Starting at 1 is correct and must stay, because the body reads `aapl_closes[i - 1]`."],
'''fixed_returns = []
for i in range(1, len(aapl_closes)):
    fixed_returns.append((aapl_closes[i] - aapl_closes[i - 1]) / aapl_closes[i - 1])

print("How many:", len(fixed_returns))
print("Same as Q1?", len(fixed_returns) == len(aapl_returns))''',
f"**{N_RET}, and it matches Q1.** The bug is an `IndexError` on the very last pass. It is worth being precise about the two ends of this loop, because they are different bugs: the **start** at 1 protects `[i - 1]` at the bottom, and the **stop** at `len(prices)` protects `[i]` at the top. Change either one and you get a wrong answer or a crash.")

# ================================================================ Q8
q("Q8", "\U0001f5c2️", "All five stocks at once",
r"""This is where a dictionary earns its keep. `closes` holds five price series, each under its ticker.

Loop over it and build a second dictionary, `vol`, holding each stock's volatility. Your `volatility`
function does all the work, so the loop body is two short lines.""",
'''vol = {}
for ticker in closes:
    ...

print(vol)''',
["Looping a dictionary gives you the **keys**, so `ticker` is `\"AAPL\"`, `\"KO\"` and so on.",
 "Read the prices with `closes[ticker]`, then store the answer with `vol[ticker] = volatility(closes[ticker])`."],
'''vol = {}
for ticker in closes:
    vol[ticker] = volatility(closes[ticker])

print(vol)''',
"Five volatilities, each under its ticker: " + ", ".join(f"**{tk}** {V[tk]:.2%}" for tk in TICKERS) +
". One loop applied a function you wrote to every stock you have. Adding a sixth stock would mean adding one CSV, and not one line of this analysis.")

# ================================================================ Q9
q("Q9", "\U0001f3c6", "The riskiest and the calmest",
r"""Rank them. Find the ticker with the **highest** volatility and the one with the **lowest**, using
only a loop and an `if`.

Python does have shortcuts for this, and section L of the exercises shows you `sorted` and
`max(vol.values())`. Do it the long way here anyway, once, so you know what those shortcuts are doing
on your behalf.

Keep a best-so-far as you go. Start `highest` at **0**, because every volatility is above zero, and
start `lowest` at **1.0**, because no stock moves 100% on a typical day.""",
'''riskiest = ""
calmest = ""
highest = 0
lowest = 1.0

for ticker in vol:
    ...

print("Riskiest:", riskiest)
print("Calmest: ", calmest)''',
["Two `if` statements inside the one loop, each comparing `vol[ticker]` against the best seen so far.",
 "`if vol[ticker] > highest:` then update **both** `highest = vol[ticker]` and `riskiest = ticker`. The lowest works the same way with `<` and `lowest`."],
'''riskiest = ""
calmest = ""
highest = 0
lowest = 1.0

for ticker in vol:
    if vol[ticker] > highest:
        highest = vol[ticker]
        riskiest = ticker
    if vol[ticker] < lowest:
        lowest = vol[ticker]
        calmest = ticker

print("Riskiest:", riskiest)
print("Calmest: ", calmest)''',
f"**Riskiest {RISKIEST} ({V[RISKIEST]:.2%} a day), calmest {CALMEST} ({V[CALMEST]:.2%} a day).** {NAMES[RISKIEST]} moves about {V[RISKIEST] / V[CALMEST]:.1f} times as much on a typical day as {NAMES[CALMEST]}, which is the sort of gap that decides how much of each you can hold. The pattern here, carry a best-so-far and replace it whenever you find better, is how you find an extreme in one pass, and it is what `max` does internally.")

# ================================================================ Q10
q("Q10", "\U0001f4dd", "The page of the report",
r"""The payoff. Using the values you have stored, print one line per stock, ordered as they sit in the
dictionary, reading like:

```
AAPL  vol 1.41%
KO    vol 0.80%
```

Then add a closing line naming the riskiest and calmest stock.""",
'''for ticker in vol:
    ...

summary = ...
print(summary)''',
['Inside the loop, `print(ticker, f"vol {vol[ticker]:.2%}")` is enough.',
 '`summary = f"Riskiest: {riskiest} | Calmest: {calmest}"`, reusing what Q9 stored.'],
'''for ticker in vol:
    print(ticker, f"vol {vol[ticker]:.2%}")

summary = f"Riskiest: {riskiest} | Calmest: {calmest}"
print(summary)''',
"Five formatted lines and a verdict, and every number in them was computed by a function you wrote. "
f"`Riskiest: {RISKIEST} | Calmest: {CALMEST}`. This is a real page of a risk report: not a rough gauge from two days, but a measure that used all 251 daily moves of every stock.")

checkpoint("you have ranked five stocks on a proper measure of risk, built entirely from your own functions.")

# ================================================================ Q11 framing
md(
"### \U0001f52e Q11 · Where the case goes next\n\n"
"Part 2 leaves the report in decent shape. You measured risk honestly, on every trading day, for "
"five stocks. Three things are still missing, and each is a later part of the case.\n\n"
"1. **Do it for everything, and see it.** Five stocks took five CSV files and a loop. The full set "
"has eleven, along with a decade of history rather than one year, and none of it has been plotted. "
"A data table (pandas) holds all of it at once, and a chart shows in one glance what a column of "
"numbers hides. *(Session 3.)*\n"
f"2. **Ask whether it lasts.** You measured {NAMES[RISKIEST]} as the most volatile stock of 2024. "
"Was it also the most volatile in 2023? Does a stock that moved a lot last year tend to move a lot "
"next year? If volatility persists, then measuring it is not only description, it is the beginning "
"of a forecast. *(Session 4.)*\n"
"3. **Say what you would predict, and from what.** To go further you have to be precise about what "
"you are predicting (a target) and what you would use to predict it (features), and about never "
"letting information from the future leak into either. That framing is the foundation of machine "
"learning. *(Session 4.)*\n\n"
"No model is fitted in these four sessions. The work here is the honest groundwork: measure "
"carefully, then state exactly what question you would ask next."
)
md("---")

# ================================================================ carry forward
md(
"## \U0001f4cc Part 2 findings · carry into Part 3\n\n"
"The investigation now leaves these in the notebook's memory:\n\n"
"- `daily_returns(prices)`, `mean(values)` and `volatility(prices)`: three functions you wrote\n"
"- `aapl_vol` and `ko_vol`: the two stocks from Part 1, measured properly\n"
"- `vol`: a dictionary of all five stocks' volatilities\n"
"- `riskiest` and `calmest`: the two ends of the ranking\n\n"
f"Apple's daily volatility is {V['AAPL']:.2%} against Coca-Cola's {V['KO']:.2%}, so the Part 1 ranking "
f"survives a much better measurement. Across all five, {RISKIEST} is the most volatile at "
f"{V[RISKIEST]:.2%} a day and {CALMEST} the least at {V[CALMEST]:.2%}.\n\n"
"Session 3 continues from here. Everything you did with loops and dictionaries, pandas does on a "
"whole table at once, for eleven stocks and ten years, and matplotlib finally lets you look at it."
)
md(
"## \U0001f3c1 Part 2 complete\n\n"
"Using loops, conditions, functions and dictionaries, you turned a year of prices into a year of "
"returns, built the standard measure of risk from its definition, checked it against the crude gauge "
"from Part 1, fixed an off-by-one bug, and ranked five stocks without a sorting function.\n\n"
"The measure you built by hand here is the one every risk system in the industry starts from. From "
"Session 3 you will get it in a single call, and you will know exactly what that call is doing.\n\n"
"*Stuck for more than 15 minutes? Ask a friend, ask an AI for a hint (not the answer), or email me "
"at `jobo@econ.au.dk`.*"
)

# ---------------------------------------------------------------- write
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
# Deterministic cell ids: nbformat assigns a fresh random uuid to every cell,
# which would make every regeneration look like a rewrite in git.
for _i, _c in enumerate(nb.cells):
    _c["id"] = f"c{_i:04d}"

OUT.write_text(nbf.writes(nb), encoding="utf-8")
n_q = sum(1 for c in cells if c.cell_type == "markdown" and c.source.startswith("### "))
print(f"wrote {OUT}  ({len(cells)} cells, {n_q} questions)")
print("vol:", {k: round(v, 5) for k, v in V.items()})
print("riskiest:", RISKIEST, "calmest:", CALMEST)
