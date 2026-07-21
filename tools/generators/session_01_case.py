# -*- coding: utf-8 -*-
"""Regenerate session_01_case.ipynb (v2, per instructor review).

The case is a single cumulative investigation: each question reuses values the
earlier ones stored (Q2 reuses Q1; the risk gauge reuses the extremes; the
final report reuses all four stored results). No star badges; difficulty rises
naturally. Same polish as the exercises (emoji headers, folded hints/solutions,
stated formulas, no em-dashes, two-cell error pattern). Only Session 1 tools.
No modelling: it ends by framing the prediction question for later sessions.
"""
import os
from pathlib import Path
import pandas as pd
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "session_01" / "session_01_case.ipynb"
REPO = Path(__file__).resolve().parents[2]

# ---- compute exact answers from the real CSVs ----
def load(f):
    for d in ["data", "session_01/data"]:
        p = REPO / d / f
        if p.exists():
            t = pd.read_csv(p)
            return t["date"].tolist(), t["close"].tolist()
    raise FileNotFoundError(f)

ad, ac = load("aapl_2024_closes.csv")
kd, kc = load("ko_2024_closes.csv")
A = dict(
    n=len(ac), first=ac[0], last=ac[-1],
    d101=ad[100], p101=ac[100],
    janret=(ac[0:21][-1] - ac[0:21][0]) / ac[0:21][0], jann=len(ac[0:21]),
    high=max(ac), low=min(ac),
    hidate=ad[ac.index(max(ac))], lodate=ad[ac.index(min(ac))],
    ret=(ac[-1] - ac[0]) / ac[0], risk=(max(ac) - min(ac)) / ac[0],
)
K = dict(
    first=kc[0], last=kc[-1], high=max(kc), low=min(kc),
    ret=(kc[-1] - kc[0]) / kc[0], risk=(max(kc) - min(kc)) / kc[0],
)

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
"# \U0001f5c2️ The Analyst's Notebook · Part 1\n"
"### Session 1\n\n"
"This is the course case, which you work on between sessions. Across the four "
"sessions you take the role of a junior analyst building a risk report on a small "
"set of US stocks. It is a single investigation that continues from one session "
"to the next: the same case each time, growing more detailed as you learn more "
"tools.\n\n"
"For Part 1 the question is straightforward. Apple and Coca-Cola both ended 2024 "
"higher than they started. Which of the two moved around more during the year, "
"and is moving more the same thing as being worse?\n\n"
"You answer it using only the Python from Session 1: lists, indexing, slicing, a "
"few functions, and the simple-return formula. A data-table library (pandas) "
"would make this shorter; it is introduced in Session 3."
)

md(
"## How to work through this\n\n"
"This is one continuous investigation, not a set of separate exercises. Work "
"**top to bottom, in order**: each question stores a result (a return, a high, a "
"low) that a later question picks up and uses. Fill in each cell before moving to "
"the next, and the results carry through to a final summary you put together at "
"the end.\n\n"
"- Replace each `...` with real code and run the cell.\n"
"- Folded **\U0001f4a1 Hint** and **✅ Solution** blocks sit under each question. "
"Try first, then check.\n"
"- One cell in the middle is **broken on purpose** (marked ⚠️) so you can practise "
"reading a real error. Everything else runs cleanly.\n\n"
"**Formulas you will use today**\n\n"
"**Simple return** over a period:\n\n"
r"$$r=\dfrac{p_{\text{last}}-p_{\text{first}}}{p_{\text{first}}}$$"
"\n\n"
"**Relative range** (today's rough risk gauge):\n\n"
r"$$\text{risk}=\dfrac{\max-\min}{p_{\text{first}}}$$"
"\n"
)
md("---")

# ================================================================ setup
md(
"## \U0001f504 Setup · load the data\n\n"
"Run this cell first. It reads two real price series that ship with the course, "
"Apple (AAPL) and Coca-Cola (KO) daily closes for every trading day of 2024, and "
"hands you four plain lists: `aapl_dates`, `aapl_closes`, `ko_dates`, `ko_closes`.\n\n"
"You are not expected to follow this cell yet. It borrows pandas, the data-table "
"library introduced in Session 3. For now, just run it."
)
code(
'''# --- Course data loader (provided; you will understand it after Session 3) ---
# It reads two CSV files and hands you four plain Python lists to work with
# using this week's skills: aapl_dates, aapl_closes, ko_dates, ko_closes.
import os
import pandas as pd

# The data ships in a folder called "data". We look in a few likely places so
# the notebook works from the repository root or from the session_01 folder.
CANDIDATE_DIRS = ["data", os.path.join("..", "data"), "."]

# If the files are not next to the notebook, as in Google Colab, they are
# downloaded from the course repository instead. Nothing to set up.
REPO_RAW_URL = "https://raw.githubusercontent.com/theill95/mlfin-2026/main/data/"


def load_closes(filename):
    """Return (dates, closes) as two plain lists from a two-column CSV."""
    for folder in CANDIDATE_DIRS:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            table = pd.read_csv(path)
            return table["date"].tolist(), table["close"].tolist()
    if REPO_RAW_URL is not None:
        table = pd.read_csv(REPO_RAW_URL + filename)
        return table["date"].tolist(), table["close"].tolist()
    raise FileNotFoundError(
        f"Could not find {filename}. Run this notebook from the course folder "
        f"so the data/ folder is nearby, upload the CSV into Colab, or set "
        f"REPO_RAW_URL to the course repository data URL."
    )


aapl_dates, aapl_closes = load_closes("aapl_2024_closes.csv")
ko_dates, ko_closes = load_closes("ko_2024_closes.csv")

print("Loaded Apple:", len(aapl_closes), "trading days")
print("Loaded Coca-Cola:", len(ko_closes), "trading days")'''
)
md("---")

# ================================================================ Q1
q("Q1", "\U0001f4cb", "Get your bearings",
r"""Before analysing anything, an analyst checks the shape of the data. For Apple, store three
facts in named variables (you will reuse them below): the number of trading days, the **first**
close of the year, and the **last**.""",
'''n_days = ...
aapl_first = ...
aapl_last = ...
print("Trading days:", n_days)
print("First close:", aapl_first)
print("Last close:", aapl_last)''',
"`len(...)` counts the items; index `0` is the first, `-1` is the last.",
'''n_days = len(aapl_closes)
aapl_first = aapl_closes[0]
aapl_last = aapl_closes[-1]
print("Trading days:", n_days)
print("First close:", aapl_first)
print("Last close:", aapl_last)''',
f"**{A['n']} trading days, first close {A['first']:.2f}, last close {A['last']:.2f}.** A US stock trades about {A['n']} days a year, a handy number to carry in your head. Keep `aapl_first` and `aapl_last` in mind: the next question uses them.")

# ================================================================ Q2
q("Q2", "\U0001f4c8", "Apple's year",
r"""Now use the two prices you just stored to compute Apple's **return over the whole of 2024**, and
show it as a percentage:

$$r=\dfrac{p_{\text{last}}-p_{\text{first}}}{p_{\text{first}}}$$

Store it as `aapl_return`, because the final report will need it.""",
'''aapl_return = ...
print("Apple 2024 return (decimal):", aapl_return)''',
["Reuse `aapl_first` and `aapl_last` from Q1 in the return formula.",
 "`aapl_return = (aapl_last - aapl_first) / aapl_first`, then read it back with `f\"{aapl_return:.2%}\"`."],
'''aapl_return = (aapl_last - aapl_first) / aapl_first
print(f"Apple 2024 return: {aapl_return:.2%}")''',
f"**Apple 2024 return: {A['ret']:.2%}.** A strong year. Notice you did not use the raw list here; you built this from the two values Q1 stored. Later questions reuse earlier results in the same way.")

# ================================================================ Q3
q("Q3", "\U0001f50e", "Look up one day",
r"""The date lists line up with the price lists position by position: `aapl_dates[i]` is the date of
`aapl_closes[i]`. Report the date **and** the price of Apple's **101st** trading day of 2024
(the 101st day sits at index 100).""",
'''day_index = 100
that_date = ...
that_price = ...
print("Date:", that_date, " Price:", that_price)''',
"Use the same index `100` on both lists: one gives the date, the other the price.",
'''day_index = 100
that_date = aapl_dates[day_index]
that_price = aapl_closes[day_index]
print(f"On {that_date}, Apple closed at {that_price:.2f}")''',
f"**On {A['d101']}, Apple closed at {A['p101']:.2f}.** Two lists kept in step, joined only by a shared position. In Session 3 a pandas table will keep the date and price together in one row, so they cannot drift apart.")

# ================================================================ Q4
q("Q4", "\U0001f5d3️", "A month on its own",
r"""Apple's first **21** entries are the trading days of January 2024 (positions 0 to 20). Slice out
January's closes into `january`, then compute January's own return from its first close to its last,
and show it as a percentage.""",
'''january = ...
january_return = ...
print("January return (decimal):", january_return)''',
["To take positions 0 through 20, slice `[0:21]`; the stop index is excluded, so it is one past the last position you want.",
 "The return uses `january[0]` and `january[-1]` in the simple-return formula."],
'''january = aapl_closes[0:21]
january_return = (january[-1] - january[0]) / january[0]
print(f"January trading days: {len(january)}")
print(f"January return: {january_return:.2%}")''',
f"**{A['jann']} trading days, January return {A['janret']:.2%}.** Apple slipped a little in January before its strong year. That is a description of what happened, not a prediction of what came next; the course keeps that distinction clear.")

# ================================================================ Q5
q("Q5", "\U0001f3d4️", "Extremes, and when they happened",
r"""An analyst reports extremes. Find Apple's **highest** and **lowest** closes of 2024 and the
**dates** they fell on. Store `aapl_high` and `aapl_low`; the risk gauge in Q7 will reuse them.

Use `max`/`min` for the prices, `.index(...)` to find *where* each sits, then read that same
position in the date list.""",
'''aapl_high = ...
aapl_low = ...
high_date = ...      # date of the highest close
low_date = ...       # date of the lowest close
print("Highest:", aapl_high, "on", high_date)
print("Lowest: ", aapl_low, "on", low_date)''',
["`max(aapl_closes)` is the highest price; `aapl_closes.index(aapl_high)` is its position.",
 "Feed that position into `aapl_dates[...]` to get the date, exactly like the lecture's whole-week slide."],
'''aapl_high = max(aapl_closes)
aapl_low = min(aapl_closes)
high_date = aapl_dates[aapl_closes.index(aapl_high)]
low_date = aapl_dates[aapl_closes.index(aapl_low)]
print(f"Highest: {aapl_high:.2f} on {high_date}")
print(f"Lowest:  {aapl_low:.2f} on {low_date}")''',
f"**Highest {A['high']:.2f} on {A['hidate']}, lowest {A['low']:.2f} on {A['lodate']}.** The low came in spring and the high near year-end: Apple climbed across the year. `aapl_high` and `aapl_low` are now saved for the risk calculation to come.")

checkpoint("you have profiled Apple: its size, its year's return, a single day, a month, and its extremes.")

# ================================================================ Q6 (two-cell bug)
q_fix("Q6", "\U0001f41e", "Fix a colleague's bug",
r"""⚠️ **Run the cell below.** A colleague tried to grab Apple's last close by asking for position
252, and it fails. Read the error, then fix it underneath so `checked_last` holds the true final
price. (It should match the `aapl_last` you stored in Q1.)""",
'''checked_last = aapl_closes[252]
print(checked_last)''',
'''checked_last = ...
print("Last close:", checked_last)
print("Matches Q1?", checked_last == aapl_last)''',
["There are 252 prices, so the valid positions are 0 to 251. Position 252 does not exist, which is the `IndexError`.",
 "The last close is `aapl_closes[-1]` (or `aapl_closes[251]`), never `aapl_closes[252]`."],
'''checked_last = aapl_closes[-1]
print("Last close:", checked_last)
print("Matches Q1?", checked_last == aapl_last)''',
f"**{A['last']:.2f}, and it matches Q1.** The bug is an `IndexError`: a 252-item list stops at position 251, so 252 runs one step off the end. `[-1]` gives the last item whatever the length, so it avoids this kind of off-by-one error.")

# ================================================================ Q7
q("Q7", "\U0001f4cf", "A first risk gauge for Apple",
r"""A first, rough measure of risk is a stock's price range relative to where it started, which
captures how far it moved over the year:

$$\text{risk}=\dfrac{\max-\min}{p_{\text{first}}}$$

Build it for Apple from the values you already stored, `aapl_high`, `aapl_low` (Q5) and `aapl_first`
(Q1). Store it as `aapl_risk`.""",
'''aapl_risk = ...
print("Apple relative range (decimal):", aapl_risk)''',
["Reuse your saved values: `(aapl_high - aapl_low) / aapl_first`.",
 "Then read it as a percentage with `f\"{aapl_risk:.2%}\"`."],
'''aapl_risk = (aapl_high - aapl_low) / aapl_first
print(f"Apple relative range: {aapl_risk:.2%}")''',
f"**Apple relative range: {A['risk']:.2%}.** Its price moved across a band worth about half its starting value. This is a rough measure: it uses only the two extreme prices, not the day-to-day movement. Session 2 introduces loops, which let you use every daily return and compute a proper volatility. For now it is a reasonable first gauge, saved as `aapl_risk`.")

# ================================================================ Q8
q("Q8", "\U0001f964", "Do the whole thing for Coca-Cola",
r"""To compare the two fairly, Coca-Cola needs the same numbers. In one cell, compute and store
Coca-Cola's **year return** (`ko_return`) and its **relative range** (`ko_risk`), reusing the exact
recipes from Q2 and Q7, now on the `ko_closes` list.""",
'''ko_first = ...
ko_return = ...
ko_risk = ...
print("Coca-Cola return (decimal):", ko_return)
print("Coca-Cola relative range (decimal):", ko_risk)''',
["`ko_return` is `(ko_closes[-1] - ko_closes[0]) / ko_closes[0]`.",
 "`ko_risk` is `(max(ko_closes) - min(ko_closes)) / ko_closes[0]`, the same shape as `aapl_risk`."],
'''ko_first = ko_closes[0]
ko_return = (ko_closes[-1] - ko_first) / ko_first
ko_risk = (max(ko_closes) - min(ko_closes)) / ko_first
print(f"Coca-Cola return: {ko_return:.2%}")
print(f"Coca-Cola relative range: {ko_risk:.2%}")''',
f"**Coca-Cola return {K['ret']:.2%}, relative range {K['risk']:.2%}.** A positive but much gentler year than Apple's. These are the same two formulas from Q2 and Q7, applied to a different list.")

# ================================================================ Q9
q("Q9", "⚖️", "Head to head",
r"""The payoff. You now hold four saved numbers: `aapl_return`, `ko_return`, `aapl_risk`, `ko_risk`.
Assemble them into a single report line that lays the two stocks side by side, something like:

`Apple: return 35.56%, risk 51.22%  |  Coca-Cola: return 7.26%, risk 26.15%`""",
'''report = ...
report''',
["Build one f-string that drops all four saved values in, each formatted with `:.2%`.",
 '`report = f"Apple: return {aapl_return:.2%}, risk {aapl_risk:.2%}  |  Coca-Cola: return {ko_return:.2%}, risk {ko_risk:.2%}"`'],
'''report = f"Apple: return {aapl_return:.2%}, risk {aapl_risk:.2%}  |  Coca-Cola: return {ko_return:.2%}, risk {ko_risk:.2%}"
report''',
f"`'Apple: return {A['ret']:.2%}, risk {A['risk']:.2%}  |  Coca-Cola: return {K['ret']:.2%}, risk {K['risk']:.2%}'`. Every number here was computed and stored earlier; this line only gathers them. Apple was both the higher earner and the more volatile of the two. That is the main point of the report: risk and return are separate measurements, and a stock can rank high or low on each independently. The course treats them separately throughout.")

checkpoint("you have compared the two stocks on both risk and return.")

# ================================================================ Q10 framing
md(
"### \U0001f52e Q10 · Where the case goes next\n\n"
"So far you have *described* 2024: Apple moved more than Coca-Cola, and returned more as well. "
"Description is where analysis starts. The three questions below are ones you cannot answer yet, "
"and each points to a later part of the case.\n\n"
"1. **Measure risk properly.** The relative range used only two days, the highest and the lowest. "
"A better measure uses *every* day's move. That is volatility, and computing it over a full year is "
"what loops are for. *(Session 2.)*\n"
"2. **Scale to many stocks.** Two stocks was manageable. The full set has eleven, and doing this "
"list by list would be tedious. A data table (pandas) handles all of them at once, and lets you "
"plot the results. *(Session 3.)*\n"
"3. **Ask whether it could be predicted.** Apple was more volatile in 2024, but could you have "
"known that on 1 January? Does last year's movement tell you anything about next year's? Setting "
"this up properly, with features and a target, is the foundation of machine learning. *(Session 4.)*\n\n"
"No model is fitted in these first four sessions. The task now is the honest first step: looking "
"carefully at what happened, and naming what you would need to go further. Building the models "
"themselves comes later in the course."
)
md("---")

# ================================================================ carry forward
md(
"## \U0001f4cc Part 1 findings · carry into Part 2\n\n"
"The investigation leaves four results stored in the notebook's memory:\n\n"
"- `aapl_return` and `ko_return`: each stock's 2024 return\n"
"- `aapl_risk` and `ko_risk`: each stock's relative-range risk gauge\n\n"
f"Apple was the more volatile of the two ({A['risk']:.2%} against Coca-Cola's {K['risk']:.2%}), "
f"even though it also returned more ({A['ret']:.2%} against {K['ret']:.2%}). Moving more is not the "
"same as being worse.\n\n"
"Session 2 continues from here: it re-loads the data and restates these findings, then uses loops "
"and functions to replace the crude relative-range gauge with a proper volatility computed from "
"every daily return. Scaling up to all eleven stocks with pandas comes in Session 3."
)
md(
"## \U0001f3c1 Part 1 complete\n\n"
"Using only Session 1 Python, you produced the first page of a risk report: you checked the shape "
"of two real price series, measured returns over a month and a year, found the extreme days, fixed "
"an `IndexError`, computed a first risk gauge, and set the two stocks side by side, carrying each "
"result forward to the next step.\n\n"
"In Session 2, loops and functions will let you compute a proper volatility from every daily "
"return, rather than just the highest and lowest price. Pandas, plots, and all eleven stocks follow "
"in Session 3.\n\n"
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
