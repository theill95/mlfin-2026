# -*- coding: utf-8 -*-
"""Build session_03_exercises.ipynb.

Same conventions as Sessions 1 and 2: pleasant intro, 4-star badges, toolkit
card with <abbr> hover docs, task -> work cell (blank-safe `...`) -> 1-2 folded
hints -> folded solution, no em-dashes, plain academic tone.

Only tools taught in the Session 3 deck (plus Sessions 1 and 2). That means:
numpy arrays / slicing / aggregations / masks / 2-D arrays / .T / @ /
linalg.inv / linalg.solve; pandas Series and DataFrame / read_csv / head /
shape / dtypes / columns / column selection / boolean filtering / new columns /
pct_change / dropna / std / mean / sort_values / groupby / agg / pivot / loc;
matplotlib subplots / plot / barh / hist / scatter / set_title / set_ylabel /
set_xlabel / legend / savefig.  NOT taught, so never required outside section
K: describe, nlargest, corr, iloc beyond a passing mention, resample, apply.

BLANK-SAFE RULES for this session:
- every work cell assigns `x = ...` and then displays `x` bare. Never call a
  method on, index into, format, or do arithmetic with a placeholder.
- plotting cells always contain a complete `fig, ax = plt.subplots(...)` and a
  bare `...` line for the student's calls, so a blank cell draws empty axes
  instead of raising.
- nothing depends on an earlier exercise having been solved: the shared setup
  cell at the top provides every variable the tasks use.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "session_03" / "session_03_exercises.ipynb"

cells = []
LABELS = {1: "Warm-up", 2: "Core", 3: "Challenge", 4: "Stretch"}


def badge(n):
    return ("★" * n + "☆" * (4 - n)) + " " + LABELS[n]


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


def ex(sid, title, n, task, work, hints, sol_code, sol_note):
    md(f"### {sid} · {title}  {badge(n)}\n\n{task}")
    code(work)
    _hints_solution(hints, sol_code, sol_note)


def section(header):
    md(header)


# ---- real numbers, computed here so every solution note is exact -----------
PX = pd.read_csv(ROOT / "data" / "prices.csv", parse_dates=["date"])
WIDE = PX.pivot(index="date", columns="ticker", values="close")
RET = WIDE.pct_change()
AAPL = PX[PX["ticker"] == "AAPL"].reset_index(drop=True)
A24 = pd.read_csv(ROOT / "data" / "aapl_2024_closes.csv", parse_dates=["date"])
CLOSE24 = A24["close"].to_numpy()
R24 = (CLOSE24[1:] - CLOSE24[:-1]) / CLOSE24[:-1]

WEEK = np.array([183.56, 182.19, 179.87, 179.15, 183.48])
WEEK_R = (WEEK[1:] - WEEK[:-1]) / WEEK[:-1]

VOL_FULL = RET.std().sort_values(ascending=False)
VOL_24 = (RET.loc["2024"].std() * np.sqrt(252)).sort_values(ascending=False)
AAPL_STD = AAPL["close"].pct_change().std()
KO_STD = WIDE["KO"].pct_change().std()

# ---------------------------------------------------------------- top matter
md(
"# \U0001f4d3 Session 3 · Exercises\n"
"### NumPy, pandas, and matplotlib\n\n"
"Last session you wrote the loops yourself. These exercises are about handing "
"that work to three packages, so you can ask bigger questions of a much bigger "
"table.\n\n"
"Most tasks ask you to write or complete a short piece of code. They start easy "
"and build up, and they all use the same real price data as the lecture."
)

md(
"## How to use this notebook\n\n"
"- Run the **setup cell** below first. It imports the three packages and loads "
"the price data, and every exercise uses what it creates.\n"
"- Each exercise has a **task**, then a **code cell** for your work. Cells with "
"`...` are blanks to fill in. Replace them with real code.\n"
"- Stuck? Open the **\U0001f4a1 Hint**, but only after a genuine attempt. Open the "
"**✅ Solution** to *check* yourself, not to skip the thinking.\n"
"- Every cell runs cleanly even with the blanks still in place, so pressing "
"**Run all** never floods you with errors.\n"
"- Exercises do not depend on each other. If one defeats you, move on.\n\n"
"**You are not expected to finish all of these.** Do what you can, and come back "
"to the rest when you revise. Short on time? Read the hint, then the solution. A "
"worked solution you genuinely understand is real learning too."
)

md(
"### Difficulty\n\n"
"| badge | level | what to expect |\n"
"|:--|:--|:--|\n"
"| ★☆☆☆ | **Warm-up** | one idea, straight from the lecture |\n"
"| ★★☆☆ | **Core** | the skill you will actually use. Most exercises live here |\n"
"| ★★★☆ | **Challenge** | combine a few ideas, or think one step past what you were shown |\n"
"| ★★★★ | **Stretch** | a real puzzle for those who want one. Rare, and always solvable with today's tools |\n"
)

md(
"## \U0001f9f0 Your toolkit for today\n\n"
"Everything you need is on this card, plus everything from Sessions 1 and 2. Only "
"section **K** goes past it, deliberately: that section is about looking things up, "
"which is the skill that matters most once a package has hundreds of functions.\n\n"
"> Names in brackets (`values`, `frame`, `column`, ...) are **placeholders**: put "
"your own variable there. **Hover any tool** to see what it does.\n\n"
'<p><strong>NumPy arrays</strong><br>\n'
'<abbr title="Turn a list of numbers into an array, which does arithmetic on every element at once."><code>np.array(list)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="How many rows and columns. For a 1-D array it prints as (n,)."><code>values.shape</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Everything from position 1 onwards. Same slicing as a list."><code>values[1:]</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Everything except the last element."><code>values[:-1]</code></abbr> &nbsp;·&nbsp; '
'<abbr title="The square root, element by element."><code>np.sqrt(x)</code></abbr></p>\n\n'
'<p><strong>Summarising an array</strong><br>\n'
'<abbr title="The average of the values."><code>values.mean()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="The standard deviation, which is what volatility is."><code>values.std()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="The smallest value."><code>values.min()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="The largest value."><code>values.max()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Add every value up."><code>values.sum()</code></abbr></p>\n\n'
'<p><strong>Asking a question of every element</strong><br>\n'
'<abbr title="An array of True and False, one per element."><code>values &gt; 0</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Keep only the elements where the condition is True."><code>values[values &gt; 0]</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Count the Trues, because True counts as 1."><code>(values &gt; 0).sum()</code></abbr></p>\n\n'
'<p><strong>Matrices</strong><br>\n'
'<abbr title="A 2-D array: a list of rows, each row a list of numbers."><code>np.array([[1, 2], [3, 4]])</code></abbr> &nbsp;·&nbsp; '
'<abbr title="One element: row 0, column 1. Rows first, then columns."><code>X[0, 1]</code></abbr> &nbsp;·&nbsp; '
'<abbr title="A whole row, or a whole column. The colon means all of them."><code>X[0, :]</code></abbr> &nbsp;·&nbsp; '
'<abbr title="The transpose: rows become columns."><code>X.T</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Matrix multiplication. For two vectors this is the dot product: multiply pairwise, then add."><code>A @ B</code></abbr> &nbsp;·&nbsp; '
'<abbr title="A column of ones, n long. Useful as the intercept column of a regression."><code>np.ones(n)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Glue 1-D arrays together side by side, as columns of a matrix."><code>np.column_stack([a, b])</code></abbr> &nbsp;·&nbsp; '
'<abbr title="The inverse: the matrix that undoes A."><code>np.linalg.inv(A)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Solve A x = b for x. Safer and faster than inverting."><code>np.linalg.solve(A, b)</code></abbr></p>\n\n'
'<p><strong>pandas: one column</strong><br>\n'
'<abbr title="A Series: values with a label on each one. Build it from a dictionary or a list."><code>pd.Series(data)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Look a value up by its label, like a dictionary."><code>s[\'AAPL\']</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Sort by value. Pass ascending=False for largest first."><code>s.sort_values()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="A Series summarises itself, exactly like an array."><code>s.mean()</code></abbr></p>\n\n'
'<p><strong>pandas: a table</strong><br>\n'
'<abbr title="Read a CSV file into a DataFrame. parse_dates turns a text date column into real dates."><code>pd.read_csv(path, parse_dates=[\'date\'])</code></abbr> &nbsp;·&nbsp; '
'<abbr title="The first n rows, 5 by default."><code>frame.head(n)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Rows and columns, as a pair of numbers."><code>frame.shape</code></abbr> &nbsp;·&nbsp; '
'<abbr title="What each column holds: dates, text, numbers."><code>frame.dtypes</code></abbr> &nbsp;·&nbsp; '
'<abbr title="The column names."><code>frame.columns</code></abbr></p>\n\n'
'<p><strong>pandas: choosing what you want</strong><br>\n'
'<abbr title="One column, as a Series."><code>frame[\'close\']</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Several columns, as a smaller table. Note the two sets of brackets."><code>frame[[\'date\', \'close\']]</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Keep only the rows where the condition is True."><code>frame[frame[\'ticker\'] == \'AAPL\']</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Take your own copy, so pandas knows you mean to change this selection and not the original."><code>.copy()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Rows by label. With a date index you can slice with dates."><code>frame.loc[\'2024\']</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Sort the rows by the values in a column."><code>frame.sort_values(\'ret\')</code></abbr></p>\n\n'
'<p><strong>pandas: new columns and groups</strong><br>\n'
'<abbr title="Create a column by assigning to a name that does not exist yet."><code>frame[\'ret\'] = ...</code></abbr> &nbsp;·&nbsp; '
'<abbr title="The percentage change from each row to the next. The first one is NaN."><code>frame[\'close\'].pct_change()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Throw away the rows with no value (NaN)."><code>s.dropna()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Split by the values in a column, do something inside each group, put the answers back together."><code>frame.groupby(\'ticker\')</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Several summaries at once, as a table."><code>g.agg([\'mean\', \'std\'])</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Reshape: one row per date, one column per ticker."><code>frame.pivot(index=\'date\', columns=\'ticker\', values=\'close\')</code></abbr></p>\n\n'
'<p><strong>Building tables out of what you have</strong><br>\n'
'<abbr title="A Series straight from a dictionary: the keys become the labels."><code>pd.Series(a_dict)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="A DataFrame from a dictionary of columns. Series are lined up on their labels for you."><code>pd.DataFrame({\'vol\': s1, \'ret\': s2})</code></abbr> &nbsp;·&nbsp; '
'<abbr title="A DataFrame from a list of dictionaries: one row per dictionary. Useful when you build rows in a loop."><code>pd.DataFrame(list_of_dicts)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="The plain numbers underneath, as a NumPy array, with the labels dropped."><code>frame.values</code></abbr></p>\n\n'
'<p><strong>Still yours from Sessions 1 and 2</strong><br>\n'
'<abbr title="Run the indented body once for every item."><code>for item in items:</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Run the block only when the condition is True. elif adds further cases, else catches the rest."><code>if / elif / else</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Define your own function. The indented body is what it does, and return hands a value back."><code>def name(x, rate=0.05):</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Hand a value back to whoever called the function."><code>return value</code></abbr> &nbsp;·&nbsp; '
'<abbr title="An empty list, ready to be filled by a loop, and the method that fills it."><code>results = []  ·  results.append(x)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="A dictionary: values stored under a name rather than a position."><code>{\'AAPL\': 0.36}</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Show r as a percentage with 2 decimals."><code>f\"{r:.2%}\"</code></abbr></p>\n\n'
'<p><strong>matplotlib</strong><br>\n'
'<abbr title="Make a figure and one axes to draw on. figsize is in inches."><code>fig, ax = plt.subplots(figsize=(9, 3))</code></abbr> &nbsp;·&nbsp; '
'<abbr title="A line: something over time."><code>ax.plot(x, y)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Horizontal bars: comparing named things."><code>ax.barh(names, values)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="A histogram: the shape of one variable."><code>ax.hist(values, bins=40)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="A scatter: one thing against another."><code>ax.scatter(x, y)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="The title. loc=\'left\' puts it on the left, which reads better."><code>ax.set_title(text, loc=\'left\')</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Label the axes. Always say what the numbers are, and in what units."><code>ax.set_ylabel(text)</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Show the legend. Needs label= on each line you drew."><code>ax.legend()</code></abbr> &nbsp;·&nbsp; '
'<abbr title="Save the figure to a file."><code>fig.savefig(\'name.png\', dpi=200)</code></abbr></p>\n\n'
"**Formulas you will reach for**\n\n"
r"| what | formula |"
"\n|:--|:--|\n"
r"| Simple return | $r_i=\dfrac{p_i-p_{i-1}}{p_{i-1}}$ |"
"\n"
r"| Volatility (daily) | $\sigma=\sqrt{\dfrac{1}{n}\sum_i (r_i-\bar{r})^2}$ |"
"\n"
r"| Annualised volatility | $\sigma_{\text{year}}=\sigma\sqrt{252}$ |"
"\n"
r"| OLS estimate | $\hat{\beta}=(X'X)^{-1}X'y$ |"
"\n"
)

md("---")

# ---------------------------------------------------------------- setup cell
md(
"## ⚙️ Setup: run this first\n\n"
"This imports the three packages and loads the price data. If you are in Google "
"Colab, either upload `prices.csv` and `aapl_2024_closes.csv`, or set "
"`REPO_RAW_URL` to the address your instructor gave you."
)

code(
'''import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# The whole universe: eleven stocks, 2015 to 2024, one row per stock per day
prices = load_csv("prices.csv", parse_dates=["date"])

# Apple's 2024 closes, as a plain NumPy array, for the array exercises
close24 = load_csv("aapl_2024_closes.csv")["close"].to_numpy()

# A single week of Apple closes, small enough to check by hand
week = np.array([183.56, 182.19, 179.87, 179.15, 183.48])

print("prices :", prices.shape, "rows x columns")
print("close24:", close24.shape[0], "trading days in 2024")
print("week   :", week)'''
)

md("---")

# ================================================================= A. Arrays
section(
"## \U0001f9ee A · Arrays\n\n"
"An array is a list that does arithmetic on every element at once. Everything "
"you learned about indexing and slicing lists still works."
)

ex("A1", "From a list to an array", 1,
   "Turn the list `[10, 20, 30, 40]` into a NumPy array called `values`.",
   "values = ...\nvalues",
   "`np.array(...)` takes a list and gives back an array.",
   "values = np.array([10, 20, 30, 40])\nvalues",
   "It prints as `array([10, 20, 30, 40])`. It looks like a list, but it behaves quite differently.")

ex("A2", "The last three closes", 1,
   "Using the `week` array from the setup cell, take the **last three** closes into "
   "`last_three`.",
   "last_three = ...\nlast_three",
   "Negative slicing from Session 1 still works: `week[-3:]`.",
   "last_three = week[-3:]\nlast_three",
   "`array([179.87, 179.15, 183.48])`. Slicing an array is exactly like slicing a list.")

ex("A3", "Everything at once", 1,
   "Every price in `week` is in dollars. Convert the whole array to cents (multiply "
   "by 100) and store it as `cents`.",
   "cents = ...\ncents",
   "Multiply the array itself. There is no loop.",
   "cents = week * 100\ncents",
   "One short line does what took a loop and an empty list last session.")

ex("A4", "Two arrays, side by side", 2,
   "Here are two arrays: an opening price and a closing price for four days. "
   "Compute the **change** on each day (close minus open) into `change`.",
   "opens = np.array([100.0, 102.0, 101.5, 104.0])\ncloses = np.array([102.0, 101.5, 104.0, 103.5])\n\nchange = ...\nchange",
   "Subtract one array from the other. NumPy lines them up position by position.",
   "opens = np.array([100.0, 102.0, 101.5, 104.0])\ncloses = np.array([102.0, 101.5, 104.0, 103.5])\n\nchange = closes - opens\nchange",
   "`array([ 2. , -0.5,  2.5, -0.5])`. Element by element, no loop in sight.")

ex("A5", "Lining up the day before", 2,
   "From `week`, build two arrays: `today` (every close from the second onwards) "
   "and `yesterday` (every close except the last).",
   "today = ...\nyesterday = ...\n\nprint(today)\nprint(yesterday)",
   ["`week[1:]` starts at the second element. `week[:-1]` stops before the last.",
    "Both arrays end up four long, and `today[i]` is the day after `yesterday[i]`."],
   "today = week[1:]\nyesterday = week[:-1]\n\nprint(today)\nprint(yesterday)",
   "Two views of the same prices, offset by one day. This is the whole trick behind "
   "computing returns without a loop.")

ex("A6", "Every daily return, in one line", 2,
   "Now use those two views to compute all four daily returns from `week`, into "
   "`returns`.",
   "returns = ...\nreturns",
   ["The formula is (today minus yesterday) divided by yesterday.",
    "Written out: `(week[1:] - week[:-1]) / week[:-1]`."],
   "returns = (week[1:] - week[:-1]) / week[:-1]\nreturns",
   f"Four returns from five prices, the first being `{WEEK_R[0]:.6f}`. Last session "
   "this was a loop, a `range`, an empty list and an `.append`.")

ex("A7", "The loop you no longer need", 2,
   "Build the same four returns from `week` the Session 2 way, with a loop and an "
   "empty list, then turn the result into an array. Compare it with `(week[1:] - "
   "week[:-1]) / week[:-1]`.",
   "loop_returns = []\nfor i in range(1, len(week)):\n    ...\n\nloop_returns = np.array(loop_returns)\nloop_returns",
   ["Inside the loop, append `(week[i] - week[i - 1]) / week[i - 1]`.",
    "`np.array(a_list)` turns the finished list into an array, which is how most "
    "arrays start life."],
   "loop_returns = []\nfor i in range(1, len(week)):\n    loop_returns.append((week[i] - week[i - 1]) / week[i - 1])\n\nloop_returns = np.array(loop_returns)\nloop_returns",
   "Identical to the one-line version. The loop is not wrong, and understanding it is "
   "why the one-liner makes sense to you. From here on, write the one-liner.")

md("---")

# =========================================================== B. Summarising
section(
"## \U0001f4ca B · Summarising an array\n\n"
"The things you wrote by hand in Session 2 are already methods on the array."
)

ex("B1", "Four numbers about a year", 1,
   "`close24` holds Apple's closing price on every trading day of 2024. Print its "
   "mean, minimum and maximum.",
   "print('mean:', ...)\nprint('min :', ...)\nprint('max :', ...)",
   "`close24.mean()`, and the same pattern for the other two.",
   "print('mean:', close24.mean())\nprint('min :', close24.min())\nprint('max :', close24.max())",
   f"Apple averaged about ${CLOSE24.mean():.2f} in 2024, between "
   f"${CLOSE24.min():.2f} and ${CLOSE24.max():.2f}.")

ex("B2", "Volatility, the short way", 2,
   "Compute Apple's 2024 **daily volatility**: the returns first, then their "
   "standard deviation.",
   "returns24 = ...\nvol = ...\nvol",
   ["The returns line is the same slicing formula as A6, on `close24`.",
    "Volatility is `returns24.std()`."],
   "returns24 = (close24[1:] - close24[:-1]) / close24[:-1]\nvol = returns24.std()\nvol",
   f"About `{R24.std():.4f}`, so roughly {R24.std():.2%} on a typical day. That is "
   "the number your Session 2 case computed with two loops and a function.")

ex("B3", "In the units finance uses", 2,
   "Turn that daily volatility into an **annualised** one by multiplying by the "
   "square root of 252.",
   "returns24 = (close24[1:] - close24[:-1]) / close24[:-1]\n\nannual_vol = ...\nannual_vol",
   "`np.sqrt(252)` gives the square root you need.",
   "returns24 = (close24[1:] - close24[:-1]) / close24[:-1]\n\nannual_vol = returns24.std() * np.sqrt(252)\nannual_vol",
   f"About `{R24.std()*np.sqrt(252):.4f}`, so roughly {R24.std()*np.sqrt(252):.0%} a "
   "year. This is the figure you would see quoted.")

ex("B4", "How far it travelled", 2,
   "Compute Apple's **total return** over 2024: from the first close of the year to "
   "the last.",
   "total_return = ...\ntotal_return",
   ["The first close is `close24[0]` and the last is `close24[-1]`.",
    "It is the same simple-return formula, applied to those two prices."],
   "total_return = (close24[-1] - close24[0]) / close24[0]\ntotal_return",
   f"About `{(CLOSE24[-1]-CLOSE24[0])/CLOSE24[0]:.4f}`, a gain of "
   f"{(CLOSE24[-1]-CLOSE24[0])/CLOSE24[0]:.1%} over the year.")

ex("B5", "Putting returns on a common scale", 3,
   "**Standardise** Apple's 2024 returns: subtract the mean, then divide by the "
   "standard deviation. Check that the result has mean 0 and standard deviation 1.",
   "returns24 = (close24[1:] - close24[:-1]) / close24[:-1]\n\nstandardised = ...\n\nprint('mean:', ...)\nprint('std :', ...)",
   ["The formula is $(r - \\bar{r}) / \\sigma$, and it works on the whole array at once.",
    "`(returns24 - returns24.mean()) / returns24.std()`."],
   "returns24 = (close24[1:] - close24[:-1]) / close24[:-1]\n\nstandardised = (returns24 - returns24.mean()) / returns24.std()\n\nprint('mean:', standardised.mean())\nprint('std :', standardised.std())",
   "The mean comes out as something like `2e-17`, which is zero with a rounding "
   "residue, and the standard deviation is exactly 1. A standardised value says how "
   "many standard deviations from average a day was, so the largest here is about "
   "5.\n\nThis is worth remembering: nearly every model you meet later in the "
   "course wants its inputs standardised, because otherwise a variable measured in "
   "millions drowns out one measured in percent.")

md("---")

# ================================================================= C. Masks
section(
"## ⚖️ C · Asking a question of every element\n\n"
"Comparing an array to a number gives you an answer for each element, which you "
"can then count or filter with. This replaces the `if` inside a loop."
)

ex("C1", "True or false, one per day", 1,
   "Using `week_returns` below, produce the array of `True`/`False` saying whether "
   "each day was up.",
   "week_returns = (week[1:] - week[:-1]) / week[:-1]\n\nwas_up = ...\nwas_up",
   "Compare the array to zero: `week_returns > 0`.",
   "week_returns = (week[1:] - week[:-1]) / week[:-1]\n\nwas_up = week_returns > 0\nwas_up",
   "`array([False, False, False,  True])`. One answer per element, and no loop.")

ex("C2", "Counting the up days", 2,
   "Count how many of Apple's 2024 days were **up days**, using `returns24`.",
   "returns24 = (close24[1:] - close24[:-1]) / close24[:-1]\n\nup_days = ...\nup_days",
   ["Make the True/False array first, then add it up.",
    "`(returns24 > 0).sum()` counts the `True`s, because `True` counts as one."],
   "returns24 = (close24[1:] - close24[:-1]) / close24[:-1]\n\nup_days = (returns24 > 0).sum()\nup_days",
   f"`{int((R24>0).sum())}` up days out of {len(R24)}. A little over half, which is "
   "what you would expect from a year the stock rose in.")

ex("C3", "Keeping only some of them", 2,
   "Keep only the **negative** returns from `returns24`, into `down_moves`.",
   "returns24 = (close24[1:] - close24[:-1]) / close24[:-1]\n\ndown_moves = ...\ndown_moves",
   "Put the condition inside the square brackets: `returns24[returns24 < 0]`.",
   "returns24 = (close24[1:] - close24[:-1]) / close24[:-1]\n\ndown_moves = returns24[returns24 < 0]\ndown_moves",
   f"{int((R24<0).sum())} numbers, all of them negative. You now have the down days "
   "on their own, ready to summarise.")

ex("C4", "How bad is a bad day?", 3,
   "Using only the down days, compute their **average** size. Then compare it to "
   "the average up day.",
   "returns24 = (close24[1:] - close24[:-1]) / close24[:-1]\n\navg_down = ...\navg_up = ...\n\nprint('average down day:', avg_down)\nprint('average up day  :', avg_up)",
   ["Filter first, then take the mean of what is left.",
    "`returns24[returns24 < 0].mean()`, and the mirror image for up days."],
   "returns24 = (close24[1:] - close24[:-1]) / close24[:-1]\n\navg_down = returns24[returns24 < 0].mean()\navg_up = returns24[returns24 > 0].mean()\n\nprint('average down day:', avg_down)\nprint('average up day  :', avg_up)",
   f"About `{R24[R24<0].mean():.4f}` against `{R24[R24>0].mean():.4f}`. The typical "
   "down day is a little larger than the typical up day, which is normal for stocks: "
   "there are simply more up days.")

ex("C5", "A share, not a count", 2,
   "What **fraction** of Apple's 2024 days were up days? Do it without dividing "
   "anything yourself.",
   "returns24 = (close24[1:] - close24[:-1]) / close24[:-1]\n\nup_share = ...\nup_share",
   ["The average of an array of `True` and `False` is the share that are `True`, "
    "because `True` counts as 1 and `False` as 0.",
    "`(returns24 > 0).mean()`."],
   "returns24 = (close24[1:] - close24[:-1]) / close24[:-1]\n\nup_share = (returns24 > 0).mean()\nup_share",
   "About `0.566`, so 57% of days were up. `.sum()` on a condition counts, and "
   "`.mean()` on the same condition gives the share. You will use both constantly.")

md("---")

# ============================================================== D. Matrices
section(
"## \U0001f9f1 D · Matrices\n\n"
"A matrix is a two-dimensional array: rows and columns. Everything a dataset or "
"a model is made of."
)

ex("D1", "Building one", 1,
   "Build this 3 by 2 matrix as `X`, and print its shape:\n\n"
   "```text\n1.0   0.5\n1.0   1.5\n1.0   2.5\n```",
   "X = ...\nprint(X)",
   "A list of rows, each row a list: `np.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]])`.",
   "X = np.array([[1.0, 0.5],\n              [1.0, 1.5],\n              [1.0, 2.5]])\nprint(X)\nprint(X.shape)",
   "`X.shape` is `(3, 2)`: three rows, two columns. Rows always come first.")

ex("D2", "Reaching into it", 2,
   "From the `X` below, pull out the element in row 1 and column 0, the whole "
   "second row, and the whole second column.",
   "X = np.array([[1.0, 0.5],\n              [1.0, 1.5],\n              [1.0, 2.5]])\n\none = ...\nrow = ...\ncolumn = ...\n\nprint(one)\nprint(row)\nprint(column)",
   ["`X[row, column]`, and `:` means all of them.",
    "The second row is `X[1, :]`. The second column is `X[:, 1]`."],
   "X = np.array([[1.0, 0.5],\n              [1.0, 1.5],\n              [1.0, 2.5]])\n\none = X[1, 0]\nrow = X[1, :]\ncolumn = X[:, 1]\n\nprint(one)\nprint(row)\nprint(column)",
   "`1.0`, then `[1.  1.5]`, then `[0.5 1.5 2.5]`. Counting starts at zero in both "
   "directions.")

ex("D3", "A portfolio return", 2,
   "You hold 50% Apple, 30% Coca-Cola and 20% Nvidia. On one day they returned "
   "1.2%, -0.4% and 3.1%. Compute the portfolio's return for that day.",
   "weights = np.array([0.5, 0.3, 0.2])\nreturns = np.array([0.012, -0.004, 0.031])\n\nportfolio_return = ...\nportfolio_return",
   ["Multiply pairwise and add up. That is the dot product.",
    "`weights @ returns` does it in one step."],
   "weights = np.array([0.5, 0.3, 0.2])\nreturns = np.array([0.012, -0.004, 0.031])\n\nportfolio_return = weights @ returns\nportfolio_return",
   "`0.011` exactly: `0.5*0.012 + 0.3*(-0.004) + 0.2*0.031`. A 1.1% day for the "
   "portfolio.")

ex("D4", "Fitted values for a whole sample", 2,
   "With `X` (3 rows, 2 columns) and coefficients `b = [0.001, 1.2]`, compute the "
   "fitted value for every row at once.",
   "X = np.array([[1.0, 0.012],\n              [1.0, -0.004],\n              [1.0, 0.008]])\nb = np.array([0.001, 1.2])\n\nfitted = ...\nfitted",
   "`X @ b`. Three rows in, three numbers out.",
   "X = np.array([[1.0, 0.012],\n              [1.0, -0.004],\n              [1.0, 0.008]])\nb = np.array([0.001, 1.2])\n\nfitted = X @ b\nfitted",
   "`array([ 0.0154, -0.0038,  0.0106])`. This is a regression's prediction step for "
   "the whole sample, written once.")

md("---")

# ======================================================== E. Linear algebra
section(
"## ➗ E · Linear algebra\n\n"
"The same maths as your econometrics course, one line per step."
)

ex("E1", "Undoing a matrix", 2,
   "Compute the inverse of `A`, then multiply it by `A` and look at what you get.",
   "A = np.array([[4.0, 2.0],\n              [1.0, 3.0]])\n\nA_inv = ...\ncheck = ...\n\nprint(A_inv)\nprint(check)",
   ["`np.linalg.inv(A)` gives the inverse.",
    "Then multiply them with `@`, in either order."],
   "A = np.array([[4.0, 2.0],\n              [1.0, 3.0]])\n\nA_inv = np.linalg.inv(A)\ncheck = A_inv @ A\n\nprint(A_inv)\nprint(check)",
   "`check` is the identity: ones on the diagonal, zeros elsewhere. Some of the "
   "zeros print in scientific notation, as something like `-5.5e-17`. That is a "
   "rounding residue, not a real number. Read it as zero.")

ex("E2", "Solving a system", 2,
   "Solve $Ax = b$ for $x$, with the `A` and `b` below. Use `np.linalg.solve`.",
   "A = np.array([[4.0, 2.0],\n              [1.0, 3.0]])\nb = np.array([10.0, 10.0])\n\nx = ...\nx",
   "`np.linalg.solve(A, b)` takes the matrix first and the right-hand side second.",
   "A = np.array([[4.0, 2.0],\n              [1.0, 3.0]])\nb = np.array([10.0, 10.0])\n\nx = np.linalg.solve(A, b)\nx",
   "`array([1., 3.])`. Check it if you like: `A @ x` gives back `[10., 10.]`.")

ex("E3", "OLS by hand, once", 3,
   "Estimate a regression of `y` on a constant and `x`, using the formula "
   "$\\hat{\\beta}=(X'X)^{-1}X'y$. Build the `X` matrix first: a column of ones, "
   "then `x`.",
   "x = np.array([0.5, 1.5, 2.5, 3.5])\ny = np.array([1.0, 2.0, 2.6, 3.8])\n\nX = ...\nbeta = ...\nbeta",
   ["`np.column_stack([np.ones(len(x)), x])` glues a column of ones onto `x`.",
    "Then `beta = np.linalg.solve(X.T @ X, X.T @ y)`."],
   "x = np.array([0.5, 1.5, 2.5, 3.5])\ny = np.array([1.0, 2.0, 2.6, 3.8])\n\nX = np.column_stack([np.ones(len(x)), x])\nbeta = np.linalg.solve(X.T @ X, X.T @ y)\nbeta",
   "`array([0.55, 0.9 ])`: an intercept of 0.55 and a slope of 0.90. The same answer "
   "your econometrics software would give.")

ex("E4", "What the model missed", 3,
   "Using the `beta` you just estimated, compute the **fitted values** and the "
   "**residuals** (actual minus fitted), and check that the residuals average out to "
   "about zero.",
   "x = np.array([0.5, 1.5, 2.5, 3.5])\ny = np.array([1.0, 2.0, 2.6, 3.8])\nX = np.column_stack([np.ones(len(x)), x])\nbeta = np.linalg.solve(X.T @ X, X.T @ y)\n\nfitted = ...\nresiduals = ...\n\nprint('fitted   :', fitted)\nprint('residuals:', residuals)",
   ["The fitted values are `X @ beta`, one per row.",
    "The residuals are what is left over: `y - fitted`."],
   "x = np.array([0.5, 1.5, 2.5, 3.5])\ny = np.array([1.0, 2.0, 2.6, 3.8])\nX = np.column_stack([np.ones(len(x)), x])\nbeta = np.linalg.solve(X.T @ X, X.T @ y)\n\nfitted = X @ beta\nresiduals = y - fitted\n\nprint('fitted   :', fitted)\nprint('residuals:', residuals)\nprint('mean residual:', residuals.mean())",
   "The residuals are `[0.0, 0.1, -0.2, 0.1]` and their mean is zero to rounding. "
   "That is not luck: including the column of ones is exactly what forces it. "
   "Residuals are how every model in this course will be judged.")

md("---")

# ================================================================ F. Series
section(
"## \U0001f3f7️ F · Series\n\n"
"A Series is one column of values with a label on each one. Think of it as a "
"dictionary and an array at the same time."
)

ex("F1", "From a dictionary", 1,
   "Turn this dictionary of annualised volatilities into a Series called `vol`:\n\n"
   "```python\n{'AAPL': 0.227, 'KO': 0.128, 'NVDA': 0.525, 'JPM': 0.235}\n```",
   "vol = ...\nvol",
   "`pd.Series(...)` accepts a dictionary directly.",
   "vol = pd.Series({'AAPL': 0.227, 'KO': 0.128, 'NVDA': 0.525, 'JPM': 0.235})\nvol",
   "The tickers become the labels, and the numbers become the values.")

ex("F2", "Looking one up", 1,
   "From the `vol` Series below, read out Nvidia's number.",
   "vol = pd.Series({'AAPL': 0.227, 'KO': 0.128, 'NVDA': 0.525, 'JPM': 0.235})\n\nnvda_vol = ...\nnvda_vol",
   "Square brackets and the label, just like a dictionary.",
   "vol = pd.Series({'AAPL': 0.227, 'KO': 0.128, 'NVDA': 0.525, 'JPM': 0.235})\n\nnvda_vol = vol['NVDA']\nnvda_vol",
   "`0.525`. Reading by name is one of the two things a Series does.")

ex("F3", "Maths on the whole thing", 2,
   "Those are annual figures. Convert the whole Series back to **daily** volatility "
   "by dividing by the square root of 252.",
   "vol = pd.Series({'AAPL': 0.227, 'KO': 0.128, 'NVDA': 0.525, 'JPM': 0.235})\n\ndaily_vol = ...\ndaily_vol",
   "Divide the Series itself, exactly as you would an array. The labels come along.",
   "vol = pd.Series({'AAPL': 0.227, 'KO': 0.128, 'NVDA': 0.525, 'JPM': 0.235})\n\ndaily_vol = vol / np.sqrt(252)\ndaily_vol",
   "Every value is divided and every label is kept. That combination is the whole "
   "reason pandas exists.")

ex("F4", "Riskiest first", 2,
   "Sort the `vol` Series so the **largest** volatility comes first.",
   "vol = pd.Series({'AAPL': 0.227, 'KO': 0.128, 'NVDA': 0.525, 'JPM': 0.235})\n\nranked = ...\nranked",
   "`.sort_values()` sorts smallest first. There is an argument to reverse it.",
   "vol = pd.Series({'AAPL': 0.227, 'KO': 0.128, 'NVDA': 0.525, 'JPM': 0.235})\n\nranked = vol.sort_values(ascending=False)\nranked",
   "NVDA, JPM, AAPL, KO. `ascending=False` is the argument, and it is exactly the "
   "kind of detail you look up rather than memorise.")

md("---")

# ============================================================ G. DataFrames
section(
"## \U0001f4c4 G · A real table\n\n"
"`prices` is the whole universe: eleven stocks, every trading day from 2015 to "
"2024, one row per stock per day."
)

ex("G1", "First look", 1,
   "Print the shape of `prices`, and show its first five rows.",
   "print(...)\n\nprices.head()",
   "`prices.shape` gives the rows and columns. `.head()` shows the top five rows.",
   "print(prices.shape)\n\nprices.head()",
   f"`{PX.shape}`: {PX.shape[0]:,} rows and {PX.shape[1]} columns. Always look before "
   "you compute.")

ex("G2", "What is in each column", 1,
   "Show what type of data each column of `prices` holds.",
   "kinds = ...\nkinds",
   "`prices.dtypes`.",
   "kinds = prices.dtypes\nkinds",
   "`date` is a real date, `ticker` is text (`object`), `close` is a float and "
   "`volume` is a whole number. A price column that arrived as text is a common and "
   "silent bug, and this is how you catch it.")

ex("G3", "One column", 1,
   "Take the `close` column of `prices` on its own, into `closes`.",
   "closes = ...\ncloses",
   "A column name in square brackets: `prices['close']`.",
   "closes = prices['close']\ncloses.head()",
   "One name gives back a Series: that column, with the row labels kept.")

ex("G4", "Two columns", 2,
   "Take just the `date` and `close` columns, into `small`.",
   "small = ...\nsmall",
   ["A **list** of names inside the brackets, so there are two sets of brackets.",
    "`prices[['date', 'close']]`."],
   "small = prices[['date', 'close']]\nsmall.head()",
   "A list of names gives back a smaller DataFrame, not a Series. That is why the "
   "brackets are doubled.")

md("---")

# ================================================== H. Filtering and columns
section(
"## \U0001f50e H · Choosing rows, adding columns\n\n"
"Filtering a table is the same idea as a mask on an array, done to 27,676 rows "
"at once."
)

ex("H1", "One stock", 2,
   "Keep only the Coca-Cola rows of `prices`, into `ko`.",
   "ko = ...\nko",
   ["The condition is `prices['ticker'] == 'KO'`.",
    "Put that condition inside `prices[...]`, and use `.shape` for the row count."],
   "ko = prices[prices['ticker'] == 'KO']\nprint(ko.shape)\n\nko.head()",
   f"`({(PX.ticker=='KO').sum()}, 4)`: one row per trading day over ten years, for "
   "one stock.")

ex("H2", "One stock, one condition", 2,
   "From the Apple rows, keep only the days where the close was above 200 dollars, "
   "into `expensive`. How many such days were there?",
   "aapl = prices[prices['ticker'] == 'AAPL']\n\nexpensive = ...\nexpensive",
   "Filter the already-filtered table: `aapl[aapl['close'] > 200]`.",
   "aapl = prices[prices['ticker'] == 'AAPL']\n\nexpensive = aapl[aapl['close'] > 200]\nprint(expensive.shape)\nexpensive.head()",
   f"`{int(((PX.ticker=='AAPL') & (PX.close>200)).sum())}` days out of 2,516. Apple "
   "only crossed 200 dollars late in the decade.")

ex("H3", "A returns column", 2,
   "Take the Microsoft rows, then add a `ret` column holding the daily return.",
   "msft = prices[prices['ticker'] == 'MSFT'].copy()\n\nmsft['ret'] = ...\nmsft[['date', 'close', 'ret']].head()",
   "`msft['close'].pct_change()` is the daily return of that column.",
   "msft = prices[prices['ticker'] == 'MSFT'].copy()\n\nmsft['ret'] = msft['close'].pct_change()\nmsft[['date', 'close', 'ret']].head()",
   "The first row is `NaN`, because the first day has no day before it. The `.copy()` "
   "is what stops pandas warning you that it cannot tell whether you also meant to "
   "change `prices`.")

ex("H4", "Volatility from a table", 2,
   "Using that `ret` column, compute Microsoft's daily volatility over the whole "
   "period.",
   "msft = prices[prices['ticker'] == 'MSFT'].copy()\nmsft['ret'] = msft['close'].pct_change()\n\nvol = ...\nvol",
   "`.std()` on the column. It skips the `NaN` for you.",
   "msft = prices[prices['ticker'] == 'MSFT'].copy()\nmsft['ret'] = msft['close'].pct_change()\n\nvol = msft['ret'].std()\nvol",
   f"About `{VOL_FULL['MSFT']:.4f}`, so {VOL_FULL['MSFT']:.2%} on a typical day. "
   "You did not have to drop the `NaN` yourself: most summary methods ignore it.")

ex("H5", "The three worst days", 3,
   "Sort Apple's rows by return, smallest first, into `worst`. The three worst days "
   "and their dates should end up at the top.",
   "aapl = prices[prices['ticker'] == 'AAPL'].copy()\naapl['ret'] = aapl['close'].pct_change()\n\nworst = ...\nworst",
   ["`.sort_values('ret')` sorts the whole table by that column.",
    "Smallest first is the default, so you do not need `ascending`."],
   "aapl = prices[prices['ticker'] == 'AAPL'].copy()\naapl['ret'] = aapl['close'].pct_change()\n\nworst = aapl.sort_values('ret')\nworst[['date', 'close', 'ret']].head(3)",
   "16 March 2020 at about -12.9%, then 3 January 2019 and 12 March 2020. The first "
   "and third are the covid crash; the middle one is the day after Apple cut its "
   "revenue guidance.")

ex("H6", "A slice of time", 2,
   "Keep only the rows of `prices` from 2023 onwards, into `recent`, and check how "
   "many rows that leaves.",
   "recent = ...\nrecent",
   ["`prices['date'] >= '2023-01-01'` is a condition on the date column, and it "
    "works because the quick setup parsed those dates properly.",
    "Put it in brackets: `prices[prices['date'] >= '2023-01-01']`."],
   "recent = prices[prices['date'] >= '2023-01-01']\nprint(recent.shape)\nrecent.head()",
   "5,522 rows: 502 trading days for each of the eleven. Cutting a table by date is "
   "how you will separate the period a model learns from, from the period you test "
   "it on.")

md("---")

# ================================================== I. groupby and reshaping
section(
"## \U0001f9ee I · All eleven at once\n\n"
"`groupby` splits the table by a column, does something inside each group, and "
"puts the answers back together. It is the loop from Session 2, written once."
)

ex("I1", "How many rows each", 2,
   "Count how many rows `prices` holds for each ticker.",
   "counts = ...\ncounts",
   "`prices.groupby('ticker').size()` counts the rows in each group.",
   "counts = prices.groupby('ticker').size()\ncounts",
   "2,516 for every one of the eleven, which is a good sign: no stock is missing days.")

ex("I2", "Average price per stock", 2,
   "Compute the average closing price of each stock over the whole period.",
   "avg_close = ...\navg_close",
   "Group by ticker, pick the `close` column, then `.mean()`.",
   "avg_close = prices.groupby('ticker')['close'].mean()\navg_close",
   "Eleven answers, each under its ticker. Note that these are not comparable across "
   "stocks: a share price depends on how many shares exist, not on how good the "
   "company is.")

ex("I3", "Volatility for every stock", 3,
   "Add a `ret` column holding each stock's own daily return, then compute the "
   "volatility of every stock and rank them riskiest first.",
   "prices['ret'] = ...\n\nvol = ...\nvol",
   ["`prices.groupby('ticker')['close'].pct_change()` computes the return **within** "
    "each stock, so no return is ever taken across two different companies.",
    "Then group again: `prices.groupby('ticker')['ret'].std()`, and sort it with "
    "`ascending=False`."],
   "prices['ret'] = prices.groupby('ticker')['close'].pct_change()\n\nvol = prices.groupby('ticker')['ret'].std().sort_values(ascending=False)\nvol",
   f"Nvidia is far out in front at about {VOL_FULL['NVDA']:.2%} a day, and the S&P 500 "
   f"ETF is calmest at about {VOL_FULL['SPY']:.2%}. Grouping before `pct_change` is "
   "the important part: without it, the first return of each stock would be computed "
   "from the last price of the previous one.")

ex("I4", "A small risk report", 3,
   "Produce a table with the mean, standard deviation, minimum and maximum daily "
   "return for every stock.",
   "prices['ret'] = prices.groupby('ticker')['close'].pct_change()\n\nreport = ...\nreport",
   "`.agg(['mean', 'std', 'min', 'max'])` after the groupby gives all four at once.",
   "prices['ret'] = prices.groupby('ticker')['close'].pct_change()\n\nreport = prices.groupby('ticker')['ret'].agg(['mean', 'std', 'min', 'max']).round(4)\nreport",
   "One row per stock, four columns. That is a risk report, and it took one line.")

ex("I5", "One column per stock", 2,
   "Reshape `prices` so that there is one row per date and one column per ticker, "
   "holding the closing price. Call it `wide`.",
   "wide = ...\nwide",
   "`prices.pivot(index='date', columns='ticker', values='close')`.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nwide.tail(3)",
   "The tall shape is better for storing and grouping; this wide one is better for "
   "comparing stocks and for plotting.")

md("---")

# =========================================================== J. matplotlib
section(
"## \U0001f4c8 J · Figures\n\n"
"Every plot is the same three steps: make the axes, draw on them, then say what "
"the reader is looking at."
)

ex("J1", "A labelled line", 1,
   "Draw Apple's 2024 closing price. Give the plot a title and a y-axis label.",
   "fig, ax = plt.subplots(figsize=(9, 3))\n...\nplt.show()",
   ["`ax.plot(close24)` draws the line. Then `ax.set_title(...)` and "
    "`ax.set_ylabel(...)`.",
    "Put the title on the left with `loc='left'`, which reads better on a slide or "
    "in a report."],
   "fig, ax = plt.subplots(figsize=(9, 3))\nax.plot(close24)\nax.set_title('Apple (AAPL) closing price, 2024', loc='left')\nax.set_ylabel('price (USD)')\nplt.show()",
   "Without the title and the label this is just a wiggly line. The x-axis here is "
   "the day number, because `close24` is a plain array with no dates attached.")

ex("J2", "Two lines and a legend", 2,
   "Using `wide` below, draw Apple and Microsoft's 2024 prices on the same axes, "
   "with a legend.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\np24 = wide.loc['2024']\n\nfig, ax = plt.subplots(figsize=(9, 3))\n...\nplt.show()",
   ["Two `ax.plot(...)` calls, each with `label='AAPL'` or `label='MSFT'`.",
    "Then one `ax.legend()` at the end."],
   "wide = prices.pivot(index='date', columns='ticker', values='close')\np24 = wide.loc['2024']\n\nfig, ax = plt.subplots(figsize=(9, 3))\nax.plot(p24.index, p24['AAPL'], label='AAPL')\nax.plot(p24.index, p24['MSFT'], label='MSFT')\nax.set_ylabel('price (USD)')\nax.legend()\nplt.show()",
   "As soon as there is more than one line, the reader has to be told which is "
   "which. `label=` on each, then one `ax.legend()`.")

ex("J3", "Ranking, as bars", 2,
   "Draw the volatility ranking below as horizontal bars, with a title.",
   "prices['ret'] = prices.groupby('ticker')['close'].pct_change()\nvol = prices.groupby('ticker')['ret'].std().sort_values()\n\nfig, ax = plt.subplots(figsize=(9, 3.5))\n...\nplt.show()",
   "`ax.barh(vol.index, vol.values)` puts the names down the side and the bars "
   "across.",
   "prices['ret'] = prices.groupby('ticker')['close'].pct_change()\nvol = prices.groupby('ticker')['ret'].std().sort_values()\n\nfig, ax = plt.subplots(figsize=(9, 3.5))\nax.barh(vol.index, vol.values)\nax.set_title('Daily volatility, 2015 to 2024', loc='left')\nplt.show()",
   "Sorting before you plot is what makes a bar chart readable. `barh` rather than "
   "`bar` because the labels are names.")

ex("J4", "The shape of a year", 2,
   "Draw a histogram of Apple's daily returns using `returns24`, with 40 bins and a "
   "title.",
   "returns24 = (close24[1:] - close24[:-1]) / close24[:-1]\n\nfig, ax = plt.subplots(figsize=(9, 3))\n...\nplt.show()",
   "`ax.hist(returns24, bins=40)`.",
   "returns24 = (close24[1:] - close24[:-1]) / close24[:-1]\n\nfig, ax = plt.subplots(figsize=(9, 3))\nax.hist(returns24, bins=40)\nax.set_title('Apple daily returns, 2024', loc='left')\nax.set_xlabel('daily return')\nplt.show()",
   "Most days are small moves near zero, with a few large ones out at the sides. "
   "That shape is what volatility is measuring the width of.")

ex("J5", "One thing against another", 2,
   "Draw a scatter of Apple's daily returns against the market's (SPY), for 2024. "
   "Label both axes.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nr24 = wide.loc['2024'].pct_change().dropna()\n\nfig, ax = plt.subplots(figsize=(5.5, 4))\n...\nplt.show()",
   ["`ax.scatter(r24['SPY'], r24['AAPL'])`, with the market on the x-axis because "
    "that is the thing doing the explaining.",
    "Then `ax.set_xlabel(...)` and `ax.set_ylabel(...)`."],
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nr24 = wide.loc['2024'].pct_change().dropna()\n\nfig, ax = plt.subplots(figsize=(5.5, 4))\nax.scatter(r24['SPY'], r24['AAPL'], s=10, alpha=0.6)\nax.set_xlabel('market return (SPY)')\nax.set_ylabel('Apple return')\nax.set_title('Apple against the market, 2024', loc='left')\nplt.show()",
   "An upward-sloping cloud. The slope of a line through it is Apple's **beta**, "
   "which you computed with `np.linalg.solve` in the lecture.")

md("---")

# ======================================================== K. Looking it up
section(
"## \U0001f50d K · Looking it up\n\n"
"Everything so far came off the toolkit card. Real work does not. These three "
"ask for functions this course never showed you, which you find the same way a "
"professional does: hover the name in VS Code, or call `help()` on it.\n\n"
"> Both work with no internet, because the text ships inside the package. That "
"matters at the exam."
)

ex("K1", "How many rows does head show?", 2,
   "`head()` showed five rows every time. Find out how to ask for a different "
   "number, then show the first **three** rows of `prices`.",
   "help(pd.DataFrame.head)\n\nfirst_three = ...\nfirst_three",
   "Read the first line that `help` prints. There is one argument, and it has a "
   "default.",
   "help(pd.DataFrame.head)     # head(self, n: 'int' = 5)\n\nfirst_three = prices.head(3)\nfirst_three",
   "The argument is `n`, and its default is 5. That is why you got five rows without "
   "asking. `prices.head(3)` or `prices.head(n=3)` both work.")

ex("K2", "A function nobody taught you", 3,
   "pandas has a method that gives you count, mean, standard deviation and the "
   "quartiles of a column in one go. It is called `describe`. Use `help` to see what "
   "it does, then run it on Apple's closing prices.",
   "help(pd.Series.describe)\n\nsummary = ...\nsummary",
   ["`prices[prices['ticker'] == 'AAPL']['close']` gets the column you want.",
    "Then just call `.describe()` on it."],
   "help(pd.Series.describe)\n\naapl_close = prices[prices['ticker'] == 'AAPL']['close']\nsummary = aapl_close.describe()\nsummary",
   "Eight numbers, including the median (shown as `50%`). Useful for a first look at "
   "any column, and you now know how to find out what any of the others do.")

ex("K3", "Finding the largest values", 3,
   "You want the three most volatile stocks, largest first. You could sort and take "
   "the top three, but pandas has a method that does exactly this in one call. Its "
   "name begins with `nl`. Find it with `dir`, read it with `help`, then use it.",
   "prices['ret'] = prices.groupby('ticker')['close'].pct_change()\nvol = prices.groupby('ticker')['ret'].std()\n\nprint([name for name in dir(vol) if name.startswith('nl')])\n\ntop3 = ...\ntop3",
   ["`dir(vol)` lists everything a Series can do. The filter narrows it to the "
    "name you want.",
    "It is `nlargest`, and it takes the number you want: `vol.nlargest(3)`."],
   "prices['ret'] = prices.groupby('ticker')['close'].pct_change()\nvol = prices.groupby('ticker')['ret'].std()\n\nprint([name for name in dir(vol) if name.startswith('nl')])\n\ntop3 = vol.nlargest(3)\ntop3",
   "NVDA, AAPL and DIS. This is the whole skill: you knew what you wanted, you did "
   "not know the name, and you found it without leaving the notebook.")

ex("K4", "Yesterday, next to today", 3,
   "You want each day's return sitting next to the **previous** day's, in the same "
   "row. pandas has a method that slides a column down by one. Find it with `dir`, "
   "then build a small table with both columns.",
   "aapl = prices[prices['ticker'] == 'AAPL'].copy()\naapl['ret'] = aapl['close'].pct_change()\n\nprint([name for name in dir(aapl['ret']) if 'shif' in name])\n\naapl['ret_yesterday'] = ...\naapl[['date', 'ret', 'ret_yesterday']].head()",
   ["The method is `shift`. `help(pd.Series.shift)` shows it takes how many places "
    "to move by.",
    "`aapl['ret'].shift(1)` puts each value one row further down, so row *i* holds "
    "the value from row *i - 1*."],
   "aapl = prices[prices['ticker'] == 'AAPL'].copy()\naapl['ret'] = aapl['close'].pct_change()\n\naapl['ret_yesterday'] = aapl['ret'].shift(1)\naapl[['date', 'ret', 'ret_yesterday']].head()",
   "The first two rows are `NaN`, because there is no yesterday for the first day and "
   "no return at all for the day before that.\n\nThis is the single most important "
   "move in time-series machine learning. To predict today from the past you must "
   "line the past up next to today, and `shift` is how. Do it the wrong way and you "
   "have used tomorrow's information to predict today, which flatters your model and "
   "means nothing.")

ex("K5", "Volatility that changes over time", 3,
   "A single volatility for a whole year hides the calm months and the wild ones. "
   "pandas can compute a statistic over a sliding window of rows. The method is "
   "called `rolling`. Use it to get Apple's volatility over the last 20 trading days, "
   "at every point in 2024.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\n\nrolling_vol = ...\nrolling_vol",
   ["`help(pd.Series.rolling)`. You call `.rolling(20)` and then the statistic you "
    "want, exactly as you would on the whole column.",
    "`wide['AAPL'].pct_change().rolling(20).std() * np.sqrt(252)` gives an "
    "annualised figure for every day, using the 20 days up to it."],
   "wide = prices.pivot(index='date', columns='ticker', values='close')\n\nrolling_vol = wide['AAPL'].pct_change().rolling(20).std() * np.sqrt(252)\nrolling_vol.loc['2024'].tail()",
   "The first 19 values are `NaN`, because a 20-day window needs 20 days. Across 2024 "
   "the number runs from about 10% to about 35%, so the single 22% figure you "
   "computed earlier is an average of very different months.\n\n`rolling` is where "
   "most financial features come from: today's volatility, today's average, today's "
   "momentum. Session 4 calls these **features**.")

md("---")

# ============================================ L. old tools, new data
section(
"## \U0001f501 L · The tools you already had\n\n"
"Packages did not replace loops, functions, `if` and dictionaries. They replaced "
"the *tedious* uses of them. These exercises put the two sessions together, "
"because that combination is what real analysis code looks like."
)

ex("L1", "A function that takes a Series", 2,
   "Write a function `annualise(daily_vol)` that turns a daily volatility into an "
   "annual one, and use it on the whole Series below in one call.",
   "vol = pd.Series({'AAPL': 0.0141, 'KO': 0.0080, 'NVDA': 0.0331})\n\ndef annualise(daily_vol):\n    ...\n\nannualise(vol)",
   ["The body is one `return` line: multiply by `np.sqrt(252)`.",
    "Because the multiplication works on a whole Series, the same function works "
    "for one number and for eleven."],
   "vol = pd.Series({'AAPL': 0.0141, 'KO': 0.0080, 'NVDA': 0.0331})\n\ndef annualise(daily_vol):\n    \"\"\"Turn a daily volatility into an annual one.\"\"\"\n    return daily_vol * np.sqrt(252)\n\nannualise(vol)",
   "A function you wrote in Session 2 style, now doing eleven numbers at a time "
   "because pandas passes the whole Series through the arithmetic. Give it a "
   "docstring and your editor will show it when you hover the name.")

ex("L2", "A function with a default", 2,
   "Write `volatility(prices_series, periods=252)` that takes a Series of prices and "
   "returns its annualised volatility. Call it once with the default and once for a "
   "monthly figure (`periods=21`).",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\n\ndef volatility(prices_series, periods=252):\n    ...\n\nprint('annual :', volatility(wide['KO']))\nprint('monthly:', volatility(wide['KO'], periods=21))",
   ["Inside: `.pct_change()`, then `.std()`, then multiply by `np.sqrt(periods)`.",
    "`return prices_series.pct_change().std() * np.sqrt(periods)`."],
   "wide = prices.pivot(index='date', columns='ticker', values='close')\n\ndef volatility(prices_series, periods=252):\n    \"\"\"Annualised volatility of a price series.\"\"\"\n    return prices_series.pct_change().std() * np.sqrt(periods)\n\nprint('annual :', volatility(wide['KO']))\nprint('monthly:', volatility(wide['KO'], periods=21))",
   "About `0.178` a year and `0.051` a month. The default argument from Session 2 "
   "earns its keep here: one function, two questions, no copy of the formula.")

ex("L3", "A loop over stocks, a dictionary of answers", 2,
   "Using the `volatility` function below, loop over three tickers and collect their "
   "volatilities in a **dictionary** keyed by ticker.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\n\ndef volatility(prices_series, periods=252):\n    return prices_series.pct_change().std() * np.sqrt(periods)\n\ntickers = ['AAPL', 'KO', 'NVDA']\nvols = {}\nfor ticker in tickers:\n    ...\n\nvols",
   ["`wide[ticker]` is that stock's price column.",
    "Inside the loop: `vols[ticker] = volatility(wide[ticker])`."],
   "wide = prices.pivot(index='date', columns='ticker', values='close')\n\ndef volatility(prices_series, periods=252):\n    return prices_series.pct_change().std() * np.sqrt(periods)\n\ntickers = ['AAPL', 'KO', 'NVDA']\nvols = {}\nfor ticker in tickers:\n    vols[ticker] = volatility(wide[ticker])\n\nvols",
   "Every idea in that cell came from Session 2: a loop, a function, a dictionary. "
   "Only the thing inside the function is new.")

ex("L4", "From a dictionary back to pandas", 2,
   "Turn the dictionary you just built into a **Series**, sort it riskiest first, and "
   "compare it with what `groupby` gives for the same three stocks.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nvols = {t: wide[t].pct_change().std() * np.sqrt(252) for t in ['AAPL', 'KO', 'NVDA']}\n\nby_hand = ...\nby_hand",
   "`pd.Series(vols)` takes the dictionary straight, then `.sort_values("
   "ascending=False)`.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nvols = {t: wide[t].pct_change().std() * np.sqrt(252) for t in ['AAPL', 'KO', 'NVDA']}\n\nby_hand = pd.Series(vols).sort_values(ascending=False)\nby_hand",
   "The same three numbers `groupby` would give you. A dictionary is a Series waiting "
   "to happen, which is why Session 2 spent time on dictionaries.")

ex("L5", "Labelling with if, elif, else", 3,
   "Write `risk_label(annual_vol)` that returns `'high'` above 0.30, `'medium'` above "
   "0.18, and `'low'` otherwise. Then loop over every stock's 2024 volatility and "
   "collect the labels in a dictionary.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nvol24 = wide.loc['2024'].pct_change().std() * np.sqrt(252)\n\ndef risk_label(annual_vol):\n    ...\n\nlabels = {}\nfor ticker in vol24.index:\n    ...\n\nlabels",
   ["The function is three branches: `if annual_vol > 0.30: return 'high'`, then "
    "`elif annual_vol > 0.18: return 'medium'`, then `else: return 'low'`.",
    "In the loop, `labels[ticker] = risk_label(vol24[ticker])`."],
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nvol24 = wide.loc['2024'].pct_change().std() * np.sqrt(252)\n\ndef risk_label(annual_vol):\n    \"\"\"Bucket an annualised volatility into high, medium or low.\"\"\"\n    if annual_vol > 0.30:\n        return 'high'\n    elif annual_vol > 0.18:\n        return 'medium'\n    else:\n        return 'low'\n\nlabels = {}\nfor ticker in vol24.index:\n    labels[ticker] = risk_label(vol24[ticker])\n\nlabels",
   "Nvidia alone is `high`; Disney, JPMorgan, Apple, Microsoft and Exxon are "
   "`medium`; the rest are `low`.\n\nYou have just turned a number into a category, "
   "which is exactly what a **classification** problem predicts. Session 4 will call "
   "these labels a *target*.")

ex("L6", "One row per stock, built by a loop", 3,
   "Write `summarise(ticker)` that returns a **dictionary** with the ticker, its 2024 "
   "annualised volatility and its average daily return. Loop over three stocks, "
   "collect the dictionaries in a list, and hand the list to `pd.DataFrame`.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nr24 = wide.loc['2024'].pct_change()\n\ndef summarise(ticker):\n    ...\n\nrows = []\nfor ticker in ['AAPL', 'KO', 'NVDA']:\n    ...\n\ntable = pd.DataFrame(rows)\ntable",
   ["The function returns something like `{'ticker': ticker, 'vol': ..., 'mean_ret': "
    "...}`, using `r24[ticker]`.",
    "In the loop, `rows.append(summarise(ticker))`. `pd.DataFrame` turns a list of "
    "dictionaries into a table, one row per dictionary."],
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nr24 = wide.loc['2024'].pct_change()\n\ndef summarise(ticker):\n    \"\"\"One row of a summary table, as a dictionary.\"\"\"\n    return {\n        'ticker': ticker,\n        'vol': r24[ticker].std() * np.sqrt(252),\n        'mean_ret': r24[ticker].mean(),\n    }\n\nrows = []\nfor ticker in ['AAPL', 'KO', 'NVDA']:\n    rows.append(summarise(ticker))\n\ntable = pd.DataFrame(rows)\ntable",
   "A list of dictionaries into `pd.DataFrame` is one of the most useful patterns "
   "there is, and it is how you will assemble a table of features one row at a time "
   "when the calculation is too awkward for `groupby`.")

ex("L7", "The same answer, two ways", 3,
   "Compute every stock's 2024 annualised volatility twice: once with a loop over "
   "`wide.columns`, and once with pandas in a single expression. Then check that the "
   "two agree.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\np24 = wide.loc['2024']\n\nloop_vols = {}\nfor ticker in p24.columns:\n    ...\n\npandas_vols = ...\npandas_vols",
   ["Inside the loop: `loop_vols[ticker] = p24[ticker].pct_change().std() * "
    "np.sqrt(252)`.",
    "The one-liner is `p24.pct_change().std() * np.sqrt(252)`."],
   "wide = prices.pivot(index='date', columns='ticker', values='close')\np24 = wide.loc['2024']\n\nloop_vols = {}\nfor ticker in p24.columns:\n    loop_vols[ticker] = p24[ticker].pct_change().std() * np.sqrt(252)\n\npandas_vols = p24.pct_change().std() * np.sqrt(252)\n\nprint(pd.Series(loop_vols).sort_index().round(6).equals(pandas_vols.sort_index().round(6)))",
   "`True`. Eleven lines of work and one line of work, agreeing to six decimals. "
   "Checking a fast method against a slow one you trust is a habit worth keeping, "
   "and it is how you will catch the mistakes that do not raise an error.")

md("---")

# ================================ M. groundwork for machine learning
section(
"## \U0001f9ea M · Groundwork for Session 4\n\n"
"Next session is about what a model is actually asked to do. It has a vocabulary: "
"an **observation** is one row, a **feature** is one column you are allowed to use, "
"and the **target** is the column you are trying to predict. Everything below is "
"ordinary pandas. The point is the shape you put the data into."
)

ex("M1", "A table of features", 3,
   "Build a table with one row per stock and three columns describing its 2024: "
   "annualised volatility, average daily return, and the share of up days.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nr24 = wide.loc['2024'].pct_change()\n\nvol = ...\nmean_ret = ...\nup_share = ...\n\nfeatures = ...\nfeatures",
   ["Each one is a whole-table method: `r24.std() * np.sqrt(252)`, `r24.mean()`, and "
    "`(r24 > 0).mean()`. Each gives a Series labelled by ticker.",
    "Then `pd.DataFrame({'vol': vol, 'mean_ret': mean_ret, 'up_share': up_share})` "
    "lines the three up on those labels for you."],
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nr24 = wide.loc['2024'].pct_change()\n\nvol = r24.std() * np.sqrt(252)\nmean_ret = r24.mean()\nup_share = (r24 > 0).mean()\n\nfeatures = pd.DataFrame({'vol': vol, 'mean_ret': mean_ret, 'up_share': up_share})\nfeatures.round(4)",
   "Eleven rows and three columns. **This shape is what every model in this course "
   "eats**: one row per observation, one column per feature, and the row labels "
   "saying which observation is which.")

ex("M2", "Features and target", 2,
   "From that table, split off `y` (the volatility, the thing you would try to "
   "predict) and `X` (the two columns you would be allowed to use).",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nr24 = wide.loc['2024'].pct_change()\nfeatures = pd.DataFrame({'vol': r24.std() * np.sqrt(252), 'mean_ret': r24.mean(), 'up_share': (r24 > 0).mean()})\n\ny = ...\nX = ...\n\nprint('y:', y)\nprint('X:', X)",
   ["`y` is one column, so single brackets: `features['vol']`.",
    "`X` is several columns, so a list inside the brackets: `features[['mean_ret', "
    "'up_share']]`."],
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nr24 = wide.loc['2024'].pct_change()\nfeatures = pd.DataFrame({'vol': r24.std() * np.sqrt(252), 'mean_ret': r24.mean(), 'up_share': (r24 > 0).mean()})\n\ny = features['vol']\nX = features[['mean_ret', 'up_share']]\n\nprint('y:', y.shape)\nprint('X:', X.shape)",
   "`y` is `(11,)` and `X` is `(11, 2)`: eleven observations, two features. The "
   "names `X` and `y` are a convention you will see in every piece of machine "
   "learning code you ever read.")

ex("M3", "From a table to a matrix", 2,
   "Machine-learning libraries want plain numbers, not labels. Convert `X` to a "
   "NumPy array and check its shape.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nr24 = wide.loc['2024'].pct_change()\nX = pd.DataFrame({'mean_ret': r24.mean(), 'up_share': (r24 > 0).mean()})\n\nX_matrix = ...\nX_matrix",
   "`.values` on a DataFrame gives the numbers underneath, as a 2-D NumPy array. "
   "`.to_numpy()` does the same thing.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nr24 = wide.loc['2024'].pct_change()\nX = pd.DataFrame({'mean_ret': r24.mean(), 'up_share': (r24 > 0).mean()})\n\nX_matrix = X.values\n\nprint(type(X_matrix))\nprint(X_matrix.shape)",
   "An `ndarray` of shape `(11, 2)`. This is the same `n` by `p` matrix from section "
   "D, and the same one that goes into $(X'X)^{-1}X'y$. The column names are gone, "
   "which is precisely why you keep the DataFrame around too.")

ex("M4", "Putting the columns on the same scale", 3,
   "One feature is a daily return, around 0.001. The other is a share, around 0.5. "
   "Standardise every column of `X` so each has mean 0 and standard deviation 1.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nr24 = wide.loc['2024'].pct_change()\nX = pd.DataFrame({'mean_ret': r24.mean(), 'up_share': (r24 > 0).mean()})\n\nX_scaled = ...\nX_scaled",
   ["The same formula as B5, but on a whole table: subtract the mean and divide by "
    "the standard deviation.",
    "`(X - X.mean()) / X.std()`. pandas applies it column by column, which is what "
    "you want."],
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nr24 = wide.loc['2024'].pct_change()\nX = pd.DataFrame({'mean_ret': r24.mean(), 'up_share': (r24 > 0).mean()})\n\nX_scaled = (X - X.mean()) / X.std()\nX_scaled.round(3)",
   "Both columns now run over roughly the same range, and a value of 2 means the same "
   "thing in either. Without this step, a method that measures distances would decide "
   "that `up_share` is the only variable that exists.")

ex("M5", "Splitting on time, not at random", 3,
   "Split the price history into an **early** period (2015 to 2022) and a **late** "
   "one (2023 to 2024), and report how many trading days each contains.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\n\nearly = ...\nlate = ...\n\nprint('early:', early)\nprint('late :', late)",
   ["The index is dates, so `.loc` slices with them: `wide.loc['2015':'2022']`.",
    "The second slice is `wide.loc['2023':'2024']`."],
   "wide = prices.pivot(index='date', columns='ticker', values='close')\n\nearly = wide.loc['2015':'2022']\nlate = wide.loc['2023':'2024']\n\nprint('early:', early.shape)\nprint('late :', late.shape)",
   "2,014 days against 502. Splitting **by time** rather than at random is not a "
   "detail: with prices, a randomly chosen test day would sit between two days you "
   "had already looked at, and any model would look far better than it is.")

ex("M6", "Does the past tell you about the future?", 4,
   "Compute the annualised volatility of every stock in the early period and again in "
   "the late one, put both in one table, and look at whether the ranking held.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nearly = wide.loc['2015':'2022']\nlate = wide.loc['2023':'2024']\n\nearly_vol = ...\nlate_vol = ...\n\ncomparison = ...\ncomparison",
   ["Each one is `period.pct_change().std() * np.sqrt(252)`, which gives a Series "
    "labelled by ticker.",
    "`pd.DataFrame({'early': early_vol, 'late': late_vol})` puts them side by side. "
    "Then sort by the early column to see whether the late one follows: "
    "`.sort_values('early', ascending=False)`."],
   "wide = prices.pivot(index='date', columns='ticker', values='close')\nearly = wide.loc['2015':'2022']\nlate = wide.loc['2023':'2024']\n\nearly_vol = early.pct_change().std() * np.sqrt(252)\nlate_vol = late.pct_change().std() * np.sqrt(252)\n\ncomparison = pd.DataFrame({'early': early_vol, 'late': late_vol})\ncomparison.sort_values('early', ascending=False).round(3)",
   "Roughly, yes. Nvidia is the most volatile in both periods and Coca-Cola and SPY "
   "are among the calmest in both. But it is far from exact: Apple falls from second "
   "riskiest to sixth, and Johnson & Johnson rises.\n\nThat gap is the whole "
   "subject of the rest of this course. A quantity that carried over perfectly would "
   "need no model; one that carried over not at all could not be predicted. "
   "Volatility sits in between, which is what makes it worth forecasting.")

md("---")

# =================================================== N. Putting it together
section(
"## \U0001f9e9 N · Putting it together\n\n"
"Three questions that need more than one tool. This is what the case asks of "
"you, so treat these as a rehearsal."
)

ex("N1", "The ranking, annualised and plotted", 3,
   "Compute the **annualised** volatility of every stock over the whole period, sort "
   "it, and draw it as horizontal bars with a title and an x-axis label.",
   "prices['ret'] = prices.groupby('ticker')['close'].pct_change()\n\nannual_vol = ...\n\nfig, ax = plt.subplots(figsize=(9, 3.5))\n...\nplt.show()",
   ["Volatility per stock is `prices.groupby('ticker')['ret'].std()`. Annualising "
    "multiplies it by `np.sqrt(252)`.",
    "Sort it with `.sort_values()` before plotting, then `ax.barh(annual_vol.index, "
    "annual_vol.values)`."],
   "prices['ret'] = prices.groupby('ticker')['close'].pct_change()\n\nannual_vol = (prices.groupby('ticker')['ret'].std() * np.sqrt(252)).sort_values()\n\nfig, ax = plt.subplots(figsize=(9, 3.5))\nax.barh(annual_vol.index, annual_vol.values)\nax.set_title('Annualised volatility, 2015 to 2024', loc='left')\nax.set_xlabel('standard deviation of daily returns, annualised')\nplt.show()",
   f"Nvidia at about {VOL_FULL['NVDA']*np.sqrt(252):.0%} a year, the S&P 500 ETF at "
   f"about {VOL_FULL['SPY']*np.sqrt(252):.0%}. The ETF being calmest is the point of "
   "diversification, in one bar.")

ex("N2", "Which stock rose on the most days?", 3,
   "For every stock, count how many days its return was positive, and rank them.",
   "prices['ret'] = prices.groupby('ticker')['close'].pct_change()\n\nup_days = ...\nup_days",
   ["`prices['ret'] > 0` gives True/False for every row.",
    "Make that a column, then group by ticker and `.sum()` it, because `True` counts "
    "as one."],
   "prices['ret'] = prices.groupby('ticker')['close'].pct_change()\nprices['was_up'] = prices['ret'] > 0\n\nup_days = prices.groupby('ticker')['was_up'].sum().sort_values(ascending=False)\nup_days",
   "The S&P 500 ETF comes top, just ahead of Nvidia, and every one of them is close "
   "to half. Rising on more days than not is a different question from rising a lot, "
   "which is why this ranking does not match the return ranking in L3.")

ex("N3", "Ten years in one number", 4,
   "For every stock, compute the **total return** over the whole period: last close "
   "against first close. Rank them, largest first.",
   "wide = prices.pivot(index='date', columns='ticker', values='close')\n\ntotal_return = ...\ntotal_return",
   ["`wide.iloc[0]` is the first row, one price per stock, and `wide.iloc[-1]` is "
    "the last. Both are Series, so you can do arithmetic with them directly.",
    "`(wide.iloc[-1] - wide.iloc[0]) / wide.iloc[0]`, then sort it."],
   "wide = prices.pivot(index='date', columns='ticker', values='close')\n\ntotal_return = ((wide.iloc[-1] - wide.iloc[0]) / wide.iloc[0]).sort_values(ascending=False)\ntotal_return",
   "Nvidia returned about 27,700%, turning one dollar into 278. Note how differently the "
   "stocks rank here compared to the volatility ranking: risk and return are not the "
   "same question, and Session 4 is about telling them apart.")

md("---")

# ---------------------------------------------------------------- closing
md(
"## ✅ Done\n\n"
"You have now done, with three packages, everything you spent two sessions doing "
"by hand: returns, volatility, counting, ranking, and looking at the result.\n\n"
"Two sections are worth going back to. **K** is about looking things up: this "
"notebook covered a few dozen functions, pandas alone has hundreds, and nobody "
"carries them in their head. **M** is the shape of the data rather than the tools: "
"one row per observation, one column per feature, a target you are trying to "
"predict, and a split by time rather than at random. Session 4 is built on those "
"four ideas, and you have now done all of them by hand.\n\n"
"**Next:** open `session_03_case.ipynb` (*The Analyst's Notebook, Part 3*) and "
"take the risk report from two stocks to all eleven.\n\n"
"*Stuck for more than 15 minutes on anything? Ask a friend, ask an AI for a hint "
"(not the answer), or email me at `jobo@econ.au.dk`.*"
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

n_ex = sum(1 for c in cells if c.cell_type == "markdown" and c.source.startswith("### "))
tiers = {}
for c in cells:
    if c.cell_type == "markdown" and c.source.startswith("### "):
        for k, lab in LABELS.items():
            if lab in c.source.splitlines()[0]:
                tiers[lab] = tiers.get(lab, 0) + 1
print(f"wrote {OUT}  ({len(cells)} cells, {n_ex} exercises)")
print("difficulty:", tiers)
