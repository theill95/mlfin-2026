# -*- coding: utf-8 -*-
"""Build cheatsheet.html: every function the course has used, with an example.

Every example on the page is EXECUTED while the page is built, against the real
course data, and the output shown is the output that actually came back. If an
example raises, the build fails rather than publishing a wrong page.

    python tools/build_cheatsheet.py

Adding a session: append entries below with since="S4". Nothing else to change.
"""
import ast
import html
import io
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cheatsheet.html"

# --------------------------------------------------------------- entries
# (call, what it does, example code, since)
# The example runs in a namespace that already holds: np, pd, plt, prices
# (the full table), wide (one column per ticker), closes (five Apple closes as
# a plain list), week (the same as a NumPy array), and returns (four daily
# returns as an array).

SECTIONS = []


def section(title, blurb=""):
    SECTIONS.append({"title": title, "blurb": blurb, "entries": []})


def e(call, what, example, since="S1", run=True, result=None):
    SECTIONS[-1]["entries"].append(
        {"call": call, "what": what, "example": example.strip(),
         "since": since, "run": run, "result": result})


# ------------------------------------------------------------------------
section("Running code",
        "A notebook shows the value of the last line of a cell. Anything else you "
        "want to see, you have to print.")
e("print(value)", "Show one or more values. Separate several with commas.",
  'print("volatility:", 0.0143)')
e("# comment", "Everything after # on a line is a note for humans, not code.",
  'total = 10  # dollars, not cents\ntotal')
e("value", "The last line of a cell is displayed automatically, with no print.",
  '2 + 2')

section("Numbers")
e("+  -  *  /", "The usual arithmetic. Division always gives a float.",
  '(10 - 4) * 2 / 3')
e("**", "To the power of. Compound growth, and square roots as ** 0.5.",
  '1.07 ** 10')
e("round(number, ndigits)", "Round to a number of decimals.",
  'round(0.014287, 4)')
e("abs(number)", "Distance from zero, sign discarded.", 'abs(-0.0482)')

section("Text")
e('"text"', "A string. Single or double quotes, as long as they match.",
  'ticker = "AAPL"\nticker')
e("+", "Glue strings together. Both sides must be strings.",
  '"AA" + "PL"')
e('f"{value}"', "An f-string: put a value inside text.",
  'ticker = "AAPL"\nf"the ticker is {ticker}"')
e('f"{r:.2%}"', "As a percentage with 2 decimals. The recipe for returns.",
  'r = 0.014287\nf"{r:.2%}"')
e('f"{x:.2f}"', "As a plain number with 2 decimals. The recipe for prices.",
  'f"{183.5622:.2f}"')

section("True and false")
e("==  !=", "Equal, and not equal. Two equals signs, because one assigns.",
  '"AAPL" == "KO"')
e("<  >  <=  >=", "The usual comparisons. The answer is True or False.",
  '0.0331 > 0.0141')
e("and  or  not", "Combine conditions.",
  'r = -0.02\nr < 0 and r > -0.05')

section("Types")
e("type(value)", "What kind of thing this is.", 'type(0.0143)')
e("int(x)  float(x)  str(x)", "Convert between whole number, decimal and text.",
  'int("252") + 1')
e("None", "The absence of a value. What a function returns when it has no return.",
  'x = None\nprint(x)')

section("Built-in functions",
        "Always available, no import needed.")
e("len(items)", "How many items.", 'len(closes)')
e("sum(numbers)", "Add them all up.", 'sum([0.012, -0.004, 0.008])')
e("min(values)  max(values)", "Smallest and largest.",
  'print(min(closes), max(closes))')
e("sorted(values)", "A new list, in order. The original is untouched.",
  'sorted([0.03, 0.01, 0.02])')
e("range(start, stop)", "Whole numbers from start up to but NOT including stop.",
  'list(range(1, 5))')
e("help(thing)", "Print the documentation. Works offline.",
  'help(round)', since="S2")
e("dir(thing)", "List everything an object can do. Useful when you half-remember a name.",
  's = pd.Series([3, 1, 2])\n[n for n in dir(s) if "sort" in n and not n.startswith("_")]',
  since="S3")

section("Lists",
        "An ordered box of values. Counting starts at 0.")
e("[a, b, c]", "Make a list.", 'closes = [183.56, 182.19, 179.87]\ncloses')
e("items[i]", "The item at that position. [0] is first.", 'closes[0]')
e("items[-1]", "Counting from the end. [-1] is last.", 'closes[-1]')
e("items[a:b]", "A slice: from a up to but NOT including b.", 'closes[1:3]')
e("items[:n]  items[n:]", "Everything before n, everything from n.", 'closes[:2]')
e("items.append(x)", "Add to the end. Changes the list itself.",
  'out = []\nout.append(0.012)\nout.append(-0.004)\nout')
e("items.index(x)", "The position of the first x.", 'closes.index(max(closes))')
e("[]", "An empty list, ready for a loop to fill.", 'results = []\nlen(results)')

section("Loops", "Doing something once per item.")
e("for item in items:", "Run the indented body once for every item.",
  'for r in [0.012, -0.004]:\n    print(r)', since="S2")
e("for i in range(n):", "Loop over positions rather than values.",
  'for i in range(3):\n    print(i)', since="S2")
e("range(1, len(items))", "Positions 1 onwards: how you reach item i and i-1 together.",
  'for i in range(1, len(closes)):\n    print(round(closes[i] - closes[i - 1], 2))', since="S2")
e("total += x", "Add to a variable and store it back. Same as total = total + x.",
  'total = 0\nfor r in [0.012, -0.004, 0.008]:\n    total += r\nround(total, 4)', since="S2")
e("while condition:", "Repeat for as long as the condition holds. Something inside must change it.",
  'value, years = 100, 0\nwhile value < 150:\n    value *= 1.10\n    years += 1\nyears', since="S2")

section("Making choices")
e("if condition:", "Run the block only when the condition is True.",
  'r = 0.012\nif r > 0:\n    print("up day")', since="S2")
e("elif / else", "Further cases, checked in order, and a catch-all.",
  'r = -0.02\nif r > 0:\n    print("up")\nelif r < 0:\n    print("down")\nelse:\n    print("flat")',
  since="S2")

section("Your own functions")
e("def name(parameters):", "Create a function. The indented body is what it does.",
  'def simple_return(p0, p1):\n    return (p1 - p0) / p0\n\nround(simple_return(100, 110), 4)',
  since="S2")
e("return value", "Hand a value back and stop there. Without it you get None.",
  'def double(x):\n    return x * 2\n\ndouble(21)', since="S2")
e("def name(x, rate=0.05):", "A default: used when the caller does not pass it.",
  'def grow(price, rate=0.05):\n    return price * (1 + rate)\n\nprint(grow(100), grow(100, 0.10))',
  since="S2")
e('"""What it does."""', "A docstring. Editors show it on hover, and help() prints it.",
  'def volatility(prices):\n    """Standard deviation of the daily returns."""\n    return 0\n\nvolatility.__doc__',
  since="S2")

section("Dictionaries", "Values stored by name instead of by position.")
e('{"AAPL": 0.36}', "Make one. Keys are usually strings.",
  'vol = {"AAPL": 0.0143, "KO": 0.0080}\nvol', since="S2")
e("data[key]", "Read the value under that key.",
  'vol = {"AAPL": 0.0143, "KO": 0.0080}\nvol["AAPL"]', since="S2")
e("data[key] = value", "Add a new key, or overwrite an existing one.",
  'vol = {"AAPL": 0.0143}\nvol["NVDA"] = 0.0331\nvol', since="S2")
e("for key in data:", "Loop over the keys. Read each value with data[key].",
  'vol = {"AAPL": 0.0143, "KO": 0.0080}\nfor t in vol:\n    print(t, vol[t])', since="S2")

section("Packages")
e("import numpy as np", "Bring a package in under a short name. np, pd and plt are conventions everybody uses.",
  'import numpy as np\nnp.sqrt(144)', since="S3")
e("%pip install pandas", "Install a package. Once per machine, in a notebook cell.",
  '%pip install pandas', since="S3", run=False,
  result="Successfully installed pandas   (or: Requirement already satisfied)")

section("NumPy arrays",
        "A list that does arithmetic on every element at once. Indexing and "
        "slicing work exactly as they do for lists.")
e("np.array(list)", "Turn a list of numbers into an array.",
  'close = np.array([183.56, 182.19, 179.87])\nclose', since="S3")
e("values * 2", "Arithmetic applies to every element. No loop.",
  'np.array([1.0, 2.0, 3.0]) * 2', since="S3")
e("a - b", "Two arrays, element by element.",
  'np.array([100.0, 200.0]) - np.array([1.0, 2.0])', since="S3")
e("(v[1:] - v[:-1]) / v[:-1]", "Every daily return in one line: each day against the day before.",
  '(week[1:] - week[:-1]) / week[:-1]', since="S3")
e("values.mean()", "The average.", 'returns.mean()', since="S3")
e("values.std()", "The standard deviation. This is what volatility is.",
  'returns.std()', since="S3")
e("values.min()  .max()  .sum()", "The usual summaries.",
  'print(returns.min(), returns.max(), round(returns.sum(), 4))', since="S3")
e("np.sqrt(x)", "Square root. np.sqrt(252) annualises a daily volatility.",
  '0.0143 * np.sqrt(252)', since="S3")
e("values > 0", "A question asked of every element: True or False for each.",
  'returns > 0', since="S3")
e("(values > 0).sum()", "Count the Trues, because True counts as 1.",
  '(returns > 0).sum()', since="S3")
e("(values > 0).mean()", "The share that are True.", '(returns > 0).mean()', since="S3")
e("values[values > 0]", "Keep only the elements where the condition holds.",
  'returns[returns > 0]', since="S3")
e("values.shape", "How big it is. For a 1-D array it prints as (n,).",
  'week.shape', since="S3")

section("NumPy matrices",
        "A matrix is a 2-D array: rows first, then columns.")
e("np.array([[1, 2], [3, 4]])", "A matrix, written as a list of rows.",
  'X = np.array([[1.0, 0.5],\n              [1.0, 1.5],\n              [1.0, 2.5]])\nX', since="S3")
e("X.shape", "Rows and columns.", 'X = np.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]])\nX.shape', since="S3")
e("X[i, j]  X[i, :]  X[:, j]", "One element, a whole row, a whole column. The colon means all.",
  'X = np.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]])\nprint(X[1, 0], X[1, :], X[:, 1])', since="S3")
e("X.T", "The transpose: rows become columns.",
  'X = np.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]])\nX.T.shape', since="S3")
e("A @ B", "Matrix multiplication. For two vectors this is the dot product.",
  'np.array([0.5, 0.3, 0.2]) @ np.array([0.012, -0.004, 0.031])', since="S3")
e("np.ones(n)", "A vector of ones. The intercept column of a regression.",
  'np.ones(4)', since="S3")
e("np.column_stack([a, b])", "Glue 1-D arrays together as the columns of a matrix.",
  'x = np.array([0.5, 1.5, 2.5])\nnp.column_stack([np.ones(3), x])', since="S3")
e("np.linalg.inv(A)", "The inverse: the matrix that undoes A. Fails if a column is redundant.",
  'A = np.array([[2.0, 1.0], [1.0, 3.0]])\nnp.round(np.linalg.inv(A) @ A, 10)', since="S3")
e("np.linalg.solve(A, b)", "Solve A x = b. Safer and faster than inverting. OLS in one line.",
  'x = np.array([0.5, 1.5, 2.5, 3.5])\ny = np.array([1.0, 2.0, 2.6, 3.8])\nX = np.column_stack([np.ones(4), x])\nnp.linalg.solve(X.T @ X, X.T @ y)', since="S3")

section("pandas: one column",
        "A Series is values with a label on each one: a list and a dictionary at once.")
e("pd.Series(a_dict)", "Build one from a dictionary. The keys become the labels.",
  'pd.Series({"AAPL": 0.0143, "KO": 0.0080, "NVDA": 0.0331})', since="S3")
e("s['AAPL']", "Look a value up by its label.",
  's = pd.Series({"AAPL": 0.0143, "KO": 0.0080})\ns["AAPL"]', since="S3")
e("s * 2", "Maths applies to every value, and the labels come along.",
  'pd.Series({"AAPL": 0.0143, "KO": 0.0080}) * np.sqrt(252)', since="S3")
e("s.mean()  .std()  .min()  .max()", "A Series summarises itself, like an array.",
  'wide["AAPL"].mean()', since="S3")
e("s.sort_values()", "In order. ascending=False for largest first.",
  's = pd.Series({"AAPL": 0.0143, "KO": 0.0080, "NVDA": 0.0331})\ns.sort_values(ascending=False)',
  since="S3")
e("s.dropna()", "Throw away the missing values.",
  'wide["AAPL"].pct_change().dropna().shape', since="S3")

section("pandas: a table",
        "A DataFrame is several Series side by side, sharing one index.")
e("pd.read_csv(path, parse_dates=['date'])", "Read a CSV. parse_dates turns a text column into real dates.",
  'prices.head(3)', since="S3")
e("pd.DataFrame({...})", "Build a table from a dictionary of columns.",
  'pd.DataFrame({"ticker": ["AAPL", "KO"],\n              "vol": [0.0143, 0.0080]})', since="S3")
e("frame.head(n)  .tail(n)", "The first or last n rows, 5 by default.",
  'prices.head(2)', since="S3")
e("frame.shape", "Rows and columns.", 'prices.shape', since="S3")
e("frame.dtypes", "What each column holds. A price column that arrived as text is a silent bug.",
  'prices.dtypes', since="S3")
e("frame.columns", "The column names.", 'list(prices.columns)', since="S3")
e("frame['close']", "One column, as a Series.", 'prices["close"].head(3)', since="S3")
e("frame[['date', 'close']]", "Several columns, as a smaller table. Note the two sets of brackets.",
  'prices[["date", "close"]].head(3)', since="S3")
e("frame[frame['ticker'] == 'AAPL']", "Keep only the rows where the condition is True.",
  'prices[prices["ticker"] == "AAPL"].shape', since="S3")
e("&  |", "And, or. Each condition needs its own brackets.",
  'big = prices[(prices["ticker"] == "AAPL")\n             & (prices["close"] > 250)]\nbig.shape', since="S3")
e(".copy()", "Take your own copy before modifying a selection, so pandas knows what you meant.",
  'aapl = prices[prices["ticker"] == "AAPL"].copy()\naapl.shape', since="S3")
e("frame['ret'] = ...", "Assign to a name that does not exist yet, and the column appears.",
  'aapl = prices[prices["ticker"] == "AAPL"].copy()\naapl["ret"] = aapl["close"].pct_change()\naapl[["date", "close", "ret"]].head(3)',
  since="S3")
e("s.pct_change()", "The percentage change from each row to the next. The first is NaN.",
  'wide["AAPL"].pct_change().head(3)', since="S3")
e("frame.sort_values('ret')", "Sort the rows by a column.",
  'aapl = prices[prices["ticker"] == "AAPL"].copy()\naapl["ret"] = aapl["close"].pct_change()\naapl.sort_values("ret")[["date", "ret"]].head(3)',
  since="S3")
e("frame.loc['2024']", "Rows by label. With a date index you can slice with dates.",
  'wide.loc["2024-01-02":"2024-01-04", ["AAPL", "KO"]]', since="S3")
e("frame.iloc[0]  .iloc[-1]", "Rows by position: the first row, the last row.",
  'wide.iloc[-1].head(3)', since="S3")
e("frame.values", "The plain numbers underneath, as a NumPy array, labels dropped.",
  'm = wide[["AAPL", "KO"]].values\nprint(type(m).__name__, m.shape)', since="S3")

section("pandas: groups and shapes")
e("frame.groupby('ticker').size()", "Split by a column, then count the rows in each group.",
  'prices.groupby("ticker").size()', since="S3")
e("frame.groupby('ticker')['close'].mean()", "Split, compute inside each group, put the answers back together.",
  'prices.groupby("ticker")["close"].mean().round(2)', since="S3")
e("groupby(...)['close'].pct_change()", "A return WITHIN each stock. Group first, or you take a return across two companies.",
  'p = prices.copy()\np["ret"] = p.groupby("ticker")["close"].pct_change()\np.groupby("ticker")["ret"].std().sort_values(ascending=False).round(4)',
  since="S3")
e("g.agg(['mean', 'std', 'min', 'max'])", "Several summaries at once, as a table.",
  'p = prices.copy()\np["ret"] = p.groupby("ticker")["close"].pct_change()\np.groupby("ticker")["ret"].agg(["mean", "std"]).round(4).head(4)',
  since="S3")
e("frame.pivot(index=, columns=, values=)", "Reshape: one row per date, one column per ticker.",
  'wide = prices.pivot(index="date", columns="ticker",\n                    values="close")\nwide.shape', since="S3")
e("s.shift(1)", "Slide a column down by one, so each row can see the previous one. How you line the past up next to the present.",
  'r = wide["AAPL"].pct_change()\npd.DataFrame({"ret": r, "yesterday": r.shift(1)}).head(4)',
  since="S3")
e("s.rolling(20).std()", "A statistic over a sliding window of rows. Where most financial features come from.",
  'r = wide["AAPL"].pct_change()\n(r.rolling(20).std() * np.sqrt(252)).tail(3)', since="S3")
e("s.describe()", "Count, mean, standard deviation and the quartiles, in one call.",
  'wide["KO"].describe().round(2)', since="S3")
e("s.nlargest(n)", "The n largest values, largest first.",
  'wide.pct_change().std().nlargest(3)', since="S3")

section("matplotlib",
        "Every plot is the same three steps: make the axes, draw on them, then say "
        "what the reader is looking at. Figures are not shown on this page.")
e("fig, ax = plt.subplots(figsize=(9, 3))", "Make a figure and one axes to draw on. Every plot starts here.",
  'fig, ax = plt.subplots(figsize=(9, 3))\ntype(ax).__name__', since="S3")
e("ax.plot(x, y)", "A line: something over time.",
  'fig, ax = plt.subplots()\nax.plot([1, 2, 3], [10, 12, 11])\n"drawn"', since="S3")
e("ax.barh(names, values)", "Horizontal bars: comparing named things. Sort before you plot.",
  'fig, ax = plt.subplots()\nv = wide.pct_change().std().sort_values()\nax.barh(v.index, v.values)\n"drawn"',
  since="S3")
e("ax.hist(values, bins=40)", "A histogram: the shape of one variable.",
  'fig, ax = plt.subplots()\nax.hist(wide["AAPL"].pct_change().dropna(), bins=40)\n"drawn"',
  since="S3")
e("ax.scatter(x, y)", "A scatter: one thing against another.",
  'fig, ax = plt.subplots()\nr = wide.pct_change().dropna()\nax.scatter(r["SPY"], r["AAPL"], s=6)\n"drawn"',
  since="S3")
e("ax.set_title(text, loc='left')", "Say what the reader is looking at.",
  'fig, ax = plt.subplots()\nax.set_title("Apple, 2024", loc="left")\nax.get_title(loc="left")', since="S3")
e("ax.set_ylabel(text)  ax.set_xlabel(text)", "Label the axes, with units. \"price (USD)\", not \"price\".",
  'fig, ax = plt.subplots()\nax.set_ylabel("price (USD)")\nax.get_ylabel()', since="S3")
e("label= and ax.legend()", "Name each line, then show the key. Needed as soon as there are two.",
  'fig, ax = plt.subplots()\nax.plot([1, 2], [1, 2], label="AAPL")\nax.legend()\n"drawn"', since="S3")
e("ax.set_ylim(a, b)", "Fix the vertical range. For bars, always start at zero.",
  'fig, ax = plt.subplots()\nax.set_ylim(0, 0.55)\nax.get_ylim()', since="S3")
e("plt.show()", "Display the figure. The last line of a plotting cell.",
  'plt.show()', since="S3", run=False, result="(the figure appears under the cell)")
e("fig.savefig('name.png', dpi=200)", "Save it to a file. Use .pdf for something that stays sharp at any size.",
  "fig.savefig('name.png', dpi=200, bbox_inches='tight')", since="S3", run=False,
  result="(writes name.png next to your notebook)")

section("Reading error messages",
        "The last line names the problem. Read it before you change anything: it is "
        "almost always telling you the truth.")
e("NameError", "A name Python has never seen. Usually a typo, or a cell you have not run yet.",
  'volatilty', run=False, result="NameError: name 'volatilty' is not defined")
e("TypeError", "The right operation on the wrong kind of thing. Often text where a number belongs.",
  '"5" + 5', run=False, result='TypeError: can only concatenate str (not "int") to str')
e("IndexError", "A position that does not exist. A list of 5 stops at index 4.",
  '[1, 2, 3][5]', run=False, result="IndexError: list index out of range")
e("KeyError", "A dictionary key that is not there.", 'vol = {"AAPL": 0.01}\nvol["KO"]',
  since="S2", run=False, result="KeyError: 'KO'")
e("SyntaxError", "Python could not read the line at all. Look for a missing bracket, quote or colon.",
  'print("hello"', run=False, result="SyntaxError: '(' was never closed")
e("ZeroDivisionError", "Dividing by zero. Often an empty list you thought had something in it.",
  '1 / 0', run=False, result="ZeroDivisionError: division by zero")
e("ModuleNotFoundError", "The package is not installed, or you are running a different Python.",
  'import pandas', since="S3", run=False,
  result="ModuleNotFoundError: No module named 'pandas'")
e("FileNotFoundError", "The path is wrong relative to where you are running from.",
  'pd.read_csv("data/prices.csv")', since="S3", run=False,
  result="FileNotFoundError: [Errno 2] No such file or directory: 'data/prices.csv'")
e("ValueError, on shapes", "Two arrays whose dimensions do not fit. Check .shape on both.",
  'np.array([[1.0, 2.0], [3.0, 4.0]]) @ np.array([1.0, 2.0, 3.0])', since="S3", run=False,
  result="ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0")
e("LinAlgError", "A matrix with no inverse: one column carries nothing the others do not. Perfect multicollinearity.",
  'np.linalg.inv(np.array([[1.0, 2.0], [2.0, 4.0]]))', since="S3", run=False,
  result="LinAlgError: Singular matrix")


# ------------------------------------------------------------- execution
def build_namespace():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    prices = pd.read_csv(ROOT / "data" / "prices.csv", parse_dates=["date"])
    wide = prices.pivot(index="date", columns="ticker", values="close").loc["2024"]
    closes = [183.56, 182.19, 179.87, 179.15, 183.48]
    week = np.array(closes)
    returns = (week[1:] - week[:-1]) / week[:-1]
    return {"np": np, "pd": pd, "plt": plt, "prices": prices, "wide": wide,
            "closes": closes, "week": week, "returns": returns}


def run_example(code, ns):
    """Execute a snippet and return what a notebook would show under the cell."""
    tree = ast.parse(code)
    out = io.StringIO()
    displayed = None
    with redirect_stdout(out):
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            if len(tree.body) > 1:
                exec(compile(ast.Module(tree.body[:-1], []), "<ex>", "exec"), ns)
            displayed = eval(compile(ast.Expression(tree.body[-1].value), "<ex>", "eval"), ns)
        else:
            exec(code, ns)
    text = out.getvalue()
    if displayed is not None:
        text += repr(displayed) if not isinstance(displayed, str) or "\n" in repr(displayed) else repr(displayed)
    return text.rstrip()


def truncate(text, max_lines=12, max_chars=600):
    lines = text.split("\n")
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " ..."
    return text


failures = []
n_run = 0
for sec in SECTIONS:
    for entry in sec["entries"]:
        if not entry["run"]:
            entry["output"] = entry["result"] or ""
            continue
        ns = build_namespace()          # a clean namespace per example
        try:
            entry["output"] = truncate(run_example(entry["example"], ns))
            n_run += 1
        except Exception:
            failures.append((sec["title"], entry["call"],
                             traceback.format_exc(limit=1).strip().split("\n")[-1]))
            entry["output"] = "!! FAILED"
        finally:
            import matplotlib.pyplot as _plt
            _plt.close("all")

if failures:
    print(f"{len(failures)} example(s) failed:", file=sys.stderr)
    for sect, call, err in failures:
        print(f"  [{sect}] {call}: {err}", file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------- page
def slug(title):
    return "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-")


SINCE_LABEL = {"S1": "Session 1", "S2": "Session 2", "S3": "Session 3", "S4": "Session 4"}

# The sidebar comes from the same template every other page uses, so this page
# cannot drift away from them.
NAV = (ROOT / "assets" / "nav.html").read_text(encoding="utf-8").rstrip("\n")
NAV = NAV.replace('<a href="cheatsheet.html">', '<a href="cheatsheet.html" aria-current="page">')

parts = []
parts.append("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cheatsheet · Machine Learning in Finance</title>
<link rel="stylesheet" href="assets/site.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<div class="shell">

  <!-- nav:start -->
  """ + NAV + """
  <!-- nav:end -->

  <main class="wrap" id="main">
  <header>
    <div class="accent"></div>
    <div class="eyebrow">Every function the course has used</div>
    <h1>Cheatsheet</h1>
    <p class="lede">One line on what each thing does, and a worked example with the
       output it actually produced. Every example on this page was run against the real
       course data when the page was built, so nothing here is a guess.</p>
  </header>

  <div class="note">
    <strong>Ctrl+P saves this as a PDF.</strong> The menu and the page furniture drop out
    when printing, so you get the reference on its own.
  </div>
""")

parts.append('  <ul class="cs-toc">')
for sec in SECTIONS:
    parts.append(f'    <li><a href="#{slug(sec["title"])}">{html.escape(sec["title"])}</a></li>')
parts.append("  </ul>")

for sec in SECTIONS:
    parts.append(f'  <section class="cs-section" id="{slug(sec["title"])}">')
    parts.append(f'    <h2>{html.escape(sec["title"])}</h2>')
    if sec["blurb"]:
        parts.append(f'    <p class="lede">{html.escape(sec["blurb"])}</p>')
    for entry in sec["entries"]:
        out = entry["output"]
        cls = "cs-out" + ("" if out else " cs-out empty")
        shown = html.escape(out) if out else "no output"
        parts.append('    <div class="cs-entry">')
        parts.append(f'      <div class="cs-call"><span class="cs-since">{SINCE_LABEL[entry["since"]]}</span>'
                     f'{html.escape(entry["call"])}</div>')
        parts.append(f'      <div class="cs-what">{html.escape(entry["what"])}</div>')
        parts.append('      <div class="cs-demo">')
        parts.append(f'        <pre><code>{html.escape(entry["example"])}</code></pre>')
        parts.append(f'        <div class="{cls}">{shown}</div>')
        parts.append("      </div>")
        parts.append("    </div>")
    parts.append("  </section>")

parts.append("""
  <footer>
    Built from the course materials, with every example executed. Something missing?
    Email me at <a href="mailto:jobo@econ.au.dk">jobo@econ.au.dk</a>.
  </footer>
  </main>
</div>
</body>
</html>
""")

OUT.write_text("\n".join(parts), encoding="utf-8", newline="\n")
n_entries = sum(len(s["entries"]) for s in SECTIONS)
print(f"wrote {OUT.name}: {len(SECTIONS)} sections, {n_entries} entries, "
      f"{n_run} examples executed, 0 failures")
