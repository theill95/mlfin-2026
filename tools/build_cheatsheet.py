# -*- coding: utf-8 -*-
"""Build cheatsheet.html: every function the course has used, one row each.

A reference page rather than a set of worked examples. Each section opens with
the imports it needs, then one row per function: the call, written with general
argument names, on the left, and what it does on the right. Nothing is run.
Instead the build checks every name in every call: a bare function has to be
imported in its section (or be a builtin), and an attribute written against a
known object (np.x, pd.x, plt.x, ax.x, s.x, frame.x, ...) has to exist in that
library. A misspelt name fails the build rather than reaching a student.

    python tools/build_cheatsheet.py

The previous format, with an executed example under every entry, is archived
in tools/archive/ together with its rendered page and CSS.

Adding a session: append rows below with since="S11" and add it to SINCE_LABEL.
Rows belong with their topic (a NumPy function under NumPy, whichever session
introduced it); the session tag at the edge of the row says when it arrived.
"""
import builtins
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cheatsheet.html"

SECTIONS = []


def section(title, blurb="", imports=None, names=None):
    """A card: an optional import block shown once, an optional line naming
    the conventions its rows use, then the rows."""
    SECTIONS.append({"title": title, "blurb": blurb, "rows": [],
                     "imports": imports.strip() if imports else None, "names": names})


def r(signature, what, since="S1"):
    """One row: the call with general argument names, and what it does."""
    SECTIONS[-1]["rows"].append({"call": signature, "what": what, "since": since})


# ======================================================================
section("Running code",
        "A notebook shows the value of the last line of a cell. Anything else you want "
        "to see, you have to print.")
r("print(value, other)", "Show one or more values. Separate several with commas.")
r("# comment", "Everything after # on a line is a note for humans, not code.")
r("value", "The last line of a cell is displayed automatically, with no print.")

section("Numbers")
r("+  -  *  /", "The usual arithmetic. Division always gives a float.")
r("x ** n", "To the power of. Compound growth, and square roots as ** 0.5.")
r("round(number, ndigits)", "Round to a number of decimals.")
r("abs(number)", "Distance from zero, sign discarded.", "S2")

section("Text")
r('"text"', "A string. Single or double quotes, as long as they match.")
r('"AA" + "PL"', "Glue strings together. Both sides must be strings.")
r('f"{value}"', "An f-string: put a value inside text.")
r('f"{r:.2%}"', "As a percentage with 2 decimals. The recipe for returns.")
r('f"{x:.2f}"', "As a plain number with 2 decimals. The recipe for prices.")
r('f"{name:<6}"  f"{x:>7.1%}"', "A field width: pad left-aligned text, or right-align a number, so a column of output lines up.", "S2")
r("text.upper()  text.lower()", "A copy of the string in one case. Tickers arrive in every mixture of both.")
r("text.strip()", "A copy with the spaces at each end removed. The first thing you do to text from a file.", "S2")
r("text.count(part)", "How many times a piece of text appears inside a string.")

section("True and false")
r("==  !=", "Equal, and not equal. Two equals signs, because one assigns.")
r("<  >  <=  >=", "The usual comparisons. The answer is True or False.")
r("and  or  not", "Combine conditions. In pandas the same job is done by &, | and ~.", "S2")

section("Types")
r("type(value)", "What kind of thing this is.")
r("int(x)  float(x)  str(x)", "Convert between whole number, decimal and text.")
r("None", "The absence of a value. What a function returns when it has no return.")

section("Built-in functions", "Always available, no import needed.")
r("len(items)", "How many items.")
r("sum(numbers)", "Add them all up.")
r("min(values)  max(values)", "Smallest and largest.")
r("sorted(values)", "A new list, in order. The original is untouched.", "S2")
r("range(start, stop)", "Whole numbers from start up to but NOT including stop.", "S2")
r("enumerate(items)", "The position and the value together, so you need no counter of your own.", "S2")
r("zip(a, b)", "Walk two lists in step, with no indexing. Stops at the shorter one.", "S2")
r("help(thing)", "Print the documentation. Works offline.", "S2")
r("dir(thing)", "List everything an object can do. Useful when you half-remember a name.", "S3")

section("Lists", "An ordered box of values. Counting starts at 0.")
r("[a, b, c]", "Make a list.")
r("items[i]", "The item at that position. [0] is first.")
r("items[-1]", "Counting from the end. [-1] is last.")
r("items[a:b]", "A slice: from a up to but NOT including b.")
r("items[:n]  items[n:]", "Everything before n, everything from n.")
r("items.append(x)", "Add to the end. Changes the list itself.")
r("items.index(x)", "The position of the first x.")
r("[]", "An empty list, ready for a loop to fill.")

section("Loops", "Doing something once per item.")
r("for item in items:", "Run the indented body once for every item.", "S2")
r("for i in range(n):", "Loop over positions rather than values.", "S2")
r("for i in range(1, len(items)):", "Positions 1 onwards: how you reach item i and i-1 together.", "S2")
r("total += x", "Add to a variable and store it back. Same as total = total + x.", "S2")
r("while condition:", "Repeat for as long as the condition holds. Something inside must change it.", "S2")

section("Making choices")
r("if condition:", "Run the block only when the condition is True.", "S2")
r("elif condition:  else:", "Further cases, checked in order, and a catch-all.", "S2")

section("Your own functions")
r("def name(parameters):", "Create a function. The indented body is what it does.", "S2")
r("return value", "Hand a value back and stop there. Without it you get None.", "S2")
r("def name(x, rate=0.05):", "A default: used when the caller does not pass it.", "S2")
r('"""What it does."""', "A docstring, the first line of the body. Editors show it on hover, and help() prints it.", "S2")

section("Dictionaries", "Values stored by name instead of by position.")
r('{"AAPL": 0.36, "KO": 0.12}', "Make one. Keys are usually strings.", "S2")
r("data[key]", "Read the value under that key.", "S2")
r("data[key] = value", "Add a new key, or overwrite an existing one.", "S2")
r("for key in data:", "Loop over the keys. Read each value with data[key].", "S2")
r("data.keys()  data.values()", "The names on their own, or the numbers on their own. max, min, sum and for all accept them.", "S2")
r("for key, value in data.items():", "The key and the value together, so you need no lookup.", "S2")
r("{}", "An empty dictionary, ready for a loop to fill. The counterpart of [].", "S2")
r("min(data, key=data.get)", "The KEY with the smallest value, not the value itself. max works the same way.", "S5")
r("sorted(data, key=data.get)", "All the keys, ordered by their values. Largest first with reverse=True.", "S5")

section("Packages")
r("import numpy as np", "Bring a package in under a short name. np, pd and plt are conventions everybody uses.", "S3")
r("%pip install pandas", "Install a package. Once per machine, in a notebook cell.", "S3")

# ======================================================================
section("NumPy arrays",
        "A list that does arithmetic on every element at once. Indexing and slicing "
        "work exactly as they do for lists.",
        imports="import numpy as np",
        names="On this card: values is a one-dimensional array, mask an array of True and False.")
r("np.array(list)", "Turn a list of numbers into an array.", "S3")
r("values * 2", "Arithmetic applies to every element. No loop.", "S3")
r("a - b", "Two arrays, element by element.", "S3")
r("(v[1:] - v[:-1]) / v[:-1]", "Every daily return in one line: each day against the day before.", "S3")
r("values.mean()", "The average.", "S3")
r("values.std()", "The standard deviation. This is what volatility is.", "S3")
r("values.min()  values.max()  values.sum()", "The usual summaries.", "S3")
r("np.sqrt(x)", "Square root. np.sqrt(252) annualises a daily volatility.", "S3")
r("values > 0", "A question asked of every element: True or False for each.", "S3")
r("(values > 0).sum()", "Count the Trues, because True counts as 1.", "S3")
r("(values > 0).mean()", "The share that are True.", "S3")
r("values[values > 0]", "Keep only the elements where the condition holds.", "S3")
r("values.shape", "How big it is. For a 1-D array it prints as (n,).", "S3")
r("values.argmax()  values.argmin()", "The POSITION of the largest or smallest element. Feed it to an index to get the label.", "S5")
r("(a == b).all()", "True only when every element of a comparison is True: the vectorised way to check that two arrays or columns agree everywhere.", "S8")
r("np.where(mask, a, b)", "An array that takes a where the mask is True and b elsewhere. Keeps one group's values and blanks the rest before argmin or argmax.", "S8")
r("np.zeros(n, dtype=int)", "An array of n zeros, as integers; without dtype, as floats. A prediction of 0 on every row is np.zeros(len(test), dtype=int).", "S8")
r("np.arange(start, stop, step)", "Values from start up to but not including stop, a fixed step apart. np.linspace fixes the count instead of the step.", "S8")
r("np.logspace(0, 4, 5)", "Values spaced by a constant factor: 1, 10, 100, 1000, 10000. The right shape for a grid of alphas.", "S6")
r("np.random.default_rng(0)", "A random number generator with a fixed seed, so the same code gives the same numbers every time.", "S4")

section("NumPy matrices", "A matrix is a 2-D array: rows first, then columns.",
        imports="import numpy as np")
r("np.array([[1, 2], [3, 4]])", "A matrix, written as a list of rows.", "S3")
r("X.shape", "Rows and columns.", "S3")
r("X[i, j]  X[i, :]  X[:, j]", "One element, a whole row, a whole column. The colon means all.", "S3")
r("X.T", "The transpose: rows become columns.", "S3")
r("X.sum(axis=1)", "Sum along each row, one total per row; axis=0 sums down each column instead. The softmax divides every row of scores by its own total, so each row of probabilities adds to one.", "S9")
r("A @ B", "Matrix multiplication. For two vectors this is the dot product.", "S3")
r("np.ones(n)", "A vector of ones. The intercept column of a regression.", "S3")
r("np.column_stack([a, b])", "Glue 1-D arrays together as the columns of a matrix.", "S3")
r("np.linalg.inv(A)", "The inverse: the matrix that undoes A. Fails if a column is redundant.", "S3")
r("np.linalg.solve(A, b)", "Solve A x = b. Safer and faster than inverting. OLS in one line: solve(X.T @ X, X.T @ y).", "S3")

# ======================================================================
section("pandas: one column",
        "A Series is values with a label on each one: a list and a dictionary at once.",
        imports="import pandas as pd",
        names="On this card: s is a Series.")
r("pd.Series(a_dict)", "Build one from a dictionary. The keys become the labels.", "S3")
r('s["AAPL"]', "Look a value up by its label.", "S3")
r("s * 2", "Maths applies to every value, and the labels come along.", "S3")
r("s.mean()  s.std()  s.min()  s.max()", "A Series summarises itself, like an array.", "S3")
r("s.median()", "The middle value. Compare it with the mean: a gap between them means a skewed tail.", "S4")
r("s.sort_values(ascending=False)", "In order. Leave ascending out for smallest first.", "S3")
r("s.value_counts(normalize=True)", "The share of each distinct value, largest first; leave normalize out for the counts. The largest share is what the majority rule scores.", "S9")
r("s.dropna()", "Throw away the missing values.", "S3")
r("s.abs()", "The size of every value, sign dropped. abs() on its own does the same for one number.", "S4")

section("pandas: a table", "A DataFrame is several Series side by side, sharing one index.",
        imports="import numpy as np\nimport pandas as pd",
        names="On this card: frame is a DataFrame, mask a column of True and False.")
r('pd.read_csv(path, parse_dates=["date"])', "Read a CSV. parse_dates turns a text column into real dates.", "S3")
r("pd.DataFrame({name: values, ...})", "Build a table from a dictionary of columns.", "S3")
r("frame.head(n)  frame.tail(n)", "The first or last n rows, 5 by default.", "S3")
r("frame.shape", "Rows and columns.", "S3")
r("frame.dtypes", "What each column holds. A price column that arrived as text is a silent bug.", "S3")
r("frame.columns", "The column names. list(frame.columns) for a plain list.", "S3")
r('frame["close"]', "One column, as a Series.", "S3")
r('frame[["date", "close"]]', "Several columns, as a smaller table. Note the two sets of brackets.", "S3")
r('frame[frame["ticker"] == "AAPL"]', "Keep only the rows where the condition is True.", "S3")
r("(mask_a) & (mask_b)   (mask_a) | (mask_b)", "And, or. Each condition needs its own brackets.", "S3")
r("~mask", "Flip a mask: every True becomes False. How you get the other half without writing the condition twice.", "S4")
r("frame.copy()", "Take your own copy before modifying a selection, so pandas knows what you meant.", "S3")
r('frame["new"] = values', "Assign to a name that does not exist yet, and the column appears.", "S3")
r("s.pct_change()", "The percentage change from each row to the next. The first is NaN.", "S3")
r('frame.sort_values("col")', "Sort the rows by a column.", "S3")
r('frame.loc["2024-01-02":"2024-01-31"]', "Rows by label. With a date index you can slice with dates, and both ends are included.", "S3")
r("frame.iloc[0]  frame.iloc[-1]", "Rows by position: the first row, the last row.", "S3")
r("frame.values", "The plain numbers underneath, as a NumPy array, labels dropped.", "S3")
r('frame.set_index("date")', "Move a column into the index, so .loc can slice by it.", "S4")
r('frame.loc["2023-11-14 13:00":"2023-11-14 16:00", ["a", "b"]]', "Rows by date and hour on an hourly index, and columns by name, in one pair of brackets. Both ends of the slice are included.", "S10")
r("frame.dropna()", "Drop every row with a missing value in any column, such as the first rows a shift leaves empty. Compare the number of rows before and after.", "S10")
r("for col in frame:", "Looping over a table gives its column names, one at a time: this is how the columns from get_dummies are copied into another table.", "S10")
r('pd.cut(s, [0, 0.2, 0.4, 1.0])', "Sort every value into one of the ranges given, and return the range it fell in. Group by the result to summarise each range.", "S9")
r('pd.cut(s, edges, labels=["low", "mid", "high"])', "The same, with a name for each range instead of the range itself. This is how a number becomes a label with more than two values.", "S9")
r("pd.cut(s, [-np.inf, -50, 50, np.inf])", "Ranges open at both ends, so every value lands in one: down by more than 50, less than 50 either way, up by more than 50.", "S10")

section("pandas: groups and shapes", imports="import pandas as pd",
        names="On this card: frame is a DataFrame, s a Series, g a grouped table such as frame.groupby(\"ticker\").")
r('frame.groupby("ticker").size()', "Split by a column, then count the rows in each group.", "S3")
r('frame.groupby("ticker")["close"].mean()', "Split, compute inside each group, put the answers back together.", "S3")
r('frame.groupby("ticker")["close"].pct_change()', "A return WITHIN each stock. Group first, or you take a return across two companies.", "S3")
r('frame.groupby(buckets, observed=True)["y"].agg(["size", "mean"])', "Group by the ranges pd.cut made, keeping only ranges that have rows: how many rows fell in each range, and the share of ones among them. This is how a calibration table is built.", "S9")
r('frame.groupby("hour")["price"].shift(1)', "The value one row earlier WITHIN each group. With one row per hour, grouped by the clock hour, that is the same hour the day before; .shift(2) reaches two days back. The first row of each group is NaN.", "S10")
r('g.agg(["mean", "std", "min", "max"])', "Several summaries at once, as a table.", "S3")
r('frame.pivot(index="date", columns="ticker", values="close")', "Reshape: one row per date, one column per ticker.", "S3")
r("s.shift(1)", "Slide a column down by one, so each row can see the previous one. How you line the past up next to the present.", "S3")
r("s.rolling(window).std()", "A statistic over a sliding window of rows. Where most financial features come from. Also .mean() and .sum().", "S3")
r("s.groupby(s.index.year).mean()", "A dated Series grouped by the year in its index, one mean per year. From a Series of True and False, one share per year.", "S8")
r("s.describe()", "Count, mean, standard deviation and the quartiles, in one call.", "S3")
r("s.nlargest(n)", "The n largest values, largest first.", "S3")

# ======================================================================
section("matplotlib",
        "Every plot is the same three steps: make the axes, draw on them, then say what "
        "the reader is looking at.",
        imports="import matplotlib.pyplot as plt")
r("fig, ax = plt.subplots(figsize=(9, 3))", "Make a figure and one axes to draw on. Every plot starts here.", "S3")
r("ax.plot(x, y)", "A line: something over time.", "S3")
r("ax.barh(names, values)", "Horizontal bars: comparing named things. Sort before you plot.", "S3")
r("ax.hist(values, bins=40)", "A histogram: the shape of one variable.", "S3")
r("ax.scatter(x, y)", "A scatter: one thing against another. s=6 makes the dots small.", "S3")
r('ax.set_title(text, loc="left")', "Say what the reader is looking at.", "S3")
r("ax.set_ylabel(text)  ax.set_xlabel(text)", 'Label the axes, with units. "price (USD)", not "price".', "S3")
r('ax.plot(x, y, label="AAPL")  ax.legend()', "Name each line, then show the key. Needed as soon as there are two.", "S3")
r("ax.set_ylim(a, b)", "Fix the vertical range. For bars, always start at zero.", "S3")
r("ax.axhline(y)", "A horizontal reference line: zero on an error plot, or an average across bars. ax.axvline(x) is the vertical one.", "S4")
r('ax.set_xscale("log")', "A logarithmic axis, so factors of ten are evenly spaced. For a validation curve over alpha or C.", "S6")
r("plt.show()", "Display the figure. The last line of a plotting cell.", "S3")
r('fig.savefig("name.png", dpi=200)', "Save it to a file. Use .pdf for something that stays sharp at any size.", "S3")

# ======================================================================
section("Features and a target",
        "One row per observation, one column per feature, and one column you are trying "
        "to predict. Every model assumes it.",
        names="On this card: s is a Series, X the feature table.")
r("s.shift(-1)", "Move a column UP, so tomorrow's value sits on today's row. This is how a target is built.", "S4")
r("s.rolling(20).std()", "A feature over the last twenty rows. It only ever looks backwards, which is what makes it legal.", "S4")
r("(s > 0).astype(int)", "Turn a number into a 0/1 label, which turns a regression into a classification.", "S4")
r("s.corr(other)", "How strongly two columns move together, between -1 and +1. A feature that correlates almost perfectly with the target is a leak, not a discovery.", "S4")
r("s.diff()", "The difference between each row and the one before it. On logged prices this is the log return.", "S4")
r('frame["x"] - frame.groupby("hour")["x"].shift(1)', "A change as a feature: this hour's value minus the same hour the day before, such as tomorrow's wind forecast minus today's. Often more telling than the level itself.", "S10")
r('frame["change"].abs()', "The size of a change, sign dropped. A logistic regression cannot build this shape from the change itself, so a big move either way needs the column.", "S10")
r("n, p = X.shape", "The two numbers that describe a learning problem: observations, and feature columns. The target is not a feature.", "S4")

section("Missing values",
        "Real tables have holes in them. What you do about the holes changes your answer, "
        "so it is a decision rather than a formality.",
        imports="import pandas as pd")
r('pd.date_range(start, end, freq="D")', "Every calendar day between two dates.", "S4")
r("frame.reindex(index)", "Force a table onto different row labels. Rows that did not exist arrive empty.", "S4")
r("frame.isna().sum()", "Count the missing values in each column.", "S4")
r("s.ffill()", "Carry the last known value forward. It only looks backwards, so a forecast may use it.", "S4")
r("s.fillna(value)", "Fill with something you choose. Filling with the column mean shrinks the variance, so be deliberate.", "S4")
r("s.interpolate()", "Draw a straight line across the gap. It uses the value AFTER the gap, so never in a forecast.", "S4")
r("s.isna().astype(int)", "Record that a value was missing. Sometimes that fact is itself the signal.", "S4")

section("Extremes, and changing the scale",
        imports="import numpy as np\nimport pandas as pd")
r("s.quantile([0.25, 0.75])", "The percentiles. Q1 and Q3 are the edges of the middle half of the data.", "S4")
r("(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)", "The IQR rule, with iqr = q3 - q1: flag anything outside Q1 - 1.5 IQR to Q3 + 1.5 IQR. On returns it flags far more than it should.", "S4")
r("np.log(values)", "The natural log. Log returns add up over time, and a log tames a long right tail.", "S4")
r("(s - s.mean()) / s.std()", "Standardise: mean 0, standard deviation 1. Compute those two numbers on the training rows only.", "S4")
r('pd.get_dummies(frame, columns=["sector"], dtype=int)', "One 0/1 column per category. This is how text gets into a feature table.", "S4")
r('pd.get_dummies(s, prefix="weekday", dtype=int)', "One 0/1 column per value of a single column, named weekday_0, weekday_1 and so on. A logistic regression then gives each weekday its own level, where the weekday as one number could only give it one slope across the week.", "S10")

# ======================================================================
section("Scoring a prediction of a number",
        "Choose the metric before you see the answer, and always report it beside a baseline.",
        imports="import numpy as np",
        names="On this card: y is what happened and p the prediction, two arrays of the same length.")
r("np.abs(y - p).mean()", "MAE, mean absolute error: the average size of a mistake, in the units of the target.", "S4")
r("((y - p) ** 2).mean()", "MSE, mean squared error: one large miss dominates it, which is sometimes exactly what you want.", "S4")
r("np.sqrt(((y - p) ** 2).mean())", "RMSE, the root of the MSE, back in the units of the target. The one to report.", "S4")
r("1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()", "R squared: how much better than always predicting the average. Zero means no better at all, and out of sample it can be negative.", "S4")
r("np.full(len(y), y_train.mean())", "The baseline prediction: one number, the training average, repeated. Nothing counts as a result until it beats this.", "S4")

section("Scoring a prediction of a label",
        "No residuals here. Four counts, and every classification metric is built out of them.",
        names="On this card: said and was are columns of True and False, the prediction and what happened; "
              "tp, fp, fn and tn are the four counts.")
r("(said & was).sum()", "True positives. False positives are (said & ~was), misses (~said & was), true negatives (~said & ~was).", "S4")
r("(tp + tn) / (tp + fp + fn + tn)", "Accuracy: the share called correctly. Meaningless when one class is rare.", "S4")
r("tp / (tp + fp)", "Precision: of the times the model said yes, how often was it right? Low precision means false alarms.", "S4")
r("tp / (tp + fn)", "Recall: of the times it really happened, how many were caught? Low recall means misses.", "S4")
r("2 * precision * recall / (precision + recall)", "F1: the harmonic mean of the two, which sits close to the smaller one.", "S4")
r("cost_miss * fn + cost_false * fp", "What a set of decisions costs when the two mistakes are not worth the same. Sweep the threshold and keep the cheapest, on training rows only.", "S9")
r("cost_false / (cost_false + cost_miss)", "The cheapest threshold, written down rather than swept for. It only works on probabilities that are calibrated.", "S9")

section("Training rows and test rows",
        "A score computed on the rows a model learned from is not evidence.",
        imports='import numpy as np\nfrom sklearn.model_selection import train_test_split')
r('frame.loc[:"2022-12-31"]', 'Split by date, never at random: the rows up to a date for training, frame.loc["2023-01-01":] for testing. A shuffled time series lets a model train on the future.', "S4")
r("train_test_split(frame, test_size=0.3, random_state=0, stratify=y)", "Shuffle the rows and cut them in two. Right for a cross-section where rows are separate cases, wrong for a time series. stratify keeps a label's share equal in both halves, which matters when it is rare.", "S9")
r("np.polyfit(x, y, degree)", "Fit a polynomial of that degree. Raising the degree always lowers training error.", "S4")
r("np.polyval(coef, x)", "Evaluate a fitted polynomial, so you can score it on rows it never saw.", "S4")

# ======================================================================
SK_NAMES = ("On this card: X is the feature table, always with two brackets, y the target column, "
            "model a fitted model, folds a TimeSeriesSplit.")

section("Fitting a model",
        "Four lines: import, create, fit, predict. Every model in scikit-learn takes them, "
        "so the only thing that changes later is the first one.",
        imports="from sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import mean_squared_error",
        names=SK_NAMES)
r('frame[["col"]]', "A LIST of column names, so the result stays a table. X always has to be two-dimensional.", "S5")
r("model = LinearRegression()", "Create the model. It installs as scikit-learn and imports as sklearn.", "S5")
r("model.fit(X, y)", "Read the training rows and compute the coefficients. Features first, target second.", "S5")
r("model.predict(X)", "One prediction per row of X, in the same order, as a plain NumPy array.", "S5")
r("model.intercept_  model.coef_", "What the fit found. A trailing underscore means it came from the data rather than from you.", "S5")
r("mean_squared_error(y, p)", "The truth goes first. Take np.sqrt of it to get back into the units of the target.", "S5")

section("Choosing between models",
        "Rows used to compare candidates can no longer give the winner an honest score. That is "
        "what the folds are for.",
        imports="from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_score",
        names=SK_NAMES)
r("folds = TimeSeriesSplit(n_splits=5)", "Folds that move forward in time, so every scored block comes after the rows fitted on.", "S5")
r("TimeSeriesSplit(n_splits=5, max_train_size=500)", "A rolling window instead of an expanding one: the oldest rows drop out.", "S5")
r("TimeSeriesSplit(n_splits=5, gap=20)", "Drop the rows whose target reaches into the block about to be scored.", "S5")
r("KFold(n_splits=5, shuffle=True, random_state=0)", "The textbook folds. They assume the rows are interchangeable, which a time series is not.", "S5")
r("for fit_rows, score_rows in folds.split(X):", "The row positions in each fold, as pairs. cross_val_score loops over this for you.", "S5")
r('cross_val_score(model, X, y, cv=folds, scoring="neg_root_mean_squared_error")', "Fit and score once per fold. Errors come back negative, because every score is reported so that larger is better; put a minus in front.", "S5")

section("Penalised regression",
        "Linear regression with a charge on the size of the coefficients, the scaling that "
        "charge needs, and a search over how strong it should be.",
        imports="from sklearn.linear_model import Ridge, Lasso, ElasticNet\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import GridSearchCV",
        names=SK_NAMES + " scaler is a fitted StandardScaler, pipe a fitted Pipeline, search a fitted GridSearchCV.")
r("Ridge(alpha=1000)", "Linear regression plus alpha times the sum of squared slopes. alpha=0 is OLS; larger alpha pulls every slope towards zero.", "S6")
r("Lasso(alpha=0.1)", "The same with a charge on the sum of absolute slopes. Sets some coefficients to exactly zero.", "S6")
r("ElasticNet(alpha=0.1, l1_ratio=0.5)", "Both penalties at once. l1_ratio=1 is the lasso, l1_ratio=0 is ridge.", "S6")
r("Lasso(alpha=0.001, max_iter=10000)", "More passes for the solver. Raise it when the ConvergenceWarning appears; standardise first, which is the usual cause.", "S6")
r("scaler = StandardScaler().fit(X_train)", "Learn each column's mean and standard deviation from the TRAINING rows. The penalty charges by coefficient size, and size depends on units.", "S6")
r("scaler.transform(X)", "Subtract the learned mean and divide by the learned standard deviation. Returns a NumPy array; use it on the test rows too, with the training numbers.", "S6")
r("scaler.mean_  scaler.scale_", "The numbers the scaler learned, one per column, in column order.", "S6")
r('Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1000))])', "Two steps as one model: .fit scales then fits, .predict scales with the fitted scaler then predicts. Cross-validation refits the scaler inside every fold.", "S6")
r('pipe.named_steps["ridge"].coef_', "Reach the fitted object inside a step. The pipeline itself has no coef_.", "S6")
r('grid = {"ridge__alpha": [1, 10, 100, 1000, 10000]}', "The setting to search and the values to try: step name, two underscores, argument name. Several keys multiply.", "S6")
r('search = GridSearchCV(pipe, grid, cv=folds, scoring="neg_root_mean_squared_error")', "Fit every value on every fold, keep the best mean score, and refit it on all the training rows. Always pass cv; the default ignores time order.", "S6")
r("search.fit(X, y)  search.best_params_  search.best_score_", "Run the search; then the winning setting, and its mean score over the folds.", "S6")
r("search.best_estimator_  search.predict(X)", "The winning pipeline, refitted on every training row, and predictions from it. Open the test rows once, here.", "S6")
r("pd.DataFrame(search.cv_results_)", "The whole grid as a table: every value tried, its mean score over the folds, the spread, and its rank.", "S6")
r("model.get_params()", "Every argument of an object and its current value. help(Ridge) prints the documentation.", "S6")

section("Classification",
        "A label as the target, logistic regression, the probabilities it gives, and the "
        "scores that judge a classifier.",
        imports='from sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.calibration import CalibratedClassifierCV\nfrom sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV\nfrom sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,\n    recall_score, roc_curve, roc_auc_score, log_loss, f1_score,\n    average_precision_score, precision_recall_curve, brier_score_loss,\n    classification_report)',
        names="On this card: X is the feature table, y the label column of ones and zeros, model a fitted "
              "classifier, p the probability column model.predict_proba(X)[:, 1], predicted the 0/1 predictions.")
r("(a > b).astype(int)", "A label from a comparison: True and False become 1 and 0. Its mean is the share of ones, and predicting the majority class everywhere is the accuracy a model has to beat.", "S8")
r('LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=100)', "The classifier, shown with its defaults: a ridge penalty of strength 1/C, solved in at most 100 steps. Create it, then .fit(X, y), as for every model.", "S8")
r("model.intercept_  model.coef_", "The fitted numbers, on the log-odds scale. coef_ has one row per boundary between classes, so coef_[0, 0] is the first slope.", "S8")
r("-model.intercept_[0] / model.coef_[0, 0]", "With one column: the value of the column at which the probability is one half, where .predict() switches from 1 to 0.", "S8")
r("np.exp(model.coef_)", "The odds factor: one unit more of a column multiplies the odds of a 1 by this. A coefficient adds to the log-odds; its exponential multiplies the odds.", "S8")
r("model.classes_", "The classes, in the order predict_proba uses for its columns.", "S8")
r("model.predict_proba(X)", "One row per observation, one column per class, each row summing to one. [:, 1] is the probability of a 1.", "S8")
r("model.predict(X)", "The 0/1 predictions: 1 where the probability of a 1 is at least one half.", "S8")
r("(p >= threshold).astype(int)", "Predictions at any other threshold, from the probability column. The model is unchanged; only the comparison changed.", "S8")
r("model.n_iter_", "How many steps the solver took. Log-loss has no formula for the coefficients, so .fit() improves them step by step; raise max_iter when the ConvergenceWarning appears, or standardise first.", "S8")
r("log_loss(y, p)", "The objective, averaged over the rows: minus the log of the probability given to what happened. A confident wrong probability costs the most.", "S8")
r("accuracy_score(y, predicted)", "The share of predictions that match the label. True labels first, predictions second, like every metric. Compare it with the majority rule.", "S8")
r("confusion_matrix(y, predicted)", "A 2 by 2 array: rows are what happened, columns are the prediction, both in the order of classes_. Top left true negatives, bottom right true positives.", "S8")
r("precision_score(y, predicted)", "The share of predicted 1s that were 1. Falls when the threshold falls.", "S8")
r("recall_score(y, predicted)", "The share of real 1s that were predicted. Falls when the threshold rises.", "S8")
r("roc_curve(y, p)", "Returns fpr, tpr, thresholds: the false positive rate and the recall at every threshold where a point moves. Joined up, the ROC curve.", "S8")
r("roc_auc_score(y, p)", "The area under the ROC curve: 0.5 for a random ranking, 1 for a perfect one, no threshold involved. Pass the probabilities, never the 0/1 predictions.", "S8")
r('cross_val_score(model, X, y, cv=folds, scoring="roc_auc")', 'One score per fold; larger is better, so no minus sign. Other strings: "accuracy", "neg_log_loss", "precision", "recall".', "S8")
r("LogisticRegression(C=0.01)", "The strength of the penalty, upside down: C = 1/alpha, so a small C is a strong penalty. Standardise in a pipeline first and choose C on a grid in factors of ten.", "S8")
r('{"logit__C": [0.0001, 0.001, 0.01, 0.1, 1, 10]}', 'A grid for GridSearchCV: the step name, two underscores, the argument. Then GridSearchCV(pipe, grid, cv=folds, scoring="roc_auc").fit(X, y), and best_params_, best_score_, predict_proba as before.', "S8")
r('LogisticRegression(penalty="l1", solver="liblinear", C=0.01)', "The lasso's penalty on a classifier, which sets coefficients to exactly zero. It needs a solver that can handle absolute values; the default lbfgs cannot.", "S8")
r("confusion_matrix(y, predicted).ravel()", "The 2 by 2 array flattened into tn, fp, fn, tp, in reading order. The four counts every other score is built from.", "S9")
r('LogisticRegression(class_weight="balanced")', "Count each class in inverse proportion to its size while fitting. Raises recall on a rare class and lowers accuracy, leaves the AUC alone, and breaks the probabilities.", "S9")
r("average_precision_score(y, p)", "The whole precision-recall curve in one number. Its no-information line is the share of ones, not 0.5, so report the two together. More informative than AUC when the class is rare.", "S9")
r("precision_recall_curve(y, p)", "Returns precision, recall and the thresholds, for drawing the curve or reading the precision at a recall you care about.", "S9")
r("brier_score_loss(y, p)", "The mean squared error of the probabilities, with what happened as 1 or 0. Smaller is better. It separates two models that the AUC cannot tell apart.", "S9")
r('CalibratedClassifierCV(model, method="sigmoid", cv=5)', "Wrap a model and learn a second, small model mapping its numbers onto probabilities, on folds. It repairs a distorted level; it cannot repair a base rate that has changed since fitting.", "S9")
r("classification_report(y, predicted, labels=order, digits=3)", "Precision, recall and F1 for every class, with the macro and weighted averages underneath. labels= sets the order.", "S9")
r('f1_score(y, predicted, average="macro")', 'One F1 per class, averaged equally. "weighted" averages by class size instead, and None returns the scores themselves. Read macro when the class you care about is the small one.', "S9")
r("confusion_matrix(y, predicted, labels=order)", "The matrix with the rows and columns in an order you choose rather than alphabetically. Without it the middle row of a three-class matrix is rarely the class you expect.", "S9")
r("model.decision_function(X)", "The raw score per class, before the exponential and the division that turn scores into probabilities.", "S9")
r("np.exp(scores) / np.exp(scores).sum()", "The softmax: make every score positive, then divide by the total so they add to one. With two classes it is the sigmoid.", "S9")
r("move[p >= t].sum() - move[p <= 1 - t].sum() - cost * n_trades", "A bet on the direction, in money: long where the probability of a rise is at least t, short where it is at most 1 - t, every trade paying a cost. The best t rises with the cost; choose it on the training rows.", "S10")

section("k-nearest neighbours",
        "A classifier with no coefficients: it stores the training rows and lets "
        "the nearest ones vote.",
        imports="from sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import GridSearchCV",
        names=SK_NAMES)
r("KNeighborsClassifier(n_neighbors=k)", "Classify a row by a vote among the k training rows closest to it. Works for any number of classes with no change.", "S9")
r('Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier())])', "The scaler is not optional here: distance is dominated by whichever column has the largest spread, so an unscaled model reads only that one.", "S9")
r("k, as the dial", "A small k fits the training rows and little else; k of 1 scores a perfect training AUC. A large k is a simple model, the way a small C was.", "S9")
r('{"knn__n_neighbors": [1, 5, 15, 51, 151, 301]}', "A grid for GridSearchCV. k cannot exceed the SMALLEST fold, not the training size; above that the search returns nan and warns.", "S9")
r("model.predict_proba(X)", "The share of the k neighbours in each class, so the probabilities are multiples of 1/k. A small k gives coarse ones.", "S9")
r('["wind_change", "weekday", "hour"]', "k-NN can take a category as one plain number: once the columns are scaled, the nearest rows to a Monday at 18:00 are Mondays near 18:00. A logistic regression needs the weekday as columns instead.", "S10")

# ======================================================================
section("Reading error messages",
        "The last line names the problem. Read it before you change anything: it is almost "
        "always telling you the truth.")
r("NameError", "A name Python has never seen. Usually a typo, or a cell you have not run yet.")
r("TypeError", "The right operation on the wrong kind of thing. Often text where a number belongs, or a placeholder ... left in a cell.")
r("IndexError", "A position that does not exist. A list of 5 stops at index 4.")
r("KeyError", "A dictionary key, or a column name, that is not there.", "S2")
r("SyntaxError", "Python could not read the line at all. Look for a missing bracket, quote or colon.")
r("ZeroDivisionError", "Dividing by zero. Often an empty list you thought had something in it.")
r("ModuleNotFoundError", "The package is not installed, or you are running a different Python.", "S3")
r("FileNotFoundError", "The path is wrong relative to where you are running from.", "S3")
r("ValueError, on shapes", "Two arrays whose dimensions do not fit. Check .shape on both.", "S3")
r("LinAlgError", "A matrix with no inverse: one column carries nothing the others do not. Perfect multicollinearity.", "S3")
r("ConvergenceWarning", "Not an error: the solver ran out of steps before the coefficients settled. Raise max_iter, or standardise the columns first.", "S6")

# ======================================================================
SINCE_LABEL = {"S1": "Session 1", "S2": "Session 2", "S3": "Session 3",
               "S4": "Session 4", "S5": "Session 5", "S6": "Session 6",
               "S8": "Session 8", "S9": "Session 9", "S10": "Session 10"}


# --------------------------------------------------------------- checks
def check_names():
    """Every bare function must be imported in its section or be a builtin,
    and every attribute written against a known object must exist."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    known = {"np": np, "pd": pd, "plt": plt, "ax": Axes, "fig": Figure,
             "s": pd.Series, "frame": pd.DataFrame, "g": pd.core.groupby.DataFrameGroupBy,
             "values": np.ndarray, "text": str, "items": list, "data": dict,
             "scaler": StandardScaler, "pipe": Pipeline, "search": GridSearchCV,
             "folds": TimeSeriesSplit}
    placeholders = {"name"}                  # def name(...): a name the reader chooses
    problems = []
    for sec in SECTIONS:
        imported = set(re.findall(r"\b([A-Za-z_]\w*)\b", sec["imports"] or ""))
        for row in sec["rows"]:
            for head in re.findall(r"(?<![\w.\"'])([A-Za-z_][\w.]*)\s*\(", row["call"]):
                parts = head.split(".")
                if len(parts) == 1:
                    if head not in imported and not hasattr(builtins, head) and head not in placeholders:
                        problems.append((sec["title"], row["call"], f"{head} is not imported in this section"))
                elif parts[0] in known:
                    obj = known[parts[0]]
                    for part in parts[1:]:
                        obj = getattr(obj, part, None)
                        if obj is None:
                            problems.append((sec["title"], row["call"], f"no such attribute: {head}"))
                            break
    return problems


problems = check_names()
if problems:
    print("NAME CHECK FAILED:")
    for p in problems:
        print("  ", *p, sep="  ")
    sys.exit(1)


# --------------------------------------------------------------- render
def slug(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


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
    <p class="lede">One row per function: the call, with its arguments, and what it does.
       Grouped by topic, with the imports a topic needs at the top of its card and the
       session that introduced each row at its edge.</p>
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
    if sec["imports"]:
        parts.append(f'    <pre class="cs-imports"><code>{html.escape(sec["imports"])}</code></pre>')
    if sec["names"]:
        parts.append(f'    <p class="cs-names">{html.escape(sec["names"])}</p>')
    parts.append('    <div class="cs-ref">')
    for row in sec["rows"]:
        # two spaces in a call separate alternatives: rendered as a faint dot
        call = ' <span class="cs-or">&middot;</span> '.join(html.escape(x) for x in re.split(r"  +", row["call"]))
        parts.append(f'      <div class="cs-sig"><code>{call}</code></div>')
        parts.append(f'      <div class="cs-desc">{html.escape(row["what"])}'
                     f'<span class="cs-since">{SINCE_LABEL[row["since"]]}</span></div>')
    parts.append('    </div>')
    parts.append("  </section>")

parts.append("""
  <footer>
    Built from the course materials, with every name checked against the library it
    comes from. Something missing? Email me at
    <a href="mailto:jobo@econ.au.dk">jobo@econ.au.dk</a>.
  </footer>
  </main>
</div>
</body>
</html>
""")

OUT.write_text("\n".join(parts), encoding="utf-8")
n_rows = sum(len(sec["rows"]) for sec in SECTIONS)
print(f"wrote {OUT.name}: {len(SECTIONS)} sections, {n_rows} rows, every name checked")
