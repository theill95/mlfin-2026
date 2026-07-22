# -*- coding: utf-8 -*-
"""Build session_02_exercises.ipynb.

Same conventions as Session 1 (approved 2026-07-20): pleasant intro, 1-5 star
badges, toolkit card with title= hover docs, task -> work cell (blank-safe
`...`) -> 1-2 folded hints -> folded solution, two-cell pattern for errors,
no em-dashes, plain academic tone.

Only tools taught in session_01.qmd + session_02.qmd. That means NO sorted,
NO //, and NO dict .keys()/.values()/.items() (the deck only taught looping
over a dictionary's keys).

BLANK-SAFE RULES that matter for this session's new constructs:
- while: the line that moves the condition towards False is always GIVEN, so a
  blank cell still terminates (never an infinite loop under Run all).
- dict: the final display is the whole dict or a key that already exists,
  never a key the student was asked to add (that would KeyError when blank).
- append: display the whole list, never an index into a possibly-empty list.
- functions: a blank body returns None, so the call result is displayed bare,
  never formatted or used in arithmetic.
"""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "session_02" / "session_02_exercises.ipynb"

cells = []
def badge(n, revisits=None):
    """Five unnamed stars, plus an optional note that this one reaches back.

    The scale restarts each session: it rates the work against what THIS
    session has taught, so a three-star task here is not the same size as a
    three-star task in a later notebook.
    """
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

# Exercises whose earlier-session tool is genuinely load-bearing, not incidental.
# Kept in one place so the tags can be read and revised as a set.
REVISITS = {
    "A4": "S1", "C4": "S1", "C5": "S1", "I4": "S1", "K5": "S1",
    "L7": "S1", "L8": "S1", "M1": "S1", "M2": "S1", "M3": "S1", "M4": "S1",
}


def ex(sid, title, n, task, work, hints, sol_code, sol_note, revisits=None):
    md(f"### {sid} · {title}  {badge(n, revisits or REVISITS.get(sid))}\n\n{task}")
    code(work)
    _hints_solution(hints, sol_code, sol_note)

def ex_fix(sid, title, n, task, demo_code, work, hints, sol_code, sol_note, raises=True, revisits=None):
    md(f"### {sid} · {title}  {badge(n, revisits)}\n\n{task}")
    code(demo_code, raises=raises)
    md("*Your fix:*")
    code(work)
    _hints_solution(hints, sol_code, sol_note)

def section(header):
    md(header)

def checkpoint(text):
    md(f"> ✅ **Checkpoint** · {text}")
    md("---")

# ---- numbers used in solution notes, computed here so the text is exact ----
WEEK = [183.56, 182.19, 179.87, 179.15, 183.48]
RET5 = [0.012, -0.004, 0.008, -0.011, 0.006]
def rets(p):
    return [(p[i] - p[i - 1]) / p[i - 1] for i in range(1, len(p))]
def mean(x):
    return sum(x) / len(x)
WEEK_R = rets(WEEK)

# compounded return over RET5
_v = 1.0
for _r in RET5:
    _v *= (1 + _r)
COMPOUNDED = _v - 1

# G4: years until a 10000 fund paying 500 a year, growing 4%, drops to 2000
def _drawdown_years():
    v, y = 10000.0, 0
    while v > 2000:
        v = v * 1.04 - 500
        y += 1
    return y
DD_YEARS = _drawdown_years()

# ---------------------------------------------------------------- top matter
md(
"# \U0001f4d3 Session 2 · Exercises\n"
"### Functions, Loops, and Dictionaries\n\n"
"Last session you did everything one value at a time. These exercises are about "
"getting the computer to do the repeating for you, and about packaging your work "
"so you can reuse it.\n\n"
"Most tasks ask you to write or complete a short piece of code. Work at your own "
"pace: they start easy and build up."
)

md(
"## How to use this notebook\n\n"
"- Each exercise has a **task**, then a **code cell** for your work. Cells with "
"`...` are blanks to fill in. Replace them with real code.\n"
"- Stuck? Open the **\U0001f4a1 Hint**, but only after a genuine attempt. Open the "
"**✅ Solution** to *check* yourself, not to skip the thinking.\n"
"- Every cell you fill in runs cleanly even with the blanks still in place, so "
"pressing **Run all** never floods you with errors. (A few cells in the "
"**Errors** section are broken on purpose so you can see a real error message. "
"They are clearly marked with ⚠️.)\n"
"- No files to download, and every exercise stands on its own. The last section "
"(**L**) is the one exception in spirit: it is about looking things up, and it "
"uses Python's own built-in documentation, which also works offline.\n\n"
"**You are not expected to finish all of these.** Do what you can. The rest is "
"here whenever you want to come back, and revising before the exam is a perfect "
"time. Short on time? Read the hint, then the solution. A worked solution you "
"genuinely understand is real learning too."
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
"The stars rate the work against **this** session. A three-star task in a later "
"notebook assumes everything before it, so it is a bigger piece of work than a "
"three-star task here, even though both sit in the middle of their own scale.\n\n"
"Some exercises also carry a **revisits** tag. Those need something from an "
"earlier session as well as today's material, and they are there on purpose: "
"the skills are meant to accumulate, not to be handed in at the end of each week."
)

md(
"## \U0001f9f0 Your toolkit for today\n\n"
"Everything you need is on this card, plus everything from Session 1. Sections **A** "
"to **K** ask for nothing beyond it.\n\n"
"> Section **L** deliberately goes past this card. Real work constantly needs a "
"function nobody taught you, so that section is about finding and reading the "
"documentation for tools this course never mentioned.\n\n"
"> The names inside the brackets (`item`, `items`, `value`, `prices`, ...) are "
"**placeholders**: you put your own value or variable there. Only the keyword, the "
"colon, the brackets and the indentation are fixed syntax. **Hover any tool** to "
"see what it does.\n\n"
'<p style="line-height:2.1"><strong>Repeating</strong><br>\n'
'<code style="cursor:help" title="Run the indented body once for every item in the list. item takes each value in turn.">for item in items:</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Run the body once for each whole number 0, 1, ... , n-1. Use it when you care about positions.">for i in range(n):</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The whole numbers from start up to but NOT including stop. range(1, 5) gives 1, 2, 3, 4.">range(start, stop)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Keep running the body for as long as the condition stays True. Something inside must change, or it never stops.">while condition:</code></p>\n\n'
'<p style="line-height:2.1"><strong>Choosing</strong><br>\n'
'<code style="cursor:help" title="Run the indented block only when the condition is True.">if condition:</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Checked only if the earlier conditions were False. Short for else-if. You can have several.">elif condition:</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Runs when none of the conditions above were True.">else:</code></p>\n\n'
'<p style="line-height:2.1"><strong>Updating a variable from itself</strong><br>\n'
'<code style="cursor:help" title="Add value to total and store it back. Exactly the same as total = total + value.">total += value</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Multiply value by factor and store it back. Same as value = value * factor.">value *= factor</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Subtract from the variable and store it back. Same as total = total - value.">total -= value</code></p>\n\n'
'<p style="line-height:2.1"><strong>Collecting results</strong><br>\n'
'<code style="cursor:help" title="A brand new list with nothing in it yet, ready to be filled by a loop.">results = []</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Add value to the end of the list. It changes the list itself, so there is no results = ... on the left.">results.append(value)</code></p>\n\n'
'<p style="line-height:2.1"><strong>Your own functions</strong><br>\n'
'<code style="cursor:help" title="Create a function called name that takes these parameters. The indented body is what it does.">def name(parameters):</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Hand a value back to whoever called the function, and stop there. Without it the function returns None.">return value</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="A default: if the caller does not pass rate, it is 0.05. Defaults come after the ordinary parameters.">def name(x, rate=0.05):</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="A docstring: a short description in triple quotes just under the def. Editors show it when you hover the function.">"""What it does."""</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Print a function\'s documentation, including its docstring.">help(name)</code></p>\n\n'
'<p style="line-height:2.1"><strong>Dictionaries</strong> &nbsp; values stored by name instead of position<br>\n'
'<code style="cursor:help" title="A dictionary literal: each key points to a value. Keys are usually strings such as a ticker.">{"AAPL": 0.36}</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Read the value stored under that key. Fails with a KeyError if the key is not there.">data[key]</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Store a value under that key. Adds the key if it is new, overwrites it if it already exists.">data[key] = value</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Loop over a dictionary and key takes each KEY in turn. Read its value with data[key].">for key in data:</code></p>\n\n'
'<p style="line-height:2.1"><strong>Still yours from Session 1</strong><br>\n'
'<code style="cursor:help" title="Show one or more values. Separate several with commas.">print(value)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="How many items are in the list.">len(items)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Add up every number in a list.">sum(numbers)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The largest / smallest of the values.">max(values)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The largest / smallest of the values.">min(values)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Round number to ndigits decimals.">round(number, ndigits)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The item at that position, counting from 0. Negative counts from the end, so [-1] is the last.">prices[i]</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Show r as a percentage with 2 decimals. 0.0231 becomes 2.31%.">f\"{r:.2%}\"</code></p>'
)

md(
"**Formulas you will reach for**\n\n"
r"| what | formula |"
"\n|:--|:--|\n"
r"| Simple return | $r=\dfrac{p_i-p_{i-1}}{p_{i-1}}$ |"
"\n"
r"| Average | $\bar{r}=\dfrac{1}{n}\sum_i r_i$ |"
"\n"
r"| Volatility | $\sigma=\sqrt{\dfrac{1}{n}\sum_i (r_i-\bar{r})^2}$ |"
"\n"
r"| Compound growth | $FV = P\,(1+r)^{n}$ |"
"\n"
)
md("---")

# ================================================================ A. For loops
section(
"## \U0001f501 A · For loops\n\n"
"*Do something once for every item in a list, instead of writing the same line "
"over and over.*\n\n"
"**Drills:** `for item in items:`, the loop variable, the indented body."
)

ex("A1", "Print every close", 1,
r"""Write a loop that prints **each** of these closing prices on its own line. Two lines of code, not
five `print` calls.""",
'''prices = [183.56, 182.19, 179.87, 179.15, 183.48]

# print each price, one per line:
...''',
"The shape is `for price in prices:` and then an **indented** `print(price)` underneath.",
'''prices = [183.56, 182.19, 179.87, 179.15, 183.48]

for price in prices:
    print(price)''',
"Five lines of output from two lines of code. Add a sixth price to the list and the loop handles it without any change, which is the whole point.")

ex("A2", "Name the loop variable yourself", 1,
r"""The loop variable is an ordinary variable and **you** choose its name. Loop over the tickers below
using a sensible name, and print each one.""",
'''tickers = ["AAPL", "KO", "NVDA", "JNJ"]

...''',
"`for ticker in tickers:` reads well. Any legal name works, but a good one makes the loop readable.",
'''tickers = ["AAPL", "KO", "NVDA", "JNJ"]

for ticker in tickers:
    print(ticker)''',
"`AAPL`, `KO`, `NVDA`, `JNJ`. Naming the loop variable for a single item (`ticker`) and the list for many (`tickers`) is a small habit that makes loops read like English.")

ex("A3", "Label each line", 2,
r"""Loop over the prices and print each one **with a label**, so the output reads:

```
close: 183.56
close: 182.19
```
""",
'''prices = [183.56, 182.19, 179.87]

...''',
'`print` takes several values separated by commas: `print("close:", price)`.',
'''prices = [183.56, 182.19, 179.87]

for price in prices:
    print("close:", price)''',
"Anything you can write in a normal `print` also works inside a loop. The body is just ordinary code that happens to run several times.")

ex("A4", "Format inside the loop", 2,
r"""These are daily returns as decimals. Loop over them and print each as a **percentage with two
decimals**, using the `:.2%` recipe from Session 1.""",
'''returns = [0.012, -0.004, 0.008]

...''',
'Combine the loop with an f-string: `print(f"{r:.2%}")`.',
'''returns = [0.012, -0.004, 0.008]

for r in returns:
    print(f"{r:.2%}")''',
f"`{RET5[0]:.2%}`, `{RET5[1]:.2%}`, `{RET5[2]:.2%}`. Formatting is per item, so every value comes out tidy without you touching them one at a time.")

ex("A5", "Two lines in the body", 2,
r"""A loop body can be as long as you like, as long as every line is indented the same amount. For each
price, print the price **and** what it would be after a 1% rise, on two separate lines.""",
'''prices = [100.0, 200.0]

for price in prices:
    ...
    ...''',
["Both lines go inside the loop, indented equally. The second one prints `price * 1.01`.",
 'Line one: `print(price)`. Line two: `print(price * 1.01)`.'],
'''prices = [100.0, 200.0]

for price in prices:
    print(price)
    print(price * 1.01)''',
"`100.0`, `101.0`, `200.0`, `202.0`. The whole indented block runs once per item, top to bottom, before the loop moves on to the next value.")

checkpoint("you can run a block of code once for every item in a list.")

# ================================================================ B. Accumulator
section(
"## \U0001f9ee B · Building up a result\n\n"
"*Start a variable before the loop, update it inside, and read it after. This "
"one pattern covers summing, counting, and averaging.*\n\n"
"**Drills:** the accumulator pattern, `+=`, `*=`."
)

ex("B1", "Add up a week by hand", 2,
r"""Compute the total of these prices **with a loop**, not with `sum`. Start `total` at zero, then add
each price to it inside the loop.""",
'''prices = [183.56, 182.19, 179.87, 179.15, 183.48]

total = ...
for price in prices:
    ...
print(total)''',
["Start with `total = 0`. Inside the loop, `total = total + price`.",
 "Or use the shorthand from the lecture: `total += price`."],
'''prices = [183.56, 182.19, 179.87, 179.15, 183.48]

total = 0
for price in prices:
    total += price
print(total)''',
f"`{sum(WEEK)}`, the same answer `sum(prices)` gives. Now you have seen what `sum` does inside: set up, add once per item, read the result afterwards.")

ex("B2", "Count without len", 2,
r"""Count how many prices are in the list **without** using `len`. The idea is identical to summing, but
you add **1** each time round instead of the value.""",
'''prices = [183.56, 182.19, 179.87, 179.15, 183.48]

count = ...
for price in prices:
    ...
print(count)''',
"Start `count = 0`, then `count += 1` inside the loop. The loop variable is not even used.",
'''prices = [183.56, 182.19, 179.87, 179.15, 183.48]

count = 0
for price in prices:
    count += 1
print(count)''',
f"`{len(WEEK)}`. Counting is the accumulator pattern with a step of 1. Notice `price` is never read: sometimes you only care that the loop went round, not what it held.")

ex("B3", "The average, from scratch", 3,
r"""Combine the two: build the total with a loop, then divide by how many there are, to get the average
price.

$$\bar{p}=\dfrac{1}{n}\sum_i p_i$$""",
'''prices = [183.56, 182.19, 179.87, 179.15, 183.48]

total = ...
for price in prices:
    ...
average = ...
print(average)''',
["Build `total` exactly as in B1.",
 "After the loop (not indented), `average = total / len(prices)`."],
'''prices = [183.56, 182.19, 179.87, 179.15, 183.48]

total = 0
for price in prices:
    total += price
average = total / len(prices)
print(average)''',
f"`{mean(WEEK)}`. The division sits **after** the loop and is not indented, because you average once at the end, not on every pass. Putting it inside the loop is one of the most common beginner bugs.")

ex("B4", "Add up the returns", 2,
r"""Same pattern, now on returns rather than prices. Add all five daily returns into `total`, then show
the result as a percentage.""",
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]

total = ...
for r in returns:
    ...
print(total)''',
["`total = 0`, then `total += r` inside the loop.",
 'To show it as a percentage, replace the last line with `print(f"{total:.2%}")`.'],
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]

total = 0
for r in returns:
    total += r
print(f"{total:.2%}")''',
f"`{sum(RET5):.2%}`. Adding returns like this is a rough approximation. Strictly they compound, which is the next exercise.")

ex("B5", "Compounding, one day at a time", 3,
r"""Returns do not add, they **compound**: each day multiplies what you had by $(1+r)$. Start a
`value` at **1.0** and multiply it by `(1 + r)` once per return, then subtract 1 to get the total
return over the week.

$$r_\text{total} = \prod_i (1+r_i) - 1$$""",
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]

value = ...
for r in returns:
    ...
compounded = ...
print(compounded)''',
["Start at `value = 1.0` (the neutral value for multiplying, the way 0 is for adding).",
 "Inside: `value *= (1 + r)`. After the loop: `compounded = value - 1`."],
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]

value = 1.0
for r in returns:
    value *= (1 + r)
compounded = value - 1
print(compounded)''',
f"`{COMPOUNDED}`, that is **{COMPOUNDED:.4%}**, against the naive sum of {sum(RET5):.4%}. The accumulator starts at 1 and uses `*=` because the neutral value for multiplication is one, not zero.")

checkpoint("you can set up a variable, update it once per item, and read the result after the loop.")

# ================================================================ C. range
section(
"## \U0001f522 C · Positions, with range\n\n"
"*Sometimes you need the position, not just the value. That is what `range` is "
"for, and it is how you compare each day with the one before it.*\n\n"
"**Drills:** `range(n)`, `range(start, stop)`, indexing with `i` and `i - 1`."
)

ex("C1", "Count to four", 1,
r"""Use `range` to print the numbers **0, 1, 2, 3**, one per line.""",
'''...''',
"`range(4)` gives 0, 1, 2, 3. It stops *before* the number you give it.",
'''for i in range(4):
    print(i)''',
"`0`, `1`, `2`, `3`. `range(n)` always starts at 0 and stops one short of `n`, which is exactly how list positions are numbered.")

ex("C2", "Start somewhere else", 1,
r"""Print the numbers **1, 2, 3, 4** using the two-argument form of `range`.""",
'''...''',
"`range(start, stop)` begins at `start` and stops before `stop`, so you want `range(1, 5)`.",
'''for i in range(1, 5):
    print(i)''',
"`1`, `2`, `3`, `4`. Just like slicing, the stop value is not included, so `range(1, 5)` gives four numbers and not five.")

ex("C3", "Number each day", 2,
r"""Print each price **with its position**, so the output reads:

```
0 183.56
1 182.19
2 179.87
```

Loop over the positions and use each one to look up the price.""",
'''prices = [183.56, 182.19, 179.87]

...''',
["Loop with `for i in range(len(prices)):` so `i` takes each valid position.",
 "Inside, `print(i, prices[i])`."],
'''prices = [183.56, 182.19, 179.87]

for i in range(len(prices)):
    print(i, prices[i])''',
"`range(len(prices))` is the standard way to walk every valid position of a list: it gives 0 up to one less than the length, which is exactly the set of positions that exist.")

ex("C4", "Compare each day with the day before", 3,
r"""To compare consecutive days you need **two** prices at once, `prices[i]` and `prices[i - 1]`. Print
each day's **change** (today minus yesterday).

Start the range at **1**, because day 0 has no day before it.""",
'''prices = [100.0, 102.0, 101.0, 105.0]

for i in range(1, len(prices)):
    ...''',
["The change is `prices[i] - prices[i - 1]`.",
 "Print it inside the loop: `print(prices[i] - prices[i - 1])`."],
'''prices = [100.0, 102.0, 101.0, 105.0]

for i in range(1, len(prices)):
    print(prices[i] - prices[i - 1])''',
"`2.0`, `-1.0`, `4.0`. Four prices give three changes. Starting at 1 is what keeps `prices[i - 1]` valid: at `i = 0` it would be `prices[-1]`, the *last* price, and quietly give a wrong answer.")

ex("C5", "Every daily return", 4,
r"""The central skill of this session. Turn a list of prices into a list of **daily returns**, using a
loop over positions and the simple-return formula:

$$r_i=\dfrac{p_i-p_{i-1}}{p_{i-1}}$$

Collect them into `returns` with `.append(...)`.""",
'''prices = [183.56, 182.19, 179.87, 179.15, 183.48]

returns = ...
for i in range(1, len(prices)):
    ...
print(returns)''',
["Start with an empty list: `returns = []`. Loop from 1 as in C4.",
 "Inside: `r = (prices[i] - prices[i - 1]) / prices[i - 1]`, then `returns.append(r)`."],
'''prices = [183.56, 182.19, 179.87, 179.15, 183.48]

returns = []
for i in range(1, len(prices)):
    r = (prices[i] - prices[i - 1]) / prices[i - 1]
    returns.append(r)
print(returns)''',
f"Five prices give **four** returns: {[round(x, 6) for x in WEEK_R]}. This exact loop is the foundation of the case, and of everything the rest of the course does with prices.")

checkpoint("you can loop over positions and reach two days at once, which is what turns prices into returns.")

# ================================================================ D. Lists
section(
"## \U0001f4e5 D · Collecting results\n\n"
"*Start from an empty list and grow it as the loop runs.*\n\n"
"**Drills:** `results = []`, `.append(...)` inside a loop."
)

ex("D1", "An empty list, filled by hand", 1,
r"""Create an **empty** list called `collected`, then append the values **10** and **20** to it.""",
'''collected = ...
...
...
print(collected)''',
"An empty list is written `[]`. Then `collected.append(10)` and `collected.append(20)`.",
'''collected = []
collected.append(10)
collected.append(20)
print(collected)''',
"`[10, 20]`. Note there is no `collected = collected.append(...)`: `.append` changes the list in place and hands nothing back.")

ex("D2", "Build a list with a loop", 2,
r"""Start from an empty list and use a loop to collect **each price doubled**.""",
'''prices = [10.0, 20.0, 30.0]

doubled = ...
for price in prices:
    ...
print(doubled)''',
["`doubled = []` before the loop.",
 "Inside the loop: `doubled.append(price * 2)`."],
'''prices = [10.0, 20.0, 30.0]

doubled = []
for price in prices:
    doubled.append(price * 2)
print(doubled)''',
"`[20.0, 40.0, 60.0]`. The empty list has to be created **before** the loop. Putting `doubled = []` inside would wipe it clean on every pass and leave you with one item.")

ex("D3", "Growth factors", 2,
r"""Given daily returns, build a list of the matching **growth factors** $(1+r)$, which is what you
multiply a price by to move it one day forward.""",
'''returns = [0.012, -0.004, 0.008]

factors = ...
for r in returns:
    ...
print(factors)''',
"`factors = []`, then inside the loop `factors.append(1 + r)`.",
'''returns = [0.012, -0.004, 0.008]

factors = []
for r in returns:
    factors.append(1 + r)
print(factors)''',
f"`{[1 + x for x in RET5[:3]]}`. A factor above 1 is an up day and below 1 is a down day. This is the shape of most data work: take one list, produce another with a value derived from each item.")

checkpoint("you can create an empty list and fill it as a loop runs.")

# ================================================================ E. if
section(
"## \U0001f500 E · Making choices\n\n"
"*Run a block only when a condition holds.*\n\n"
"**Drills:** `if`, `else`, `elif`, conditions on numbers."
)

ex("E1", "Only when it is true", 1,
r"""Print `"above 250"` only if `price` is greater than **250**. Then change `price` to `240`, run it
again, and confirm that nothing is printed.""",
'''price = 260

...''',
"`if price > 250:` and then an indented `print(\"above 250\")`.",
'''price = 260

if price > 250:
    print("above 250")''',
"`above 250`. With `price = 240` the condition is `False`, the indented block is skipped, and the cell produces no output at all. Nothing breaks; the block simply does not run.")

ex("E2", "The other case too", 2,
r"""Extend it: print `"above 250"` when the price is above 250, and `"250 or below"` otherwise.""",
'''price = 240

...''',
"Add an `else:` branch under the `if`, with its own indented `print`.",
'''price = 240

if price > 250:
    print("above 250")
else:
    print("250 or below")''',
"`250 or below`. Exactly one of the two branches runs, always. `else` needs no condition of its own: it catches everything the `if` did not.")

ex("E3", "Up, down, or flat", 2,
r"""A return can be positive, negative, or exactly zero. Print `"up day"`, `"down day"` or `"flat day"`
for the value of `r`, using `if` / `elif` / `else`.""",
'''r = -0.02

...''',
["Three branches: `if r > 0:`, then `elif r < 0:`, then `else:`.",
 "The `else` catches exactly zero, so it needs no condition."],
'''r = -0.02

if r > 0:
    print("up day")
elif r < 0:
    print("down day")
else:
    print("flat day")''',
"`down day`. The conditions are checked **in order** and Python stops at the first true one. Because `> 0` and `< 0` are already taken, `else` can only mean exactly zero.")

ex("E4", "How big was the move?", 3,
r"""Classify a daily return by size. Print `"large move"` if it is **above 2%** or **below -2%**,
and `"normal move"` otherwise.

You have not been shown a way to say "or", so think about what a large move means regardless of
direction, and reach for a Session 1 tool that removes the sign.""",
'''r = -0.031

...''',
["A move of -3.1% and a move of +3.1% are both large. What single number describes the *size* of a move whatever its direction?",
 "`max(r, -r)` is the size of the move, ignoring the sign. Compare that with 0.02."],
'''r = -0.031

size = max(r, -r)
if size > 0.02:
    print("large move")
else:
    print("normal move")''',
"`large move`. `max(r, -r)` is the distance of `r` from zero, so one comparison covers both directions. Python does have a built-in for this (`abs`), and this is a fair way to get there with only the tools you have been given.")

checkpoint("you can branch on a condition, and chain several conditions in order.")

# ================================================================ F. if in loop
section(
"## \U0001f50e F · Choosing, inside a loop\n\n"
"*The combination that does real work: check a condition for every item.*\n\n"
"**Drills:** counting matches, summing a subset, collecting a subset."
)

ex("F1", "Count the up days", 2,
r"""Count how many of these returns are **positive**. Put an `if` inside the loop and increase the
counter only when the condition holds.""",
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]

up_days = ...
for r in returns:
    ...
print(up_days)''',
["`up_days = 0` before the loop, and `if r > 0:` inside it.",
 "The `up_days += 1` is indented **twice**: once for the loop, once for the `if`."],
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]

up_days = 0
for r in returns:
    if r > 0:
        up_days += 1
print(up_days)''',
f"`{sum(1 for x in RET5 if x > 0)}`. The two levels of indentation say it exactly: for every return, and only if it is positive, add one. Counting how often something happens is the most common thing you will do with a condition in a loop.")

ex("F2", "Add up only the gains", 2,
r"""Sum only the **positive** returns, ignoring the down days entirely.""",
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]

gains = ...
for r in returns:
    ...
print(gains)''',
["Same shape as F1, but add `r` rather than 1.",
 "`if r > 0:` then `gains += r`."],
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]

gains = 0
for r in returns:
    if r > 0:
        gains += r
print(gains)''',
f"`{sum(x for x in RET5 if x > 0)}`. Only the three up days contributed. Swapping `+= 1` for `+= r` turns a count into a conditional sum, and the rest of the structure is untouched.")

ex("F3", "Keep the bad days", 2,
r"""Collect all the **negative** returns into a new list called `losses`.""",
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]

losses = ...
for r in returns:
    ...
print(losses)''',
["Start `losses = []`, and put an `if r < 0:` inside the loop.",
 "Inside the `if`, `losses.append(r)`."],
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]

losses = []
for r in returns:
    if r < 0:
        losses.append(r)
print(losses)''',
f"`{[x for x in RET5 if x < 0]}`. Loop, test, append: this is filtering, and it is one line of pandas in Session 3. Doing it by hand once makes that line make sense.")

ex("F4", "How many big days?", 3,
r"""Count how many daily returns were **larger than 1%** in size, in either direction. Reuse the
sign-removing trick from E4.""",
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]

big_days = ...
for r in returns:
    ...
print(big_days)''',
["The size of a move, ignoring direction, is `max(r, -r)`.",
 "`if max(r, -r) > 0.01:` then `big_days += 1`."],
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]

big_days = 0
for r in returns:
    if max(r, -r) > 0.01:
        big_days += 1
print(big_days)''',
f"`{sum(1 for x in RET5 if max(x, -x) > 0.01)}`: the +1.2% day and the -1.1% day. Counting how often a series moves more than some threshold is a genuine, if crude, way to describe risk.")

ex("F5", "The worst day, and when it happened", 4,
r"""Find the **worst** return and the **position** it sits at, using only a loop and an `if`. No `min`,
no `.index`.

Track two things as you go: the worst value seen so far, and where you saw it. Start the worst value
at the **first** return, so you are always comparing against something real.""",
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]

worst = ...
worst_at = ...
for i in range(len(returns)):
    ...
print(worst, worst_at)''',
["Start `worst = returns[0]` and `worst_at = 0`, then walk every position with `for i in range(len(returns)):`.",
 "Inside: `if returns[i] < worst:` then update **both** `worst = returns[i]` and `worst_at = i`."],
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]

worst = returns[0]
worst_at = 0
for i in range(len(returns)):
    if returns[i] < worst:
        worst = returns[i]
        worst_at = i
print(worst, worst_at)''',
f"`{min(RET5)} {RET5.index(min(RET5))}`. This is `min` and `.index` in one pass, written out. Keeping a best-so-far and updating it when you find something better is a pattern worth recognising: it is how you find an extreme in data too large to sort.")

checkpoint("you can count, sum, and filter a list according to a condition.")

# ================================================================ G. while
section(
"## ⏳ G · While loops\n\n"
"*Repeat for as long as a condition holds, when you do not know in advance how "
"many rounds it will take.*\n\n"
"**Drills:** `while`, counting the rounds, making sure the loop can stop.\n\n"
"> In each of these the line that changes the value is already written for you, "
"so the loop always stops. Your job is to count the rounds."
)

ex("G1", "How long to reach 1500?", 3,
r"""**1000** grows at **8%** a year. Count how many whole years it takes to reach **1500**. The growth
line is given; add the line that counts the year.""",
'''value = 1000
years = 0
while value < 1500:
    value *= 1.08
    ...
print(years)''',
"Count one year on every pass: `years += 1`, indented inside the loop.",
'''value = 1000
years = 0
while value < 1500:
    value *= 1.08
    years += 1
print(years)''',
"`6`. After five years the value is still below 1500 and the sixth pass pushes it over. You never had to know the answer in advance, which is exactly when a `while` beats a `for`.")

ex("G2", "Halving", 2,
r"""A position worth **1000** is halved again and again. Count how many halvings it takes to fall
**below 1**.""",
'''value = 1000.0
halvings = 0
while value >= 1:
    value = value / 2
    ...
print(halvings)''',
"Same shape as G1: `halvings += 1` inside the loop.",
'''value = 1000.0
halvings = 0
while value >= 1:
    value = value / 2
    halvings += 1
print(halvings)''',
"`10`. After nine halvings the value is about 1.95, still at or above 1, and the tenth takes it to about 0.98. The condition is tested **before** each pass, so the loop stops the moment it is no longer true.")

ex("G3", "Doubling your money", 2,
r"""At **7%** a year, how many whole years until **100** becomes **200**?""",
'''value = 100.0
years = 0
while value < 200:
    value *= 1.07
    ...
print(years)''',
"`years += 1` inside the loop, exactly as before.",
'''value = 100.0
years = 0
while value < 200:
    value *= 1.07
    years += 1
print(years)''',
"`11`. The familiar rule of thumb divides 72 by the rate, giving about 10.3 years, and the honest count is 11 whole years. A short loop settles a question that would otherwise need logarithms.")

ex("G4", "Drawing down a fund", 3,
r"""A fund holds **10000**, grows **4%** a year, and pays out **500** at the end of each year. Count how
many years until it falls to **2000** or below.

The line that updates the balance is given. Note that it both grows and pays out in one step.""",
'''value = 10000.0
years = 0
while value > 2000:
    value = value * 1.04 - 500
    ...
print(years)''',
["`years += 1` inside the loop.",
 "The balance falls because the 500 payout is larger than 4% of the balance, so the loop is guaranteed to end."],
'''value = 10000.0
years = 0
while value > 2000:
    value = value * 1.04 - 500
    years += 1
print(years)''',
f"`{DD_YEARS}` years. Each pass applies the growth and the payout together, and because the payout outweighs the growth the balance falls a little further every year until the condition finally fails.")

checkpoint("you can repeat until a condition is met, and count how many rounds it took.")

# ================================================================ H. Functions
section(
"## \U0001f6e0️ H · Your own functions\n\n"
"*Name a calculation once, then reuse it anywhere. This is how code stops being "
"a pile of one-off cells.*\n\n"
"**Drills:** `def` and `return`, parameters, defaults, docstrings, a function "
"that contains a loop."
)

ex("H1", "Your first function", 1,
r"""Write a function `double(x)` that **returns** twice its input, then call it on **21**.""",
'''def double(x):
    ...

result = double(21)
print(result)''',
"The body is a single line: `return x * 2`.",
'''def double(x):
    return x * 2

result = double(21)
print(result)''',
"`42`. `def` names the calculation, `x` stands for whatever you pass in, and `return` hands the answer back so you can store it.")

ex("H2", "The return formula, named", 2,
r"""Write `simple_return(p0, p1)` that returns the simple return from `p0` to `p1`:

$$r=\dfrac{p_1-p_0}{p_0}$$

Then call it for a price that moved from **183.56** to **182.19**.""",
'''def simple_return(p0, p1):
    ...

r = simple_return(183.56, 182.19)
print(r)''',
["The body is one line: `return (p1 - p0) / p0`.",
 "The order matters: the *later* price minus the earlier one, divided by the earlier one."],
'''def simple_return(p0, p1):
    return (p1 - p0) / p0

r = simple_return(183.56, 182.19)
print(r)''',
f"`{(182.19 - 183.56) / 183.56}`, that is {((182.19 - 183.56) / 183.56):.2%}. Write the formula once and you can stop worrying about getting it wrong: every later call reuses the same tested line.")

ex_fix("H3", "It computes, but hands nothing back", 2,
r"""⚠️ **Run the cell below.** It does not error, but `x` comes out as `None`, because the function
**prints** its answer instead of **returning** it. Look at the output, then rewrite the function
underneath so that `x` really holds the number.""",
'''def show(p0, p1):
    print((p1 - p0) / p0)

x = show(100, 110)
print("x is:", x)''',
'''def give(p0, p1):
    ...

x = give(100, 110)
print("x is:", x)''',
["`print` shows a value on screen; `return` hands it back to the caller. They are not the same thing.",
 "Replace the `print(...)` line with `return (p1 - p0) / p0`."],
'''def give(p0, p1):
    return (p1 - p0) / p0

x = give(100, 110)
print("x is:", x)''',
"`x is: 0.1`. A function without `return` gives back `None`, so the value is displayed once and then lost. If you want to keep a result, you must return it.",
raises=False)

ex("H4", "A default rate", 2,
r"""Write `grow(price, rate=0.05)` that returns the price after one year of growth, using **5%** unless
a different rate is passed. Then call it **both** ways.""",
'''def grow(price, rate=0.05):
    ...

print(grow(100))
print(grow(100, 0.10))''',
"The body is `return price * (1 + rate)`. The default is already written in the `def` line.",
'''def grow(price, rate=0.05):
    return price * (1 + rate)

print(grow(100))
print(grow(100, 0.10))''',
"`105.0` and `110.00000000000001`. The default makes the common case short while leaving the parameter available when you need it. (The trailing digits are ordinary floating-point dust; `round(..., 2)` tidies them.)")

ex("H5", "Document it", 1,
r"""Give `simple_return` a **docstring**: a one-line description in triple quotes, on the first line of
the body. Then run `help(simple_return)` to read it back.""",
'''def simple_return(p0, p1):
    ...
    return (p1 - p0) / p0

help(simple_return)''',
'A docstring is just a string on its own line: `"""Percentage change in price from p0 to p1."""`',
'''def simple_return(p0, p1):
    """Percentage change in price from p0 to p1."""
    return (p1 - p0) / p0

help(simple_return)''',
"`help` prints the function's name, its parameters, and your sentence. Your editor shows the same text when you hover the function, so one line written now saves you re-reading the body later.")

ex("H6", "A function with a loop inside", 3,
r"""Package the accumulator pattern: write `average(values)` that returns the mean of a list, using a
loop inside the function.

$$\bar{v}=\dfrac{1}{n}\sum_i v_i$$""",
'''def average(values):
    ...

print(average([10, 20, 30]))''',
["Inside the function: start `total = 0`, loop `for v in values:` adding each one.",
 "Then `return total / len(values)`, outside the loop but inside the function."],
'''def average(values):
    total = 0
    for v in values:
        total += v
    return total / len(values)

print(average([10, 20, 30]))''',
"`20.0`. Four lines of work now have one name. Watch the indentation carefully: the loop is indented inside the function, and the `return` is indented inside the function but **not** inside the loop, so it runs once at the end.")

ex_fix("H7", "Not enough arguments", 2,
r"""⚠️ **Run the cell below.** `simple_return` needs **two** prices but is given only one, and it fails
with a `TypeError`. Read the message, then call it correctly underneath for a move from **100** to
**110**.""",
'''def simple_return(p0, p1):
    return (p1 - p0) / p0

simple_return(100)''',
'''def simple_return(p0, p1):
    return (p1 - p0) / p0

r = ...
print(r)''',
["The message names the parameter Python never received: `p1`.",
 "Pass both arguments: `r = simple_return(100, 110)`."],
'''def simple_return(p0, p1):
    return (p1 - p0) / p0

r = simple_return(100, 110)
print(r)''',
"`0.1`. Python will not guess a missing argument. If you want one to be optional, give it a default in the `def`, as in H4.")

checkpoint("you can define a function, return a value, give a parameter a default, document it, and put a loop inside it.")

# ================================================================ I. Dicts
section(
"## \U0001f5c2️ I · Dictionaries\n\n"
"*Store values under a name you choose, instead of a position you have to "
"remember.*\n\n"
"**Drills:** creating, looking up, adding, updating, looping over keys."
)

ex("I1", "Two stocks, by name", 1,
r"""Build a dictionary called `year_return` holding **AAPL: 0.36** and **KO: 0.07**, then display it.""",
'''year_return = ...
print(year_return)''',
'Curly braces, with `key: value` pairs separated by commas: `{"AAPL": 0.36, "KO": 0.07}`.',
'''year_return = {"AAPL": 0.36, "KO": 0.07}
print(year_return)''',
"`{'AAPL': 0.36, 'KO': 0.07}`. The keys are strings, so they need quotes. The values here are numbers, but they can be anything.")

ex("I2", "Look one up", 1,
r"""Read Coca-Cola's return out of the dictionary using its **key**, and store it in `ko`.""",
'''year_return = {"AAPL": 0.36, "KO": 0.07}

ko = ...
print(ko)''',
'Square brackets, but with the key instead of a position: `year_return["KO"]`.',
'''year_return = {"AAPL": 0.36, "KO": 0.07}

ko = year_return["KO"]
print(ko)''',
"`0.07`. The brackets look like list indexing, but the thing inside is a name you chose rather than a position you have to count. `year_return[\"KO\"]` says what it means.")

ex("I3", "Add one, change one", 2,
r"""Add **NVDA** with a return of **1.79** to the dictionary, and at the same time **correct** Apple's
figure to **0.3556**. Both are done by assigning to a key.""",
'''year_return = {"AAPL": 0.36, "KO": 0.07}

...
...
print(year_return)''',
["Assigning to a key that does not exist **adds** it; assigning to one that does **replaces** it.",
 '`year_return["NVDA"] = 1.79` and `year_return["AAPL"] = 0.3556`.'],
'''year_return = {"AAPL": 0.36, "KO": 0.07}

year_return["NVDA"] = 1.79
year_return["AAPL"] = 0.3556
print(year_return)''',
"`{'AAPL': 0.3556, 'KO': 0.07, 'NVDA': 1.79}`. One piece of syntax does both jobs. There is no separate 'add' and 'update', which is convenient but does mean a typo in a key quietly creates a new entry instead of correcting an old one.")

ex("I4", "Report every stock", 3,
r"""Loop over the dictionary and print each ticker together with its return as a percentage, so the
output reads:

```
AAPL 35.56%
KO 7.26%
```
""",
'''year_return = {"AAPL": 0.3556, "KO": 0.0726}

...''',
["`for ticker in year_return:` gives you each **key** in turn.",
 'Read the value with `year_return[ticker]`, then `print(ticker, f"{year_return[ticker]:.2%}")`.'],
'''year_return = {"AAPL": 0.3556, "KO": 0.0726}

for ticker in year_return:
    print(ticker, f"{year_return[ticker]:.2%}")''',
"`AAPL 35.56%` and `KO 7.26%`. Looping a dictionary gives you the **keys**, not the values, so you use the key to look the value up. That is the one thing people forget on their first try.")

ex("I5", "A dictionary of price series", 2,
r"""A dictionary's values can be whole lists. Build `prices` holding a short series for each stock, then
print **only Coca-Cola's** list.""",
'''prices = ...

# now print just Coca-Cola's list:
...''',
["The value after each key is a full list in square brackets.",
 '`prices = {"AAPL": [183.56, 182.19, 179.87], "KO": [59.20, 59.05, 59.40]}`, then `print(prices["KO"])`.'],
'''prices = {
    "AAPL": [183.56, 182.19, 179.87],
    "KO": [59.20, 59.05, 59.40],
}
print(prices["KO"])''',
"`[59.2, 59.05, 59.4]`. One object now holds every stock's history, each under its own ticker. This is the structure the case uses to handle several stocks at once.")

checkpoint("you can store values by name, look them up, update them, and walk a whole dictionary.")

# ================================================================ J. Errors
section(
"## \U0001f41e J · Reading errors\n\n"
"*Each cell here breaks on purpose, so you meet this session's errors somewhere "
"safe and learn to repair them.*\n\n"
"**Drills:** `IndentationError`, `SyntaxError`, `KeyError`, `IndexError`."
)

ex_fix("J1", "Nothing to run", 2,
r"""⚠️ **Run the cell below.** It fails with an **IndentationError**: the `print` after the colon was
never indented, so the loop has no body. Read the message, then write it correctly underneath.""",
'''prices = [10, 20, 30]
for price in prices:
print(price)''',
'''prices = [10, 20, 30]

# write the loop correctly:
...''',
["A colon at the end of a line promises an indented block underneath it.",
 "Indent the `print(price)` by four spaces so it sits inside the loop."],
'''prices = [10, 20, 30]

for price in prices:
    print(price)''',
"`10`, `20`, `30`. Indentation is not decoration in Python: it is the only thing that says which lines are inside the loop. Any consistent amount works, and four spaces is the convention.")

ex_fix("J2", "A missing colon", 1,
r"""⚠️ **Run the cell below.** It fails with a **SyntaxError** before it runs at all, because the `for`
line is missing its colon. Read the message, then write the corrected loop underneath so it prints the
running total.""",
'''total = 0
for r in [0.01, 0.02]
    total += r
print(total)''',
'''total = 0

# write the loop correctly:
...
print(total)''',
"Every `for`, `while`, `if` and `def` line ends with a colon. Add it and the block is well-formed.",
'''total = 0

for r in [0.01, 0.02]:
    total += r
print(total)''',
"`0.03`. A `SyntaxError` is about shape rather than meaning: Python cannot even begin until the line is written legally. The colon is the single most commonly forgotten character in Python.")

ex_fix("J3", "No such key", 2,
r"""⚠️ **Run the cell below.** It fails with a **KeyError**, because there is no `"MSFT"` in the
dictionary. Read the message, then fix it underneath: add Microsoft with a volatility of **0.0125**,
and then read it back.""",
'''vol = {"AAPL": 0.0141, "KO": 0.0080}
vol["MSFT"]''',
'''vol = {"AAPL": 0.0141, "KO": 0.0080}

...
print(vol)''',
["A `KeyError` means you asked for a key the dictionary does not have. Check your spelling first, then whether it was ever added.",
 '`vol["MSFT"] = 0.0125` adds it. After that, `vol["MSFT"]` works.'],
'''vol = {"AAPL": 0.0141, "KO": 0.0080}

vol["MSFT"] = 0.0125
print(vol)''',
"`{'AAPL': 0.0141, 'KO': 0.008, 'MSFT': 0.0125}`. Reading a missing key is an error, but *writing* to one is not: it simply creates the entry. That asymmetry is worth remembering, because a mistyped key never warns you on the way in.")

ex_fix("J4", "One step past the end", 3,
r"""⚠️ **Run the cell below.** It fails with an **IndexError**: the range runs one position too far, so
the last pass asks for a price that does not exist. Read the message, then write the corrected loop
underneath so it prints all three prices.""",
'''prices = [100, 102, 101]
for i in range(1, len(prices) + 1):
    print(prices[i])''',
'''prices = [100, 102, 101]

# write the loop correctly, printing every price:
...''',
["Three prices sit at positions 0, 1 and 2. The `+ 1` pushes the last pass to position 3, which does not exist.",
 "To visit every position, use `range(len(prices))`, which gives 0, 1, 2."],
'''prices = [100, 102, 101]

for i in range(len(prices)):
    print(prices[i])''',
"`100`, `102`, `101`. `range(len(prices))` is the safe way to walk every position, because it stops one short of the length by design. The only time you start at 1 is when you deliberately need `prices[i - 1]` as well, as in C4.")

checkpoint("you can read this session's error messages and repair the cause.")

# ================================================================ K. Together
section(
"## \U0001f4c8 K · Putting it together\n\n"
"*Loops, conditions, functions and dictionaries aimed at real analysis. This is "
"the work the case will ask of you.*\n\n"
"**Drills:** statistics written as functions, applying one function to many "
"stocks."
)

ex("K1", "An average you can reuse", 2,
r"""Write `mean(values)` returning the average of a list, then use it on the returns below. You will
want this function again in the case.""",
'''def mean(values):
    ...

returns = [0.012, -0.004, 0.008, -0.011, 0.006]
m = mean(returns)
print(m)''',
["Inside the function, `sum(values) / len(values)` is enough for a one-liner.",
 "Or build the total with a loop, as in H6. Both are correct."],
'''def mean(values):
    return sum(values) / len(values)

returns = [0.012, -0.004, 0.008, -0.011, 0.006]
m = mean(returns)
print(m)''',
f"`{mean(RET5)}`, the average daily return, about {mean(RET5):.3%} a day. Small and named, this is the kind of function you write once and lean on constantly.")

ex("K2", "Distances from the average", 4,
r"""Volatility is built from how far each return sits from the average. Compute the **sum of squared
distances** from the mean:

$$\sum_i (r_i-\bar{r})^2$$

The mean is given. Accumulate the squares with a loop.""",
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]
m = sum(returns) / len(returns)

squared_total = ...
for r in returns:
    ...
print(squared_total)''',
["Start `squared_total = 0`, then add one squared distance per return.",
 "The distance is `r - m`, and squaring uses `** 2`: `squared_total += (r - m) ** 2`."],
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]
m = sum(returns) / len(returns)

squared_total = 0
for r in returns:
    squared_total += (r - m) ** 2
print(squared_total)''',
f"`{sum((x - mean(RET5)) ** 2 for x in RET5)}`. Squaring does two jobs: it removes the sign, so moves up and down both count as risk, and it weights big moves much more heavily than small ones.")

ex("K3", "The typical distance", 3,
r"""A plainer cousin of volatility is the **mean absolute deviation**: the average distance from the
mean, without squaring. Write it as a function `typical_distance(values)`.

Use `max(d, -d)` to take a distance without its sign, as you did in E4.""",
'''def typical_distance(values):
    m = sum(values) / len(values)
    total = ...
    for v in values:
        ...
    return ...

print(typical_distance([0.012, -0.004, 0.008, -0.011, 0.006]))''',
["The distance of `v` from the mean, ignoring sign, is `max(v - m, m - v)`.",
 "Accumulate those distances, then return `total / len(values)`."],
'''def typical_distance(values):
    m = sum(values) / len(values)
    total = 0
    for v in values:
        total += max(v - m, m - v)
    return total / len(values)

print(typical_distance([0.012, -0.004, 0.008, -0.011, 0.006]))''',
f"`{sum(max(x - mean(RET5), mean(RET5) - x) for x in RET5) / len(RET5)}`, about {(sum(max(x - mean(RET5), mean(RET5) - x) for x in RET5) / len(RET5)):.3%} a day. It answers the same question as volatility (how far does this move on a typical day) in a way that is easier to explain, though volatility is the standard because squaring has properties that make the mathematics work out.")

ex("K4", "What share of days were up?", 3,
r"""Write `up_share(returns)` returning the **fraction** of days that were positive, as a decimal. Count
the up days with a loop, then divide by how many days there were.""",
'''def up_share(returns):
    ...

r = [0.012, -0.004, 0.008, -0.011, 0.006]
print(up_share(r))''',
["Inside the function: `up = 0`, loop with `if r > 0: up += 1`.",
 "Careful with names: if the parameter is called `returns`, use a different name for the loop variable. Then `return up / len(returns)`."],
'''def up_share(returns):
    up = 0
    for r in returns:
        if r > 0:
            up += 1
    return up / len(returns)

r = [0.012, -0.004, 0.008, -0.011, 0.006]
print(up_share(r))''',
f"`{sum(1 for x in RET5 if x > 0) / len(RET5)}`, so {sum(1 for x in RET5 if x > 0)} of the {len(RET5)} days were up. Everything from this session in six lines: a function, an accumulator, a loop, a condition, and a result handed back.")

ex("K5", "Every stock, at once", 4,
r"""The capstone, and a preview of the case. `series` holds three stocks, each with its own list of
prices. Using the `average` idea and a loop over the dictionary, build a **new dictionary** `avg_price`
holding each ticker's average price.

Then print each ticker with its average, formatted to two decimals.""",
'''series = {
    "AAPL": [183.56, 182.19, 179.87],
    "KO": [59.20, 59.05, 59.40],
    "JNJ": [156.20, 157.10, 155.90],
}

avg_price = ...
for ticker in series:
    ...

# now print each ticker with its average:
...''',
["Start `avg_price = {}` (an empty dictionary, just as `[]` is an empty list).",
 'In the first loop, `prices = series[ticker]` then `avg_price[ticker] = sum(prices) / len(prices)`. In the second, `print(ticker, f"{avg_price[ticker]:.2f}")`.'],
'''series = {
    "AAPL": [183.56, 182.19, 179.87],
    "KO": [59.20, 59.05, 59.40],
    "JNJ": [156.20, 157.10, 155.90],
}

avg_price = {}
for ticker in series:
    prices = series[ticker]
    avg_price[ticker] = sum(prices) / len(prices)

for ticker in avg_price:
    print(ticker, f"{avg_price[ticker]:.2f}")''',
f"`AAPL {mean([183.56, 182.19, 179.87]):.2f}`, `KO {mean([59.20, 59.05, 59.40]):.2f}`, `JNJ {mean([156.20, 157.10, 155.90]):.2f}`. One dictionary in, one dictionary out, with a calculation applied to every stock. Swap the average for a volatility and you have the case, which is exactly what you are about to do.")

# ================================================================ L. Looking things up
section(
"## \U0001f50d L · Functions nobody taught you\n\n"
"*No course covers everything, and this one has deliberately covered very little. "
"The skill that outlasts any syllabus is being able to find a tool you have never "
"seen, work out what it does, and use it correctly.*\n\n"
"**Three places to look**\n\n"
"1. **`help(name)`**, Python's own manual, built into the language. It works "
"offline and it is always the version you are actually running. Every exercise "
"below starts here.\n"
"2. **The official documentation**, at `docs.python.org`. Longer, with examples "
"and the reasoning behind a function.\n"
"3. **Ask an AI to explain a function** (what it does, what its arguments mean). "
"Asking for an explanation you then verify is good practice. Asking it to write "
"your answer is how you fail the exam.\n\n"
"**Reading a signature.** `help` opens with a line like `round(number, ndigits=None)`. "
"That tells you the order of the arguments and that anything shown with an `=` has a "
"**default**, so you may leave it out. It is the same rule you met when you wrote "
"`def grow(price, rate=0.05)` in H4.\n\n"
"**Some help pages are long, and that is normal.** For `enumerate` and `zip` below, "
"Python prints every internal method the object has, with names wrapped in double "
"underscores like `__iter__`. Ignore all of it. Read the **signature** at the top and "
"the **first paragraph** underneath, and stop there. Learning what to skip is half of "
"reading documentation.\n\n"
"**Drills:** `help`, `sorted`, `abs`, `.keys()`, `.values()`, `.items()`, "
"`enumerate`, `zip`."
)

ex("L1", "Read the manual", 1,
r"""Start with a function you already know, so you can check the manual against your own understanding.

Run the cell to read `help(round)`. Its signature is `round(number, ndigits=None)`. Use that to round
**3.14159** to **3** decimals, and then to round it with the second argument left out entirely.""",
'''help(round)

three_dp = ...
no_second_argument = ...
print(three_dp, no_second_argument)''',
["The signature shows two parameters, and the `=None` on the second means it is optional.",
 "`round(3.14159, 3)` passes it; `round(3.14159)` leaves it out."],
'''help(round)

three_dp = round(3.14159, 3)
no_second_argument = round(3.14159)
print(three_dp, no_second_argument)''',
f"`{round(3.14159, 3)} {round(3.14159)}`. Leaving out `ndigits` rounds to a whole number, and note the type changes: you get the integer `3`, not `3.0`. The signature told you both facts before you ran anything, which is what makes reading it worth the ten seconds.")

ex("L2", "A function for putting things in order", 2,
r"""You have never been shown how to sort. Python has `sorted`.

Read `help(sorted)`, then use it to put the prices in order from smallest to largest. Print the
original list afterwards as well, and notice what did or did not happen to it.""",
'''prices = [183.56, 179.87, 183.48, 179.15, 182.19]

help(sorted)

ordered = ...
print("ordered: ", ordered)
print("original:", prices)''',
["`sorted(prices)` is the whole call.",
 "Look at the first line of the help text: it says it returns a **new** list. That is the point of printing the original afterwards."],
'''prices = [183.56, 179.87, 183.48, 179.15, 182.19]

help(sorted)

ordered = sorted(prices)
print("ordered: ", ordered)
print("original:", prices)''',
f"`ordered: {sorted([183.56, 179.87, 183.48, 179.15, 182.19])}` while `original: [183.56, 179.87, 183.48, 179.15, 182.19]`, unchanged. `sorted` **returns a new list** and leaves yours alone. (Lists also have a `.sort()` method that reorders in place and returns nothing, which is a classic source of confusion. The documentation is how you tell the two apart.)")

ex("L3", "When you do not even know the name", 3,
r"""`help` only works if you know what to type. Finding the *name* is the part that sends you to a search
engine or an AI, and that is a legitimate move.

In E4 and F4 you wrote `max(r, -r)` to get the size of a move without its sign. Python has a builtin
for exactly that, and it is called `abs`. Read its help, then redo F4 with it: count the returns
larger than **1%** in either direction.""",
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]

help(abs)

big_days = ...
for r in returns:
    ...
print(big_days)''',
["`abs(-0.011)` is `0.011`. It replaces the whole `max(r, -r)` trick.",
 "`if abs(r) > 0.01:` then `big_days += 1`."],
'''returns = [0.012, -0.004, 0.008, -0.011, 0.006]

help(abs)

big_days = 0
for r in returns:
    if abs(r) > 0.01:
        big_days += 1
print(big_days)''',
f"`{sum(1 for x in RET5 if max(x, -x) > 0.01)}`, the same answer as F4, and the code now says what it means. `max(r, -r)` was a correct workaround, and this is the lesson of the section: your workaround usually already exists as a named function, and finding it makes the code shorter and clearer.")

ex("L4", "What a dictionary can hand you", 2,
r"""Dictionaries carry methods of their own. Read `help(dict.values)`, then pull out the keys and the
values of this dictionary separately.

Finish by finding the largest volatility, using a function you already know.""",
'''vol = {"AAPL": 0.0141, "KO": 0.0080, "NVDA": 0.0330}

help(dict.values)

names = ...
numbers = ...
biggest = ...
print(names)
print(numbers)
print(biggest)''',
["`vol.keys()` and `vol.values()`, called with a dot like any other method.",
 "`max` accepts the values directly: `biggest = max(vol.values())`."],
'''vol = {"AAPL": 0.0141, "KO": 0.0080, "NVDA": 0.0330}

help(dict.values)

names = vol.keys()
numbers = vol.values()
biggest = max(vol.values())
print(names)
print(numbers)
print(biggest)''',
"`dict_keys(['AAPL', 'KO', 'NVDA'])`, `dict_values([0.0141, 0.008, 0.033])` and `0.033`. They display with those wrappers rather than as plain lists, but `max`, `min`, `sum` and `for` all accept them happily. `max(vol.values())` in one line replaces an entire best-so-far loop.")

ex("L5", "Both halves at once", 3,
r"""In I4 you looped a dictionary and looked each value up with `vol[ticker]`. `.items()` hands you the
key and the value together.

Read `help(dict.items)`, then print each ticker with its volatility as a percentage. To catch both
halves, name **two** variables in the `for` line, separated by a comma.""",
'''vol = {"AAPL": 0.0141, "KO": 0.0080, "NVDA": 0.0330}

help(dict.items)

...''',
["The loop line takes two names: `for ticker, v in vol.items():`.",
 'Then `print(ticker, f"{v:.2%}")`, with no lookup needed.'],
'''vol = {"AAPL": 0.0141, "KO": 0.0080, "NVDA": 0.0330}

help(dict.items)

for ticker, v in vol.items():
    print(ticker, f"{v:.2%}")''',
"`AAPL 1.41%`, `KO 0.80%`, `NVDA 3.30%`. `.items()` gives back a **pair** each time round, and naming two variables splits the pair between them. You saw the same idea in Session 1 when you wrote `w1, w2, w3 = 0.5, 0.3, 0.2`: several names on the left, several values on the right.")

ex("L6", "Position and value together", 3,
r"""In C3 you numbered the days with `for i in range(len(prices))` and then looked up `prices[i]`.
`enumerate` does both at once.

Read `help(enumerate)`, then print each position next to its price. Look at the signature while you
are there: it has a second parameter with a default.""",
'''prices = [183.56, 182.19, 179.87]

help(enumerate)

...''',
["`for i, price in enumerate(prices):` gives the position and the value together.",
 "The signature also shows `start=0`, so `enumerate(prices, start=1)` would number the days from 1."],
'''prices = [183.56, 182.19, 179.87]

help(enumerate)

for i, price in enumerate(prices):
    print(i, price)''',
"`0 183.56`, `1 182.19`, `2 179.87`, exactly what C3 produced with more machinery. The `start=0` default is worth remembering: `enumerate(prices, start=1)` numbers them 1, 2, 3, which is usually what you want when a human reads the output.")

ex("L7", "Two lists, walked in step", 3,
r"""Session 1 kept dates and closes in two parallel lists, joined only by position, and warned that they
could drift apart. `zip` walks them together.

Read `help(zip)`, then print each date next to its close, with no indexing at all.""",
'''dates  = ["Jan 02", "Jan 03", "Jan 04"]
closes = [183.56, 182.19, 179.87]

help(zip)

...''',
["`zip(dates, closes)` produces one pair per position.",
 "`for date, close in zip(dates, closes):` then print the two."],
'''dates  = ["Jan 02", "Jan 03", "Jan 04"]
closes = [183.56, 182.19, 179.87]

help(zip)

for date, close in zip(dates, closes):
    print(date, close)''',
"`Jan 02 183.56`, `Jan 03 182.19`, `Jan 04 179.87`. No `i`, no `[i]`, and no chance of reading position 3 of one list against position 4 of the other. Worth knowing from the help text: if the lists are different lengths, `zip` stops at the shorter one rather than complaining.")

ex("L8", "Rank the stocks, properly", 5,
r"""Put the section together. In the case you find the riskiest and calmest stock with a best-so-far
loop. With `sorted` you can rank **all** of them at once, from most volatile to least.

The obstacle: `sorted` orders a list, and you have a dictionary. So build a list first, and put the
**number first** in each entry, because `sorted` compares the first element of each item.

Read `help(sorted)` again for the parameter that reverses the order.""",
'''vol = {"AAPL": 0.0141, "KO": 0.0080, "NVDA": 0.0330, "JNJ": 0.0094, "JPM": 0.0148}

pairs = []
for ticker, v in vol.items():
    ...

ranked = ...
...''',
["Each entry is a small list with the number first: `pairs.append([v, ticker])`.",
 "`ranked = sorted(pairs, reverse=True)` puts the largest first. Then `for v, ticker in ranked:` to print them."],
'''vol = {"AAPL": 0.0141, "KO": 0.0080, "NVDA": 0.0330, "JNJ": 0.0094, "JPM": 0.0148}

pairs = []
for ticker, v in vol.items():
    pairs.append([v, ticker])

ranked = sorted(pairs, reverse=True)
for v, ticker in ranked:
    print(ticker, f"{v:.2%}")''',
"`NVDA 3.30%`, `JPM 1.48%`, `AAPL 1.41%`, `JNJ 0.94%`, `KO 0.80%`. Two ideas earned this: `sorted` compares the **first** element, so ordering by volatility means putting the volatility first, and `reverse=True` was sitting in the signature the whole time. This is the whole section in one exercise: a problem you could not solve with the lecture's tools, solved by reading two help pages.")

checkpoint("you can find a function you were never taught, read its signature, and use it correctly.")

# =========================================== M. Still yours from Session 1
section(
"## \U0001f501 M · Still yours from Session 1\n\n"
"*Nothing here is new. Every one of these needs a Session 1 tool, used inside a "
"Session 2 loop. That combination is what the case runs on, and it is what the "
"exam asks for.*\n\n"
"**Drills:** f-strings inside a loop, string methods on messy input, slicing a "
"window, and the float trap that catches everybody once."
)

ex("M1", "A report, one line per stock", 2,
r"""`year_return` holds four stocks. Loop over it and print one line each, with the ticker
**left-aligned in 6 characters** and the return as a **percentage with one decimal**:

```
AAPL   36.0%
KO      7.0%
```

The alignment recipe is `f"{ticker:<6}"`, and you already know `:.1%`.""",
'''year_return = {"AAPL": 0.36, "KO": 0.07, "NVDA": 1.79, "JNJ": 0.02}

for ticker in year_return:
    ...''',
["Inside the loop you need both the key and the value: `year_return[ticker]`.",
 '`print(f"{ticker:<6}{year_return[ticker]:>6.1%}")` lines the numbers up as well.'],
'''year_return = {"AAPL": 0.36, "KO": 0.07, "NVDA": 1.79, "JNJ": 0.02}

for ticker in year_return:
    print(f"{ticker:<6}{year_return[ticker]:>6.1%}")''',
"A readable report in three lines. The f-string is pure Session 1, the loop and the dictionary are today's, and the two together are how every result in this course gets shown to somebody.")

ex("M2", "Clean the tickers first", 3,
r"""Ticker symbols have arrived from four different systems, so they are a mess: stray spaces, mixed
case. Before you can count anything you have to normalise them.

Loop over `raw`, strip the spaces and force each to uppercase, collect the results into `clean`,
then report how many of them are **exactly three characters** long.""",
'''raw = ["  aapl ", "KO", " Jnj", "nvda  ", " ko"]

clean = []
for ticker in raw:
    ...

three_letter = 0
for ticker in clean:
    ...

print(clean)
print("three-letter:", three_letter)''',
["`ticker.strip().upper()` does both jobs in one go, then `.append` it.",
 "For the count, `len(ticker) == 3` inside an `if`, exactly like the up-day counter in F1."],
'''raw = ["  aapl ", "KO", " Jnj", "nvda  ", " ko"]

clean = []
for ticker in raw:
    clean.append(ticker.strip().upper())

three_letter = 0
for ticker in clean:
    if len(ticker) == 3:
        three_letter += 1

print(clean)
print("three-letter:", three_letter)''',
"`['AAPL', 'KO', 'JNJ', 'NVDA', 'KO']` and **three** three-letter tickers. Real data always needs this step, and it is always the boring Session 1 methods that do it. Notice that `KO` appears twice: cleaning does not deduplicate, and nobody told you it would.")

ex("M3", "Every three-day window", 3,
r"""A **rolling window** is the idea behind most financial features. Using slicing inside a `range`
loop, build every consecutive three-day window of `closes`, and report the average of each.

With seven prices, how many such windows fit?""",
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4, 73.9, 75.2]

n_windows = 1          # <- replace with the right count

for i in range(n_windows):
    window = ...
    print(i, window)''',
["A window starting at `i` is `closes[i:i + 3]`. The last one that fits starts at `len(closes) - 3`.",
 "So the loop is `for i in range(len(closes) - 2):`, and the average is `sum(window) / len(window)`."],
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4, 73.9, 75.2]

n_windows = len(closes) - 2

for i in range(n_windows):
    window = closes[i:i + 3]
    print(i, window, round(sum(window) / len(window), 2))''',
"**Five windows.** The `- 2` that stops you running off the end is the same arithmetic you met when pairing consecutive days, and writing it as `len(closes) - 2` rather than as `5` is what makes the code survive a longer list.\n\nIn Session 3 this whole cell becomes `.rolling(3).mean()`, one call, and it will help to have built it by hand first.")

ex("M4", "The comparison that fails", 4,
r"""⚠️ **Run the cell below.** Three returns of 0.1 are added up in a loop, and the total is compared
with 0.3. Python says they are **not** equal.

Nothing is broken. Print `total` to see why, then repair the comparison so it says `True`.""",
'''returns = [0.1, 0.1, 0.1]

total = 0
for r in returns:
    total = total + r

print(total)
print(total == 0.3)''',
["Look at what `total` actually prints. Decimals that are simple in base 10 are not simple in base 2.",
 "Compare rounded values instead: `round(total, 10) == round(0.3, 10)`."],
'''returns = [0.1, 0.1, 0.1]

total = 0
for r in returns:
    total = total + r

print(total)
print(round(total, 10) == round(0.3, 10))''',
"`0.30000000000000004`, so the raw comparison is `False`. This is not a Python defect, it is how binary floating point works in every language.\n\n**Never test two computed decimals with `==`.** Round both sides first, or ask whether the difference is small. Every language you will ever use has this trap, and this is the cheapest possible place to meet it.")

# ---------------------------------------------------------------- closing
md(
"## \U0001f3c1 Done\n\n"
"That covers all of Session 2: for loops, building results with accumulators and "
"`+=`, positions with `range`, collecting into lists, `if` / `elif` / `else`, "
"conditions inside loops, `while` loops, writing your own functions with "
"defaults and docstrings, dictionaries, and statistics written as reusable "
"functions.\n\n"
"The last section went further on purpose. `sorted`, `abs`, `.items()`, "
"`enumerate` and `zip` are ordinary, everyday Python that this course simply has "
"no time to teach, and you found them yourself from the documentation. That is "
"how you will meet almost every function you use from here on, including most of "
"pandas next session.\n\n"
"**Next:** open `session_02_case.ipynb` (*The Analyst's Notebook, Part 2*) and "
"use these tools to replace last session's rough risk gauge with a proper "
"volatility, computed from every daily return of a full year.\n\n"
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

import re as _re

_heads = [c.source.splitlines()[0] for c in cells
          if c.cell_type == "markdown" and _re.match(r"^### [A-Z]\d+ ", c.source)]
n_ex = len(_heads)
tiers = {}
for _h in _heads:
    _m = _re.search(r"(★+)", _h)
    if _m:
        _k = len(_m.group(1))
        tiers[_k] = tiers.get(_k, 0) + 1
tiers = {f"{k}star": tiers[k] for k in sorted(tiers)}
n_revisit = sum(1 for _h in _heads if "revisits" in _h)
print(f"wrote {OUT}  ({len(cells)} cells, {n_ex} exercises)")
print("stars:", tiers, " revisits tags:", n_revisit)
