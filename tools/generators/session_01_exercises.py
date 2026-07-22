# -*- coding: utf-8 -*-
"""Regenerate session_01_exercises.ipynb (v2, per instructor review).

Changes from v1:
- Toolkit: placeholder names made explicit + hover docs (title= on the chip).
- Em-dashes removed throughout.
- B2 reframed (average of three, a real parentheses trap, not a math puzzle).
- B6 reframed to a 'translate the stated formula' exercise (present value).
- Errors use a two-cell pattern: a cell that really errors (tagged
  raises-exception) then a blank fix cell. H1 rebuilt (case-sensitivity).
- A five-star ladder, rated against this session only, with a few genuinely hard
  challenges (worst-day offset, median without sorting, ...).
Only tools taught in session_01.qmd are used (no sorted, no //, no loops/if).
"""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "session_01" / "session_01_exercises.ipynb"

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

def ex(sid, title, n, task, work, hints, sol_code, sol_note, revisits=None):
    md(f"### {sid} · {title}  {badge(n, revisits)}\n\n{task}")
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

# ---------------------------------------------------------------- top matter
md(
"# \U0001f4d3 Session 1 · Exercises\n"
"### Beginning Python for Financial Data\n\n"
"These exercises put the ideas from the lecture into practice. Programming is "
"learned by doing, so most of the tasks ask you to write or complete a short "
"piece of code yourself.\n\n"
"Work through them at your own pace. They start easy and build up gradually."
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
"- No files, no internet. Every exercise stands on its own.\n\n"
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
"Everything you need is on this card. No exercise asks for anything beyond it.\n\n"
"> The names inside the brackets (`value`, `number`, `text`, `prices`, ...) are "
"**placeholders**: you put your own value or variable there. Only the tool's "
"name, the dots, the brackets and the quotes are fixed syntax. **Hover any tool** "
"to see what it does and what goes in.\n\n"
'<p style="line-height:2.1"><strong>Functions</strong> &nbsp; you call these by name<br>\n'
'<code style="cursor:help" title="Show one or more values. Separate several with commas; each prints with a space between.">print(value)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Tell you what kind of value it is: int, float, str or bool.">type(value)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Round number to ndigits decimals. round(3.14159, 2) gives 3.14.">round(number, ndigits)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="How many items are inside: characters in a string, elements in a list.">len(sequence)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The smallest of the values. Give it a list, or several values.">min(values)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The largest of the values. Give it a list, or several values.">max(values)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Add up every number in a list.">sum(numbers)</code></p>\n\n'
'<p style="line-height:2.1"><strong>Methods</strong> &nbsp; these hang off a value with a dot<br>\n'
"<code style=\"cursor:help\" title=\"An UPPERCASE copy of the string. 'aapl'.upper() is 'AAPL'.\">text.upper()</code> &nbsp;&nbsp; "
'<code style="cursor:help" title="How many times part appears inside the string.">text.count(part)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Add value to the end of the list. It changes the list itself.">prices.append(value)</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="The position (counting from 0) where value first appears in the list.">prices.index(value)</code></p>\n\n'
'<p style="line-height:2.1"><strong>Turn text into a number</strong><br>\n'
"<code style=\"cursor:help\" title=\"Turn a piece of text into a decimal number. float('52.40') is 52.4.\">float(text)</code></p>\n\n"
'<p style="line-height:2.1"><strong>Format inside an f-string</strong><br>\n'
'<code style="cursor:help" title="Show price with 2 decimal places, e.g. 248.83.">f\"{price:.2f}\"</code> &nbsp;&nbsp; '
'<code style="cursor:help" title="Show r as a percentage with 2 decimals. 0.0231 becomes 2.31%.">f\"{r:.2%}\"</code></p>'
)

md(
"**Formulas you will reach for**\n\n"
r"- **Simple return**: $r=\dfrac{p_1-p_0}{p_0}$" "\n"
r"- **Compound growth**: $FV = P\,(1+r)^{n}$" "\n"
r"- **Average**: $\bar{p}=\dfrac{\text{sum of values}}{\text{how many}}$" "\n"
r"- **Range**: $\text{range}=\max-\min$" "\n"
)
md("---")

# ================================================================ A. Notebooks
section(
"## \U0001f4d3 A · Notebooks & output\n\n"
"*Get code to run, and show exactly what you want to see.*\n\n"
"**Drills:** `print`, several outputs from one cell, the last-line rule."
)

ex("A1", "Show the day's prices", 1,
r"""A trading day has an **open**, a **high**, and a **low** price. They are stored below.
Add three `print(...)` calls so that running the cell shows all three, each on its own line.""",
'''open_price = 189.30
high_price = 193.05
low_price = 188.60

# print each of the three prices below:
...''',
"`print(open_price)` shows one value. You need three such lines.",
'''open_price = 189.30
high_price = 193.05
low_price = 188.60

print(open_price)
print(high_price)
print(low_price)''',
"Three lines in, three lines out. `print` is how you make a cell show more than just its final value.")

ex("A2", "Make them all appear", 1,
r"""When you run the cell below, **only `78.0` appears**, because a cell shows just the value of its
**last** line. Change the cell so that all three results are displayed.""",
'''12.0 + 18.0
30.0 + 25.0
40.0 + 38.0''',
"Wrap each line in `print(...)`. The last-line rule gives you only *one* automatic output; `print` gives you as many as you like.",
'''print(12.0 + 18.0)
print(30.0 + 25.0)
print(40.0 + 38.0)''',
"`30.0`, `55.0`, `78.0`. Without `print`, the first two results are computed and silently thrown away. Only the last line displays on its own.")

ex("A3", "Label your output", 1,
r"""A bare number on its own tells the reader nothing. Using a single `print(...)`, produce
exactly this line:

`Latest close: 248.83`""",
'''close = 248.83

# one print that produces:  Latest close: 248.83
...''',
'`print` can take several things separated by commas: `print("Latest close:", close)`.',
'''close = 248.83

print("Latest close:", close)''',
"`print` accepts several arguments separated by commas and puts a space between them. Handy for labelling a value so future-you knows what it means.")

checkpoint("you can run a cell and make it show precisely the output you intend.")

# ================================================================ B. Calculator
section(
"## \U0001f522 B · Python as a calculator\n\n"
"*Arithmetic, powers, and the order of operations.*\n\n"
"**Drills:** `+ - * /`, `**`, parentheses, comments."
)

ex("B1", "A quick invoice", 1,
r"""You buy **120** shares at **45.50** each. Write one expression that computes the total cost,
and add a `#` comment saying what it represents.""",
'''# total cost of the trade:
...''',
"Total cost is price times quantity. A comment is any text after a `#` on the line.",
'''120 * 45.50    # 120 shares at 45.50 each = total cost''',
"`5460.0`. Anything after `#` is a note for humans; Python ignores it. Leave yourself comments, because next-week-you will be grateful.")

ex("B2", "The average price, done right", 2,
r"""Three desks quote a stock at **182.50**, **179.80**, and **184.20**. Compute the average of the
three in a single expression.

A trap lives here: because `/` runs before `+`, writing `a + b + c / 3` divides only the last
price. Group correctly so all three are averaged.""",
'''q1 = 182.50
q2 = 179.80
q3 = 184.20

average = ...
average''',
["The three additions must all happen *before* the division. What forces that?",
 "Wrap the sum in parentheses: `(q1 + q2 + q3) / 3`."],
'''q1 = 182.50
q2 = 179.80
q3 = 184.20

average = (q1 + q2 + q3) / 3
average''',
f"`{(182.50 + 179.80 + 184.20) / 3}`. Without the parentheses you would get `{182.50 + 179.80 + 184.20 / 3}`, a silently wrong answer. Add parentheses whenever the order of operations matters.")

ex("B3", "Compound growth", 2,
r"""Money left to grow at a fixed rate follows

$$FV = P\,(1+r)^{n}$$

where $P$ is the amount invested, $r$ the yearly rate, and $n$ the number of years.
Invest **2000** at **6%** for **8** years, and complete the formula.""",
'''principal = 2000
rate = 0.06
years = 8

future_value = ...
future_value''',
["Recall `1000 * (1 + 0.04) ** 5` from the lecture. Same shape, new numbers.",
 "`future_value = principal * (1 + rate) ** years`"],
'''principal = 2000
rate = 0.06
years = 8

future_value = principal * (1 + rate) ** years
future_value''',
f"`{2000 * (1 + 0.06) ** 8}`. Round it with `round(future_value, 2)` to get **{round(2000 * (1.06) ** 8, 2)}**. The `**` operator is compounding: repeated multiplication, written once.")

ex("B4", "How many times your money?", 3,
r"""Reuse the compound-growth formula for **5000** at **3.5%** over **12** years. Then also compute
the **growth multiple** $(1+r)^{n}$, which is how many times the original the investment became.""",
'''principal = 5000
rate = 0.035
years = 12

future_value = ...
multiple = ...
print(future_value, multiple)''',
["`future_value` is the B3 formula with these numbers.",
 "`multiple = (1 + rate) ** years`, the same power term without the principal."],
'''principal = 5000
rate = 0.035
years = 12

future_value = principal * (1 + rate) ** years
multiple = (1 + rate) ** years
print(future_value, multiple)''',
f"`{5000 * (1.035) ** 12}` and `{(1.035) ** 12}`. The money grows about **{round((1.035)**12, 2)}×**. Notice the formula never changed, only the inputs. Reusing a stored value instead of retyping it is what variables are for.")

ex("B5", "Up 4%, then down 3%", 2,
r"""A stock at **100** rises **4%** one week, then falls **3%** the next. Write one expression for its
final price by multiplying by the two growth factors in turn.""",
'''# start at 100, then +4%, then -3%:
...''',
"A +4% move multiplies by `1.04`; a -3% move multiplies by `0.97`. Chain them: `100 * 1.04 * 0.97`.",
'''100 * 1.04 * 0.97''',
f"`{100 * 1.04 * 0.97}`. Up 4% then down 3% does **not** return to 100, because the second move acts on a bigger base. A small fact with large consequences for returns.")

ex("B6", "What is it worth today?", 3,
r"""A payment of **50000** arrives in **10** years. With money earning **5%** a year, its value
*today* (its present value) is given by the compound-growth formula rearranged for you:

$$PV = \dfrac{FV}{(1+r)^{n}}$$

Translate that formula into code.""",
'''future_payment = 50000
rate = 0.05
years = 10

present_value = ...
present_value''',
"Read the formula straight across: divide `future_payment` by `(1 + rate) ** years`.",
'''future_payment = 50000
rate = 0.05
years = 10

present_value = future_payment / (1 + rate) ** years
present_value''',
f"`{50000 / (1.05) ** 10}`, about **{round(50000 / (1.05)**10, 2)}**. This is *discounting*, one of the most-used ideas in finance, and it is the same `**` you already know, just dividing instead of multiplying.")

checkpoint("you can turn a written formula into working arithmetic and keep the order of operations honest.")

# ================================================================ C. Variables
section(
"## \U0001f3f7️ C · Variables\n\n"
"*Give a value a name so you can reuse it, update it, and read your own code "
"later.*\n\n"
"**Drills:** assignment, reuse, overwriting, naming rules."
)

ex("C1", "Name a holding", 1,
r"""Store a price of **189.50** and a quantity of **30** shares in well-named variables, then compute
the holding's value from those names (not from the raw numbers).""",
'''price = ...
shares = ...
holding_value = ...
holding_value''',
"Three lines: assign `price`, assign `shares`, then multiply the two names.",
'''price = 189.50
shares = 30
holding_value = price * shares
holding_value''',
f"`{189.50 * 30}`. Once a value has a name, you write the name and never retype the number. Change `price` on one line and everything downstream updates.")

ex("C2", "Reuse what you built", 2,
r"""Continuing from a holding value, a position grows **10%**. Using the variable you already have
(not the number), compute the new value.""",
'''price = 189.50
shares = 30
holding_value = price * shares

new_value = ...
new_value''',
"Multiply the existing `holding_value` by `1.10`.",
'''price = 189.50
shares = 30
holding_value = price * shares

new_value = holding_value * 1.10
new_value''',
f"`{(189.50 * 30) * 1.10}`. Building new results out of earlier ones, rather than one giant expression, is what keeps real analysis readable.")

ex("C3", "Overwrite, on purpose", 2,
r"""A price of **100** rises **2%** two days running. Model it with the *same* variable, reassigned each
day. Read `price = price * 1.02` from the right: take the current price, grow it, store it back.""",
'''price = 100.0

# day 1: grow price by 2%
...
# day 2: grow price by 2% again
...
price''',
"Write `price = price * 1.02` twice. The name on the left is updated to the value computed on the right.",
'''price = 100.0

price = price * 1.02
price = price * 1.02
price''',
f"`{round(100.0 * 1.02 * 1.02, 4)}`. A variable can appear on both sides of `=`. The right side is computed first with the *old* value, then the result replaces it.")

ex("C4", "Fix the illegal names", 2,
r"""Two names below are illegal Python. The rules: a name may use only letters, digits and
underscores, **may not start with a digit**, and **may not contain a space**. Give each value a
legal, readable name and print both.

```python
2015_return = 0.05      # starts with a digit
share price = 45.50     # contains a space
```""",
'''# choose legal, readable names for these two values:
a = 0.05      # rename 'a'
b = 45.50     # rename 'b'
print(a, b)''',
"For example `return_2015` and `share_price`. Underscores stand in for the spaces you are not allowed.",
'''return_2015 = 0.05
share_price = 45.50
print(return_2015, share_price)''',
"`0.05 45.5`. `2015_return` breaks the no-leading-digit rule; `share price` breaks the no-spaces rule. Good names read like English and cost nothing.")

ex("C5", "The swap", 3,
r"""`morning` and `evening` hold two prices. Swap their values, so afterwards `morning` holds what
`evening` did and vice versa, using only assignment (no lists, no tricks you have not seen).""",
'''morning = 10.0
evening = 12.5

# swap them, so morning becomes 12.5 and evening becomes 10.0
...

print(morning, evening)''',
["If you write `morning = evening` first, the old `morning` is gone forever. You need somewhere to *park* it first.",
 "Use a third variable: `temp = morning`, then `morning = evening`, then `evening = temp`."],
'''morning = 10.0
evening = 12.5

temp = morning
morning = evening
evening = temp

print(morning, evening)''',
"`12.5 10.0`. A temporary variable holds one value while the other is moved. It works because `=` stores a value; it does not create a link between the two names.")

checkpoint("you can name values, reuse and update them, and name them well.")

# ================================================================ D. Runtime
section(
"## \U0001f9e0 D · The notebook's memory\n\n"
"*A notebook remembers what earlier cells defined, and only what has actually "
"run.*\n\n"
"**Drills:** using a name before it exists, execution order."
)

ex_fix("D1", "Nothing there yet", 2,
r"""⚠️ **Run the cell below first.** On a fresh notebook it fails, because it reads `balance` before
anything was stored under that name. Read the error, then fix it underneath: give `balance` a
starting value of **1000**, then add 100, then show it.""",
'''balance = balance + 100''',
'''# make this run cleanly on a fresh notebook:
...''',
["The failing line uses `balance` on the right before any value exists under that name. That is a `NameError`.",
 "First `balance = 1000`, then `balance = balance + 100`, then `balance`."],
'''balance = 1000
balance = balance + 100
balance''',
"`1100`. The original line asks Python to *read* `balance` before it was ever *assigned*. Define first, then update. The runtime only knows what it has been told.")

ex("D2", "Put them in order", 2,
r"""These three lines were run in the order shown and failed, because each name is used before it
exists:

```python
total = price * qty     # ran first
price = 45.50
qty = 10
```

Rewrite the three lines in an order that runs cleanly from top to bottom and leaves `total` equal
to **455.0**.""",
'''# the three lines, reordered so each name exists before it is used:
...''',
"A name must be assigned *above* the line that uses it. `price` and `qty` have to come before `total`.",
'''price = 45.50
qty = 10
total = price * qty
total''',
"`455.0`. A notebook runs top to bottom, so define the ingredients before you combine them. This is exactly why *Restart & run all* is the real test that a notebook is correct.")

checkpoint("you understand that cells share one memory, filled in the order you run them.")

# ================================================================ E. Types
section(
"## \U0001f9ec E · Types, text & formatting\n\n"
"*Numbers, text, and true/false, plus why mixing them up quietly costs money.*\n\n"
"**Drills:** `type`, `int`/`float`, division, strings, `float()`, f-strings, comparisons."
)

ex("E1", "Ask for the type", 1,
r"""For each value below, write a line that prints its **type**. Run it and read the four answers.""",
'''# print the type of each value:
...   # 250
...   # 19.95
...   # "AAPL"
...   # 19.95 > 20''',
"`type(250)` gives the type; wrap it in `print(...)` to see it. Four lines.",
'''print(type(250))
print(type(19.95))
print(type("AAPL"))
print(type(19.95 > 20))''',
"`int`, `float`, `str`, `bool`. A whole number is an `int`, one with a decimal point is a `float`, quoted text is a `str`, and any comparison is a `bool`.")

ex("E2", "Division makes decimals", 1,
r"""Write the division of **100** by **4**, store it, and print the value together with its type.
Notice what kind of number you get back.""",
'''result = ...        # 100 divided by 4
print(result, type(result))''',
"Just `100 / 4`.",
'''result = 100 / 4
print(result, type(result))''',
"`25.0 <class 'float'>`. Division in Python **always** returns a `float`, even when it divides evenly. You get `25.0`, not `25`.")

ex_fix("E3", "The silent string bug", 3,
r"""⚠️ **Run the cell below.** It does **not** error, yet the answer is nonsense: with `price` still
text, `price * shares` repeats the text into `"848484"` instead of multiplying. Look at the output,
then fix it underneath so `cost` is the real number **252.0**.""",
'''price = "84"      # arrived from a file, as text
shares = 3
price * shares''',
'''price = "84"
shares = 3

cost = ...
cost''',
["`price` is text. Turn it into a number first with `float(...)`.",
 "`cost = float(price) * shares`"],
'''price = "84"
shares = 3

cost = float(price) * shares
cost''',
"`252.0`. No error is **not** the same as no bug, and this is the dangerous kind: a wrong answer with no complaint. Prices read from files often arrive as text, so convert before you compute.",
raises=False)

ex("E4", "Glue text together", 1,
r"""Given a ticker, use `+` to build the sentence `AAPL is a US stock`. Remember that `+` between
strings joins them.""",
'''ticker = "AAPL"

sentence = ...
sentence''',
'`ticker + " is a US stock"`. Mind the leading space inside the quotes.',
'''ticker = "AAPL"

sentence = ticker + " is a US stock"
sentence''',
'`\'AAPL is a US stock\'`. Between strings, `+` means join, not add. The space had to sit inside the quotes, or you would get `"AAPLis a US stock"`.')

ex("E5", "Two prices from a file", 2,
r"""Two prices arrived as text: `"52.40"` and `"48.10"`. Add them as real numbers to get their sum.""",
'''p1 = "52.40"
p2 = "48.10"

total = ...
total''',
["`\"52.40\" + \"48.10\"` would glue the text. Convert each with `float(...)` first.",
 "`total = float(p1) + float(p2)`"],
'''p1 = "52.40"
p2 = "48.10"

total = float(p1) + float(p2)
total''',
"`100.5`. Without `float`, `+` would produce the text `\"52.4048.10\"`. Converting first is the difference between a sum and a mess.")

ex("E6", "Show a tidy price", 2,
r"""An f-string drops values into text, and a `:.2f` inside the braces formats a number to **two
decimals**. Produce exactly:

`Apple closed at 248.83`""",
'''price = 248.834

message = ...
message''',
'Put an `f` before the quotes and the variable in braces with its format: `f"Apple closed at {price:.2f}"`.',
'''price = 248.834

message = f"Apple closed at {price:.2f}"
message''',
"`'Apple closed at 248.83'`. `:.2f` rounds *for display* to two decimals, exactly what prices want. The underlying value is left untouched.")

ex("E7", "Show a return as a percent", 3,
r"""The other format recipe, `:.2%`, turns a decimal into a percentage. A stock went from **183.56**
to **248.83**. Compute the simple return and format it:

$$r=\dfrac{p_1-p_0}{p_0}$$

Target output: `Return: 35.56%`""",
'''p0 = 183.56
p1 = 248.83

r = ...
message = ...
message''',
["`r = (p1 - p0) / p0`, the simple-return formula.",
 '`message = f"Return: {r:.2%}"`. The `:.2%` multiplies by 100 and adds the `%` for you.'],
'''p0 = 183.56
p1 = 248.83

r = (p1 - p0) / p0
message = f"Return: {r:.2%}"
message''',
f"`'Return: {((248.83-183.56)/183.56):.2%}'`. Note that `:.2%` does the ×100 *and* the `%` sign, so never multiply by 100 yourself as well, or you will report 3556%.")

ex("E8", "Ask a yes/no question", 1,
r"""A comparison produces a `bool` (`True` or `False`). Today's close is **251.20**. Write one
comparison that asks whether it closed **above 250**, and print the answer.""",
'''close = 251.20

beat_250 = ...
print(beat_250)''',
"Use the `>` operator: `close > 250`.",
'''close = 251.20

beat_250 = close > 250
print(beat_250)''',
"`True`. Comparisons (`>`, `<`, `>=`, `<=`, `==`, `!=`) answer yes/no questions. They look modest now, but later they select rows of data, and later still they become the thing a model predicts.")

ex("E9", "Convert, compare, report", 4,
r"""A price arrives as text: `"205.50"`. Without using `if`, build a single line that reports both the
tidy price and whether it is above 200, exactly like:

`205.50 above 200? True`""",
'''price_text = "205.50"

report = ...
report''',
["You will need `float(price_text)` twice: once formatted with `:.2f`, once inside a `> 200` comparison.",
 '`report = f"{float(price_text):.2f} above 200? {float(price_text) > 200}"`'],
'''price_text = "205.50"

report = f"{float(price_text):.2f} above 200? {float(price_text) > 200}"
report''',
"`'205.50 above 200? True'`. An f-string can hold *any* expression in its braces: a conversion, a format, even a whole comparison. Four ideas from this section in one line.")

ex("E10", "The f that went missing", 2,
r"""⚠️ **Run the cell below.** It does not error, and the output is still wrong: it prints the braces
and the variable name instead of the price.

Find the one character that is missing and fix it.""",
'''close = 248.834

message = "Apple closed at {close:.2f}"
message''',
["Compare it with E6. What is written immediately before the opening quote there?",
 "Without the `f`, Python sees an ordinary string and leaves the braces exactly as typed."],
'''close = 248.834

message = f"Apple closed at {close:.2f}"
message''',
"`'Apple closed at 248.83'`. A missing `f` is the quietest bug in this section, because nothing fails: you just publish a report with `{close:.2f}` printed in the middle of it.")

checkpoint("you can tell the types apart, convert text to numbers, format for humans, and ask true/false questions.")

# ================================================================ F. Functions
section(
"## \U0001f6e0️ F · Functions & methods\n\n"
"*Named tools you call with `(...)`, and methods you call with a dot: "
"`value.method(...)`.*\n\n"
"**Drills:** `round`, `len`, `.upper()`, `.count()`."
)

ex("F1", "Two little tools", 1,
r"""Write two lines: one that rounds **19.876** to **2** decimals, and one that reports how many
characters are in the word `"portfolio"`. Print each.""",
'''# round 19.876 to 2 decimals, and count the letters in "portfolio":
...
...''',
"`round(x, n)` takes the number and how many decimals; `len(text)` counts characters.",
'''print(round(19.876, 2))
print(len("portfolio"))''',
"`19.88` and `9`. `round` and `len` are two of the small tools you will use in almost every task.")

ex("F2", "Fix the ticker", 1,
r"""A ticker was typed in lowercase. Use the string method that makes text uppercase to fix it.""",
'''ticker = "jpm"

fixed = ...
fixed''',
"A method is called with a dot: `ticker.upper()`.",
'''ticker = "jpm"

fixed = ticker.upper()
fixed''',
"`'JPM'`. `.upper()` is a **method**: it belongs to the string and is called with a dot, `value.method()`. That same grammar will run pandas and everything after it.")

ex("F3", "Count occurrences", 2,
r"""`.count(part)` reports how many times a piece of text appears inside a string. Count how many
times `"a"` appears in `"banana"`.""",
'''word = "banana"

how_many = ...
how_many''',
"`word.count(\"a\")`, the same dot-grammar as `.upper()`.",
'''word = "banana"

how_many = word.count("a")
how_many''',
'`3`. Methods take arguments in their brackets just like functions do. The difference is that they hang off a value with a dot.')

ex("F4", "Assemble a headline", 3,
r"""Combine a method and a format recipe. Given a lowercase ticker and a price, build:

`JPM closed at 231.45`""",
'''ticker = "jpm"
price = 231.451

headline = ...
headline''',
'A method call is fine *inside* an f-string: `f"{ticker.upper()} closed at {price:.2f}"`.',
'''ticker = "jpm"
price = 231.451

headline = f"{ticker.upper()} closed at {price:.2f}"
headline''',
"`'JPM closed at 231.45'`. Braces in an f-string happily hold a method call and a format spec at once.")

ex("F5", "How much of the name?", 3,
r"""For `name = "Coca-Cola Company"`, compute the fraction of its characters that are a lowercase
`"o"`, and show it as a percentage (for example `17.65%`). You will need a method *and* a
function.""",
'''name = "Coca-Cola Company"

fraction_o = ...
message = ...
message''',
['Count the `"o"`s with `name.count("o")` and the total characters with `len(name)`, then divide.',
 '`fraction_o = name.count("o") / len(name)`, then `f"{fraction_o:.2%}"`.'],
'''name = "Coca-Cola Company"

fraction_o = name.count("o") / len(name)
message = f"{fraction_o:.2%}"
message''',
f"`'{(('Coca-Cola Company'.count('o'))/len('Coca-Cola Company')):.2%}'`, that is {'Coca-Cola Company'.count('o')} of the {len('Coca-Cola Company')} characters. A method and a function, combined and formatted: the shape of most real code.")

checkpoint("you can call functions and methods, and you recognise the `value.method(...)` pattern everywhere.")

# ================================================================ G. Lists
section(
"## \U0001f4ca G · Lists\n\n"
"*Many values, in order, under one name. Lists are the structure you will use "
"most this week.*\n\n"
"**Drills:** create, `len`, indexing, negative indexing, slicing, `.append`, "
"`.index`, `min`/`max`/`sum`."
)

ex("G1", "Collect a week", 1,
r"""Put these five closing prices into a list called `closes`, in order:
**183.56, 182.19, 179.87, 179.15, 183.48**. Then report how many values it holds.""",
'''closes = ...
count = ...
print(count)''',
"A list is written with square brackets and commas: `[183.56, 182.19, ...]`. `len(closes)` counts the items.",
'''closes = [183.56, 182.19, 179.87, 179.15, 183.48]
count = len(closes)
print(count)''',
"`5`. One name now holds the whole week. A real year is about 252 of these, far too many for one variable each.")

ex("G2", "Pick out three days", 1,
r"""From the list below, pull the **first** day, the **third** day, and the **last** day into three
variables. Remember that positions count from **0**, and `-1` is the last.""",
'''closes = [61.20, 62.05, 60.90, 63.10, 64.00, 63.75]

first = ...
third = ...
last  = ...
print(first, third, last)''',
"First is `closes[0]`, third is `closes[2]` (not `[3]`), last is `closes[-1]`.",
'''closes = [61.20, 62.05, 60.90, 63.10, 64.00, 63.75]

first = closes[0]
third = closes[2]
last  = closes[-1]
print(first, third, last)''',
"`61.2 60.9 63.75`. The index is the *distance from the start*, so the third item sits at position 2. `[-1]` is useful: it gives the latest value, whatever the length.")

ex("G3", "Yesterday vs today", 2,
r"""Using **negative** indexing only, compute how much the **last** price changed from the
**second-to-last** (the last minus the one before it).""",
'''closes = [61.20, 62.05, 60.90, 63.10, 64.00, 63.75]

change = ...
change''',
"The last is `closes[-1]`; the one before it is `closes[-2]`.",
'''closes = [61.20, 62.05, 60.90, 63.10, 64.00, 63.75]

change = closes[-1] - closes[-2]
change''',
f"`{round(63.75 - 64.00, 2)}`, so the price slipped on the final day. Negative indices count from the end, so `[-2]` is always the second-last value without your knowing the length.")

ex_fix("G4", "Off the end", 2,
r"""⚠️ **Run the cell below.** It asks for a seventh price from a six-item list and fails. Read the
error, then fix it underneath so `sixth` holds the **sixth (last)** price.""",
'''prices = [61.20, 62.05, 60.90, 63.10, 64.00, 63.75]
prices[6]''',
'''prices = [61.20, 62.05, 60.90, 63.10, 64.00, 63.75]

sixth = ...
sixth''',
["Six items fill positions **0 to 5**. Position 6 does not exist, which is the `IndexError`.",
 "The sixth item is `prices[5]`, or equivalently `prices[-1]`."],
'''prices = [61.20, 62.05, 60.90, 63.10, 64.00, 63.75]

sixth = prices[5]
sixth''',
"`63.75`. The count is 6, but the positions are 0 to 5, the classic *off-by-one*. `prices[-1]` sidesteps it entirely.")

ex("G5", "A window in the middle", 2,
r"""A **slice** `list[start:stop]` takes from `start` up to, but **not including**, `stop`. From the
ten-day list, extract the four days at positions **3, 4, 5, 6**.""",
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4, 73.9, 75.2, 74.8, 76.0, 75.5]

window = ...
window''',
"To include position 6, the `stop` must be 7 (one past the last one you want): `closes[3:7]`.",
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4, 73.9, 75.2, 74.8, 76.0, 75.5]

window = closes[3:7]
window''',
f"`{[71.2, 72.0, 70.5, 73.1, 74.4, 73.9, 75.2, 74.8, 76.0, 75.5][3:7]}`. The item count equals `stop - start`, here `7 - 3 = 4`. The excluded `stop` feels odd at first, then becomes second nature.")

ex("G6", "Both ends", 2,
r"""Leave a slice's start or stop blank to mean *from the beginning* or *to the end*. From the same
list, take the **first three** days and, separately, the **last three** days.""",
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4, 73.9, 75.2, 74.8, 76.0, 75.5]

first_three = ...
last_three  = ...
print(first_three)
print(last_three)''',
"`closes[:3]` starts from the beginning; `closes[-3:]` runs to the end.",
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4, 73.9, 75.2, 74.8, 76.0, 75.5]

first_three = closes[:3]
last_three  = closes[-3:]
print(first_three)
print(last_three)''',
f"`{[71.2,72.0,70.5,73.1,74.4,73.9,75.2,74.8,76.0,75.5][:3]}` and `{[71.2,72.0,70.5,73.1,74.4,73.9,75.2,74.8,76.0,75.5][-3:]}`. An empty side just means all the way in that direction.")

ex("G7", "A new close arrives", 1,
r"""Markets closed and a new price of **77.10** came in. Use `.append(...)` to add it to the end of the
list, then report the new length.""",
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4]

...
print(len(closes))''',
"`closes.append(77.10)` adds to the end, then `len(closes)`.",
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4]

closes.append(77.10)
print(len(closes))''',
"`6`. `.append` changes the list in place, so notice there is no `closes = ...`. The list itself grows by one.")

ex("G8", "Where is the low?", 3,
r"""`.index(value)` reports the **position** of a value in a list. Find the lowest price with
`min(...)`, then use `.index(...)` to find *which day* (position) it fell on.""",
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4, 73.9, 75.2, 74.8, 76.0, 75.5]

lowest = ...
low_position = ...
print(lowest, low_position)''',
["`min(closes)` gives the lowest value.",
 "Feed that straight in: `closes.index(min(closes))`."],
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4, 73.9, 75.2, 74.8, 76.0, 75.5]

lowest = min(closes)
low_position = closes.index(lowest)
print(lowest, low_position)''',
"`70.5 2`, so the low was at position 2. Pairing `min` with `.index`, that is *what* was the extreme and *where*, is a move you will make constantly.")

ex("G9", "Summarise the week", 2,
r"""From one list, report three summaries: the **highest** price, the **lowest** price, and the
**average**. The average is the sum divided by the count:

$$\bar{p}=\dfrac{\text{sum of prices}}{\text{how many prices}}$$""",
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4, 73.9, 75.2, 74.8, 76.0, 75.5]

highest = ...
lowest  = ...
average = ...
print(highest, lowest, average)''',
["`max(closes)` and `min(closes)` give the extremes.",
 "`average = sum(closes) / len(closes)`."],
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4, 73.9, 75.2, 74.8, 76.0, 75.5]

highest = max(closes)
lowest  = min(closes)
average = sum(closes) / len(closes)
print(highest, lowest, average)''',
f"`76.0 70.5 {sum([71.2,72.0,70.5,73.1,74.4,73.9,75.2,74.8,76.0,75.5])/10}` (wrap the average in `round(..., 2)` to tidy it). `sum`/`len` is your first statistic built from raw parts, exactly what a data table will compute for you in one step later in the course.")

ex("G10", "How wide did it swing?", 3,
r"""A stock's range relative to its floor is a rough gauge of how much it moved. Compute the range as
a **percentage of the lowest** price:

$$\text{relative range}=\dfrac{\max-\min}{\min}$$

and format it as a percent.""",
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4, 73.9, 75.2, 74.8, 76.0, 75.5]

relative_range = ...
message = ...
message''',
["First the range: `max(closes) - min(closes)`. Then divide by `min(closes)`.",
 '`relative_range = (max(closes) - min(closes)) / min(closes)`, then `f"{relative_range:.2%}"`.'],
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4, 73.9, 75.2, 74.8, 76.0, 75.5]

relative_range = (max(closes) - min(closes)) / min(closes)
message = f"{relative_range:.2%}"
message''',
f"`'{((max([71.2,72.0,70.5,73.1,74.4,73.9,75.2,74.8,76.0,75.5])-min([71.2,72.0,70.5,73.1,74.4,73.9,75.2,74.8,76.0,75.5]))/min([71.2,72.0,70.5,73.1,74.4,73.9,75.2,74.8,76.0,75.5])):.2%}'`. Three tools, `max`, `min`, and a format recipe, combined into a genuine if crude risk measure.")

ex("G11", "From an empty cell", 3,
r"""No scaffolding this time. Write the whole thing yourself, from the empty cell.

Starting from the five closes **71.2, 72.0, 70.5, 73.1, 74.4**:

1. put them in a list called `closes`
2. a sixth day closes at **77.10**, so add it
3. the first day turns out to belong to the previous week, so drop it with a slice
4. report how many days are left, and what the new first day is""",
'''...''',
["Four lines, one per step. `closes = [...]`, then `.append(...)`, then `closes = closes[1:]`.",
 "The last step is a single `print` with two values in it: `len(closes)` and `closes[0]`."],
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4]
closes.append(77.10)
closes = closes[1:]
print(len(closes), closes[0])''',
"`5 72.0`. Nothing here is new, and that is the point: with no blanks to fill in you have to remember the order the pieces go in, which is what the exam actually asks of you.")

checkpoint("you can build a list, reach any day or window, grow it, locate values, and summarise it.")

# ================================================================ H. Errors
section(
"## \U0001f41e H · Reading errors\n\n"
"*An error message is a diagnosis, not a scolding. Each of these cells breaks on "
"purpose so you can meet the four errors of week one and repair them.*\n\n"
"**Drills:** `NameError`, `TypeError`, `SyntaxError`, `IndexError`."
)

ex_fix("H1", "A capital mistake", 2,
r"""⚠️ **Run the cell below.** It fails with a **NameError**. Look closely at the capital letters: the
value was stored as `Revenue`, but the last line asks for `revenue`, and Python treats those as two
different names. Read the message, then fix it underneath.""",
'''Revenue = 5000
costs = 3200
revenue - costs''',
'''Revenue = 5000
costs = 3200

profit = ...
profit''',
["`Revenue` and `revenue` are different names. Only one of them holds 5000.",
 "Match the capital: `profit = Revenue - costs`."],
'''Revenue = 5000
costs = 3200

profit = Revenue - costs
profit''',
"`1800`. A `NameError` almost always means a typo, a wrong capital letter, or a cell you forgot to run. Check the exact spelling first.")

ex_fix("H2", "Wrong kind of thing", 2,
r"""⚠️ **Run the cell below.** It fails with a **TypeError**, because you cannot add a number to text.
Read the message, then fix it underneath so `final` is the price plus the markup, as a number.""",
'''price_text = "230.10"     # text, from a file
markup = 5
price_text + markup''',
'''price_text = "230.10"
markup = 5

final = ...
final''',
["A `TypeError` means an operation met the wrong type. Convert the text to a number first.",
 "`final = float(price_text) + markup`"],
'''price_text = "230.10"
markup = 5

final = float(price_text) + markup
final''',
"`235.1`. A `TypeError` is Python refusing to guess what you meant between mismatched types, usually the sign that a number is secretly text.")

ex_fix("H3", "Python cannot even start", 1,
r"""⚠️ **Run the cell below.** It fails with a **SyntaxError**, before it runs at all, because a bracket
is missing. Read the message, then write the corrected line underneath so it prints `Total: 500`.""",
'''total = 500
print("Total:", total''',
'''total = 500

# write the corrected print statement:
...''',
"Count the brackets in the broken line: one `(` opens, but nothing closes it.",
'''total = 500

print("Total:", total)''',
"`Total: 500`. A `SyntaxError` is about *shape*, not meaning: a missing bracket, quote, or comma. Python cannot even begin until the line is well-formed.")

ex_fix("H4", "One step too far", 2,
r"""⚠️ **Run the cell below.** It fails with an **IndexError**: with four prices, the positions are 0 to
3, so `prices[days]` (position 4) runs off the end. Read the message, then fix it underneath so
`last` holds the actual final price.""",
'''prices = [61.20, 62.05, 60.90, 63.10]
days = len(prices)
prices[days]''',
'''prices = [61.20, 62.05, 60.90, 63.10]

last = ...
last''',
["Four items sit at positions 0, 1, 2, 3. The length is 4, but there is no position 4.",
 "Use `prices[3]`, or the length-proof `prices[-1]`."],
'''prices = [61.20, 62.05, 60.90, 63.10]

last = prices[-1]
last''',
"`63.1`. An `IndexError` means you reached past the end of the list, nearly always an off-by-one. Using `len(prices)` as an index is the classic version of the mistake, since a length of 4 points one step beyond the last item. `[-1]` is the safe way to say the last one.")

checkpoint("you can recognise and repair all four of week one's errors from their messages.")

# ================================================================ I. Together
section(
"## \U0001f4c8 I · Putting it together, the analyst's toolkit\n\n"
"*Everything from today, aimed at real prices. This is the exact work the case "
"will ask of you.*\n\n"
"**Drills:** simple returns, parallel lists, locating extremes, weighting, "
"compounding, a formatted report."
)

ex("I1", "One day's return", 2,
r"""Yesterday a stock closed at **183.56**, today at **182.19**. Compute the simple return and show it
as a percentage:

$$r=\dfrac{p_1-p_0}{p_0}$$""",
'''p0 = 183.56
p1 = 182.19

r = ...
message = ...
message''',
["`r = (p1 - p0) / p0`.",
 '`message = f"Return: {r:.2%}"`'],
'''p0 = 183.56
p1 = 182.19

r = (p1 - p0) / p0
message = f"Return: {r:.2%}"
message''',
f"`'Return: {((182.19-183.56)/183.56):.2%}'`, a small down day. The percentage change in price is the basic quantity used throughout the course.")

ex("I2", "A whole window's return, twice", 4,
r"""There are two ways to measure what a window did, and a good analyst checks one against the other.

**Route A:** compare the two ends directly, $\dfrac{p_{\text{last}}-p_{\text{first}}}{p_{\text{first}}}$.

**Route B:** take the first three days only, and compound their two daily returns:
$(1+r_1)(1+r_2)-1$.

Compute both over the **first three closes**, and print whether they agree.""",
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4]
first_three = closes[:3]

route_a = ...
r1 = ...
r2 = ...
route_b = ...

print("A:", route_a)
print("B:", route_b)
print("agree:", ...)''',
["Route A uses `first_three[0]` and `first_three[-1]`. Route B needs the two daily returns first.",
 "The two answers are floats, so compare them rounded: `round(route_a, 10) == round(route_b, 10)`."],
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4]
first_three = closes[:3]

route_a = (first_three[-1] - first_three[0]) / first_three[0]
r1 = (first_three[1] - first_three[0]) / first_three[0]
r2 = (first_three[2] - first_three[1]) / first_three[1]
route_b = (1 + r1) * (1 + r2) - 1

print("A:", route_a)
print("B:", route_b)
print("agree:", round(route_a, 10) == round(route_b, 10))''',
"Both give **-0.98%**, and they agree. That is not a coincidence: compounding the daily steps *is* the end-to-end return, which is why you may never add returns together. Note the rounding in the comparison. Two floats that are mathematically equal can still differ in the last bits, so `==` on raw floats is a trap.")

ex("I3", "Which day was best?", 3,
r"""Two lists run in parallel, a date and a price at each position. Find the **highest** close and,
using `.index`, the **date** it happened on.""",
'''dates  = ["Jan 02", "Jan 03", "Jan 04", "Jan 05", "Jan 08"]
closes = [183.56, 182.19, 179.87, 179.15, 183.48]

best_price = ...
best_date  = ...
print(best_date, best_price)''',
["`max(closes)` is the highest price.",
 "Find its position and read the *same* position in `dates`: `dates[closes.index(max(closes))]`."],
'''dates  = ["Jan 02", "Jan 03", "Jan 04", "Jan 05", "Jan 08"]
closes = [183.56, 182.19, 179.87, 179.15, 183.48]

best_price = max(closes)
best_date  = dates[closes.index(best_price)]
print(best_date, best_price)''',
"`Jan 02 183.56`. Two lists kept *in step*, connected only by position: `.index` finds *where* in one, and you read the same spot in the other. Next week a table keeps them together so they can never drift apart.")

ex("I4", "The worst day", 4,
r"""From the week below, find the **worst single-day return** and the **date** it fell on. Five prices
give **four** daily returns. Build them by hand into a list, then locate the minimum.

Watch the offset: the return in position `k` of your list happened on day `dates[k+1]`, because a
return needs the *previous* day too.""",
'''dates  = ["Jan 02", "Jan 03", "Jan 04", "Jan 05", "Jan 08"]
closes = [183.56, 182.19, 179.87, 179.15, 183.48]

r1 = (closes[1] - closes[0]) / closes[0]
r2 = ...
r3 = ...
r4 = ...
returns = [r1, r2, r3, r4]

worst = ...
worst_date = ...
print(worst_date, worst)''',
["Each return follows the same shape as `r1`, shifted along one: `r2` uses `closes[2]` and `closes[1]`, and so on.",
 "`worst = min(returns)`; its position is `returns.index(worst)`; the date is `dates[returns.index(worst) + 1]`."],
'''dates  = ["Jan 02", "Jan 03", "Jan 04", "Jan 05", "Jan 08"]
closes = [183.56, 182.19, 179.87, 179.15, 183.48]

r1 = (closes[1] - closes[0]) / closes[0]
r2 = (closes[2] - closes[1]) / closes[1]
r3 = (closes[3] - closes[2]) / closes[2]
r4 = (closes[4] - closes[3]) / closes[3]
returns = [r1, r2, r3, r4]

worst = min(returns)
worst_date = dates[returns.index(worst) + 1]
print(worst_date, f"{worst:.2%}")''',
f"`Jan 04 {min([(182.19-183.56)/183.56,(179.87-182.19)/182.19,(179.15-179.87)/179.87,(183.48-179.15)/179.15]):.2%}`. The `+1` is the whole puzzle: a list of 4 returns lines up against days 2 to 5, not 1 to 5. Doing this by hand for four returns is exactly why loops (the next module) exist.")

ex("I5", "Recover yesterday's price, and prove it", 4,
r"""A stock closed today at **185.00**, up **2.5%** on the day. What was *yesterday's* close? The return
formula rearranged gives:

$$p_0=\dfrac{p_1}{1+r}$$

Compute it, then **check your own answer**: put your `p0` back into the ordinary return formula and
confirm you get 2.5% out again.""",
'''p1 = 185.00
r = 0.025

p0 = ...
check = ...

print("yesterday :", p0)
print("recomputed:", check)
print("matches   :", ...)''',
["Read the formula across: divide today's price by `(1 + r)`.",
 "The check runs the normal formula forwards: `(p1 - p0) / p0`. Compare it to `r` with `round(..., 10)` on both sides, because these are floats."],
'''p1 = 185.00
r = 0.025

p0 = p1 / (1 + r)
check = (p1 - p0) / p0

print("yesterday :", p0)
print("recomputed:", check)
print("matches   :", round(check, 10) == round(r, 10))''',
f"`{round(185.00/1.025, 2)}`, and the check returns 2.5% exactly. Today's price is yesterday's grown by `(1 + r)`, so dividing it back out recovers where you started.\n\nThe habit matters more than the answer: when you rearrange a formula, run it forwards again. It costs one line and it catches an inverted sign every time.")

ex("I6", "A weighted portfolio return", 3,
r"""A portfolio holds three stocks with **weights** 0.5, 0.3, 0.2 (they sum to 1) and daily returns
0.04, -0.01, 0.02. The portfolio return is each weight times its return, all added up:

$$r_p = w_1 r_1 + w_2 r_2 + w_3 r_3$$

Compute it and show it as a percentage.""",
'''w1, w2, w3 = 0.5, 0.3, 0.2
r1, r2, r3 = 0.04, -0.01, 0.02

portfolio_return = ...
message = ...
message''',
["Multiply each weight by its own return, then add the three products.",
 '`portfolio_return = w1*r1 + w2*r2 + w3*r3`, then format with `:.2%`.'],
'''w1, w2, w3 = 0.5, 0.3, 0.2
r1, r2, r3 = 0.04, -0.01, 0.02

portfolio_return = w1*r1 + w2*r2 + w3*r3
message = f"Portfolio: {portfolio_return:.2%}"
message''',
f"`'Portfolio: {(0.5*0.04 + 0.3*-0.01 + 0.2*0.02):.2%}'`. A weighted average is how every real portfolio return is built. The weights sum to 1, so the result always sits among the individual returns.")

ex("I7", "The middle price, no sorting", 5,
r"""Three desks quote **182.19**, **179.87**, and **183.48**. Find the **median** (the middle value)
**without** sorting them and without `if`.

Think about what is left over: if you take the running total of all three and remove the largest and
the smallest, what single value remains?""",
'''a = 182.19
b = 179.87
c = 183.48

median = ...
median''',
["The three values add up to a fixed total. The median is that total with the biggest and the smallest taken away.",
 "`median = a + b + c - max(a, b, c) - min(a, b, c)`. Note `max` and `min` accept several arguments, not only a list."],
'''a = 182.19
b = 179.87
c = 183.48

median = a + b + c - max(a, b, c) - min(a, b, c)
median''',
f"`{182.19 + 179.87 + 183.48 - max(182.19, 179.87, 183.48) - min(182.19, 179.87, 183.48)}`, which is 182.19 (the trailing digits are floating-point dust, and `round(median, 2)` tidies them away). Nothing new was needed, only a fresh way to combine `sum`-style addition with `max` and `min`. That reframing, the middle is what remains once the extremes are removed, is the kind of step the harder exercises look for.")

ex("I8", "Compounding beats adding", 3,
r"""Four daily returns are given. The true return over all four days is **not** their sum: each day
compounds on the last, so you multiply the growth factors and subtract 1:

$$r_\text{total} = (1+r_1)(1+r_2)(1+r_3)(1+r_4) - 1$$

Compute the compounded return and format it as a percentage.""",
'''r = [0.012, -0.004, 0.008, -0.011]

compounded = ...
message = ...
message''',
["Each factor is `1 + r[k]`. Multiply all four together, then subtract 1.",
 "`compounded = (1 + r[0]) * (1 + r[1]) * (1 + r[2]) * (1 + r[3]) - 1`."],
'''r = [0.012, -0.004, 0.008, -0.011]

compounded = (1 + r[0]) * (1 + r[1]) * (1 + r[2]) * (1 + r[3]) - 1
message = f"Compounded: {compounded:.2%}"
message''',
f"`'Compounded: {((1+0.012)*(1-0.004)*(1+0.008)*(1-0.011) - 1):.2%}'`, against a naive sum of `{sum([0.012,-0.004,0.008,-0.011]):.2%}`. Close over a few small days, but the gap grows fast, which is why returns are compounded, not added.")

ex("I9", "The one-line risk report", 4,
r"""The capstone. From one week of closes, build a **single** formatted line reporting the week's return
(as a percent), the average price (2 decimals), and the highest and lowest prices. Aim for something
like:

`Week 3.68% | avg 74.10 | high 76.00 | low 70.50`""",
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4, 73.9, 75.2, 74.8, 76.0, 75.5]

week_return = ...
average = ...
report = ...
report''',
["Build the pieces first: `week_return` (first vs last), `average` (`sum`/`len`), and use `max`/`min` directly in the f-string.",
 'The f-string can hold it all: `f"Week {week_return:.2%} | avg {average:.2f} | high {max(closes):.2f} | low {min(closes):.2f}"`.'],
'''closes = [71.2, 72.0, 70.5, 73.1, 74.4, 73.9, 75.2, 74.8, 76.0, 75.5]

week_return = (closes[-1] - closes[0]) / closes[0]
average = sum(closes) / len(closes)
report = f"Week {week_return:.2%} | avg {average:.2f} | high {max(closes):.2f} | low {min(closes):.2f}"
report''',
f"`'Week {((75.5-71.2)/71.2):.2%} | avg {(sum([71.2,72.0,70.5,73.1,74.4,73.9,75.2,74.8,76.0,75.5])/10):.2f} | high {max([71.2,72.0,70.5,73.1,74.4,73.9,75.2,74.8,76.0,75.5]):.2f} | low {min([71.2,72.0,70.5,73.1,74.4,73.9,75.2,74.8,76.0,75.5]):.2f}'`. Returns, an average, extremes, formatting: every idea from today working together. This is the kind of summary an analyst produces.")

# ---------------------------------------------------------------- closing
md(
"## \U0001f3c1 Done\n\n"
"That covers all of Session 1: running and formatting output, arithmetic with "
"powers and percentages, variables, types, functions and methods, and lists, "
"ending with a few realistic tasks such as a weighted return and a one-line risk "
"report.\n\n"
"**Next:** open `session_01_case.ipynb` (*The Analyst's Notebook, Part 1*) and "
"apply these tools to a full year of real Apple and Coca-Cola prices.\n\n"
"*Stuck for more than 15 minutes on anything? Ask a friend, ask an AI for a hint "
"(not the answer), or email me at `jobo@econ.au.dk`.*"
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
