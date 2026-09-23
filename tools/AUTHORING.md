# Authoring a session

How the lecture deck, the exercises and the case for a session are written,
distilled from the review rounds on Sessions 1 to 6. `tools/README.md` covers
the mechanics (build, check, publish); this page covers what the material
has to be like. Read it before writing anything, and read it again before
handing a draft over for review.

Who it is for: master's students in finance and economics who know OLS and
started the course with no Python. Everything is calibrated to what they have
actually written in the exercises so far, not to what is idiomatic.

---

## 0 · The rules that apply to everything

**Voice.** Plain and explanatory. Complete sentences, ordinary words, one fact
per sentence, and stop. The register is a careful academic talking, not
editorial copy. Second person; never "I". No em-dashes anywhere. Say "last
time" and "next time", never "Session 4", never "last week".

The seven habits that get a draft rejected, and what to do instead:

| habit | example that was cut | do this instead |
|:--|:--|:--|
| verbless emphasis fragments | "Three lines." "No fitting at all." "Both penalties at once." | make it a sentence with a subject and a verb |
| stage directions about the deck | "the next slide asks", "a few slides ago", "it is the subject of Part 3" | say the thing; do not narrate the deck |
| knowing asides | "everybody trips over that once", "nothing magic inside", "the laziest rule you can think of" | delete |
| litotes | "not bad, and clearly not right either", "it has earned nothing" | say it directly |
| anthropomorphism | "the model wants", "no test row was touched", "training error votes for" | give the verb to the analyst or the code |
| aphoristic closers | "One cut date is one experiment." "The spread is information." | end on the specific consequence |
| elegant variation | "the scored block", "a stretch of market", "the rows it has never seen" for the same thing | pick one name and repeat it |
| decorative trailing clauses | "the threshold, and where the cheapest one sits", "the probabilities, and how to put them right", "a second classifier, judged on the same folds", "a second table, where the event worth predicting happens to one borrower in five" | cut everything after the noun. "the threshold, set from the cost of each mistake" |
| coy phrasing for a plain thing | "where the cheapest one sits" for the cheapest threshold, "how to put them right" for calibrating them, "what the fitted model grows" for the extra rows of `coef_` | name the thing |

Also out: marketing and "bro" register ("workout", "grab a coffee", "that's a
wrap"), taglines, catchphrases, motivational lines, and adjectives that
editorialise ("honest", "elegant", "beautiful") where a number would do.

**Length, as a hard limit.** A paragraph under a code cell or a figure is **at
most two sentences and about 30 words**. Three sentences is already too long;
so is one sentence with two subordinate clauses. Two examples, both cut in
review for being too long:

> Last time a logistic regression on two columns predicted whether the index's
> volatility over the next 20 days would be higher than over the last 20. On
> the 482 test days it was right on 74 percent of them, against 53 percent for
> predicting the majority class. *(47 words: borderline)*

> The label is 1 on 48 percent of the days, so the two classes are almost the
> same size and a day with a rise is as common as a day without one. The
> decisions a bank, an insurer or a regulator makes are rarely like that: the
> event worth predicting happens to a small share of the cases, and being wrong
> in one direction costs more than being wrong in the other. *(73 words: far
> too long)*

The second became: "The label is 1 on 48 percent of the days, so the two
classes are almost the same size. Most decisions a bank or an insurer makes are
not like that." Everything else was either already obvious or belonged on
another slide. Audit a finished deck by counting: any block over 36 words is a
rewrite, and the median slide should sit near 25.

**A slide carries one text size besides its code or figure.** Body prose, a
`.keypoint`, a `.formula` with its `.formula-note`, a `.muted .small` aside and
a `.why-grid` are five different sizes; a slide may use one of them, not three.
The pattern that keeps getting flagged is a formula, then a small centred note,
then a full-size paragraph: pick the formula plus one paragraph and fold the
note into it. Prose plus a `.keypoint` is fine, because the keypoint is a
boxed conclusion rather than a second voice. Prose plus a `.muted .small` is
fine only on a "Your turn" slide, where the small text is the hint.

**Numbers are measured, never asserted.** Every figure on a slide and every
value in a solution note is computed from the real data by the same code the
student runs. If a claim cannot be measured on this dataset, change the
example until it can (Session 6 switched from Apple to the index because on
Apple no penalised model beat one column). When the honest result is
inconvenient, keep it and say so; that is usually the better lesson.

**One model family per session, only tools already taught.** Nothing appears
before it has been introduced, and a new function is explained at its first
use. If an exercise needs something the deck did not cover, the task names it
in one line. The lecture stands alone: it does not assume the previous
session's exercises or case were done.

**The scale of a piece of code** is set by the previous session's exercise
work cells. At the end of Session 4 a three-star task was one expression;
Session 5's deck therefore used bare expressions and `print`. No
comprehensions, no `%`-formatting, no `pd.Series(values, index=...)`, no
clever indexing, until an exercise has made students write them.

**Look backwards on purpose.** Later sessions keep re-testing loops,
functions, dictionaries, f-strings, boolean masks, vectorised arithmetic and
matplotlib, in the new session's context. The skills are meant to accumulate.

---

## 1 · The lecture deck

### Shape

- A Quarto `live-revealjs` deck, 1280 x 720, flat (every slide is a level-1
  `#`), roughly 60 to 70 slides. Many small slides, one idea each. Nothing
  may go past 720 px; audit every deck for overflow.
- Divider slides between parts, "Today" up front, and at the end a recap
  grid, a "where this goes next" timeline and a "before next time" slide with
  the exercises, the case and the help line (`jobo@econ.au.dk`).
- Every "Your turn" is followed by an answer slide with static code and the
  result, so nothing has to be typed live. Your-turn and predict cells have
  `autorun: false`; demonstration cells autorun.

### Order within a topic

Never open a topic with the code or with the answer. Inside every part:

1. a problem is met, on the running example, with the tools from last time;
2. why the problem exists, with a figure or a diagram that shows the
   mechanism;
3. how it is solved, as an idea;
4. how to do that in Python.

Code comes last. A divider followed directly by a live cell is wrong. The
chain should be crisp: "524,288 subsets, in or out; infeasible; solution:
charge for coefficient size" is one visual step, not three prose slides. Do
not let the pattern dominate to the point where sections feel mechanical, but
when in doubt, add the motivating slide.

### Starting a deck

Begin where the previous deck ended, with its own tools running on the
running example, and reach the new idea by showing where the old tool stops.
Restate only what the new deck needs (three numbers in a stat band is
enough). Do not build the dataset live: ship a prepared CSV, load it in one
cell, and make building it an exercise.

### What goes on a slide

- **The code cell is the slide.** One sentence of setup, a 3 to 8 line live
  `{pyodide}` cell, one or two sentences on what ran. Roughly two thirds of
  teaching slides carry code; a slide about an idea gets a figure or a
  diagram instead. Code must earn its place; forcing a cell onto every slide
  reads as padding.
- **One short paragraph per slide**, two or three lines, or a figure plus two
  lines. A large-font paragraph followed by a small-font paragraph looks
  unbalanced and was flagged repeatedly. `.muted .small` asides are for one
  line, not for the overflow that did not fit. Do not fix an overflow by
  demoting text.
- **Titles name the thing being taught** ("The loop variable", "Ridge in
  code"), never a slogan, a conclusion or a vague pronoun ("Why it gets it
  wrong"). A question is allowed only where the slide answers it. Titles cut in
  review: "An accuracy that looks respectable" (editorial), "One number for the
  whole page" (coy), "The errors are almost never two steps" (a conclusion),
  "Two steps to fix that" (points at the previous slide). They became "Accuracy
  against the majority rule", "The Brier score", "Where the mistakes fall" and
  "Making the scores into probabilities".
- **Visuals do the arguing.** Diagrams for mechanisms (SVG, flush-left, no
  blank lines inside), matplotlib figures precomputed in `{python}` cells,
  fragment animations where a process unfolds (a loop trace, folds taking
  turns, a grid of alphas by folds). Restrained look: deep blue, sparing
  amber and brick, off-white ground, no decorative animation, no rainbow
  palettes, one axis per chart, plain tick labels ("1,000" not "10^3").
- **Check every rendered figure** for labels colliding with lines or axes,
  legends over data, clipped circles, and every SVG for text overlaying
  another element. These are spotted instantly in review.
- Maths in KaTeX where it helps, stated once, with the terms named in a
  short list underneath.
- **Arguments of a new function** are explained when they matter, and a
  session that introduces objects with many settings gets one "when to
  change what" slide (always / sometimes / rarely, each with a few words on
  what the setting is) rather than tables of every argument.
- **Build a new idea up, never drop it in as a formula.** The order that
  survived review, for the softmax: the concrete case first (one day, three
  scores, and why they are not probabilities), then the fix on those actual
  numbers (exponentiate: 1.061, 0.724, 1.302; divide by 3.087), then the
  general statement with N classes and its name, then the code. A formula
  slide that arrives before the reader has seen the arithmetic gets cut. Say
  in one line how the new idea relates to what is already known ("with two
  classes it is the sigmoid from last time") rather than re-deriving it.
- Every number quoted in prose must match the live cell's output. Wrap bare
  expressions in `print()` (NumPy 2 prints `np.float64(...)`), and end a cell
  with `print(model)` rather than the estimator's HTML widget.

### Before handing a deck over

Render it, run the overflow audit after the autorun cells have finished,
click-test every live cell in a browser, screenshot the figure and diagram
slides, and run the prose checklist above over every slide. Hand over a
served URL: opened from `file://` the live cells do not appear at all, and the
deck will be reviewed as having no code.

Run these four counts over the finished `.qmd` before handing it over. Each of
them has caught something that would otherwise have come back in review:

1. **Word count per prose block.** Anything over 36 words is a rewrite.
2. **Text sizes per slide.** More than one of {prose, `.formula-note`,
   `.muted`, `.why-grid`, `.lede`} on a slide is a rewrite.
3. **Titles.** Read the list of `^# ` lines on its own. Every one should name a
   thing; none should state a conclusion, carry an adjective, or point at
   another slide.
4. **Decorative clauses.** Grep the prose for `, and how`, `, and where`,
   `, and what`, `judged on`, `which is what`. Most hits are the habit above.

And one rendering trap, because it is silent: **never put a `###` heading
inside a `::: {.compare}` or any other pandoc fence on a reveal slide.** Pandoc
promotes the heading to its own `<section>`, which closes the slide early, so
the title renders on top of the content and the overflow audit reports a
meaningless small number instead of flagging it. Write two-column blocks as raw
HTML (`<div class="compare"><div class="compare-col"><h3>..</h3><p>..</p>`), as
Session 8 does. After rendering, assert that every `slide level1` section
contains more than its own `<h1>`.

---

## 2 · The exercises

### Purpose and size

About 50 to 60 exercises per session, mostly write-from-scratch or
fill-in-the-blank. They are where the learning happens; the lecture only
introduces. Test everything the lecture covered, from more than one angle,
and keep re-testing the earlier sessions' core skills in the new context.

### Structure

- One notebook, generated by `tools/generators/session_NN_exercises.py` and
  never edited by hand. Intro (pleasant and plain; students are not expected
  to finish; short on time means read the hint, then the solution), the
  difficulty table, a toolkit card with `title=` hover docs for everything
  new this session, the formulas, then one setup cell, then sections A to K.
- Per exercise: title with a 1 to 5 star badge (no tier names; the scale
  restarts each session and rates the work against what this session
  taught), an optional `revisits Sn` tag when an earlier session's tool is
  load-bearing, the task, one work cell with `...` blanks, one or two folded
  hints, a folded solution with the code and one plain paragraph that states
  the number and what it means.
- Use at least four of the five star levels; sanity-check the histogram.
- Vary the shapes: fix-the-bug, do-it-two-ways-and-check-they-agree,
  refactor a loop into one line, fix-the-figure, write-the-assertion,
  from-scratch with no scaffold, a `while` with a `break`, a function that
  wraps an earlier cell.
- **About six exercises per notebook ask the student to draw something**, and
  most of them are one or two stars. Sessions 5 and 6 have three, Session 8 has
  six; a session that ends up with two was sent back. The simple shapes are
  enough: a bar chart of a rate with a dashed reference line, a swept curve
  with the chosen point marked, one score per class as bars, two lines against
  a setting on a log axis with a legend. Count them before handing over.

### Self-contained, with short clusters

Exercises stand alone by default. Where a chain is natural (build a table
step by step; fit a pipeline, then read it, then cross-validate it) a
cluster of two to four exercises shares one setup, the intro lists the
clusters, and each task names what it continues from ("`ols` from B1"). A
later section that needs a function from an earlier one says so in its
header.

**A work cell never pre-fills an earlier exercise's answer**, even if it is
lecture-level code: a student who peeks down would see the solution.
Outside a cluster the student refits what is needed (a two-line fit as a
blank is fine). Pre-filled lines are limited to scaffolding no exercise asked
for (`spot = columns.index('vol_20d')`, `fig, ax = plt.subplots(...)`), and to
deliberate bugs the exercise exists to fix. Loop variables must not overwrite
a cluster's shared names.

### Blank safety

Every work cell must run cleanly with its `...` untouched, so Run-all never
floods the student with errors. A blank is the right-hand side of an
assignment, a bare `...` statement, or an argument to `print()`. Pre-written
lines never call a method on, index into, iterate over, do arithmetic with,
or pass as an argument a placeholder, and never call anything on an object a
chained exercise was supposed to produce. `while True:` is not blank-safe;
bound the loop. `frame['x'] = ...` is not blank-safe if the frame is fitted
on later. A function whose body is `...` returns `None`, so nothing may call
`.shape` on its result. Deliberate-error demonstrations use the two-cell
pattern: a cell that really errors, tagged `raises-exception`, then a blank
fix cell.

### What the solutions must be

Correct against the real data, with every quoted number computed in the
generator's preamble from the same code. Outputs print plain numbers, not
`np.float64(...)` (wrap with `float()`, `int()`, `round()`). Solution notes
state the result and what it means; they do not moralise and they do not
copy the deck's sentences.

### Checks before handing over

`nbconvert --execute` on the untouched notebook; the solution verifier
(all solutions, in order, in one namespace, against the real data);
`tools/verify/blank_safety.py`; `colab_markup.py`; `exercise_ladder.py`;
`colab_data_access.py` once the data is on GitHub. Then read the whole
notebook as a student would.

---

## 3 · The case

### What it is

One investigation, the risk report on the eleven-instrument universe, that
runs across every session and builds within each part. Later questions reuse
the variables earlier ones stored, and each part ends by framing the next.
About 15 to 18 questions, no star badges, a natural easy-to-hard progression.

### Continuity

- Open with a **quick load** that restores the previous part's facts as
  named variables and prints them, so the investigation continues without
  re-reading the earlier notebook.
- Q1 rebuilds and re-verifies where the previous part stopped (one cell, a
  `True`).
- Units and the instrument carry forward: the case has used plain decimals
  and Apple since Part 1, even when the lecture used percent and another
  instrument. State the difference in the quick load and use it when it
  teaches something (in Part 6, the lecture's alpha on raw decimal columns
  collapses the model to the average guess).
- Close with a table of what now exists and where it lives, "what changed
  since the previous part", "where this leaves the risk report", and one
  line on the next part.
- The last question is always the report: a dictionary of what would be
  defended and a `summarise()` function that prints it with a verdict.

### Different from the exercises

Same tools, different questions. Where the exercises and the case would run
the same computation, give it to one of them and find something else for the
other (Part 6: the case counts where the wide model beats one column across
the desk and splits the test years into calm and busy days; the exercises do
Nvidia, which alpha each instrument picks, and how many columns the lasso
keeps per instrument). No verbatim copying from the lecture either: the
homework must not let a function or snippet the deck showed be pasted; make
it assembled from separate steps, or applied to different data.

### Honesty

The case reports what the data says. Part 5 found the persistence rule
positive on only 4 of 11 instruments; Part 6 found the penalised wide model
losing to one column on Apple and winning on 7 of 11. Those go in the report
with both numbers. A result checked on one stock is one observation.

### Blank safety, cumulative version

Same rules as the exercises, plus: no pre-written line may call anything on
a variable an earlier question produced, because that question may still be
blank. `len()` of an earlier variable, indexing a dictionary an earlier
question fills, iterating `folds.split(train)` where `train` is an earlier
blank, all break Run-all. Every such dependency sits inside the student's own
blank. Test a case two ways: blank Run-all from the session folder, and every
solution executed in order in one shared namespace with the closing table's
promised variables checked for presence.

---

## 4 · Handing over

- Say what was measured and what was decided without the reviewer, in a few
  lines each. Do not say "I".
- Give a served URL for a deck, and the file paths for notebooks.
- Record every correction from the review in the memory notes, with the
  reason, before starting the next session.
