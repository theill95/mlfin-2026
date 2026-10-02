# -*- coding: utf-8 -*-
"""Build revision.html: what each session should leave you with, then a glossary.

The first half is copied from the decks themselves: every regular deck ends with
a "What stays with you" slide (a .recap-grid of .recap-item boxes), and Session 13
has "The routine behind every model". This reads them out of the rendered HTML,
so the page follows the decks whenever they are rendered again, and links each
session back to its slide. The second half is the glossary below, written by
hand: each term, the session that introduced it, and a line or two.

    python tools/build_revision.py

release.py runs it before the sidebar is stamped into the pages.
"""
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision.html"

# (term, sessions, definition). Sessions are where the term arrived.
GLOSSARY = [
    ("Accuracy", [4, 8], "The share of predictions that match the label. Misleading when one class is rare: always predicting the majority class can score well."),
    ("AUC", [8], "The area under the ROC curve: the chance that a randomly chosen 1 gets a higher score than a randomly chosen 0. 0.5 is a random ranking and 1 a perfect one; no threshold is involved."),
    ("Bagging", [11], "Grow many deep trees, each on a bootstrap sample of the training rows, and average their forecasts. The average cancels much of what one tree gets wrong by chance."),
    ("Baseline", [4, 5], "The simplest rule a model has to beat: the training average, the majority class, or persistence. A score means little until it sits next to the baseline's."),
    ("Bias and variance", [4], "Bias is the error of a model too simple to follow the truth; variance the error of a model so flexible it follows the noise of its training rows. More flexibility trades the first for the second."),
    ("Boosting", [12], "Start from the mean and add small trees one at a time, each fitted to what the trees so far still get wrong and shrunk by the learning rate."),
    ("Bootstrap sample", [11], "As many rows as the data, drawn at random with replacement, so some rows come twice and about a third not at all."),
    ("Calibration", [9], "Whether a probability means what it says: of the days given 0.3, about 30 percent should turn out to be 1s."),
    ("Classification", [4, 8], "Predicting a label, such as whether next month will be busier than this one, rather than a number."),
    ("Confusion matrix", [4, 8], "The four counts behind every score of a classifier: true positives, false positives, false negatives and true negatives."),
    ("Cross-validation", [5], "Choosing between models with the training rows only: fit on some folds, score on the next, and average. Time-series folds always score rows that come after the ones fitted on."),
    ("Data leakage", [4], "Information the forecast could not have had at the time, slipping into a feature or into the fit, such as a scaler fitted on the test rows. It makes a model look better than it is."),
    ("Decision tree", [11], "A model that asks one question of one column at a time and forecasts the mean, or the class shares, of the training rows in each leaf."),
    ("Early stopping", [12], "Add trees until the score on later rows stops improving, and keep the number of trees at which it was best."),
    ("Elastic net", [6], "The ridge and lasso penalties at once, in a mix set by <code>l1_ratio</code>."),
    ("F1", [4, 9], "The harmonic mean of precision and recall, which sits close to the smaller of the two."),
    ("Feature", [4], "A column used to predict, built only from what was known when the forecast is made."),
    ("Feature importance", [12], "How much a model leans on each column: from its cuts, measured on the training rows, or by permutation, on rows it did not fit."),
    ("Fold", [5], "One of the blocks the training rows are cut into for cross-validation."),
    ("Gini impurity", [11], "How mixed the labels in a box are: 2p(1 - p) for two classes, and zero when the box is pure. A classification tree cuts where it falls most."),
    ("Gradient", [12], "The direction in which a loss rises fastest. For squared error, minus the gradient is the residual, which is why boosting fits its trees to residuals."),
    ("Grid search", [6], "Trying every value of a setting on the folds and keeping the best; <code>GridSearchCV</code> then refits it on all the training rows."),
    ("Hyperparameter", [6], "A setting chosen before fitting, such as alpha, C, k, the depth of a tree or the number of trees. Chosen on validation rows, never on the test rows."),
    ("k-nearest neighbours", [9], "Classify a row by a vote among the k training rows closest to it, on standardised columns."),
    ("Lasso", [6], "Linear regression with a charge on the sum of the absolute coefficients. It sets some coefficients to exactly zero."),
    ("Learning rate", [12], "The share of each new tree that boosting adds. A smaller rate needs more trees for the same fit."),
    ("Log loss", [8], "What logistic regression minimises: minus the log of the probability given to what happened. A confident wrong probability costs the most."),
    ("Logistic regression", [8], "A linear model for a label: a weighted sum of the columns, put through the sigmoid to become a probability."),
    ("MAE, MSE and RMSE", [4], "The mean absolute, mean squared and root mean squared error of a forecast of a number. The RMSE is in the target's own units."),
    ("Majority class", [8, 9], "Predicting the most common label everywhere: the baseline for accuracy."),
    ("Mean reversion", [5], "A value far from normal tends to be followed by one closer to normal, such as a calm month after a busy one."),
    ("Monotone constraint", [13], "Forcing a model's forecast to rise, or to fall, as one column rises, everything else unchanged."),
    ("Out-of-bag", [11], "Scoring each training row with the trees whose bootstrap sample left it out. Fair for separate cases; too kind on days, whose neighbours were in the sample."),
    ("Overfitting", [4], "Learning the noise of the training rows along with their pattern: a lower training error and a higher test error."),
    ("Penalty", [6], "A charge on the size of the coefficients, added to what the fit minimises, with a strength set by alpha, or by 1/C in logistic regression. Also called regularisation."),
    ("Permutation importance", [12], "How far a model's score falls when one column is shuffled, on rows the model did not fit."),
    ("Persistence", [4, 5], "Forecasting that the next value equals the latest one: next month's volatility is this month's. Hard to beat in finance."),
    ("Pipeline", [6], "Steps chained into one model, such as a scaler and then ridge, so cross-validation refits every step inside every fold."),
    ("Precision", [4, 8], "Of the times the model said yes, the share that were right."),
    ("Random forest", [11], "Bagging in which each question chooses among a few columns drawn at random, so the trees differ more and their average is steadier."),
    ("Recall", [4, 8], "Of the real yeses, the share the model caught."),
    ("Recursive binary splitting", [11], "How a tree grows: the best single cut, then the best cut inside each new box, never revising an earlier one."),
    ("Regression", [4, 5], "Predicting a number, such as next month's volatility."),
    ("Residual", [5, 12], "What happened minus what was forecast."),
    ("Return", [1, 4], "The relative change in a price, (p1 - p0) / p0. The log return, log(p1 / p0), adds up over time."),
    ("Ridge", [6], "Linear regression with a charge on the sum of the squared coefficients: every coefficient shrinks towards zero, none to exactly zero."),
    ("ROC curve", [8], "The recall against the false positive rate, at every threshold."),
    ("Sigmoid", [8], "1 / (1 + exp(-z)): turns any number into a probability between 0 and 1."),
    ("Softmax", [9], "The sigmoid for more than two classes: exponentiate each class's score and divide by the total."),
    ("Standardise", [4, 6], "Subtract a column's mean and divide by its standard deviation, both learned on the training rows."),
    ("Stump", [11, 12], "A tree with one question and two leaves."),
    ("Supervised learning", [4], "Learning a rule from rows whose answer is known. Unsupervised learning looks for structure with no answer column."),
    ("Target", [4], "The column to predict, set on the row of the features it is predicted from."),
    ("Test rows", [4], "Rows held back until the end and used once, to score the model that was chosen."),
    ("Threshold", [8, 9], "The probability from which a classifier says 1. The default is 0.5; when the two mistakes cost different amounts, another threshold is cheaper."),
    ("Training rows", [4], "The rows a model is fitted on."),
    ("Validation", [5], "Scoring candidates on rows they were not fitted on, before the test rows: the folds, or a later block."),
    ("Volatility", [2, 3], "The standard deviation of returns: how much a price moves. Annualised by the square root of 252 trading days."),
    ("XGBoost and LightGBM", [12], "Fast boosting libraries that cut each column only between bins. LightGBM grows its trees leaf by leaf."),
]


def deck_title(n):
    text = (ROOT / f"session_{n:02d}" / f"session_{n:02d}.html").read_text(encoding="utf-8")
    return html.unescape(re.search(r"<title>(.*?)</title>", text, re.S).group(1)).strip()


def recap(n, slide_id_prefix):
    """The recap items of one deck, as HTML, and the slide's own id."""
    text = (ROOT / f"session_{n:02d}" / f"session_{n:02d}.html").read_text(encoding="utf-8")
    m = re.search(rf'<section id="({slide_id_prefix}[^"]*)"[^>]*>(.*?)</section>', text, re.S)
    if not m:
        return None, []
    items = re.findall(r'<div class="recap-item">\s*(.*?)\s*</div>', m.group(2), re.S)
    return m.group(1), items


def slug(term):
    return "g-" + re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")


sessions = []
for n in range(1, 14):
    slide, items = recap(n, "what-stays-with-you")
    if items:
        sessions.append((n, slide, items))
assert len(sessions) >= 10, f"only {len(sessions)} decks have a recap slide"
routine_slide, routine = recap(13, "the-routine-behind-every-model")
assert routine, "Session 13's routine slide is missing"

parts = []
for n, slide, items in sessions:
    boxes = "\n".join(f'        <div class="rv-item">{item}</div>' for item in items)
    parts.append(f'''  <section class="card rv-session" id="s{n}">
    <h2>Session {n} · {html.escape(deck_title(n))}</h2>
    <p class="card-meta"><a href="session_{n:02d}/session_{n:02d}.html#/{slide}">the slide</a> ·
       <a href="session_{n:02d}/session_{n:02d}.html">the lecture</a> ·
       <a href="downloads/lectures/session_{n:02d}.pdf">its PDF</a></p>
    <div class="rv-grid">
{boxes}
    </div>
  </section>''')

routine_boxes = "\n".join(f'        <div class="rv-item">{item}</div>' for item in routine)
glossary = "\n".join(
    f'    <dt id="{slug(term)}">{term} <span class="g-s">'
    + ", ".join(f'<a href="session_{s:02d}/session_{s:02d}.html">Session {s}</a>' for s in ss)
    + f"</span></dt>\n    <dd>{text}</dd>" for term, ss, text in GLOSSARY)
toc = " ".join(f'<li><a href="#s{n}">Session {n}</a></li>' for n, _, _ in sessions)

page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Revision · Machine Learning in Finance</title>
<meta name="description" content="What each session of Machine Learning in Finance should leave you with, the routine behind every model, and a glossary of the course's terms.">
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<meta property="og:site_name" content="Machine Learning in Finance">
<meta property="og:title" content="Revision · Machine Learning in Finance">
<meta property="og:description" content="What each session of Machine Learning in Finance should leave you with, the routine behind every model, and a glossary of the course's terms.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://theill95.github.io/mlfin-2026/revision.html">
<meta property="og:image" content="https://theill95.github.io/mlfin-2026/assets/social-card.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<link rel="stylesheet" href="assets/site.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<div class="shell">

  <!-- nav:start -->
  <!-- nav:end -->

  <main class="wrap" id="main">
  <header>
    <div class="accent"></div>
    <div class="eyebrow">Generated from the lectures</div>
    <h1>Revision</h1>
    <p class="lede">What each session should leave you with, as its last slide puts it,
       then the routine every model went through, and a glossary of the course's terms.
       Each session links back to its slide.</p>
  </header>

  <ul class="cs-toc">
    {toc} <li><a href="#routine">The routine</a></li> <li><a href="#glossary">Glossary</a></li>
  </ul>

{chr(10).join(parts)}

  <section class="card rv-session" id="routine">
    <h2>The routine behind every model</h2>
    <p class="card-meta">From Session 13: <a href="session_13/session_13.html#/{routine_slide}">the slide</a></p>
    <div class="rv-grid">
{routine_boxes}
    </div>
  </section>

  <h2 id="glossary">Glossary</h2>
  <p class="lede">The course's terms, each with the session that introduced it.</p>
  <dl class="glossary">
{glossary}
  </dl>

  <footer>
    Built from the lectures by <code>tools/build_revision.py</code>. Something missing?
    Email <a href="mailto:jobo@econ.au.dk">jobo@econ.au.dk</a>.
  </footer>
  </main>
</div>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script>
  // The recap boxes keep the decks' maths as TeX in span.math; typeset it once KaTeX is in.
  window.addEventListener("load", function () {{
    if (!window.katex) return;
    document.querySelectorAll("span.math").forEach(function (el) {{
      katex.render(el.textContent, el, {{ throwOnError: false, displayMode: el.classList.contains("display") }});
    }});
  }});
</script>
</body>
</html>
'''
OUT.write_text(page, encoding="utf-8", newline="\n")
print(f"wrote {OUT.name}: {len(sessions)} sessions, {sum(len(i) for _, _, i in sessions)} recap items, "
      f"{len(routine)} routine items, {len(GLOSSARY)} glossary terms")
