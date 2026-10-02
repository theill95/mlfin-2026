# Machine Learning in Finance

Teaching materials for a master's-level course introducing machine learning
to finance and economics students. The course assumes econometrics (OLS) but
**no prior Python or machine-learning experience**, and begins in Google
Colab.

Each session is a single self-contained folder. All thirteen sessions are complete.

## Live site

This repository is published:

- **Home page (put this link on the learning platform):**
  <https://theill95.github.io/mlfin-2026/>
- Interactive lecture, Session 1:
  <https://theill95.github.io/mlfin-2026/session_01/session_01.html>
- Interactive lecture, Session 2:
  <https://theill95.github.io/mlfin-2026/session_02/session_02.html>
- Interactive lecture, Session 3:
  <https://theill95.github.io/mlfin-2026/session_03/session_03.html>
- Interactive lecture, Session 4:
  <https://theill95.github.io/mlfin-2026/session_04/session_04.html>
- Interactive lecture, Session 5:
  <https://theill95.github.io/mlfin-2026/session_05/session_05.html>
- Interactive lecture, Session 6:
  <https://theill95.github.io/mlfin-2026/session_06/session_06.html>
- Interactive lecture, Session 7:
  <https://theill95.github.io/mlfin-2026/session_07/session_07.html>
- Interactive lecture, Session 8:
  <https://theill95.github.io/mlfin-2026/session_08/session_08.html>
- Interactive lecture, Session 9:
  <https://theill95.github.io/mlfin-2026/session_09/session_09.html>
- Interactive lecture, Session 10:
  <https://theill95.github.io/mlfin-2026/session_10/session_10.html>
- Interactive lecture, Session 11:
  <https://theill95.github.io/mlfin-2026/session_11/session_11.html>
- Interactive lecture, Session 12:
  <https://theill95.github.io/mlfin-2026/session_12/session_12.html>
- Interactive lecture, Session 13:
  <https://theill95.github.io/mlfin-2026/session_13/session_13.html>
- Repository: <https://github.com/theill95/mlfin-2026>

The site has ten pages, linked from the sidebar:

| page | what it is |
|:--|:--|
| `index.html` | the sessions: lecture, exercises and case for each, and the questions students ask |
| `case.html` | the course case on one page: how the parts work, each part's question, what it added and the number it ended on, with a Colab link to each |
| `revision.html` | each session's "What stays with you" slide, Session 13's routine, and a glossary of the course's terms. **Generated** by `tools/build_revision.py` from the rendered decks, so render a deck before running it |
| `cheatsheet.html` | every function the course has used, one row each: the call with general argument names, and what it does, with a filter by word and by session. **Generated** by `tools/build_cheatsheet.py`, which checks every name against the library it comes from, so do not edit it by hand. The earlier worked-example format is kept in `tools/archive/` |
| `search.html` | one search over every slide, exercise and case question, cheatsheet row, glossary term and page section; a slide result opens at that slide. It reads `assets/search-index.json`, **generated** by `tools/build_search.py` from the rendered decks, the notebooks and the built pages. `/` on any page jumps to a search box |
| `math.html` | the maths the course assumes: matrix and vector notation, dimensions, vectorisation, inverses and multicollinearity, and how to read the objective functions behind OLS, ridge, lasso, logistic regression and trees. Hand-written; maths renders through KaTeX from a CDN |
| `setup.html` | installing Python and VS Code, and the extras (scripts vs notebooks, virtual environments, Git), each marked needed or optional |
| `resources.html` | official docs, ISLP, Kaggle, the AI-use policy, the data and its terms, and how the materials may be reused |
| `downloads.html` | every notebook and CSV, every lecture as a PDF, plus `downloads/mlfin-course.zip` |
| `exam_info.html` | the exam at a glance, a get-ready checklist, the offline copies (course ZIP, the 13 lecture PDFs, cheatsheet and maths PDFs, `setup_check.ipynb` in the ZIP), the practicalities guide and three mock exam student bundles |

The PDFs are built by `tools/build_handouts.py` (Playwright drives Edge): it opens
each deck in Reveal's PDF view, scrolls every slide into view until every live cell
has run, prints the deck once, and splices in one page per step for the few slides
that build up in replacing steps. Printing from the browser would lose the live
cells' output. `release.py` runs it and rebuilds only what changed;
`tools/verify/handouts.py` fails if a deck was rendered again without its PDF.

Every page has a light and a dark look: colours are tokens at the top of
`assets/site.css`, the dark set follows the system unless the reader picks one
with the sidebar's toggle (remembered in localStorage), and print is always
light. `favicon.svg` and `assets/social-card.png` give the site its icon and its
link preview (each page carries description and Open Graph tags); `404.html`
answers a missing address with the sidebar; `.nojekyll` tells GitHub Pages to
serve the files as they are, without Jekyll.

Shared styling lives in `assets/site.css` and the sidebar in `assets/nav.html`,
which `tools/build_nav.py` stamps into every page. The student ZIP is built from
an allowlist by `tools/build_download_bundle.py`, so `tools/`, `_extensions/`
and this README are never handed to students, and no page links to the
repository. Every push to `main` updates the site automatically within a minute or two.

## Giving students one simple link (recommended)

Students should never need a terminal, a script, or a download to get an
interactive version. The repository is already published to **GitHub Pages**;
just put the home-page link above on the learning platform. It shows a clean
menu with one button for the **interactive lecture** and one button each for
the **exercises** and **case** that open straight in Google Colab.

If you ever recreate the site from scratch, the one-time setup is:

1. Put this repository on GitHub as a **public** repo (Colab can only open
   notebooks from public repos).
2. In the repo: **Settings → Pages → Build and deployment → Deploy from a
   branch**, pick `main` and `/ (root)`, save.
3. Your materials are now live at
   `https://<your-user>.github.io/<your-repo>/` — that is the link you give
   students. It always shows the latest version you push.

The home page builds the "Open in Colab" links automatically from that address,
so there is nothing to edit. (Hosting somewhere other than GitHub Pages? Set
`REPO_OVERRIDE` at the bottom of `index.html` to `"user/repo"`.)

If you would rather **upload files to the learning platform** instead of
linking out: the exercise and case `.ipynb` files can be uploaded to Colab by a
student (in Colab, **File → Upload notebook**). But the *lecture's* live code
only works when the page is served over the web, so for the lecture the link
above is the smooth path.

**Data in Colab.** Every notebook that needs data looks for a local `data/`
folder first and, failing that, downloads what it needs from this repository's
raw URL. Colab has no `data/` folder, so that fallback is what makes the "Open
in Colab" buttons work with nothing to upload. If the repository ever moves,
update `REPO_RAW_URL` in `tools/generators/`, regenerate, and re-run
`python tools/verify/colab_data_access.py`, which proves the download path by
emptying the local search path and confirming the result still matches.

## What is in a session

```
session_01/
├── session_01.qmd            # the lecture: source for the slides + live code
├── session_01.html           # the rendered Reveal.js presentation (open this to teach)
├── session_01_files/         # assets the rendered HTML needs (keep alongside the .html)
├── session_01_exercises.ipynb    # exercises, with folded hints and solutions
├── session_01_case.ipynb         # the longitudinal case, Part 1, with folded solutions
└── data/                     # the small CSV files this session uses
```

Sessions 2 to 12 have the same shape, except 7 and 10, which like 13 are case
lectures with no notebooks. Current contents:

| session | lecture | exercises | case |
|:--|:--|:--|:--|
| 1 · Beginning Python for financial data | `session_01.qmd` | 55 | Part 1, 10 questions |
| 2 · Functions, loops and dictionaries | `session_02.qmd` | 64 | Part 2, 11 questions |
| 3 · Packages: NumPy, pandas and matplotlib | `session_03.qmd` | 76 | Part 3, 15 questions |
| 4 · Foundations of machine learning | `session_04.qmd` | 73 | Part 4, 17 questions |
| 5 · Model selection and cross-validation | `session_05.qmd` | 63 | Part 5, 18 questions |
| 6 · Penalised regression | `session_06.qmd` | 58 | Part 6, 16 questions |
| 7 · Pricing a house in Aarhus (recap) | `session_07.qmd` | none | none |
| 8 · Classification with logistic regression | `session_08.qmd` | 64 | Part 8, 15 questions |
| 9 · Classification in practice | `session_09.qmd` | 68 | Part 9, 16 questions |
| 10 · Electricity prices in West Denmark (recap) | `session_10.qmd` | none | none |
| 11 · Trees and forests | `session_11.qmd` | 62 | Part 11, 14 questions |
| 12 · Boosting and reading a model | `session_12.qmd` | 59 | Part 12, 15 questions (the last part) |
| 13 · The two cases with trees (recap) | `session_13.qmd` | none | none |

Sessions 3 to 12 load numpy, pandas and matplotlib into the browser runtime
(about 20 MB, once per page load). Sessions 5 to 13 add scikit-learn, which
brings scipy and a BLAS with it, so their cold load is nearer 45 MB; Sessions 12
and 13 also load xgboost, and Session 12 lightgbm, about 2 MB more. Session 13's
figures are drawn when the deck is rendered, so it loads no matplotlib. Open the
deck and run one cell several minutes before class so it is warm.

From Session 4 on, the decks have mathematics on the slides; it renders
through KaTeX, set in the `.qmd` front matter.

The `.qmd` file **is** the lecture: it is at once the slide deck, the lecture
narrative, and the source of every executable code demonstration. There is no
separate lecture notebook.

## Running the presentation

The rendered `session_01/session_01.html` opens in any modern browser and
supports keyboard navigation: arrow keys, `f` for fullscreen, `s` for the
speaker view, `m` for a menu of every slide by title, and `e` to lay the whole
deck out as one scrollable page (which is also how you print it to PDF).
Inside a code cell, `Ctrl+Enter` runs it; `Shift+Enter` does nothing there.

**Serve it over HTTP, do not open it from a `file://` path.** The lecture's
code cells are **live and editable** — you can run and change them in front of
the class. They use [quarto-live](https://r-wasm.github.io/quarto-live/), which
runs Python in the browser with Pyodide, and browsers block that runtime on
bare `file://` URLs.

On Windows, just **double-click `serve.cmd`** in the repository root, then open
the address it prints. Or, from any terminal in the repository root:

```bash
python -m http.server 8000
# then visit http://localhost:8000/session_01/session_01.html
```

### Getting the interactive lecture without a terminal (recommended)

The local server works, but the cleanest option is to **publish the deck once
to a web address** — then you (and anyone) just open a link, the live cells
work over HTTPS, and there is no script to run. Pick one:

```bash
# Option A — Quarto Pub (free, no repository needed; one command)
quarto publish quarto-pub session_01/session_01.qmd

# Option B — GitHub Pages (if the course lives in a GitHub repo)
quarto publish gh-pages session_01/session_01.qmd
```

Either gives a stable URL you can bookmark and reuse each year. Re-run the same
command to update it.

**About Google Colab.** The lecture deck is a Reveal.js *presentation*, not a
notebook, so it does not "open in Colab". That is fine, because **students do
not need the interactive deck** — their hands-on work is the exercise and case
**notebooks**, which open in Colab directly and are where they actually write
code. Think of it as: the deck (a hosted URL) is *your* tool for lecturing; the
Colab notebooks are *their* tool for practising.

Notes for presenting:

- The **first** time a code cell runs, the browser downloads the Pyodide
  runtime (about 8 MB) once from a CDN — so the classroom needs internet, and
  it is worth running one cell a minute before class to warm it up. After that,
  every cell is fast.
- Most cells **run themselves on load** (their output appears automatically);
  "predict" and "your turn" cells stay blank until you press **Run**, so the
  reveal is under your control.
- All static content — text, diagrams, tables, and the year-price chart — is
  rendered ahead of time and always displays, with or without the live runtime.

## Rebuilding the presentation

You only need this if you edit a `.qmd` file. Install
[Quarto](https://quarto.org) (1.4 or newer) and the Python environment
(below), then:

```bash
quarto render session_01/session_01.qmd
```

Rendering executes every Python cell from a clean state, so a successful
render is also a check that the lecture code all runs.

## Building and checking

```bash
python tools/release.py
```

Rebuilds everything generated (the notebooks, the cheatsheet, the sidebar,
the download bundle) and runs the checks. Run it before every push. See
[`tools/README.md`](tools/README.md) for the details, and for the checklist to
follow when adding a session. In short: edit the generator, not the notebook.

## Python environment

`requirements.txt` pins the versions students install (the setup page tells
them to run `%pip install -r https://theill95.github.io/mlfin-2026/requirements.txt`):
NumPy 2.3, pandas 2.3, matplotlib 3.10, scikit-learn 1.7, SciPy 1.16, XGBoost
2.1.4 and LightGBM 4.6, on Python 3.12 to 3.14 (3.15 has no installers for
these yet). Build with exactly that environment:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r tools/requirements-dev.txt
python tools/release.py        # re-runs itself under .venv
```

The generators measure numbers while they write the notebooks, and a few move
between versions (scikit-learn 1.6 changed some forest results in the fourth
decimal; XGBoost 3 grows different trees from 2.1), so `release.py` starts by
checking the environment against the pins and stops if they differ. Colab has
its own versions, recorded in its public `pip-freeze.txt`
(github.com/googlecolab/backend-info); on 28 September 2026 it ran scikit-learn
1.6.1, pandas 2.2.3, NumPy 2.1.3 and XGBoost 3.4.1, with which every notebook
gives the same numbers except Session 12's XGBoost results, so Session 12's
setup cells install XGBoost 2.1.4 in Colab first. The lectures run Pyodide
0.28.1 in the browser (scikit-learn 1.7, XGBoost 2.1.4).

Sessions 5 to 13 use scikit-learn in the materials themselves, in the deck,
the exercises and the case, and Sessions 12 and 13 the two boosting libraries.
Session 6 also reads
`data/market_features.csv`, a nineteen-column table built from the prices by
`tools/build_session06_table.py`, and Session 7 reads `data/aarhus_houses.csv`,
7,621 Aarhus house sales from January 2021 to September 2024, built once by
`tools/build_session07_table.py` from a public compilation of boliga.dk sale
records and geocoded with the national address register (no addresses are
kept). Session 10 reads `data/power.csv`, 26,133 hours of day-ahead prices and
wind and sun forecasts for West Denmark from 2022 to 2024, built once by
`tools/build_session10_table.py` from Energinet's Energi Data Service; its
interactive slides are drawn by `tools/deck10_widgets.py` from numbers that
`tools/deck10_data.py` computes with the same models as the live cells.
Session 13 works the same way with `tools/deck13_widgets.py`, except that
`tools/deck13_data.py` saves its numbers to `session_13/data/deck13.json` (and
the map to `map13.png`) and the deck only reads that file when it renders, so
the Python that Quarto renders with needs no xgboost. Rebuild the file with
`python tools/deck13_data.py session_13/data` after changing a model. All
core materials run **offline** once this environment is installed
and the repository is cloned.

## Working the notebooks (students)

Open a notebook in Google Colab, then **File → Save a copy in Drive** before
you start, so your work is your own.

- **Exercises** and the **case** contain `...` placeholders — replace them
  with your own code. The notebooks run top to bottom from a fresh kernel
  even before you fill anything in, so "Run all" never floods you with errors.
- Hints and full solutions are folded under each task (click to expand). Try
  first; expand to check.
- Every notebook's first cell loads the data. It looks for a `data/` folder
  next to the notebook first and otherwise reads the files from this
  repository, so in Colab nothing needs uploading.
- From Part 2 on, each case part opens with a **quick load** cell that restores
  what the earlier parts worked out, so a part can be started on its own.

## Data

Five tables, each downloaded or built once and committed, so the materials never
depend on a live connection. `resources.html` lists them for students, with
their sources and terms.

| file | what | sessions | source |
|:--|:--|:--|:--|
| `prices.csv` (+ 2024 extracts) | daily closes of eleven US instruments, 2015–2024 | 1–6, 8, 9, 11, 12 | Yahoo Finance via yfinance (personal-use terms: kept for teaching) |
| `market_features.csv` | the nineteen-column index table | 6, 8, 9, 11, 12 | built from the prices by `tools/build_session06_table.py` |
| `aarhus_houses.csv` | 7,621 Aarhus house sales, 2021–2024 | 7, 13 | Martin Frederiksen's boliga.dk compilation (Kaggle, educational use), geocoded with DAWA, by `tools/build_session07_table.py` |
| `credit.csv` | 30,000 credit card holders, Taiwan 2005 | 9, 11, 12 | Yeh (2009), UCI Machine Learning Repository, CC BY 4.0 |
| `power.csv` | 26,133 hours of West Denmark day-ahead prices and forecasts, 2022–2024 | 10, 13 | Energinet's Energi Data Service, CC BY 4.0, by `tools/build_session10_table.py` |

The price files in more detail: daily closing prices for eleven US instruments
(AAPL, MSFT, NVDA, JPM, KO, PG, XOM, JNJ, WMT, DIS, and the S&P 500 ETF SPY),
2015–2024.

- `data/prices.csv` - all eleven, tidy long format (`date, ticker, close,
  volume`); used from Session 3 onward, once pandas is introduced. Session 3's
  deck reads it live in the browser, and both of its notebooks read it too.
- Small single-stock 2024 extracts (`date, close`), used in Sessions 1 and 2
  before pandas exists: `aapl_2024_closes.csv` and `ko_2024_closes.csv`
  (Session 1), plus `nvda_2024_closes.csv`, `jnj_2024_closes.csv` and
  `jpm_2024_closes.csv` for the five-stock ranking in the Session 2 case. Each
  session's folder carries a copy of the files it needs in `session_NN/data/`,
  so a session folder is self-contained.

**Provenance.** Prices are split- and dividend-adjusted closes, obtained once
from Yahoo Finance via the `yfinance` package on 2026-07-19 and committed to
the repository. The course itself never downloads data at runtime. To rebuild
the files from source, `pip install yfinance` and run:

```bash
python tools/build_dataset.py
```

## Course case

A single financial investigation, a **risk report** on this stock universe,
runs across every session. It starts (Session 1) with two stocks analysed as
plain Python lists; uses functions and loops to measure volatility properly
(Session 2); scales up to all eleven stocks with pandas tables and plots
(Session 3); is reframed as a machine-learning prediction problem, with
features and a target, in Session 4; gets its first model, chosen by
cross-validation, in Session 5; in Session 6 is given every column the desk
can offer, with a penalty chosen on the folds to keep it honest; gains a
classifier in Sessions 8 and 9, a warning before a big move with a threshold
set by what each mistake costs; and in Sessions 11 and 12 holds the forecast
and the warning to trees, forests and boosted trees, before Part 12 sets every
model the report has fitted side by side. Sessions 7, 10 and 13 are case
lectures of their own (house prices in Aarhus, power prices in West Denmark,
and both again with trees) and have no part.

## Licence

The course materials (the lectures, notebooks, exercises, case, cheatsheet,
site pages, and the code in them) are © 2026 Jonas Theill Bøjstrup and
licensed under [CC BY 4.0](LICENSE). They may be copied, adapted and taught
from, with credit:

> Jonas Theill Bøjstrup, Machine Learning in Finance, Aarhus University (2026), https://theill95.github.io/mlfin-2026/, CC BY 4.0

Not covered: the data files keep their sources' terms (see [Data](#data)), and
the libraries and tools the site is built with keep their own licences.
Questions about reuse: jobo@econ.au.dk.
