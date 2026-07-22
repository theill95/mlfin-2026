# Machine Learning in Finance

Teaching materials for a master's-level course introducing machine learning
to finance and economics students. The course assumes econometrics (OLS) but
**no prior Python or machine-learning experience**, and begins in Google
Colab.

Each session is a single self-contained folder. Sessions 1 to 3 are complete;
later sessions are added as the course is built.

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
- Repository: <https://github.com/theill95/mlfin-2026>

The site has five pages, linked from the sidebar:

| page | what it is |
|:--|:--|
| `index.html` | the sessions: lecture, exercises and case for each |
| `cheatsheet.html` | every function the course has used, with a worked example. **Generated** by `tools/build_cheatsheet.py`, which executes every example against the real data, so do not edit it by hand |
| `setup.html` | installing Python and VS Code, and the extras (scripts vs notebooks, virtual environments, Git), each marked needed or optional |
| `resources.html` | official docs, ISLP, Kaggle, and the AI-use policy |
| `downloads.html` | every notebook and CSV, plus `downloads/mlfin-course.zip` |

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

Sessions 2 to 4 have the same shape. Current contents:

| session | lecture | exercises | case |
|:--|:--|:--|:--|
| 1 · Beginning Python for Financial Data | `session_01.qmd` | 53 | Part 1, 10 questions |
| 2 · Functions, Loops, and Dictionaries | `session_02.qmd` | 60 | Part 2, 10 questions |
| 3 · Packages: NumPy, pandas, matplotlib | `session_03.qmd` | 70 | Part 3, 14 questions |
| 4 · Foundations of Machine Learning | `session_04.qmd` | 65 | Part 4, 14 questions |

Sessions 3 and 4 load numpy, pandas and matplotlib into the browser runtime
(about 20 MB, once per page load). Open the deck and run one cell a few minutes
before class so it is warm.

Session 4 is the only deck with mathematics on the slides; it renders through
KaTeX, set in the `.qmd` front matter.

The `.qmd` file **is** the lecture: it is at once the slide deck, the lecture
narrative, and the source of every executable code demonstration. There is no
separate lecture notebook.

## Running the presentation

The rendered `session_01/session_01.html` opens in any modern browser and
supports keyboard navigation (arrow keys, `f` for fullscreen, `s` for the
speaker view, `Esc` for the slide overview).

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

Rebuilds everything generated (the six notebooks, the cheatsheet, the sidebar,
the download bundle) and runs the checks. Run it before every push. See
[`tools/README.md`](tools/README.md) for the details, and for the checklist to
follow when adding a session. In short: edit the generator, not the notebook.

## Python environment

```bash
pip install -r requirements.txt
```

Built and tested with Python 3.12, pandas 2.2, numpy 1.26, matplotlib 3.9,
scikit-learn 1.5. All core materials run **offline** once this environment is
installed and the repository is cloned.

## Working the notebooks (students)

Open a notebook in Google Colab, then **File → Save a copy in Drive** before
you start, so your work is your own.

- **Exercises** and the **case** contain `...` placeholders — replace them
  with your own code. The notebooks run top to bottom from a fresh kernel
  even before you fill anything in, so "Run all" never floods you with errors.
- Hints and full solutions are folded under each task (click to expand). Try
  first; expand to check.
- The case notebook's first cell loads the data. It looks for the `data/`
  folder next to the notebook first. In Colab, either upload the two CSV files
  or set `REPO_RAW_URL` in that cell to the repository's raw data URL.

## Data

Daily closing prices for eleven US instruments (AAPL, MSFT, NVDA, JPM, KO, PG,
XOM, JNJ, WMT, DIS, and the S&P 500 ETF SPY), 2015–2024.

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
runs across Sessions 1–4. It starts (Session 1) with two stocks analysed as
plain Python lists; uses functions and loops to measure volatility properly
(Session 2); scales up to all eleven stocks with pandas tables and plots
(Session 3); and is reframed as a machine-learning prediction problem, with
features and a target, in Session 4. Building the models themselves comes later
in the course.
