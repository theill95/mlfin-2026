# tools

Scripts that build and check the course materials. None of them is course
content: students never see this folder.

Everything here is run from the **repository root**, and works out where the
repository is from its own location, so there are no paths to edit.

## The one command

```bash
python tools/release.py           # rebuild everything generated, then check it
python tools/release.py --check   # check only, change nothing
```

Run this before every push. It regenerates the six notebooks, the cheatsheet,
the sidebar and the download bundle, then runs the verifiers. If you only ever
remember one thing on this page, remember this command.

It deliberately does **not** render the Quarto decks (slow, and needs Quarto)
and does not commit anything.

## Adding a session

In order, and `release.py` covers steps 4 to 6:

1. Write `session_04/session_04.qmd`, then
   `quarto render session_04/session_04.qmd`.
2. Write `tools/generators/session_04_exercises.py` and `session_04_case.py`,
   copying an existing pair. Keep `REPO_RAW_URL` set, or Colab breaks.
3. Add the new session to `BUILD` and, if you write verifiers for it, `CHECK`
   in `tools/release.py`; and to `NOTEBOOKS` in
   `tools/verify/colab_data_access.py` if its notebooks load data.
4. Add cheatsheet entries for anything new, with `since="S4"`, in
   `tools/build_cheatsheet.py`.
5. Add a card to `index.html` (and its id to the `NOTEBOOKS` map in the script
   at the bottom), and a section to `downloads.html` with `id="session-4"`.
6. Add the new notebooks and data to `INCLUDE_GLOBS` in
   `tools/build_download_bundle.py` if they do not already match.
7. Update the session table in the root `README.md`.

The download bundle is built from an **allowlist**, so a new session's files
are not published to students until step 6. That is on purpose: it is safer to
forget to include something than to accidentally ship the instructor's notes.

## Building the data

```bash
python tools/build_dataset.py        # needs: pip install yfinance
```

Downloads the price history once and writes `data/prices.csv` plus the small
single-stock extracts. The course itself never downloads anything at runtime,
so you only need this if you want to rebuild the dataset from source.

## Building the notebooks

The four student notebooks per session are **generated**, not hand-edited. Edit
the generator, re-run it, and commit the result.

```bash
python tools/generators/session_01_exercises.py
python tools/generators/session_01_case.py
python tools/generators/session_02_exercises.py
python tools/generators/session_02_case.py
python tools/generators/session_03_exercises.py
python tools/generators/session_03_case.py
python tools/generators/session_04_exercises.py
python tools/generators/session_04_case.py
```

Cell ids are stamped deterministically (`c0000`, `c0001`, ...), so re-running a
generator you have not changed produces **no diff at all**. If you see a diff,
it is a real one.

Editing a notebook by hand in Jupyter or Colab works, but the next time anyone
runs the generator your edit is gone. Change the generator instead.

### The difficulty ladder

Exercises carry **one to five stars and no tier names**. The scale restarts each
session: it rates the work against what that session has taught, so a three-star
task in Session 4 is a bigger piece of work than a three-star task in Session 1
even though both sit in the middle of their own notebook.

| stars | what it means |
|:--|:--|
| ★ | one step, straight from the lecture |
| ★★ | the same idea on new data, or two steps in a row |
| ★★★ | combine two ideas, or adapt a pattern rather than copy it |
| ★★★★ | you choose the approach; something has to be worked out first |
| ★★★★★ | a genuine puzzle: an insight, or a constraint ruling out the obvious route |

An exercise that leans on an **earlier** session carries a `revisits` tag, which
renders after the stars. Those tags live in a `REVISITS` dict near the top of
each generator rather than on the individual `ex()` calls, so the whole set can
be read and revised at once. Tag an exercise only when the earlier tool is
genuinely load-bearing; a tag that is merely true is noise.

`tools/verify/exercise_ladder.py` enforces all of this: unique consecutive ids,
stars in range, at least four of the five levels used per session, and every
`revisits` tag pointing at an **earlier** session.

### Markdown Colab will actually render

Colab does not use Jupyter's markdown renderer. The two agree on almost
everything, and disagree on one thing that has already bitten us: **once a
markdown cell emits block-level raw HTML, Colab stops parsing what follows as
markdown.** Jupyter carries on, so the cell looks right locally and reaches
students as a wall of literal `|` characters, one stacked line per row. That is
what happened to the toolkit card, whose `<p>` chips were followed by the
"Formulas you will reach for" table, which is why that table now lives in its
own cell.

The rule, which costs nothing to obey:

> markdown first, raw HTML last, and never markdown again after the HTML.

Put anything that must stay markdown, a table above all, in its **own cell**.

The second rule is narrower: **inside a table cell, maths must use `$$...$$`,
never `$...$`.** Colab typesets display maths in a table happily and collapses
the row on inline maths, into the stacked literal text you would otherwise ship.
Jupyter renders both, which is what makes this so easy to miss. Outside tables
both forms are fine, and these notebooks use plenty of each.

`tools/verify/colab_markup.py` enforces both rules across all eight notebooks.

### Maths in Colab

Measured in Colab, not reasoned about, after two wrong guesses cost a push each.
The same five formulas were rendered every way that seemed plausible:

| layout | Colab |
|:--|:--|
| markdown table, `$...$` | **broken**, row collapses |
| HTML table, `$...$` | **broken**, identical to the markdown one |
| markdown table, `$$...$$` | **works** |
| HTML table, `$$...$$` | **works** |
| any table, `\(...\)` | **broken**, prints the LaTeX literally |
| bullet list, `$...$` | works |
| HTML table, Unicode, no LaTeX | works, but reads poorly for sums and fractions |

Two conclusions worth keeping. The table **type** is irrelevant, so reaching for
raw HTML to dodge a markdown problem buys nothing here. And `\(...\)` is not a
delimiter Colab knows anywhere, in or out of a table: only `$` and `$$` are.

**When Colab and Jupyter disagree again, probe before you push.** Build a
throwaway notebook that renders the same content every plausible way, put it
under `tools/`, where the download allowlist keeps it away from students, and
open it once in Colab. One look beats a chain of confident guesses, and each
guess costs a push and a round trip.

Give each probe round its **own filename**. Colab caches a notebook it has
already opened from GitHub, so pushing a new round over the same path serves the
old one and the new blocks appear to be missing. Delete the probe once the
question is answered; the answer belongs in the table above, not in a stray
notebook.

The toolkit chips are `<code style="cursor:help" title="...">`, not `<abbr>`.
`abbr` draws a dotted underline that reads like a spelling error and fights the
code styling; a plain `<code>` inherits whatever the platform already uses for
inline code and looks right in Jupyter, Colab and VS Code alike. The `title` is
what carries the hover doc, and the cursor is the only affordance for it, so
keep both. Note the escaping: in a double-quoted Python string the attribute
quotes have to be `\"`, or the string ends early and the generator will not
parse.

## The site pages

```bash
python tools/build_nav.py              # stamp assets/nav.html into every page
python tools/build_download_bundle.py  # downloads/mlfin-course.zip
python tools/build_cheatsheet.py       # cheatsheet.html
```

The sidebar lives once, in `assets/nav.html`, and `build_nav.py` copies it into
each page between the `<!-- nav:start -->` and `<!-- nav:end -->` markers. Edit
the template, not the pages.

`build_download_bundle.py` assembles the student ZIP from an **allowlist**, so
`tools/`, the Quarto extensions, the instructor README and the build scripts
cannot end up in a student download. It refuses anything matching its `NEVER`
list even if a glob would have caught it, and it writes fixed timestamps so an
unchanged bundle produces no git diff.

`build_cheatsheet.py` writes `cheatsheet.html`. Every example on that page is
executed against the real course data while the page is built, and the output
you see is the output that actually came back. The script fails loudly if any
example raises, rather than publishing a guess.

## Checking the materials

```bash
python tools/verify/colab_data_access.py      # run this after touching any loader cell
python tools/verify/session_03_exercises.py   # every folded solution, against real data
python tools/verify/session_03_case.py        # the case solutions, chained in one namespace
python tools/verify/session_03_deck.py        # every live cell in the deck
python tools/verify/session_04_deck.py        # same, for Session 4
```

`colab_data_access.py` is the important one. It runs each notebook's loader
cell with the local search path emptied, which forces the download that Google
Colab depends on, and confirms the result matches the local files. Colab has no
`data/` folder, so if `REPO_RAW_URL` is ever unset or wrong, every "Open in
Colab" button on the site leads to a notebook that dies on its first cell.

It also checks its own coverage: any notebook that reads a file and is **not**
in its `NOTEBOOKS` list is a failure. Sessions 1 and 2 keep their prices in
literal lists and so are absent on purpose; without the check, adding a loader
to one of them would go unnoticed until a student hit it in Colab.

Nothing else is needed at runtime: across all eight notebooks the only
third-party imports are numpy, pandas and matplotlib, which Colab preinstalls,
so no notebook needs a `pip install` cell.

The `session_0N_exercises.py` and `session_0N_case.py` verifiers extract each
folded solution and run it, so the stated answers cannot drift away from the
data. The case verifier runs them **in order in one namespace**, which is what
proves the investigation still chains together.

A blank run is checked separately, and matters just as much: a student who
presses Run All before filling anything in must not see a wall of errors.

`tools/verify/blank_safety.py` checks this statically, across all eight
notebooks, in under a second. It parses every work cell, tracks which names are
bound to `...`, and refuses anything that would call a method on one, index into
it, iterate it, do arithmetic with it, or pass it to `len`/`range`/`sum`. It
also refuses `while ...:` outright, which does not raise at all: `Ellipsis` is
truthy, so that cell **hangs the kernel forever**, which is far worse than a
traceback and invisible in a quick read.

Run the real thing as well before publishing, because the static check cannot
see everything:

```bash
python -m jupyter nbconvert --to notebook --execute --output /tmp/out.ipynb session_03/session_03_exercises.ipynb
```

## Rendering the decks

```bash
quarto render session_03/session_03.qmd
```

Needs Quarto and the `quarto-live` extension in `_extensions/`. The rendered
`session_NN.html` and its `session_NN_files/` folder are both committed: the
deck will not run without the second one, and it cannot be self-contained
because the Pyodide runtime cannot load its worker from a `data:` URI.

Editing `theme/mlfin.scss` changes the compiled stylesheet's hash, so every
re-render leaves the previous `quarto-<hash>.css` behind inside
`session_NN_files/`. After a theme change, re-render **all** the decks so they
share one stylesheet, then:

```bash
python tools/prune_render_cruft.py            # list the orphans
python tools/prune_render_cruft.py --delete   # remove them
```

It only removes a stylesheet that no `session_NN.html` refers to.
