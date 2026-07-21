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
```

Cell ids are stamped deterministically (`c0000`, `c0001`, ...), so re-running a
generator you have not changed produces **no diff at all**. If you see a diff,
it is a real one.

Editing a notebook by hand in Jupyter or Colab works, but the next time anyone
runs the generator your edit is gone. Change the generator instead.

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
```

`colab_data_access.py` is the important one. It runs each notebook's loader
cell with the local search path emptied, which forces the download that Google
Colab depends on, and confirms the result matches the local files. Colab has no
`data/` folder, so if `REPO_RAW_URL` is ever unset or wrong, every "Open in
Colab" button on the site leads to a notebook that dies on its first cell.

The `session_0N_exercises.py` and `session_0N_case.py` verifiers extract each
folded solution and run it, so the stated answers cannot drift away from the
data. The case verifier runs them **in order in one namespace**, which is what
proves the investigation still chains together.

A blank run is checked separately, and matters just as much: a student who
presses Run All before filling anything in must not see a wall of errors.

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
