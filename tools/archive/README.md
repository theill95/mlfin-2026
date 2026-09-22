# Archived cheatsheet, worked-examples format

The cheatsheet was rebuilt on 2026-09-22 as a reference page: imports once per
section, then one row per function with general argument names and a sentence,
nothing executed. This folder keeps the previous format, in which every entry
carried an example that was run against the course data at build time.

- `build_cheatsheet_examples.py`: the old builder, as committed in 1197b5d.
- `cheatsheet_examples.html`: the page it produced (styled only when it sits
  next to `assets/site.css`, so copy it to the repo root to look at it).
- `cheatsheet_examples.css`: the CSS that format needs.

To go back: copy the builder to `tools/build_cheatsheet.py`, put the CSS block
back into `assets/site.css`, and run `python tools/build_cheatsheet.py`.
`tools/` is never shipped to students, so nothing here reaches the site.
