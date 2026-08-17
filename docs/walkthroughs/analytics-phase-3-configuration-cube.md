# Analytics Phase 3: Configuration Cube (docs/PLAN-analytics.md)

**Status:** implemented, awaiting your sign-off.
**You can now:** see a "Configuration cube" figure — every trial plotted in
hyperparameter space, colored by score, with X/Y (and optional Z, for a 3D
scatter) pickers that let you look at any pair or triple of hyperparameters
without leaving the page.

Opens Phase 3 of `docs/PLAN-analytics.md`.

---

## Why this figure needed its own plumbing, not the `views` mechanism

Every prior multi-view figure (importance, interactions, performance) picks
from a small, fixed, enumerable vocabulary — three games, two renderings,
two axis choices — so the server can precompute one JSON payload per
combination and the client just switches which one is showing. The cube's
"view" is which two or three of an experiment's *N* hyperparameters are on
which axis: for a model with, say, 6 hyperparameters, that's `C(6,2) +
C(6,3) = 15 + 20` combinations, growing combinatorially with the model. There
is no fixed set to enumerate.

So instead of a `views` tuple, `configuration_cube_plot`
(`ui/figures/plots.py`) ships **one** default 2D scatter (the first two
hyperparameters, in config-space order — the same order the Trials table's
columns already use), but every hyperparameter's full per-trial values ride
along on that same trace's `customdata` (one row per trial), with the column
order recorded in `layout.meta["hp_names"]`. The client then remaps which
columns are `x`/`y`/`z` — and switches between a 2D `scatter` and a 3D
`scatter3d` trace — entirely by re-slicing that one array, per the plan's
explicit instruction to remap client-side rather than round-trip to the
server per axis change.

**Deliberately self-contained, not reusing `hp_names`/`trial_rows`:** the view
already computes and ships `hp_names`/`trial_rows` for the Trials table, but
both are only present in the page's HTML *inside that table's own template
block* — meaning they vanish if the Trials figure is toggled off. Following
`test_hiding_the_performance_figure_keeps_the_page_working`'s own precedent
(no figure may depend on another figure's visibility), the cube figure
carries its own copy of the same information in its own trace instead.

## The change

- `ui/figures/plots.py`: `configuration_cube_plot(result, display_metric)` —
  builds the default `go.Scatter`, marker-colored by `display_metric`'s
  score (`Viridis`, with a colorbar), `customdata` = every hyperparameter's
  value per trial, `layout.meta.hp_names` = the column order. Returns `None`
  with no trials or fewer than two hyperparameters (a cube needs at least
  two axes to mean anything).
- `ui/figures/catalog.py`: new `ConfigurationCube` figure — `per_metric =
  True` (color depends on the metric), `width = FULL` (axis pickers plus a
  potential 3D plot need the room), no `views` declared (see above). `plot()`
  is a thin wrapper, matching `TrialDuration`'s shape.
- `ui/templates/ui/figures/configuration_cube.html`: three `<select>`s (X,
  Y, Z — Z's first option is "None (2D)"), populated from `hp_names`, which
  is already unconditionally in the page context regardless of the Trials
  figure's visibility.
- `experiment_detail.html`: `applyCubeAxes()` — reads the three pickers,
  re-slices the currently-drawn trace's `customdata` by
  `layout.meta.hp_names`, and calls `Plotly.react` with a fresh
  `scatter`/`scatter3d` trace and layout. Wired to each picker's `change`
  event, and called after every `redraw()` in `show(metric)` so a metric
  switch's fresh default trace has the user's own axis choice reapplied over
  it — the same "view survives a metric switch" pattern every other
  multi-view figure already follows, just implemented by hand here since
  there's no `currentView` entry to fall back on.

## Deviations / notes

- **A real bug, caught by screenshot, not by a test:** the first draft of
  `applyCubeAxes()` set `layout.xaxis.title` (and `yaxis`/`scene.*axis`) to a
  bare string. `plotly.py`'s own `xaxis_title=` shorthand always produces the
  object form (`{"title": {"text": "..."}}`), and the bundled Plotly.js
  turned out to silently drop the bare-string shorthand — no console error,
  no exception, the axis just rendered with no title at all. Caught only by
  actually looking at a screenshot after the first remap; fixed by matching
  the object form everywhere `applyCubeAxes` sets a title. Recorded here
  because it is exactly the kind of silent failure that would not have
  surfaced from `fig.to_json()`-only test coverage (the *server's* JSON was
  always correct — only the client-side remap code got it wrong), and
  because "no error means it's not a rendering bug" is not a safe assumption
  for Plotly's shorthand properties.
- **No MDS projection**: the plan explicitly scoped this to real
  hyperparameter axes rather than DeepCave's dimensionality-reduced
  embedding, to stay dependency-free and keep every axis independently
  meaningful. A projected view is not on this phase's list.
- **Picking the same hyperparameter for two axes is allowed, not guarded
  against** — it produces a degenerate (line/point) plot, not a crash, and
  guarding against it would be validating a harmless, self-correcting user
  choice rather than a real error.

## Checklist

✅ `configuration_cube_plot`, self-contained `customdata` + `layout.meta`,
no cross-figure dependency · ✅ 2D default, 3D via the Z picker, remapped
entirely client-side · ✅ axis choice survives a metric switch · ✅ correct
axis titles after remap (object form, not the silently-dropped bare-string
shorthand) · ✅ `FULL` width, automatic settings checkbox · ✅ tests
(`test_plots.py`, `test_figure_visibility.py`, `test_figures_view.py`) ·
✅ manual browser verification: default 2D scatter with real data, axis
remap changes both the plotted points and the axis title, Z picker switches
to a real 3D scatter, axis choice survives a metric switch, no console
errors.

## Verification

```bash
python -m pytest -q -m "not slow"
# 744 passed, 5 skipped, 36 deselected in 155.27s — flat against Phase 2's
# own clean baseline (153.06s).
python manage.py check
python manage.py makemigrations --check --dry-run   # no model changes
```

Manual: ran a fresh 12-trial Random Forest / Random Search experiment (4
hyperparameters). Confirmed: the default 2D scatter shows real, distinct
points colored by accuracy; remapping the Y axis redraws with new points and
a correct new axis title; picking a Z axis switches to a real 3D scatter
with correct axis titles on all three; switching metric (accuracy → f1)
keeps the chosen axes and trace type rather than resetting to the server's
2D default. No browser console errors throughout.
