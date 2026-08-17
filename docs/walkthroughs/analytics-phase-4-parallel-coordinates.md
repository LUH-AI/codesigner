# Analytics Phase 4: Parallel Coordinates (docs/PLAN-analytics.md)

**Status:** implemented, awaiting your sign-off.
**You can now:** see a "Parallel coordinates" figure — every trial as one
line across its hyperparameters, ending at its score, axes ordered by that
metric's own HyperSHAP tunability (most important first) — DeepCave's own
best tool for spotting hyperparameter interactions at a glance, and
codesigner had no equivalent before this.

Closes Phase 4 of `docs/PLAN-analytics.md`.

---

## The simplest phase in this roadmap so far

Unlike the cube (Phase 3), axis order here is a full, fixed ranking per
metric — every hyperparameter appears exactly once, in importance order —
not a per-experiment unbounded *combination* of a subset. That means it fits
the ordinary `per_metric` contract every figure before Phase 3 already used:
one precomputed `go.Parcoords` plot per metric, no `views`, no client-side
remapping, no bespoke JS at all. The generic `payloadFor`/`redraw`/`show()`
machinery handles it exactly like `hyperparameter_importance` did before
Phase 0 ever introduced the views concept.

## The change

- `ui/figures/plots.py`: `parallel_coordinates_plot(result, display_metric)`
  — one `go.Parcoords` dimension per hyperparameter, ordered by
  `result.hyperparameter_importance[display_metric]` descending (falls back
  to config order if importance isn't available for this metric — e.g. a
  result predating Phase 1a, or HyperSHAP failed), plus a final dimension for
  the score itself. Lines are colored by that same score (`Viridis`), so a
  cluster of high-scoring lines all bending through one region of one axis
  reads as "this hyperparameter matters."
  - **Non-numeric axes**: `go.Parcoords` dimensions are strictly numeric.
    Booleans and strings are recoded to their sorted-unique values' integer
    position, with `ticktext` naming the originals — booleans specifically
    are treated as categorical rather than raw 0/1, since "True"/"False"
    reads better than a bare number for a hyperparameter that's
    conceptually a toggle, not a magnitude.
- `ui/figures/catalog.py`: new `ParallelCoordinates` figure — `per_metric =
  True`, `width = FULL` (one line per trial across every axis needs the
  room), no `views`. `plot()` is a thin wrapper, matching `TrialDuration`'s
  and `ConfigurationCube`'s shape.
- `ui/templates/ui/figures/parallel_coordinates.html`: a plain card, no
  selectors — there is nothing for the client to choose between.
- No JS changes at all in `experiment_detail.html` — the existing generic
  per-metric draw/redraw path is the entire client-side story for this
  figure.

## Deviations / notes

- **HyperSHAP-first ordering, not DeepCave's fANOVA**: the plan called for
  this explicitly — codesigner's explanations are HyperSHAP throughout, so
  the one figure DeepCave orders by fANOVA gets reordered to match rather
  than importing a second, inconsistent notion of "importance."
- **No brushing/filtering UI added**: Plotly's `Parcoords` trace ships
  interactive axis brushing/filtering natively, client-side, for free — no
  code needed here to get it, and confirmed working in the browser check
  below without any wiring on this app's part.

## Checklist

✅ `parallel_coordinates_plot`, HyperSHAP-tunability axis order, numeric and
recoded-categorical dimensions both covered · ✅ final axis is the score,
line color matches it · ✅ `FULL` width, automatic settings checkbox, no
special JS · ✅ tests (`test_plots.py`, `test_figure_visibility.py`,
`test_figures_view.py`) · ✅ manual browser verification: real data across
all axes, axis order changes correctly per metric, no console errors.

## Verification

```bash
python -m pytest -q -m "not slow"
# 750 passed, 5 skipped, 36 deselected in 154.46s — flat against Phase 3's
# own clean baseline (155.27s).
python manage.py check
python manage.py makemigrations --check --dry-run   # no model changes
```

Manual: ran a fresh 12-trial Random Forest / Random Search experiment (4
hyperparameters). Confirmed: five dimensions render (four hyperparameters
ordered by tunability, then the score), with real per-trial values and
score-colored lines; switching metric (accuracy → f1) redraws with that
metric's own tunability ordering (`max_features, n_estimators, max_depth,
min_samples_split` for f1 vs. `max_features, n_estimators, min_samples_split,
max_depth` for accuracy — genuinely different rankings, not a fixed order).
No browser console errors throughout.
