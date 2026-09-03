# Analytics Phase 6: Partial Dependence (PDP/ICE) (docs/PLAN-analytics.md)

**Status:** implemented, awaiting your sign-off.
**You can now:** see a "Partial dependence (PDP/ICE)" figure — pick any
hyperparameter and see its marginal effect on the current metric: a bold
Partial Dependence curve (the average effect) plus every trial's own
Individual Conditional Expectation curve (thin, translucent) underneath it.
Lines that all move together say this hyperparameter doesn't interact with
anything else; lines that fan out somewhere on the grid say it does, right
where they fan out.

Closes Phase 6 of `docs/PLAN-analytics.md` — the last item on the roadmap.

---

## Same shape as local ablation, for the same underlying reason

Which hyperparameter is showing isn't one of a small, fixed set of ways to
look at the *same* data — it changes what's being explained, and there's no
enumerable list of options the server could precompute a JSON payload per
entry for the way importance's game/rendering or performance's axis choices
work. So, exactly like the importance figure's "Local (selected trial)"
game from Phase 1b: no `views`, nothing precomputed, `plot()` always returns
`None`, and the real work happens behind a dedicated endpoint
(`partial_dependence`), fetched lazily by the client — including the very
first render, not just later switches.

**Unlike local ablation, the default isn't "the best trial" — it's "the
most important hyperparameter."** Each metric's panel gained a `top_hp`
field (its own top-tunability hyperparameter, mirroring `best_idx`'s "most
interesting trial" instinct), and the picker resets to it on every metric
switch, the same way `lastClickedIdx` gets forgotten on one.

## The change

- `core/optimizers/base.py`:
  - `_hp_grid(hp, n_points=20)` (new, module-level) — a categorical
    hyperparameter's full set of choices; sorted-unique rounded integers for
    an integer one (so the grid never suggests a value it couldn't actually
    take); otherwise `n_points` evenly spaced floats across its bounds.
  - `BaseOptimizer.compute_partial_dependence(config_space, trials,
    metric_name, hp_name, seed=0, n_points=20)` (new) — fits a surrogate via
    Phase 5's `fit_surrogate` (the exact same stand-in every global
    HyperSHAP game's fallback rung already uses), then for every grid value
    of *hp_name*, holds each trial's *other* hyperparameters fixed at its
    own actual values and predicts with only *hp_name* swapped in — one ICE
    point per trial per grid value. The Partial Dependence curve is the
    grid-wise mean across every trial's ICE curve — the standard
    definition, and DeepCave's own. A synthetic configuration the config
    space rejects (never happens with this app's current registry models,
    none of which have conditional/forbidden-clause hyperparameters, but
    kept defensive for a future model that might) contributes `None` at
    that grid point rather than raising.
- `ui/figures/plots.py`: `partial_dependence_plot(hp_name, grid, ice_lines,
  pdp)` — every ICE row batched into **one** `None`-separated trace (so
  trial count never multiplies trace count — a real scaling concern this
  figure has that no earlier one did, since a long SMAC run can have
  hundreds of trials) underneath a bold Partial Dependence trace. x is a
  categorical axis even for a numeric hyperparameter's evenly-spaced grid —
  visually identical for an even grid, and it means a categorical
  hyperparameter's grid never needs special-case handling here.
- `ui/figures/catalog.py`: new `PartialDependence` figure — `per_metric =
  True`, `width = FULL`, no `views`, no `plot()` override (stays the base
  class's `None`, exactly like `BestConfiguration`/`SelectedConfiguration`).
- `ui/views.py`:
  - `_partial_dependence_data(built, metric, hp_name)` — same shape as
    `_local_ablation_data`: resolves the config space from `built["model"]`
    (`None`, reported as an unavailable-not-crashed warning, for a
    custom-model experiment viewed read-only), calls
    `compute_partial_dependence`, builds the figure.
  - `partial_dependence` (new view, `GET
    /experiments/<pk>/partial-dependence/?metric=&hp=`) — same validation
    shape as `trial_ablation`.
  - Each metric's panel gains `top_hp`.
- `experiment_detail.html`: `refreshPartialDependence(metric)` — a close
  sibling of `refreshLocalAblation`, cached by `"metric:hp"`. Wired to the
  picker's own `change`, and to every `show(metric)` call (metric switch
  resets the picker to that metric's `top_hp` first, then fetches) — unlike
  local ablation, there's no "is this game even active" gate, since this
  figure only ever shows partial dependence.
- New URL, permissions-route-audit entry
  (`tests/ui/test_permissions.py`), tests (`test_hp_importance.py`,
  `test_plots.py`, new `test_partial_dependence_view.py`,
  `test_figure_visibility.py`, `test_figures_view.py`).

## Deviations / notes

- **A real timing artifact caught in manual verification, ruled out by
  re-checking**: an early browser check appeared to show the plot not
  updating after switching hyperparameters (same x-axis title in two
  screenshots taken moments apart). Re-checked by reading
  `layout.xaxis.title.text` directly (not a screenshot) with a longer wait —
  it *had* updated correctly; Plotly's own console warning ("Too many
  auto-margin redraws," from this figure's many rotated category tick
  labels) meant that particular render just took longer than the first
  check's wait allowed. Recorded here as due diligence, not as a shipped
  bug — worth remembering that a screenshot is a snapshot of a race, not
  proof either way, when a render is mid-flight.
- **Test suite runtime increased for a real, attributable reason this
  time** (contrast Phase 1b's own investigation, where the jump turned out
  to be a stray dev server): 158.78s (Phase 5) → ~181s here. Confirmed via
  `--durations`: the new `test_partial_dependence_*` tests each do a real
  surrogate fit plus grid predictions (unlike Phase 2's interactions, which
  were a free re-extraction of data already being computed) — three of
  them alone account for ~18s. This is the same kind of cost Phase 1a
  disclosed for its own two new games: genuinely new computation, not
  noise, and not treated as a blocker.
- **2D interaction plots**, the plan's own explicitly-optional stretch goal
  for this phase, were not attempted — the plan marked them not required to
  ship, and nothing about this implementation blocks adding them later
  (the surrogate and grid machinery already generalizes to two
  hyperparameters at once).

## Checklist

✅ `_hp_grid` (categorical/integer/float, all three exercised by tests) ·
✅ `compute_partial_dependence`, reuses Phase 5's `fit_surrogate`, no new
surrogate-fitting code path · ✅ ICE batched into one trace regardless of
trial count · ✅ lazy fetch + cache + picker reset on metric switch, wired
to change/show, mirroring local ablation's own pattern · ✅ graceful "model
unavailable" path for custom-model experiments viewed read-only · ✅ tests
(`test_hp_importance.py`, `test_plots.py`, `test_partial_dependence_view.py`,
`test_figure_visibility.py`, `test_figures_view.py`, route-audit entry) ·
✅ manual browser verification: default fetch on load for the metric's
top-tunability hyperparameter, refetch on hyperparameter switch (confirmed
via live DOM state, not just a screenshot), picker resets to the new
metric's own top hyperparameter on a metric switch with a fresh fetch,
cache hit confirmed for both a repeated hyperparameter and a repeated
metric, no browser console errors.

## Verification

```bash
python -m pytest -q -m "not slow"
# 768 passed, 5 skipped, 36 deselected in ~181s — up from Phase 5's 158.78s
# for a real, attributed reason (see "Deviations" above), not a regression.
python manage.py check
python manage.py makemigrations --check --dry-run   # no model changes
```

Manual: ran a fresh 12-trial Random Forest / Random Search experiment.
Confirmed: the figure fetches and renders on load with no click needed,
showing that metric's top-tunability hyperparameter (`max_features` for
this fixture); switching hyperparameters redraws with a genuinely different
curve (confirmed both visually and via `layout.xaxis.title.text`); switching
metric resets the picker to the new metric's own top hyperparameter and
fetches fresh; revisiting an already-fetched (metric, hyperparameter)
combination does not refetch. No browser console errors (aside from benign
WebGL performance diagnostics unrelated to correctness).

---

This closes out every phase in `docs/PLAN-analytics.md`. Nothing scheduled
remains except the two explicitly-out-of-scope items (Pareto Front, Budget
Correlation — both blocked on capabilities this app doesn't have) and the
"explicitly not scheduled" list (Configuration Footprint, Symbolic
Explanations) noted there for future reconsideration only.
