# Analytics Phase 2: pairwise HyperSHAP interactions (docs/PLAN-analytics.md)

**Status:** implemented, awaiting your sign-off.
**You can now:** see a "Hyperparameter interactions" figure next to
Hyperparameter importance — a heatmap of every pairwise HyperSHAP tunability
interaction (diagonal = that hyperparameter's own importance, off-diagonal =
how much the pair matters together beyond their individual effects), plus a
bar-chart alternate view of the top 10 strongest pairs.

Opens Phase 2 of `docs/PLAN-analytics.md`.

---

## The one real finding: this was already being computed

The plan flagged "cost is an open question, not yet measured" for this phase.
It turned out there's no cost to measure: `_compute_hp_game`
(`core/optimizers/base.py`) calls HyperSHAP's `tunability()` (and
`sensitivity()`/`mistunability()`) with no `order` argument, and every one of
those methods defaults to `order=2` — confirmed by reading
`hypershap/hypershap.py`'s own signatures, not assumed. So every optimizer run
was *already* asking HyperSHAP for pairwise interaction values; the existing
code just extracted `iv.get_n_order(order=1)` and threw the rest away. This
phase is a re-extraction of an existing `InteractionValues` object, not a new
HyperSHAP call — zero marginal compute cost, verified by reading `shapiq`'s
`get_n_order` (`.venv/.../shapiq/interaction_values.py:642`), which confirms
`order=2` returns exactly the pairwise terms `(i, j)` with `i < j`, cleanly
separable from the order-1 terms already in use.

## The change

- `BaseOptimizer._extract_pairwise(iv, params)` (`core/optimizers/base.py`,
  new `@staticmethod`) — reshapes an `InteractionValues` object's order-1 and
  order-2 terms into one square `{hp_a: {hp_b: value}}` grid, symmetric, with
  the diagonal holding that hyperparameter's own **signed, unnormalized**
  order-1 value. Deliberately not `abs()`-ed/normalized like
  `compute_hp_importance`'s own return — a heatmap comparing interaction
  *strength* needs sign (positive reads as synergy, negative as redundancy)
  and one shared scale across the whole grid, not a share-of-100%.
- `_compute_hp_game` now returns a 3-tuple —
  `(importance, warning, interactions)` — instead of 2. `interactions` is the
  `_extract_pairwise` grid on the HyperSHAP-success path, `{}` on either
  fallback rung (a plain `feature_importances_` or uniform weights carry no
  pairwise structure to report). `compute_hp_importance`/`_sensitivity`/
  `_mistunability` unpack the 3-tuple and discard the third element, so their
  existing 2-tuple contract (and every test against it) is unchanged.
- `compute_hp_games`'s per-game return also grows to a 3-tuple
  (`importance_by_metric, warning_by_metric, interactions_by_metric`) — every
  game's interactions are computed as the same free byproduct, but only
  `games["tunability"][2]` is actually wired to a field below.
  Sensitivity's/mistunability's interaction grids exist in that return value
  right now but aren't stored anywhere — a natural, zero-cost follow-up if
  wanted later, not done here since the plan scoped this as one new figure,
  not a third game selector.
- `BaseOptimizer.compute_hp_interactions(config_space, trials, metric_name,
  seed)` — a standalone convenience wrapper (same contract shape as
  `compute_hp_importance`) for callers that want just this, without also
  computing importance. Internally calls `_compute_hp_game` itself, so a
  *direct* call to this alongside a separate `compute_hp_importance` call
  would recompute HyperSHAP twice — production code doesn't do that (see
  below); it's for tests and any future one-off caller.
- `OptimizationResult` gains `hyperparameter_interactions` (`metric →
  {hp_a: {hp_b: value}}`) and `hyperparameter_interactions_warning` (`metric →
  str | None`), both defaulted to `{}` so old serialized results still
  deserialize. `serialize_result`/`deserialize_result` (base and SMAC's
  override) round-trip both.
- `grid_optimizer.py`/`random_optimizer.py`/`smac_optimizer.py`: each already
  calls `compute_hp_games` once per `optimize()`; now each also reads
  `games["tunability"][2]` (interactions) and `games["tunability"][1]` (the
  same warning tunability's own importance already carries — same call, same
  failure) into the two new fields. No new HyperSHAP call, no new surrogate
  refit — this is exactly the "free byproduct" the finding above promised.
- `ui/figures/plots.py`: `hyperparameter_interactions_heatmap_plot` (default
  view; `go.Heatmap`, diverging `RdBu` colorscale centered at zero, params
  ordered by `|diagonal|` descending — the figure's "most important first"
  convention) and `hyperparameter_interactions_bar_plot` (top-10 off-diagonal
  pairs by `|value|`, signed and colored by sign like
  `hyperparameter_ablation_plot`, labeled `"a × b"`).
- `ui/figures/catalog.py`: new `HyperparameterInteractions` figure, `per_metric
  = True`, `views = ("heatmap", "bar")` — fully precomputed like the three
  global-game views, unlike the importance figure's lazy "local" game, since
  nothing here depends on a live selection.
- `ui/views.py`: each metric's panel gains `interactions_warning` (shares
  tunability's own warning — same underlying call, so whatever explains an
  empty importance figure also explains an empty interactions one).
- `experiment_detail.html`: one flat `<select>` (heatmap/bar) wired through
  the existing generic `switchView`; a `syncInteractionsWarning(metric)`
  parallel to `syncImportanceWarning`, called from `show(metric)` only (unlike
  the importance figure's warning, this one doesn't depend on which view is
  showing, so a view switch alone never needs to re-sync it).
- No new endpoint, no new URL — every other figure this phase touches is
  fully precomputed, so there was nothing to fetch lazily for.

## Deviations / notes

- **Scope held to the plan's literal ask**: one new figure, tunability-backed
  only. Sensitivity/mistunability interactions are computed (for free) inside
  `compute_hp_games` but not exposed — adding a game selector to this figure,
  mirroring the importance figure's, would be a small, well-scoped follow-up
  if wanted, not attempted here to avoid scope creep beyond what was asked.
- **Diagonal convention**: putting the signed order-1 value on the heatmap's
  diagonal (rather than leaving it 0 or NaN) is a judgment call, not
  something the plan specified — it's a standard fANOVA-style reading (a
  hyperparameter's own effect next to how it interacts with everything else)
  and reuses data already in hand, but is worth flagging as a design choice
  rather than a spec requirement.
- **`compute_hp_interactions` vs. production call sites**: the standalone
  method recomputes HyperSHAP from scratch (it's a convenience wrapper, not
  a cache reader), so it exists for tests/ad-hoc callers only — every
  production path reads interactions off `compute_hp_games`'s existing
  tunability entry instead, to avoid paying for the same HyperSHAP call
  twice per `optimize()`.

## Checklist

✅ confirmed (not assumed) that order=2 was already the default, from
`hypershap`'s own source · ✅ `_extract_pairwise`, symmetric grid, signed
diagonal · ✅ `_compute_hp_game`'s 3-tuple, existing 2-tuple callers
unaffected · ✅ zero new HyperSHAP calls in any optimizer's `optimize()` ·
✅ heatmap + bar views, ordered/ranked consistently with the importance
figure's own conventions · ✅ warning shared with tunability's importance,
not a separate failure mode · ✅ tests (`test_hp_importance.py`,
`test_plots.py`, `test_figures_view.py`) · ✅ automatic settings checkbox
(`show_hyperparameter_interactions`), no manual wiring needed.

## Verification

```bash
python -m pytest -q -m "not slow"
# 739 passed, 5 skipped, 36 deselected in 153.06s — in line with Phase 1b's own
# clean baseline (148.62s), confirming the "zero marginal cost" finding above.
python manage.py check
python manage.py makemigrations --check --dry-run   # no model changes
```

Manual: ran a fresh 12-trial Random Forest / Random Search experiment (real
HyperSHAP interactions, not the stale pre-Phase-2 fixture) and loaded its
detail page. Confirmed: the heatmap renders with real, non-empty values,
ordered `max_features, n_estimators, min_samples_split, max_depth` (matching
the importance pie's own descending order); switching to "Top pairs" draws a
signed, sign-colored bar of the same data; switching metric (accuracy → f1)
redraws both views with that metric's own numbers while staying on "Top
pairs" (view choice persists across a metric switch, matching every other
multi-view figure). No browser console errors throughout.
