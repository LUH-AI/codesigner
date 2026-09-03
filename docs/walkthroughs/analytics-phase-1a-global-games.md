# Analytics Phase 1a: sensitivity and mistunability (docs/PLAN-analytics.md)

**Status:** implemented, awaiting your sign-off.
**You can now:** switch the Hyperparameter importance figure between three
HyperSHAP games — Tunability (the original importance numbers, unchanged),
Sensitivity, and Mistunability — independently of which of pie/bar/table is
drawing them. Nine views total (3 games × 3 renderings), composed from two
selectors the same way Performance over time's trial/time × score/error
already compose from two.

First half of Phase 1 from `docs/PLAN-analytics.md`. The second half — a
"Local (selected trial)" view, using HyperSHAP's `ablation` game against the
currently-selected trial — is split off as its own follow-up: `ablation`
takes a specific configuration rather than being a plain global game, so it
needs a different call shape and a live recompute wired to the existing
click-to-select mechanism, which is enough of its own thing to review
separately.

---

## What the spike found

The plan called for confirming HyperSHAP's actual API before designing
against it. Reading `hypershap.hypershap.HyperSHAP`'s source directly (not
just its docstrings) turned up two things worth designing around:

- **A fourth game exists**: `mistunability`, sibling to `tunability`
  (MAX aggregation — how much upside does tuning this hyperparameter offer)
  and `sensitivity` (VAR aggregation — how much does performance vary as it
  moves). `mistunability` uses MIN aggregation — how much downside risk does
  getting it wrong carry. Identical call shape to the other two, so it rides
  along for free.
- **`optimizer_bias` doesn't fit this app's model.** It needs a live
  `optimizer_of_interest` and an `optimizer_ensemble` — actual searcher
  objects to compare against each other — not just one run's trial history.
  Every other game this app surfaces explains *one completed run*; this one
  explains *a choice of search strategy*, which would mean re-running the
  search under several strategies just to explain the choice. Not offered.
- **`ablation` needs a specific configuration**, not just trial history —
  confirmed it's the mechanism for Phase 1b's "local" view, not a fourth
  global one.
- Timed on a realistic case (the `test2.ihpo` fixture, 30 trials, 4
  hyperparameters): building the `ExplanationTask` is ~35ms; `tunability`/
  `sensitivity`/`mistunability` each take ~0.25-0.3s (their internal
  `n_samples=10_000` random-config sampling dominates); `ablation` against
  one specific configuration is ~45ms — cheap enough to compute live, on a
  click, in Phase 1b, with no caching needed.

## The change

**Concept.** `BaseOptimizer.compute_hp_importance` was one method, one game.
It's now three public methods (`compute_hp_importance` kept, unchanged
signature, since existing tests and every optimizer's serialize path already
name it — plus `compute_hp_sensitivity`, `compute_hp_mistunability`) sharing
one private `_compute_hp_game(config_space, trials, metric, game, seed)`, and
a `compute_hp_games(...)` convenience wrapper that loops every metric × every
game in `HP_GAMES` once, returning `{game: (importance_by_metric,
warning_by_metric)}` — what every optimizer's `optimize()` now calls instead
of its own per-metric loop (previously duplicated three times, once per
optimizer, for tunability alone; would have been nine times for three games).

- `OptimizationResult` gains `hyperparameter_sensitivity`/`_warning` and
  `hyperparameter_mistunability`/`_warning`, mirroring the existing
  `hyperparameter_importance`/`_warning` shape exactly, defaulted to `{}` so
  an `.ihpo` from before this exists still deserializes.
- `serialize_result`/`deserialize_result` (base, and SMAC's own override,
  which doesn't call `super()`) carry the four new fields.
- All three optimizers (`grid_optimizer.py`, `random_optimizer.py`,
  `smac_optimizer.py`) call `self.compute_hp_games(...)` once at the end of
  `optimize()` instead of looping `compute_hp_importance` per metric.
- `hyperparameter_importance_plot` (`ui/figures/plots.py`) now takes a plain
  `importance: dict` instead of `(result, metric)` — the three games produce
  the exact same shape, so one renderer serves all of them; the caller picks
  which dict.
- `HyperparameterImportance.views` (`ui/figures/catalog.py`) becomes the
  9-way product of `HP_GAME_FIELDS` (which `OptimizationResult` field pair
  backs each game) × `HP_RENDERINGS` (`pie`, `bar`, `table`); `plot()` splits
  the view key on `-` to get both. `HP_GAME_FIELDS` is exported from
  `ui/figures` since `views.py`'s `panels` loop needs the same mapping to
  build every game's table rows and warning per metric
  (`panel.importance_by_game[game].table`/`.warning`).
- Client-side: the importance figure now owns two selects (game, rendering)
  composing into one view key, the same pattern `performance_over_time`
  already established for trial/time × score/error. `syncImportanceTable`
  now matches on `data-game` as well as `data-metric`; a new
  `syncImportanceWarning` picks the right game's warning instead of assuming
  there's only one to show.

## Deviations / notes

- **Test suite runtime grew from ~104s to ~138s** (fast suite). Every test
  that runs a real optimizer now pays for three HyperSHAP games instead of
  one, and two of the three (`sensitivity`, `mistunability`) cost ~0.25-0.3s
  each from their internal `n_samples=10_000` sampling. Not fixed here —
  reducing `n_samples` would trade estimate quality for speed and wasn't
  asked for, and in production this overhead is one extra ~0.5s at the end
  of a run that already takes much longer, not a per-page-load cost. Worth
  revisiting if it becomes a real bottleneck (lower `n_samples`, or only
  compute sensitivity/mistunability lazily) rather than pre-optimizing now.
- **Incidental finding, not a regression**: manually testing against a real
  10-trial run where HyperSHAP gave one hyperparameter ~100% tunability
  (`max_features` at `0.9999999999954595`, the rest at ~1e-11) found that
  Plotly's pie chart doesn't render a visible arc for a single near-360°
  slice — the wedge and its label don't show, only the tiny slivers' leader
  lines do. Confirmed this is pre-existing (the `go.Pie(...)` call itself is
  untouched by this phase) and that the bar view — added in Phase 0 for
  exactly this kind of case — renders it perfectly clearly. Left as-is: it's
  a real argument for the alternate-views feature existing at all, not a bug
  introduced here, and fixing Plotly's own near-degenerate-pie rendering is
  out of scope.
- **The RandomForest fallback doesn't distinguish games.** If HyperSHAP fails
  for any of the three, all three fall back to the same generic
  `feature_importances_` — a "best effort, not the real thing" signal that
  already existed for tunability alone; the warning message now says which
  game failed, but the fallback numbers themselves are identical regardless
  of which game asked. Matches the existing fallback's own honesty framing
  (it says what happened) rather than pretending each game has a distinct
  fallback it doesn't.

## Checklist

✅ `mistunability` added, `optimizer_bias` explicitly excluded (needs a live
optimizer ensemble, not just trial history) · ✅ shared `_compute_hp_game` +
`compute_hp_games` convenience wrapper, one call per optimizer instead of a
per-metric loop repeated per game · ✅ `OptimizationResult` fields + serialize/
deserialize (base and SMAC's override) · ✅ 9-view catalog entry (3 games × 3
renderings) · ✅ per-game warnings and table rows · ✅ tests updated
(`test_plots.py`, `test_figures_view.py`, `test_figure_visibility.py`) plus
existing `test_hp_importance.py`/`test_serialization.py` unmodified and still
green · ✅ manual browser verification with a real 10-trial run (not just a
fixture) — confirmed tunability/sensitivity/mistunability give genuinely
different, sensible distributions, not copies of each other; no console
errors.

## Verification

```bash
python -m pytest -q -m "not slow"   # 717 passed, 5 skipped (was 717 before too — this phase net-zero on count, since it revises existing tests rather than only adding)
python manage.py check
python manage.py makemigrations --check --dry-run   # no model changes — OptimizationResult isn't a Django model
```

Manual: created and ran a real experiment (Random Forest + Random Search,
iris, 10 trials) through the browser, switched the importance figure through
all three games × all three renderings, confirmed the numbers genuinely
differ per game and the warning/table sync follows the game selector, not
just the metric selector. No console errors.
