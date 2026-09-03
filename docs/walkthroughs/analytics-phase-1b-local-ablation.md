# Analytics Phase 1b: local explanation via ablation (docs/PLAN-analytics.md)

**Status:** done. Closes out Phase 1 of `docs/PLAN-analytics.md`.
**You can now:** switch the Hyperparameter importance figure to "Local
(selected trial)" and see a signed, diverging bar chart explaining the
*currently selected* trial specifically — which of its hyperparameter values
helped or hurt versus the config space's default, not a global share like
the other three games. It follows whatever trial is clicked on Performance
over time, and defaults to the metric's best trial before any click, the
same way the selected-config panel already does.

Second half of Phase 1 from `docs/PLAN-analytics.md`, closing it out.

---

## Why this one needed its own design, not just a fourth row

The three global games (Phase 1a) are static: computed once, at run
completion, stored on `OptimizationResult`, read back on every page view for
free. Local ablation cannot work that way — its answer depends on *which
trial* you're asking about, and that changes with every click. Two
consequences shaped the whole implementation:

1. **`Figure.plot(result, metric, view)` cannot compute it.** That signature
   only ever receives the stored result — no model, no config space, no
   notion of "which trial." Every other view lives entirely inside that
   contract; this one structurally can't. `HyperparameterImportance.views`
   still lists `"local-bar"` (so `figure_views` correctly advertises it as a
   real option), but `plot()` returns `None` for it and `views.py`'s
   `plot_json` explicitly skips it when building the page's initial payload —
   the real computation happens elsewhere, described below.
2. **It costs real time, repeatedly** — confirmed by Phase 1a's own spike
   (~45ms for the ablation game itself, plus ~35ms to refit the surrogate
   from trial history each time, since nothing keeps one around between
   calls). That's cheap *once*, but the initial design draft of this phase
   had `_detail_context` precompute a default (best-trial) answer for every
   metric on every page load — i.e., paying that cost on every single
   *view* of the page, not once at run completion like the other three
   games. Caught before shipping it: reverted in favor of fetching lazily,
   the same way a click already fetches the selected-config panel.

## The change

- `BaseOptimizer.compute_hp_ablation(config_space, trials, metric_name,
  config_of_interest, seed)` (`core/optimizers/base.py`) — HyperSHAP's
  `ablation` game, `config_of_interest` = the trial being explained,
  baseline = the config space's default. **Signed, deliberately** — no
  `abs()`, no normalize-to-1, unlike `_compute_hp_game`'s three siblings:
  direction (did this hyperparameter's value help or hurt) is the entire
  point of a local explanation, and zeroing it out would answer a different
  question. No RandomForest-fallback rung either — a plain
  `feature_importances_` has no sign and doesn't answer the same question,
  so failure is reported as a warning rather than answered with something
  that resembles an answer but isn't one.
- `hyperparameter_ablation_plot(ablation)` (`ui/figures/plots.py`) — a
  diverging bar, colored by sign, ranked by magnitude (the same "which
  mattered most" ordering as the other three games' bar view, just allowing
  negative bars). No pie, no table: a share-of-a-whole framing doesn't apply
  to signed values, and this game only ever gets the one rendering.
- `ui/views.py`:
  - `_rebuild_experiment` (new) exposes the full `build_experiment()` dict —
    the model (for its config space) and the optimizer instance, neither of
    which the stored `OptimizationResult` alone carries. `_rebuild_result`
    is now a thin wrapper over it, unchanged for its existing callers.
  - `_local_ablation_data(built, metric, idx)` — the shared logic behind
    both the lazy default and a click: resolves the config space from
    `built["model"]` (`None` for a custom-model experiment viewed
    read-only — reported as an unavailable-not-crashed warning, not a 500),
    calls `compute_hp_ablation`, builds the figure.
  - `trial_ablation` (new view, `GET /experiments/<pk>/trial-ablation/?metric=&idx=`) —
    the same shape as the existing `trial_panel` endpoint (same validation,
    same `@experiment_view(VIEW)`), returning JSON (`{figure, warning}`)
    instead of an HTML fragment, since there's a chart to draw rather than a
    panel to swap in.
  - `panels[]` gains `best_idx` (the array index of each metric's best
    trial — distinct from the existing `best_n`, that trial's own number) so
    the client knows what to fetch by default, before any click.
- Client-side (`experiment_detail.html`): `refreshLocalAblation(metric)` — a
  no-op unless "Local" is the active game, called from every place that could
  change what it should be showing (switching to it, a click while it's
  already showing, a metric switch while it's showing). Resolves the trial
  index from `lastClickedIdx` (already tracked, from Phase 0) or falls back
  to `bestIdxByMetric`; caches by `"metric:idx"` so switching games and back
  doesn't refetch. The rendering select (pie/bar/table) hides itself while
  "Local" is active, since none of that applies — one less thing on screen
  that would do nothing if touched.

## Deviations / notes

- **The eager-precompute design was reverted before shipping**, not after —
  see "Why this one needed its own design" above. Worth calling out
  explicitly since it's the kind of thing that's easy to ship by default
  (every other view *is* precomputed, so it was the path of least
  resistance) and only wrong once you notice the cost model differs.
- **A brief "no data" flash is possible** the first time a trial's ablation
  is requested — `draw()` shows the empty-message caption for anything
  falsy, and a missing dict key (before the fetch resolves) is exactly as
  falsy as `None`. Consistent with how the selected-config panel already
  behaves on a fresh click (no loading spinner anywhere in this app for
  fetches this fast); not treated as a gap worth a spinner for ~100-200ms.
- **Test suite runtime measured, then re-measured clean**: an initial full-suite
  run came back at 217.27s against Phase 1a's own final figure of 140.82s — a
  jump too large to wave off. `--durations=25` showed it spread across
  pre-existing tests that run real optimizer searches, not concentrated in
  the new ablation tests (the two that showed up in the top 25 total under
  16s combined), so it wasn't Phase 1b's actual code. The likely cause: a
  `manage.py runserver` left over from this phase's own manual browser
  verification was still running underneath the measurement. Killing it and
  re-running came back at 194.85s, then 148.62s on a second clean run —
  within noise of Phase 1a's baseline. Recorded here since silently
  swallowing a 55%-looking regression (or silently trusting the first
  number) would both have been wrong; the honest answer is "no real
  regression, confirm the dev server is down before trusting a timing run."

## Checklist

✅ `compute_hp_ablation`, signed, no fallback rung · ✅ diverging bar view,
no pie/table · ✅ `_rebuild_experiment` exposing model/optimizer/seed ·
✅ `trial_ablation` endpoint, same validation shape as `trial_panel` ·
✅ lazy fetch + cache + rendering-select hide, wired to switch/click/metric-switch ·
✅ graceful "model unavailable" path for custom-model experiments viewed
read-only · ✅ tests (`test_hp_importance.py`, `test_plots.py`,
`test_trial_ablation.py`, `test_permissions.py`'s route audit) · ✅ manual
browser verification: default fetch on switching to Local, refetch on click,
cache hit on switching away and back, refetch on metric switch, no console
errors.

## Verification

```bash
python -m pytest -q -m "not slow"
# 730 passed, 5 skipped, 36 deselected in 148.62s, clean (no dev server running)
python manage.py check
python manage.py makemigrations --check --dry-run   # no model changes
```

Manual: on a real 10-trial run, switched the importance figure to "Local,"
confirmed the rendering select hid itself and a request went out for the
metric's best trial; clicked a different point on Performance over time and
confirmed a fresh request went out and the bar chart changed to a genuinely
different (and correctly signed) distribution; switched to Tunability and
back to Local and confirmed no new request fired (cache hit); switched
metric and confirmed a fresh request went out for the new metric's best
trial. No browser console errors throughout.
