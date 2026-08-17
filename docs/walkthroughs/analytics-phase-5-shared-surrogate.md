# Analytics Phase 5: shared surrogate-fitting utility (docs/PLAN-analytics.md)

**Status:** implemented, awaiting your sign-off.
**You can now:** nothing new to look at — this is pure infrastructure,
unblocking Phase 6 (Partial Dependencies). It also fixes a real, verified
inefficiency spotted while reviewing the codebase: the three global HyperSHAP
games were each independently re-fitting an identical surrogate.

Closes Phase 5 of `docs/PLAN-analytics.md`.

---

## Two things in one phase, both about not duplicating a fit

**1. The planned part**: factor the `RandomForestRegressor`-fitting logic
`_compute_hp_game`'s own fallback rung already had into a standalone
function, so Phase 6's Partial Dependencies doesn't have to duplicate it —
`optimize()` never keeps the real model around after it returns (SMAC's
facade, in particular, is a local variable, discarded at the end of the
call), so anything needing a model afterward needs a fresh stand-in, same as
this fallback already builds one.

**2. An unplanned but verified finding**: `compute_hp_games` calls
`_compute_hp_game` once per `(game, metric)` pair, and the old
`_compute_hp_game` built its own `ExplanationTask`/`HyperSHAP` explainer
*inside itself* every single call — meaning for one metric, the identical
training data (the same trials, the same scores) got a fresh surrogate fit
**three times over**, once each for tunability, sensitivity, and
mistunability, even though only which aggregation (MAX/VAR/MIN) is asked of
the *already-fitted* explainer differs between them. Confirmed by reading
the code, not assumed — and confirmed fixed via a mock-spy test
(`test_compute_hp_games_builds_the_explainer_once_per_metric_not_per_game`)
asserting the explainer-building step is called exactly once per metric
across all three games, not three times.

## The change

- `core/optimizers/base.py`, two new **module-level** functions (not
  methods — nothing about them needs an optimizer instance, and a plain
  function is the easiest thing for Phase 6 to import on its own):
  - `_pair_trials_with_scores(config_space, trials, metric_name)` — the
    "turn trials into `[(Configuration, score), ...]`" loop that used to be
    copy-pasted in `_compute_hp_game`, `compute_hp_ablation`, and (before
    this phase) the fallback rung a third time. Now the one place that
    happens.
  - `fit_surrogate(config_space, trials, metric_name, seed=0)` — fits a
    `RandomForestRegressor` over that paired data. Returns `(surrogate,
    warning)`; `surrogate` is `None` (with a reason) when there are fewer
    than two usable trials or the fit itself raises.
- `BaseOptimizer._build_explainer(config_space, trials, metric_name)` (new,
  private) — the shared "pair trials, fit the HyperSHAP explainer" step
  behind all three global games. Returns `(hs, warning, insufficient)`:
  `insufficient` is `True` only for "fewer than two usable trials," which
  skips straight to uniform weights with no HyperSHAP or RandomForest
  attempt at all (there's nothing to fit either from); `False` with `hs`
  `None` means HyperSHAP itself failed to build one, which still warrants
  trying the RandomForest rung.
- `BaseOptimizer._compute_hp_game(..., explainer=None)` — gained an optional
  `explainer` parameter, `_build_explainer`'s own return tuple. `None` (the
  default) builds its own, preserving the exact original behavior for
  standalone callers (`compute_hp_importance`/`_sensitivity`/
  `_mistunability`, and every existing test against them). The fallback rung
  itself now calls the new `fit_surrogate` instead of its own inline
  RandomForest fit.
- `BaseOptimizer.compute_hp_games` — now builds one explainer per metric via
  `_build_explainer`, then passes it into `_compute_hp_game` for all three
  games in `HP_GAMES`, instead of each game rebuilding its own.
- `BaseOptimizer.compute_hp_ablation` — its own duplicate pairing loop
  replaced with a call to `_pair_trials_with_scores`; behavior unchanged
  (still no RandomForest-fallback rung, per Phase 1b's own reasoning — a
  signed local explanation has no honest unsigned stand-in).

## Deviations / notes

- **Minor, accepted duplication remains**: `fit_surrogate` re-derives
  `_pair_trials_with_scores` from raw `trials` rather than accepting an
  already-paired `data` list, so `_compute_hp_game`'s fallback rung re-pairs
  trials it (or `_build_explainer`) already paired once earlier in the same
  call. Deliberate: pairing is a handful of cheap dict/Configuration
  operations, not a model fit, and keeping `fit_surrogate`'s signature
  `(config_space, trials, metric_name, seed)` — matching every other
  `compute_hp_*` method in this file exactly — means Phase 6 (or any future
  caller) never needs to know the paired-tuple shape at all. Optimizing away
  a cheap loop at the cost of a more awkward public API was the wrong trade.
- **Test-suite runtime is not visibly faster**: the fix removes roughly
  two-thirds of a ~35ms fit, three times, per metric per `optimize()` call —
  real, but small (well under 100ms) against everything else one real
  optimizer run does (SMAC's Bayesian loop, cross-validation, actual model
  fits). The full suite measured 158.78s here, statistically indistinguishable
  from Phase 4's own 154.46s baseline given the noise already characterized
  in Phase 1b's walkthrough. The savings are real per-request (verified by
  the spy test counting explainer builds, not by timing), just not large
  enough to separate from ambient measurement noise in aggregate.
- **Sensitivity/mistunability's own interaction grids remain unwired** —
  unchanged from Phase 2's own note; this phase didn't touch that decision.

## Checklist

✅ `fit_surrogate`/`_pair_trials_with_scores`, module-level, no optimizer
instance needed · ✅ `_build_explainer`, shared by `compute_hp_games` and
standalone `compute_hp_game` callers alike · ✅ explainer built once per
metric, not once per (game, metric) — verified by a mock-spy test, not
inferred · ✅ `compute_hp_ablation`'s duplicate pairing loop removed ·
✅ every existing `compute_hp_importance`/`_sensitivity`/`_mistunability`/
`_ablation` test still passes unmodified — the standalone-call contract is
unchanged · ✅ new tests for `fit_surrogate` directly (insufficient trials,
a usable fitted regressor) and for the once-per-metric guarantee.

## Verification

```bash
python -m pytest -q -m "not slow"
# 753 passed, 5 skipped, 36 deselected in 158.78s — within noise of
# Phase 4's 154.46s baseline; see "Deviations" above for why this fix
# doesn't show up as a visible aggregate speedup.
python manage.py check
python manage.py makemigrations --check --dry-run   # no model changes
```

No manual browser verification for this phase — pure backend infrastructure,
no template/JS touched, and every existing figure that depends on
`compute_hp_games`'s output (importance, interactions) is covered by the
existing test suite's assertions on its actual values, which are unchanged.
