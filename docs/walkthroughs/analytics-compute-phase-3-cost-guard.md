# Analytics Compute Phase 3: eager cost guard and cancel-skip (docs/PLAN-analytics-compute.md)

**Status:** implemented, awaiting your sign-off.
**You can now:** cancel a run and have it actually stop, instead of waiting out
the analytics for a run you just cancelled. And uploading a model with many
hyperparameters no longer silently appends minutes to every single run — past a
configurable budget the importance analytics are skipped, and the page says so.

New deployment setting: `ANALYTICS_EAGER_MAX_COALITIONS` (default 1024).

---

## The cost is exponential in hyperparameters, not in trials

This is the fact the whole phase turns on, and it's counter-intuitive enough to
state plainly: a shapiq `ExactComputer` evaluates **2^n_hp coalitions** per game,
and each coalition costs 15–19ms **regardless of how many trials the run had.**

So **a short run of a wide model is the expensive case.** A 500-trial search over
4 hyperparameters is cheap; a 12-trial search over 10 is not.

Measured, over 4 metrics and all three games:

| Hyperparameters | Coalitions | Eager cost |
|---|---|---|
| 4 (Random Forest) | 192 | **3.6s** measured |
| 6 (SVM Classifier) | 768 | **12.7s** measured |
| 8 | 3,072 | **45.6s** measured |
| 10 (a custom upload) | 12,288 | ~3 minutes, extrapolated |

I measured the 8-hyperparameter point specifically to check the extrapolation
rather than trusting it — and it's just as well, because it came in at 45.6s
against the ~51s I'd predicted. The per-coalition constant isn't flat: it drifts
down as the count grows (18.6 → 16.5 → 14.8ms) as each game's fixed overhead
amortizes. That revised the headline 10-hyperparameter figure from ~3.5 minutes
down to ~3, and every number in the roadmap, settings comments and `.env.example`
was corrected to match.

## Why the budget counts coalitions

Not seconds, and not hyperparameters:

- **Seconds** would bake a machine-calibrated constant into the code. The 15–19ms
  is *this* machine; a slower host would want a different number for the same
  policy.
- **Hyperparameter count** ignores that metric count multiplies it. Four metrics
  cost four times one metric at the same width.

Coalitions are the unit the work is actually done in, so the budget means the same
thing on any machine. `2^n_hp × 3 games × n_metrics`, checked before anything is
fitted.

The default of 1024 was chosen against the table above: it admits both models
that ship (192 and 768) and turns away the custom-upload pathology (12,288). `0`
means no limit.

## Where the decision lives, given `core/` can't see Django settings

`compute_hp_games` is called from inside each optimizer's `optimize()`, and
`core/` is deliberately Django-free. Rather than move the analytics out to the run
service — considered and rejected in the roadmap, since it's a wide refactor for
no user-visible gain — `BaseOptimizer` carries a plain
`analytics_max_coalitions` attribute defaulting to a module constant, and
`ui/services/run.py` sets it from the setting before calling `optimize()`. One
line in `ui/`, no Django import in `core/`, and `optimize()`'s signature (which
tests encode) is untouched.

`test_the_run_service_applies_the_deployment_setting` pins that wiring, because a
silent failure there means the guard never fires in production while every unit
test still passes.

## Cancelling a run stopped paying for analytics

Every optimizer breaks out of its trial loop on `cancel_event` — and then called
`compute_hp_games` unconditionally. So Cancel bought you the full analytics bill
on a run you had just stopped: on a wide model, minutes of it.

`compute_hp_games` now takes `cancel_event` and declines up front. Both guards
live there rather than being repeated at three call sites, which is also why the
three optimizers each needed only a one-argument change.

Checked cancel-first, deliberately: a cancelled run shouldn't be lectured about a
budget it never got to spend.

**Nothing is permanently lost.** Resuming recomputes the games over
`previous + new` trials, so a cancelled run's analytics are one resume away — and
the warning text says so, rather than leaving you to guess.

## No new UI, and that claim is tested

A skip produces empty values plus the reason in the per-metric
`hyperparameter_*_warning` fields — **deliberately the same shape a HyperSHAP
failure has always produced**, so nothing downstream has to tell "couldn't" from
"wouldn't". The page already renders those warnings, so a skipped run explains
itself with no template changes at all.

Two figures read importance for something other than drawing it, and "they
degrade cleanly" was the argument for building no new UI — so it's tested rather
than asserted:

- **Parallel coordinates** orders its axes by importance. With none, `.get(h, 0.0)`
  makes every key equal and Python's stable sort leaves config order intact.
- **The partial-dependence picker** defaults to the most important hyperparameter
  (`top_hp`), falling back to the first config key.

Confirmed in a browser as well as in tests: the amber warning box carries the full
reason on both the importance and interactions figures, those two show their "No
… data available" captions instead of blank charts, and every other figure —
performance over time, trial duration, best/selected config, parallel
coordinates, partial dependence — works normally. No console errors.

## Deviations from the plan

- **Both guards went into `compute_hp_games` rather than into the three
  optimizers.** The plan said "guard the call on `cancel_event` in all three
  optimizers"; putting it one level down means one implementation instead of
  three, and lets the cancel and budget skips share the `_skipped_games` shape.
- **The cost numbers changed**, as above. The 8-hyperparameter measurement was
  not in the plan; I added it because publishing an extrapolation as the
  justification for a default felt too thin, and it turned out to be off by ~10%.
- No `tests/config/test_settings.py` was added for the default (the plan
  suggested one) — there is no such file and no precedent for testing settings
  defaults in isolation; the default is covered where it matters, through
  `test_the_default_budget_admits_both_registry_models` and the run-service test.

## Checklist

- [x] `ANALYTICS_EAGER_MAX_COALITIONS` setting, documented, in `.env.example`
- [x] budget checked as `2^n_hp × games × metrics` before anything is fitted
- [x] wired from the run service, keeping `core/` Django-free
- [x] analytics skipped on cancel, with the recovery path named in the warning
- [x] cancel reported before budget
- [x] skip reported through the existing warning channel — no new UI
- [x] degradation tested at both consumers of importance
- [x] 18 new tests

## Verification

- `pytest -q -m "not slow"` → **804 passed, 5 skipped, 36 deselected in 196.23s**,
  up 18 from Phase 2's 786. Runtime is flat against Phase 2's 190.9/193.4/196.5s
  band; the new tests cost 6.5s measured (5.87s + 0.68s), most of them cheap
  precisely because the guard makes them skip the expensive path.
- `manage.py check` → no issues. `makemigrations --check --dry-run` → no changes.
- **The guard, end to end at three widths** with the real default of 1024:
  4 hyperparameters → 192 coalitions, computed in 3.575s; 6 → 768, computed in
  12.704s; 10 → 12,288, **skipped in 0.000s** with the reason attached.
- **The extrapolation, checked**: 8 hyperparameters forced through with no budget
  → 45.6s for 3,072 coalitions.
- **In the browser**, on an experiment whose stored result carries a skip: warning
  visible with full text on both affected figures, parallel-coordinates axes in
  config order (`max_depth, max_features, min_samples_split, n_estimators,
  Accuracy`), partial dependence defaulted to `max_depth` and drew both traces,
  importance figure showed its empty caption with zero traces, no console errors.
  (That experiment was a synthetic row I created for the check and deleted
  afterwards — it was not a real run, and leaving a fabricated warning in the dev
  database would have been misleading.)
