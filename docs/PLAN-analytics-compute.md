# Analytics compute policy: what gets computed when

## Context

`docs/PLAN-analytics.md` (Phases 0–6) is closed out: every DeepCave-inspired
figure that codesigner set out to build is built. But it was built
figure-by-figure, each phase deciding for itself whether its numbers were
cheap enough to ship in the page payload or expensive enough to fetch on
demand. Nobody ever stood back and asked the question across all of them at
once.

Doing that turns up one policy problem and three bugs.

The policy problem: **a page reload refits models.** The partial-dependence
endpoint fires automatically on every page load and takes 1.55 seconds. There
is no server-side cache anywhere in this repo — no `django.core.cache`, no
`CACHES` block in `config/settings.py`, no `lru_cache` or `cached_property`
(checked, not assumed). The only caching is two JavaScript objects
(`pdpCache`, `localAblationCache` in `experiment_detail.html`) that are
discarded on reload.

This roadmap inventories every non-trivial computation behind the analytics,
fixes what is redundant, and gives the *user* the say over what runs
automatically.

## What "non-trivial" means here

Anything past simple arithmetic over already-stored trial data: fitting a
surrogate, evaluating a HyperSHAP/shapiq game, reconstructing a
`ConfigurationSpace`, predicting across a grid. Deliberately excluded as
simple algebra: `incumbent_scores`, `best_idx`, the cube/parallel-coordinates/
trial-duration/trials-table builders, selected-config deltas.

All costs below were **measured**, on `tests/fixtures/test2.ihpo` (Random
Forest, 4 hyperparameters, 30 trials, 4 metrics) plus a synthetic 6-HP SVM
run — not estimated from reading the code.

| # | Computation | Where | Measured | Scales with |
|---|---|---|---|---|
| 1 | `Configuration` construction + config-space validation per trial | `base.py::_pair_trials_with_scores` | 0.5ms / 30 trials | n_trials |
| 2 | HyperSHAP explainer — `ExplanationTask.from_data`, **fits a RandomForest internally** | `base.py::_build_explainer` | 33ms | n_trials |
| 3 | One HyperSHAP global game — a shapiq `ExactComputer` over **2^n_hp coalitions**, each a batched 10,000-row surrogate predict | `base.py::_compute_hp_game` | 0.27s @ 4 HPs, 1.07s @ 6 HPs | **2^n_hp**, *not* n_trials |
| 4 | `_extract_pairwise` order-1/order-2 reshape | `base.py::_extract_pairwise` | free | — |
| 5 | Plain surrogate fit, `RandomForestRegressor(100)` | `base.py::fit_surrogate` | 36ms | n_trials |
| 6 | PDP/ICE — surrogate fit + n_trials × n_points predictions | `base.py::compute_partial_dependence` | **1.55s** (a defect — see Phase 1) | n_trials × n_points |
| 7 | Local ablation — a *second* `ExplanationTask` fit + the ablation game over 2^n_hp | `base.py::compute_hp_ablation` | 77ms @ 4 HPs, 194ms @ 6 HPs | **2^n_hp** |
| 8 | Result rebuild from stored JSON, per request | `base.py::deserialize_result` | O(n_trials) | n_trials |
| 8b | **SMAC only** — `mkdtemp` + rewriting every embedded state file, per request, never cleaned up | `smac_optimizer.py::deserialize_result` | 0.5ms + an unbounded temp-dir leak | every request |
| 9 | ~57 `go.Figure` builds + `to_json` per render | `views.py::_detail_context` | unmeasured | figures × metrics × views |

The per-coalition cost is a stable **~17ms** (0.27s/16 and 1.07s/64 both give
it), so a run's eager analytics cost is
`0.017 × 2^n_hp × 3 games × n_metrics`:

| Hyperparameters | Eager cost at run completion (4 metrics) |
|---|---|
| 4 (Random Forest) | 3.4s — measured |
| 6 (SVM Classifier) | 13s — measured |
| 8 | ~51s — extrapolated |
| 10 (a custom upload) | **~3.5 minutes** — extrapolated |

That last row is the motivating pathology: the cost is exponential in
hyperparameter count and **independent of how long the run was**. A short run
of a wide model is the expensive case, which is the opposite of most people's
intuition.

## The categorization

**Computed automatically after a run, and stored.** The three global HyperSHAP
games (tunability / sensitivity / mistunability) and the pairwise interaction
grid — #2, #3, #4 above. This is already what happens, and it is right. The
cost argument is only half of it: these are *inputs to other figures*.
Parallel coordinates orders its axes by `hyperparameter_importance`, and each
metric panel's default PDP hyperparameter (`top_hp`) is the most important
one. Deferring them would quietly degrade three figures, not one.

**Computed only when asked, and never precomputed.** Local ablation (#7) is
one value per (metric, **trial**) — precomputing means n_trials × n_metrics
games. PDP (#6) is n_trials × n_points floats per (metric, hyperparameter); a
500-trial, 10-HP, 4-metric run would put megabytes of ICE data into
`Experiment.result`. Both are rejected for precomputation on *storage*
grounds, independently of what they cost to compute.

**Neither, today: computed on request, but requested automatically.** PDP
auto-fires on page load. That is the thing that literally matches "recomputing
on reload," and Phase 4 puts it under the user's control.

## Decisions taken up front

- **No server-side cache.** After Phase 1's fix a reload costs ~52ms, measured,
  most of it the one surrogate fit. The only genuine crossover is local ablation at
  **n_hp ≥ 7**, where 2^n starts to bite. Building an invalidation fingerprint,
  a size cap and an eviction policy for a cost that no longer exists would be
  infrastructure for its own sake. Revisit if that crossover is ever reached.
  (An in-memory cache was rejected regardless: deployment is
  `gunicorn --workers 3` plus a separate huey consumer, so a reload would land
  on a warm worker about a third of the time. Intermittently fast is worse
  than consistently fast enough.)
- **Per-figure control, spelled as booleans.** Two states is a boolean, so the
  `bool()` coercion in `_posted_settings` needs no change and no new widget
  type is needed. The third state a tri-state would have offered — "never" —
  already exists as the `show_<key>` visibility flag.
- **The eager cost guard is a deployment setting**, not a per-experiment one:
  it is a statement about machine capacity, not about this experiment's taste.
- **Analytics stay inside `optimize()`.** Moving them to the run service was
  considered and rejected — the one bug that argued for it (cancel not
  skipping analytics) turns out to be a one-line guard where `cancel_event` is
  already in scope.

## Phase 1 — Stop wasting work

> **Status:** done. See `docs/walkthroughs/analytics-compute-phase-1-waste.md`.

No new concepts, no settings, no schema change. Three independent defects.

- **Partial dependence spends almost all of its time on sklearn call overhead.**
  The grid loop does one `Configuration` construction *and one single-row
  `rf.predict([row])`* per (trial, grid point). Measured: 600 single predicts
  = 1.544s; the same 600 rows in one batched `rf.predict(matrix)` = 0.0035s
  (**~440× on the prediction alone**); all 600 `Configuration` constructions =
  0.010s. Batching the prediction takes the whole endpoint from **1553ms to
  52ms — 30×** — after which the 36ms surrogate fit is the dominant cost and
  the grid work is noise.
- **`compute_hp_ablation` ignores its `seed`.** The parameter is declared and
  never used — `ExplanationTask.from_data` always fits at `random_state=0`, so
  a local explanation silently ignores the experiment's seed.
- **SMAC leaks a temp directory per request.** `deserialize_result` calls
  `tempfile.mkdtemp()` and rewrites the entire run history on *every* rebuild
  — page load, trial click, ablation fetch, PDP fetch — and nothing ever
  deletes it. Verified: 10 rebuilds, 10 directories. Carry the state in memory
  instead; nothing on any read path needs the files.

## Phase 2 — Compute derived values once per render

> **Status:** done. See `docs/walkthroughs/analytics-compute-phase-2-derived-once.md`.
> Also fixed a pre-existing 500 found alongside it: a result with no trials
> (importable — `io.parse` accepts an empty `data`) crashed the detail page.

Pure cleanup, no visible change.

- `best_idx` is computed six times per metric per render, and `incumbent_scores`
  once per view. The cost is trivial; the risk is that six copies of an argmax
  can disagree about tie-breaking. One memoized accessor on `OptimizationResult`.
- The RandomForest fallback rung inside `_compute_hp_game` is not shared across
  the three games, unlike the explainer, which deliberately is — so on the
  fallback path it refits an identical model up to three times.

## Phase 3 — Eager cost guard and cancel-skip

- A coalition budget in `config/settings.py`, checked against
  `2^n_hp × games × metrics`, so a wide custom model can't append minutes to
  every run. When it trips, the reason goes into the existing per-metric
  `_warning` fields — which the page already surfaces, so this needs no new UI.
- Cancelling a run currently still pays the full analytics cost. On a 10-HP
  model that means waiting ~3.5 minutes for a run you just cancelled. Skip on
  cancel; resuming recomputes over all trials, so nothing is permanently lost.

## Phase 4 — Per-figure control over deferred computation

The direct answer to "nothing should be recomputed on reload."

Each deferred computation gets an `autocompute_<name>` boolean, declared on the
figure that owns it. The setting keys name the *computation*, not the figure,
because local ablation is a **view of** the importance figure rather than a
figure of its own — `compute_hyperparameter_importance` would be actively
misleading.

Combined with the existing visibility flag, that gives three states without a
tri-state widget:

| `show_<key>` | `autocompute_<name>` | Behaviour |
|---|---|---|
| off | — | never computed |
| on | on | computed on load and on metric switch (today's behaviour) |
| on | off | a "Compute" button; **a reload computes nothing** |

## Explicitly not doing

- **A server-side cache**, per the decision above.
- **Persisting PDP/ICE or ablation results** into `Experiment.result` — the
  storage cost is unbounded in trial count.
- **Moving analytics out of `optimize()`** into the run service.
- **Removing the unused sensitivity/mistunability interaction grids.** They are
  extracted (`_extract_pairwise`) for all three games but only tunability's is
  stored or read. The extraction is a free reshape of data already in memory,
  so deleting it is churn, and *storing* it is a new feature (an interactions
  game selector), not a cleanup.
- **Optimizing the ~57-figure render cost (#9).** Genuinely unmeasured, and
  plausibly the largest remaining page cost once Phase 1 lands. Measure it
  after Phase 1 and decide then; it is its own epic, not a footnote to this one.

## Verification (per phase)

Matching the habit the rest of this project already follows:

- `python -m pytest -q -m "not slow"` green after each phase.
- `manage.py check` and `makemigrations --check --dry-run` clean.
- One walkthrough per phase under `docs/walkthroughs/`, one commit per phase.
- Any change in test-suite runtime reported, not silently absorbed.
