# Scoring that knows what it is measuring, and reading somebody else's run

**Status: the import path is complete. Stages 0–4, 9 and 10 landed, plus the
halves of 7 and 8 that an import needs. Stages 5, 6 and the registration half of
8 remain — all of them *running*-side work, none of them reachable from an
import, which computes no scores at all.**

## Context

Three asks that turn out to be one piece of work in a fixed order.

**Read a SMAC output directory.** SMAC writes five files. The `.ihpo` `result`
block is already a strict superset of `runhistory.json`, and `optimizer_state`
already embeds the other four verbatim — so the trial data slots in almost free.
The one real obstacle was that `deserialize_result` hardcoded
`score = 1.0 - cost`, exact for a run codesigner wrote and nonsense for an RMSE
or a log-loss. Decision taken: **import the cost as-is**, as a lower-is-better
metric. Which means the scoring work has to come first.

**Align `.ihpo` with SMAC's layout.** Mostly we are already aligned or ahead.
One change is worth making, and it fixes an existing gap: the config space is
embedded in `optimizer_state` and never read, while `_config_space_for`
re-derives one from the model — so an `.ihpo` imported *without* its custom
model `.py` silently loses every surrogate-backed figure.

Re-examined when the importer was built, and the answer stands: **no further
alignment is worth doing.** `.ihpo`'s `result` block already *is*
`runhistory.json` — the same four top-level keys, each data entry carrying SMAC's
ten fields verbatim plus our three. The three places they differ are each right
as they stand, and the importer normalizes them at the boundary rather than the
format bending toward a foreign one:

1. **`configs` keying.** We hold `config_id == trial number`, which the whole
   selection layer addresses trials through. SMAC keys its data by
   `(config_id, instance, seed, budget)`, so one config can head several rows.
   Relaxing the invariant to accommodate multi-fidelity data we do not run would
   change every figure's selection; renumbering on import does not.
2. **`config_origins` keying.** Ours is by trial over recorded trials, SMAC's by
   config id over every config it held. `smac_optimizer.py` already records why
   ours is the more accurate one: SMAC reads origins off live `Configuration`
   objects that the local-search maximizer relabels in place.
3. **`Infinity`/`NaN`.** Ours is the stricter format — see Stage 10, where it
   turned out not to be strict enough yet.

**AUC, and metrics that are not 0–1.** AUC's blocker was never the range — AUC
is 0–1 and higher-better. It is that `fit_predict` returns one predicted *label*
per row, and `roc_auc_score` needs probabilities. A contract change, separate
from the orientation work, and both are wanted.

Full design: `~/.claude/plans/with-that-in-mind-gentle-cloud.md`.

---

## Landed

### Stage 0–1 — a metric says what it is

`core/metrics.py`. `METRICS` is now `dict[str, Metric]`: `name`, `fn`,
`higher_is_better`, `bounds`, `needs`, `null_score`. The four original metrics
keep their keys, their `fn` bodies verbatim, and exactly the assumptions the
application used to make about all scores.

`to_cost`/`from_cost` **derive** the number an optimizer minimises instead of
hardcoding `1.0 - score` at six call sites:

- higher-better with a finite upper bound → `hi - score` → `1.0 - score` for
  every existing metric, so no file changes and no convention flag
- higher-better and unbounded → `-score`
- lower-better → `score`, unchanged — which is what makes importing a raw SMAC
  cost honest rather than a guess about what the number meant

### Stage 2 — orientation, everywhere

Direction now comes from the metric at ~25 sites: `best_index`,
`incumbent_scores`, `rebase_history`, `target_score`, the collector's incumbent,
the surrogate-uncertainty slice, all three optimizers' result assembly, and
SMAC's two cost conversions. `BaseOptimizer.new_collector` resolves the metric
once, so none of the three optimizers has a line about it — the same seam that
already carries the cancellation gate and the progress callback.

`OptimizationResult.declared_metrics` arrived here rather than in Part B, because
`deserialize_result` needs a direction to read a cost back and an imported run's
objective exists only inside its own file. A field rather than a global the
importer mutates: two experiments open at once must be able to disagree about
what "cost" means. The registry's own metrics are deliberately **never** written
into a file — a later build correcting what `f1` means must not be overruled by
every file written before the correction.

**Verified neutral: 1208 passed, 0 failed**, with `test_serialization.py` and
`test_metric_change.py` — which pin `cost == 1 - score` — passing *untouched*.
That is the whole proof of this stage.

### Stage 3 — what a failure scores, and a JSON bug

A failed trial scored `0.0` on every metric, written down as though that were a
property of failure. It is a property of *those four metrics*. On an RMSE, 0.0
is a **perfect** score. A failed trial now scores the metric's null score — what
a model that knows nothing would get — still exactly 0.0 for every metric that
predates this.

Two bugs came out of it, both pre-existing:

- **A crash could be the incumbent.** `TrialCollector.record` had no failure
  check, so in an all-failing run the first crash passed `0.0 > -inf` and the
  page reported a crashed configuration as the best one. Being ≤ every real
  score is what kept it invisible, not anything that made it right.
- **`-Infinity` in a stored file.** After a failing start the incumbent is still
  the metric's worst sentinel, and `ui/services/run.py` writes a partial result
  while that is true. `json.dumps` emits the bare token `-Infinity`, which Python
  reads back and nothing else does — so that `.ihpo` is not openable by a strict
  parser. `best_score` and each trial's `incumbent_score` are now guarded at the
  serialization boundary and written as `null`, which flows back through the
  collector's existing "nothing to beat yet" path with no special case.

### Stage 4 — probabilities

`fit_predict_proba` is optional on the SDK and returns `(y_pred, y_proba,
classes)` in **one call**, so a run scoring both accuracy and AUC does not fit
twice per trial. `classes` is returned rather than inferred: the scoring code
never sees the model, and guessing the column order is right for scikit-learn
and silently wrong for anything else — which does not fail, it scores the wrong
class.

The wire gains `capabilities` in `hello`, `want_proba` on a trial, and
`y_proba`/`classes` on a result — all optional keys, **no `PROTOCOL_VERSION`
bump**, because the check is exact equality and a bump would stop every already-
built environment until it was rebuilt.

Two things worth recording:

- **The planned capability check was wrong, and a test caught it.** The design
  said to duck-type with `hasattr`. Once `BaseModel` *defines* the method, every
  model advertises probabilities. The SDK now marks its own stub with `_is_stub`:
  an older cached SDK has neither method nor flag, this one has both, an override
  has the method and not the flag — all three read correctly with no reference to
  the SDK from the host.
- **The SVM fits differently when asked.** `probability=True` costs an internal
  five-fold Platt calibration, and `predict` on the calibrated estimator can
  disagree with the `argmax` of its own `predict_proba`. `fit_predict_proba`
  returns the calibrated model's labels, since those are the ones that belong
  with those probabilities. That divergence is exactly why this is a separate
  method rather than a flag, and why a run that never asks never pays.

### Stage 7 (display half) — LANDED

An imported cost is lower-is-better and unbounded, and it goes on the page, so
the display half was required:

- `core/metrics.py` gained `to_error` and `error_range`. `to_error` is *not*
  `to_cost`: a cost only has to reverse an ordering, so negating an unbounded
  score is a fine cost, while an error is a distance from a fixed best and an
  unbounded metric has none. The error view returns None there rather than
  drawing an axis measured from nowhere.
- `absolute_scale` became `Figure.absolute_scale_for(metric)`, resolved per
  metric and per view, and the page payload is keyed by metric. A metric with no
  bounds yields `{}`, and the toggle *hides* rather than pinning the axis to some
  other metric's range. Accuracy reproduces the old `[0, 1]` / `[-3, 0]` exactly.
- Colour direction: `reversescale` on the cube, the projection and parallel
  coordinates' colour bar; parallel coordinates' worst-first z-order now sorts by
  an oriented shade, so colour and z-order cannot disagree about which end is
  good.
- The incumbent's improvement test and the panel's delta arrow come from
  `metric.better` rather than from `> 0`.
- `target_score`'s bounds come from the metric, and an out-of-range value is
  **clamped rather than dropped**. Dropping made it *absent*, so a run asking for
  nothing else was then refused for having no criterion at all — which names a
  different problem from the one the reader created. The field is
  `type="number"`; its old text pattern rejected a minus sign and exponent
  notation.

### Stage 8 (orientation half) — LANDED

`_pair_trials_with_oriented_scores`, beside `_pair_trials_with_scores`, on the
explainer path only — `_build_explainer` and `compute_hp_ablation`'s own
surrogate. `fit_surrogate` stays in raw units as planned: partial dependence
labels its axis with the metric, and the uncertainty field's tree spread is
orientation-free.

Negation rather than `metric.high - score`, because an imported objective has no
upper bound and both preserve every difference between two scores.

Measured on the fixture run: the incumbent's `depth` reads `+0.547` oriented and
`-0.547` unoriented. Without this, the best configuration's best hyperparameter
is reported as the one that hurt it most.

`_GAME_PIECES` labels are unchanged, as planned.

### Stage 9 — LANDED

An optional top-level `space` section holding ConfigSpace's serialized dict,
byte-identical to SMAC's `configspace.json` (verified: `to_json` is
`json.dump(to_serialized_dict())`). `_config_space_for` prefers it and falls back
to the model; `io.config_space_from_serialized` deep-copies before decoding,
since `from_serialized_dict` pops `"type"` and destroys its input.

Three call sites reached past `_config_space_for` to `model.get_config_space` —
local ablation, partial dependence, local effects. Routing them through it is
what lets those three draw for a run with no model.

`Experiment.config_space` is filled from the file on import, and **from the model
at the moment a run has one** (`services/run.py::_remember_config_space`). That
second source fixes the case the stage was written for: a custom model runs in
its own process and is gone afterwards, so a page rendered later had nobody to
ask.

The latent `normalize()` bug is fixed — a format newer than this build is refused
by name rather than run through the format-1 lifter — and the check sits in
`parse()` *before* its field validation, which otherwise reported whichever
format-1 field a future file happened to lack.

### Stage 10 — LANDED

`core/smac_import.py`, one entry point taking `{path: parsed json}` and returning
a format-2 snapshot. Everything downstream is untouched. Two doors: a
`webkitdirectory` picker on the import page, and `manage.py import_ihpo --smac`.

Every edge case this plan predicted was real and is handled — repeated
configurations renumbered with the original key kept in `additional_info`,
RUNNING rows dropped, multi-objective refused by name, non-finite costs dropped.
One refinement: a *finite* crash cost is kept and marked failed, since that is a
real measurement of a configuration that does not work, and the failure marking
built for `plan_today.md` Step 1 then draws it correctly.

Three things this plan did not anticipate:

- **`model_fingerprint` re-derived `kind` from `model_path` alone**, so an export
  relabelled an imported run as a registry model it could not then resolve. It
  takes the kind from its caller now, and `_model_kind` is the single decision.
- **The objective name must be namespaced** (`smac:cost`). `serialize_result`
  writes a declaration only for a name the registry lacks, so an objective that
  happened to be called `accuracy` would lose its declaration on the first
  re-export and read back as a 0–1 higher-is-better score.
- **Django keeps only an upload's basename**, so a directory picker's relative
  paths do not survive the post. The importer matches on basenames, and two run
  directories at once are refused rather than blended.

**A pre-existing bug found here.** Every `.ihpo` ever written for a SMAC run
contains bare `Infinity` — four of them in `tests/fixtures/test.ihpo` — because
`optimizer_state` carries `scenario.json` verbatim and its `crash_cost` and
`walltime_limit` default to infinity. Stage 3 guarded `best_score` and
`incumbent_score`; this was the other leak in the same wall. `io.to_bytes` is now
the one place a snapshot becomes a file, and it sanitises. `default=` on
`json.dumps` cannot do this — it is never called for a float.

### What an import buys

Every figure in the catalog. Verified against a real 25-trial run: the
trial-based figures draw directly, and importance, the five interaction views,
partial dependence, local effects and local explanation fill in through the
**existing** `experiment_compute_analytics` endpoint. They need the config space
and the trials; neither the dataset nor the model.

Not runnable, and not resumable. A `Scenario` records nothing about what was
being optimized — no dataset name, no path, no digest — so a dataset attached
here could never be checked, which is exactly the case `import_experiment`
already refuses via `provenance.dataset_mismatch`.

---

## Outstanding

### Stage 5 — decide what is scoreable before the run, not during it

The blocker for registering anything new. `ui/views.py:404` has every new
experiment declare `list(METRICS)`, and `core/io.py`'s `_check_trial_scores` then
demands all of them on every trial — so registering `auc` would require an AUC
score on every trial of every new experiment. AUC also legitimately fails when a
validation fold happens to lack a class, and today *any* scoring exception fails
the whole trial.

Add `core/metrics.py::usable(metrics, splits, model) -> (usable, {name: reason})`,
called once from a shared helper in `base.py` before the loop. It checks the
model's capability and dry-runs each metric against each fold's own labels.
Reasons go into an additive `result["metric_notes"]`; `_check_trial_scores`
relaxes to **all-or-nothing per metric**, still refusing the partial-presence
case that is the actual KeyError-at-render bug. If the *primary* metric is
unusable the run is refused before any trial runs, with the reason. The metric
selector greys out a noted metric.

### Stage 6 — register `auc`

Six lines, because 1–5 did the work. `needs=PROBABILITIES`, `bounds=(0, 1)`,
`null_score=0.5` — a coin flip, because **AUC 0.0 is perfectly wrong**, which
would tell the surrogate that region is maximally anti-predictive. A bug the
naive design introduces even for a 0–1 higher-better metric.

### Stage 7 (the rest) — starting a run on such a metric

`surpass_target` rounds in the improving direction, and its absolute `10**-4`
nudge becomes magnitude-relative. Only reachable once a lower-better metric can
be *run*, which is stages 5, 6 and 8 below.

## Verification

Every stage leaves the suite green on its own. `python -m pytest -q` is ~20
minutes including the SMAC end-to-end runs.

**Do not edit the tree while a suite is running.** Twice in this work a
background run collected mid-edit and reported failures that did not exist —
`harness.py` is read from disk per subprocess while `client.py` is already
imported, so the two ends disagree and every subprocess test goes red.

The stage-2 signal is worth restating: `test_serialization.py` and
`test_metric_change.py` pin `cost == 1 - score` and must stay green **untouched**.
If they go red, the derivation is wrong — do not adjust the test.

New for stages 0–4: `tests/core/test_metric_orientation.py`, plus additions to
`tests/core/test_metrics.py` and `tests/core/test_modelhost.py`.

New for the import path:

- `tests/core/test_smac_import.py` — the importer against
  `tests/fixtures/smac_run`, a real 25-trial run committed for the purpose, plus
  hand-built runhistories for what a healthy run does not contain.
- `tests/core/test_game_orientation.py` — the explainer's training data is
  turned, `fit_surrogate`'s is not.
- `tests/ui/storage/test_stored_config_space.py` — the `space` section through
  the file, the row and the page.
- `tests/ui/runs/test_run_captures_config_space.py` — a run records the space it
  drew from.
- `tests/ui/results/test_metric_direction_on_the_page.py` — a lower-is-better
  unbounded metric on every figure, asserted against `accuracy` in parallel so
  the unchanged case is pinned too.
- `tests/ui/storage/test_smac_import_page.py` — upload, browse, compute, export,
  re-import.

Suite at 1295 passing when the import path landed, from 1184 before it.
