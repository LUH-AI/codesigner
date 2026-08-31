# Scoring that knows what it is measuring, and reading somebody else's run

**Status: stages 0–4 of 10 landed and verified. 5–10 outstanding.**

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

### Stage 7 — the UI for scores that are not 0–1

- error axis (`plots.py:251-252`): `hi - score` when bounded above; a
  lower-better metric already *is* an error; higher-better and unbounded has no
  definition, so return `None` and let the figure show its empty caption
- `absolute_scale` (`catalog.py:263-272`) becomes per-metric, `None` when
  unbounded so the toggle hides
- `target_score` bounds come from the metric; **stop silently dropping
  below-low values** (`views.py:146-147`) — typing `-1` today removes the
  criterion and the run is then refused with a misleading message
- the input `pattern` rejects a minus sign and exponent notation → `type="number"`
- `surpass_target` rounds in the improving direction; its absolute `10**-4`
  nudge becomes magnitude-relative
- delta arrow direction computed in Python; incumbent markers `>` → `!=`
- colour direction: `reversescale` for a lower-better metric, and reverse the
  worst-first z-order sort at `plots.py:1209`

### Stage 8 — register `rmse`/`logloss`; orient the HyperSHAP games

MAX/VAR/MIN mean tunability/sensitivity/mistunability only if higher is better.
**Orient the training data, not the games**: a separate
`_pair_trials_with_oriented_scores` on the explainer path only, so no stored
field changes meaning. `fit_surrogate` stays in raw units — PDP labels its axis
with the metric, and the uncertainty field's tree spread is orientation-free — so
these must be genuinely separate functions, not one with a flag.

Do **not** swap the `_GAME_PIECES` labels: `hyperparameter_mistunability` and
friends are in every `.ihpo`.

### Stage 9 — hoist the config space

An optional top-level `space` section in the format-2 snapshot, holding the
ConfigSpace serialized dict — byte-identical to `configspace.json` and exactly
what `from_serialized_dict` consumes. `_config_space_for` prefers it, falling
back to the model. Deep-copy before decoding: `from_serialized_dict` pops
`"type"` and destroys its input (`core/modelhost/client.py:296` is the pattern).

Additive, so no format bump. But fix one latent bug while here: `normalize()`
dispatches on exact `== SNAPSHOT_FORMAT`, so a future format 3 would be run
through the format-1 lifter and mangled rather than refused.

### Stage 10 — the importer

`core/smac_import.py`, one entry point taking `{relative path: parsed json}` and
returning a format-2 snapshot. Everything downstream is untouched — no second
internal representation. Upload via `<input webkitdirectory multiple>` on the
existing import page.

Edge cases that will bite:

- SMAC's `data` is keyed by `(config_id, instance, seed, budget)`, so **one
  config can appear many times**. Number trials sequentially in `data` order
  rather than reusing `config_id`, and keep the original key in
  `additional_info`.
- **RUNNING rows** (`status: 0`) carry `cost: 2147483647.0` — drop them.
- **`Infinity`/`NaN`**: SMAC writes bare JSON tokens. Sanitise on the way in; a
  crashed entry records its status and **no score**, since the importer has no
  dataset and so cannot compute a null score.
- **Multi-objective `cost` is a list** — refuse with a reason naming it, rather
  than silently importing the first element.
- `build_experiment` raises when `model.name` does not resolve and `model.path`
  is empty — hence a new `model.kind: "external"`, yielding `model = None` and
  `can_run = False`, the state an `.ihpo` imported without its dataset is already
  in.

---

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

New: `tests/core/test_metric_orientation.py`, plus additions to
`tests/core/test_metrics.py` and `tests/core/test_modelhost.py`.
