# How a run is bounded, and how a trial is scored

**Status:** implemented, awaiting your sign-off.
**You can now:** change an experiment's optimized metric without corrupting its
search; end a run on a target score, a deadline, a compute budget or stagnation
rather than only a trial count; and score trials with k-fold cross-validation
instead of a single holdout.

Three commits: `576ef40`, `b22752d`, `f72b841`. They came out of planning the
A3S integration — two are gaps that integration will need, one is a live
correctness bug — and A3S is deferred until after them.

A theme runs through all three: **an experiment's evaluation must be fixed for
its whole life.** Changing the metric mid-experiment broke that, which is fix 1;
it is also why cross-validation lands on the `Experiment` at creation rather
than in the mutable settings layer.

---

## 1. A changed metric no longer poisons the optimizer

### What was wrong

Two faults, one of them silent and expensive.

`smac_optimizer.optimize` resumed by reusing the stored SMAC output directory
with `overwrite=False`, so SMAC **reloaded the previous runhistory** — whose
costs were `1 - old_metric` — and then received new trials told as
`1 - new_metric`. One Gaussian process, fitted across two different cost
functions. Every suggestion after that was steered by an objective that does not
exist. Quieter and affecting all four optimizers: `TrialCollector` was seeded
with the *old* metric's best, so the incumbent trajectory the figure draws was
for an objective nobody was optimizing.

And `decide_run` warned **once**, on the first divergence, then ran with the
current primary metric whatever the form asked for — so a second change was
silently discarded. That made everything above unreachable after the first
switch.

### What it does now

Nothing is thrown away, because **every trial records a score for every metric**.
`rebase_history` re-reads the accumulated trials under the new metric: each
trial's score becomes its score for that metric, and the incumbent trajectory is
recomputed from those. The trials themselves are untouched — the same
configurations were evaluated on the same data.

SMAC additionally discards its stored state and **replays** the history into a
fresh facade with recomputed costs. Verified against the local SMAC3 checkout
that `tell` takes a plain `TrialInfo` and does not require one from `ask()`
(`smac/facade/abstract_facade.py:292`). The replay also fixes an older gap: a
resume whose stored directory had gone used to start blind while the page still
showed the history.

The stored metric was already available — `OptimizationResult.primary_metric` is
serialized and deserialized — so the `cost_metric` field the plan called for
turned out to be unnecessary.

Every change is confirmed now, not just the first, and "keep" means the metric
you are on rather than the original. The confirmation says what is actually
true: the search continues from the accumulated trials rather than restarting,
but they were *selected* under the old objective, so the run inherits that bias.

**Proof it catches the bug:** with the discard disabled, the regression test
fails with `0.033333333` (`1 − accuracy`) sitting where `0.033416876`
(`1 − f1`) belongs.

## 2. A run can stop on more than a trial count

Five criteria. Any of them can be set at once, the first to fire ends the run,
and the run records which.

| | |
|---|---|
| `n_trials` | The trial cap. **Always present** — a target that is never reached has to end somewhere. |
| `target_score` | The incumbent reached what you asked for. |
| `max_seconds` | Wall-clock for the run. |
| `max_trial_seconds` | Time spent inside trials — the compute actually consumed, which is what a shared machine budgets. |
| `no_improvement_trials` | The score stopped improving. |

All of it lives in `TrialCollector.done`, the condition every optimizer loop is
already written against, so all four got it without knowing. `Run.stopping`
holds the optional four and `Run.n_trials` stays as the always-present one, so
no criterion has two homes. `Run.stopped_by` records the answer and the page
says *stopped early: the target score was reached* rather than leaving you to
work out why there are 12 trials and not 30.

The reason is **latched** — `done` is read once per loop iteration and the wall
clock keeps moving, so a run that finished its trials must not be relabelled a
timeout on the next read.

On **confidence bound on the incumbent**, which you asked for: a real
statistical criterion needs a variance estimate the single-holdout protocol
could not supply. Stagnation is the honest proxy. Cross-validation now makes the
real thing possible — see below.

Measured end to end on iris, 25-trial cap:

```
trial cap only   trials=25  stopped_by='n_trials'
target 0.5       trials= 1  stopped_by='target_score'
stagnation 2     trials= 3  stopped_by='no_improvement_trials'
budget 0.001s    trials= 1  stopped_by='max_trial_seconds'
```

## 3. Cross-validation

k-fold in codesigner: one metric optimized, the rest tracked, the fold scores
averaged.

### One mechanism, not two

Holdout and k-fold are both **a list of (train, val) index pairs**
(`core/splits.py`). Everything downstream then has one shape to handle: the
trial loop iterates folds, the wire protocol sends folds, and a holdout is the
case where there is one. A boolean and two code paths would have had to be
threaded through the optimizers, the subprocess protocol and the harness twice
over.

A trial still yields **one score per metric**, because the averaging happens
inside `evaluate_trial`. No figure, no resume path and no `.ihpo` knows the
difference. A fold that fails fails the trial — a configuration that works on
four fifths of the data and dies on the rest is not a partial success to hand
the search.

### Where the setting lives

On the **`Experiment`**, chosen at creation and fixed thereafter. Trials scored
different ways are not comparable, and an experiment's own history has to be —
the same fault fix 1 exists to prevent. That also sidesteps `SETTING_DEFAULTS`
being all-boolean. Absent on an older `.ihpo` means holdout, so every existing
file round-trips unchanged.

### Protocol 2

The child now receives the **whole feature matrix once** plus the fold
divisions, and a trial names a fold. The dataset crossing once instead of once
per fold per trial is the point.

`RemoteModel.use_fold(index)` is set before the call rather than passed to it,
because `fit_predict` is the contract user models implement and must not grow an
argument no user model will ever use. `evaluate_trial` asks for the method with
`getattr` rather than requiring it, so a minimal local model still works.

### The honest consequence

With one fold the child is never sent a validation label at all — the strong
statement holds exactly. With k folds every row trains in k−1 of them, so across
one trial the union of what the child receives is **every** label. It is never
told which rows it is about to be scored on, which is what stops an accidental
leak, but a model deliberately caching what it was sent could reconstruct them.
Closing it means one process per fold — k times the memory for the whole run —
and that is not a trade worth making by default. Written down in
`core/splits.py`, `core/modelhost/protocol.py` and the README rather than
glossed.

### Why it was worth doing

The same experiment on iris, four SMAC trials:

```
holdout   trials=4  best=1.0000  wall=2.9s
5-fold    trials=4  best=0.9467  wall=8.9s
```

The holdout calls the best configuration perfect. It is not.

---

## Verify

```bash
python -m pytest -m "not slow and not uv"   # 528
python -m pytest -m slow                    # 10
python -m pytest -m uv                      # 3, needs uv
python manage.py migrate                    # 0010, 0011
```

New tests: `tests/core/test_metric_change.py` (11), `test_stopping.py` (16),
`test_cross_validation.py` (15), plus four cross-validation cases across the
pipe in `test_modelhost.py` and four view-level ones in `test_run_views.py`.

Live-checked: all four stopping criteria ending a real run with the right
`stopped_by`; a holdout and a 5-fold run of the same experiment agreeing on
shape and disagreeing on score; the `.ihpo` round trip preserving `cv_folds`;
and the create form rendering the four scoring choices.

One fixture fixed on the way through: `tests/ui/model_envs/conftest.py`
hardcoded `"protocol": 1` in its fake-uv greeting, so the bump would have shown
up as a wall of unrelated environment failures. It now interpolates
`PROTOCOL_VERSION`.

## What this opens up

- **A real incumbent-confidence stopping criterion.** The per-fold scores exist
  now, so the variance does too. It would need `TrialResult` to keep the fold
  scores rather than only their mean.
- **The A3S integration**, which is where all three of these came from. Their
  generated target function cross-validates 5 ways; codesigner can now match
  that rather than measuring an agent-designed pipeline differently from
  everything else on the instance.
