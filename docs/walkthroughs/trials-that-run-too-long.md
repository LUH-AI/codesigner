# Trials that run too long

**Status:** implemented, awaiting your sign-off.
**You can now:** set how long any one trial may take, per run, and choose
between a fixed number of seconds and a deadline predicted from the
configuration itself. And a run now *survives* a timeout, which it did not
before.

Closes Step 2 of `docs/plan_today.md`.

---

## The thing the plan did not know about

The plan reads Step 2 as mostly exposure: the timeout already exists
(`DEFAULT_TRIAL_TIMEOUT`, wired from `settings.MODEL_TRIAL_TIMEOUT`), so the
work is offering it and adding a second mode.

It is enforced by killing the child that missed its deadline — `_await` does
that, and has to, since the whole point of a timeout is that the model is not
going to answer. What nothing accounted for is that **one child serves a whole
run**. So the kill ended the run as well as the trial: every later
`fit_predict` found a dead pipe, raised `ModelProcessError`, and was recorded by
`evaluate_trial`'s catch-all as a **crashed trial**. Confirmed by running it,
not by reading it — a model that sleeps on `k == 1` and answers instantly
otherwise:

```
FIRST  status: 3 (timeout)   the model did not answer within the time allowed
SECOND status: 2 (crashed)   scores 0.0 across every metric
```

The second trial never ran. It was a fabricated failure, scored zero, and
indistinguishable on the page from a configuration the model genuinely rejects —
which is exactly the confusion Step 1 built the ✕ marks to end.

Today that mostly hides behind `max_failures = 1`: the run stops at the timeout
and never reaches the fabrication. Anyone who raises that limit — which the
Step 1 help text explicitly invites, "raise it when a search is deliberately
probing a region you expect to break" — gets a run of invented crashes instead.
And prediction-relative deadlines are *designed* to fire more often than a
600-second catch-all, so shipping the setting on top of this would have made a
rare bug into a common one.

**So the child is now replaced.** `model_session._spawn` is the old `__enter__`
body, callable more than once; `RemoteModel._live_process` calls it when it
finds a dead process at the start of a trial. Lazily, so a timeout on the last
trial of a run — the common case, with `max_failures` at 1 — pays for nothing,
and the dataset is not rewritten, since `_arrays_dir` lives until `__exit__`. A
restart that itself fails ends the session for good rather than spending an
interpreter start per trial rediscovering it, and a cancelled run is refused a
replacement outright (`start()` has no cancel flag to read, so a child spawned
for a run nobody wants would be waited on for the full 120s start timeout).

Same probe, after:

```
FIRST  status: 3 (timeout)
SECOND status: 1 (success)   accuracy 1.0 — a real measurement
```

## The deadline as a policy

`core/modelhost/deadline.py` is new. The deadline is now asked per call rather
than fixed at the top of the run, and `trial_timeout` accepts either a bare
number (what every existing caller passes, unchanged) or a policy.

**`FixedDeadline`** is the old behaviour, named, plus a zero that means "no
limit" — matching every other numeric field on the run form, where an empty box
has never meant "stop immediately".

**`PredictedDeadline`** fits a `RandomForestRegressor` over the durations this
run has already measured and allows each configuration `factor` times what the
forest predicts *for that configuration*. Same estimator and same encoding as
`core.optimizers.base.fit_surrogate` — `Configuration.get_array()` — so the
duration model and the performance model see a configuration identically.

Prediction-relative and not median-relative, for the reason the plan gives:
duration tracks capacity, capacity tracks performance, so a multiple of the
typical trial systematically prunes the high-capacity end of the space. It kills
a trial for being big. A prediction conditioned on the configuration does not,
and there is a test that says so — a forest fed `k=1 → 1s` and `k=5 → 100s`
gives `k=5` more than ten times the deadline it gives `k=1`.

Three bounds, all of which earn their place:

- **`ceiling`** — the fixed number, still asked for, still the outer bound. It
  is what covers the forest's inability to extrapolate: a prediction never
  leaves the range of the targets it was trained on, so a configuration far
  larger than anything measured yet is predicted at roughly the slowest thing
  seen so far. Without a ceiling that would be a licence; with one it is a cap.
- **`floor`** (30s) — the fastest measured trials are inside a tenth of a
  second, and `2 × 0.05s` is a deadline shorter than the round trip carrying it.
- **`MIN_OBSERVATIONS`** (5) — until then the policy *is* its ceiling, so the
  earliest and least-informed trials get exactly the fixed deadline they would
  have had anyway.

**It only learns from calls that finished.** A timeout is censored — all it says
is "longer than what it was given" — and a crash died early rather than ran.
Training on either would teach the forest that the region is fast and tighten
the deadline over it, which is the one feedback loop worth refusing outright.

## Where it is set

The run form, in a third fieldset — not the settings page. `experiment_settings`
says outright that "how the search runs is not here", and `_posted_settings`
reads integers only; a deadline is neither a display preference nor an integer.
Its own fieldset rather than a third "give up if…" field, because the two above
it end the *run* and this one ends a *trial*: the run carries on with that trial
recorded as failed, and which of the two failure limits then fires is their
decision, not this one's.

Stored in a new `Run.trial_timeout` JSON column rather than in `Run.stopping`,
which is documented as "every stopping criterion, on equal footing" and would
have been wrong twice over — the collector filters `stopping` to
`STOPPING_CRITERIA` and would have dropped it silently, and a reader of the row
would have been told the run stops on something it does not.

Carried through the metric-change confirmation, like the criteria, since a
confirmation that dropped it would start a run under a deadline nobody chose.

## The limitation, asserted rather than commented

A registry model's `fit_predict` runs in the run's own thread and Python cannot
interrupt that, so **no deadline applies on the in-process path** and none can
without running every trial out of process. Per the plan, this is now documented
by a test rather than by a comment — `test_an_in_process_model_is_not_subject_to_any_deadline`
sleeps past a deadline and asserts it was not interrupted. The run form leaves
the fields out entirely for such a model rather than showing fields that would
be quietly ignored.

## A cancelled trial is no longer recorded at all

Found while reasoning about the `except` chain above, verified by running it,
and fixed on your call rather than left as a note.

`RemoteModel` raises `TrialCancelled` when the flag trips mid-trial —
cancellation reaches a *running* trial now, which it did not always. But nothing
caught it: `evaluate_trial`'s catch-all clause did, and recorded the attempt as
a **crashed trial scoring 0.0**. So a cancelled run left behind a trial that
never happened, at the worst score there is:

```
trials recorded: 3
  #1 score=0.5 status=1
  #2 score=0.5 status=1
  #3 score=0.0 status=2  err=TrialCancelled: the run was cancelled
metadata stopped_by: all_failing        <- it was interrupted, not failing
```

Three consequences, the third of which is the one that gives it away. The
phantom trial appeared in the table and on every figure, marked failed by
Step 1's own ✕. It pulled the surrogates towards a minimum nobody measured. And
it counted against `max_failures` — default 1 — so the collector latched
`all_failing` and the stored `.ihpo` said the run gave up on failures when in
fact somebody pressed cancel.

**The gate is `TrialCollector.record`, not the three loops.** `evaluate_trial`
marks the attempt with `CANCELLED` and `record` returns `None` without appending
it, without moving the incumbent, without touching `_since_improvement`,
`_trial_seconds` or either failure counter — and latches
`stopped_by = "cancelled"`, so `while not collector.done` ends the loop on the
collector's own terms rather than relying on each optimizer to re-read the flag.

Nothing in `RandomOptimizer`, `GridOptimizer` or `SmacOptimizer` changed. All
three already call `evaluate_trial` and then `record`, which is what made the
collector the right place. The one that looked like it would need its own line
is SMAC, which calls `smac.tell(...)` *before* `record` — but
`SmacOptimizer.serialize_result` already keeps only runhistory entries matching
a recorded trial, for exactly this hazard, in its own words: "a trial asked for
and not told because the run stopped ... a trial that never happened being
reported as one that scored nothing."

`CANCELLED` is a marker beside the status rather than a fourth `STATUS_` int, on
purpose: those are SMAC's `StatusType` values, `smac_optimizer` passes them
straight to `StatusType(...)`, and that enum has only
RUNNING/SUCCESS/CRASHED/TIMEOUT/MEMORYOUT. It is also outside `RUN_INFO_KEYS`,
so it cannot reach a file even if some future caller does record the trial.

Scope worth knowing: this only ever bit models running in their own process.
`TrialCancelled` comes only from `core.modelhost`, and an in-process model is
cancelled between trials, never during one.

`tests/core/test_cancelled_trials.py` (14) covers the gate directly and then
asserts all three optimizers agree — parametrized precisely because none of them
has code for it.

## Verification

```
python -m pytest -q                   1188 passed, 5 skipped (20:22)
python -m pytest -q -m "not slow"     1149 passed, 5 skipped, 39 deselected
python manage.py check                no issues
python manage.py makemigrations --check --dry-run    no changes
```

The whole suite rather than the usual `not slow` subset, because the collector
change below is in the path every optimizer takes — including the SMAC
end-to-end runs, which are the slow ones.

`ui/migrations/0018_run_trial_timeout.py` adds the column; every run that
predates it has `{}`, which `as_deadline` reads as the deployment default —
exactly what those runs already behaved as.

New: `tests/core/test_trial_deadline.py` (16),
`tests/ui/runs/test_trial_timeout.py` (12) and
`tests/core/test_cancelled_trials.py` (14).

One existing test changed: `test_what_is_left_out_is_about_the_reader_not_the_experiment`
audits every detail-page context key, so the two the run form opens on had to be
classified — `default_trial_timeout` as about this instance (it is this
deployment's `MODEL_TRIAL_TIMEOUT`), `default_trial_factor` as a built-in
constant.

## Deviations, and one thing deliberately left alone

- **The restart is not in the plan.** It is in this step because the setting is
  not safe without it, for the reasons at the top. It is also, incidentally, the
  fix for a model that dies mid-run for any other reason.
- **The prediction is the forest's mean, not an upper quantile.** The plan says
  "cap at k × prediction" and that is what this is. Per-tree variance is
  available at no extra fitting cost — Step 4 already plans to use it for the
  confidence field — and a deadline at, say, the 90th percentile over trees
  would be better calibrated than a mean times five. Worth doing when Step 4
  builds the variance machinery, rather than building it twice.
- **Nothing primes the forest from previous runs.** Only per-fold durations are
  predictable and only whole-trial durations are stored, so cross-run priming
  would mean dividing by the fold count and calling it a measurement. A run
  learns from itself, and with `MIN_OBSERVATIONS` at 5 most of a thirty-trial
  run is adaptive.
- **The deadline is per model fit, not per trial**, and the form now says so.
  It has always been enforced on one `fit_predict` round trip, so under
  cross-validation a 5-fold trial is given it five times over — which would have
  made a "trial" label wrong by a factor of `cv_folds` for most experiments
  here, cross-validation being the default. The forest is fitted on the same
  unit, so the prediction and the limit agree. "One fit per fold" is already how
  the new-experiment form describes it.
