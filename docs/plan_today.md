# Failed trials, visible; then timeouts; then live figures

## Context

A trial can already fail without ending a run, and none of it reaches the page.
`core/optimizers/trial.py::evaluate_trial` catches a model raising, a trial
exceeding its deadline, and predictions that cannot be scored; it scores the
trial 0.0 on every metric, sets `run_info["status"]` to `STATUS_CRASHED` or
`STATUS_TIMEOUT`, and puts the reason in `additional_info["error"]`. Those
fields are in `RUN_INFO_KEYS`, so they serialize into the `.ihpo` and come back
on deserialize. `MAX_CONSECUTIVE_FAILURES = 15` already ends a run of nothing
but failures, with `STOPPED_BY_ALL_FAILING`.

So this is mostly a **surfacing** job. Every figure treats a crashed trial as a
trial that scored zero, and the reader has no way to tell one from the other.

Decisions taken:

- **Failures keep scoring 0.0 everywhere.** No computation changes; only the
  display. Recorded risk: a cluster of crashes becomes a cluster of real-looking
  minima that tunability, PDP and the projections will explain as if measured —
  ten crashes near `max_depth=50` read as "max_depth matters enormously". The
  marks are what make that legible rather than invisible.
- **The threshold is configurable**, and so is the consecutive one — the comment
  claiming it should not be is overruled.
- **Tracebacks go in the result, and the export asks** whether they travel,
  alongside the timestamps question already there.
- **Still to gain is disabled** — the branch where you did that is not reachable
  from this clone (`FETCH_HEAD` is from July 23; `origin/main` is the only ref
  that resolves), so it is disabled here directly and can be rebased later.

---

## Step 0 — Turn off "still to gain"

**Not ready for use, computation included.** The presentation problems were the
visible part, but the numbers are not verified either, and there is direct
evidence against them: the same 40-trial seed-0 run produced tunability +0.096 in
one process and +0.070 in another, while being bit-identical within a process.
Whatever causes that also lands in the headroom subtraction, so "still to gain"
is quoting a difference between two quantities one of which moves between
processes for reasons nobody has found. The existing tests check the arithmetic
is self-consistent, which is not the same as the measure being sound.

So: a flag, not a revert, and the flag is "unverified" rather than "cosmetic".
`ui/views.py`'s `_tuning_progress` returns `([], False)` behind a module constant
(`TUNING_PROGRESS_ENABLED = False`), which empties `split`, `headroom` and `rows`
in the `trial_ablation` payload. `syncRemainingControl` then hides the checkbox
the same way it already hides it under the other two games, and the table's three
columns stay empty.

The code stays in place, unreferenced by the page, so it can be picked up when
the cross-process instability is understood — that instability is the thing worth
fixing before this feature is worth re-enabling, and it is not in this plan.

## Step 1 — Failed trials, visible

**`TrialResult.failed`** — a property on the dataclass in `core/optimizers/base.py`:
`run_info.get("status", STATUS_SUCCESS) != STATUS_SUCCESS`. Every builder already
takes the `result`, so nothing needs a new argument, and the templates can ask
the same question the figures do.

**The traceback.** `evaluate_trial` keeps `str(exc)`; it gains
`traceback.format_exc()` under `additional_info["traceback"]`. Same place, same
round trip, no new storage.

**The export asks.** `ui/templates/ui/export_confirm.html` grows a second
question beside the timestamps one — a traceback names absolute paths, package
versions and sometimes environment details, which is the same exposure the
timestamps question exists for. `experiment_export` reads both answers; the
buttons become "Export" plus two checkboxes rather than two buttons, since two
independent questions no longer fit one either/or.

**The marks**, in `ui/figures/plots.py`. Failures are a second trace per figure
rather than a restyle of the first, so the selection machinery is untouched —
`_selection_meta` keeps naming the succeeded trace, and a click on a failure
resolves through its own entry:

| Figure | Failures |
| --- | --- |
| Trial performance | thick red ✕ (`marker symbol="x"`, `NEGATIVE_COLOR`, `line.width` 3) |
| Hyperparameter space projection | thick red ✕, all three views |
| Trials table | row tinted `NEGATIVE_COLOR` at low alpha |
| Trial duration | bar in `NEGATIVE_COLOR` at low alpha |
| Parallel coordinates | omitted — a line to a fabricated zero is a line through the picture |
| Everything else | unchanged; they read the 0.0 as today |

Transparency on the two where the mark is a *fill* (row, bar) and full strength
on the two where it is a *symbol* (✕), which is what you asked for and is also
the right way round: a tint says "this row is different", a cross says "this
point is not a measurement".

**The link.** `ui/templates/ui/_selected_config_inner.html` gains, under the
configuration table and only when the selected trial failed, the one-line reason
and a link to the full traceback — a `trial_traceback` endpoint returning
`text/plain` from the stored result. Read-only, no new storage, and it 404s for a
trial that did not fail.

**The thresholds**, in `core/optimizers/base.py`'s collector:
`MAX_CONSECUTIVE_FAILURES` stops being a constant and both counts become
stopping criteria beside the six already there — `max_failures` (total, default
1) and `max_consecutive_failures` (default 15, its current value). Both go in
`STOPPING_CRITERIA`, both get a field in the run form's `<details>`, and
`stopped_by` gains a reason for the total one. The collector already tracks
`_consecutive_failures`; the total is one more counter in the same place.

## Step 2 — Timeouts

**What already exists, and its limit.** `core/modelhost/client.py` enforces
`DEFAULT_TRIAL_TIMEOUT = 600.0` per trial and raises `TrialTimeout`, which
`evaluate_trial` records as `STATUS_TIMEOUT`. It is wired from
`settings.MODEL_TRIAL_TIMEOUT` through `ui/services/modelenv.py`.

**But it only covers the subprocess path.** A registry model's `fit_predict`
runs in the run's own thread, and Python cannot interrupt that — no timeout can
be enforced there without running every trial out of process. Worth knowing
that doing so would also fix runs dying when `runserver` reloads.

**On the threshold itself** — you were right to be suspicious of a fixed number,
and the tempting alternative is worse:

- *Fixed wall clock.* Honest and predictable, and nobody knows the right value
  before their first run.
- *Relative to the other trials* (k × median). Adapts by itself, and is
  **biased**: duration tracks capacity — `n_estimators=500` is legitimately 50×
  `n_estimators=10` — and capacity tracks performance. A median-relative cap
  systematically prunes the high-capacity end of the space, which is often where
  the best configuration is. It kills a trial for being big.
- *Predicted from the configuration.* The principled one, and the data is
  already here: `duration` is recorded per trial and there is already a Trial
  duration figure. Fit the same RandomForest surrogate PDP uses against
  duration, and cap at k × prediction. A configuration predicted to take ten
  minutes gets ten minutes; one predicted at ten seconds and still running after
  five minutes is stuck.

So Step 2 is: expose the existing per-experiment timeout as a setting, and offer
prediction-relative as the second mode. Not median-relative.

## Step 3 — Figures during a run, and after an early stop

**There is no rendering constraint — there is a persistence one.** `execute_run`
writes `experiment.result` exactly once, at `ui/services/run.py:234`, after
`_optimize` returns. During a run there is genuinely nothing new to draw, and no
figure is being withheld.

Two costs to writing sooner, and they separate cleanly:

- Re-serializing the whole result per trial is quadratic in bytes over a run.
  Mitigated by writing every *k* trials rather than every one.
- The eager analytics are the expensive part (2^p coalitions per game per
  metric) and must not run per trial. But they don't have to: the
  **trial-based** figures — trial performance, trial duration, the trials table,
  the cube's axes view, parallel coordinates, the projection, best and selected
  configuration — need no surrogate at all. Only importance, interactions, PDP
  and local effects do.

So live updates are cheap for exactly the figures worth watching mid-run, which
includes all four that get failure marks. A partial result simply has its
`hyperparameter_*` fields empty, which is already the cancelled-run case, so the
page's tolerance for that exists.

**The channel is the database, not the process.** Under real huey the run is in a
worker process, so nothing in-process can hand the page anything; the partial
write is the same filtered `Experiment.objects.filter(pk=...).update(result=...)`
already used at the end, which is also what keeps a mid-run deletion from
resurrecting the experiment.

**The page side needs one small addition, not a new mechanism.** `poll.js` is a
fragment swapper — it replaces the polled element's HTML and re-wires it — so
`ui/templates/ui/_run_status.html` can carry the fresh payloads in a
`<script type="application/json">` block, requiring no new endpoint and no new
poll loop. What's missing is a way for the page to notice a swap happened:
`poll.js` gains a `CustomEvent` dispatched after `el.replaceWith(fresh)`, and
`experiment_detail.html` listens for it and redraws the trial-based figures
through the same `draw()` the metric switch already uses.

**And after an early stop.** `compute_hp_games` deliberately *skips* the
analytics when a run is cancelled (`_skipped_games`, reason "cancelled",
"Resuming recomputes them") — because cancelling used to still buy the full
analytics bill. So a cancelled run's importance and interactions are empty by
design, permanently. The fix that satisfies both: make them **fetchable on
demand** afterwards, exactly like the three deferred figures already are. Then a
cancelled run costs nothing until someone asks, and asking works.

## Step 4 — Surrogate confidence in hyperparameter space (design pass first)

Feasible, and needs its own pass rather than a sketch here.

- The analytics surrogate is a `RandomForestRegressor`; per-point variance across
  its trees is the uncertainty estimate, at no new fitting cost.
- For the **axes** view it is a 2-D grid over the two chosen hyperparameters with
  the rest at a reference — a PDP-shaped computation, so per axis pair and
  therefore deferred, like PDP is.
- For **PCA/PLS** it is genuinely harder: a grid in component space maps back to
  configuration space only through the pseudo-inverse, and the points it lands on
  may not be valid configurations. Possibly axes-only.
- **Not green.** `ACCENT_COLOR` means "the best-so-far" across every figure, and
  a field underneath the points is not that. A background field wants a pale
  single-hue ramp of its own, under the marks — the palette block in `plots.py`
  is the place to add it and say what it means.

## Step 5 — A fabricated failed trial, to look at

Last, and it needs no run: **copy an existing experiment and edit the copy's
stored result** so one trial reads as crashed. Then the page can be looked at
directly, with none of Step 1's code exercised by a real crash.

A one-off `manage.py shell` script (kept in `scratch/`, not committed):

```python
exp = Experiment.objects.get(identifier="…")
exp.pk, exp.identifier = None, generate_identifier()
exp.name = f"{exp.name} (fabricated failure)"
exp.save()                       # a copy, so the original is untouched

data = exp.result
entry = data["data"][7]          # some mid-run trial
entry["status"] = 2              # STATUS_CRASHED
entry["cost"] = …                # the zeroed score, in whatever orientation it stores
entry["additional_info"] = {
    "error": "ValueError: max_features must be in (0, n_features]",
    "traceback": "Traceback (most recent call last):\n  …",
}
exp.save(update_fields=["result"])
```

`pk = None` + a fresh `identifier` is the whole copy — the `FileField`s point at
the same stored dataset, which is right for a read-only look. Two things will
look odd and are expected: the copy has **no `Run` rows**, so the run history and
overhead boxes are empty, and its `created_at` is now. Every figure reads
`result`, so all of them draw.

Worth doing *before* the figure work rather than after: it makes the marks
reviewable in the browser while they're being written, and it is the cheapest
possible fixture for "does a zeroed score with a crashed status render sensibly".
The exact key names come from `RUN_INFO_KEYS` and `deserialize_result` — I'll read
them off the serializer rather than guess, and confirm the round trip by loading
the copy's page.

---

## Verification

```
python -m pytest -q -m "not slow"
python manage.py check
python manage.py makemigrations --check --dry-run   # Step 1 adds settings, which are JSON
```

Per step:

- **Step 0** — the checkbox is absent, the payload's `split`/`headroom`/`rows` are
  empty, and the page is otherwise byte-identical to the pre-feature figure.
- **Step 1** — a model that raises on a known configuration produces a trial with
  `failed` true, a stored traceback, and the run continuing; each of the four
  figures carries its failure trace and parallel coordinates does not; the export
  honours both answers independently; both thresholds stop a run and say which
  did. A fixture with one deliberately-crashing configuration is the thing worth
  committing, since every one of these reads from it.
- **Step 2** — a model that sleeps past its deadline is recorded
  `STATUS_TIMEOUT`, not `STATUS_CRASHED`; the in-process path is asserted *not*
  to enforce it, so the limitation is documented by a test rather than by a
  comment.
- **Step 3** — a partially-written result renders every trial-based figure; a
  cancelled run's analytics are empty on the page and non-empty after being
  asked for.

In the browser, starting with Step 5's fabricated copy since it costs nothing:
confirm the ✕s, the tinted row and the tinted bar all read as failures at a
glance and that parallel coordinates omits the trial; click the failed point and
read its traceback from the sidebar. Then against a real run: a crashing
configuration mid-run, confirming the marks arrive as it goes, and a cancel,
confirming every figure still draws.