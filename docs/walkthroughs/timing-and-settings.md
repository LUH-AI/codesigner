# Timing & settings (post-parity epic)

**Status:** implemented, awaiting your sign-off.
**You can now:** see how long each trial took and where a run's time went, judge
the marginal value of extra trials, and control per experiment (with a global
default) whether an exported `.ihpo` reveals *when* it was run.

Built after feature parity, in three commit groups (A/B/C).

---

## A. Timing foundation

**Concept.** SMAC's runhistory has nine per-trial fields
(`instance, seed, budget, time, cpu_time, status, starttime, endtime,
additional_info`), but in ask/tell mode SMAC can't time trials itself, so they
were all `0.0`/null. A shared helper now measures each trial and records the
full set for **every** optimizer.

- `core/optimizers/timing.py` — `timed_evaluation(seed, …)` context manager +
  `RUN_INFO_KEYS`. `time` = perf_counter duration, `cpu_time` = thread_time,
  `starttime`/`endtime` = POSIX timestamps, `status` = SUCCESS, and null
  `instance`/`budget` (single-instance, no multi-fidelity).
- `TrialResult` gains a `run_info` dict + a `duration` property.
  `TrialCollector.record(run_info=…)` stores it; base `serialize_result`
  splats it into each entry and `deserialize_result` collects it back (empty →
  no keys, so old `.ihpo` files round-trip unchanged).
- Every loop wraps `model.train_evaluate` in `timed_evaluation`. Grid/Random/
  Fake store it directly; SMAC also feeds it to `TrialValue` so its runhistory
  (copied verbatim by the serialize override) carries the real values.
- **Trials table** gains a per-trial **Duration** column.

**Checklist:** ✅ helper · ✅ run_info on TrialResult + duration · ✅ record/
serialize/deserialize · ✅ all four optimizers · ✅ Duration column · ✅ old
files still round-trip.

## B. Run summary + two figures

**Concept.** Surface the timing: how long a run took overall vs. how much was
spent evaluating trials, and which trials were worth their time.

- `Run` gains `trial_seconds` (Σ this run's trial durations) and a `duration`
  property; `execute_run` stores `trial_seconds` for the trials it *added*
  (resumed-from trials excluded). The detail page shows **"Last run finished in
  T s — X s in trials, O s overhead"** (overhead = total − trial time = the
  search/bookkeeping SMAC and Codesigner spent between evaluations).
- Two figures in `charts.py`: **Trial duration** (per-trial seconds,
  metric-independent, rendered once) and **Gain per unit time** (per-metric:
  each trial's incumbent improvement ÷ its duration — the marginal value of the
  trial; first/non-improving/zero-duration trials are 0). Placed in a timing
  row below the 2×2 analytics grid; gain switches with the metric selector.

**Checklist:** ✅ trial_seconds + duration · ✅ overhead note · ✅ duration
figure · ✅ gain-per-time figure (guards divide-by-zero) · 🔁 absolute
start/end times are recorded but **not shown** (per the decision — users care
about durations, not wall-clock).

## C. Settings system

**Concept.** A settings layer with a global default and per-experiment
overrides, so sharing an `.ihpo` need not leak *when* it ran.

- `Experiment` gains `settings` (overrides) + `use_default_settings`; a
  `GlobalSettings` singleton holds the default experiment settings.
  `ui/services/settings.py` `resolve_settings(exp)` resolves per key:
  experiment override → global default → built-in `SETTING_DEFAULTS`.
- First setting: **`export_absolute_times`** (default on). When off,
  `experiment_export` scrubs `starttime`/`endtime` from each trial entry (on a
  deep copy, so the detail page is unaffected; durations kept; re-import
  unaffected since deserialize ignores them).
- Pages: **⚙ Settings** on the experiment page → a per-experiment settings page
  (inherit via a checkbox / override / reset-to-default); a **global settings**
  link in the sidebar → a page linking to the **default experiment settings**
  subpage. `GlobalSettings` is registered in the admin.

**Checklist:** ✅ storage + singleton · ✅ resolve (override→global→builtin) ·
✅ export scrub + re-import safe · ✅ per-experiment page (inherit/override/
reset) · ✅ global + defaults pages · ✅ ⚙ + sidebar links · ✅ admin.

## What is now possible

- Read each trial's **duration** in the trials table and as a chart.
- See a run's **total time and overhead**, and each trial's **value per
  second** (gain-per-time) to judge diminishing returns.
- Set, per experiment or as a global default, whether an exported `.ihpo`
  **includes absolute timestamps**; turn it off to share results without
  revealing when they were produced.

## Verify

```bash
python -m pytest tests/timing tests/settings     # this epic
python -m pytest                                   # full suite (incl. slow SMAC)
python manage.py runserver
```

Automated: `tests/timing/` (helper, record, serialize round-trip, real Grid/
Random + slow SMAC timing, duration column, both figures, run summary) and
`tests/settings/` (resolve, export scrub + re-import, all pages). Live-checked:
the Duration column, both timing figures, the overhead note, all three settings
pages, and that export scrubbing strips `starttime`/`endtime` when the setting
is off.

Design notes: `run_info` mirrors SMAC's entry (so the SMAC override does nothing
and the base splat is trivial); `cpu_time` uses thread_time (undercounts BLAS
worker threads — acceptable, recorded not shown); figures/overhead show zeros
for pre-existing `.ihpo` files lacking timing and populate on new runs.
