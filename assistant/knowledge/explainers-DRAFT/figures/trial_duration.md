---
id: figure-trial-duration
title: Trial duration
kind: figure
about: [figure:trial_duration]
related: [figure-performance-over-time, figure-trials, figure-selected-configuration, page-dashboard-sidebar]
keywords: [how long each trial took, duration, seconds, timeout, slow trials, failed trials]
status: draft
sources:
  - ui/figures/catalog.py:563-575 (TrialDuration)
  - ui/figures/plots.py:769-794 (trial_duration_plot), 123-126 (FAILURE_FILL)
  - ui/templates/ui/figures/trial_duration.html
  - core/optimizers/base.py:75-78 (TrialResult.duration)
  - ui/templates/ui/experiment_detail.html:493-611 (applySelection), 3443-3493 (live updates)
  - config/figure_layout.toml (misc)
---

## What it shows

How long each trial's evaluation took, as one bar per trial. It is the same for every metric, so it is drawn once and does not change with the metric selector.

## How to read it

- **X axis**: trial number. **Y axis**: "Duration (s)", the wall-clock seconds the trial's evaluation took (0 when it was not recorded).
- **Blue bars**: ordinary trials. The selected trial's bar is drawn in the selection colour (coral).
- **Red-tinted bars**: trials that failed. Their length is still a real measurement — a failure really took that long — and often the telling one: a crash is usually short, and a trial stopped at its deadline is usually the longest bar.
- Under cross-validation a trial includes one model fit per fold, so its duration covers all of them.

Read it beside Trial performance: the "vs. time" view there uses the sum of these durations as its x axis.

## Controls

None of its own. Clicking a bar selects that trial on the whole page (Selected Configuration, the other figures, the Trials table).

## When it appears

- **Misc** tab by default.
- Not per metric; nothing to compute on request.
- Updates while a run is going, as trials arrive.
- Only shown once the experiment has trials.
- Switched off with "Trial duration" in the experiment's settings.

## Questions people ask

**Why is one trial so much longer?** Its configuration was more expensive to fit, or it ran until its deadline. If it is tinted red, it failed; click it to read the reason in Selected Configuration.

**Why are some bars red?** Those trials failed. The bar is tinted rather than removed because the time was still spent.

**Does this change with the metric?** No. Duration does not depend on which score you are viewing.

**Is this the total run time?** No. It is each trial's own evaluation time; time spent by the optimizer between trials is not included.
