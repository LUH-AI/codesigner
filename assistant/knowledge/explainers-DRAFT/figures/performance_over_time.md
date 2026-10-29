---
id: figure-performance-over-time
title: Trial performance
kind: figure
about: [figure:performance_over_time, element:view-select-performance_over_time-y, element:view-select-performance_over_time-x]
related: [concept-metrics, figure-selected-configuration, figure-trial-duration, figure-best-configuration, page-dashboard-sidebar]
keywords: [score over time, running best, incumbent, convergence, error over time, trials vs time, absolute scale, log scale]
status: draft
sources:
  - ui/figures/catalog.py:183-239 (PerformanceOverTime, absolute_scale_for)
  - ui/figures/plots/performance.py:22-109 (performance_over_time_plot)
  - ui/figures/plots/style.py:1-45 (palette), 25-97 (failure marks)
  - ui/templates/ui/figures/performance_over_time.html
  - ui/templates/ui/experiment_detail.html:304-411 (scale toggle), 3476-3484 (view selects), 3165-3250 (live updates)
  - core/metrics.py:371-390 (to_error, error_range)
  - ui/figure_layout.toml (overview, misc)
---

## What it shows

Every trial's outcome on the viewed metric, in the order the trials ran, with a step line for the best score found so far (the incumbent). It is the chart for "is the search still improving?", and the one the other figures are read against.

## How to read it

- **Points** (blue, faded): one per trial. The selected trial is drawn larger in the selection colour (coral).
- **Green step line** ("Incumbent"): the running best. It only moves when a trial improves on it, in the metric's own direction (up for accuracy, down for an error).
- **Green diamonds** ("New incumbent"): the trials that set a new best.
- **Red crosses**: trials that failed. Their value is a placeholder, not a measurement.
- A long flat stretch of the green line means recent trials have not beaten the best.

Common misreading: in the "vs. time" view the x axis is the summed duration of the trials, not wall-clock time, so pauses between runs do not appear.

## Controls

Two selects in the card head, read as "score vs. trials":

- **Y axis** (`view-select-performance_over_time-y`): **score** — the metric's own value; **error** — distance from a perfect score (for a lower-is-better metric, the score itself). The error axis is logarithmic when every error is above zero, with values floored at 0.001.
- **X axis** (`view-select-performance_over_time-x`): **vs. trials** — trial number; **vs. time** — cumulative trial duration in seconds.

A toolbar button toggles **Absolute scale** (pins the y axis to the metric's full range) and **Relative scale** (fits the data). It is hidden for a metric with no fixed range. Clicking a point selects that trial everywhere on the page.

## When it appears

- On **Overview** and **Misc** by default, full width.
- Follows the metric selector. Needs no Compute button.
- Updates during a run as trials arrive.
- Empty ("No data available.") with no trials, or in the error view for a higher-is-better metric with no upper bound, which has no "perfect" score to measure from.
- Switched off with "Trial performance" in the experiment's settings.

## Questions people ask

**Has the search converged?** If the green line has been flat for many trials, recent trials are not improving on the best.

**Why is the error view empty?** The metric has no upper bound, so there is no distance from perfect to plot. Use the score view.

**What are the red crosses?** Failed trials. Click one to see its failure reason in Selected Configuration.

**Why is there no absolute-scale button?** The metric is unbounded, so there is no full range to pin to.
