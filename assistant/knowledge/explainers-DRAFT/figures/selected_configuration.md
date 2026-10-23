---
id: figure-selected-configuration
title: Selected Configuration
kind: figure
about: [figure:selected_configuration, selected-content-<metric>, export-model, export-trial-parameters]
related: [page-dashboard-sidebar, figure-best-configuration, figure-performance-over-time, figure-trials, figure-configuration-cube, concept-metrics]
keywords: [selected trial, click to select, incumbent, difference to best, failed trial, traceback, proposal, run as trial]
status: draft
sources:
  - ui/figures/catalog.py:35-45 (SelectedConfiguration, in_sidebar)
  - ui/templates/ui/figures/selected_configuration.html
  - ui/templates/ui/_selected_config_inner.html
  - ui/templates/ui/_selected_proposal_inner.html
  - ui/views.py:3277-3312 (_selected_panel_data), 616-676 (proposal_panel, run_proposal), 680-712 (trial_traceback)
  - ui/templates/ui/experiment_detail.html:437-444 (selectionFor), 730-831 (pointExportAt, selectTrial, selectProposal, dropProposal), 3295-3318 (show)
  - ui/figures/base.py:184-205 (selects_trials, in_sidebar)
---

## What it shows

The one trial that is currently selected on the dashboard: its trial number, its score on the metric you are viewing, how far that score is from the best trial's, and every hyperparameter value it ran with. Until you click something, the selected trial is the best trial for the viewed metric. It sits in the sidebar, not in a tab, so it stays visible while you click through the figures.

## How to read it

- **Caption**: "Trial N — click any point on the graph to select".
- **Big number**: the trial's score, with the metric name in brackets.
- **Line under it**: the difference to the best trial (score minus best score). It is shown even for the best trial itself, where it is 0 and marked "(Incumbent)". Its colour says whether the difference is in the good direction for this metric; for a lower-is-better metric a positive number is worse.
- **Table**: hyperparameter and value.
- **Failed trial**: a warning says the trial produced no measurement and the score is a placeholder, with the failure reason and a "Full traceback →" link when a traceback was stored.
- **Proposal**: clicking a sampled point in the Hyperparameter space projection's point-cloud layer shows "Proposal n of N" instead: the surrogate's predicted score ± spread, the difference to the incumbent, and **Run as Trial**, which runs it as a one-trial run (disabled, with the reason, when you cannot run now).

## Controls

None of its own. It changes when you click a trial in any figure that selects trials (Trial performance, Trial duration, Parallel coordinates, Hyperparameter space projection, Local effects across trials, the Trials table). The header's export buttons (`export-model`, `export-trial-parameters`) follow the same selection.

## When it appears

On the dashboard of any experiment with at least one trial, unless "Selected Configuration" is switched off in the experiment's settings. There is one panel per metric; switching the metric selector shows that metric's panel and resets the selection to that metric's best trial. A proposal is let go by clicking empty space on the page, which returns to the incumbent.

## Questions people ask

**Why did my selection jump back?** Switching the evaluation metric resets the selection to that metric's best trial.

**What does the number under the score mean?** The selected trial's score minus the best trial's score for this metric.

**Why is the score 0?** If a warning says the trial produced no measurement, it failed; 0 is the placeholder the search was given, not a result.

**Which figures depend on this selection?** Local explanation, Forecast, and, when "Compare against" is set to Selected, every HyperSHAP figure.
