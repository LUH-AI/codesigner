---
id: figure-local-explanation
title: Local explanation (selected trial)
kind: figure
about: [figure:local_explanation, ablation-compute-btn, hypershap-baseline]
related: [concept-explanation-games, figure-local-effects, figure-hyperparameter-importance, figure-selected-configuration, page-dashboard-sidebar]
keywords: [local explanation, ablation, waterfall, why did this trial score well, helped, hurt, HyperSHAP]
status: draft
sources:
  - ui/figures/catalog.py:136-158 (LocalExplanation)
  - ui/templates/ui/figures/local_explanation.html
  - ui/figures/plots.py:500-543 (hyperparameter_ablation_plot)
  - ui/views.py:715-770 (trial_ablation), 2980-3015 (_local_ablation_data)
  - core/optimizers/base.py:775-804 (_pair_trials_with_oriented_scores), 1587-1664 (compute_hp_ablation, _ablate)
  - ui/templates/ui/experiment_detail.html:1198-1261 (refreshLocalAblation), 835-851 (selectionMoved), 3710-3718 (baseline handler)
  - ui/services/settings.py (autocompute_local_ablation off by default)
---

## What it shows

Why one trial scored as it did: its score built up from a baseline configuration one hyperparameter at a time, using HyperSHAP's ablation game. Unlike the importance figure, which explains the whole search, this explains the trial selected in the sidebar.

## How to read it

A waterfall, read left to right as a walk:

- **First bar** ("Default"): the starting point, at 0.
- **One step per hyperparameter**, largest effect first: switching that hyperparameter from the baseline's value to this trial's value. **Blue, up**: this trial's value helped. **Red, down**: it hurt. Each step is labelled with its signed size.
- **Last bar** ("This trial", grey): the total — how far this trial got from the baseline.
- **Y axis**: "vs. default", relative to the baseline, not an absolute score.

Up always means better, also for metrics where lower wins. The effects come from a surrogate model fitted to the trials, so they are estimates.

## Controls

- No controls on the card besides **Compute** (`ablation-compute-btn`), shown when it waits to be asked.
- It follows the **selected trial**: click a trial in any figure or the Trials table.
- It follows the sidebar's **Compare against** (`hypershap-baseline`): **Default** measures from the search space's default configuration; **Incumbent** and **Selected** measure from that trial instead.

<!-- REVIEW: the waterfall's first bar and y axis are always labelled "Default" / "vs. default" (plots.py hyperparameter_ablation_plot), even when Compare against is Incumbent or Selected. Check whether the label should follow the baseline. -->

## When it appears

- **HyperSHAP** tab. Follows the metric selector (the selection resets to that metric's best trial).
- Deferred: "Local explanation (selected trial)" under "Analytics computed as soon as a figure needs them" is off by default, so a **Compute** button appears for each newly selected trial not already computed.
- "This trial is the baseline." when the selected trial is the one being compared against — for example the page's opening state, where Compare against is Incumbent and the selected trial is the incumbent.
- Other messages: not enough trials, HyperSHAP failed, no search space, or a proposal rather than a trial is selected.
- Does not compute on its own while a run is adding trials.
- Switched off with "Local explanation (selected trial)" in settings.

## Questions people ask

**Why did this trial do well?** The tallest blue steps are the hyperparameter values that carried it furthest from the baseline.

**Why does it say "This trial is the baseline"?** You are comparing a trial with itself. Select another trial, or set Compare against to Default.

**Is the last bar the trial's score?** No, it is the difference from the baseline configuration.

**How is this different from Local effects across trials?** This explains one trial; Local effects shows the same kind of explanation for many trials at once.
