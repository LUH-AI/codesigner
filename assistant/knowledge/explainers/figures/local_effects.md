---
id: figure-local-effects
title: Local effects across trials
kind: figure
about: [figure:local_effects, element:local-effects-compute-btn, element:hypershap-baseline]
related: [figure-local-explanation, concept-explanation-games, figure-hyperparameter-importance, figure-selected-configuration, page-dashboard-sidebar]
keywords: [beeswarm, local effects, spread of effects, helped or hurt, ablation across trials, consistent effect]
status: draft
sources:
  - ui/figures/catalog.py:340-354 (LocalEffects)
  - ui/templates/ui/figures/local_effects.html
  - ui/figures/plots/surrogate.py:273-317 (local_effects_plot)
  - core/analytics/surrogate.py:132-141 (_sample_evenly)
  - core/analytics/effects.py:69-106 (compute_local_effects)
  - ui/views/analytics.py:213-232 (local_effects), 423-437 (_local_effects_data)
  - ui/templates/ui/experiment_detail.html:1501-1551 (refreshLocalEffects), 834-850, 3467-3475
  - ui/forms.py:374, 388-390 (local_effects_max_trials label)
  - experiments/services/settings.py (local_effects_max_trials 100, autocompute_local_effects off)
---

## What it shows

The local explanation of many trials at once, as a beeswarm: for each hyperparameter, one point per trial showing how much that trial's value of it helped or hurt. It shows the spread of each hyperparameter's effect, which a single average cannot.

## How to read it

- **One row per hyperparameter**, the widest spread at the top: a wide row is the interesting finding here.
- **One point per trial** in the sample, jittered up and down so overlapping points stay countable.
- **X axis**: "Effect vs. default (left: hurt, right: helped)", with a line at zero. **Blue** points helped, **red** points hurt. Right is always better, also for metrics where lower wins.
- A tight cluster: the hyperparameter behaves the same wherever the search is. Points fanning across zero: it helps in some regions and hurts in others.
- The selected trial's points are ringed in every row, if it is in the sample. Hover a point for its trial and value.

Each point is HyperSHAP's ablation game for that trial against the **Compare against** baseline, from one shared surrogate model.

<!-- REVIEW: the x-axis title says "vs. default" even when Compare against is Incumbent or Selected; same question as for the local explanation's labels. -->

## Controls

- **Compute** (`local-effects-compute-btn`), shown when it waits to be asked.
- Sidebar **Compare against** (`hypershap-baseline`) changes the baseline; under **Selected** it is recomputed whenever the selection changes.
- Click a point to select that trial on the whole page.

## When it appears

- **HyperSHAP** tab, two columns wide. Follows the metric selector.
- Deferred, and the most expensive figure on the page (one ablation game per sampled trial): "Local effects across trials" under "Analytics computed as soon as a figure needs them" is off by default, so a **Compute** button appears.
- How many trials it explains is the setting "Trials explained on the local-effects figure" (100 by default, 0 for all), sampled evenly from the start of the run to the end.
- Does not compute on its own while a run is adding trials.
- Messages: not enough trials, no search space, or (under Selected) a proposal is selected rather than a trial.
- Switched off with "Local effects across trials" in settings.

## Questions people ask

**Does this hyperparameter always help?** If its points sit on one side of zero, yes; if they straddle zero, it depends on where in the space you are.

**Why are fewer trials shown than I ran?** Only up to the setting's number are explained, spread evenly through the run.

**Why are some of my clicks not highlighted here?** The selected trial may not be in the sample.
