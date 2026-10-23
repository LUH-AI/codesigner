---
id: figure-acquisition-slice
title: Acquisition function
kind: figure
about: [figure:acquisition_slice, acq-hp-select, acq-show-slice, acq-next-trials, acq-requery, acq-compute-btn]
related: [figure-prior, figure-configuration-cube, concept-priors, figure-partial-dependence, concept-metrics]
keywords: [acquisition function, expected improvement, EI, PI, what the optimizer tries next, next trials, requery, candidates, prior weighting, surrogate]
status: draft
sources:
  - ui/figures/catalog.py:460-497 (AcquisitionSlice)
  - ui/templates/ui/figures/acquisition_slice.html, _next_trials.html, _acquisition_shared.html
  - ui/figures/plots.py:1431-1591 (acquisition_slice_plots)
  - ui/static/ui/acquisition.js:203-440 (compute), 487-530 (priorTraces), 1193-1321 (candidates), 1765-1819 (refresh)
  - ui/static/ui/next_trials.js
  - ui/views.py:882-922 (_slice_model), 1263-1357 (_next_trials, next_trials), 1402-1549 (_slice_figures, acquisition_slice), 1647-1663 (_weighting_warning)
  - core/optimizers/base.py:1096-1143 (slice_surrogate, acquisition_function), 1948-2081 (sample_prediction_space), 2111-2289 (compute_incumbent_slice)
  - ui/services/settings.py (autocompute_acquisition_slice on by default)
---

## What it shows

How interesting the optimizer finds each value of one hyperparameter: its acquisition function (expected improvement, or probability of improvement if the run uses that), computed from the run's surrogate model and weighted by any stated priors. A prior does not change the surrogate's prediction; it multiplies the acquisition, so this curve is where a prior's effect shows.

## How to read it

- **X axis**: the chosen hyperparameter's values. **Y axis**: "Acquisition", scaled so the tallest curve reaches 1; only the ranking matters, not the numbers.
- **Blue filled curve** ("Best sampled, weighted"): for each value, the best acquisition among configurations sampled across the whole space, with the other hyperparameters free — what the search can actually reach there.
- **Blue dotted curve** ("Best sampled, unweighted"): the same without priors; appears only when a prior is stated. The gap between the two is what the prior did.
- With **Acq. w/ inc. HPs** ticked: a dashed "Acquisition" curve and an amber "Weighted by the prior" curve along a slice with every other hyperparameter held at the incumbent's values, with the incumbent marked.
- **Green diamonds along the top** ("Optimizer's Candidate"): what the optimizer would run next. Position only; height means nothing. A number on a diamond counts candidates merged at that spot; hover gives rank ("runs next" for rank 1), origin and configuration.

## Controls

- **Hyperparameter** (`acq-hp-select`), kept in step with the Prior card's picker.
- **Acq. w/ inc. HPs** (`acq-show-slice`): adds the incumbent slice.
- **Next Trials** (`acq-next-trials`) and **Requery** (`acq-requery`): ask the real optimizer, rebuilt from this run's settings and replayed over its trials under the saved priors, what it would run next. Takes a few seconds. Shared with the projection's pair; disabled while a run is adding trials.
- **Compute** (`acq-compute-btn`) when it waits to be asked.

## When it appears

- **Acquisition** tab, full width. Follows the metric selector.
- "Acquisition and priors" computes automatically by default; unticked, a **Compute** button appears (the prior is still drawn). It does not compute on its own during a run.
- Warnings: no search space; a SMAC without the acquisition weight layer (priors would be ignored); the optimizer cannot be asked (random or grid search fits no model).
- "No acquisition data available." when there is nothing to draw.
- Switched off with "Acquisition function" in settings.

## Questions people ask

**What will the optimizer try next?** Tick Next Trials; the rank-1 diamond runs next.

**Why is the next trial not at the curve's peak?** The search moves every hyperparameter at once; the sampled curve, not the incumbent slice, shows what it can reach.

**Did my prior change anything?** Compare the solid and dotted sampled curves.
