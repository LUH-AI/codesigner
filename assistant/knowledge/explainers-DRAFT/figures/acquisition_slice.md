---
id: figure-acquisition-slice
title: Acquisition function
kind: figure
about: [figure:acquisition_slice, element:acq-hp-select, element:acq-show-slice, element:acq-next-trials, element:acq-requery, element:acq-compute-btn]
related: [figure-prior, figure-configuration-cube, concept-priors, figure-partial-dependence, concept-metrics]
keywords: [acquisition function, expected improvement, EI, PI, what the optimizer tries next, next trials, requery, candidates, prior weighting, surrogate]
status: draft
sources:
  - ui/figures/catalog.py:313-325 (AcquisitionSlice)
  - ui/templates/ui/figures/acquisition_slice.html, _next_trials.html, _acquisition_shared.html
  - ui/figures/plots.py:964-1151 (acquisition_slice_plots)
  - ui/static/ui/acquisition.js:207-444 (compute), 475-518 (priorTraces), 1178-1306 (candidates), 1750-1804 (refresh)
  - ui/static/ui/next_trials.js
  - ui/views/priors.py:136-167 (_slice_model), 319-393 (_next_trials, next_trials), 401-509 (_slice_figures, acquisition_slice), 512-524 (_weighting_warning)
  - core/optimizers/base.py:544-575 (slice_surrogate, acquisition_function)
  - core/analytics.py:473-568 (sample_prediction_space), 590-697 (compute_incumbent_slice)
  - experiments/services/settings.py (autocompute_acquisition_slice on by default)
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
