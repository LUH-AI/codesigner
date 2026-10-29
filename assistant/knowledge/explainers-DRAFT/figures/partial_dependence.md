---
id: figure-partial-dependence
title: Partial dependence (PDP/ICE)
kind: figure
about: [figure:partial_dependence, element:pdp-hp-select, element:pdp-compute-btn]
related: [figure-hyperparameter-importance, figure-acquisition-slice, figure-parallel-coordinates, concept-metrics, page-dashboard-sidebar]
keywords: [partial dependence, PDP, ICE, individual conditional expectation, effect of one hyperparameter, predicted score, surrogate]
status: draft
sources:
  - ui/figures/catalog.py:298-310 (PartialDependence)
  - ui/templates/ui/figures/partial_dependence.html
  - ui/figures/plots/surrogate.py:13-42 (partial_dependence_plot)
  - core/analytics/surrogate.py:49-70 (fit_surrogate), 101-115 (_hp_grid), 132-141 (_sample_evenly)
  - core/analytics/effects.py:109-165 (compute_partial_dependence)
  - ui/views/analytics.py:109-127 (partial_dependence), 406-420 (_partial_dependence_data)
  - ui/views/dashboard.py:440-451 (top_hp)
  - ui/templates/ui/experiment_detail.html:1443-1494, 1609-1614, 3117-3121
  - ui/forms.py:373, 385-387 (ice_max_curves label)
  - experiments/services/settings.py (ice_max_curves 100, autocompute_partial_dependence off)
---

## What it shows

How the predicted score changes as one hyperparameter changes, according to a model (a random-forest surrogate) fitted to the trials. The bold line is the average effect; the thin lines show the same effect for each trial separately.

## How to read it

- **X axis**: values of the chosen hyperparameter, up to 20 points spread evenly on its own scale (logarithmically for a log-scaled one, whole numbers for an integer one), or every choice for a categorical one.
- **Y axis**: "Predicted score" for the viewed metric.
- **Thin blue lines** ("Individual trials (ICE)"): for one trial, its other hyperparameters held fixed and only this one varied.
- **Bold green line** ("Partial dependence"): the average of the thin lines at each value.
- Thin lines that move together: this hyperparameter acts the same whatever the others are. Lines that fan out somewhere: it interacts with other hyperparameters there.

The average is taken over the trials the search actually ran, not over the whole search space, so it describes what the optimizer saw. It is a model's prediction, not measured scores.

## Controls

- **Hyperparameter** (`pdp-hp-select`): which hyperparameter is on the x axis. Opens on the viewed metric's most important hyperparameter by tunability, and resets to it when the metric changes.
- **Compute** (`pdp-compute-btn`): shown when the figure waits to be asked.

## When it appears

- **Misc** tab by default, full width. Follows the metric selector.
- Deferred: "Partial dependence (PDP/ICE)" under "Analytics computed as soon as a figure needs them" is off by default, so a **Compute** button appears; each new hyperparameter or metric asks again, unless already computed.
- How many trials get a thin line is the setting "Trials drawn on the partial-dependence figure" (100 by default, 0 for all), sampled evenly through the run. It caps the work too: the average is over those trials.
- Does not compute on its own while a run is adding trials.
- Warnings explain an empty figure: too few trials to fit a surrogate, no valid configuration on the grid, or no search space available.
- Switched off with "Partial dependence (PDP/ICE)" in settings.

## Questions people ask

**What value of this hyperparameter is best?** Where the bold line peaks (or bottoms out, for a lower-is-better metric), as far as the surrogate can tell.

**Why do the thin lines spread apart?** The effect depends on the other hyperparameters' values there: an interaction.

**Why must I press Compute?** It fits a model and predicts across a grid; it waits to be asked unless its autocompute setting is on.
