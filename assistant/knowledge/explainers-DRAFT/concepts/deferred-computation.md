---
id: concept-deferred-computation
title: What is computed when
kind: concept
about: [field:autocompute_hypershap_baseline, field:autocompute_local_ablation, field:autocompute_surrogate_uncertainty, field:autocompute_partial_dependence, field:autocompute_acquisition_slice, field:autocompute_local_effects, field:autocompute_forecast, field:ice_max_curves, field:local_effects_max_trials]
related: [concept-explanation-games, concept-surrogate-and-acquisition, concept-priors, concept-permissions, page-experiment-settings, page-dashboard, figure-partial-dependence, figure-local-effects, figure-local-explanation]
keywords: [compute, compute button, autocompute, computed automatically, deferred, on request, cost, slow, compute explanations, during a run, live, partial result, analytics]
status: draft
sources:
  - ui/figures/base.py:58-137
  - ui/figures/catalog.py:76-96
  - ui/figures/catalog.py:242-274
  - ui/figures/catalog.py:298-310
  - experiments/services/settings.py:1-68
  - ui/forms.py:373-433
  - ui/templates/ui/_experiment_settings_fields.html
  - ui/templates/ui/_side_explanation.html
  - core/analytics.py:23-32
  - core/analytics.py:21-55
  - core/analytics.py:234-256
  - core/analytics.py:712-762
  - ui/views/priors.py:124-171
  - ui/views/priors.py:381-393
  - ui/views/common.py:160-205
  - ui/views/analytics.py:349-354
  - config/settings.py:91
---

## In short

Some analyses are computed once when a run finishes and stored with the result. Some are cheap enough to draw from the trials on every page load. The rest depend on a choice (which trial, which hyperparameter, which pair of axes) and are computed on request. Each on-request analysis has a "computed automatically" setting. With it off, the figure shows a "Compute" button, and opening the page costs nothing.

## In detail

### At the end of a run

The three HyperSHAP games, measured from the search space's default, are computed for every metric, with their interactions, and stored. The cost is 2^(number of hyperparameters) coalition evaluations per game per metric, so it grows with the width of the search, not its length. They are skipped in three cases:

- the run was cancelled;
- every figure that shows them is switched off;
- the cost would exceed the instance's budget (1,024 evaluations by default, `ANALYTICS_EAGER_MAX_COALITIONS`).

"Compute explanations" in the sidebar computes them later. It is refused while a run is in progress.

### Drawn from the trials alone

These cost almost nothing and are shipped with the page:

- Trial performance;
- Trial duration;
- Parallel coordinates;
- the Hyperparameter space projection (HP Space, PCA, PLS);
- the Trials table;
- the best and selected configuration panels;
- the stored importance and interaction figures.

### On request

| Analysis (setting label) | What it costs | Automatic by default |
|---|---|---|
| HyperSHAP against the incumbent or selected trial | the three games again, for one metric | yes |
| Acquisition and priors | a surrogate fit and a prediction over 8,192 sampled configurations; also feeds the projection's layers | yes |
| Local explanation (selected trial) | one ablation game | no |
| Partial dependence (PDP/ICE) | a forest fit and predictions for up to "Trials drawn on the partial-dependence figure" trials (default 100) × 20 grid points | no |
| Local effects across trials | one ablation game per trial, up to "Trials explained on the local-effects figure" (default 100) | no |
| Surrogate uncertainty (axes view) | a forest fit and a 20 × 20 grid for one pair of axes | no |
| Forecast | runs the model once per backtest | no |

Partial dependence [Fri01, GKBP15] and local effects sample trials evenly through the run, covering both exploration and exploitation. A cap of 0 means every trial.

Three things are never automatic and always wait for a button:

- **Next Trials**: rebuilds the optimizer and asks it. This takes a few seconds and is not cached.
- **Evaluate Prior**: a model fit plus local-search climbs.
- **Compute explanations**: described above.

The fitted surrogate and its sample are reused across the acquisition figure and the projection for the same experiment, metric and number of trials.

### During a run

While trials are arriving, the page polls every 5 to 30 seconds, depending on how long trials take. The figures drawn from the trials alone update. Importance, interactions, partial dependence and local effects stay as they were, because their analyses are not run per trial; they fill in when the run ends. The projection's layers are disabled while a run is adding trials.

### Settings

Each figure has a "show" setting under "Figures shown on an experiment page". Each on-request analysis has an autocompute setting under "Analytics computed as soon as a figure needs them".

- Figure off: nothing is computed for it.
- Figure on, autocompute on: computed as soon as the figure needs it.
- Figure on, autocompute off: a "Compute" button.

An experiment uses the instance's defaults until "Use default experiment settings" is unticked on its settings page.

## In Codesigner

- The experiment settings page, and the default settings page for whoever may change instance defaults.
- "Compute" buttons on the figures. For the HyperSHAP baseline the button sits in the sidebar's "HyperSHAP" section. "Show surrogate uncertainty" is on the projection.
- Opening a figure's analysis only needs view rights. "Compute explanations" and "Evaluate Prior" need compute rights.

## Questions people ask

**Why is a figure showing a Compute button?** Its analysis is not set to compute automatically for this experiment.

**Why are the importance figures empty after I cancelled?** A cancelled run skips them. Press "Compute explanations".

**Why don't the HyperSHAP figures update during a run?** They are computed when the run ends.

**How do I make the page open faster?** Turn off automatic computation, or unused figures.
