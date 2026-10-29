---
id: figure-configuration-cube
title: Hyperparameter space projection
kind: figure
about: [figure:configuration_cube, element:view-select-configuration_cube, element:cube-dimensions, element:cube-axis-x, element:cube-axis-y, element:cube-axis-z, element:cube-layer, element:cube-layer-style, element:cube-next-trials, element:cube-requery]
related: [concept-projection, figure-acquisition-slice, figure-parallel-coordinates, figure-selected-configuration, concept-priors]
keywords: [configuration cube, where the search went, PCA, PLS, projection, HP space, scatter, prediction layer, acquisition layer, point cloud, proposal]
status: draft
sources:
  - ui/figures/catalog.py:242-274 (ConfigurationCube)
  - ui/templates/ui/figures/configuration_cube.html
  - ui/figures/plots.py:608-697 (configuration_cube_plot, configuration_projection_plot)
  - core/projection.py:162-192, 340-349
  - core/modelhost/deadline.py:125-138
  - core/analytics.py:473-568 (sample_prediction_space)
  - ui/templates/ui/experiment_detail.html:1677-1684 (UNCERTAINTY_DISABLED), 1382-2069 (layer), 2505-2520 (candidates), 2668-3048 (applyCubeAxes, pickers), 3497-3525 (opening state)
  - ui/views/analytics.py:256-302 (surrogate_prediction)
  - ui/views/dashboard.py:392 (can_requery)
---

## What it shows

Every trial as a point in the search space, coloured by its score: where the search went and where it scored well. Three views: the hyperparameters' own axes (**HP Space**), or the whole space projected onto its most telling directions (**PCA**, **PLS**). A layer underneath can show the model's prediction or the optimizer's interest.

## How to read it

- **Points**: trials, light to dark blue, darker is better. Crosses are failed trials; the selected trial is redrawn in coral. Hover shows the trial's score and whole configuration.
- **HP Space**: positions are real hyperparameter values (log-scaled ones on log axes), but distances between points mix units and mean nothing.
- **PCA**: directions along which the trials varied most; labels give each component's share of variance. **PLS**: directions that move the metric most. Both standardise every hyperparameter first, so distances roughly mean something; each axis is a weighted mix of hyperparameters.
- **Layer**: **Prediction** is the surrogate's predicted score on the trials' colour scale; **Acquisition** is the optimizer's interest on its own orange-red scale, weighted by stated priors. Drawn from about 8,000 configurations sampled evenly. In PCA/PLS an outline marks where configurations can exist at all.

## Controls

- **Method** (`view-select-configuration_cube`): HP Space, PCA, PLS. Opens on PLS in 2D.
- PCA/PLS: **in** 1D / 2D / 3D (`cube-dimensions`).
- HP Space: three axis pickers (`cube-axis-x`, `-y`, `-z`); how many you fill sets 1D, 2D or 3D. 3D rotates by dragging.
- **Layer** (`cube-layer`): None, Prediction, Acquisition; **as** (`cube-layer-style`) Scalar Field or Point Cloud. Clicking a point-cloud sample selects it as a proposal you can run.
- **Next Trials** / **Requery** (`cube-next-trials`, `cube-requery`): what the optimizer would run next (filled diamond = next), with the local-search path that led there.

<!-- REVIEW: a "Show surrogate uncertainty" button and the setting "Surrogate uncertainty (axes view)" exist, but the page disables the feature (UNCERTAINTY_DISABLED = true in experiment_detail.html); left out. -->

## When it appears

- **Acquisition** tab, also pinned to **Overview**. Follows the metric selector.
- The trials and projections need no Compute. The layer opens on Acquisition, or None when "Acquisition and priors" is not computed automatically.
- During a run the points update; the layer is disabled.
- Projections need at least three trials, more than one hyperparameter and some variety. Next Trials appears only for optimizers with a surrogate.
- Switched off with "Hyperparameter space projection" in settings.

## Questions people ask

**Where did the search spend its time?** Dense clusters of points, best read in PCA or PLS.

**Which direction improves the score?** PLS's first axis.

**Why is HP Space empty?** Pick a hyperparameter for an axis.

**Why "No configurations to plot."?** PCA and PLS need three trials, two hyperparameters and some variety; try HP Space.

