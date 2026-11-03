---
id: concept-projection
title: Hyperparameter space projection
kind: concept
about: [figure:configuration_cube, element:view-select-configuration_cube, element:cube-dimensions, element:cube-axis-x, element:cube-axis-y, element:cube-axis-z, element:cube-layer, element:cube-layer-style]
related: [concept-priors, concept-surrogate-and-acquisition, figure-configuration-cube, figure-parallel-coordinates]
keywords: [projection, HP Space, configuration cube, PCA, PLS, principal components, partial least squares, one-hot, standardised, log scale, layer, prediction, acquisition, scalar field, point cloud]
status: draft
sources:
  - core/projection.py
  - ui/figures/catalog.py (ConfigurationCube)
  - ui/templates/ui/figures/configuration_cube.html
  - core/analytics/__init__.py
---

## In short

The "Hyperparameter space projection" figure shows where the search went, in one of three views. HP Space puts real hyperparameters on the axes. PCA and PLS project the whole space onto two or three directions, so configurations that are alike land close together.

## In detail

### The three views

- **HP Space** (`axes`): up to three hyperparameters you pick, with log axes for log-scaled ones. Positions are readable values, but distances mix unrelated units.
- **PCA** [JC16]: the directions the trials varied in most. Each axis gives its share of the variance, for example "PC 1 (48%)".
- **PLS** [WSE01]: the directions that move the metric most, fitted against the scores. The figure opens on PLS. Its axes shift as trials arrive.

For PCA and PLS, a log-scaled hyperparameter enters as log10, and a categorical or boolean one becomes one 0/1 column per value, so no order is invented between choices. Every column is then standardized to mean 0 and variance 1, so a plotted distance approximates a real one. Both need at least 3 trials and show at most min(3, trials − 1, columns) components. An outline marks where any legal configuration can land.

### Layers

A layer under the trials shows the surrogate's **Prediction** or the **Acquisition**, as a **Scalar Field** or a **Point Cloud**. It comes from an even sample of 8,192 configurations placed through the same projection (see concept-surrogate-and-acquisition). Acquisition is SMAC-only and weighted by every stated prior. Layers are disabled while a run adds trials.

## In Codesigner

- Acquisition tab, "Hyperparameter space projection": method (`view-select-configuration_cube`), "in" 1D/2D/3D (`cube-dimensions`), axes (`cube-axis-x/y/z`), "Layer" (`cube-layer`), "as" (`cube-layer-style`), and "Next Trials"/"Requery".
- Priors are stated one hyperparameter at a time on the Prior card (see concept-priors); the Acquisition layer shows how they weigh the whole space.

## Questions people ask

**PCA or PLS?** PCA shows where the search spread. PLS shows which directions changed the score.

**Why do the PLS axes move?** PLS is refitted to the scores as trials arrive, so the directions that matter most can change.

**Can I state a prior in this figure?** No. Priors are stated per hyperparameter on the Prior card; this figure shows their effect through the Acquisition layer.
