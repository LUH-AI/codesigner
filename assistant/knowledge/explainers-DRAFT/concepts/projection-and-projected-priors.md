---
id: concept-projection-and-projected-priors
title: Hyperparameter space projection and projected priors
kind: concept
about: [figure:configuration_cube, figure:projected_prior, field:view-select-configuration_cube, field:cube-dimensions, field:cube-axis-x, field:cube-axis-y, field:cube-axis-z, field:cube-layer, field:cube-layer-style, field:cube-draw-prior, field:pp-decay, field:pp-beta, field:pp-delay]
related: [concept-priors, concept-surrogate-and-acquisition, concept-initial-design, figure-configuration-cube, figure-projected-prior, figure-parallel-coordinates]
keywords: [projection, HP Space, configuration cube, PCA, PLS, principal components, partial least squares, one-hot, standardised, log scale, layer, prediction, acquisition, scalar field, point cloud, draw prior, projected prior, point, ridge, slab, position, spread, width]
status: draft
sources: [core/projection.py:1-483, ui/figures/catalog.py:330-397, ui/figures/catalog.py:520-535, ui/templates/ui/figures/configuration_cube.html, ui/templates/ui/figures/projected_prior.html, main:ui/templates/ui/figures/projected_prior.html, core/projected_prior.py:253-275, main:core/projected_prior.py:1-60, main:core/projected_prior.py:289-592, main:core/projected_prior.py:595-668, ui/services/projected.py:1-218, ui/views.py:459-596, ui/views.py:1786-1852, core/optimizers/base.py:1953-2081, core/optimizers/smac_optimizer.py:1000-1026, git show f621b00, git show a8c2cf0]
---

## In short

The "Hyperparameter space projection" figure shows where the search went, in one of three views. HP Space puts real hyperparameters on the axes. PCA and PLS project the whole space onto two or three directions. In any view you can draw a belief as a point, a line or a plane: a projected prior. It weighs whole configurations, so it can say "these two go together", which per-hyperparameter priors cannot.

## In detail

### The three views

- **HP Space** (`axes`): up to three hyperparameters you pick, with log axes for log-scaled ones. Positions are readable values, but distances mix unrelated units.
- **PCA** [JC16]: the directions the trials varied in most. Each axis gives its share of the variance, for example "PC 1 (48%)".
- **PLS** [WSE01]: the directions that move the metric most, fitted against the scores. The figure opens on PLS. Its axes shift as trials arrive.

For PCA and PLS, a log-scaled hyperparameter enters as log10, and a categorical or boolean one becomes one 0/1 column per value, so no order is invented between choices. Every column is then standardized to mean 0 and variance 1, so a plotted distance approximates a real one. Both need at least 3 trials and show at most min(3, trials − 1, columns) components. An outline marks where any legal configuration can land.

### Layers

A layer under the trials shows the surrogate's **Prediction** or the **Acquisition**, as a **Scalar Field** or a **Point Cloud**. It comes from an even sample of 8,192 configurations placed through the same projection (see concept-surrogate-and-acquisition). Acquisition is SMAC-only and weighted by every stated prior. Layers are disabled while a run adds trials.

### Projected priors

Tick **Draw Prior** and click to put down points, at most as many as the view has dimensions:

- one point: a bump;
- two points: a ridge along their line;
- three points: a slab around their plane.

The "Projected prior" card shapes the belief:

- **Position**: where along the points it peaks, 0 at the first point and 1 at the second. Default 0.5.
- **Spread**: how far along the points it reaches. Default 0.5.
- **Width**: how far it reaches away from them, in the view's units. Default a tenth of the view's extent.

Density, peak 1: exp(−½ [Σ((tⱼ − positionⱼ)/spreadⱼ)² + r²/width²]), where tⱼ is the place along each direction and r the distance off the line or plane.

It has a decay and "β / N" as on the Prior card (see concept-priors), and on branch `main` also "Delay decay until after the initial design" (`pp-delay`). **State Prior** saves it, with the projection frozen in, so later PLS changes do not move it. After that, its points are fixed; "Edit" changes parameters and decay, and "Withdraw" removes it.

**Combination.** All projected and per-hyperparameter priors are multiplied into one prior, so every belief must hold at once. Evaluate Prior judges them together.

**Sampling.** SMAC also draws candidates from the prior.

- On the current branch, 20 times as many configurations as needed are drawn and resampled by the prior's weight, which rarely reaches a narrow belief.
- On `main`, HP Space draws the Gaussian on its axes directly, cut to the space. PCA and PLS move random configurations toward places drawn from the Gaussian, then weigh them so the result follows the belief.

### Why it needs trials

Drawing uses the surrogate's sample of the space, which exists only once there are trials ("There is no sample of this space to draw on."). Draw Prior is offered only for SMAC, to someone who may edit. In HP Space only numeric axes can carry a prior, and points must lie inside the space.

## In Codesigner

- Acquisition tab, "Hyperparameter space projection": method (`view-select-configuration_cube`), "in" 1D/2D/3D (`cube-dimensions`), axes (`cube-axis-x/y/z`), "Layer" (`cube-layer`), "as" (`cube-layer-style`), "Next Trials"/"Requery", "Draw Prior" (`cube-draw-prior`) and "Clear".
- "Projected prior" card: "Normal" (the only shape), decay (`pp-decay`), "β / N" (`pp-beta`), "State Prior", "Discard", and the stated beliefs.

## Questions people ask

**PCA or PLS?** PCA shows where the search spread. PLS shows which directions changed the score.

**Why can't I draw a prior?** No trials yet, not SMAC, a categorical axis, no edit rights, or a run in progress.

**Does it replace my per-hyperparameter priors?** No, they are multiplied.

