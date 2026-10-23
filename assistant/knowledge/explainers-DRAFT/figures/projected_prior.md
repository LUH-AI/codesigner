---
id: figure-projected-prior
title: Projected prior
kind: figure
about: [figure:projected_prior, cube-draw-prior, cube-clear-prior, pp-decay, pp-beta, pp-delay, pp-state, pp-discard, pp-stated]
related: [concept-priors, concept-projection-and-projected-priors, figure-configuration-cube, figure-prior, figure-acquisition-slice]
keywords: [projected prior, draw prior, belief in the projection, bump, ridge, slab, PCA, PLS, joint belief, two hyperparameters together]
status: draft
sources:
  - ui/figures/catalog.py:520-535 (ProjectedPrior)
  - git show main:ui/templates/ui/figures/projected_prior.html ([data-pp-strings], pp-delay)
  - git show main:ui/static/ui/projected_prior.js
  - git show main:core/projected_prior.py (module docstring, frame, density, profile)
  - ui/templates/ui/figures/configuration_cube.html (Draw Prior, Clear)
  - ui/templates/ui/experiment_detail.html:2514-2730 (drawing in the cube), 2491-2501 (weights)
  - ui/views.py:503-595 (projected_prior_preview, state_projected_prior, withdraw_projected_prior), 3588-3592 (can_draw_priors)
---

## What it shows

A belief drawn in the Hyperparameter space projection instead of stated on one hyperparameter: a bump around a point, a ridge along a line, or a slab around a plane, in whichever view the projection shows (HP Space, PCA or PLS). Because it is one function of a whole configuration, a ridge running diagonally across two hyperparameters can say "these go together", which no pair of single-hyperparameter priors can. It weights the acquisition, multiplied with any per-hyperparameter priors.

## How to read it

- **The card's plot** draws the belief in its own frame: a bump as weight against distance across; a ridge along "point 1 → point 2" against across; a slab over the plane through its points. The diamond is the peak.
- **In the projection** the draft is drawn in amber (points, line or plane, width, peak), and the acquisition layer is re-weighted by it.
- **Stated priors** are listed as "label · from trial N", N being the trial count when it was stated.

## Controls

1. In the projection tick **Draw Prior** (`cube-draw-prior`) and click: one point per click, up to as many as the view has dimensions (one point a bump, two a ridge, three a slab). **Clear** (`cube-clear-prior`) removes them. Points outside the search space are refused.
2. On the card, adjust each **Point**'s coordinates, and **Position** (where along the points it peaks: 0 at the first, 1 at the second), **Spread** (how far along it reaches) and **Width** (how far away from the points, in the view's units).
3. Choose **Decay** (`pp-decay`) and **β / N** (`pp-beta`), as on the Prior card; **Delay decay until after the initial design** (`pp-delay`) shows while the design is not over. The distribution is Normal only.
4. **State Prior** (`pp-state`) saves it; **Discard** (`pp-discard`) drops the draft. Each stated prior has **Edit** (parameters and decay; its points stay fixed, because a different shape is a different belief) and **Withdraw**.

<!-- REVIEW: the "Delay decay until after the initial design" box (pp-delay) is on branch main (commit a8c2cf0) but not yet on branch assistant; keep only if main is merged. -->

## When it appears

- **Acquisition** tab, beside the projection. Not per metric.
- Only once there are trials. Drawing needs an optimizer with a surrogate and edit rights.
- Switching the metric, method, dimensions or axes drops an unstated draft. A PCA/PLS projection is frozen into the stated prior, so it keeps its meaning as trials arrive.
- "Draw Prior in the space projection." when nothing is drawn or stated.
- Switched off with "Projected prior" in settings.

## Questions people ask

**How do I say two hyperparameters should rise together?** In HP Space with those two on the axes, place two points along the direction you believe in.

**Why can't I move a stated prior's points?** A different shape is a different belief; withdraw it and draw a new one.

**Does it combine with per-hyperparameter priors?** Yes, they are multiplied.
