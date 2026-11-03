---
id: figure-parallel-coordinates
title: Parallel coordinates
kind: figure
about: [figure:parallel_coordinates]
related: [figure-hyperparameter-importance, figure-configuration-cube, figure-selected-configuration, concept-explanation-games, concept-metrics]
keywords: [parallel coordinates, lines, every trial, axes order, good region, hyperparameter values vs score]
status: draft
sources:
  - ui/figures/catalog.py:277-295 (ParallelCoordinates)
  - ui/templates/ui/figures/parallel_coordinates.html
  - ui/figures/plots/space.py:202-319 (parallel_coordinates_plot and helpers), 18-30 (_INTENSITY_SCALE)
  - ui/templates/ui/experiment_detail.html:513-529 (line highlight), 857-978 (click via hover)
  - ui/figure_layout.toml (misc)
---

## What it shows

Every trial as one line running across a vertical axis per hyperparameter and ending on a final axis for its score. Lines that score well and bend through the same region of an axis suggest that axis matters and where its good values are.

## How to read it

- **Axes, left to right**: hyperparameters ordered by HyperSHAP tunability for the viewed metric (measured from the default configuration), most important first, then the score axis, named after the metric. Without importance numbers the order is the configuration's own.
- Each axis has its own range, with its real values written beside it. A hyperparameter that never varied sits in the middle.
- **Categorical and true/false** hyperparameters are placed at evenly spaced positions, with the choices written on the ticks; the spacing is not a distance.
- **Log-scaled** hyperparameters are drawn on a log scale, with their real values on the ticks.
- **Colour**: light to dark blue by score, darker is better whichever direction the metric runs. The best trials are drawn on top.
- **Failed trials are not drawn**: their score is a placeholder, so the line count is the number of trials that produced a measurement.
- The selected trial is drawn as a thicker coral line on top.

## Controls

None. Click a line (anywhere along it) to select that trial on the whole page; hover shows its trial number and score. Unlike some parallel-coordinates tools, axes cannot be dragged to reorder or filtered by brushing.

## When it appears

- **Misc** tab by default, full width.
- Follows the metric selector: both the colour and the axis order are per metric. It does not follow the sidebar's Game or Compare against.
- Updates during a run as trials arrive. Needs no Compute button.
- "No trials to plot yet." when there are no trials, or none that succeeded.
- Switched off with "Parallel coordinates" in the experiment's settings.

## Questions people ask

**Where are the good values of a hyperparameter?** Follow the darkest lines and see where they cross that hyperparameter's axis.

**Why is this axis first?** It had the largest tunability share for this metric; the order follows HyperSHAP importance.

**Why are there fewer lines than trials?** Failed trials are left out.

**Why are the axes in a plain order?** Importance was not available for this metric (for example, explanations were not computed), so the configuration's order is used.
