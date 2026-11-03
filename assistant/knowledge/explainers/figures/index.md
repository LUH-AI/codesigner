---
id: page-figure-index
title: Dashboard figures
kind: page
about: [figure:selected_configuration, figure:best_configuration, figure:hyperparameter_importance, figure:performance_over_time, figure:interactions_heatmap, figure:interactions_top_pairs, figure:interactions_graph, figure:interactions_coalitions, figure:interactions_orders, figure:trial_duration, figure:forecast, figure:parallel_coordinates, figure:partial_dependence, figure:local_explanation, figure:local_effects, figure:acquisition_slice, figure:prior, figure:configuration_cube, figure:trials]
related: [page-dashboard-sidebar, concept-explanation-games, concept-priors, concept-projection, concept-metrics]
keywords: [which figure, dashboard, tabs, overview, HyperSHAP tab, acquisition tab, misc tab]
status: draft
sources:
  - ui/figures/catalog.py (FIGURES, home_tab)
  - ui/figures/base.py:13-20 (tabs)
  - ui/views/dashboard.py:231-237 (TAB_LABELS)
  - ui/figure_layout.toml (default placement and Overview pins)
---

Each figure lives on one tab (its home tab). **Overview** holds copies pinned from the other tabs; by default these are Best configuration, Hyperparameter importance, Trials, Trial performance, Forecast and Hyperparameter space projection. Readers can pin, move and resize figures; this lists the defaults. Any figure can be switched off in the experiment's settings.

## Sidebar

- [Selected Configuration](selected_configuration.md) (`figure-selected-configuration`): which trial is selected, its score, distance from the best, and configuration.

## HyperSHAP

- [Hyperparameter importance (HyperSHAP)](hyperparameter_importance.md) (`figure-hyperparameter-importance`): which hyperparameters matter most, under the chosen game?
- [Interactions: heatmap](interactions_heatmap.md) (`figure-interactions-heatmap`): how does every pair of hyperparameters interact?
- [Interactions: top pairs](interactions_top_pairs.md) (`figure-interactions-top-pairs`): which pairs interact most strongly?
- [Interactions: graph](interactions_graph.md) (`figure-interactions-graph`): which hyperparameters interact, including as groups of three or more?
- [Interactions: coalitions](interactions_coalitions.md) (`figure-interactions-coalitions`): which named groups of hyperparameters carry the most value?
- [Interactions: by order](interactions_orders.md) (`figure-interactions-orders`): does each hyperparameter matter on its own, in pairs, or in larger groups?
- [Local explanation (selected trial)](local_explanation.md) (`figure-local-explanation`): why did the selected trial score as it did, hyperparameter by hyperparameter?
- [Local effects across trials](local_effects.md) (`figure-local-effects`): does a hyperparameter's effect stay the same across the run, or help in some places and hurt in others?

## Acquisition

- [Acquisition function](acquisition_slice.md) (`figure-acquisition-slice`): which values of a hyperparameter does the optimizer find promising, and what would it run next?
- [Prior](prior.md) (`figure-prior`): where do I believe a hyperparameter's good values are, and how should that belief fade?
- [Hyperparameter space projection](configuration_cube.md) (`figure-configuration-cube`): where did the search go, where did it score well, and what does the model predict elsewhere?

## Misc

- [Best configuration](best_configuration.md) (`figure-best-configuration`): which trial is best so far for this metric, and with which values?
- [Trial performance](performance_over_time.md) (`figure-performance-over-time`): is the search still improving?
- [Trial duration](trial_duration.md) (`figure-trial-duration`): how long did each trial take, and which failed?
- [Trials](trials.md) (`figure-trials`): every trial's scores, duration and configuration, sortable.
- [Parallel coordinates](parallel_coordinates.md) (`figure-parallel-coordinates`): which ranges of each hyperparameter do the good trials pass through?
- [Partial dependence (PDP/ICE)](partial_dependence.md) (`figure-partial-dependence`): how does the predicted score change with one hyperparameter, and does that depend on the others?
- [Forecast](forecast.md) (`figure-forecast`): what did the selected configuration forecast at each backtest? Forecast experiments only.
