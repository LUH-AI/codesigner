---
id: figure-interactions-top-pairs
title: "Interactions: top pairs"
kind: figure
about: [figure:interactions_top_pairs, element:explanation-game, element:hypershap-baseline]
related: [concept-explanation-games, figure-interactions-heatmap, figure-interactions-coalitions, figure-hyperparameter-importance, page-dashboard-sidebar]
keywords: [strongest interactions, pairs, synergy, redundancy, ranked, FSII, order 2]
status: draft
sources:
  - ui/figures/catalog.py:113-152 (_Interactions, InteractionsTopPairs)
  - ui/templates/ui/figures/interactions_top_pairs.html, _before_this_run.html
  - ui/figures/plots.py:370-388 (hyperparameter_interactions_bar_plot)
  - core/analytics.py:859-867 (_extract_pairwise)
  - ui/templates/ui/experiment_detail.html:1431-1441, 3452-3475
---

## What it shows

The handful of hyperparameter pairs that interact most strongly, as a ranked bar chart. It is the same data as Interactions: heatmap, read as a list so you do not have to scan a grid.

## How to read it

- One bar per pair, labelled "a × b"; at most the 10 strongest pairs, ranked by size from left to right.
- Bar height is the signed interaction value (y axis "Interaction strength").
- **Blue, above zero**: synergy — the pair is worth more tuned together than each alone. **Red, below zero**: redundancy — together they are worth less.
- Ranking is by magnitude, so a large red bar can come before a smaller blue one.
- Values are raw, signed FSII order-2 values for the game chosen in the sidebar, not percentage shares.

## Controls

None of its own. The sidebar's **Game** (`explanation-game`) and **Compare against** (`hypershap-baseline`) switch it together with the importance figure and the other interaction figures. Hover a bar for its value.

## When it appears

- **HyperSHAP** tab. Follows the metric selector.
- Computed with the importance figure when a run finishes; against Incumbent or Selected, on request with it.
- "Waiting for run to finish…" during a run.
- Empty with "Press Compute explanations to see results." when the run's explanations are missing, or "No interaction data available." when there are none — including a model with a single hyperparameter (no pairs) and runs where HyperSHAP fell back to surrogate feature importances, which carry no interactions.
- Switched off with "Interactions: top pairs" in settings.

## Questions people ask

**Which two hyperparameters should I tune together?** The tallest blue bars are the pairs whose joint effect exceeds their separate effects.

**Why is the strongest pair negative?** Pairs are ranked by size, not sign. A large red bar means the two overlap: tuning one already captures much of what the other offers.

**Why only ten pairs?** The figure shows the strongest ten; Interactions: heatmap shows every pair.

**Do these include groups of three or more?** No. For larger groups, see Interactions: graph or Interactions: coalitions.
