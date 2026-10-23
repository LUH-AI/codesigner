---
id: figure-interactions-heatmap
title: "Interactions: heatmap"
kind: figure
about: [figure:interactions_heatmap, explanation-game, hypershap-baseline]
related: [concept-explanation-games, figure-interactions-top-pairs, figure-interactions-graph, figure-hyperparameter-importance, page-dashboard-sidebar]
keywords: [interaction, pairwise, synergy, redundancy, FSII, order 2, heatmap, hyperparameter pairs]
status: draft
sources:
  - ui/figures/catalog.py:161-211 (_Interactions, InteractionsHeatmap)
  - ui/templates/ui/figures/interactions_heatmap.html, _before_this_run.html
  - ui/figures/plots.py:546-570 (hyperparameter_interactions_heatmap_plot), 615-633 (_SIGN_SCALE)
  - core/optimizers/base.py:1561-1585 (compute_hp_interactions), 2441-2564 (_compute_hp_game), 2566-2578 (_extract_pairwise)
  - ui/templates/ui/experiment_detail.html:1437-1447 (syncInteractionsWarning), 3695-3718 (game and baseline handlers)
  - ui/views.py:2124-2138 (_analytics_absent)
---

## What it shows

Every pair of hyperparameters at once, as a grid of HyperSHAP interaction values for the game chosen in the sidebar. It answers "do these two hyperparameters work together, or does one make the other redundant?" for all pairs in one picture.

## How to read it

- Rows and columns are the hyperparameters, ordered by the size of their own (diagonal) value, largest first, so the strongest single effects sit in the top-left corner.
- **Diagonal cell**: the hyperparameter's own signed order-1 value.
- **Off-diagonal cell**: the signed interaction between the two.
- **Blue** (positive) is synergy: the pair is worth more together than the sum of each alone. **Red** (negative) is redundancy: together they are worth less. White is near zero.
- Values are raw, signed numbers, not the percentage shares the importance figure shows.

These are HyperSHAP's FSII values from an order-2 fit. They are not the same numbers as the graph, coalitions and by-order figures, which read the Möbius decomposition; this is deliberate, so do not expect them to match.

## Controls

No controls of its own. The sidebar's **Game** (`explanation-game`) and **Compare against** (`hypershap-baseline`) switch this figure together with the importance figure and the other interaction figures. Hover a cell for its value.

## When it appears

- **HyperSHAP** tab. Follows the metric selector.
- Computed with the importance figure when a run finishes, at no extra cost; against Incumbent or Selected it is computed on request with the importance figure (see that figure's Compute button).
- "Waiting for run to finish…" during a run.
- "Press Compute explanations to see results." when the run's explanations are missing; "No interaction data available." when there are none.
- Empty while the importance figure is still filled when HyperSHAP fell back to surrogate feature importances or uniform weights: those have no interactions. The warning above the figure says so.
- Switched off with "Interactions: heatmap" in settings.

## Questions people ask

**Which hyperparameters interact?** Look for the darkest off-diagonal cells, or use Interactions: top pairs for a ranked list.

**What does red mean?** Redundancy: tuning both together gains less than tuning each alone would suggest.

**Why are the numbers different from the graph's?** The heatmap shows FSII values; the graph shows Möbius values, a different decomposition of the same game.

**Why is it empty when importance is not?** The importance figure fell back to a method that has no interaction values; the warning gives the reason.
