---
id: figure-interactions-graph
title: "Interactions: graph"
kind: figure
about: [figure:interactions_graph, explanation-game, hypershap-baseline]
related: [concept-explanation-games, figure-interactions-coalitions, figure-interactions-orders, figure-interactions-heatmap, page-dashboard-sidebar]
keywords: [Möbius graph, interaction graph, higher-order interactions, coalitions, hyperedge, synergy, redundancy]
status: draft
sources:
  - ui/figures/catalog.py:161-228 (_Interactions, InteractionsGraph)
  - ui/templates/ui/figures/interactions_graph.html, _before_this_run.html
  - ui/figures/plots.py:636-754 (_circle_positions, hyperparameter_graph_plot)
  - core/optimizers/base.py:977-992 (_moebius_terms), 2441-2564 (_compute_hp_game)
  - git show e933531^:docs/walkthroughs/hypershap-plots.md (Möbius value meaning)
---

## What it shows

The Möbius interaction graph: hyperparameters placed on a circle, with lines for the interactions between them. It is the only figure on the page that shows interactions of three or more hyperparameters at once, for the game chosen in the sidebar.

## How to read it

- **Node** (blue circle): a hyperparameter. Its size is its own (order-1) value; hover says "on its own".
- **Line between two nodes**: a pair's interaction.
- **Hub with spokes**: a group of three or more; spokes run from the group's centre to each member, meaning they interact as a set rather than as separate pairs.
- **Line width**: the size of the interaction. **Colour**: blue for synergy (worth more together), red for redundancy (worth less).
- Only the 15 strongest interactions are drawn.
- Hover the small grey circle in the middle of a line or at a hub for its members and signed value.
- The circle is a layout convention: distances between nodes mean nothing.

Each value is a Möbius term: what exactly that set of hyperparameters contributes beyond what its smaller subsets already explain. These are not the FSII numbers the heatmap and top pairs show.

## Controls

None of its own. The sidebar's **Game** (`explanation-game`) and **Compare against** (`hypershap-baseline`) switch it with the other HyperSHAP figures.

## When it appears

- **HyperSHAP** tab. Follows the metric selector.
- Computed with the importance figure when a run finishes; against Incumbent or Selected, on request with it.
- "Waiting for run to finish…" during a run.
- Empty ("No interaction data available." or "Press Compute explanations to see results.") with fewer than two hyperparameters, when explanations are missing, or when HyperSHAP fell back to surrogate feature importances.
- Switched off with "Interactions: graph" in settings.

## Questions people ask

**Do these hyperparameters interact as a group?** A hub with spokes to several nodes says yes; check its hover for the value.

**Why is it hard to read?** With many hyperparameters the graph becomes crowded. Interactions: coalitions shows the same data as a ranked list and stays legible past about six hyperparameters.

**Why do some interactions I saw in the heatmap not appear?** Only the 15 strongest Möbius terms are drawn, and Möbius values differ from the heatmap's FSII values.

**Does a big node mean an important hyperparameter?** It means a large effect on its own; its group effects are in the lines.
