---
id: figure-interactions-coalitions
title: "Interactions: coalitions"
kind: figure
about: [figure:interactions_coalitions, element:explanation-game, element:hypershap-baseline]
related: [concept-explanation-games, figure-interactions-graph, figure-interactions-orders, figure-interactions-top-pairs, page-dashboard-sidebar]
keywords: [UpSet plot, coalitions, groups of hyperparameters, Möbius, higher-order interactions, ranked]
status: draft
sources:
  - ui/figures/catalog.py:113-171 (_Interactions, InteractionsCoalitions)
  - ui/templates/ui/figures/interactions_coalitions.html, _before_this_run.html
  - ui/figures/plots/interactions.py:147-198 (hyperparameter_upset_plot)
  - core/analytics/games.py:62-70 (_moebius_terms)
---

## What it shows

An UpSet plot of the strongest groups of hyperparameters: which named combinations carry value, ranked. It reads the same Möbius decomposition as Interactions: graph, as a list instead of a picture, so it stays legible for searches with many hyperparameters where the graph becomes crowded (around six and up).

## How to read it

- **Top panel**: one bar per group (coalition) of two or more hyperparameters, at most the 12 strongest, ranked by size. The bar is the group's signed Möbius value (y axis "Interaction"). **Blue, above zero**: synergy. **Red, below zero**: redundancy.
- **Bottom panel**: a matrix with one row per hyperparameter. Under each bar, **filled dots** joined by a line are the members of that group; **hollow dots** are the hyperparameters left out.
- Hover a bar for the group's members ("a × b × c") and value.
- A Möbius value is what exactly that group contributes beyond its smaller subsets, for the game chosen in the sidebar.

## Controls

None of its own. The sidebar's **Game** (`explanation-game`) and **Compare against** (`hypershap-baseline`) switch it with the other HyperSHAP figures.

## When it appears

- **HyperSHAP** tab. Follows the metric selector.
- Computed with the importance figure when a run finishes; against Incumbent or Selected, on request with it.
- "Waiting for run to finish…" during a run.
- Empty when there is no group of two or more: a single-hyperparameter model, missing explanations ("Press Compute explanations to see results."), or a run where HyperSHAP fell back to surrogate feature importances ("No interaction data available.").
- Switched off with "Interactions: coalitions" in settings.

## Questions people ask

**Which group of hyperparameters matters most together?** The leftmost bar; the filled dots under it name the members.

**Why is a bar below zero?** The group is redundant: its members together are worth less than their separate and smaller-group effects add up to.

**Why do pairs here differ from Interactions: top pairs?** This figure uses Möbius values; top pairs uses FSII values. They are different decompositions of the same game.

**Where are single hyperparameters?** Groups of one are left out here; see Interactions: by order for each hyperparameter's own share.
