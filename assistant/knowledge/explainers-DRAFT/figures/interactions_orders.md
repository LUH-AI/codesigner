---
id: figure-interactions-orders
title: "Interactions: by order"
kind: figure
about: [figure:interactions_orders, explanation-game, hypershap-baseline]
related: [concept-explanation-games, figure-interactions-graph, figure-interactions-coalitions, figure-hyperparameter-importance, page-dashboard-sidebar]
keywords: [on its own, in pairs, in larger groups, Möbius, Harsanyi dividend, attributed influence, interaction order]
status: draft
sources:
  - ui/figures/catalog.py:161-248 (_Interactions, InteractionsOrders)
  - ui/templates/ui/figures/interactions_orders.html, _before_this_run.html
  - ui/figures/plots.py:824-889 (_ORDER_BANDS, hyperparameter_orders_plot)
  - core/optimizers/base.py:977-992 (_moebius_terms)
---

## What it shows

For each hyperparameter, where its attributed influence comes from: from itself alone, from its pairs with others, or from larger groups. It answers a question no other figure does: does this hyperparameter matter on its own, or only in combination?

## How to read it

- One stacked bar per hyperparameter, ordered by total height, largest first.
- **Dark blue — "On its own"**: its own (order-1) Möbius term.
- **Medium — "In pairs"**: its share of every pair it is in.
- **Light — "In larger groups"**: its share of every group of three or more.
- Y axis: "Attributed influence".
- Each Möbius term is split equally among its members (the usual reading of a Harsanyi dividend).
- Bars are built from magnitudes, not signed values, so helping and hurting effects do not cancel out.

A tall dark base means the hyperparameter is worth tuning by itself. A bar that is mostly light means it pays off only alongside others, which is different advice. Values are for the game chosen in the sidebar.

## Controls

None of its own. The sidebar's **Game** (`explanation-game`) and **Compare against** (`hypershap-baseline`) switch it with the other HyperSHAP figures. Hover a segment for its value.

## When it appears

- **HyperSHAP** tab. Follows the metric selector.
- Computed with the importance figure when a run finishes; against Incumbent or Selected, on request with it.
- "Waiting for run to finish…" during a run.
- Empty when the Möbius decomposition is missing: explanations not computed ("Press Compute explanations to see results."), a result from before it was stored, or a run where HyperSHAP fell back ("No interaction data available.").
- Switched off with "Interactions: by order" in settings.

## Questions people ask

**Can I tune this hyperparameter on its own?** If most of its bar is dark blue, yes; if most is light, it matters mainly together with others.

**Why does this ranking differ from the importance figure?** The importance figure ranks order-1 FSII shares; this ranks total Möbius magnitude, including group effects.

**Why are all values positive?** Magnitudes are summed, so a hyperparameter that helps in one combination and hurts in another still shows as influential.

**Which other hyperparameters is it interacting with?** This figure does not say; see Interactions: graph or coalitions.
