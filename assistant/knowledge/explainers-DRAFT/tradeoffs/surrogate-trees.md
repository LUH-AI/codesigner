---
id: tradeoff-surrogate-trees
title: How many trees the search's forest has
kind: tradeoff
about: [field:opt_rf_trees]
verdict: trade-off
related: [concept-surrogate-and-acquisition, tradeoff-search-strategy, tradeoff-forest-shape]
keywords: [surrogate trees, number of trees, forest uncertainty, calibration, random forest surrogate]
status: draft
sources:
  - core/optimizers/smac/params.py (FACADE_DEFAULTS: n_trees)
  - core/analytics/surrogate.py (the figures' forest of 100 trees)
  - Bre01
  - HXHL14
---

## The trade-off

More trees give a steadier prediction and a steadier spread, the spread being what drives exploration under the forest [Bre01, HXHL14]. Fitting time grows with the number of trees. The default of ten is few. Codesigner's own figures fit their forests with 100.

## When to stray

- Trials that take seconds or more: raise it (50 to 100); the forest still fits in well under a second.

## Questions people ask

**Are these the trees of my model?** No. They belong to the search's own model of the score, not to the model being tuned.

**Can I change it between runs?** No, it is fixed once the experiment has trials.
