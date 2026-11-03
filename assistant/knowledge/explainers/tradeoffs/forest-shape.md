---
id: tradeoff-forest-shape
title: The shape of the search's trees
kind: tradeoff
about: [field:opt_rf_max_depth, field:opt_rf_min_samples_split, field:opt_rf_min_samples_leaf, field:opt_rf_feature_ratio, field:opt_rf_bootstrapping]
verdict: default is better
related: [concept-surrogate-and-acquisition, tradeoff-surrogate-trees, tradeoff-random-trial-rate]
keywords: [surrogate tree depth, split threshold, leaf size, feature ratio, bootstrapping, noisy scores, forest uncertainty]
status: draft
sources:
  - core/optimizers/smac/params.py (the rf_ settings)
  - ui/optimizer_labels.py (the settings' help)
  - Bre01
  - HXHL14
---

## The trade-off

By default the search's trees grow fully, consider every hyperparameter at each split, and are each fitted to a resample of the trials. Fully grown trees fit every trial; shallower trees or larger leaves average neighbouring trials, which smooths over noise and blurs real detail. A feature ratio below 1 makes the trees differ more: wider uncertainty and more exploration, from less accurate trees [Bre01]. Resampling is what makes the trees disagree where trials are few [HXHL14]; without it the forest's uncertainty collapses and the search turns greedy, leaving exploration to the random trial rate. With the few hyperparameters of the built-in models, none of this changes much.

## When to stray

- Noisy scores (a small dataset, few folds, scores that vary widely between folds): a leaf size of a few trials, to smooth them.
- A custom model with many hyperparameters under the forest, whose search keeps to one region: lower the feature ratio (0.8, say).
- Bootstrapping off: none in practice.

## Questions people ask

**Are these settings of my random forest model?** No. They shape the search's own forest, which models the score; the model being tuned has its own settings of similar names.

**Why keep bootstrapping on?** It is what gives the forest its sense of where it is unsure.
