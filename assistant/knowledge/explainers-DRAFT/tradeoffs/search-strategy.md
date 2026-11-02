---
id: tradeoff-search-strategy
title: Gaussian process or random forest
kind: tradeoff
about: [field:opt_search_strategy]
verdict: trade-off
related: [concept-optimizers, concept-surrogate-and-acquisition, concept-priors, tradeoff-random-trial-rate, tradeoff-refit-interval]
keywords: [search strategy, Gaussian process, random forest, surrogate, categorical hyperparameters, conditional hyperparameters, many hyperparameters, long runs, BlackBoxFacade, HyperparameterOptimizationFacade]
status: draft
sources:
  - core/optimizers/smac/params.py (STRATEGIES, FACADE_DEFAULTS)
  - core/optimizers/smac/facade.py (_surrogate)
  - core/models/svm.py (the SVM's space)
  - LEF+22
  - HHL11
  - HXHL14
  - RW06
---

## The trade-off

The Gaussian process (`gp`, the default) learns the most from few trials when the hyperparameters are few and numeric, and its uncertainty is a real posterior [RW06]. Fitting it grows with the cube of the trials, so past a few hundred trials every pick slows. The random forest (`rf`) copes with categorical hyperparameters and with many of them, and fits quickly at any number of trials [HHL11, LEF+22]. Its uncertainty is how much its trees disagree, which is overconfident far from the trials [HXHL14], and its predictions are steps rather than a curve. The strategy also sets other defaults: the random trial rate, the candidates per trial and the refit interval.

## When to stray

- A categorical hyperparameter decides whether others matter (SVM's kernel: degree counts only for poly), or most of them are categorical (k-Nearest Neighbors, ETS): `rf`, whose trees can split on the category first.
- A custom model with more than about ten hyperparameters: `rf`.
- Several hundred cheap trials or more, where refitting the process for each pick would take longer than the trial: `rf`.
- Keep `gp` when the person wants to judge a prior with Evaluate Prior, which works only under `gp`.

## Questions people ask

**Which strategy suits the built-in boosting models?** The Gaussian process: their four hyperparameters are numeric, and a classification adds one two-way class weighting.

**Can I switch strategy after the first run?** No. The strategy is fixed once the experiment has trials, because the next run continues their search.

**Is the forest worse at finding the best configuration?** Not in general. It is less sure-footed with few trials and few numeric hyperparameters, and better with categories, many hyperparameters or long runs.
