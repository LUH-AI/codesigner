---
id: tradeoff-refit-interval
title: How often the model is refitted
kind: tradeoff
about: [field:opt_retrain_after]
verdict: trade-off
related: [concept-surrogate-and-acquisition, tradeoff-search-strategy, tradeoff-candidate-search]
keywords: [refit interval, retrain after, stale model, surrogate refit, optimizer overhead]
status: draft
sources:
  - core/optimizers/smac/params.py (FACADE_DEFAULTS)
  - LEF+22
---

## The trade-off

The interval is the number of trials between refits of the surrogate. In between, picks come from the last fit's ranking. A higher interval spends less time fitting and picks from staler information. The defaults are every trial under the Gaussian process and every eighth under the forest [LEF+22], a choice made for cheap trials.

## When to stray

- Under the forest, with trials that take seconds or more and a budget under a few hundred: 1, so every pick uses every result. A forest refit takes a fraction of a second.
- Under the Gaussian process, with hundreds of cheap trials, when refitting the process takes much of the run's time: raise it (4 to 8).

## Questions people ask

**Why does the forest refit only every eighth trial?** SMAC's default for the forest assumes cheap trials, where refitting each time would cost more than it gains.

**Can I change it between runs?** No, it is fixed once the experiment has trials.
