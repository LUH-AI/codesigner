---
id: tradeoff-sampling-method
title: How the design's points are drawn
kind: tradeoff
about: [field:opt_initial_design]
verdict: default is better
related: [concept-initial-design, tradeoff-initial-design-size]
keywords: [sampling method, Sobol, Latin hypercube, uniformly random, defaults only, space filling, quasi-random, continued run]
status: draft
sources:
  - core/optimizers/smac/params.py (INITIAL_DESIGNS, EXTENDABLE_DESIGNS)
  - core/optimizers/smac/facade.py (_pinned_points)
  - Sob67
  - MBC79
  - COKT26
---

## The trade-off

Sobol, the default, spreads points evenly over the whole space and extends: a continued run that asks for more design points keeps the old ones and adds to them [Sob67]. A Latin hypercube guarantees each hyperparameter's range is split evenly for any number of points [MBC79], which Sobol does exactly only at powers of two. Its points all move when its size changes, so a continued run keeps the first size. Uniformly random covers the space less evenly than either, and the search then needs more trials in total [COKT26]. Defaults only is a single point.

## When to stray

- Latin hypercube: a design whose size is far from a power of two, for a run that will not be continued. The gain over Sobol is small.
- Uniformly random: only to compare against a plain random start.
- Defaults only: a budget too small for a design. Expect the first guided trials to go to the edges of the space [COKT26].

## Questions people ask

**Why Sobol rather than random points?** Sobol points fill the space evenly, so a design of the same size tells the model more.

**Why does a Latin hypercube design not grow on a continued run?** A different size moves every point, so the run keeps the size it started with.

**Can I change the sampling method after the first run?** No, it is fixed once the experiment has trials.
