---
id: tradeoff-improvement-margin
title: The improvement margin
kind: tradeoff
about: [field:opt_acquisition_xi]
verdict: trade-off
related: [concept-surrogate-and-acquisition, tradeoff-acquisition-function, tradeoff-search-strategy]
keywords: [improvement margin, xi, exploration, exploitation, stuck search, units of the margin]
status: draft
sources:
  - core/optimizers/smac/facade.py (_facade: xi passed to EI and PI)
  - core/metrics.py (to_cost: a score becomes a cost on the metric's scale)
  - JSW98
  - Gar23
---

## The trade-off

The margin is how much better than the best so far a candidate must promise to be. A higher margin explores more and refines less [Gar23]. The default is 0. Under the Gaussian process it is in the metric's own units: 0.01 is one percentage point of accuracy. Under the forest the scores are rescaled to the range seen and logged before the margin applies, so a value there has no meaning in the metric's units.

## When to stray

- For a later run under the Gaussian process whose picks keep landing next to the best without improving it: a margin about the size of an improvement worth having.
- Under the forest: leave it at 0.

## Questions people ask

**What margin should I use?** Under the Gaussian process, about the smallest improvement you would care about, such as 0.005 for accuracy.

**Why leave it at 0 under the forest?** The forest's search applies it to rescaled, logged scores, where no value means a fixed amount of your metric.
