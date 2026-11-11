---
id: tradeoff-acquisition-function
title: Expected or probable improvement
kind: tradeoff
about: [field:opt_acquisition]
verdict: default is better
related: [concept-surrogate-and-acquisition, tradeoff-improvement-margin]
keywords: [acquisition function, expected improvement, EI, probability of improvement, PI, exploitation, refine, polish the best]
status: draft
sources:
  - core/optimizers/smac/facade.py (_facade: EI or PI)
  - JSW98
  - Kus64
  - Gar23
---

## The trade-off

Expected improvement, the default, weighs how much better a candidate might be [JSW98]. Probability of improvement weighs only whether it might be better [Kus64], so it favours small, likely gains next to the best and explores less. Expected improvement is the robust choice for a whole run [Gar23].

## When to stray

- For a short last run that refines around the best, when too few trials remain for exploring elsewhere to pay off: probability of improvement. Under the Gaussian process pair it with an improvement margin, or it settles for negligible gains.

## Questions people ask

**Which finds better configurations?** Over a whole run, expected improvement; probability of improvement tends to settle near the first good region it finds.

**Can I change it between runs?** Yes, in the middle column of the Evaluation card.
