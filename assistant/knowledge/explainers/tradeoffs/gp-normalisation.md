---
id: tradeoff-gp-normalisation
title: Normalising the scores for the Gaussian process
kind: tradeoff
about: [field:opt_gp_normalize_y]
verdict: default is better
related: [concept-surrogate-and-acquisition, tradeoff-gp-fitting]
keywords: [normalise surrogate targets, normalize y, score scale, Gaussian process fit]
status: draft
sources:
  - core/optimizers/smac/facade.py (_surrogate: normalize_y)
  - RW06
---

## The trade-off

The process assumes scores centred on 0 with a spread of about 1, and normalising, on by default, brings any metric to that [RW06]. Off, a metric on another scale, an RMSE in the thousands or a log loss near 0.05, is fitted badly.

## When to stray

- None in practice.

## Questions people ask

**Does normalising change my scores?** No. Only the search's own model sees them rescaled; every score is recorded as measured.
