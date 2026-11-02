---
id: tradeoff-candidate-search
title: How hard each pick searches
kind: tradeoff
about: [field:opt_challengers, field:opt_local_search_iterations]
verdict: trade-off
related: [concept-surrogate-and-acquisition, tradeoff-refit-interval]
keywords: [candidates per trial, challengers, local search iterations, acquisition maximizer, optimizer overhead, slow picks, cheap trials, expensive trials]
status: draft
sources:
  - core/optimizers/smac/params.py (FACADE_DEFAULTS)
  - ui/optimizer_labels.py (the settings' help)
  - HHL11
---

## The trade-off

Before each trial the search looks for the best candidate under the model: it scores many configurations, then climbs from the promising ones [HHL11]. More candidates and more climbing find a better candidate and make each pick slower. The defaults are 1,000 candidates under the Gaussian process and 10,000 under the forest, whose predictions are cheap, with 10 local search iterations under both.

## When to stray

- Trials that take a minute or more: raise them (5,000 candidates, say), since the extra seconds per pick do not matter.
- Trials of a second or two and a budget of hundreds, when the run's time goes to the search rather than to the fits (compare the trial durations with the run's length): lower them.

## Questions people ask

**Why is my run slower than its trials add up to?** The search spends time on each pick: fitting the model and scoring candidates. With very fast trials that time can dominate.

**Can I change these between runs?** No, they are fixed once the experiment has trials.
