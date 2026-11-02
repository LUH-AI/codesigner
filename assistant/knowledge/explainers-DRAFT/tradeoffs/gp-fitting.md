---
id: tradeoff-gp-fitting
title: How the Gaussian process is fitted
kind: tradeoff
about: [field:opt_gp_model_type, field:opt_gp_restarts]
verdict: trade-off
related: [concept-surrogate-and-acquisition, tradeoff-search-strategy, tradeoff-gp-normalisation]
keywords: [surrogate fitting, vanilla, MCMC, maximum likelihood, kernel hyperparameters, fit restarts, calibration, expensive trials]
status: draft
sources:
  - core/optimizers/smac/facade.py (_surrogate)
  - core/optimizers/smac/params.py (GP_MODELS)
  - SLA12
  - RW06
---

## The trade-off

Vanilla, the default, fits the kernel's own hyperparameters once, by maximum likelihood, from 10 starting points, and can be confidently wrong while the trials are few. MCMC averages over them: better calibrated with few trials, at about ten times the fitting time per pick, growing with the trials [SLA12]. More restarts make a poor maximum-likelihood fit less likely, and each one adds fitting time to every pick [RW06].

## When to stray

- Trials that take a minute or more and a budget under about 100: MCMC. The extra seconds per pick are small next to a trial.
- Trials that take a minute or more, under vanilla: more restarts (20, say).
- Hundreds of cheap trials, when fitting takes much of the run's time: fewer restarts.

## Questions people ask

**When is MCMC worth its time?** When trials are slow and few, so the search's own fitting is small next to them and its better sense of uncertainty matters most.

**Can I change it between runs?** No, it is fixed once the experiment has trials.
