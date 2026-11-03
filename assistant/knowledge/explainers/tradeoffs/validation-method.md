---
id: tradeoff-validation-method
title: How trials are validated
kind: tradeoff
about: [field:evaluation_scheme, field:evaluation_value]
verdict: trade-off
related: [concept-evaluation, concept-metrics, tradeoff-optimized-metric]
keywords: [validation method, cross-validation, folds, held-out share, holdout, backtests, noise, cost per trial, optimistic best score, overtuning, time order, data leakage]
status: draft
sources:
  - core/splits.py
  - ui/forms.py (the validation method's choices and ranges)
  - BBL+23
  - FH19
  - Tas00
---

## The trade-off

Every trial's score is an estimate, and the validation method sets how noisy it is and what it costs. Cross-validation averages over several splits, which gives a less noisy score than one held-out share [BBL+23], at one model fit per fold: 5 folds is five fits per trial. A held-out share is one fit, the cheapest, and on a small table so noisy that the search can end up chasing the split. Noise matters more to a search than to a single evaluation: the best of many trials is partly the one whose noise flattered it, so its score is optimistic, the more so with small data, fewer splits and more trials [BBL+23, FH19]. The default is 5-fold cross-validation. The method is fixed for the experiment's life.

## When to stray

- Hundreds of rows or fewer, and fits that take seconds: 10 folds, for less noise at twice the fits.
- Tens of thousands of rows, or slow fits: a held-out share (0.2), one fit per trial; the large validation share keeps the noise down. Or 3 folds.
- A time limit and slow fits: 3 folds buy more trials, each noisier.
- A class with only a few dozen rows: keep cross-validation, which validates all of them; a held-out share validates a few, and a per-class metric gets noisy.
- Rows recorded over time, when the model will predict later ones: order by the time column. Shuffled folds let the model learn from rows after the ones it is scored on, which overstates the score [Tas00].
- Future values to forecast: backtests, as many as the history allows (each needs a horizon more of it), with the horizon the model will forecast in use.
- Labels that arrive late: a gap between training and validation rows.

## Questions people ask

**Why not always use 10 folds?** Each fold is one more model fit per trial, so 10 folds take twice as long as 5 for a less noisy score.

**Why is the best score better than what I see on new data?** The best trial was picked because its score was high, partly by luck. The more trials and the smaller the data, the more its score flatters it [BBL+23].

**Can I change the validation method later?** No. Trials scored differently cannot be compared, so it is fixed for the experiment.
