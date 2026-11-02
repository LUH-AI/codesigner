---
id: tradeoff-optimized-metric
title: Which metric the search optimizes
kind: tradeoff
about: [field:optimize_metric]
verdict: trade-off
related: [concept-metrics, concept-evaluation, tradeoff-validation-method]
keywords: [optimized metric, evaluation metric, accuracy, balanced accuracy, F1, precision, recall, ROC AUC, log loss, RMSE, MAE, R², MASE, sMAPE, imbalanced classes, ties, outliers, proxy metric]
status: draft
sources:
  - core/metrics.py (each metric's definition, direction and range)
  - core/optimizers/smac/history.py (replay: a metric change re-reads the trials)
  - assistant/tools/planning.py (list_metrics)
  - HK06
  - HA21
  - BBL+23
---

## The trade-off

The search optimizes one metric, though every trial is scored on all of the task's metrics. The one it optimizes is what the surrogate learns from and what decides the best trial. Two things can pull apart: the metric that says what the person cares about, and the metric that gives the search the most to learn from. Metrics that count right and wrong labels (accuracy, balanced accuracy, F1) move in steps: with 150 rows, a cross-validated accuracy moves in steps of 1/150, and many configurations tie, which leaves the surrogate little to tell them apart by. Metrics on class probabilities (log loss, ROC AUC) move smoothly and separate them, but they are not the decision the model will make. The best configuration under one metric need not be the best under another.

## When to stray

- Classes of very different sizes, and the small ones matter: balanced accuracy (the same number as recall(macro)) instead of accuracy, which a model can score well on by favouring the largest class. Weighted F1 and precision do not fix this, since large classes dominate them too. Balanced accuracy is noisier when a class has few rows per fold.
- A small classification dataset where many trials tie on accuracy: optimize log loss and judge the best trials by accuracy in the trials table. Log loss rewards calibrated probabilities and punishes confident mistakes without limit, so its best may not be the most accurate. It needs a model that gives probabilities.
- The model's probabilities will be used, to rank cases or with a threshold set later: ROC AUC for the ranking, log loss for probabilities that mean what they say.
- Precision or recall on their own: Codesigner averages them over all classes, so neither targets one class's errors; they are rarely the right thing to optimize.
- A regression target with outliers that should not dominate: MAE instead of RMSE. When large errors are what is costly: RMSE. R² ranks configurations almost exactly as RMSE does; choose between them for readability, not for the search.
- A forecast: MASE, the default, is scale-free and comparable across series, and fails only when a series' history is flat [HK06]. sMAPE reads as a percentage, but it is unstable near zero and, despite its name, treats over- and under-forecasts differently [HK06, HA21]; avoid it for series that reach zero.
- A later run may optimize another metric. It re-reads every trial under the new one, so nothing is lost, but the earlier trials were chosen for the old one.

## Questions people ask

**Why optimize log loss if I care about accuracy?** On small data many configurations tie on accuracy, and log loss still tells them apart, so the search learns more per trial. Check the best trials' accuracy afterwards.

**Why do so many trials have the same accuracy?** Accuracy counts rows; with 150 rows it can only change in steps of 1/150.

**Can I change the metric later?** Yes. The metric is chosen for each run, and a run under a new metric re-reads all the trials.

**Is F1 the right metric for imbalanced classes?** Not as Codesigner computes it: it is weighted by class size, so the large classes dominate. Balanced accuracy counts each class equally.
