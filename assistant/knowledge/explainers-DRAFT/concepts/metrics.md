---
id: concept-metrics
title: Metrics
kind: concept
about: [field:optimize_metric, field:metric-select]
related: [concept-evaluation, concept-tasks-and-models, concept-stopping, concept-priors, figure-performance-over-time, figure-trials, page-dashboard-sidebar]
keywords: [metric, score, accuracy, F1, precision, recall, balanced accuracy, ROC AUC, log loss, RMSE, MAE, R², MASE, sMAPE, null score, baseline score, failed trial, higher is better, lower is better]
status: draft
sources:
  - core/metrics.py:34-131
  - core/metrics.py:162-258
  - core/metrics.py:257-264
  - core/metrics.py:371-380
  - core/optimizers/trial.py:58-75
  - core/optimizers/trial.py:175-183
  - ui/drafts.py:296-302
  - ui/drafts.py:330-342
  - experiments/services/run.py:35-42
  - experiments/services/priors.py:42-64
  - assistant/data.py:490-506
  - ui/templates/ui/_side_evaluation.html:16-28
  - ui/templates/ui/metric_change.html
---

## In short

A metric turns one trial's predictions into one number. Codesigner scores every trial on every metric that fits the experiment's task, and the search optimizes the one picked under "Evaluation metric". Each metric knows its direction (whether higher or lower is better), its range, and what a model that knows nothing would score.

## In detail

### The metrics

| Name | Measures | Better | Range | For |
|---|---|---|---|---|
| `accuracy` | share of rows labelled right | higher | 0–1 | classification |
| `f1` | F1, class-size weighted | higher | 0–1 | classification |
| `precision` | precision, class-size weighted | higher | 0–1 | classification |
| `recall(macro)` | recall, plain mean over classes | higher | 0–1 | classification |
| `balanced_accuracy` | accuracy averaged per class | higher | 0–1 | classification |
| `roc_auc` | ROC AUC, one class vs the rest, averaged | higher | 0–1 | classification, probabilities |
| `log_loss` | log loss of class probabilities | lower | 0 to ∞ | classification, probabilities |
| `rmse` | root mean squared error | lower | 0 to ∞ | regression |
| `mae` | mean absolute error | lower | 0 to ∞ | regression |
| `r2` | R², share of variance explained | higher | −∞ to 1 | regression |
| `mase` | MAE ÷ MAE of a seasonal naive forecast on each series' history [HK06] | lower | 0 to ∞ | forecast only |
| `smape` | symmetric mean absolute percentage error, % | lower | 0–200 | forecast only |

MASE below 1 beats repeating last season. MASE and sMAPE are offered only for backtests with a numeric target, and a forecast optimizes MASE unless told otherwise.

ROC AUC and log loss need class probabilities. They are tracked only for a classification whose model gives probabilities from its own fit. The SVM does not, because its probabilities cost an extra calibration on every trial. A run that optimizes one of them with a model that gives no probabilities is refused.

With several folds or backtests, a trial's score is the average over them.

### The "know-nothing" score

A trial that fails (raises an error, misses its deadline, or returns something that cannot be scored) is still recorded. It scores what a model that knows nothing would score on the validation rows (`Metric.null_score`):

- 0 for accuracy, F1, precision, recall, balanced accuracy, and R²;
- 0.5 for ROC AUC;
- for log loss, the loss of always predicting each class's share;
- for RMSE, the target's standard deviation; for MAE, the mean absolute distance from its mean (always predicting the average);
- 1 for MASE; for sMAPE, the sMAPE of predicting the mean.

A failed trial never becomes the best trial. The assistant's `baseline_scores` tool reports these numbers, computed on all labelled rows.

### Scale

`core.metrics.scale` is how far apart a perfect and a worthless score are:

- a two-ended metric's range (1 for accuracy, 200 for sMAPE);
- otherwise, the distance from the best end to the null score (the target's spread for RMSE, 1 for R²).

The Evaluate Prior tolerance is a share of it. When no null score can be computed, the spread of the trials' scores is used.

### What SMAC minimizes

SMAC minimizes a cost. A higher-is-better metric with an upper end becomes best end − score (for example 1 − accuracy). A lower-is-better metric is used as it is.

## In Codesigner

- Dashboard sidebar, "Experiment Evaluation": "Evaluation metric" (`metric-select`) switches every figure. In the Run form it is also the metric the next run optimizes (`optimize_metric`).
- Picking a different metric than the one being optimized asks "Change the optimized metric?". The trials are kept and read again under the new metric.
- Figures use short titles ("ROC AUC", "RMSE", "R²", "sMAPE (%)"). The Trials table has a column per metric.

## Questions people ask

**Why is a failed trial's RMSE not zero?** Zero would be perfect. It gets the error of always predicting the average.

**Why can't I pick ROC AUC?** The task is regression, or the model gives no probabilities from its own fit.

**Must I choose the metric before running?** Every metric is computed anyway. The choice only decides what the search optimizes.

**Why is R² negative?** That is worse than always predicting the mean.
