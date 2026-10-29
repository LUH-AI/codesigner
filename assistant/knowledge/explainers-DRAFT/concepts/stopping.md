---
id: concept-stopping
title: When a run stops
kind: concept
about: [field:max_trials, field:target_score, field:max_seconds, field:max_trial_seconds, field:no_improvement_trials, field:incumbent_confidence, field:max_failures, field:max_consecutive_failures, field:trial_timeout_mode, field:trial_timeout_seconds, field:trial_timeout_factor, data-assist:run]
related: [concept-metrics, concept-optimizers, concept-surrogate-and-acquisition, page-dashboard-sidebar, figure-performance-over-time, figure-trial-duration]
keywords: [stopping criteria, run until, trials, budget, target score, time limit, compute time, no improvement, confidence, nothing better remains, give up, failures, consecutive failures, deadline, timeout, model fit, predicted duration, cancel, stopped because]
status: draft
sources:
  - ui/templates/ui/_side_evaluation.html:39-213
  - ui/run_form.py:7-76
  - core/optimizers/base.py:274-346
  - core/optimizers/base.py:339-521
  - core/optimizers/smac_optimizer.py:118-126
  - core/optimizers/smac_optimizer.py:898-933
  - core/optimizers/smac_optimizer.py:1085-1087
  - core/modelhost/deadline.py
  - ui/optimizer_labels.py:7-35
  - ui/views/runs.py:168-173
  - experiments/services/run.py:248-334
  - experiments/services/run.py:502-514
  - config/settings.py:71
---

## In short

A run continues until one of the limits you set is reached. At least one limit must be something other than a failure limit, and the first limit to fire ends the run. Separately, a run gives up when trials keep failing, and each model fit can have a deadline.

## In detail

### "Run until…"

Empty means "do not stop for this". No single field is required, but at least one must be filled.

- **…this many trials have run** (`max_trials`, default 30): new trials in this run.
- **…performance surpasses** (`target_score`): the best score must strictly beat this, in the metric's direction (for RMSE, get under it). The field opens at the current best score and follows the metric selector until you type in it.
- **…the run has taken this many seconds** (`max_seconds`): wall-clock time.
- **…this many seconds have been spent in trials** (`max_trial_seconds`): time inside trials only.
- **…this many trials in a row have not improved** (`no_improvement_trials`).
- **…the search is this sure nothing better remains** (`incumbent_confidence`, 0–1, SMAC only). SMAC samples 1,000 configurations. Each x has chance Φ((μ(best) − μ(x)) / σ(x)) of beating the best, on costs. The confidence is 1 minus the largest chance. It is checked only once the run has 10 trials and a fitted model. It is confidence under the model's own assumptions, and a narrow optimum no sample lands near is missed.

With only a target or a confidence set, the form warns that these may never happen.

### "Give up if…"

These two do not count as the required limit.

- **…this many trials have failed** (`max_failures`, default 1). The first failure is usually the one worth reading. Raise this if you expect part of the space to break.
- **…this many have failed in a row** (`max_consecutive_failures`, default 15). Catches a model that cannot fit the data at all.

A trial fails when the model raises an error, misses its deadline, or returns something unscoreable. It is recorded with its reason, and a traceback where there is one, and gets the know-nothing score (see concept-metrics).

### "Give up on a model fit after…"

The deadline is per model fit. Under 5-fold cross-validation, each fold's fit gets it.

- **…a deadline set** (`trial_timeout_mode`): "a fixed number of seconds" or "a multiple of its predicted duration".
- **…this many seconds (0 for no limit)** (`trial_timeout_seconds`): it opens at the instance default (600 s unless configured), and it is also the upper bound in predicted mode.
- **…this many times the fit's predicted duration** (`trial_timeout_factor`, default 5).

Predicted mode works like this:

1. A random forest is fitted to how long this run's completed fits took.
2. Each fit is allowed factor × its prediction, at least 30 s and at most the fixed deadline.
3. Until 5 fits have completed, the fixed deadline applies.
4. Timed-out or crashed fits are never learned from.

A fit past its deadline is stopped and counts as a failed trial, and the run continues. Deadlines need the model to run in its own process. The fields are hidden when it runs inside Codesigner's process for lack of uv.
<!-- REVIEW: built-in models have env_status "none", so the deadline fieldset is shown for them, yet run.py imports built-ins in-process (`_model_launch` returns None) and no deadline is applied. Check whether the fields should be hidden, or the text changed, for built-in models. -->

### Other endings

- "every configuration had been tried": a small discrete space ran out.
- "it was interrupted": Cancel was pressed. Finished trials are kept. A run that stops answering can then be force-stopped, which loses what was not written.

## In Codesigner

- Dashboard sidebar, "Experiment Evaluation" → "Next run": the three fieldsets, then "▶ Run".
- The header then says "Last run: … Stopped because …", for example "the score had stopped improving".
- Only the person presses Run or Cancel. The assistant can fill the fields and point at the button.

## Questions people ask

**Why did my run stop after one trial?** By default the first failure ends it. Read the reason in the Trials table, or raise "…this many trials have failed".

**Is time in trials the same as run time?** No. It leaves out the optimizer's own work between trials.

**Why is the confidence option missing?** Only SMAC has a model to ask.

**Does the deadline cover the whole trial?** No, each model fit separately.
