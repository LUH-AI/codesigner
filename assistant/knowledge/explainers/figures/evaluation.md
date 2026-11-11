---
id: figure-evaluation
title: Experiment Evaluation
kind: figure
about: [figure:evaluation, element:metric-select, field:optimize_metric, field:max_trials, field:target_score, field:max_seconds, field:max_trial_seconds, field:no_improvement_trials, field:incumbent_confidence, field:max_failures, field:max_consecutive_failures, field:trial_timeout_mode, field:trial_timeout_seconds, field:trial_timeout_factor, field:next_run_settings, data-assist:run]
related: [page-dashboard, page-dashboard-sidebar, concept-metrics, concept-stopping, concept-optimizers, concept-permissions, figure-best-configuration]
keywords: [Experiment Evaluation, evaluation metric, run form, run, start a run, run button, next run, run until, stopping criteria, give up, failures, deadline, timeout, cancel run, give up on it, optimizer settings between runs]
status: draft
sources:
  - ui/figures/catalog.py (Evaluation)
  - ui/templates/ui/figures/evaluation.html
  - ui/templates/ui/_metric_select.html
  - ui/templates/ui/metric_change.html
  - ui/templates/ui/_run_status.html
  - ui/templates/ui/_optimizer_params.html
  - ui/templates/ui/experiment_prerun.html
  - ui/figure_layout.toml (overview, misc)
  - ui/run_form.py
  - ui/views/runs.py
  - ui/optimizer_form.py
  - experiments/services/run_logic.py
  - core/modelhost/deadline.py
  - core/optimizers/base.py
---

## What it shows

The card that starts a run: which metric the figures show and the run optimizes, when the run should end, when it should give up, and the **▶ Run** button. While a run is going it shows the run's status instead.

## How to read it

- **First column, Run until…** (fill at least one): **…this many trials have run** (`max_trials`, 30, or what is left of a planned budget), **…performance surpasses** (`target_score`, opens at the score there is to beat and follows the metric until you type), **…the run has taken this many seconds** (`max_seconds`), **…this many seconds have been spent in trials** (`max_trial_seconds`), **…this many trials in a row have not improved** (`no_improvement_trials`), and for SMAC **…the search is this sure nothing better remains** (`incumbent_confidence`, 0–1, the optimizer's own confidence, not a guarantee). The run ends at the first criterion reached.
- **Middle column, after the first run only**: the optimizer settings a later run may change (`next_run_settings`), such as the random trial rate and the initial design caps, all listed, the advanced ones included. They are saved when the run starts. The rest stay as the first run had them.
- **Last column**: **Evaluation metric** (`metric-select`), which every figure shows and the next run optimizes (`optimize_metric`); **Give up if…**: **…this many trials have failed** (`max_failures`, 1) and **…this many have failed in a row** (`max_consecutive_failures`, 15); **Give up on a model fit after…** (not for a model running in-process): **…a deadline set** (`trial_timeout_mode`: a fixed number of seconds, or a multiple of its predicted duration), **…this many seconds (0 for no limit)** (`trial_timeout_seconds`, normally 600), and in predicted mode **…this many times the fit's predicted duration** (`trial_timeout_factor`, 5); then **▶ Run** (`data-assist="run"`).
- The deadline is per model fit, so each fold gets it. A fit past it fails the trial, not the run.
- While a run is going: the metric selector, "Optimizing… (status)", **Cancel run**, then "Cancelling — it stops after the trial it is on." and **Give up on it**.

## Controls

- **▶ Run**: starts the run with the settings in the card. Choosing a different metric from the one being optimized first asks "Change the optimized metric?" (**Optimize *new***, **Keep optimizing *current***, **Cancel**).
- **Cancel run**: stops after the current trial and keeps the finished trials.
- **Give up on it**: only if cancelling does nothing; it does not keep unwritten work.

## When it appears

- Before the first trial: on the experiment's page, first, beside the Optimizer card and the Prior. Those three cards stay in place and keep their size.
- After the first trial: on the **Misc** tab, and pinned to **Overview** by default. It can be moved, resized, and unpinned from Overview like any figure.
- It follows nothing; the metric selector in it is what the other figures follow.
- During a run the form gives way to the run's status.
- Viewers, and experiments loaded without a dataset, get the metric selector only. No setting switches the card off.

## Questions people ask

**Where is the Run button?** In the Experiment Evaluation card: first on the page before the first run, and on Overview and Misc afterwards.

**Why did my run stop early?** The header's "Stopped because …" names the criterion reached.

**Can I run more trials later?** Yes. Press **▶ Run** again; the next run continues the same search from the trials already there.

**Can I change the optimizer between runs?** Only the settings in the card's middle column. The rest stay as the first run had them, since its trials were made with them.

**Does the assistant start runs?** No. It may fill in the card's fields, but only the person presses **▶ Run** or **Cancel run**.
