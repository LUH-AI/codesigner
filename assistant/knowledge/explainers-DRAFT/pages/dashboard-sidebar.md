---
id: page-dashboard-sidebar
title: Experiment page sidebar
kind: page
about: [element:metric-select, field:optimize_metric, field:max_trials, field:target_score, field:max_seconds, field:max_trial_seconds, field:no_improvement_trials, field:incumbent_confidence, field:max_failures, field:max_consecutive_failures, field:trial_timeout_mode, field:trial_timeout_seconds, field:trial_timeout_factor, data-assist:run, element:explanation-game, element:hypershap-baseline, element:hypershap-compute-btn, figure:selected_configuration]
related: [page-dashboard, page-export-import, figure-selected-configuration, figure-hyperparameter-importance, figure-configuration-cube, concept-metrics, concept-stopping, concept-explanation-games, concept-optimizers, concept-permissions]
keywords: [sidebar, Experiment Evaluation, evaluation metric, run form, next run, run until, stopping criteria, give up, failures, deadline, timeout, run, cancel run, give up on it, HyperSHAP, game, compare against, baseline, compute explanations, selected configuration, proposal, run as trial, export model]
status: draft
sources:
  - ui/templates/ui/_side_evaluation.html
  - ui/templates/ui/_metric_select.html
  - ui/templates/ui/metric_change.html
  - ui/templates/ui/_run_status.html
  - ui/templates/ui/_side_explanation.html
  - ui/templates/ui/figures/selected_configuration.html
  - ui/templates/ui/_selected_config_inner.html
  - ui/templates/ui/_selected_proposal_inner.html
  - ui/run_form.py:7-76
  - experiments/services/run_logic.py:7-17
  - ui/optimizer_labels.py:13-23
  - ui/views/priors.py:49-57
  - ui/views/analytics.py:306-346
  - ui/views/runs.py:25-71
  - ui/views/analytics.py:359-370
  - ui/views/runs.py:167-185
  - ui/figures/catalog.py:36-64
  - ui/templates/ui/experiment_detail.html:719-756
  - config/settings.py:71
  - core/modelhost/deadline.py:24
  - core/optimizers/base.py:270-275
---

## What it is for

The experiment page's sidebar holds the choices the figures answer to, and stays in view while you read them. Below Experiment Selection are three sections: **Experiment Evaluation**, **HyperSHAP** and **Selected Configuration**. Before the first trial, only Experiment Evaluation is there.

## What is on it

**Experiment Evaluation**
- **Evaluation metric** (`metric-select`): which metric every figure shows. It is also the metric the next run optimizes (`optimize_metric`).
- **Next run** (a fold, open until there are results):
  - **Run until…** (fill at least one): **…this many trials have run** (`max_trials`, 30), **…performance surpasses** (`target_score`, opens at the score there is to beat and follows the metric until you type), **…the run has taken this many seconds** (`max_seconds`), **…this many seconds have been spent in trials** (`max_trial_seconds`), **…this many trials in a row have not improved** (`no_improvement_trials`), and for SMAC **…the search is this sure nothing better remains** (`incumbent_confidence`, 0–1, the optimizer's own confidence, not a guarantee). The run ends at the first criterion reached.
  - **Give up if…**: **…this many trials have failed** (`max_failures`, 1) and **…this many have failed in a row** (`max_consecutive_failures`, 15).
  - **Give up on a model fit after…** (not for a model running in-process): **…a deadline set** (`trial_timeout_mode`: a fixed number of seconds, or a multiple of its predicted duration), **…this many seconds (0 for no limit)** (`trial_timeout_seconds`, the instance's default, normally 600), and in predicted mode **…this many times the fit's predicted duration** (`trial_timeout_factor`, 5). The limit is per model fit, so each fold gets it. A fit past it fails the trial, not the run.
  - **▶ Run** (`data-assist="run"`).
- While a run is going, a status box replaces the fold: "Optimizing… (status)", **Cancel run**, then "Cancelling — it stops after the trial it is on." and **Give up on it**.

**HyperSHAP**
- **Game** (`explanation-game`): **Tunability**, **Mistunability** or **Sensitivity**. All six HyperSHAP figures answer the one chosen.
- **Compare against** (`hypershap-baseline`): **Default**, **Incumbent** (the default choice) or **Selected**: the configuration the figures measure gains from.
- **Compute** (`hypershap-compute-btn`): shown when that measurement waits to be asked for.
- **Compute explanations**: shown when the run stored none (a cancelled or imported run).

**Selected Configuration**
- A trial: "Trial N — click any point on the graph to select", its score, its difference from the best ("(Incumbent)" for the best), and its hyperparameter values. A failed trial says its score is a placeholder, with the reason and **Full traceback →**.
- A proposal picked in the space projection: tagged **Proposal**, its predicted score ± uncertainty, its difference "vs. incumbent", and **Run as Trial**.

## How to use it

1. Choose the metric. The figures redraw at once.
2. Open **Next run**, set when it should end, press **▶ Run**. Choosing a different metric from the one being optimized first asks "Change the optimized metric?" (**Optimize *new***, **Keep optimizing *current***, **Cancel**).
3. To stop: **Cancel run** keeps the finished trials. Use **Give up on it** only if cancelling does nothing; it does not keep unwritten work.
4. Click a point in any trial figure to fill Selected Configuration. 📦 **Export model** and 🧮 **Export trial parameters** in the header then refer to that trial.

## Rules and limits

- A run needs at least one **Run until…** criterion; the give-up limits do not count. Otherwise: "Set at least one stopping criterion, so the run has something to end on."
- Running, cancelling, **Run as Trial** and **Compute explanations** need permission to run (owner, contributor or group lead). Viewers, and experiments loaded without a dataset, get the metric selector only.
- Starting or stopping a run is the person's alone. The assistant may fill in the run form but never presses **Run** or **Cancel run**.
- **Run as Trial** runs one trial on that configuration, optimizing the current metric.

## Questions people ask

**Why did my run stop early?** The header's "Stopped because …" names the criterion reached.

**Why is Export trial parameters greyed out?** That trial's fitted model was not kept (the tooltip says why), or a proposal is selected, which has no model.

**Can I run more trials later?** Yes. The next run continues the same search from the trials already there.
