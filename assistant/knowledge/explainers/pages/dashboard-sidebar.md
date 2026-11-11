---
id: page-dashboard-sidebar
title: Experiment page sidebar
kind: page
about: [element:explanation-game, element:hypershap-baseline, element:hypershap-compute-btn, figure:selected_configuration]
related: [page-dashboard, figure-evaluation, page-export-import, figure-selected-configuration, figure-hyperparameter-importance, figure-configuration-cube, concept-metrics, concept-stopping, concept-explanation-games, concept-optimizers, concept-permissions]
keywords: [sidebar, HyperSHAP, game, compare against, baseline, compute explanations, selected configuration, proposal, run as trial, export model]
status: draft
sources:
  - ui/templates/ui/_side_explanation.html
  - ui/templates/ui/figures/selected_configuration.html
  - ui/templates/ui/_selected_config_inner.html
  - ui/templates/ui/_selected_proposal_inner.html
  - ui/run_form.py:6-66
  - experiments/services/run_logic.py:7-17
  - ui/optimizer_labels.py:12-22
  - ui/views/priors.py:49-57
  - ui/views/analytics.py:306-346
  - ui/views/runs.py:25-67
  - ui/views/analytics.py:359-370
  - ui/views/runs.py:154-171
  - ui/figures/catalog.py:47-75
  - ui/templates/ui/experiment_detail.html:716-753
  - config/settings.py:62
  - core/modelhost/deadline.py:24
  - core/optimizers/base.py:242-247
---

## What it is for

The experiment page's sidebar holds the choices the figures answer to, and stays in view while you read them. Below Experiment Selection are two sections: **HyperSHAP** and **Selected Configuration**. Before the first trial neither is there. The metric selector and the Run form are the **Experiment Evaluation** card (its own explainer), not the sidebar.

## What is on it

**HyperSHAP**
- **Game** (`explanation-game`): **Tunability**, **Mistunability** or **Sensitivity**. All six HyperSHAP figures answer the one chosen.
- **Compare against** (`hypershap-baseline`): **Default**, **Incumbent** (the default choice) or **Selected**: the configuration the figures measure gains from.
- **Compute** (`hypershap-compute-btn`): shown when that measurement waits to be asked for.
- **Compute explanations**: shown when the run stored none (a cancelled or imported run).

**Selected Configuration**
- A trial: "Trial N — click any point on the graph to select", its score, its difference from the best ("(Incumbent)" for the best), and its hyperparameter values. A failed trial says its score is a placeholder, with the reason and **Full traceback →**.
- A proposal picked in the space projection: tagged **Proposal**, its predicted score ± uncertainty, its difference "vs. incumbent", and **Run as Trial**.

## How to use it

1. Choose the game and what to compare against. The HyperSHAP figures redraw.
2. Click a point in any trial figure to fill Selected Configuration. 📦 **Export model** and 🧮 **Export trial parameters** in the header then refer to that trial.

## Rules and limits

- **Run as Trial** and **Compute explanations** need permission to run (owner, contributor or group lead).
- **Run as Trial** runs one trial on that configuration, optimizing the current metric.

## Questions people ask

**Where did the Run button go?** It is in the Experiment Evaluation card, on Overview and Misc, and first on the page before the first run.

**Why is Export trial parameters greyed out?** That trial's fitted model was not kept (the tooltip says why), or a proposal is selected, which has no model.

**Why is HyperSHAP missing from the sidebar?** It appears once the experiment has trials to explain.
