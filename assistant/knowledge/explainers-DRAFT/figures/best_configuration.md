---
id: figure-best-configuration
title: Best configuration
kind: figure
about: [figure:best_configuration]
related: [figure-selected-configuration, figure-performance-over-time, concept-metrics, page-dashboard-sidebar]
keywords: [best trial, incumbent, best score, winning hyperparameters, best overall]
status: draft
sources:
  - ui/figures/catalog.py:27-32 (BestConfiguration)
  - ui/templates/ui/figures/best_configuration.html
  - ui/views.py:3643-3667 (_detail_context panels), 1995-2007 (_best_panels), 2054-2062 (_live_payloads best_html)
  - core/optimizers/base.py:199-228 (OptimizationResult.best_index)
  - ui/templates/ui/experiment_detail.html:3309-3318 (show), 3495-3507 (swapBestConfiguration)
  - config/figure_layout.toml (overview, misc)
---

## What it shows

The best trial so far for the metric you are viewing, as a card: which trial it was, its score, and the hyperparameter values it ran with. There is no plot. It is the configuration the search would hand back if it stopped now (the incumbent).

## How to read it

- **Caption**: "Best overall (Trial N)" — N is the trial's own number.
- **Big number**: the best score, with the metric name in brackets.
- **Table**: hyperparameter and value.

"Best" follows the metric's own direction: the highest accuracy, but the lowest error or RMSE. When two trials tie, the earlier one counts as best. A metric that no trial has a score for has no card.

This card always shows the best trial. To look at any other trial, click it in a figure; that appears in the sidebar's Selected Configuration panel, which starts on the same best trial.

## Controls

None. It follows the sidebar's **Evaluation metric** selector (`metric-select`): one card per metric, and only the viewed metric's card is shown.

## When it appears

- On the **Overview** and **Misc** tabs by default (Overview holds it as a pinned copy).
- Only once the experiment has at least one trial; before that the dashboard shows the optimizer settings and the prior instead.
- Switched off with "Best configuration" in the experiment's settings ("Figures shown on an experiment page").
- During a run it updates as trials arrive: the card's contents are replaced whenever a trial beats the current best, without a page reload.
- Nothing is computed on request; it needs no Compute button.

## Questions people ask

**Which hyperparameters gave the best result?** The table on this card lists them for the viewed metric.

**Why does the best trial change when I switch metric?** Each metric has its own best trial; an accuracy and a training time rarely agree.

**Is the best trial the same as the selected trial?** Only until you click another trial. This card does not follow clicks; Selected Configuration does.

**Why is lower shown as better?** For a lower-is-better metric (an error or a cost) the best trial is the one with the smallest score.
