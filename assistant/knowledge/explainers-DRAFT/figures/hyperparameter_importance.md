---
id: figure-hyperparameter-importance
title: Hyperparameter importance (HyperSHAP)
kind: figure
about: [figure:hyperparameter_importance, element:explanation-game, element:hypershap-baseline, element:hypershap-compute-btn, element:view-select-hyperparameter_importance-rendering, element:importance-comparison]
related: [concept-explanation-games, page-dashboard-sidebar, figure-interactions-heatmap, figure-local-explanation, figure-parallel-coordinates, figure-partial-dependence, concept-metrics]
keywords: [importance, which hyperparameter matters, tunability, mistunability, sensitivity, Shapley, HyperSHAP, pie, bar, baseline, compute explanations]
status: draft
sources:
  - core/optimizers/base.py:81-166 (HP_GAME_FIELDS, HP_GAME_LABELS, HP_GAME_HELP, HyperparameterImportance)
  - ui/templates/ui/figures/hyperparameter_importance.html, _importance_table.html, _before_this_run.html
  - ui/templates/ui/_side_explanation.html
  - ui/figures/plots.py:225-250 (hyperparameter_importance_plot)
  - ui/templates/ui/experiment_detail.html:1062-1171, 1257-1429, 3373-3475
  - ui/views/analytics.py:306-346 (hypershap_figures), 349-370 (_analytics_absent, experiment_compute_analytics), 403-416 (TUNING_PROGRESS_ENABLED)
  - core/analytics.py:21-31 (HP_GAMES), 234-256 (budget), 712-762 (compute_hp_games), 789-856 (_compute_hp_game)
  - experiments/services/settings.py (autocompute_hypershap_baseline default on, GAME_DISPLAY_FIGURES)
---

## What it shows

How much each hyperparameter contributed, according to HyperSHAP, which treats the search as a cooperative game between hyperparameters and splits the outcome the way Shapley values split a payoff. The shares add up to 100%. The sidebar's game decides the question.

## How to read it

- **Tunability**: how much there is to gain by tuning this hyperparameter.
- **Mistunability**: how much there is to lose by getting it wrong.
- **Sensitivity**: how much performance moves as it moves.

A bigger slice or bar is a bigger share of the game's total. Shares are relative: they do not say how much of the metric the whole is worth. Every share is a gain measured from one configuration, the **Compare against** baseline, so rankings can differ between baselines. Each share is the magnitude of HyperSHAP's order-1 FSII value.

## Controls

- Sidebar **Game** (`explanation-game`): Tunability, Mistunability, Sensitivity. Moves this figure and all five interaction figures together.
- Sidebar **Compare against** (`hypershap-baseline`): **Default** (the search space's default configuration), **Incumbent** (the best trial; the page opens on this), **Selected** (follows your clicks).
- Rendering select (`view-select-hyperparameter_importance-rendering`): **Pie**, **Bar** (largest first), **Table** (one column each for Default, Incumbent and Selected, side by side).
- **vs** radios (`importance-comparison`), Bar only: overlays a second baseline's bars as narrower bars in front. The sidebar's own baseline is not offered.

<!-- REVIEW: a "Still to gain" checkbox and Achievable/Banked/Still to gain table columns exist in the template but are switched off (TUNING_PROGRESS_ENABLED = False in ui/views.py); left out here. -->

## When it appears

- **HyperSHAP** tab, also pinned to **Overview**. Follows the metric selector.
- Default numbers are computed when a run finishes. Incumbent and Selected are computed on request, automatically unless "HyperSHAP against the incumbent or selected trial" is unticked in settings; then the sidebar shows a **Compute** button (`hypershap-compute-btn`).
- During a run: "Waiting for run to finish…", showing the trials before it.
- "Press Compute explanations to see results." means the explanations are missing (cancelled or imported run); the sidebar's **Compute explanations** button computes them.
- A warning explains fallbacks: too few trials (uniform weights), HyperSHAP failing (surrogate feature importances), no signal, or skipped because the model has too many hyperparameters for the budget (cost grows as 2^hyperparameters).
- Not computed when this figure and all five interaction figures are switched off.

## Questions people ask

**Which hyperparameter mattered most?** The largest share under Tunability against Default; against Incumbent, gains are measured from the best trial instead.

**Why does every hyperparameter have the same share?** The warning says why: too few trials, or HyperSHAP and its fallback both failed. Equal shares then mean "could not tell".

**Why is it empty after I cancelled a run?** A cancelled run skips the explanations. Press Compute explanations, or resume the run.

**Does a small share mean it is useless?** Not necessarily: it may matter only together with others (see Interactions: by order).
