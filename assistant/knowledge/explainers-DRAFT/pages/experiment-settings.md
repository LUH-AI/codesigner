---
id: page-experiment-settings
title: Experiment settings
kind: page
about: [page:ui:experiment_settings, page:ui:default_experiment_settings, field:use_default_settings, field:ice_max_curves, field:local_effects_max_trials, field:prior_acceptance_tolerance, field:prior_acceptance_top_k, field:keep_best_trial_models, field:keep_last_run_trial_models, field:export_timestamps]
related: [page-dashboard, page-export-import, figure-partial-dependence, figure-local-effects, figure-prior, figure-hyperparameter-importance, concept-priors, concept-permissions]
keywords: [settings, experiment settings, show figure, hide figure, figures shown, autocompute, compute button, ICE curves, local effects, prior acceptance tolerance, top k, kept trial models, export timestamps, default settings, inherit defaults, reset layout, default layout]
status: draft
sources:
  - ui/templates/ui/experiment_settings.html
  - ui/templates/ui/_experiment_settings_fields.html
  - ui/templates/ui/default_experiment_settings.html
  - ui/templates/ui/save_as_default_confirm.html
  - ui/forms.py:373-433
  - experiments/services/settings.py
  - ui/views/preferences.py:18-39
  - ui/views/preferences.py:141-154
  - ui/views/dashboard.py:258-301
  - ui/figures/base.py:116-119 (deferred), 164-168 (autocompute_key)
  - ui/figures/catalog.py:399-421
  - ui/layout.py:190-247
  - ui/templates/ui/base.html:96-99
  - ui/templates/ui/base.html:197-208
  - ui/views/dashboard.py:112-132
---

## What it is for

What one experiment's page shows and computes, and what it keeps. Opened with **⚙ Settings** in the experiment's header ("Settings: *name*"). How the search runs is not here: the optimizer is set in setup or on the page before the first trial.

## What is on it

- **Use default experiment settings** (`use_default_settings`): while ticked, the experiment follows the instance's defaults and the fields below are greyed out.
- **Figures shown on an experiment page**: one checkbox per figure (`show_<figure>`), all on by default: Selected Configuration, Best configuration, Hyperparameter importance (HyperSHAP), Trial performance, the five Interactions figures, Trial duration, Forecast, Parallel coordinates, Partial dependence (PDP/ICE), Local explanation (selected trial), Local effects across trials, Acquisition function, Prior, Hyperparameter space projection, Trials. A figure switched off is not drawn and nothing is computed for it.
- **Trials drawn on the partial-dependence figure** (`ice_max_curves`, 100; 0 draws every trial).
- **Trials explained on the local-effects figure** (`local_effects_max_trials`, 100; 0 explains every trial).
- **Prior acceptance tolerance** (`prior_acceptance_tolerance`, 0.15, 0–1): the share of the metric's scale a prior may fall short by before **Evaluate Prior** calls it unhelpful.
- **Prior acceptance top k** (`prior_acceptance_top_k`, 10, 1–100): best configurations compared on each side.
- **Trial models kept for export** (`keep_best_trial_models`, 5, 0–100; 0 keeps none), and **Also keep every trial of the last run** (`keep_last_run_trial_models`, kept until the next run starts). These are what **🧮 Export trial parameters** can download.
- **Time** (`export_timestamps`): what the export page preselects: "The exporter's preference", Real Times, Relative Times or No Times.
- **Analytics computed as soon as a figure needs them** (`autocompute_<name>`): HyperSHAP against the incumbent or selected trial, and Acquisition and priors, are on by default. Forecast, Partial dependence (PDP/ICE), Local explanation (selected trial), Local effects across trials and Surrogate uncertainty (axes view) are off. Off means the figure shows a **Compute** button instead, so opening the page costs nothing until asked.
- Buttons: **Save settings**, **Reset to default**, and **Save settings as default** (only for people allowed to change the defaults; it asks "Make these the default settings?" first).
- **Figure layout**: **Reset layout**, **Make this my default layout**, **Reset my default to the site's**.
- **Change default experiment settings** link (same permission).

## How to use it

1. Untick **Use default experiment settings** to give this experiment its own settings.
2. Change what you need and press **Save settings**.
3. To follow the defaults again, press **Reset to default**.

## Rules and limits

- The page needs permission to change the experiment (owner, contributor or group lead).
- Numbers outside their range are clamped; unreadable ones fall back to the built-in default.
- Kept trial models take storage counted against the group. When the group is full, a new dataset replaces the stalest kept trial models. A change to how many are kept applies from the next run.
- With every HyperSHAP figure (importance and the five interactions) switched off, runs skip computing the explanations.
- The figure layout is yours: resetting it or setting your default changes nobody else's page.
- The defaults (Settings → **Default experiment settings**) apply to every experiment that inherits them, including existing ones. Changing them needs a separate permission. Only a person may change them; the assistant never does.

## Questions people ask

**How do I hide a figure?** Untick it under "Figures shown on an experiment page" and save.

**Why does a figure show a Compute button?** Its analysis is not computed automatically. Tick it under "Analytics computed as soon as a figure needs them" to change that.

**How do I undo my rearranging of the figures?** Press **Reset layout**.

<!-- REVIEW: save_as_default_confirm.html:16-18 re-posts every truthy setting as "on", so numeric settings (ICE count, tolerance, top k, kept models) and the export time choice look like they fall back to the built-in defaults when "Save settings as default" is confirmed. Check, and fix or describe. -->
