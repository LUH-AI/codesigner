---
id: page-dashboard
title: Experiment page (dashboard)
kind: page
about: [page:ui:experiment_detail, field:name, field:optimizer_name, element:export-model, element:export-trial-parameters, element:rename-name]
related: [page-dashboard-sidebar, page-experiment-settings, page-export-import, page-data-handling, page-timeline, page-setup-step-2, figure-prior, figure-best-configuration, figure-hyperparameter-importance, figure-trials, figure-performance-over-time, figure-configuration-cube, figure-forecast, concept-permissions, concept-optimizers]
keywords: [dashboard, experiment page, header, rename, export, settings, delete, ownership, tabs, Overview, HyperSHAP, Acquisition, Misc, pin, unpin, wider, narrower, drag, layout, pre-run, before first run, waiting for the first trial]
status: draft
sources: [ui/templates/ui/_experiment_page.html:1-14, ui/templates/ui/_experiment_header.html:1-133, ui/templates/ui/experiment_detail.html:1-75, ui/templates/ui/experiment_prerun.html:1-75, ui/views.py:376-391, ui/views.py:394-405, ui/views.py:272-283, ui/views.py:3374-3407, ui/views.py:3436-3479, ui/views.py:3482-3557, ui/figures/base.py:24-31, ui/figures/catalog.py:27-45, config/figure_layout.toml:20-55, ui/static/ui/layout.js:1-18, ui/static/ui/layout.js:560-680, ui/layout.py:325-385, ui/services/optimizer_form.py:14-21, access/policy.py:299-326, ui/context_processors.py:12-30]
---

## What it is for

An experiment's own page, at one address for its whole life. Before its first trial it shows what can still be decided: the optimizer and the prior. Once a trial exists it shows the figures that analyse the trials, on four tabs. The sidebar (its own explainer) holds the metric, the Run form and the selected trial.

## What is on it

**Header** (also on Data Handling and the timeline):
- **Experiment: *name***, and **✎** (tooltip "Rename") to rename it (`rename-name`, then **Save**). An empty name shows as "Untitled Experiment".
- A caption: **Model**, **Optimizer**, **Metric** ("~" before the first run, "Inconsistent" when the metric now optimized is not the one the results were produced with), **Evaluation** (e.g. "5-fold CV", "20% held out"), **Seed**, **ID**. After a run: "Last run: N trials in X s — Y s in trials, Z s overhead. Stopped because …".
- Buttons: **💾 Export** ("Download as .ihpo"), **📦 Export model** (`export-model`) and **🧮 Export trial parameters** (`export-trial-parameters`), both shown once there are trials and following the selected trial, **⚙ Settings**, **🗑 Delete**.
- An ownership line (with accounts), such as "Yours, and not shared. Share it from *group*", "Shared with you by *name* — read only." or "*name*'s, in a group you lead."
- Warnings when they apply: "Run failed: …", a custom model that may not be run, or the model's environment (being prepared, failed with **Try again**, or running in-process with **Prepare an isolated environment**).

**Before the first trial**: an **Optimizer** card (the same form as setup step 2, with **Save**) and the **Prior** card. While a run starts, "Waiting for the first trial…"; the page reloads itself when the first trials arrive.

**After the first trial**: the tabs **Overview**, **HyperSHAP**, **Acquisition** and **Misc**. Every figure lives on one of the last three. Overview holds the figures pinned to it; by default Best configuration, Hyperparameter importance, Trials, Trial performance, Forecast (forecasts only) and the Hyperparameter space projection. Each figure has controls: **⠿** ("Drag to move"), **📌** ("Pin to Overview" / "Remove from Overview"), **−** ("Narrower") and **+** ("Wider"). "The search space is exhausted at N trials." appears when every configuration has been tried.

## How to use it

1. Before running: check the optimizer (change it and press **Save** if needed), optionally state a prior, then use the Run form in the sidebar.
2. Switch tabs to see different figures. Pin a figure you use often to Overview with 📌.
3. Drag a figure by ⠿ to move it within its tab; use − and + to change its width (one, two or four columns, as the figure allows).

## Rules and limits

- Rename, and deciding who it is shared with: the owner only. Settings and changing the optimizer: owner, contributors and the group's leads. Delete: owner and contributors only. Export: anyone who can open it. Without accounts, everyone may do everything.
- The optimizer can be changed only before the first trial and not during a run.
- The arrangement of figures is yours alone: it is saved for you and this experiment and changes nobody else's page. Reset it, or make it your default, under ⚙ Settings → Figure layout. A figure cannot be dragged to another tab, only pinned to Overview.
- A figure switched off in the experiment's settings is not shown on any tab.
- Running, cancelling, deleting and sharing are done by the person, never by the assistant.

## Questions people ask

**Where are the figures?** They appear once the first trial exists.

**Why does Metric say "Inconsistent"?** The metric the experiment now optimizes is not the one its results were first produced with, because a later run changed it.

**How do I get my layout back to normal?** ⚙ Settings → **Reset layout**.

**Why can I not delete or rename it?** You are not its owner (rename), or neither owner nor contributor (delete).
