---
id: page-setup-step-2
title: Optimizer (step 2)
kind: page
about: [page:ui:setup_optimizer, field:optimizer_name, field:opt_search_strategy, field:opt_use_share_cap, field:opt_share_cap, field:opt_use_trial_cap, field:opt_trial_cap, field:opt_initial_points_use_max, field:opt_random_probability, field:opt_use_default_config, field:opt_numeric_steps, data-assist:save]
related: [page-setup-step-1, page-setup-step-3, page-dashboard, concept-optimizers, concept-priors, concept-stopping]
keywords: [optimizer, SMAC, Random Search, Grid Search, search strategy, Gaussian process, random forest, initial points, share cap, trial cap, random trial rate, model defaults, advanced search settings, acquisition function, surrogate, grid steps]
status: draft
sources:
  - ui/templates/ui/setup_step_optimizer.html
  - ui/templates/ui/_optimizer_form.html
  - ui/templates/ui/_optimizer_params.html
  - ui/templates/ui/_group_initial_points.html
  - ui/templates/ui/_use_default.html
  - ui/optimizer_labels.py:29-203
  - core/optimizers/smac/facade.py:264-285
  - core/optimizers/grid_optimizer.py:32-36
  - core/optimizers/random_optimizer.py:18-19
  - core/registry.py:41-45
  - ui/optimizer_form.py
  - ui/views/setup.py:57-69
---

## What it is for

The second setup step chooses what searches the model's hyperparameters, and with what settings. The same form appears on an experiment's page until its first trial exists.

## What is on it

- **Optimizer** (`optimizer_name`): **SMAC** (the first offered, so the default), **Random Search** or **Grid Search**. Changing it swaps the settings panel below without reloading.
- **Configure optimizer**: the chosen optimizer's settings. Each label has an ⓘ with its meaning and accepted range; an empty field shows in grey what it would use.
  - **SMAC**:
    - **Search strategy** (`opt_search_strategy`): Gaussian process (default) suits a few numeric hyperparameters; Random forest suits many, or categorical ones.
    - **Initial points**: how many configurations are sampled before the model takes over. Two caps side by side. **Use share cap** (`opt_use_share_cap`) with **Share cap** (`opt_share_cap`, 0.01–1, empty means 0.25), a fraction of the run's trials. **Use trial cap** (`opt_use_trial_cap`) with **Trial cap** (`opt_trial_cap`, 1–10,000, empty means 10 per hyperparameter). Each cap has a **Use default** box that empties its field. SMAC uses the smaller of the two caps; **Use the larger of the two** (`opt_initial_points_use_max`) takes the larger instead.
    - **Random trial rate** (`opt_random_probability`, 0–1): how often a trial is drawn at random once the model has taken over.
    - **Include the model's defaults** (`opt_use_default_config`): spends one exploration trial on the model's own default configuration.
    - **Advanced search settings** (folded): **Sampling method** (Sobol, Latin hypercube, Uniformly random, Defaults only), **Acquisition function** (Expected improvement, Probability of improvement), **Improvement margin**, **Candidates per trial**, **Local search iterations**, **Refit interval**. Under Random forest also **Surrogate trees**, **Surrogate tree depth**, **Surrogate split threshold**, **Surrogate leaf size**, **Surrogate feature ratio** and **Bootstrap the surrogate's trees**. Under Gaussian process also **Surrogate fitting** (Vanilla, MCMC (slow)), **Surrogate fit restarts** and **Normalise surrogate targets**. Settings for the strategy not chosen are hidden.
  - **Grid Search**: **Grid steps** (`opt_numeric_steps`, 2–50, default 5), the values tried per numeric hyperparameter. The combinations multiply.
  - **Random Search**: no settings.
- **Save and continue** (`data-assist="save"`).

## How to use it

1. Pick the optimizer. For most experiments, leave SMAC on its defaults.
2. Change a setting only if you have a reason; an empty field leaves it to SMAC's default for the chosen strategy.
3. Press **Save and continue** to go to step 3, Data.

## Rules and limits

- Step 1 must be saved first; otherwise this step sends you back there.
- The optimizer and its settings can change until the experiment has its first trial, here or on the experiment's page. After that they are fixed ("This experiment has trials, and the next run continues their search."), and also while a run is under way.
- Priors only steer SMAC. Random Search and Grid Search ignore them. Judging a prior with "Evaluate Prior" needs SMAC with the Gaussian process strategy.
- The "…the search is this sure nothing better remains" stopping criterion is offered only for SMAC.
- Changing the optimizer of a created experiment is recorded on its timeline.

## Questions people ask

**Which optimizer should I use?** SMAC fits a model of the results and uses it to choose the next trial. Random Search samples independently. Grid Search tries evenly spaced values in order.

**What does an empty field mean?** Use the strategy's own default. The grey placeholder shows it.

**Can I switch to Grid Search after running?** No. Once there are trials the search that found them is fixed, and the next run continues it.
