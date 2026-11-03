---
id: concept-optimizers
title: Optimizers
kind: concept
about: [field:optimizer_name, field:opt_search_strategy, field:opt_random_probability, field:opt_use_default_config, field:opt_initial_design, field:opt_acquisition, field:opt_acquisition_xi, field:opt_challengers, field:opt_local_search_iterations, field:opt_retrain_after, field:opt_rf_trees, field:opt_gp_model_type, field:opt_numeric_steps]
related: [concept-initial-design, concept-surrogate-and-acquisition, concept-priors, concept-stopping, page-setup-step-2, page-dashboard]
keywords: [optimizer, SMAC, Bayesian optimization, Gaussian process, random forest, search strategy, acquisition function, expected improvement, probability of improvement, improvement margin, random trial rate, candidates per trial, local search, refit interval, MCMC, random search, grid search, grid steps]
status: draft
sources:
  - core/registry.py:41-45
  - core/optimizers/smac/params.py:9-27
  - core/optimizers/smac/params.py:44-54
  - core/optimizers/smac/optimizer.py:33-300
  - core/optimizers/smac/facade.py:184-228
  - core/optimizers/smac/facade.py:228-286
  - core/optimizers/random_optimizer.py
  - core/optimizers/grid_optimizer.py:24-125
  - core/optimizers/base.py:566-570
  - ui/optimizer_labels.py:29-198
  - ui/optimizer_form.py
  - ui/templates/ui/_optimizer_params.html:49-52
  - SMAC3/smac/facade/blackbox_facade.py
  - SMAC3/smac/facade/hyperparameter_optimization_facade.py
  - SMAC3/smac/facade/abstract_facade.py:511
---

## In short

The optimizer decides which configuration each trial runs. Codesigner offers SMAC, Random Search and Grid Search. SMAC fits a model of the score to the trials so far and picks the next configuration from it. Random Search and Grid Search choose without looking at any results.

## In detail

### SMAC

SMAC [LEF+22, HHL11] is Bayesian optimization. After an initial design (see concept-initial-design), each trial fits the surrogate, scores candidates with the acquisition function, and runs the best one (see concept-surrogate-and-acquisition). A blank setting takes SMAC's default for the chosen strategy, shown as the placeholder.

- **Search strategy** (`search_strategy`). Gaussian process (`gp`, the default) suits a few numeric hyperparameters. Random forest (`rf`) [Bre01] suits many hyperparameters, or categorical ones. The choice also changes several defaults below.
- **Initial points** group: "Use share cap", "Share cap", "Use trial cap", "Trial cap", "Use the larger of the two" (see concept-initial-design).
- **Random trial rate** (`random_probability`, 0–1). How often a trial is drawn at random instead of from the model, once the model has taken over. Blank means about 0.084 under the Gaussian process and 0.2 under the forest.
- **Include the model's defaults** (`use_default_config`). Spends one exploration trial on the model's default configuration, a reference point to beat.

Under "Advanced search settings":

- **Sampling method** (`initial_design`). Sobol (default) [Sob67], Latin hypercube [MBC79], Uniformly random, or Defaults only.
- **Acquisition function** (`acquisition`). Expected improvement (`ei`, default) [JSW98] weighs a candidate by how much better it might be. Probability of improvement (`pi`) [Kus64] counts only whether it might be better.
- **Improvement margin** (`acquisition_xi`, 0–1, default 0). How much better than the best so far a candidate must promise to be. Higher explores more.
- **Candidates per trial** (`challengers`). How many configurations the model scores before picking one. Blank means 1,000 under the Gaussian process and 10,000 under the forest.
- **Local search iterations** (`local_search_iterations`, blank 10). How far the search climbs from each promising candidate.
- **Refit interval** (`retrain_after`). Trials between refits of the surrogate. Blank means every trial under the Gaussian process and every eighth under the forest.
- Forest only: "Surrogate trees" (`rf_trees`, blank 10), depth, split threshold, leaf size, feature ratio, and "Bootstrap the surrogate's trees". With bootstrapping off, the trees agree more and the reported confidence narrows.
- Gaussian process only: "Surrogate fitting" (`gp_model_type`). Vanilla fits once by maximum likelihood. "MCMC (slow)" samples the kernel's hyperparameters and averages over them: better calibrated, about ten times slower. Also fit restarts and target normalization.

The "Surrogate…" settings belong to SMAC's own model, not to the model being tuned. Under the forest strategy, SMAC fits log-transformed costs and computes expected improvement in log space [HHLM09].

### Random Search

Random Search [BB12] draws each configuration uniformly from the search space. It has no settings, skips repeats, and stops after 200 repeats in a row.

### Grid Search

Grid Search tries every combination of a fixed set of values, in order. "Grid steps" (`numeric_steps`, 2–50, default 5) is the number of values per numeric hyperparameter, evenly spaced between its bounds. An integer hyperparameter with fewer values than that gets all of them, and a categorical one gets every choice. Combinations multiply. A resumed run skips points already run, and the search is complete when the grid is used up.
<!-- REVIEW: the grid is evenly spaced on the native scale even for log-scaled hyperparameters (np.linspace on lower/upper). Confirm that is intended before stating it. -->

### What only SMAC can do

Only SMAC fits a surrogate, so only SMAC offers:

- the "…the search is this sure nothing better remains" stopping criterion;
- the acquisition figure, Next Trials, and the projection's Acquisition layer;
- acting on priors. Random and Grid Search accept priors and ignore them.

"Evaluate Prior" (`judges_priors`) works only under the Gaussian process strategy. The forest is fitted to log-scaled costs, which makes every prior pass.

## In Codesigner

- Setup step 2 ("Optimizer"): "Optimizer" (`optimizer_name`) and one settings panel per optimizer, with fields named `opt_<setting>`.
- On the dashboard the same form is available until the experiment has trials. After that the optimizer is fixed, because the next run continues the same search.

## Questions people ask

**Which optimizer should I pick?** SMAC, unless you want an unguided baseline (Random Search) or an exhaustive sweep of a small space (Grid Search).

**Why can't I change the optimizer any more?** The experiment has trials. The next run continues their search.
