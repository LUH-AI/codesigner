# Trade-off explainers (draft, in use)

Each file weighs one choice whose default is worth questioning: what each side
gains and costs, and the circumstances that call for moving it.

- `search-strategy.md` (tradeoff-search-strategy): Gaussian process or random forest.
- `initial-design-size.md` (tradeoff-initial-design-size): the share cap, the trial cap and taking the larger; continued runs.
- `random-trial-rate.md` (tradeoff-random-trial-rate): how often a trial skips the model.
- `models-defaults.md` (tradeoff-models-defaults): one trial at the model's defaults, as a reference.
- `sampling-method.md` (tradeoff-sampling-method): Sobol, Latin hypercube, random, defaults only.
- `acquisition-function.md` (tradeoff-acquisition-function): expected or probable improvement.
- `improvement-margin.md` (tradeoff-improvement-margin): the margin, and its units under each strategy.
- `candidate-search.md` (tradeoff-candidate-search): candidates per trial and local search.
- `refit-interval.md` (tradeoff-refit-interval): trials between refits of the surrogate.
- `surrogate-trees.md` (tradeoff-surrogate-trees): the search's forest size.
- `forest-shape.md` (tradeoff-forest-shape): depth, split, leaf, feature ratio, bootstrapping.
- `gp-fitting.md` (tradeoff-gp-fitting): vanilla or MCMC, and fit restarts.
- `gp-normalisation.md` (tradeoff-gp-normalisation): normalising the scores.
- `optimized-metric.md` (tradeoff-optimized-metric): which metric the search optimizes.
- `validation-method.md` (tradeoff-validation-method): folds, a held-out share or backtests.
