---
id: concept-initial-design
title: Initial design
kind: concept
about: [field:opt_use_share_cap, field:opt_share_cap, field:opt_use_trial_cap, field:opt_trial_cap, field:opt_initial_points_use_max, field:opt_initial_design, field:opt_use_default_config, field:acq-prior-delay]
related: [concept-optimizers, concept-priors, concept-projection, concept-surrogate-and-acquisition, figure-prior, page-setup-step-2]
keywords: [initial design, initial points, exploration, share cap, trial cap, Sobol, Latin hypercube, random, defaults only, model-free, delay decay, prior cannot affect]
status: draft
sources:
  - core/optimizers/smac_optimizer.py:54-96
  - core/optimizers/smac_optimizer.py:104-132
  - core/optimizers/smac_optimizer.py:390-426
  - core/optimizers/smac_optimizer.py:761-783
  - core/optimizers/smac_optimizer.py:998-1063
  - ui/optimizer_labels.py:35-74
  - ui/optimizer_labels.py:179-193
  - core/priors.py:207-225
  - core/priors.py:228-236
  - ui/views/analytics.py:292-329
  - experiments/services/priors.py:85-91
  - ui/templates/ui/figures/prior.html
  - ui/templates/ui/figures/_acquisition_shared.html
  - ui/static/ui/acquisition.js:618-630
  - SMAC3/smac/initial_design/abstract_initial_design.py:28-45
---

## In short

The initial design is the set of configurations SMAC runs before it has a model of the score. They are spread evenly over the search space so the surrogate has something to learn from. Since no surrogate exists yet, nothing that weights the acquisition function can affect these trials, priors included.

## In detail

### Its size

SMAC bounds the number of initial points in two ways and uses the smaller:

- **Share cap** (`share_cap`, placeholder 0.25): a fraction of the run's trial budget. A longer run explores for longer.
- **Trial cap** (`trial_cap`, placeholder "10 per hyperparameter"): a fixed count.

Each cap has a switch ("Use share cap", "Use trial cap"). With both on, "Use the larger of the two" (`initial_points_use_max`) takes the larger instead. With both off, nothing bounds the design and the whole budget is sampled. The size is always at least 1 and at most the budget, less one if the model's defaults are included.

The budget is the run's trial limit plus the trials already run. A run with no trial limit (bounded by time or a target score) uses 100 as the budget for this purpose.

Example: a 4-hyperparameter model and a 30-trial first run give share cap 0.25 × 30 = 7 and trial cap 10 × 4 = 40. The design is 7 trials.

"Include the model's defaults" (`use_default_config`) adds the model's default configuration to the same model-free phase.

### How it is drawn

"Sampling method" (`initial_design`) offers:

- Sobol (default) [Sob67] and Latin hypercube [MBC79], which cover the space more evenly than chance;
- Uniformly random;
- Defaults only, a single configuration.

When a stopped experiment is run again, a Sobol, random or defaults design continues where it left off, because a bigger draw contains the smaller one. A Latin hypercube keeps the size it first had, because changing the size moves every point.

### Why priors cannot reach it

A prior multiplies the acquisition function, and the acquisition function is only consulted once a surrogate exists. The initial design comes before that.

The prior card says so while a run is inside its design: "Trial {trials} of {size} in the initial design ({name}). These are drawn before any surrogate is fitted, so a prior set here cannot affect them — it starts to count once the design is exhausted." The size is read from the design SMAC recorded, never counted from the trials. Random Search, Grid Search and imported runs without that record have no design, and show no notice.

### Delay decay

A decaying prior fades from the trial it was stated at. Stated before or during the design, it would spend the design fading while doing nothing, and reach the first trial it can influence already weakened.

"Delay decay until after the initial design" makes the belief wait. The search applies it only once the design's trials have run, so it arrives at full strength on the first trial it can influence. Until then the card reports it as not yet counting.

The checkbox is on by default. It is shown only while the experiment has no trials yet or is still inside its design, and only for a non-uniform prior. Stated after the design, the flag is not kept. All beliefs form one prior applied at one moment, so if any asks to wait, all wait.

## In Codesigner

- Setup step 2 and the dashboard's optimizer form, SMAC panel: the "Initial points" group, plus "Sampling method" and "Include the model's defaults".
- The "Prior" figure: the notice above the plot (`acq-initial`) and the checkbox (`acq-prior-delay`).

## Questions people ask

**Why did my prior change nothing in the first trials?** Those trials were the initial design, drawn before any model existed.

**How many initial trials will I get?** The smaller of share cap × budget and the trial cap, unless "Use the larger of the two" is on.

**Should I keep "Delay decay" on?** Usually yes. Otherwise a decaying prior loses strength during trials it cannot affect.

**Can I skip it?** Set a small trial cap; at least one configuration remains.
