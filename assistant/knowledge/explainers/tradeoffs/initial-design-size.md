---
id: tradeoff-initial-design-size
title: How big the initial design is
kind: tradeoff
about: [field:opt_use_share_cap, field:opt_share_cap, field:opt_use_trial_cap, field:opt_trial_cap, field:opt_initial_points_use_max]
verdict: trade-off
related: [concept-initial-design, concept-priors, tradeoff-sampling-method, tradeoff-random-trial-rate]
keywords: [initial design, initial points, share cap, trial cap, use the larger, exploration, design size, continued run, resume, more design points, model-guided trials]
status: draft
sources:
  - core/optimizers/smac/facade.py (initial_points)
  - core/optimizers/smac/params.py (INITIAL_POINTS, RUN_SAFE)
  - ui/optimizer_labels.py (the caps' help)
  - tests/fixtures/make_wine_fixture.py (the wine-150 case)
  - COKT26
  - Sob67
---

## The trade-off

A larger initial design covers the space before the model guides anything. The surrogate starts from a truer picture, and the explanation figures (importance, partial dependence, coverage) rest on evenly spread trials. A smaller one leaves more trials to the model, and to priors, which cannot act during the design. The total number of trials a search needs is U-shaped in the design's size: too small and expected improvement spends its first guided trials on the edges and corners of the space, too large and trials go unguided. When unsure, err larger [COKT26]. By default the design is the smaller of a quarter of the budget and 10 per hyperparameter, so the quarter binds on budgets under 40 per hyperparameter.

## When to stray

- A budget of less than about 10 trials per hyperparameter, where a quarter is a handful of points (5 hyperparameters and 20 trials give 5): raise the share cap (0.4 gives 8), since too small a design costs more trials than it saves.
- The person wants to understand the space (which hyperparameters matter and how), not only the best score: raise the caps, or tick "Use the larger of the two", for more evenly spread trials.
- Noisy scores (a small dataset, few folds, scores that vary widely between folds): raise the caps; the surrogate needs more points to tell signal from noise.
- A trusted prior, or the model's defaults expected to be near the best, and the goal is the best score in few trials: lower the trial cap, so the model and the prior take over sooner.
- A continued run: the design is sized over every trial, old and new, so a longer total can draw more design points mid-search (150 planned, cancelled at 86, continued to 236: the design grows from 37 to 50, 13 more unguided trials). To let the model keep guiding, set the trial cap to the design size already drawn. To explore more first, leave the caps as they are.
- "Use the larger of the two": the design should grow with the budget and never fall below 10 per hyperparameter. On long budgets it costs many guided trials.
- Both caps off samples the whole budget, a quasi-random search: only as a baseline to compare against.

## Questions people ask

**Why not skip the design and let the model choose from the start?** With almost nothing to learn from, expected improvement goes for the space's edges and corners first, and the search needs more trials in total [COKT26].

**Why did my continued run start with unguided trials again?** The design is sized over all the trials, old and new, so a longer total can make it bigger. Setting the trial cap to the design already drawn stops that.

**Can the caps change after the first run?** Yes, on the run form's "Next run" fold; they are saved when the run starts.
