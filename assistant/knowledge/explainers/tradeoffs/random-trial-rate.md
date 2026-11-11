---
id: tradeoff-random-trial-rate
title: How often a trial is random
kind: tradeoff
about: [field:opt_random_probability]
verdict: trade-off
related: [concept-optimizers, concept-priors, tradeoff-search-strategy, tradeoff-initial-design-size]
keywords: [random trial rate, random probability, interleaved random configurations, exploration, stuck search, random search baseline, prior protection]
status: draft
sources:
  - core/optimizers/smac/params.py (FACADE_DEFAULTS)
  - core/optimizers/smac/priors.py (a prior weights the acquisition only)
  - HHL11
  - BBL+23
---

## The trade-off

After the design, each trial is drawn at random with this probability instead of from the model. Random trials keep the search from settling on a region the surrogate is wrongly sure of [HHL11, BBL+23]. They also ignore priors, which protects against a wrong prior. Each one is a trial the model did not choose. The defaults are about 0.084 under the Gaussian process and 0.2 under the forest, where the forest's overconfidence far from the trials makes random trials matter more.

## When to stray

- A small budget of expensive trials under the Gaussian process, on a space of a few numeric hyperparameters: lower it, down to 0, so every trial after the design is guided.
- A trusted prior: lower it, since random trials ignore the prior. A doubtful prior: keep it, or raise it.
- For a later run, when the best has not changed for many trials and the model's picks cluster around it: raise it (0.3, say) to look elsewhere.
- The person wants the explanation figures to cover the whole space: raise it.
- A later run meant as a random-search baseline on the same experiment: 1.0.
- Under the forest, lowering it is riskier than under the Gaussian process: the forest is most overconfident where there are no trials.

## Questions people ask

**Why does a model-guided search run random trials at all?** The model can be confidently wrong about regions it has not seen; an occasional random trial checks them.

**Does a random trial follow my prior?** No. A prior weights the model's choice, and a random trial skips the model.

**Can I change the rate between runs?** Yes, in the middle column of the Experiment Evaluation card.
