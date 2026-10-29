---
id: concept-surrogate-and-acquisition
title: Surrogate and acquisition function
kind: concept
about: [figure:acquisition_slice, figure:configuration_cube, field:acq-show-slice, element:acq-next-trials, field:cube-next-trials, field:cube-layer, field:cube-layer-style, page:ui:run_proposal]
related: [concept-optimizers, concept-priors, concept-projection, concept-initial-design, concept-explanation-games, figure-acquisition-slice, figure-configuration-cube, figure-selected-configuration, figure-partial-dependence]
keywords: [surrogate, model of the objective, Gaussian process, random forest, uncertainty, spread, confidence, acquisition function, expected improvement, probability of improvement, lower confidence bound, EI, PI, LCB, next trials, requery, proposal, run as trial, prediction layer, acquisition layer, sample]
status: draft
sources:
  - core/analytics.py:73-94
  - core/analytics.py:115-127
  - core/optimizers/base.py:535-542
  - core/analytics.py:409-470
  - core/analytics.py:473-568
  - core/analytics.py:590-697
  - core/optimizers/smac_optimizer.py:211-233
  - core/optimizers/smac_optimizer.py:475-501
  - core/optimizers/smac_optimizer.py:573-600
  - ui/static/ui/improvement.js
  - ui/static/ui/acquisition.js:228-246
  - ui/timeline.py:352-377
  - ui/views/priors.py:61-93
  - ui/views/priors.py:124-171
  - ui/views/priors.py:319-352
  - ui/views/analytics.py:175-209
  - ui/views/analytics.py:256-302
  - ui/templates/ui/figures/acquisition_slice.html
  - ui/templates/ui/figures/configuration_cube.html
  - ui/templates/ui/figures/_next_trials.html
  - ui/templates/ui/_selected_proposal_inner.html
  - SMAC3/smac/acquisition/function/confidence_bound.py:20-60
---

## In short

The surrogate is a model of the score, fitted to the trials so far. For any configuration it predicts a mean and an uncertainty. The acquisition function turns those two numbers into how promising the configuration is. SMAC runs the configuration with the highest acquisition next, after weighting it by any stated prior.

## In detail

### The surrogate

SMAC's surrogate is a Gaussian process or a random forest, as set by the search strategy. The fitted model is not stored. When the page needs it, Codesigner refits the same kind of model with the run's settings on the recorded trials, so a Gaussian-process run is read through a Gaussian process. Random Search and Grid Search fit no model. For them, and for partial dependence, the uncertainty field and HyperSHAP, Codesigner fits a plain random forest with 100 trees.

**Uncertainty.**
- A Gaussian process gives a posterior standard deviation.
- A forest's uncertainty is how much its trees disagree. That is not a calibrated posterior, and it fails in one particular way: far from the trials, every tree lands in the same extreme leaf, so the forest looks confident exactly where there is no data.

A forest's prediction is also a staircase, not a smooth curve.

### The acquisition function

Write gain = μ − η when higher is better, or η − μ when lower is better. Here μ is the predicted mean, σ the uncertainty, η the best score observed so far, and ξ the "Improvement margin".

- **Expected improvement** (EI) [JSW98]: the expected amount by which the configuration beats η by more than ξ. EI = (gain − ξ)·Φ(z) + σ·φ(z), with z = (gain − ξ)/σ. It rewards a good mean and a large uncertainty.
- **Probability of improvement** (PI) [Kus64]: the chance of beating η by at least ξ, Φ((gain − ξ)/σ). It ignores by how much.
- **Lower confidence bound** (LCB): μ − √β·σ on costs, an optimistic reading of each configuration [SKKS10]. It is not offered as the search's acquisition function. Evaluate Prior uses it to score configurations.

A stated prior multiplies the acquisition: weighted = acquisition × prior density^exponent (see concept-priors). The prior does not change the surrogate's prediction.

### The prediction space sample

For the projection's layers, the acquisition figure's cloud and proposals, Codesigner draws 8,192 configurations evenly over the whole space (a scrambled Sobol sequence [Sob67]). It predicts all of them once per metric and the current trials. Every view places those same configurations.

### Next Trials and proposals

- **Next Trials**: rebuilds the run's own SMAC, replays every trial, applies the saved priors, and asks it for its next 12 configurations in rank order. Rank 1 "runs next". The path the local search climbed to each candidate is drawn too. This takes a few seconds and is not cached; **Requery** asks again after a prior changes. SMAC only.
- **Proposals**: clicking a sampled configuration in the projection's point-cloud layer selects it as "Proposal n of N". The selected-configuration panel then shows its predicted score ± uncertainty and its difference from the incumbent. **Run as Trial** runs it as a one-trial run, optimizing the experiment's current metric.

## In Codesigner

- "Acquisition function" figure (Acquisition tab): a hyperparameter picker; the curve over the sampled cloud; "Acq. w/ inc. HPs" (`acq-show-slice`), the slice with every other hyperparameter at the incumbent's value; "Next Trials" and "Requery"; and a "Compute" button when it waits to be asked.
- "Hyperparameter space projection": "Layer" (`cube-layer`: None, Prediction, Acquisition), drawn "as" (`cube-layer-style`: Scalar Field, Point Cloud), "Next Trials", "Requery". In HP Space with exactly two axes, "Show surrogate uncertainty" draws the forest's spread over that plane, with the other hyperparameters at the best trial's values.
- Only the person can press "Run as Trial".

## Questions people ask

**Why does the acquisition peak where the prediction is not best?** EI also rewards uncertainty. An unexplored region can be worth a trial.

**Can I trust the uncertainty band?** Under a Gaussian process it is a posterior. Under a forest it is tree disagreement, and it is overconfident far from the data.

**Why is the curve stepped?** The surrogate is a random forest.

**Why are Next Trials different after I change a prior?** The prior reweights the acquisition. Press Requery to ask again.

**Is a proposal's score measured?** No, it is a prediction until you run it.
<!-- REVIEW: LogEI [ADE+23] was in the key list but is not used: under the random-forest strategy SMAC computes EI on log costs (EI(log=True), citing HHLM09/Wat24). Decide whether to mention either. -->
