---
id: figure-prior
title: Prior
kind: figure
about: [figure:prior, element:prior, element:evaluate_prior, element:prior-hp-select, element:acq-prior-kind, element:acq-prior-decay, element:acq-prior-beta, element:acq-prior-delay, element:acq-top-k, element:acq-evaluate, element:acq-reset]
related: [concept-priors, figure-acquisition-slice, figure-configuration-cube, page-dashboard-sidebar]
keywords: [prior, belief, where good values are, Normal, Beta, freeform, decay, beta over N, πBO, DynaBO, evaluate prior, reset prior, initial design]
status: draft
sources:
  - ui/figures/catalog.py:328-337 (Prior)
  - ui/templates/ui/figures/prior.html (assistant branch: data-assist="prior", "evaluate_prior"), _acquisition_shared.html
  - ui/static/ui/acquisition.js:89-132 (KINDS), 711-1019 (editing), 1021-1169 (controls), 1578-1792 (judge, reset, strength, syncControls)
  - ui/views/priors.py:186-316 (save_prior, reset_prior, evaluate_prior)
  - core/priors.py:99-161 (β ratio), 207-225 (_initial_design_notice)
  - ui/views/analytics.py:292-344 (_decayed)
  - ui/views/dashboard.py:210-227 (_prior_only_context)
  - ui/templates/ui/experiment_prerun.html, setup_step_priors.html
  - experiments/services/settings.py (prior_acceptance_tolerance 0.15, prior_acceptance_top_k 10)
---

## What it shows

A belief about where one hyperparameter's good values are, drawn as a curve. The optimizer multiplies its acquisition function by this curve (πBO-style), so values where the curve is high are tried more. A flat (Uniform) prior states nothing.

## How to read it

- **X axis**: the hyperparameter's values. The prior lives on the search's own scale, so on a log-scaled hyperparameter a Normal is a log-normal in real units.
- **Amber curve**: the prior, scaled to peak at 1. With a decay it is drawn as it counts now: flatter as the run goes on.
- **"counts at x"**: how much of the belief still counts (1 = in full, near 0 = nothing).

## Controls

- **Hyperparameter** (`prior-hp-select`), in step with the Acquisition function's picker.
- **Prior distribution** (`acq-prior-kind`): Uniform, Normal (μ, σ), Beta (α, β), Freeform. Categorical hyperparameters offer Uniform and Freeform only. Parameters have a number field and a slider.
- **Freeform**: click to add a point, drag to move, double-click a point to remove it, double-click elsewhere to withdraw.
- **Decay** (`acq-prior-decay`): No decay, Logarithmic, Linear, Quadratic, Cubic.
- **β / N** (`acq-prior-beta`): decay factor as a share of the trial budget (default 0.1); larger keeps the prior influential longer — better if it is right, worse if not.
- **Delay decay until after the initial design** (`acq-prior-delay`): shown while the initial design is not over.
- **Top k** (`acq-top-k`) and **Evaluate Prior** (`acq-evaluate`): asks the model whether the belief points somewhere promising (DynaBO's check). Needs the Gaussian process search strategy. A rejection asks whether to keep or discard it.
- **Reset** (`acq-reset`): back to what the last run searched under, or Uniform.

Every change saves as it is made; there is no Save button. Changing it needs edit rights.

## When it appears

- **Acquisition** tab; not per metric. Also before any trial: on a draft's "Priors & constraints" step and on an unrun experiment's page, alone (no acquisition curve without a surrogate).
- A caption warns while the initial design runs: those trials are drawn before any model exists, so a prior cannot affect them.
- Evaluate Prior's tolerance and default k are the settings "Prior acceptance tolerance" (0.15 of the metric's scale) and "Prior acceptance top k" (10).
- Switched off with "Prior" in settings.

## Questions people ask

**How do I tell the optimizer where good values are?** Choose Normal and set μ there; a smaller σ is a firmer belief.

**Did my prior do anything?** Look at the Acquisition function's solid and dotted sampled curves.

**Why did my first trials ignore the prior?** They belonged to the initial design.

**Why is Evaluate Prior greyed out?** The prior is Uniform, or the search strategy is not the Gaussian process.

