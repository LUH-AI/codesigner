---
id: concept-priors
title: Priors
kind: concept
about: [figure:prior, field:prior-hp-select, field:acq-prior-kind, field:acq-prior-decay, field:acq-prior-beta, field:acq-prior-delay, field:acq-top-k, assist:prior, assist:evaluate_prior, field:prior_acceptance_tolerance, field:prior_acceptance_top_k]
related: [concept-initial-design, concept-surrogate-and-acquisition, concept-projection-and-projected-priors, concept-metrics, concept-optimizers, figure-prior, figure-acquisition-slice, page-setup-step-4, page-experiment-settings]
keywords: [prior, belief, uniform, normal, beta, freeform, decay, logarithmic, linear, quadratic, cubic, beta over N, β, delay decay, reset, evaluate prior, accepted, rejected, unjudged, tolerance, top k, πBO, DynaBO]
status: draft
sources: [ui/templates/ui/figures/prior.html, ui/templates/ui/figures/_acquisition_shared.html, ui/views.py:942-1170, ui/views.py:1173-1260, ui/views.py:1571-1644, ui/services/priors.py:1-82, core/priors.py:1-234, core/optimizers/smac_optimizer.py:54-71, core/optimizers/smac_optimizer.py:339-345, core/optimizers/smac_optimizer.py:807-908, core/optimizers/smac_optimizer.py:934-1080, core/optimizers/smac_optimizer.py:1470-1535, ui/services/settings.py:32-46, ui/services/settings.py:71-78, ui/forms.py:404-410, ui/static/ui/acquisition.js:232-250, assistant/prompts.py:73-82, SMAC3/docs/advanced_usage/14_priors_and_constraints.md, SMAC3/smac/acquisition/weight/decay.py:65-157, SMAC3/smac/acquisition/weight/acceptance.py:270-328]
---

## In short

A prior is your belief about where good values of one hyperparameter lie. SMAC multiplies its acquisition function by the prior, so trials lean toward that region, and the influence fades as the surrogate learns [HSSL22]. A wrong belief costs some trials but can still be corrected. "Evaluate Prior" asks the run's own model whether the belief looks worth acting on.

## In detail

### Kinds

A prior is drawn on the hyperparameter's normalized 0–1 axis. For a log-scaled hyperparameter that axis is in log space, so a Normal there is log-normal in real units.

- **Uniform**: states nothing, and is stored as no prior.
- **Normal**: a mean and a standard deviation.
- **Beta**: two shape parameters.
- **Freeform**: a curve dragged through points. For a categorical hyperparameter the points sit on its choices.

### Decay

The search weights by density^exponent, where t is the number of trials since the belief was stated:

- No decay: exponent 1.
- Linear: β / (t + 1).
- Quadratic: β / (t + 1)².
- Cubic: β / (t + 1)³.
- Logarithmic: β / log(t + 2).

**β / N** gives β as a share of the run's budget N: β = ratio × N. It accepts 0.01–0.5 and opens on 0.1, the πBO and DynaBO value. DynaBO's measured values (0.02, 0.04, 0.1, 0.2, 0.4) are ticks [FWS+25]. A larger β keeps the prior influential longer, which helps an accurate prior and hurts a misleading one. N is the trials already run plus the run's trial limit, or plus 100 when there is no limit.

Example: β/N = 0.1 and N = 50 give β = 5. Under linear decay the exponent is 5 at t = 0 (sharper than drawn), 1 at t = 4, and 0.5 at t = 9.

"Delay decay until after the initial design" starts t when the design ends (see concept-initial-design). Editing parameters keeps the anchor; changing the kind restates the belief.
<!-- REVIEW: the figure counts t from the trial the belief was stated at (`at_trial`), and β from at_trial + the last run's limit. A run applies the prior after replaying the history, so SMAC anchors at the run's first trial and takes β from that run's own budget. These differ for a belief that stands across several runs. Check which one is intended before describing it. -->

### How SMAC uses priors

All stated beliefs become one prior over the whole space: per-hyperparameter tables multiplied with any projected priors. Before asking for the next configuration, SMAC registers it with `add_prior` and the chosen decay [FWS+25]. It weights the acquisition by the prior and also draws some candidates from it. Random Search and Grid Search ignore priors. Choosing a different model in step 1 withdraws them.

### Evaluate Prior

Evaluate Prior rebuilds the run's SMAC, replays the trials, and offers all stated beliefs to DynaBO's safeguard, `ClimbingComparisonPolicy`:

1. It draws configurations from the belief, and from a normal around the incumbent as wide as the belief.
2. It ranks each side by the surrogate's lower confidence bound, weighting the belief's side by the belief.
3. It climbs with local search from the best **Top k** on each side.
4. It scores where the climbs ended with the unweighted bound.

The belief is rejected if its mean is worse by more than:

margin = tolerance × the metric's scale

The tolerance is the "Prior acceptance tolerance" setting, default 0.15; for an accuracy the scale is 1. "Top k" defaults to "Prior acceptance top k" (10), and can be changed beside the button.

The verdict is one of:

- **accepted**: "This prior is potentially informative".
- **rejected**: "This prior is unlikely to be helpful", with a choice to "Keep the prior" or "Discard it". Nothing changes until you choose.
- **unjudged**: no trials, no scale, or nothing to compare. The belief stands as stated.

Evaluate Prior needs the Gaussian process strategy. The random forest is fitted to log-scaled costs, under which the bound is near zero everywhere and every prior would pass.

## In Codesigner

- The "Prior" figure, on the Acquisition tab and in setup step 4 "Priors & constraints", has:
  - a hyperparameter (`prior-hp-select`), the kind (`acq-prior-kind`), decay (`acq-prior-decay`) and "β / N" (`acq-prior-beta`);
  - the delay checkbox (`acq-prior-delay`);
  - "Top k", "Evaluate Prior", and "Reset", which goes back to what the last run searched under, or to uniform.
- A caption "counts at …" shows the current exponent.
- Changes save at once and need edit rights. Evaluate Prior needs compute rights. The assistant can point at these controls and explain them, but does not set priors.

## Questions people ask

**Will a prior force the search there?** No. It weights the acquisition, and the weight fades.

**Why is Evaluate Prior greyed out?** It needs SMAC with the Gaussian process strategy.

**Does it judge only the hyperparameter shown?** It judges all stated beliefs together.
