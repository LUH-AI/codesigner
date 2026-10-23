---
id: page-setup-step-4
title: Priors & constraints (step 4)
kind: page
about: [page:ui:setup_priors, data-assist:prior, data-assist:evaluate_prior, data-assist:create]
related: [page-setup-step-3, page-new-experiment, page-dashboard, figure-prior, concept-priors, concept-optimizers]
keywords: [priors, constraints, belief, optimum, create experiment, create, not ready, planned, optional step]
status: draft
sources: [ui/setup_views.py:96-122, ui/templates/ui/setup_step_priors.html:1-36, ui/services/setup.py:190-194, ui/services/setup.py:228-250, ui/services/drafts.py:143-152, ui/services/drafts.py:171-192, ui/views.py:3315-3340, ui/views.py:940-973, ui/templates/ui/figures/prior.html:1-30, ui/templates/ui/figures/prior.html:140-148, ui/services/drafts.py:333-335]
---

## What it is for

The last setup step. Here you can say what you believe about where the best values of each hyperparameter lie, before any trial, and then create the experiment. Stating a prior is optional; **Create experiment** is what turns the draft into an experiment.

## What is on it

- **Prior** card (`data-assist="prior"`): the same prior figure the experiment's page shows, as it looks before any run. Pick a hyperparameter, choose a distribution (Uniform, Normal, Beta, Freeform), set its parameters or drag the curve, and optionally a decay. **Evaluate Prior** (`data-assist="evaluate_prior"`) and **Reset** are part of the card. The figure's own explainer covers the controls.
  - If the model's search space is not known yet, the card is replaced by "This model's search space is not known yet, so there is nothing to state a belief about."
  - The card is missing if the Prior figure is switched off in the default experiment settings.
- **Constraints** card, marked **Planned**: "Bounds a configuration must keep to — a latency, a model size — which the search learns to stay inside." It has no controls yet.
- **Create experiment** (`data-assist="create"`). While something is missing it is greyed out, and the reason is shown under it: "Save step 1, Setup, first.", "Save step 2, Optimizer, first.", "Save step 3, Data, first.", "Step 1 is missing a dataset or a model." or "Step 2 is missing an optimizer."

## How to use it

1. Optionally, state a prior for any hyperparameter you have a belief about. Each change is kept as you make it; there is no separate Save button on this step.
2. Press **Create experiment**. The experiment's page opens, ready for its first run.

## Rules and limits

- Steps 1, 2 and 3 must each be saved before **Create experiment** works. This step itself never has to be saved; it shows "Optional" in the step bar.
- Creating starts the experiment's history. The timeline then shows how it was set up, the Data Handling choices, and each prior stated here.
- Saving step 1 again with a different model withdraws the priors stated here, since they described the old model's search space.
- Priors only steer SMAC; Random Search and Grid Search ignore them. **Evaluate Prior** needs SMAC with the Gaussian process search strategy (tooltip "Needs the Gaussian process search strategy.").
- Only the person presses **Create experiment**. The assistant may suggest priors but never creates the experiment.
- Priors can still be stated and changed after creation, on the experiment's page.

## Questions people ask

**Do I have to state a prior?** No. Without one the search treats every value as equally likely.

**Why is Create experiment greyed out?** One of steps 1 to 3 has not been saved. The message under the button names which.

**Can I change priors after creating?** Yes, on the Prior card of the experiment's page (Acquisition tab), before or between runs.

**What are constraints?** A planned feature. They cannot be set yet.

<!-- REVIEW: check what "Evaluate Prior" does on a draft with no trials (the view allows drafts, views.py:1022-1024); not described here. -->
