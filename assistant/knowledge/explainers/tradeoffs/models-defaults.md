---
id: tradeoff-models-defaults
title: Including the model's defaults
kind: tradeoff
about: [field:opt_use_default_config]
verdict: change is usually better
related: [concept-initial-design, concept-tasks-and-models, tradeoff-initial-design-size]
keywords: [model's defaults, default configuration, baseline, reference point, out of the box, improvement over defaults]
status: draft
sources:
  - core/optimizers/smac/facade.py (initial_points: one trial added)
  - core/models/lightgbm_model.py (a space's defaults)
  - ui/optimizer_labels.py (use_default_config's help)
---

## The trade-off

"Include the model's defaults" adds one trial at the configuration the model's search space gives as default, on top of the design. It costs one trial. It gives the run a reference point, how much the search gained over the defaults, and often a good early best. It is off by default.

## When to stray

- A built-in model: turn it on. Its space's defaults are a sensible configuration, for the boosting models the library's own.
- Leave it off for a custom model whose defaults are placeholders, or when the defaults were measured elsewhere and the person does not need them in this experiment.

## Questions people ask

**Are these the library's defaults?** For LightGBM, yes; for some models the space sets its own sensible defaults, such as a depth limit for the random forest.

**Does it replace a design point?** No, it is one trial on top of the design.

**How do I see what tuning gained?** Compare the best trial with the defaults' trial in the trials table.
