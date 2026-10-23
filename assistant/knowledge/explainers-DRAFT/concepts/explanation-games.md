---
id: concept-explanation-games
title: HyperSHAP explanation games
kind: concept
about: [field:explanation-game, field:hypershap-baseline, figure:hyperparameter_importance, figure:local_explanation, figure:local_effects, figure:interactions_heatmap, figure:interactions_top_pairs, figure:interactions_graph, figure:interactions_coalitions, figure:interactions_orders]
related: [concept-surrogate-and-acquisition, concept-deferred-computation, figure-hyperparameter-importance, figure-local-explanation, figure-local-effects, figure-interactions-heatmap, figure-interactions-top-pairs, figure-interactions-graph, figure-interactions-coalitions, figure-interactions-orders, page-dashboard-sidebar]
keywords: [HyperSHAP, Shapley value, explanation game, tunability, sensitivity, mistunability, importance, interactions, Möbius, Harsanyi dividend, FSII, ablation, local explanation, local effects, baseline, compare against, default, incumbent, selected]
status: draft
sources: [core/optimizers/base.py:1376-1396, core/optimizers/base.py:1472-1511, core/optimizers/base.py:1587-1664, core/optimizers/base.py:1666-1724, core/optimizers/base.py:2291-2396, core/optimizers/base.py:2398-2565, core/optimizers/base.py:2567-2581, core/optimizers/base.py:775-803, ui/figures/catalog.py:48-249, ui/figures/catalog.py:538-560, ui/templates/ui/_side_explanation.html, ui/templates/ui/figures/hyperparameter_importance.html, ui/templates/ui/figures/local_explanation.html, ui/views.py:716-770, ui/views.py:1715-1755, ui/views.py:1855-1903, ui/views.py:2124-2190, ui/views.py:2980-3028, ui/services/settings.py:29-65, .venv hypershap/games/tunability.py:82-180, .venv hypershap/utils.py:84-150, .venv hypershap/games/ablation.py:34-105]
---

## In short

HyperSHAP [WMFL25] treats a finished search as a cooperative game between the hyperparameters and splits the outcome between them with Shapley values [Sha53]. The game decides the question: tunability, mistunability and sensitivity explain the whole search; the local explanation explains one trial.

## In detail

### How a game is played

A random forest (100 trees, seeded by the experiment) is fitted to the trials' configurations and scores. The scores are oriented so that higher is better. To value a coalition of hyperparameters, HyperSHAP draws 10,000 random configurations. In each, the coalition's hyperparameters vary and every other hyperparameter is held at the baseline configuration. The forest predicts a score for each one, and the game reduces those predictions to a single number. The computation uses shapiq [MBF+24].

### The three global games

- **Tunability**: "How much is there to gain by tuning this hyperparameter?" Uses the best of the predictions (maximum).
- **Mistunability**: "How much is there to lose by getting this hyperparameter wrong?" Uses the worst (minimum).
- **Sensitivity**: "How much does performance move as this hyperparameter moves?" Uses their variance.

The importance figure shows each hyperparameter's order-1 value as a share of the total, so the shares sum to 100%. The raw tunability total is kept as well, in the metric's units.

### Interactions

Each game is played once, in its Möbius representation [Gra16], which gives every coalition of every size a value (its Harsanyi dividend). The order-2 Faithful Shapley Interaction Index (FSII) [TYR23] is derived from it.

- "Interactions: heatmap" and "Interactions: top pairs" read the FSII pairs. The heatmap's diagonal is each hyperparameter's own signed value.
- "Interactions: graph", "Interactions: coalitions" and "Interactions: by order" read the Möbius terms, the only reading that shows effects above order 2.

### The baseline: "Compare against"

Every game is a gain over a baseline configuration (`hypershap-baseline`):

- **Default**: the search space's default configuration. These games are computed at the end of each run and stored.
- **Incumbent**: the best trial for the viewed metric. The page opens on this one.
- **Selected**: the trial selected on the page.

Incumbent and Selected are computed on request. The baseline applies to every HyperSHAP figure, local ones included.

### Local explanation and local effects

- **Local explanation** plays the ablation game [FH16] for the selected trial (the metric's best trial until one is selected). A waterfall builds its score up from the baseline, one hyperparameter at a time. Values are signed: positive means this trial's value beat the baseline's. A trial is not compared with itself ("This trial is the baseline.").
- **Local effects across trials** plays it for many trials and draws each hyperparameter's effects as a beeswarm. A tight cluster means the hyperparameter acts the same everywhere; points spread across zero mean it helps in some regions and hurts in others. Up to "Trials explained on the local-effects figure" trials (default 100) are sampled evenly through the run.

### Cost and fallbacks

Each game evaluates 2^(number of hyperparameters) coalitions per metric. At the end of a run they are skipped if the run was cancelled, if their figures are all off, or if they would exceed the instance's budget (see concept-deferred-computation). "Compute explanations" fills them in later. With fewer than two trials the figure shows uniform weights. If HyperSHAP fails, the forest's feature importances are shown instead. If every contribution is zero, HyperSHAP "found no signal", which usually means the objective is saturated.

## In Codesigner

- Dashboard sidebar, "HyperSHAP" section: "Game" (`explanation-game`), "Compare against" (`hypershap-baseline`), a "Compute" button when the baseline waits to be asked, and "Compute explanations" when a result has none.
- On the "HyperSHAP" tab: "Hyperparameter importance (HyperSHAP)" (Pie, Bar or Table, and "vs" to overlay another baseline), the five interaction figures, "Local explanation (selected trial)" and "Local effects across trials".

## Questions people ask

**Tunability says 80% for one hyperparameter. Is that 80% accuracy?** No. It is that hyperparameter's share of the total attributed gain.

**Why do Default and Incumbent give different answers?** They measure gains from different starting configurations.

**What does a negative bar in the local explanation mean?** That hyperparameter's value in this trial did worse than the baseline's.

**How reliable is this?** It explains a surrogate fitted to the trials, so it is as good as their coverage.
