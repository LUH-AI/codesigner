# Making SMAC search, and letting users configure it

**Status:** implemented, awaiting your sign-off.
**You can now:** choose how the search models the objective and how much of the
budget it spends exploring first — and, more importantly, get a search at all.

Commit `11ba366`. It began as "which SMAC knobs should we expose" and turned up
a reason none of them would have mattered.

---

## The headline

**SMAC had never done Bayesian optimization here at any realistic run size.**

SMAC spends the first part of a run on an *initial design* — sampling the space
before it has a model worth asking — and sizes it as
`n_configs_per_hyperparameter × n_params`, capped at `max_ratio × n_trials`.
`Scenario.n_trials` was a fixed `_SMAC_MAX_TRIALS = 100_000`, so the cap was
25,000 and never applied. The design came out at 32 configurations for the
built-in Random Forest and 48 for the SVM:

| 30-trial run | Random Forest (4 HPs) | SVM (6 HPs) |
|---|---|---|
| before | 30× Sobol initial design | 30× Sobol initial design |
| after | 7 Sobol, **21 local search**, 2 random | 7 Sobol, **21 local search**, 2 random |

The Gaussian process was fitted every single iteration and asked for nothing.
What the app called SMAC was Sobol sampling with an expensive ornament.

The difference it makes, on wine at 30 trials:

```
random search   best accuracy = 0.6813
SMAC (gp)       best accuracy = 0.7188
SMAC (rf)       best accuracy = 0.6813
```

Before this, the first two lines were the same computation.

### Why the constant was there, and why it could go

Its comment: *"Fixed budget keeps the Scenario hash stable across runs."*
Resuming reused SMAC's own output directory with `overwrite=False`, which
requires the scenario — budget included — to hash identically between runs.

That requirement is gone. `_replay`, built for the metric-change fix, rebuilds a
facade and re-tells the whole history; it is `tell` calls with no evaluations
behind them, negligible beside one model fit. So every run now rebuilds, the
budget can be honest, and the `overwrite=False` path disappears — which is worth
having gone on its own account: on a mismatched directory SMAC asks the
**console** whether to continue (`smac/main/smbo.py:552`), and a worker with no
console waits forever.

When a run has no trial cap — a deadline or a target score instead — the budget
falls back to a documented constant. It only sizes the initial design, and
`exploration_ratio` is the real knob for the same idea.

---

## What became configurable

Eleven settings on `SMACOptimizer`. **Blank means the strategy's own default**,
the convention the stopping criteria already use — the two strategies disagree
about several of these and imposing one number on both would be worse than
silence. Each field's label is the setting's *name*; what it means is a tooltip
on that name, and a visually-hidden description for anyone not using a pointer.

### Always visible

| | |
|---|---|
| **Search strategy** | Gaussian process (default) or random forest — `BlackBoxFacade` / `HyperparameterOptimizationFacade`. |
| **Exploration share** | `max_ratio`: the fraction of the budget spent sampling before the model chooses. The knob the headline fix makes real. |
| **Exploration trials** | The same thing as a count — `n_configs`. Overrides the share, which otherwise clamps it back down; capped at the budget, less the default configuration if one was asked for, because SMAC raises on an initial design that does not fit. |
| **Random trial rate** | How often to try a random configuration instead of the model's pick. |
| **Include the model's defaults** | `use_default_config` — a known reference point to beat. |

### Advanced, folded away

Sampling method, acquisition function (EI or PI) and its improvement margin
(`xi`), candidates per trial, local search iterations, refit interval.

### Not exposed, deliberately

`walltime_limit` and friends duplicate the stopping criteria we already built —
one concept, one mechanism. `n_workers` is inert in ask/tell but still enters
the scenario hash, so it would look like it did something. Multi-fidelity
*raises* without budgets. `max_config_calls` and `n_seeds` collapse to one under
`deterministic=True`. `LCB`'s `beta` is not the coefficient it uses — that is
`2·log(D·t²/beta)` — and is not a number to put in front of anyone. The
runhistory encoder stays with the strategy that owns it, because mismatching it
with `EI(log=)` silently corrupts the acquisition.

---

## Where it lives

**The persistence half already existed.** `Experiment.optimizer_params`
round-trips through `.ihpo` and reconstructs via `opt_type(**params)`, proven by
Grid's `numeric_steps`. `OptimizerParam` was written in Step 2 to drive form
rendering and never wired up; `ui/forms.py` had said *"editing them is a later
step"* for eight steps. Only the UI was missing, and it is generic — the partial
loops `params_schema`, so a new setting is one line and no template changes.

**One deviation from the plan.** It said labels should be `gettext_lazy` on the
dataclass. They cannot be: `core` has no Django, deliberately. So the split
follows the stopping criteria instead — the vocabulary in `core`, the words in
`ui/optimizer_labels.py`, translated there.

**Editable between runs**, not fixed at creation like `cv_folds`. The test is
whether past trials stay comparable, and they do: the strategy chooses *which*
configurations to try, not *how* they are measured. Refused while a run is in
flight, and the page says the change applies from the next run onwards.

---

## Three things found while wiring it

**The confidence criterion would have died silently.** It asked the surrogate
for a standard deviation; `RandomForest._predict` **raises** on anything but
`"diagonal"`, and the call sits inside a bare `except` — so picking the random
forest would have switched the criterion off with no error and no output. It now
asks for variance and takes the root.

**And it was comparing across spaces.** The reference was the incumbent's
*measured* cost, but the random-forest strategy trains on log-scaled costs, so
the comparison was meaningless — it returned a flat 0.0 every time. It now uses
the incumbent's *predicted* cost, which keeps both sides inside whatever space
the model thinks in, and is how SMAC picks the reference for its own acquisition
function (`_get_x_best`).

Even so: the Gaussian process gives answers spanning 0.30–0.65 over a run, and
the random forest sits at exactly 0.5 throughout — its posterior is too flat at
these trial counts to carry information. Pinned as a test, so the criterion
being worth setting under `gp` and close to inert under `rf` is a known property
rather than a surprise.

**A withdrawn setting used to crash a page.** `opt_type(**stored)` raises
`TypeError` on an unknown keyword, and `_rebuild_result` catches only
`ValueError` — so renaming a parameter would 500 the detail page of every
experiment that stored it. `known_params()` filters against the schema first,
as the collector does for stopping keys.

---

## The rename

`SMAC (BlackBox)` was the name of the one facade it was nailed to, which is now
a setting. It is called **SMAC**. Migration `0013` updates stored rows;
`SMACOptimizer.aliases` resolves the old name, because an `.ihpo` on someone
else's disk cannot be migrated.

## Verify

```bash
python -m pytest -m "not slow and not uv"   # 583
python -m pytest -m slow
python -m pytest -m uv
python manage.py migrate                     # 0013
```

`tests/core/test_smac_configuration.py` leads with the regression: a 30-trial
run must contain more model-chosen configurations than initial-design ones.
Then `exploration_ratio` moving that line, both strategies searching, both
answering the confidence criterion, a resume not re-exploring, and changed
settings taking effect while the history survives.
`tests/ui/settings/test_optimizer_settings.py` covers the two forms, clamping,
the run-in-flight refusal, the alias, and the withdrawn-setting case.

Live-checked: the create form and the settings page render all ten with the
advanced six folded, stored values come back selected, and a saved change
survives a reload.

## For A3S

The Gaussian process is the right default for the built-in models — four
continuous hyperparameters, short budgets. It is the wrong one for an
A3S-generated sklearn pipeline, which will have many more hyperparameters and
conditionals, and `BlackBoxFacade` refuses some spaces outright. The workspace
importer should set `search_strategy="rf"` on the experiments it creates.
