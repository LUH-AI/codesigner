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
the share cap is the real knob for the same idea.

---

## What became configurable

Eleven settings on `SMACOptimizer`. **Blank means the strategy's own default**,
the convention the stopping criteria already use — the two strategies disagree
about several of these and imposing one number on both would be worse than
silence. Each field's label is the setting's *name*; what it means is behind the
circled i beside it — a button, so it opens by keyboard and by tap as well as by
pointer, and the bubble it opens is what the field's `aria-describedby` points
at. Decimal fields are plain text inputs: `type="number"` puts arrows on them
that step by one, which on a fraction between zero and one is the whole range,
and which make the field look like it wants a whole number. A stepper now means
the field takes one.

### Always visible

| | |
|---|---|
| **Search strategy** | Gaussian process (default) or random forest — `BlackBoxFacade` / `HyperparameterOptimizationFacade`. |
| **Initial points** | Its own block — see below. |
| **Random trial rate** | How often to try a random configuration instead of the model's pick. |
| **Include the model's defaults** | `use_default_config` — a known reference point to beat. |

### Initial points

SMAC bounds the sampling phase two ways at once and uses the **smaller**:

```python
int(max(1, min(n_configs, max_ratio * n_trials)))
```

where `n_configs` defaults to ten per hyperparameter in the space. So "how many
initial points does a search use" has no single answer — which bound binds
depends on the model and the budget. A four-hyperparameter model at 30 trials
gets 7 (a quarter of the budget); the same model at 600 gets 40 (ten each), not
150.

Both bounds are exposed as caps, in a two-column block because they are
alternatives to each other:

| | |
|---|---|
| **Use share cap** / **Share cap** | `max_ratio`. Blank is SMAC's 0.25. |
| **Use trial cap** / **Trial cap** | `n_configs`. Blank is ten per hyperparameter. |
| **Use the larger of the two** | Off by default, matching SMAC. |

Either cap can be switched off. With **both** off nothing bounds the phase, which
is the whole budget: a search that only samples. Strange to ask for and legible,
so it is allowed rather than reinterpreted.

**The larger is ours, not SMAC's.** There is no maximum form of that expression.
It is reachable because `n_configs` overrides the per-hyperparameter bound and
`max_ratio=1.0` disables the clamp, so `SMACOptimizer._initial_points` works the
number out and hands it over. A test pins that with nothing configured the
number is still exactly what SMAC would have reached alone.

"Use default" beside each field is **not a setting**. An empty field already
means "SMAC's own", and a second thing recording the same fact could disagree
with it; it is a control over whether the field is filled in, and is never
submitted.

### What resuming does to it

The size is worked out from the budget *this run* was told about — trials
already done plus this run's cap. Nothing stores an intent, so:

- Resume with the rest of the original budget and the totals line up. Three then
  twenty-seven reaches the same seven sampled points as thirty in one go, because
  SMAC skips design configurations already in its runhistory and every past trial
  is replayed before anything is asked of it.
- Resume with **less** and the phase is sized smaller. Stop at 3 and resume for 5
  and SMAC is told the budget is 8, so a quarter of it is 2 — which the replayed
  trials nearly cover. You asked for a quarter and got an eighth.
- Either way it is **not the same experiment** as running straight through: the
  interrupted run sized its own phase for its own small budget, so its trials are
  not the first few of the long run.

`tests/core/test_resume_initial_points.py` pins all three. Giving the experiment
a stored intended budget to size against would change the second of them; it has
not been done.

One gap it also pins. Resuming rebuilds the facade and replays every past trial
into it with `tell`, and a told trial carries no origin — so SMAC's runhistory,
which the exported `.ihpo` copies verbatim, records the replayed ones as though
the model had chosen them. The trials, configurations and scores are intact;
only the label for how each was arrived at is not. Nothing downstream reads
`config_origins`, so it costs nothing today — it costs a reader of the file,
which is what the file is for.

### Advanced, folded away

Sampling method, acquisition function (EI or PI) and its improvement margin
(`xi`), candidates per trial, local search iterations, refit interval.

### The surrogate itself

Nine more, each declaring the strategy it belongs to (`OptimizerParam.
depends_on`) so the form shows only the set that applies. Hidden, not disabled:
a disabled field is not submitted and would reset on every strategy switch, and
the names are prefixed per strategy so there is nothing to gain by dropping them.

| Random forest | Gaussian process |
|---|---|
| trees, tree depth, split threshold, leaf size, feature ratio, bootstrapping | fitting (vanilla / MCMC), fit restarts, target normalisation |

The forest comes through `HyperparameterOptimizationFacade.get_model`, which
takes all six. The process does not: `BlackBoxFacade.get_model` exposes only
`model_type` and `kernel`, so restarts and normalisation mean constructing
`GaussianProcess` directly — with the facade's own kernel, and identical to what
the facade would have returned when nothing is set.

Three things worth knowing rather than discovering.

**The naming collision.** The demo Random Forest *model* is tuned over
`max_depth` and `min_samples_split`. The random forest *surrogate* has settings
of those exact names, and both appear on the same page. Hence `rf_`/`gp_`
prefixes on the schema names and the word **surrogate** in every label, which a
test enforces.

**`ratio_features` above 1.0** makes SMAC compute `max_features = 0` — a forest
whose every split considers no features, fitted and consulted and useless, with
no error. Capped in three places, because only the last covers a hand-edited
`.ihpo`.

**MCMC must be paired with `IntegratedAcquisitionFunction`.** An MCMC process is
an ensemble; nothing in SMAC pairs them up, and an unwrapped acquisition
function scores against one arbitrary member rather than complaining — it raises
only when there are no members at all. It costs roughly six times the wall clock
per trial here, which is why the choice is labelled "MCMC (slow)".

One thing that did **not** hold up. The plan expected more trees to fix the flat
confidence readings the forest gives. It moves them, but not reliably enough to
assert on: how coarse the forest's answer is depends on the split and the data
more than on the tree count. The knob is there and works; the claim about what
it buys is not pinned by a test.

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
Then the share cap moving that line, both strategies searching, both answering
the confidence criterion, and changed settings taking effect while the history
survives.

The arithmetic behind the two caps is pinned against the design object in
`tests/core/test_initial_points.py` rather than against a search: every
combination and edge is twenty questions, and twenty thirty-trial searches is
ten minutes to answer what a constructor already knows. One real search per cap
stays in the slow file, to check that the number the arithmetic produces is the
number of trials that actually get sampled.
`tests/core/test_resume_initial_points.py` covers what a stopped run does to the
sampling phase, and `tests/ui/settings/test_optimizer_settings.py` the panel,
clamping, the run-in-flight refusal, the alias and the withdrawn-setting case.

Live-checked: the create form and the settings page render all ten with the
advanced six folded, stored values come back selected, and a saved change
survives a reload.

## For A3S

The Gaussian process is the right default for the built-in models — four
continuous hyperparameters, short budgets. It is the wrong one for an
A3S-generated sklearn pipeline, which will have many more hyperparameters and
conditionals, and `BlackBoxFacade` refuses some spaces outright. The workspace
importer should set `search_strategy="rf"` on the experiments it creates.
