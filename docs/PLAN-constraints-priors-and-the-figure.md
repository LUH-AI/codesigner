# Constraints, priors, benchmarking and the interactive figure

## Context

Six tasks across three repos. A survey of all three found that four are done,
one is nearly done with a single missing link, and one is half-built.

| # | Task | Repo | Actual state |
|---|---|---|---|
| 1 | Output constraints | SMAC3 | **Done**, refactored into a general weight layer |
| 2 | DynaBO / piBO | SMAC3 | **Done** — all three mechanisms, docs, example, ~3000 lines of tests |
| 3 | Benchmarking | CARP-S | Benchmarks built and verified; **one wiring gap**; never run |
| 4 | Auth layer | codesigner | **Done** |
| 5 | Remote job management | codesigner | **Done** |
| 6 | Interactive figure | codesigner | Maths done; **no way to state a belief**; no constraints |

Two corrections to the starting assumptions:

- **DynaBO is implemented**, on `origin/feature/output-constraints`, not on the
  locally checked-out `angrimson-output-constraints` (which is 18 commits
  behind). `PriorEnsemble` sums priors each decayed by its own age,
  `SamplingPool` draws candidates from priors (finally wiring up the long-dead
  `prior_sampling_fraction` argument), and `IncumbentComparisonPolicy` is an
  opt-in hard reject. The vocabulary is "weight"/"prior"/"anchor", not "dynabo".
- **The SMAC/CARP-S constraint channel already works.** Commit `0f4f5e447`
  made `SMBO.tell` call `extract_constraint_values` on `additional_info`, so
  ask-and-tell callers get the same treatment as the runner path. CARP-S
  already emits named floats there, so it needs **no** SMAC-specific code for
  constraint transfer.

Agnosticism requirements hold today and must survive: nothing in `carps/`
imports a SMAC constraint type, and SMAC has no knowledge of codesigner or
CARP-S. The one unavoidably SMAC-specific piece (Phase 2) lives in
`carps/optimizers/smac20.py`, which is the SMAC adapter and the right home.

## Phase 0 — Verify the branch, then consolidate (gate)

Nothing below should be built on unverified ground: **the 18 commits' tests
have never been executed on this machine.** The survey could only read them.

1. `git worktree add` (or check out) `feature/output-constraints` and run the
   full suite. Expect the 9 known pre-existing failures — 8 `test_phvi_*` and
   `test_mo_local_search_sort_keys`, all `ImportError: pymoo is required`. Any
   other failure is a real regression and blocks everything else.
2. Pay particular attention to `tests/test_priors/`, `tests/test_acquisition/`
   and `tests/test_constraints/`, which are the new coverage.
3. **Reconcile the two branches.** `angrimson-output-constraints` is a strict
   subset (`0 18`) and is stale. Either fast-forward it to
   `feature/output-constraints` or retire it, so there is one branch to pin.

Two soft spots the survey flagged, worth a decision now rather than a surprise
during benchmarking:

- `IncumbentComparisonPolicy`'s `threshold=-0.15` is in raw objective units and
  calibrated for a `[0,1]` objective. On CARP-S's synthetic problems the
  objective ranges differ per problem (Gardner 1 reaches −1.89, Branin 0.40),
  so a fixed threshold will behave inconsistently across the suite.
- `PriorEnsemble` sums, so several stale near-1.0 priors can drown a fresh
  sharp one. `prune_exponent` and `combination="max"` exist but are off.

## Phase 1 — The interactive figure

Currently three stacked panels sharing a hyperparameter axis
(`ui/figures/plots.py::acquisition_slice_plot`, drawn by
`ui/static/ui/acquisition.js`). The middle panel is permanently empty and the
two acquisition curves are identical, because `state.belief` is initialised to
`null` and never assigned.

### 1a. Per-panel legends (small, do first)

`plots.py` sets one horizontal legend at `y=1.01` for the whole figure, so
traces from all three panels land in one strip and nothing says which panel a
name belongs to. Plotly ≥5.15 supports multiple legends and plotly 6 is
vendored: give each panel its own (`legend`, `legend2`, `legend3`), anchored to
that panel's y-domain, and assign each trace with `legend=`.

### 1b. A belief the user can state

The missing input. Everything downstream already exists — `density()`,
the `acq * (w + 1e-12)^exponent` weighting mirroring SMAC's own, and
`impliedMean()`, a bisection inverting EI to produce the implied-performance
ghost.

Draw on the **middle** panel: a bell curve with a draggable centre and width.
`dragmode` and `fixedrange` are already disabled across the figure
specifically so pointer events are free and a pixel can be converted to a value
from the bounding box alone — that groundwork was laid for this.

Note the axis subtlety, already handled server-side: `positions` are ConfigSpace
*normalized* coordinates because a prior is a density in that space, while
`labels` carry native values as tick text. Drag in normalized space.

### 1c. A fourth panel — the constraint surrogate

**Constrainable outputs are concrete.** `TrialResult` (`core/optimizers/base.py:58`)
carries `scores: Dict[str, float]` — *every* metric, not just the primary one —
and a `duration` property (`run_info["time"]`). So the selector offers the
experiment's other metrics plus trial duration.

The panel is structurally a second copy of the top panel: a surrogate slice of
a measured quantity with mean and band, in that quantity's own units. Fit it
with the `values=` hook that already exists on `fit_surrogate` and
`compute_incumbent_slice`, whose docstring names exactly this use
(`lambda t: t.duration`). Nothing currently passes `values=`.

**The two interactions are deliberately different**, which is the asymmetry to
design around rather than paper over:

| | Belief panel | Constraint panel |
|---|---|---|
| Object | density over the hyperparameter axis | surrogate of a measured output |
| Units | none | seconds, or the metric's own |
| Interaction | drag a bump (centre + width) | drag a horizontal bound line |
| Effect on acquisition | `× prior^decay` | `× P(value ≤ bound)` |

Both feed the same weighted acquisition, which is what SMAC now does — priors
and feasibility compose in one `WeightedAcquisitionFunction`.

### 1d. Fidelity — use SMAC's real acquisition

The figure currently reconstructs EI in JavaScript over codesigner's own
scikit-learn random forest, so the curve is not the one SMAC maximized: plain
EI with `xi` hard-coded to 0, and tree disagreement instead of a posterior.

**Do not move the computation server-side.** The existing design deliberately
keeps EI in the browser, and the reason still holds: the belief is dragged
client-side and the server never sees it, so a server-side acquisition means a
network round trip per drag frame. Fix the *inputs and the formula* instead:

- source `mu`/`sigma` from SMAC's own surrogate rather than a fresh RF
- match the run's configured acquisition (`acquisition`, `acquisition_xi` are
  already optimizer settings; `PI` is offered and unimplemented in the figure)
- keep evaluation in the browser so dragging stays interactive

This requires re-pinning codesigner from stock `smac>=2.4.1` to the branch from
Phase 0. Flag: the `values=` groundwork and the RF's honest caveat (tree
disagreement reports *confidence* far from data, where there is none) both stay
relevant for runs whose surrogate cannot be rebuilt.

## Phase 2 — CARP-S benchmarking

The benchmark side is done and numerically verified: four synthetic problems
(`carps/objective_functions/constrained_synthetic.py`) whose constrained optima
were independently reproduced to ≤4.2e-4, plus 15 pymoo constrained tasks.
Constraint values already flow as named floats through `additional_info`.

**The one gap.** `carps/optimizers/smac20.py:248` builds
`Scenario(**scenario_kwargs)` from `carps/configs/optimizer/smac20/base.yaml`,
which has no `constraints` key. The `constraints:` block in all 19 new task
configs is inert, so a run launched today silently completes as ordinary
unconstrained BO.

Fix with a **new** `carps/configs/optimizer/smac20/constrained.yaml` rather
than touching `base.yaml` — the key exists on only 19 of ~6300 task configs,
so a `${constraints}` interpolation in the base would break every other task.

Also needed:

- **Cluster environment.** Per the decision to run on Slurm rather than
  locally: an environment with CARP-S installed and SMAC pinned at the
  consolidated branch from Phase 0. Note CARP-S lists **no** SMAC dependency at
  all (only the container recipe does, as `smac>=2.1.0`), so the pin has to be
  explicit. `pymoo` is already a hard dependency, and both new suites are pure
  Python — no container or data download.
- **Feasibility analysis.** `constrained_optimum` and `constrained_argmin` are
  written into the YAMLs and **read by nothing**. Nothing computes feasibility,
  a feasible-best trajectory, or a gap against the known optimum — which is
  the actual output of the benchmark. Add it in `carps/analysis/`, reading
  `trial_value__additional_info`.
- Watch `gather_data_utils.py:882` / `concat_rundata.py:44`: the parquet write
  now receives a dict-typed column for yahpo/mfpbench/pymoo runs that never had
  one. Runs with different key sets concatenated together are an untested path.

## Phase 3 — Deferred: steering live runs

Not in scope now. A stated belief currently changes only the chart. Making it
reach a running optimizer via `smac.add_prior` additionally needs belief
persistence and a control channel to Slurm jobs — the cluster module only
stages, submits, polls and touches a `CANCEL` file, so a `PRIOR` file would ride
the same mechanism. Worth choosing the belief data model in 1b with this in
mind, so steering is later wiring rather than a rewrite.

## Small fix, unrelated but cheap

`run.sh` exports `ALLOW_CUSTOM_MODELS=False` unconditionally ("off here
regardless of `.env`") but sets `REQUIRE_LOGIN` only inside the `auth` branch,
so a stale `.env` value silently turns the login wall on in every mode —
including `--demo`, whose warning text then lies. Mirror the sibling toggle:

```sh
export REQUIRE_LOGIN=False
if [ "$MODE" = auth ]; then export REQUIRE_LOGIN=True; fi
```

## Verification

- **Phase 0:** full SMAC suite on the consolidated branch; only the 9 known
  pymoo-import failures. This is the gate.
- **Phase 1a:** visual — each panel's legend names only its own traces.
- **Phase 1b/1c:** the existing `test_the_skeleton_leaves_the_belief_traces_empty`
  asserts the belief traces come back `None`; it should be replaced by tests
  that a stated belief produces non-empty traces and moves the weighted
  argmax. Note there is **no JS test runner** in the project, so
  `expectedImprovement`, `impliedMean` and `density` are currently untested by
  anything but source-text assertions — worth adding one.
- **Phase 1d:** on a run with a rebuildable surrogate, the figure's acquisition
  argmax should match the configuration SMAC actually chose next.
- **Phase 2:** one constrained run end to end on the cluster
  (`+task=ConstrainedSynthetic/branin_disk +optimizer/smac20=constrained`),
  then confirm from the recorded trials that constraint values were stored,
  that the reported incumbent is feasible, and that the gap to the recorded
  `constrained_optimum` (0.3979 for branin_disk) closes. Compare against the
  same task with the unconstrained optimizer config — if the two produce the
  same incumbent, the wiring is still inert.
