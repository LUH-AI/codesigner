# DeepCave-inspired analytics: an "alternate views" mechanism, then a phased plugin roadmap

> Drafted 2026-08-14. Progress is tracked per-phase in each walkthrough's own
> Status line, matching the convention `docs/PLAN.md` established; write-ups
> live in [docs/walkthroughs/](walkthroughs/).

## Context

Codesigner already positions itself as a spiritual successor to DeepCave
(the AutoML.org group's HPO-analysis tool), but today only reproduces a
fraction of DeepCave's analysis surface: an importance chart, a
performance-over-trials chart, and an error-over-time chart. The goal here is
to close that gap — porting or reproducing DeepCave's other useful plugins —
while also addressing a real UI complaint that holds even for the plugins
codesigner *does* have: nearly-identical information (a pie vs. a bar chart
vs. a table of the same importance numbers; performance-over-trials vs.
error-over-time, which are the same curve on different axes) is spread across
separate figures instead of being one entry with switchable views.

**A finding that reshapes the "fANOVA → HyperSHAP" premise this plan started
from:** codesigner's existing feature-importance computation
(`core/optimizers/base.py:554-609`, shared by all three optimizers) is
**already HyperSHAP-based**, not fANOVA-based — it computes order-1 Shapley
"tunability" values from trial history via
`hypershap.ExplanationTask`/`HyperSHAP().tunability()`, falling back to a
RandomForest's `feature_importances_` on failure. So there is no fANOVA to
swap out; instead, the highest-leverage move is exposing more of what the
already-installed `hypershap` library can do (it also ships `sensitivity`,
`ablation`, and `optimizer_bias` "explanation games," and supports *local*
single-configuration explanations, none of which codesigner surfaces yet)
alongside porting DeepCave plugins that have no HyperSHAP equivalent at all
(Configuration Cube, Parallel Coordinates, Partial Dependencies).

DeepCave's plugin list, verified against its docs/repo, and each one's
feasibility against codesigner's current single-objective, single-fidelity
setup:

| DeepCave plugin | Needs | Feasible now? |
|---|---|---|
| Importances (fANOVA/LPI) | trial history (+ fitted RF) | **Already have an equivalent** — HyperSHAP tunability |
| *(no DeepCave equivalent)* Higher-order interactions | same surrogate HyperSHAP already needs for order-1 | Yes — HyperSHAP already computes order-2 FSII via `shapiq`; only the visualization is new work |
| Cost Over Time | trial history | **Already have** — incumbent-performance/error-over-time |
| Configuration Cube | trial history | Yes, no new capability needed |
| Parallel Coordinates | trial history (+ importance for ordering, optional) | Yes |
| Partial Dependencies (PDP/ICE) | fitted RF surrogate over trial history | Yes, needs a new shared surrogate-fitting utility |
| Ablation Paths | fitted RF surrogate | Superseded by HyperSHAP's own `ablation` game — no path-trace visualization |
| Configuration Footprint | fitted RF surrogate + MDS projection | Possible later; overlaps heavily with Cube/Parallel Coordinates, lower priority |
| Symbolic Explanations | fitted RF surrogate + symbolic regression | Lowest priority / exotic |
| Pareto Front | 2+ objectives | **Out of scope** — codesigner is single-objective by design (`docs/PLAN.md`); noted for distant future only |
| Budget Correlation | 2+ fidelity/budget levels | **Out of scope** — codesigner has no budget/fidelity dimension; noted for distant future only |

Decisions made while drafting this plan: Pareto Front / Budget Correlation
are noted as blocked-on-capability, not scheduled. Ablation is HyperSHAP's
`ablation` game only, no separate path-trace visualization. New
view-selectors are session-only (client-side, reset on reload) — the same
pattern metric-switching already uses, no new persisted settings. A
dedicated phase for HyperSHAP's higher-order (pairwise+) interactions sits
ahead of Configuration Cube, once research confirmed real "existing
machinery" for it — HyperSHAP already computes order-2 interactions via a
`shapiq` dependency; see Phase 2 for what that does and doesn't hand us for
free (the computation, yes; the plotting, no — `shapiq`'s plotting is
matplotlib-only, so it needs a Plotly-native reproduction, not a straight
port, to stay consistent with the rest of the app).

**This file is the roadmap.** Each phase below is a self-contained, shippable
unit of work, executed one phase at a time with its own review/sign-off,
matching this project's per-epic-walkthrough habit — not implemented all in
one pass.

## Current architecture (what every phase builds on)

- `Figure` (`ui/figures/base.py:29-69`): `key`, `label`, `width`
  (`HALF`/`FULL`), `per_metric`, `absolute_scale`, and a `plot(result,
  metric) -> go.Figure | None` classmethod. **Strictly one plot per figure
  today** — no view/tab concept exists.
- `FIGURES` (`ui/figures/catalog.py:90-98`): ordered tuple of `Figure`
  subclasses. `ui/views.py:775-822` (`_detail_context`) filters it by
  settings, precomputes every metric's plot, and hands the page
  `metric_plots`/`static_plots`/`figure_options`.
- Rendering: `experiment_detail.html` includes `figure.template` per figure;
  JS `draw(key, plot)` mounts each by `#figure-<key>`. Metric switching
  (`experiment_detail.html:~445-453`) is a plain `<select>` `change` listener
  that re-draws by key and toggles `.panel[data-metric]` visibility — this
  exact pattern is what the new view-selector reuses, one level nested
  (metric → view).
- The only existing "alternate view" precedent is `makeScaleToggle`
  (`experiment_detail.html:~316-363`), a modebar button that flips one axis
  relayout — it doesn't switch chart type or template, but confirms the
  precomputed-payload, client-side-swap approach is already this codebase's
  idiom.
- Visibility (`ui/services/settings.py:15-18`, `ui/forms.py:134-139`):
  one `show_<key>` boolean auto-derived per `FIGURES` entry. Stays one
  checkbox per **entry** even once an entry gains multiple views — no change
  needed here beyond entries merging correctly.
- Real wall-clock per-trial timestamps already exist
  (`core/optimizers/timing.py:38-58`, `run_info["starttime"/"endtime"]`) —
  `error_over_time_plot` currently ignores them and sums `duration` instead;
  switching to real timestamps is a byproduct of Phase 0, not new capability.
- No live fitted surrogate is ever retained after a run (SMAC's facade is a
  local var, discarded at the end of `optimize()`) — anything needing a
  surrogate (Phase 5+) fits a fresh, cheap RandomForest over the trial
  history, the same way `compute_hp_importance`'s own fallback path already
  does (`core/optimizers/base.py`) — not a new kind of capability, just
  reused more often.

## Phase 0 — The "alternate views" mechanism, proven on two concrete cases

The foundation everything else leans on: importance as pie/bar/table, and the
performance/error-over-time merge, as one entry with axis toggles.

- Extend `Figure` (`ui/figures/base.py`) to optionally declare a set of named
  views; a figure with no views declared behaves exactly as today (every
  existing figure needs zero changes). A view is either a Plotly-producing
  function (like today) or a "table" kind that renders a small Django
  template partial instead of a `go.Figure` — so the table view needs no new
  client-side rendering code, just a precomputed HTML block toggled the same
  way metric panels already toggle.
- **Feature Importance entry**: becomes one entry with pie / bar / table
  views over `result.hyperparameter_importance[metric]`
  (`ui/figures/plots.py:64-78` is today's pie builder; add a bar builder —
  matching DeepCave's own default chart type for this data — and a table
  partial). The existing warning-text handling (`hyperparameter_importance_warning`)
  is shared across all three views unchanged.
- **Performance/error entry**: merge `incumbent_performance_plot`
  (`ui/figures/plots.py:29-61`) and `error_over_time_plot` (`:100-136`) into
  one entry with two independent axis toggles — x: trial index | wall-clock
  time, y: score | error — four view combinations. Both already share
  `incumbent_scores()` (`:19-26`), so the data layer barely changes; the
  time axis switches from `error_over_time_plot`'s current cumulative-summed
  `duration` to the real `starttime`/`endtime` in `run_info`
  (`core/optimizers/timing.py`), which is strictly more correct and needs no
  core changes. `absolute_scale` becomes per-view-combination (today's two
  fixed relayouts — linear 0–1, log −3..0 — need reconciling into a small
  lookup keyed by the active combination).
- New view-selector UI: one or two small `<select>`s per multi-view figure
  card, wired with the same `change` listener idiom the metric switcher
  already uses (`experiment_detail.html`) — session-only, no persistence.
- Update the settings-derived checkboxes: two old keys
  (`show_incumbent_performance`, `show_error_over_time`) collapse into one
  new merged key; old keys simply become unused JSON — no migration needed
  (`GlobalSettings`/experiment settings are JSONFields), just a one-line note
  in the walkthrough about the rename.
- Tests to update: `tests/ui/results/test_figures_view.py`,
  `test_figure_visibility.py`, `test_plots.py`, `test_metric_switcher.py` (the
  merged entry + new views); new tests for the view-switch rendering/toggle
  behavior itself.

## Phase 1 — Expose HyperSHAP's other explanation games

The highest-value, lowest-effort port: `hypershap` is already a dependency
and already wired into `compute_hp_importance`
(`core/optimizers/base.py:554-609`) for its `tunability` game only. The same
library ships `sensitivity`, `ablation`, `mistunability`, and `optimizer_bias`
games, and supports *local* (single-configuration) explanations, not just
global ones.

> **Status: split in two after the spike.** `optimizer_bias` turned out to
> need a live ensemble of optimizers to compare against, not just trial
> history, so it's excluded (doesn't fit this app's model). The rest split
> cleanly by call shape — `tunability`/`sensitivity`/`mistunability` take no
> required arguments (same signature, same cost profile); `ablation` takes a
> specific configuration to explain, a genuinely different shape and use.
>
> **1a — the three global games — done**, see
> [walkthroughs/analytics-phase-1a-global-games.md](walkthroughs/analytics-phase-1a-global-games.md).
> `mistunability` (MIN aggregation — downside risk) was added alongside
> `sensitivity`, since it shares tunability's exact call shape and rides
> along for free.
>
> **1b — the local view via `ablation` — done**, see
> [walkthroughs/analytics-phase-1b-local-ablation.md](walkthroughs/analytics-phase-1b-local-ablation.md).
> Spike confirmed it's cheap (~45ms against a real fixture, vs. ~0.25-0.3s for
> each global game), so no caching was needed on the compute side; it's
> fetched lazily per selected trial rather than precomputed, since (unlike
> the three global games) its answer depends on which trial is selected and
> that changes with every click.

- **Spike first**: confirm the installed `hypershap` version's actual call
  signatures for the other games and for local explanations — this plan's
  research could not fetch the full paper, so treat the exact API as
  unconfirmed until checked against the installed package (`pip show
  hypershap`, its own source/docstrings). *(Done — see the walkthrough.)*
- Generalize `compute_hp_importance` into sibling methods (or one method
  parameterized by game) sharing the existing trial-pairing + RandomForest-
  fallback + uniform-weights-warning scaffolding already in `base.py` — no
  need to duplicate that plumbing per game. *(Done, Phase 1a.)*
- Add a **local importance at the currently-selected trial** view, driven by
  HyperSHAP's `ablation` game (config-of-interest = the selected trial,
  baseline = the config space default) and wired to the *existing*
  click-to-select mechanism (the `trial_panel` endpoint / selected-config
  panel already built in Step 7) — the standout feature here, since it
  reuses an entire existing UI flow end to end rather than building new
  interaction. Ablation's values are signed (did tuning away from baseline
  help or hurt this specific trial), unlike the three global games' abs-value
  shares — needs its own rendering (a diverging bar at minimum), not a fourth
  drop-in option on the existing pie/bar/table. *(Done, Phase 1b.)*
- These become additional views on the Phase 0 Feature Importance entry:
  Tunability / Sensitivity / Mistunability / Local (selected trial, via
  ablation) — all *(done)*.
- Tests: unit tests for the new `compute_hp_*` methods (mirror whatever
  currently tests `compute_hp_importance` — locate via
  `tests/core/test_hp_importance.py`) *(done)*, plus a view test for the
  local-explanation-at-selected-trial wiring *(done)*.

## Phase 2 — Higher-order HyperSHAP interactions (pairwise, then beyond)

> **Status: done**, see
> [walkthroughs/analytics-phase-2-interactions.md](walkthroughs/analytics-phase-2-interactions.md).
> The "cost is an open question" note below turned out to have a definite
> answer: zero marginal cost. `order=2` was already the default on every
> HyperSHAP call this app makes, so the existing importance computation was
> already producing pairwise values and discarding them — this phase is a
> re-extraction of data already computed, not a new HyperSHAP call.

Confirmed, not guessed: HyperSHAP already computes this. Every HyperSHAP
explanation-game method (`tunability`, `sensitivity`, `ablation`,
`optimizer_bias`, ...) takes an `order` parameter — **default `order=2`** —
and returns interaction values via the Faithful Shapley Interaction Index
(FSII), as a `shapiq.InteractionValues` object (`shapiq` — Shapley
Interaction Quantification — is a genuine code dependency of HyperSHAP, not
just shared authorship; its README states outputs are `InteractionValues`
objects directly). So computing pairwise (and higher) interactions needs
**no new surrogate/black-box wrapper** beyond what Phase 1 already
generalized for order-1 games — just passing `order=2` (or higher) through
the same call.

**The one real decision, not yet made: how to render it.** `shapiq` ships
real plotting code for this (`network_plot`/`plot_network`, `plot_force`,
`plot_stacked_bar`, `plot_bar`, `plot_waterfall`, and HyperSHAP wraps several
of these directly as `plot_si_graph`/`plot_upset`/`plot_force`/`plot_waterfall`/
`plot_stacked_bar` methods) — but confirmed from source, all of it is
**matplotlib-only** (`matplotlib`/`networkx` in `shapiq`'s core dependencies,
no Plotly path anywhere). Codesigner's entire app renders through Plotly
(`plotly.min.js`, every `Figure.plot()` returns a `go.Figure`) with no
matplotlib anywhere today. Two options:

1. Render `shapiq`'s matplotlib figures as static images. Fastest to build,
   literally reuses their plotting code, but is a real inconsistency — every
   other figure in the app is a live, interactive Plotly chart, and this one
   wouldn't be.
2. Take the computed `InteractionValues` (the actual numbers — no need to
   recompute anything) and render them Plotly-natively, matching every other
   figure in the app.

**Chosen: option 2**, specifically as a **heatmap** (hyperparameter ×
hyperparameter grid, cell = pairwise interaction strength) as the primary
view plus a **bar chart of the top-k strongest pairwise interactions** as an
alternate view — both are straightforward in Plotly (`go.Heatmap`, `go.Bar`),
unlike a force-directed network graph, which Plotly has no native layout
algorithm for. This also gives this entry the same pie/bar/table-style
multi-view treatment as Phase 0's Feature Importance entry, for free.

- New `Figure` entry ("Hyperparameter Interactions"), heatmap + bar views,
  computed via the Phase 1 `compute_hp_*` scaffolding with `order=2`.
  *(Done.)*
- **Cost turned out to be zero, not "open"**: see the status note above —
  measured by reading `hypershap`'s own source rather than benchmarking,
  since there was nothing new to benchmark. *(Done.)*
- Tests: unit tests for the order-2 computation (mirroring however Phase 1
  tests its `compute_hp_*` methods), plus rendering tests for the heatmap/bar
  views. *(Done.)*

## Phase 3 — Configuration Cube

> **Status: done**, see
> [walkthroughs/analytics-phase-3-configuration-cube.md](walkthroughs/analytics-phase-3-configuration-cube.md).
> No `views` tuple after all — axis choice is a per-experiment, unbounded
> combination, not the small fixed vocabulary every other multi-view figure
> picks from, so the client remaps from one self-contained trace instead
> (`customdata` + `layout.meta`), per the plan below.

A 2D/3D scatter of every trial's configuration, colored by score. No
surrogate needed — pure trial history, like DeepCave's version.

- New `Figure` entry in `ui/figures/plots.py` + `catalog.py`. Axes are 2 or 3
  user-picked hyperparameters (simple `<select>` pickers, not DeepCave's MDS
  projection — keeps this phase dependency-free and cheap). Precompute one
  full per-trial dataset (all HP values + score) once and remap axes
  client-side (`Plotly.newPlot`/`restyle`) rather than round-tripping to the
  server per axis change. *(Done.)*
- New template partial, automatic settings checkbox (existing mechanism).
  *(Done.)*
- Tests: new file or addition to `tests/ui/results/test_plots.py`. *(Done —
  added to test_plots.py, plus test_figure_visibility.py/test_figures_view.py.)*

## Phase 4 — Parallel Coordinates

> **Status: done**, see
> [walkthroughs/analytics-phase-4-parallel-coordinates.md](walkthroughs/analytics-phase-4-parallel-coordinates.md).
> The simplest phase so far: axis order is a full, fixed per-metric ranking
> rather than an unbounded combination (contrast the cube), so it needed no
> `views`, no client-side wiring at all — just the ordinary per-metric
> precompute every figure before Phase 3 already used.

Per-trial lines across every hyperparameter axis, ending in the objective
axis — codesigner has no equivalent today, and it's DeepCave's best tool for
spotting HP interactions at a glance.

- New `go.Parcoords`-based builder, catalog entry, template, tests. Order
  axes by the existing HyperSHAP tunability importance (Phase 0/1's output)
  instead of DeepCave's fANOVA ordering — a natural fit for the
  "HyperSHAP-first" framing, and no new computation needed if Phase 1 already
  shipped. *(Done.)*

## Phase 5 — Shared surrogate-fitting utility

Pure infrastructure, no user-visible change — unblocks Phase 6.

> **Status: done**, see
> [walkthroughs/analytics-phase-5-shared-surrogate.md](walkthroughs/analytics-phase-5-shared-surrogate.md).
> Also fixed a real, verified inefficiency found while implementing this:
> the three global games were each independently re-fitting an identical
> surrogate per metric — now built once, reused across all three.

- Factor the RandomForest-fitting logic that `compute_hp_importance`'s
  fallback path already has (`core/optimizers/base.py`) into a small, reusable
  function (e.g. `fit_surrogate(trials, config_space)`), so it isn't
  duplicated by every future surrogate-consuming plugin. *(Done —
  `fit_surrogate(config_space, trials, metric_name, seed)`.)*
- Tests: unit tests against synthetic trial data only. *(Done.)*

## Phase 6 — Partial Dependencies (PDP/ICE)

Uses Phase 5's surrogate. 1D marginal-effect plot per hyperparameter (a
picker dropdown selecting which HP to show, rather than DeepCave's small
multiples, to keep this one `Figure` entry). 2D interaction plots are an
optional stretch within this phase, not required for it to ship.

- New builder, catalog entry, template, tests, following the same pattern as
  every prior phase.

## Explicitly not scheduled

- **Pareto Front** and **Budget Correlation** — blocked on capabilities
  (multi-objective, multi-fidelity) codesigner intentionally doesn't have.
  Noted here for reference only, in case either capability is ever added for
  unrelated reasons.
- **Configuration Footprint** (MDS-projected coverage/performance surface)
  and **Symbolic Explanations** — technically feasible via Phase 5's
  surrogate, but overlap heavily with Cube/Parallel Coordinates (Footprint)
  or are niche/exotic relative to everything above (Symbolic Explanations).
  Worth reconsidering only after Phases 0–6 are done and there's appetite for
  more.

## Verification (per phase, not just at the end)

Matches this project's own established habit ("every step ships the UI
needed to manually test what it built," `docs/PLAN.md`):

- `python -m pytest -q -m "not slow"` green after each phase, with that
  phase's new/updated tests included.
- Manually load an experiment's detail page with a completed run after each
  phase and exercise the new view/figure in the browser — metric switching,
  the new view-selector, and (from Phase 1 on) the click-to-select
  interaction all still need to agree with each other.
- Each phase gets its own short walkthrough doc under `docs/walkthroughs/`,
  matching the existing convention (concept explanation, feature checklist,
  "what is now possible," verification) — one epic, one write-up, signed off
  before the next phase starts.
