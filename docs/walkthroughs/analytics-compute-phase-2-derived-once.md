# Analytics Compute Phase 2: compute derived values once (docs/PLAN-analytics-compute.md)

**Status:** implemented, awaiting your sign-off.
**You can now:** nothing new — this phase is deliberately invisible. What
changed is that the detail page stops recomputing the same three things over and
over, and one class of disagreement between figures became impossible rather
than merely unlikely.

One user-facing bug did get fixed on the way, because it was one line from code
this phase was already rewriting: importing an experiment file whose result has
no trials used to 500 the detail page.

---

## The point is agreement, not speed

`best_index` was computed **six times per metric per render**: once in the
detail page's panels loop, again inside `_selected_panel_data` (which the caller
already had it for), and once per each of `performance_over_time`'s four views.
`incumbent_scores` was computed once per view — four identical lists per metric.

The cost of that is genuinely trivial, and it would be dishonest to sell this as
a performance phase. The reason to consolidate is different: **six copies of an
argmax are five chances for them to disagree about which trial is "best."** The
highlighted marker on the performance chart and the trial the "Selected
configuration" panel describes below it are supposed to be the same trial. With
six independent `max(range(...), key=...)` calls, that held only because all six
happened to break ties the same way. Now there is one implementation, and
`test_best_index_breaks_ties_toward_the_lowest_index` pins the rule.

Both accessors moved onto `OptimizationResult` and memoize per metric.

### Why the memo is not a dataclass field

`rebase_history` rebuilds a result with `dataclasses.replace`, which passes
declared fields through to `__init__`. A memo declared as a field would be
carried into the rebased copy — which is precisely the class of bug that
rebasing exists to fix. As a plain attribute it is simply absent on the new
object and gets recomputed.

Worth being straight about: **nothing observable distinguishes the two today.**
Rebasing changes each trial's `score` and `incumbent_score` but not its `scores`
dict, and these accessors read `scores`, so a carried-over memo would still have
been correct. That is exactly why it's written down and tested
(`test_the_memo_is_not_a_dataclass_field`) — so a future tidy-up that promotes it
to a field doesn't quietly make that accident load-bearing.

## The fallback surrogate was the one real duplicate fit

`_build_explainer` was already shared across the three global games, on the
grounds that refitting an identical surrogate per game is pure waste. One rung
down, the fallback wasn't: when HyperSHAP raised, each of
`tunability`/`sensitivity`/`mistunability` independently called `fit_surrogate`
with identical arguments — the same waste the explainer sharing existed to
avoid.

The `(hs, warning, insufficient)` tuple became a small `_MetricExplainer` object,
purely so the fallback has somewhere to live. It fits at most once per metric,
lazily (most runs never reach the rung at all), and memoizes the failure case
too — a fit that failed will fail the same way on the next game, and reporting it
three times is not three pieces of information.

`test_the_fallback_surrogate_is_also_fitted_once_per_metric_not_per_game` forces
all six game calls to raise across two metrics and asserts exactly two fits,
plus that the fallback still produced real per-hyperparameter numbers rather than
degrading to uniform weights.

## The bug found next door: a result with no trials

`_detail_context` guarded `result is None` but not `not result.trials`. With an
empty trials list the panels loop did an argmax over an empty range and raised.

I checked whether that was reachable rather than assuming either way: **`io.parse`
accepts a snapshot whose result has an empty `data` list**, so importing a
hand-written or truncated `.ihpo` hits it. A run never produces one —
`ui/services/run.py` only writes a result that has trials — which is why it went
unnoticed.

This predates the phase (it raised `ValueError` from `max()` before and would
have raised `TypeError` from `trials[None]` after), so consolidating the argmax
changed the exception but not the outcome: a 500 either way. Fixed by giving it
the same early return as "no result at all", since there is nothing to plot and
no best trial to describe in either case — the template's existing waiting state
already handles it. Three tests in `tests/ui/results/test_empty_result.py`, one
of which pins the premise that the parser really does accept this shape.

**This is a deliberate scope widening** — Phase 2 was meant to be pure cleanup.
Shipping a 500 I had just confirmed, one line from code I was rewriting anyway,
seemed worse than the widening.

## Deviations from the plan

- **Skipped: hoisting the metric-independent `Configuration` construction out of
  `compute_hp_games`' per-metric loop.** The plan listed it "to remove a
  divergence risk", but on inspection there is no divergence risk to remove —
  it's one function called with a different metric, not two implementations that
  could drift. And the measurement doesn't support it either: it's 0.5ms per
  metric against a 3.4s games computation, so ~2ms, or **0.06%**. Adding a cache
  parameter and a second helper for that is a worse trade than leaving it, so I
  left it.
- **Added: the empty-trials guard**, as above.
- `plots.py::incumbent_scores` is kept as a thin delegate rather than deleted —
  it's part of `ui.figures`' exported surface and tests call it directly.

## Checklist

- [x] `OptimizationResult.best_index(metric)` — one argmax, memoized per metric
- [x] `OptimizationResult.incumbent_scores(metric)` — memoized per metric
- [x] memo held off the dataclass fields, with the reason tested
- [x] all six `best_index` sites and all four `incumbent_scores` sites converted
- [x] fallback surrogate fitted once per metric, not once per game
- [x] tie-breaking pinned
- [x] empty-trials result renders instead of 500ing
- [x] 13 new tests

## Verification

- `pytest -q -m "not slow"` → **786 passed, 5 skipped, 36 deselected**, up 13
  from Phase 1's 773.
- **Runtime: 190.91s, 193.42s, 196.50s across three samples**, against Phase 1's
  178.25s. I took the second sample expecting ~178s, didn't get it, and went
  looking rather than calling it noise. The split:
  - **~5.6s is this phase**, measured directly: twelve of the thirteen new tests
    cost 0.28s in total; the thirteenth costs 5.29s because it needs a real
    five-trial optimization to have something to explain before it can force the
    fallback rung.
  - **~7s is the machine, not the code.** Every *pre-existing* test got ~6%
    slower, including ones this phase cannot touch —
    `test_grid_search_resume_skips_evaluated_configs` 12.11s → 13.45s,
    `test_grid_run_caps_at_grid_size` 10.56s → 11.60s,
    `test_an_optimizer_runs_a_cross_validated_search` 5.32s → 5.69s. And the
    three samples above rose monotonically on *identical* code, which a code
    regression cannot do.
  Consolidating an argmax and memoizing two lists cannot make grid-search resume
  slower, so I'm recording the ~5.6s as the real cost of this phase and the rest
  as measurement conditions — with the numbers above so you can judge that
  yourself rather than take it on trust.
- `manage.py check` → no issues. `makemigrations --check --dry-run` → no changes.
- **Equivalence check, since this phase must change nothing visible:** for all 7
  local experiments × all 4 metrics, `best_index` and `incumbent_scores` were
  compared against the exact inline expressions they replaced — all match — and
  all four `performance_over_time` views still build a figure. That's a stronger
  check than diffing rendered HTML, because it isolates the values this phase
  touched.
