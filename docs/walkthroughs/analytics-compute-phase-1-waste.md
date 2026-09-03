# Analytics Compute Phase 1: stop wasting work (docs/PLAN-analytics-compute.md)

**Status:** implemented, awaiting your sign-off.
**You can now:** open an experiment page and have it stop refitting things it
didn't need to. The partial-dependence figure — which fires automatically on
every page load — went from **1553ms to 52ms**. A SMAC experiment no longer
leaks a temporary directory every time you look at it. And a local explanation
now actually uses the seed the experiment was run with.

Opens `docs/PLAN-analytics-compute.md`, a follow-on roadmap to
`docs/PLAN-analytics.md` (which is closed out). No settings, no schema change,
no new concepts — three independent defects, all found by measuring rather than
by reading.

---

## Why there was a phase here at all

The analytics roadmap built ten figures, each phase deciding for itself whether
its numbers were cheap enough to ship in the page payload or expensive enough
to fetch on demand. Every one of those calls was reasonable in isolation.
Nobody ever measured the whole thing at once.

Doing that turned up one policy problem — a page reload refits models, and
nothing is cached server-side anywhere in the repo — and, on the way to
answering it, three bugs. This phase is the three bugs. The policy question is
Phases 3 and 4.

The measurements that matter, all on `tests/fixtures/test2.ihpo` (Random
Forest, 4 hyperparameters, 30 trials, 4 metrics):

| | before | after |
|---|---|---|
| `compute_partial_dependence` | 1553ms | **52ms** |
| SMAC rebuild (per request) | 0.50ms + a leaked temp dir | **0.05ms, no directory** |
| `compute_hp_ablation` | 76ms, wrong seed | 81ms, right seed |

## 1. Partial dependence was paying sklearn's per-call overhead 600 times

The grid loop did one `Configuration` construction **and one single-row
`rf.predict([row])`** per (trial, grid point). For 30 trials on a 20-point grid
that's 600 separate calls into a 100-tree forest.

Where the time actually went, measured by taking the loop apart:

| | time |
|---|---|
| 600 single-row `rf.predict([row])` calls | 1.544s |
| the *same 600 rows* in one `rf.predict(matrix)` | **0.0035s** |
| all 600 `Configuration` constructions | 0.010s |

So essentially none of it was the forest, and none of it was ConfigSpace. It
was 600 round trips through sklearn's input validation. Batching the prediction
is ~440× on that step and **30× on the endpoint** (1553ms → 52ms); what's left
is mostly the one `fit_surrogate` call, which is now the floor.

The `Configuration` construction deliberately stays per-point. It's 10ms of the
1.55s, and it is what enforces conditionals and forbidden clauses — which is
where the `None` holes in an ICE row come from. So the shape of the change is:
build every row up front remembering *where* each belongs, skip the ones the
config space refuses, predict once, and scatter the answers back into a
pre-sized grid of `None`s.

That scatter-back is the one genuinely new piece of index arithmetic in this
phase, so it got two tests rather than one:

- `test_partial_dependence_batching_matches_predicting_one_at_a_time` computes
  the reference the slow way, in the test, and asserts the batched result is
  equal. This costs the suite ~5s and is worth it: it's the difference between
  "the fast version returns plausible numbers" and "the fast version returns
  *the same* numbers".
- `test_partial_dependence_holes_where_the_config_space_refuses_a_grid_value`
  builds a config space with a `ForbiddenAndConjunction` (no registry model has
  one) so the `None` branch is actually exercised, and asserts the hole lands at
  the right index and that the PDP mean skips it rather than counting it as zero.

## 2. Every explanation ignored the seed

`compute_hp_ablation` declared a `seed` parameter and never read it. Chasing
that turned out to be broader than the one function: `_build_explainer` didn't
take a seed at all, so **the three global games ignored it too**.

The mechanism, confirmed in the installed package rather than assumed:
`ExplanationTask.from_data` forwards to `DataBasedSurrogateModel`, which does
`base_model = RandomForestRegressor(random_state=seed)` with its **own**
`seed=0` default — a hardcoded 0, not the experiment's. Leaving `base_model` out
meant every explanation in the app was computed at seed 0 regardless of how the
run was seeded, silently.

Fixed by passing the estimator explicitly on both paths.
`RandomForestRegressor(n_estimators=100, random_state=seed)` at `seed=0` is the
identical estimator to hypershap's default (sklearn's own `n_estimators`
default is already 100), which is worth stating precisely because it means
**the numbers only move where they were wrong** — every stored result and every
pinned test expectation from before this stays valid.
`test_seed_zero_keeps_hypershaps_own_default_estimator` pins exactly that, by
computing tunability both ways and asserting equality.

## 3. SMAC leaked a temp directory per request

`SMACOptimizer.deserialize_result` called `tempfile.mkdtemp()`, reconstructed
`runhistory.json`, and wrote out every embedded SMAC state file — on **every**
rebuild. That path runs on a page load, a trial click, an ablation fetch and a
partial-dependence fetch. Nothing ever deleted them. Verified before the fix:
10 rebuilds, 10 directories, and no `rmtree` for `smac_output_dir` anywhere in
the repo.

Before removing the writes I traced every consumer, because "nothing needs this"
is the kind of claim that is embarrassing to get wrong:

- `serialize_result` is the **only** reader of `metadata["smac_output_dir"]`,
  and it already falls back to the base serialization when the directory is
  missing.
- A resume never reuses the directory — `optimize()` always makes its own and
  replays history through `tell`.
- The one thing a resume *does* lift out of stored state is `initial_design`,
  and `deserialize_result` already read that from the **in-memory** dict, not
  off disk.

So the state is now carried on `result.metadata["optimizer_state"]` (`metadata`
is never serialized, so it's a pure in-process handoff) and nothing is written.

**One thing this exposed that had to be fixed alongside it.** The old
deserialize→serialize round trip only preserved `optimizer_state` *by going
through the filesystem* — deserialize wrote the files, serialize read them
back. With the writes gone, the base fallback would have silently dropped the
state, and with it the ability to resume an imported run. Hence
`_serialize_without_a_live_run`, which passes the carried dict through.
`test_carried_state_survives_re_serialization` covers it, and the pre-existing
`test_result_survives_load_save_cycle` passed unchanged, which is the better
evidence that fidelity held.

### The security property, deliberately kept

The old write loop used `safe_join` to stop a hand-edited `optimizer_state` key
(absolute, or containing `..`) from writing wherever it pointed, and
`tests/core/test_snapshot_paths.py` pins that. Removing the writes makes
traversal *unreachable here* — a stronger outcome than guarding it — but the
validation stays anyway, because the state is still passed through into
whatever gets serialized next, and a hostile key should be refused where it
enters rather than where it lands. `io.parse` checks this too
(`_check_optimizer_state`); the deserialize-time check covers a result that
reached the optimizer without passing through parse, which is exactly what its
test says it's for.

## Deviations from the plan, and notes

- **The plan said 112×, and the endpoint is 30×.** Both numbers are real and I
  had conflated their scopes: ~440× is the prediction step, 112× was the
  prediction *loop* including `Configuration` construction, and 30× is the
  whole function — because `fit_surrogate`'s 36ms is inside it and doesn't
  change. Corrected in the roadmap; the endpoint number is the one that matters
  to a page load.
- **The seed bug was wider than recorded.** The plan named
  `compute_hp_ablation`; `_build_explainer` had the same defect, affecting all
  three global games. Both fixed.
- **`_serialize_without_a_live_run` wasn't in the plan.** It's required, not
  optional — see above. Without it this phase would have quietly broken resume
  for imported SMAC runs, which no existing test would have caught in the
  direction that mattered.
- **Two tests were rewritten rather than deleted**, and both now say in their
  docstrings what they used to assert and why that changed, so the next reader
  doesn't have to guess whether the old behaviour was a requirement.
- Not touched, deliberately: the ~57 Plotly figures built per render. That is
  plausibly the largest remaining page cost now, and it is still unmeasured.
  Measuring it is the next honest step, not optimizing it.

## Checklist

- [x] `compute_partial_dependence` batches its predictions into one call
- [x] the `None`-hole path preserved, and now actually tested
- [x] `_build_explainer` and `compute_hp_ablation` honour `seed`
- [x] seed-0 behaviour proven identical to hypershap's default
- [x] `deserialize_result` writes nothing; state carried in `metadata`
- [x] `serialize_result` passes carried state through, so resume survives
- [x] path-traversal guard retained at both parse and deserialize time
- [x] 5 new tests; 2 rewritten with their history documented

## Verification

- `pytest -q -m "not slow"` → **773 passed, 5 skipped, 36 deselected in 178.25s**.
  Up 5 tests from Phase 6's 768, and *faster* than its ~181s despite the
  additions — the PDP tests themselves got quicker, which more than paid for
  the ~5s the new equivalence test costs.
- The slow SMAC subset (`-k smac`) passes, including
  `test_smac_resume_through_serialization_preserves_all_trials`, which is the
  real resume path.
- `manage.py check` → no issues. `makemigrations --check --dry-run` → no changes.
- Measured after the change: PDP 52.4ms (from 1553.2ms), SMAC rebuild 0.05ms
  (from 0.50ms), **0 temp directories leaked across 10 rebuilds** (from 10).
- In the browser (Playwright, experiments 1 and 6): three consecutive loads of a
  SMAC experiment leaked 0 temp directories; the PDP figure renders both traces
  with the right legend and 20 grid points; switching hyperparameter moves the
  axis title with it; local ablation still renders its bar; no console errors.
  Confirmed the SMAC results still deserialize fully (55 and 30 trials, state
  carried) — an early check of mine reported an empty trials table, which was a
  wrong CSS selector in the check script, not a regression.
