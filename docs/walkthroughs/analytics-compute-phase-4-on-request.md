# Analytics Compute Phase 4: per-figure control over on-request analytics (docs/PLAN-analytics-compute.md)

**Status:** implemented, awaiting your sign-off.
**You can now:** decide, per analytic, whether it works itself out as soon as the
page needs it or waits for a button. Switch both off and **opening or reloading an
experiment page fits nothing at all** — which is the thing that started this whole
piece of work.

Two new checkboxes on both settings pages: "Local explanation (selected trial)"
and "Partial dependence (PDP/ICE)", under *Compute automatically, without waiting
to be asked*. Both default on, so nothing changes unless you change it.

Closes `docs/PLAN-analytics-compute.md`.

---

## Two states, so a boolean

The plan offered a tri-state (auto / on-request / never) and I said the harder
option was warranted. It isn't, because **"never" already exists**: switching a
figure off with its `show_<key>` checkbox suppresses its fetch too. So the
genuinely new axis has exactly two states, and a boolean is the honest shape.

That matters beyond tidiness. It means:

- `_posted_settings`' blanket `bool(request.POST.get(key))` needs no special case
  — the chokepoint I'd flagged as a blocker never had to be touched.
- No new widget type, no changes to how the settings partial renders fields.
- No migration: both settings columns are JSONFields and `resolve_settings`
  backfills unknown keys, so every settings row written before this reads as on.
  Tested (`test_a_stored_setting_from_before_these_existed_still_resolves`).

Combined with the existing visibility flag, three states with no tri-state widget:

| `show_<key>` | `autocompute_<name>` | Behaviour |
|---|---|---|
| off | — | never computed |
| on | on | computed as soon as the figure needs it (today) |
| on | off | a **Compute** button; a reload computes nothing |

## The keys name the computation, not the figure

`Figure.deferred` is a tuple of `(name, label)` pairs, and the setting is
`autocompute_<name>`. The names are the computation's rather than the figure's for
one specific reason: **local ablation is a *view of* the importance figure**, whose
other nine views are computed once at run completion and stored.
`autocompute_hyperparameter_importance` would claim the importance numbers are
deferred when they are not. So:

- `HyperparameterImportance.deferred = (("local_ablation", …),)`
- `PartialDependence.deferred = (("partial_dependence", …),)`

Declared on the figure, derived like `setting_key`/`template`/`dom_id` are, so a
future figure that defers something gets its checkbox on both settings pages
without touching the forms or the templates.

## "Nothing asked for yet" is not "nothing to show"

`draw(key, null)` empties a figure *and reveals its "No … data available"
caption* — the right thing when a fetch came back with nothing. It is the wrong
thing here: that caption would sit directly under a Compute button, contradicting
it.

So the pending path uses a separate `clearPlot(key)`, which empties the figure and
keeps the caption hidden. Small, but it's the difference between a page that
explains itself and one that argues with itself.

## The button state is keyed like the cache

Pressing Compute passes `asked = true` past the gate for that one fetch. Which
(metric, hyperparameter) or (metric, trial) has already been computed is tracked by
the same `pdpCache` / `localAblationCache` keys that already existed, so:

- returning to something already computed shows it immediately, no new request;
- moving to something new shows the button again, because it genuinely is new work.

That falls out of reusing the cache keys rather than needing separate bookkeeping.

## A Phase 2 shortfall this turned up

Wiring the flags into the page meant looking at when the figure script renders at
all — and that exposed something I got half-right in Phase 2.

Phase 2 fixed a 500 on a result with no trials by returning early from
`_detail_context`. But `has_result` was `result is not None`, so the template
still rendered the **entire plotting script** while `panels`, `metric_plots` and
the rest were absent from the context. Django resolves missing variables to `""`,
so the page returned 200 and then broke in the browser on
`JSON.parse('""').forEach`.

My Phase 2 test asserted `status_code == 200`, which passed. A 500 became a page
with broken JavaScript, and the test I wrote couldn't tell the difference.

`has_result` now means "a result *with trials*", which is what the early return
actually keys on, so the empty case takes the template's "No results yet." branch.
`test_the_figure_script_is_not_rendered_at_all` asserts the script's data blocks
are absent — the check that would have caught it.

## Deviations from the plan

- **Booleans rather than the tri-state** you picked, as argued above — the same
  per-computation granularity you asked for, without the POST-coercion change,
  new widget or migration the tri-state would have needed.
- **`clearPlot` wasn't in the plan.** Required, not cosmetic: without it the
  pending state renders a contradiction.
- **The `has_result` fix wasn't in the plan** — it's a Phase 2 defect this phase
  surfaced. Fixed here rather than left, since Phase 4 is what made the figure
  script's render condition load-bearing.
- The plan sketched gating inside `refreshPartialDependence`/`refreshLocalAblation`
  and that is where it went; both grew one `asked` parameter and an early return.

## Checklist

- [x] `Figure.deferred` declares `(name, label)` per deferred computation
- [x] `autocompute_<name>` in `SETTING_DEFAULTS`, derived from the catalog
- [x] checkboxes on both settings pages, defaulting on, no migration
- [x] per-experiment override works, independently per computation
- [x] flags reach the page through `json_script`, like every other payload
- [x] Compute button per deferred figure, hidden unless pending
- [x] pending state clears the plot without claiming "no data"
- [x] button state keyed like the fetch cache
- [x] `has_result` no longer renders the figure script with nothing to read
- [x] 16 new tests

## Verification

- `pytest -q -m "not slow"` → **820 passed, 5 skipped, 36 deselected in 197.20s**,
  up 16 from Phase 3's 804. Runtime flat against the 191–197s band the machine has
  been sitting in since Phase 2.
- `manage.py check` → no issues. `makemigrations --check --dry-run` → no changes
  (the point of the JSONField-backed settings).
- **In a browser, which is where client-side gating has to be proven.**
  With both switched off, on page load: **zero deferred requests**, the Compute
  prompt visible, the plot hidden, and the "no data" caption correctly *not*
  showing. Pressing Compute issued exactly one request and drew the figure.
  Switching hyperparameter went back to the prompt with no request; switching
  *back* to the computed one redrew it with no request. Switching the importance
  figure to "Local" prompted rather than fetching, and its Compute drew the
  ablation bar. No console errors.
  With the default (on), the same page issued its one partial-dependence request
  on load and drew both traces — today's behaviour, unchanged.
- The experiment I flipped for that check has been restored to inheriting the
  defaults; no experiment in the dev database is left with an override.
