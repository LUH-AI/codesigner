# Figures while it runs

**Status:** implemented, awaiting your sign-off.
**You can now:** watch the trial-based figures fill in during a run instead of
waiting for it to end, and ask for the explanations on a run that never computed
any — a cancelled one, or one you are still watching.

Closes Step 3 of `docs/plan_today.md`.

---

## There was never a rendering constraint

The plan gets this exactly right and it is worth restating, because it is the
whole shape of the change. Nothing was withholding a figure. Trial performance,
trial duration, the projection and parallel coordinates read nothing but
`result.trials`, and the page has always been able to draw them the moment there
are any. They had nothing to draw because `execute_run` wrote
`experiment.result` **once**, at `ui/services/run.py`, after `_optimize`
returned.

So the run writes what it has as it goes, and the page is told.

### The write

`BaseOptimizer.progress` is a callback the application sets, the same way it
already sets `analytics_max_coalitions` and `analytics_wanted` — per instance,
so `core/` stays Django-free and a direct caller pays nothing for a feature it
is not using. `ui/services/run.py` points it at `_partial_result_writer`.

It reaches all three optimizers through **`BaseOptimizer.new_collector`**, a
factory replacing the three separate `TrialCollector(...)` calls. Not one line
about progress in `RandomOptimizer`, `GridOptimizer` or `SmacOptimizer` — same
reasoning as the cancellation gate: it is the same decision for all three, and
all three already funnel through the collector. A test asserts that none of them
mentions `progress` or constructs a collector directly, so the next thing every
run needs has somewhere obvious to go.

**And the poll adapts to the trials.** Two seconds however long a trial took
meant a run of ten-minute trials paid three hundred requests for one new row.
The interval is now the median duration of the last few trials, floored at five
seconds — the write interval, below which a poll can only ever answer "nothing
yet" — and capped at thirty so a long-trial run still feels live. It rides in
the fragment's own `hx-trigger`, which poll.js re-reads off each replacement, so
it adapts per swap with no state on either side.

**And it stops while the tab is hidden.** It did not before: browsers throttle
background timers but they do not stop the requests, so a forgotten tab kept
asking for figures nobody could see, for as long as the run lasted. The poll is
now *held* rather than skipped, so it fires the moment the tab comes back
instead of waiting out another interval. Navigating away needed nothing — the
timer lives on the page it was started from, so leaving the experiment ends its
polling by taking the page with it.

**Throttled by the clock, not by a trial count.** The plan says "every *k*
trials"; the trials here span three orders of magnitude, so a count cannot be
right for both ends. At one write per trial a run of 0.07s trials rewrites the
whole result a dozen times a second; a *k* that fixes that starves a run of
ten-minute trials for an hour. Serializing costs O(bytes so far), so a time
bound is what actually bounds the total: one write per five seconds however fast
the trials arrive.

**And no analytics in it.** Importance, interactions, PDP and local effects each
cost a surrogate fit or 2^n_hp coalitions per game per metric, and none of them
may run per trial. A partial result leaves those fields empty, which is a state
the page already handles — it is exactly what a cancelled run produces.

**The channel is the database, not the process.** Under a real huey consumer the
run is in a worker process and nothing in-process could hand the page anything.
Written with the same filtered `Experiment.objects.filter(pk=...).update()` the
final write uses, so an experiment deleted mid-run is not resurrected by its own
run finishing a trial.

### The page side

`poll.js` gains one thing: a `poll:swapped` `CustomEvent` after
`el.replaceWith(fresh)`. It is a fragment swapper and the page had no way to
notice a swap had happened.

Everything else rides in the run-status fragment, which is already fetched every
two seconds and already replaced wholesale — no new endpoint, no second poll
loop. `?trials=` on its own `hx-get` is the page saying how much of the run it
has drawn; `poll.js` re-reads that attribute off the replacement, so each swap
carries the next question with it and neither side keeps any state. The server
builds plots only when that number disagrees with what is stored, which under a
five-second write interval is at most every third poll.

The payloads are shaped as the page's own two maps (`metric_plots[metric][key]`,
`static_plots[key]`) and **merged**, not replaced: only the live figures are in
them, and the analytics and the three fetched figures have to keep whatever they
have. After the redraw the selection highlight goes back on and the click
handlers are reattached, exactly as a view switch already does.

**Which figures move is one declaration.** `Figure.live` drives both the payload
and the redraw, so the two cannot disagree about which figures are cheap enough:

| Live | Not live |
| --- | --- |
| Trial performance, trial duration, hyperparameter space projection, parallel coordinates | Importance, all five interaction figures, partial dependence, local effects |

The **trials table** updates too, by a different route: it is server-rendered
HTML, not a Plotly payload, so the poll carries the rows the page has not seen —
rendered from the same `_trial_rows.html` partial the table itself was built
from — and the page **appends** them. Appending rather than swapping is the
whole point: this is the one thing on the page you are *using* while reading a
chart, and replacing it under someone mid-sort would lose the sort, the page,
the scroll position and the row they had picked. Only the count in the heading
and the row order need putting right afterwards, and a `trials:appended` event
re-applies whatever sort the reader had.

**A page showing "No results yet" reloads itself, once.** It has no figure grid
to merge into, so it is sent the trial count alone and reloads into a page that
has one; from then on it updates in place. Once needs a marker that survives the
reload, hence `sessionStorage` — the alternative is a page that *cannot* render
a grid reloading itself every few seconds for ever. The grid, when it does
render, clears the marker, so a later run of the same experiment gets its own.

## Explanations for a run that computed none

`compute_hp_games` declines outright for a cancelled run, and says why. That was
the right call — pressing Cancel used to still buy the full analytics bill, so
on a wide model you waited minutes for a run you had just stopped — but it made
the emptiness **permanent**: computed once at run completion, or never, with
resuming the run as the only way to get them. Partial writes would have added a
second permanent-emptiness case.

So `experiment_compute_analytics` makes them askable afterwards. One button, on
the importance figure, shown only when the stored result has trials and no
games and no run is in flight. One computation fills all six figures that read a
game, so one button is the honest number.

**Written back onto the result, not returned as a payload.** This is where it
differs from the three fetched figures and why: those answer "which trial, which
hyperparameter", which has no small precomputable set of answers, so each one is
a fresh request. These are exactly the fields the result already carries,
computed from the trials already stored — so filling them in makes the result
what a completed run would have written. The page renders it with no live-update
path of its own, the export carries it, and nobody pays for it twice.

Synchronous, and bounded by the same `eager_analytics_budget_exceeded` guard the
eager computation is held to: anything wide enough to be a problem is turned
away with a reason rather than started and waited on, which leaves this in the
same range as the local-effects fetch the page already makes.

## Verification

```
python -m pytest -q                   1188 passed, 5 skipped (20:22)
python -m pytest -q -m "not slow"     1149 passed, 5 skipped, 39 deselected
python manage.py check                no issues
python manage.py makemigrations --check --dry-run    no changes
```

New: `tests/ui/runs/test_live_figures.py` (14) and
`tests/ui/runs/test_analytics_on_request.py` (8). Two of the fourteen are the
ones worth having: a real Random Search run over iris, watched **from inside the
model**, which is the only vantage point where "the run has not finished" is
certainly true — checking after `execute_run` returns would be satisfied by the
final write, which was always there.

Two existing audit tests changed, both of which exist to make a new declaration
deliberate rather than accidental:

- `test_page_survives_a_round_trip.py` pins every detail-page context key, so
  the three new ones were classified — `trial_count` and `analytics_absent` as
  about this render and this instance, `live_figures` as a catalog declaration.
- `test_the_declared_actions_match_what_the_routes_do` pins every route's
  permission. `experiment_compute_analytics` is RUN: it spends this machine's
  CPU and writes to the stored result, and VIEW would have made reading an
  experiment a way to make it compute.

## Deviations, and what is deliberately not live

- **Time-throttled rather than count-throttled**, for the reason above.
- **The two configuration panels do not update live** — best and selected. The
  plan lists them as trial-based and they are, but they are server-rendered HTML
  with no append-shaped update: a new incumbent replaces the panel rather than
  adding to it, and the selected-configuration panel is answering a click. Both
  are correct at the next reload, which the end of a run already triggers.
- **A partial result carries no `optimizer_state`.** SMAC's `serialize_result`
  reads its runhistory from the live output directory, and a partial result has
  no `smac_output_dir` in its metadata, so it takes the plain path. Only matters
  if the process dies mid-run: what survives is resumable from its trials, but
  without SMAC's own fitted state — which `_replay` rebuilds anyway.
- **The Compute button lives on the importance figure only.** One computation
  fills all six figures that read a game, so one button is right; but with the
  importance figure switched off and an interaction figure on, there is a
  warning and nowhere to press. The narrow case — with *both* display figures
  off the run skips the games for the separate and correct reason that nobody is
  looking, and there is no figure to put a button on either.
- **Redrawing interrupts an interaction with a live figure.** `Plotly.react`
  every five seconds will drop a hover or a zoom on the four figures that move.
  Worth knowing before you decide the interval is right; the fix, if it bothers
  you in use, is to skip the redraw while the pointer is inside the figure.
