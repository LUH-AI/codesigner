# Analytics Phase 0: alternate views (docs/PLAN-analytics.md)

**Status:** implemented, awaiting your sign-off.
**You can now:** switch the Hyperparameter importance figure between a pie
chart, a bar chart and a plain table of the same numbers; and switch the old
Performance-of-Incumbent / Error-over-time figures — now one entry,
"Performance over time" — between four views: trial index or elapsed time on
the x-axis, score or error on the y-axis.

First phase of `docs/PLAN-analytics.md`, the DeepCave-inspired analytics
roadmap. Everything after this phase builds on the mechanism introduced here.

---

## The mechanism

**Concept.** A `Figure` (`ui/figures/base.py`) used to draw exactly one plot.
It can now declare `views` — a tuple of named view keys, in order, the first
being the default — and its `plot(result, metric, view)` classmethod is
called once per view instead of once. A figure that declares no views (every
pre-existing one) is unaffected: `views` defaults to `()`, and nothing about
it changes. `absolute_scale` follows the same split — a flat relayout for a
single-view figure, or a dict keyed by view for one that needs a different
"absolute" range depending on which view is showing.

Server-side, `_detail_context`'s `plot_json` (`ui/views.py`) returns a plain
payload for a single-view figure exactly as before, or `{view: payload, ...}`
for one with views — one JSON round-trip per view, all precomputed, same as
every metric's plot always has been. A new `figure_views` context key
(`{key: [view, ...]}`, only for figures that have any) tells the page script
which payloads are keyed by view without it having to guess from shape.

Client-side (`experiment_detail.html`), the shared plumbing is one function,
`payloadFor(key, metric)`: it looks up a figure's current payload, resolving
through `currentView[key]` (defaulting to the first view) when the figure has
one. Every draw, the click-to-select highlight, and the absolute-scale toggle
go through it, so none of them had to learn views exist. What isn't shared is
the view-selector markup itself — a figure with views owns its own `<select>`
(or two, composed) in its own template, because a flat 3-item dropdown
(importance) and two independent axis pickers (performance) are different
enough shapes that one generic widget would have fit neither well.

## The two entries

**Hyperparameter importance** (`ui/figures/plots.py`,
`ui/templates/ui/figures/hyperparameter_importance.html`): pie (unchanged,
still the default) and a new bar view both come from
`hyperparameter_importance_plot`; "table" has no plot at all — the
builder returns `None` for it on purpose, and the template renders the same
numbers (via a new `panels[].importance` list, sorted descending) directly,
one block per metric, shown only while "table" is the active view. Switching
to "table" would otherwise trip the figure's generic "no data" message
(`draw()` treats a `None` payload as empty) — `syncImportanceTable` suppresses
that specifically for the table view, since table has its own empty state
inside its own block.

**Performance over time** (same two files, `performance_over_time.html`):
replaces `IncumbentPerformance` and `ErrorOverTime`. One builder,
`performance_over_time_plot`, takes `x_axis` ("trial" | "time") and `y_axis`
("score" | "error"); the x-axis toggle only changes what's on the x-axis (elapsed
time is cumulative trial *duration*, deliberately not wall-clock time —
see Deviations), the y-axis toggle only changes the score/error transform and
whether the axis is linear or log. All four views share one visual design —
raw per-trial outcome as markers, the running-best as a step line, a
diamond marking each trial that actually improved it — where the two source
figures previously differed (only one drew per-trial markers; only the other
drew improvement markers) — see Deviations.

## Deviations from the plan

- **Time axis stayed cumulative duration, not real timestamps.** The plan
  noted real wall-clock `starttime`/`endtime` exist in `run_info`
  (`core/optimizers/timing.py`) and called switching to them "strictly more
  correct." On reflection that's wrong for this chart: an experiment can be
  resumed arbitrarily long after its previous run finished (experiments
  persist indefinitely), so a raw wall-clock delta would show that idle gap
  as a dead stretch with no compute spent — exactly the "expensive dry spell"
  the original `error_over_time_plot` docstring cared about getting right,
  except fake. Cumulative `t.duration` (compute actually spent) is what
  shipped instead — no core changes needed either way.
- **The two source figures' visual styles were unified, not preserved
  separately per y-axis choice.** `incumbent_performance_plot` drew every
  trial's raw score as a scatter cloud; `error_over_time_plot` drew only the
  incumbent as a step line plus improvement markers, with no raw per-trial
  trace at all. Keeping that split would have meant "score views look like
  A, error views look like B," which isn't really four views of one thing.
  All four views now draw the same three traces (raw outcome, incumbent step
  line, improvement diamonds); `test_plots.py`'s old note that
  `error_over_time_plot` was "still being refined visually, so deliberately
  uncovered" is resolved by this unification rather than by covering the old
  design.
- **Selection persists across a view switch, resets on a metric switch.**
  Not explicitly asked for, but a bare implementation would have shown a
  click "vanish" on a view switch (each view's precomputed payload highlights
  that metric's best trial by default, same as before any click). A small
  `lastClickedIdx` cache, cleared at the start of every metric switch and
  consulted after every view switch, makes a view switch feel like a
  different way to look at the same selection rather than a reason to lose
  it — while metric-switch-resets-selection (existing, tested behavior) is
  untouched.

## Checklist

✅ `Figure.views` + per-view `plot()` · ✅ per-view JSON payloads +
`figure_views` context key · ✅ generic `payloadFor`/`redraw` client plumbing ·
✅ importance: pie/bar/table · ✅ performance/error: 4-view merge, unified
visual design · ✅ absolute-scale toggle resolves per current view · ✅
click-to-select preserved across a view switch, reset on a metric switch ·
✅ settings collapse cleanly (two old `show_*` keys become one; no migration,
per `docs/PLAN-analytics.md`'s own note) · ✅ tests updated/added
(`test_plots.py`, `test_figures_view.py`, `test_figure_visibility.py`) · ✅
manual browser verification (Playwright, headless Chromium — no
`chromium-cli`/project run-skill available in this environment, so this was
set up ad hoc; worth turning into a proper run-skill if this becomes routine).

## Verification

```bash
python -m pytest -q -m "not slow"   # 716 passed, 5 skipped
python manage.py check
python manage.py makemigrations --check --dry-run   # no model changes this phase
```

Manual: imported `tests/fixtures/test2.ihpo` with its dataset through the
browser, then exercised every view of both entries — pie → bar → table (and
back) on importance; all four trial/time × score/error combinations on
performance, clicking a specific trial marker (not just the default best),
confirming the selected-config panel updated and the highlight/selection
survived switching between all four views, then confirming it correctly
reset on switching the evaluation metric. Also toggled the absolute/relative
scale button on both the score and error y-axes to confirm it resolves the
right relayout per view. No browser console errors. Caught and fixed one bug
this way: the table view initially tripped the figure's generic "no data"
message, since its payload is `None` by design — not visible from the test
suite alone, since no existing test rendered the table view and checked for
the absence of that specific message.
