# A failure worth looking at

**Status:** implemented, awaiting your sign-off.
**You can now:** open an experiment whose trials include a fabricated crash and
a fabricated timeout, and see what Step 1 and Step 2 actually draw — without
waiting for a model to break.

Closes Step 5 of `docs/plan_today.md`. **It found a real bug**, which is the
argument for the step.

---

## The fixture

`scratch/fabricate_failed_trial.py`, not committed (`scratch/` is gitignored),
exactly as the plan asks:

```
.venv/bin/python scratch/fabricate_failed_trial.py [source-pk]
```

`pk = None` plus a fresh identifier is the whole copy — the FileFields go on
pointing at the same stored dataset, which is right for a read-only look. The
original is untouched, verified. The copy has no `Run` rows and a `created_at`
of now, both expected.

Every key it writes was read off `serialize_result`/`deserialize_result` rather
than guessed: `cost` is `1.0 - score`, `scores` is the per-metric dict the page
reads, and `status`/`additional_info` are in `RUN_INFO_KEYS`, which is what makes
them survive the round trip.

Two departures from the plan's sketch, both to make the fixture say more:

- **Two trials, not one.** Step 2 added a second kind of failure and the two are
  meant to render differently: a crash carries a traceback and the panel links
  to it; a timeout carries none — the deadline expired on this side, so a
  traceback would show the client waiting rather than the model working — and
  the panel must give the reason without offering a link to nothing.
- **It picks its own trials.** A trial that *set* the incumbent cannot be
  zeroed: every later entry carries `incumbent_score`, so the incumbent line
  would end up quoting a score no trial has. The script takes only trials whose
  incumbent is inherited from the one before, and never the one `best_config_id`
  names.

## What looking at it found

Everything the plan asked to confirm was correct except one thing, and that one
thing could not have been caught by a test in this repo:

**Selecting any trial flattened every failure mark on two of the figures.**
`applySelection`'s `recolor` branch repainted a trace wholesale — every point to
either the selection colour or the plain blue, every size to 6 or 13 — which
wiped the per-point styling the server had drawn underneath. Measured in the
browser, before and after:

| | before | after |
| --- | --- | --- |
| Trial duration, failed bars | `#636EFA` — **the tint was gone** | `rgba(255, 43, 43, 0.45)` |
| Trial performance, failed points | size 6 | size 11 |

The duration bar lost its tint outright. The crosses survived only by accident:
`marker.symbol` and `marker.line` are not touched by that branch, so the glyph
stayed while its size went. The configuration cube was unaffected — it uses the
`overlay` style, which slices the selected point into a trace of its own instead
of repainting.

Pre-existing, from Step 1: the failure marks were added to figures whose
selection repaint had always been wholesale, and nothing put the two together
until they were on screen at once.

**The fix.** `draw()` now snapshots each trace's marker arrays from the payload
— the server's own drawing, before anything on the page touches it — and the
repaint restores from that instead of from a default. A selected failure keeps
its cross and its size and turns its *outline* to the selection colour, which is
what `_failure_marks` says should happen: the shape carries the failure and the
colour carries the selection.

## Confirmed in the browser

Driven with Playwright against the dev server, reading the actual trace data
rather than squinting at pixels:

- **Trial performance** — two `x-thin` crosses at trials 12 and 29, ringed in
  `NEGATIVE_COLOR`, sitting at 0.0.
- **Hyperparameter space projection** — the same two, size 11, symbol `x-thin`.
- **Trial duration** — both bars tinted, at their real durations. A failed trial
  did take that long to fail.
- **Trials table** — both rows tinted, trial number in red with a ✕, zeros
  across every score column, duration intact.
- **Parallel coordinates** — 28 lines for 30 trials, omitting exactly 12 and 29.
  (The count needed care: there are 30 traces, two of which are the colour-bar
  carrier and the selection highlight.)
- **The sidebar** — trial 12 gives the reason *and* a traceback link; trial 29
  gives the reason and **no link**. `trial-traceback/?idx=11` serves the
  stacktrace as plain text; `?idx=28` is a 404.
- No JavaScript errors on the page.

## Verification

```
python -m pytest -q                   1188 passed, 5 skipped (20:22)
python -m pytest -q -m "not slow"     1149 passed, 5 skipped, 39 deselected
python manage.py check                no issues
python manage.py makemigrations --check --dry-run    no changes
```

Three regression guards went into `tests/ui/results/test_failed_trials_on_the_page.py`.
They assert the repaint reads what was drawn rather than painting a constant,
that the snapshot is taken from the payload rather than the element (which by
the second selection already carries the first), and that a selected failure
turns its outline. Source-level assertions, which is weaker than driving a
browser — there is no browser-test harness in this repo and adding one is a
bigger decision than this step. They would have caught this regression, which is
the bar they were written to.

## Worth knowing

- **This is the class of bug that only shows up in the app.** Every unit test
  around the failure marks passed the whole time: the server was building the
  right figure. The page then painted over it.
- **The fabricated experiment is in the dev database** (pk 13, `da256049`,
  "test3 (fabricated failures)"). `db.sqlite3` is gitignored, so it travels no
  further than this machine. Delete it whenever you like; the script rebuilds it
  in a second.
