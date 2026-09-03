# Six adjustments: layout, defaults you can read, and where the time went

Follow-up to [hypershap-plots.md](hypershap-plots.md). Six things asked for
after using the analytics page for real, plus one question about a run that took
longer than it looked like it should have.

## 1. The interactions graph fitted its card

The Möbius graph was drawn with `scaleanchor` tying the y-axis to the x-axis, so
Plotly letterboxed a square figure inside a card that is wider than it is tall.
The circle stayed round and everything around it was empty.

The anchor is gone and the axis ranges are set explicitly instead — `x` over
`[-1.6, 1.6]`, `y` over `[-1.25, 1.3]`, margins at 10. The node ring is placed by
`cos`/`sin` at even angles, so it is round because the ranges are chosen to make
it round, not because an axis constraint forces it. Labels sit outside the ring,
which is why the ranges are not symmetric about the circle.

## 2. Trial performance sits beside the selected configuration

"Performance over time" is now **Trial performance**, is a `FULL`-width figure,
and is third in `FIGURES` — immediately after Selected configuration. The two are
read together: you click a point on one and the other tells you what you clicked.

The layout used to be two grids, one for half-width figures and one for
full-width, which meant a `FULL` figure could never sit next to a `HALF` one and
the pair could not be kept adjacent. There is now one `.grid-2x2` and a
`.slot.wide` class spanning both columns:

```css
.grid-2x2 > .slot.wide { grid-column: 1 / -1; }
```

Ordering the catalog now orders the page, which it did not before.

### The reason it didn't look like that

The markup was right the whole time — the layout tests assert the spanning class
lands on the right slots, and a rendered page confirms it — but the span itself
is a stylesheet rule, and the stylesheet is served under a plain filename:
WhiteNoise's non-manifest storage keeps it that way so `{% static %}` works
without a prior `collectstatic`. A browser holding the previous `app.css` has no
way to tell it apart from the new one, so it kept laying the page out by rules
that were no longer the rules: no `wide`, everything tiling two-up, and the
figures that used to sit outside the grid at full page width suddenly at half.

`{% asset %}` (`ui/templatetags/assets.py`) puts the file's modification time in
the query string. Edit the file and every page asks for a URL nobody has cached;
leave it alone and the cached copy keeps being used. Both templates that link a
stylesheet use it — the app shell and the sign-in page, which does not extend it.

## 3. Optimizer fields say what the default actually is

Every optimizer setting that fell back to the search strategy's own default said
"search strategy default" — true, and useless while deciding what to type.

`SMACOptimizer.strategy_defaults()` reads the actual values out of SMAC's own
signatures, keyed by parameter and then by strategy, and
`ui/optimizer_labels.py` turns them into a placeholder (`0.08447` — four
significant figures, thousands separators) and a range caption (`0–1`). Both are
shown under the field, because `min`/`max` alone only speak up once a value is
already wrong and a placeholder is easy to miss inside a field you are about to
type into.

The defaults for *every* strategy ride along on the field as
`data-strategy-defaults`, so changing the strategy selector swaps the
placeholder without a round trip.

## 4. Evaluation is a scheme and a number, not a menu

Trial evaluation was one dropdown of preset combinations, which meant 7-fold and
a 70/30 split were simply not offerable. It is now a selector —
**Cross-validation** or **Dataset split** — beside a single number whose label,
bounds, step and default follow it:

| Scheme | Number means | Range | Default |
| --- | --- | --- | --- |
| Cross-validation | Folds | 2–20 | 5 |
| Dataset split | Share held out | 0.05–0.5 | 0.2 |

One number rather than two, because only one applies at a time and an inactive
second field invites a value that goes nowhere. `EVALUATION_SCHEMES` in
`ui/forms.py` is the single definition; the template's script reads it through
`json_script` rather than restating it, and a test asserts the page actually
ships that table.

Storage does not change: `cv_folds` stays the source of truth for which scheme is
in use (0 is a split, 2+ is cross-validation), and `test_size` is carried under
either scheme so switching later has somewhere to start from.

Out of range is clamped, not refused. These are two ends of one continuum and
every value between is meaningful, so there is no typo here worth failing a form
over — 40 folds becomes 20, and the experiment is still created.

## 5. Where the 44 seconds went

A 128-second run reported 84 s of trials and 44 s of "overhead". That is
expected, and the label is the misleading part.

Reproduced cold at the same shape — SMAC/gp, random forest, 5-fold, wine, 20
trials — for 54.5 s total and 13.0 s outside the trials, which decomposes as:

| Stage | Time | What it is |
| --- | --- | --- |
| SMAC ask | 10.6 s | GP refit + acquisition optimization |
| Importance analytics | 2.3 s | The eager games, at run completion |
| SMAC tell | 0.1 s | Recording the observation |
| Dataset load + fold split | ~0 | Once per run |
| Serialize | ~0 | Once per run |

So roughly **80% of "overhead" is the optimizer deciding what to try next** —
the search doing its job, not bookkeeping.

Run 48's share was higher than the cold reproduction because it was a resume:
`trial_offset` was 19, so the GP was fitting over 19→39 points (cubic in *total*
history, not in the trials this run added), `_replay` re-read the 19 prior
trials, and the analytics ran over all 39. Meanwhile `trial_seconds` counts only
the 20 new trials' fits. The two numbers are measured over different populations,
which is the whole of the discrepancy.

Nothing was changed for this. It is worth knowing that the number grows with
accumulated history rather than with run length, and that "overhead" reads as
waste when it is mostly the surrogate.

## 6. Intensity, not a two-colour ramp

The configuration cube and parallel coordinates used Viridis, which travels
between hues. Reading a hue as an ordering is a learned skill; reading "darker
means more" is not.

`_INTENSITY_SCALE` is a five-stop single-hue blue ramp (`#EEF1FE` → `#2B3AA8`).
Same data, same direction, one dimension of variation.

## 7. Pickers read as one control

Two separate causes, one symptom — controls that belong together sitting one per
row.

In the figure cards it was the page's own base rule: `select` is stretched to
`width: 100%`, which is right for a form field and wrong for a picker inside a
card. Three full-width selects become three rows, and the cube's axis pickers
stop reading as one choice about one figure. They now sit in a `.selectors` row
that sizes them to their contents — used by all five figures that have pickers,
so a select added outside one is the thing that stands out. A test asserts no
select in any figure card escapes its row.

On the new-experiment form it was the stylesheet cache above: `.field-row` is
flex, and without it two `div.field` children stack.

## 8. The cube starts empty

All three axis pickers now start at **None**, and how many are filled decides
what is drawn: nothing, a strip, a plane, a cube. The option says only "None" —
what it does to the figure is visible in the figure.

The server used to pick the first two hyperparameters in config-space order,
which answered the figure's own question before it was asked; "first" only ever
meant that something had to be. So `configuration_cube_plot` now ships the trace
with `customdata`, the marker colours and `meta` (`hp_names`, `log_hps`) but
`x`/`y` empty and no axis titles or types, and `applyCubeAxes` fills the axes
from whichever columns are picked. It already ran after every redraw, so a
metric switch still lands on the reader's choice rather than a server default.

The slots are three chances to name a hyperparameter, not three fixed axes:
picking only the third draws the same strip as picking only the first. At one
axis the points sit at a constant height with the y-axis hidden, because the
score is the marker colour at two and three axes too and moving it onto y would
make the one-axis view the only one where the axes mean something else.

One hyperparameter is now enough to draw. It used to return `None` below two,
which left a one-hyperparameter model with no cube at all.

### The empty state has to keep the data

The first cut drew the empty state as `Plotly.react(el, [], ...)`, which is a
one-way door: `el.data` is the only copy of `customdata`, so handing Plotly an
empty trace list threw the values away, and every later choice hit the
`!el.data.length` guard at the top of `applyCubeAxes` and returned. Since the
figure now *starts* empty, that ran on page load and no axis choice ever drew
anything.

So the empty state keeps the trace and empties its axes instead: `x` and `y` go
to `[]`, both axes go `visible: false`, and the colour bar is switched off so it
isn't a legend for nothing. The trace is rebuilt from `customdata` on every call
rather than edited in place, which is also what makes 2D↔3D switching a plain
re-render.

## Verify

```
python -m pytest -q -m "not slow"     # 926 passing
python manage.py check                # clean
python manage.py makemigrations --check --dry-run
```

`0017_experiment_test_size.py` adds the field item 4 needs; nothing else
migrates.

In the browser, since the suite runs no JS: switch the evaluation scheme and
confirm the label, range, help and default all follow it, and that a number
typed under one scheme comes back when you switch away and return; change the
search strategy and confirm the placeholders follow; and walk the cube through
all four states — none, one, two, three axes — then switch metric and confirm
the choice survives it.
