# One selected trial, shared by every figure that shows all of them

## What it was

Clicking a point meant something on exactly one figure — Trial performance —
and was seen by exactly one other, the Selected configuration panel. The wiring
said so: `attachPerfClickOnce` listened on `figure-performance_over_time`,
`highlightPoint` restyled that one trace with two hard-coded hex colours, and
`lastClickedIdx` was read by the panel fetch and by local ablation.

Every other figure that draws all the trials — the cube, parallel coordinates,
trial duration, local effects, the trials table — drew a point you could see an
interesting thing about and had no way to ask about.

## The declaration

`Figure.selects_trials` sits beside `width`, `per_metric` and `deferred`. Six
figures set it. The rule behind it is not "these ones look clickable": a figure
can offer a trial to click and a trial to highlight exactly when it draws the
trials themselves. Importance, interactions and partial dependence are summaries
*over* trials, and have no point that is one.

`ui/views.py` ships the keys to the page, so the script iterates a declaration
and names no figure.

## The contract

A plot says where its trials are in `layout.meta["selection"]`, built by
`_selection_meta` in `ui/figures/plots.py`:

```python
{"trials": {"0": [0, 1, 2, …]}, "style": "recolor", "segment": 1}
```

Three deliberate choices in that shape.

**Keyed by trace, not a flat list.** A trace not named here is not trials —
Trial performance's incumbent line and its new-incumbent diamonds are drawn from
the same trials but are not a trial each, and parallel coordinates has a trace
that exists only to carry a colour bar. The old code expressed this as
`curveNumber !== 0`, which was correct for one figure and unavailable to the
rest.

**Positions in `result.trials`, not `TrialResult.trial` numbers.** That is what
the trial-panel endpoint and the local-ablation cache already key on, and
resumed runs make the two diverge.

**Named rather than implied by position.** Local effects draws a *sample*
(`_sample_evenly`), so its nth point is not its nth trial. `compute_local_effects`
now carries the position in each row for exactly this. Every other figure could
have got away with identity; this one is why the contract is explicit, and one
shape beats one shape plus an exception.

`segment` is how many points make up one trial — 1 everywhere except parallel
coordinates, where a trial is a whole polyline. So a single rule reads every
figure: the trial at point *p* of trace *t* is `trials[t][p // segment]`.

### `style` is not a taste

| Figure | style | why |
| --- | --- | --- |
| Trial performance | `recolor` | marker colour carries nothing else |
| Trial duration | `recolor` | one flat colour today |
| Configuration cube | `outline` | fill colour **is** the score |
| Local effects | `outline` | fill colour **is** the sign of the effect |
| Parallel coordinates | `line` | a trial is a polyline, not a point |

Painting the selected point in a highlight colour on the cube would be a claim
about how well that trial did. So there it is a ring — except in 3D, where
Plotly draws no marker outline at all and the size change carries it alone.


## The palette

Four colours across every figure, each meaning one thing. Held by convention
rather than by machinery — nothing enforces it — but a reader who learns a
colour on one figure should not have to unlearn it on the next.

| Colour | Means |
| --- | --- |
| `MARKER_COLOR` (blue) | a trial, an observation, an unemphasised line; and the positive side of a signed quantity — synergy, "helped" |
| `SELECTION_COLOR` (orange) | the selected trial, and nothing else |
| `ACCENT_COLOR` (emerald) | the best-so-far: the incumbent line, the trials that improved it, and the partial-dependence mean — the same idea one level up, what the surrogate makes of the run rather than one point in it |
| `NEGATIVE_COLOR` (red) | the negative side of a signed quantity: redundancy between hyperparameters, a value that hurt |
| `_INTENSITY_SCALE` | anything scaled to the metric |

Three meanings meet on the performance figure alone — a trial, the best-so-far,
and the one that is selected — and they used to be two colours between them. The
incumbent line carried no colour of its own at all and took Plotly's second
colorway entry, which is the same orange the selection is drawn in: the
highlight was competing with a line drawn through every point on the chart.

`_INTENSITY_SCALE` covers the configuration cube and parallel coordinates, which
build their colours by different routes — one hands the array to Plotly, the
other interpolates per line because a Scatter line takes a single colour — so a
test pins them to the same ramp. That is where the two would drift apart.

## The bus

`selectTrial(metric, idx)` is the only thing that changes the selection.
Every figure's click handler and every table row comes through it, so they
cannot disagree about what is selected or forget to tell the panel.
`showSelection` puts the highlight on everything; `applySelection` does one
figure and is what a late arrival (local effects is fetched) or a redraw (a
metric or view switch) calls for itself.

The two colours come from `plots.py`'s `MARKER_COLOR` / `SELECTED_COLOR`
through a `json_script`. The page draws the highlight itself — a restyle, not a
rebuilt figure — so restating them in the script was one place for the page and
the plots to disagree about what "selected" looks like.

A metric switch still forgets the click and falls back to that metric's best
trial, unchanged.

## Parallel coordinates left `go.Parcoords`

Parcoords emits no click, has no per-line width, and has no way to reorder what
sits on top. All three of the things asked for are unreachable through it, so
the figure is now scatter traces: one per trial, because a Scatter line takes a
single colour and colouring by score is the whole reading of the figure — "a
cluster of high-scoring lines bending through the same region of one axis says
that axis matters".

What survives untouched is the part that took the thought: the tunability
ordering, the `log10` recoding with native numbers on the ticks, and the
categorical recoding to sorted-unique positions. What changes is what is done
with those dimensions — every axis is normalized to one shared 0–1 scale, drawn
as a vertical shape with its own values as pixel-offset annotations beside it.

Three traces follow the trials: one carrying nothing but a colour bar (with a
colour per trace there is no array for Plotly to build one from), and one empty
line added last, which the page fills by slicing the selected trial's own trace.
Last, because trace order is fixed at draw time and that is the only way to put
one line over the others.

**What it costs**, stated plainly: Parcoords' two native gifts, dragging an axis
to reorder it and dragging along one to filter the lines through it. Nothing
else on the page replaces either.

## The trials table

Rows carry `data-trial-idx` and `tabindex`, and the highlight is a class. The
identity has to be the attribute rather than the row's position, because the
outcome columns sort by reordering these very rows — after one click on a header
the nth row is not the nth trial.

## Verify

```
python -m pytest -q -m "not slow"     # 941 passing
python manage.py check
python manage.py makemigrations --check --dry-run
```

The contract is checked by construction: for every builder that declares it,
each named trace's list is as long as that trace has points and every entry is a
position that exists in the result. A figure that got this wrong would highlight
the wrong trial rather than fail.

The bus is JavaScript, which this suite does not run. In the browser:

- click a point on each of the five plots and confirm the other four, the table
  and the panel all move to the same trial
- click a row in the table and confirm the five plots follow
- confirm the cube's colour bar is back (it had been lost — a `showscale: false`
  set on a copied marker that then became the trace, so the flag was read back
  and carried forward forever), that the ring is visible against the score
  colours in 2D and the size change reads in 3D, and that the highlight survives
  an axis change
- confirm the parallel-coordinates selected line is thicker and over the others,
  and that clicking near a vertex picks the right trial
- click a point on the cube with a third axis chosen: a 3D trace reports
  `pointNumber` and no `pointIndex`, so it named no trial until the handler read
  both
- switch metric and confirm every figure resets to the new metric's best
