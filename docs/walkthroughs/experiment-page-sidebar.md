# The experiment page: a sidebar that stays

## What it was

One scrolling column. A run box, a metric selector, an identity strip and eleven
figures, with the panel that tells you what you just clicked somewhere in the
middle of them — so clicking a point at the bottom of the page meant scrolling
back up to read the answer. The inner sidebar held one thing, the list of
experiments, and scrolled away with everything else.

## Three sections

The sidebar is now a stack of sections, each one a decision the figures below are
an answer to:

| Section | Holds |
| --- | --- |
| **Experiment Selection** | ＋ New, 📂 Load, the experiment list, See all… — what the sidebar already held |
| **Experiment Evaluation** | the metric selector, always visible; what bounds the next run behind a fold; the run-status box in its place while a run is in flight |
| **Selected Configuration** | the trial currently selected, wherever it was selected from |

The first is the sidebar's own, in `templates/base.html`, and shows on every
page. A page with more to ask adds its own through a new `sidebar_sections`
block; the experiment page adds two.

And it stays put — `position: sticky; top: 0; height: 100vh; overflow-y: auto`,
the same treatment `nav.rail` already had. A control that scrolls out of view
while you are using it is one you have to go and find.

### The metric selector is the Evaluation section

Not a field inside a form that happens to be there: the metric drives every
figure on the page and is chosen constantly, while what bounds a run is set once.
So the selector is the section, and the stopping criteria and the Run button are
in a closed `<details>` beneath it.

The selector nonetheless stays *inside* the `<form>` — it is still the run's
`optimize_metric` field, which is what `_metric_select.html`'s `field_name`
switch has always been for. One `#metric-select` on the page, as before, so the
page script needed no change at all.

While a run is in flight the status box replaces the fold rather than joining
it. There is nothing to configure until it finishes, and the alternative is a
form offering to start a run you cannot have.

### Selected Configuration is still a figure

`Figure.in_sidebar` joins `selects_trials` and the rest. `SelectedConfiguration`
sets it; `ui/views.py` splits the shown figures into `grid_figures` and
`sidebar_figures`, and each renders where it belongs. The figure keeps its
catalog entry, its visibility setting, its per-metric `panels` and every id the
metric switch and the trial-panel fetch already reach for — only its address
changed.

### Always a difference

The panel used to omit the delta row when the selected trial *was* the best one,
so the panel changed height as you clicked from trial to trial. It now always
carries the number, reading `0.0000 (Incumbent)` for the best. "No difference"
and "a difference of zero" are the same fact said two ways, and only one of them
is on the page.

## Numbers: four significant figures

One rule, in `ui/formatting.py`, shared by the plot builders and a `{{ x|sig }}`
template filter. Django's `floatformat` counts *decimal places*, which is the
wrong unit for these numbers: the same four places that suit a score between 0
and 1 write a small hyperparameter effect as `0.0000`, and write `0.8` as
`0.8000` — three digits it has not earned.

Integers keep every digit and gain thousands separators. 128 estimators rounded
to 130 is a different experiment, and an integer has no rounding to hide behind.
Anything that is not a number comes back as itself, because this runs over
hyperparameter values, which are as often a kernel name as a float.

It is a display rule and only that. The full float is what is stored, what the
optimizer sees, what an exported `.ihpo` carries — and what the trials table
keeps in `data-sort`, so the sort orders the real numbers rather than the
rounded ones.

## Five interaction figures, not five views

`HyperparameterInteractions` had a selector offering heatmap, top pairs, graph,
coalitions and by-order. They are now five figures. They answer different
questions and are wanted side by side — the heatmap is a grid you scan, the graph
a shape you recognise, the coalitions a ranked list you read — and behind one
selector only one could ever be on the page at a time. Each has its own
visibility setting now.

A shared base, `_Interactions`, holds what they have in common: `per_metric`, and
a `source` saying which stored field the reading comes out of. The heatmap and
the top pairs read the order-2 FSII grid; the graph, the coalitions and the
orders read the Möbius decomposition. Those are not the same numbers and
deliberately so — see `BaseOptimizer._shared_exact_computer`.

The warning is one warning: it is the tunability game's caveat and all five read
that game, so each carries a copy and `syncInteractionsWarning` sets them
together. One that appeared only on whichever figure was showing would be one
most readers never saw.

## Page order

Catalog order is page order, and the widths do the pairing:

| Row | |
| --- | --- |
| Best configuration · Hyperparameter importance | |
| Trial performance | spans |
| Interactions: heatmap · Interactions: top pairs | |
| Interactions: graph · Interactions: coalitions | |
| Interactions: by order · Trial duration | |
| Configuration cube · Partial dependence · Local effects · Parallel coordinates · Trials | each spans |

`SelectedConfiguration` is still first in the catalog — that is where its
settings checkbox sits — but it renders into the sidebar, so its position in the
list is not a position on the grid.

## The trials table is paged

Fifty rows to a page, with the page selector and the rows-per-page field sharing
a footer beneath the table: the same decision at two scales. The pager stays
hidden until the table is longer than a page, since a selector for one page is a
control that does nothing — measured against the *smaller* of the default and the
current size, so raising the field past the row count does not take the field
away with it.

Everything in it counts rows in the order they are currently in, because the sort
reorders those very nodes. The sort announces itself with a `trials:sorted`
event rather than calling the pager, so neither script knows about the other.

Selecting a trial pages to it. A highlight on a page nobody is looking at is not
an answer, and the selection can come from any figure on the page.

## Three fixes alongside

**Selection is taken from hover, not from `plotly_click`.** That event serves
three kinds of figure here and does not serve them equally. On a 3D scene the
default drag is an orbit and the click is almost always eaten by it, so the
configuration cube stopped selecting the moment a third axis was chosen — twice
diagnosed as something else, twice wrongly. On parallel coordinates the target is
a line and a click has to land within a few pixels of one of its points.

Hover is the forgiving half of the same machinery: it has a radius
(`hoverdistance`), it works inside a scene, and it is already how a figure says
what is under the pointer. So the click is a DOM click on the graph div and the
*point* is whatever was last hovered — one path for every figure, 2D and 3D. A
pointer that travelled more than a few pixels between press and release is a
rotation or a zoom rather than a selection, and is ignored.

**Clicks are wired before anything that can fail.** `attachSelectionClicks()` now
runs immediately after the redraw loop rather than at the end of `showPlots`.
Whether the page can be clicked at all should not depend on every refresh below
it succeeding — the reported symptom was that a broken cube took the other
figures' selectability with it, which is what that ordering allowed. `showSelection`
guards each figure for the same reason: they are independent answers to one
question.

**Parallel-coordinates lines were also hard to hit.** A polyline carried one point
per axis, so there was nothing to hover over the whole middle of every span. Each
span now carries four points, sitting exactly on the straight segment, so the
picture is unchanged and there is something to find everywhere along it.
`segment` in the selection meta follows, so the click-to-trial rule divides out
the same as before.

**The incumbent markers blocked the trials beneath them.** They are drawn over
the trials they mark and are bigger, so they sat between the reader and exactly
the trials most worth selecting — and being incumbents rather than trials, a
click on one names nothing anyway. `hoverinfo="skip"` takes them out of
hit-testing entirely and the point beneath answers instead. Nothing is lost: it
is the same trial and it says so.

**The interactions heatmap was orange, and backwards.** It asked for Plotly's
named `"RdBu"`, which the bundled `plotly.min.js` defines as blue at 0 running to
red at 1 — so with `zmid=0` it put *positive* on red, the reverse of the graph,
the top-pairs bar, the waterfall and the beeswarm, with an orange band around
0.6–0.7 belonging to no meaning at all.

Worth recording why that is easy to disbelieve: Python's own scale of the same
name (`_plotly_utils.colors.diverging.RdBu`) runs the other way, red to blue. The
figure ships the string and the browser resolves it, so it is plotly.js's
definition that renders — reading the Python side tells you the opposite.

It now uses `_SIGN_SCALE`, built from the palette: `NEGATIVE_COLOR` → white →
`MARKER_COLOR`. A test holds the heatmap and the graph to the same reading of
sign, since they are two views of one set of numbers.

## Verify

```
python -m pytest -q -m "not slow"     # 960 passing
python manage.py check
python manage.py makemigrations --check --dry-run
```

In the browser, since none of the layout or the selection bus is exercised by the
suite:

- scroll the figures and confirm all three sidebar sections stay put
- collapse and expand the run configuration; start a run and confirm the status
  box takes the same place
- select the incumbent and confirm the panel reads `0 (Incumbent)`, and that
  the hyperparameter rows below it are all there — the sections are
  `flex-shrink: 0` because a squeezed one clipped its table instead of scrolling
- page through the trials table, change the rows per page, sort a column, and
  select a trial on another page
- click a cube point with a third axis chosen, then switch back to two and click
  again; rotate the scene and confirm the rotation does not select anything
- click a parallel-coordinates line halfway between two axes
- click a trial that is also an incumbent on Trial performance
