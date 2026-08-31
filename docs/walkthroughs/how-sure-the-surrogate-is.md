# How sure the surrogate is

**Status:** implemented, awaiting your sign-off.
**You can now:** put two hyperparameters on the configuration cube's axes and
ask for a field under the trials showing where the surrogate's own trees agree
with each other and where they do not.

Closes Step 4 of `docs/plan_today.md`, which asked for a design pass rather than
a sketch. This is that pass, and what it decided.

---

## The estimate

`fit_surrogate` already builds a `RandomForestRegressor` — for partial
dependence, and for the fallback rung under every global HyperSHAP game. Each of
its hundred trees predicts a point separately, and how far apart those
predictions sit is how much the forest disagrees with itself there. **No new
fit, no second model.** The uncertainty was already inside the thing that was
already being built.

**Standard deviation, not variance.** The plan says variance, and variance is
the same information; but it is in squared metric units, and nothing can be read
off a colour bar labelled "squared accuracy". The square root is in the metric's
own units, so "the trees disagree by 0.04 accuracy here" is a sentence.

**What it is not, said out loud.** Tree disagreement is not a calibrated
posterior, and it fails in a specific direction worth knowing: far outside the
region the trials cover, every tree drops into the same extreme leaf and agrees
*completely* — so the field goes confident exactly where there is no data. It
reads as "where do the trees disagree", which is a real thing to see over a
search's own footprint. It does not read as "where might the truth be", and the
docstring says so rather than leaving a reader to find out.

## A slice, not an average

The plan says "the rest at a reference" and leaves the reference open. It is the
**best trial's configuration**.

The alternative worth weighing was the PDP-shaped one: average over every
trial's own other values, the way `compute_partial_dependence` averages its ICE
curves. Two things against it. It costs `n_trials × n_points²` rows through
every tree rather than `n_points²` — at 25 trials and a 20-point grid that is
250,000 tree-predictions instead of 40,000, per axis pair, per look. And the
deciding one: a plane through the incumbent passes through a configuration that
was actually evaluated, so the slice is anchored where the forest has real data,
while a plane through the config-space default may sit somewhere the search
never went and report uniform ignorance across the whole picture.

The default *is* the right reference for tunability — it is what "what could
tuning gain you" is measured from. This is a different question and takes a
different answer.

## Axes only, and only at two

The plan suspected this and it turns out to be firmer than "possibly":

- **PCA and PLS have no way back.** `core.projection.project` returns
  coordinates and labels and discards the fitted transform, so there is no
  inverse to carry a grid in component space back to configurations. Even with
  one kept, the plan's own objection stands: the points it landed on need not be
  valid configurations.
- **Three axes** would be a volume through a point cloud — an isosurface or a
  voxel field read through the very scatter it is meant to inform. That is a
  worse picture than none.
- **One axis** is a strip whose vertical axis carries nothing, so there is no
  second dimension for a field to vary in.

So the field appears when the axes view has exactly two hyperparameters named,
and is silently absent otherwise — not a state to explain, since the figure is
simply not showing something it could sit under.

## Not green, and not blue either

`ACCENT_COLOR` means "the best so far" on every figure that uses it, and a field
underneath the points is not that. The plan says so.

The second half is the one the plan does not mention and that matters just as
much: `_INTENSITY_SCALE` ramps **blue** for the metric — on this very figure, on
the points sitting directly on top of the field. A reader who has learned that
more blue is a better score must not meet a second blue ramp meaning something
else two millimetres below it.

So `UNCERTAINTY_SCALE`: a violet the palette does not otherwise use, translucent
at both ends because this is scenery behind the trials and must never compete
with them for the eye. It ramps **up** with uncertainty — the un-inverted
quantity, so pale is where the trees agree. Inverting it to "confidence" would
have read better against the score ramp and would have invited exactly the
false reading the docstring above refuses.

The two ramps still sit on one figure and still point opposite ways in valence
(more blue is better, more violet is worse). They are separated by saturation, by the
white rings on the points, and by a second labelled colour bar set outside the
score's — outside rather than opposite, because a bar on the left of a plot
reads as belonging to the y-axis, and the figure's right margin is widened to
make room for it. Worth looking at in the browser before you agree it is
enough; I have not, since it needs a run wide enough to have two hyperparameters
worth putting on axes.

## Deferred, like partial dependence

It is per *pair* — a model with six hyperparameters has fifteen — and a reader
looks at one. So it follows the deferred contract exactly: declared on the
figure, given its own `autocompute_surrogate_uncertainty` setting, off by
default, a "Show surrogate uncertainty" button in its place, cached per
(metric, x, y) so going back to a pair already computed is instant.

Drawn as a heatmap trace appended **last** and still rendered **underneath**:
Plotly's cartesian layer order is `imagelayer, heatmaplayer, …, scatterlayer`,
so trace order does not decide depth here. Verified in the bundled
`plotly.min.js` rather than assumed — and appending is what matters, because
`layout.meta.selection` names trace 0 as the trials and trace 1 as the
highlight, and inserting a trace in front of them would silently renumber both
and break click-to-select.

## Verification

```
python -m pytest -q                   1188 passed, 5 skipped (20:22)
python -m pytest -q -m "not slow"     1149 passed, 5 skipped, 39 deselected
python manage.py check                no issues
```

New: `tests/ui/results/test_surrogate_uncertainty.py` (14). The one worth
naming asserts the field is the trees *disagreeing* and not their mean — a field
of predictions would just be the metric again, which the points already carry.

## Known, and for your call

- **A live update leaves the field stale.** New trials arrive under a field
  computed from fewer of them. Refitting per poll is precisely the cost Step 3
  excluded, and clearing it would make the button useless during a run — so it
  stands as a snapshot of the surrogate as of when it was asked for. Press it
  again for the current one.
- **The reference is the incumbent, which moves.** Ask twice during a run and
  the plane can be a different plane, because the best trial changed. Correct,
  and worth knowing before comparing two looks.
- **20×20 is hard-coded.** `compute_surrogate_uncertainty` takes `n_points` and
  the endpoint does not expose it. A grid setting is easy to add if the
  resolution turns out wrong in use; guessing at one before that seemed worse
  than leaving it.
