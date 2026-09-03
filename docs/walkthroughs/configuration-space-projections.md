# Projections: PCA and PLS beside the cube

## The same operation, over a larger set of candidates

Not two families. Both the cube and the projections keep two or three linear
coordinates of a p-dimensional space and drop the rest; picking axes is picking
a *coordinate* subspace, and PCA picks any subspace. The cube is this problem
with the search restricted to the axes. The differences are of degree, and there
are three of them.

**How much is lost.** On standardised data every hyperparameter carries one unit
of variance, so any two axes retain exactly two of the p units. PCA's first two
components retain λ1 + λ2 — at least two, and equal to two only when the
standardised hyperparameters are entirely uncorrelated. The same choice over a
larger candidate set can only do better, and on a search whose hyperparameters
move together it does much better.

**Whether a distance survives.** The cube plots raw values, so the length of a
line between two of its points mixes `n_estimators` with `min_samples_split` and
is not a length. The projections standardise every column first, which is what
makes the underlying metric coherent, and then choose the subspace that loses
least of it — so a plotted distance approximates a real one, with the discarded
variance as the size of the error. Two changes, and only the second is about
PCA: standardising the cube's axes would fix half of it, at the cost of the
thing the cube exists for, which is that a position on an axis is a
hyperparameter value you can read.

**What is being maximised.** PCA minimises what is lost from the configurations.
PLS does not — it maximises covariance with the *score*, so it will drop a
high-variance direction the metric does not care about and keep a low-variance
one it does. Its first component is the direction through configuration space
that moves the metric, which is a different question from where the search
spread out, and usually the one being asked.

| View | Chooses the subspace that | Supervised | Axes readable |
| --- | --- | --- | --- |
| Axes | you name | — | yes, they are hyperparameters |
| PCA | loses least of the configurations | no | as weighted sums |
| PLS | covaries most with the score | yes, by the metric | as weighted sums |

The general name for the family is **dimensionality reduction**; a 2D/3D scatter
of one is a **projection** or **embedding** plot.

### Where MDS would sit, if it is ever added

Not here, and the reason is worth recording, because on the face of it DeepCAVE's
Footprint plugin looks like a fourth entry in the table above.

*Classical* MDS on Euclidean distances **is PCA** — checked, not assumed: the
double-centred distance matrix has the same leading eigenvectors, and the two
give pairwise-identical pictures up to rotation. Adding it as another view would
be a slower way to draw the one already there.

MDS earns its place only with a distance that is not Euclidean-on-the-encoding —
one that handles categoricals and conditional hyperparameters as a configuration
space actually has them, which is why Footprint uses it. And at that point it is
categorically different from these three on four counts, every one of which
breaks something this figure relies on:

- **No out-of-sample map.** PCA and PLS give a matrix, so a new trial is
  projected without refitting. SMACOF gives coordinates for *these* points only:
  one more trial moves every point.
- **Stochastic.** It starts from a random init and converges to a local optimum.
  Measured at 100–1,000 points, two seeds give pictures whose pairwise distances
  differ by more than the diameter of the plot. It needs a seed, and re-running
  is not guaranteed to reproduce the picture.
- **No readable components.** A PCA or PLS component is a weighted sum of
  hyperparameters. An MDS axis means nothing individually — only distances do —
  so the "2D / 3D" selector and the component labels have nothing to say.
- **O(n²).** A full distance matrix. Measured: 0.03 s at 100 trials, 1.7 s at
  1,000, against 1.5 ms for PCA at 5,000. It would want the Compute button these
  three did not.

So: its own figure, deferred, if it is added at all — and worth adding only for
the non-Euclidean distance, not for the projection.

The figure is renamed **Hyperparameter space projection**, since "Configuration cube" now
names one of its three views and "projection" is what all three are. The
settings key stays `configuration_cube` — it is stored.

## Two rows as well as two columns

`Figure.height`, beside `Figure.width`, with `SINGLE` and `DOUBLE` mirroring
`HALF` and `FULL`. Only this figure sets `DOUBLE`.

They are separate declarations because they answer different questions — how
much room a figure needs beside it, and how much under it — and this is the
figure that needs both. All three of its views are scatters over a space with no
privileged direction, and a scatter in a single row is a strip: the vertical axis
gets a fifth of the room the horizontal one does and reports a fifth of what it
has to say.

The row span is what makes it two rows of the grid; the plot's own `min-height`
is what makes those rows tall, since the grid's rows are sized by their contents
and a full-width item alone in its row has nothing to be sized against.

## No Compute button

Asked for, and measured against: PCA takes 0.2–1.5 ms and PLS 0.4–5.2 ms, from
20 trials × 8 features up to 5,000 × 60. The three figures that do have Compute
buttons cost 80 ms to 4.3 s. All three views ship with the page, and switching
between them is a switch between payloads that are already there.

## The encoding is the whole claim

`core/projection.py`, deliberately not under `core/optimizers/`: none of this
needs an optimizer, a surrogate or a refit, only the trials and the space they
came from, and it is worth being testable without any of them.

A projection of a badly encoded matrix is not a wrong number — it is a plausible
picture of nothing. So three things happen on the way in, each for its own
reason:

- a **log-scaled** hyperparameter goes on as `log10`. `C` sampled log-uniformly
  over 0.01–100 would otherwise put three quarters of a run within a hundredth
  of itself, and the projection would report the few large values as the only
  structure there is.
- a **categorical or boolean** one becomes one column per value. Parallel
  coordinates codes categories to integers because it needs one column per axis
  to draw; here that would put `linear` one unit from `poly` and two from `rbf`
  — an ordering and a distance that do not exist. Inventing distances is exactly
  what this module must not do, since distance is what it draws. `bool` is a
  subclass of `int` in Python, so it is excluded from the numeric test by name.
- every column is then **standardised**. Without it PCA reports whichever
  hyperparameter has the widest raw range: `n_estimators` over 10–500 drowns
  `min_samples_split` over 2–10 whatever either did to the score. A column that
  never varied is left at zero rather than divided by zero.

Two degenerate cases return a caption instead of a figure, because sklearn does
not: an unknown method used to fall through to PCA and draw a picture under the
wrong name, and a set of trials that all ran the same configuration returns a
plane of NaNs and a divide-by-zero warning rather than an error.

No seed. PCA is pinned to `svd_solver="full"` rather than sklearn's `auto`,
which switches to a randomized solver on a large enough matrix — and a
randomized solver takes a seed, which would then have to be threaded down from
the experiment. Exact is deterministic, and at these sizes `auto` would choose
it anyway.

## One payload shape, one piece of client code

The projection payload is shaped exactly like the cube's: coordinates in
`customdata`, one column per component, `x`/`y` empty, and the client assembles
two or three of them. So one payload covers both dimensionalities with no round
trip — the same trick the cube already played with hyperparameter columns — and
`applyCubeAxes` draws all three views off one branch:

```js
const projected = !!meta.components;
const names = projected ? meta.components : (meta.hp_names || []);
const chosen = projected ? names.slice(0, dimensions) : /* the three pickers */;
```

Everything after that — the colour bar, the 2D/3D switch, the selection ring,
the `newPlot` rebuild when the subplot type changes — is untouched.

The method is a `view`, because three named options is exactly what `views` is
for. Which hyperparameter sits on which axis is *not*: that is unbounded and
per-experiment, which is why it rides in `customdata` instead. The two live
side by side in the same figure without either becoming the other.

## Opaque points, and the selected one on top

Two nearly overlapping points were compositing into something darker than
either. On a scale where darker means better, that is a trial that did not
happen — the picture inventing a result out of two real ones. Nothing in the
figure set an alpha (Plotly's marker default is 1), so the blending was the
renderer's; `opacity=1` is now stated out loud because it is load-bearing here
rather than incidental, and every point carries a one-pixel white ring so an
overlap reads as two points rather than as one good one.

The selection is now an **overlay**: the selected trial drawn again on top, in
the selection colour, as a trace of its own. Two reasons it is not simply
recoloured in place — the fill here *is* the score, so painting one point orange
would lose the value it was carrying; and the ring this used to be does not
exist in a 3D scene, where Plotly draws no marker outline at all, leaving the
view that most needs a landmark without one.

`_selection_meta`'s fourth style, and the same shape as `line`, which parallel
coordinates already used for the same reason: the highlight is sliced out of the
trace that is already carrying the coordinates rather than shipped twice. It
carries `hoverinfo: "skip"`, like the incumbent markers, so it never stands
between the reader and the point underneath it.

## It opens on PLS

`Figure.default_view`, beside `views`. Separate from the order of `views`
because the order they read in and the one worth showing first are different
questions: the selector lists Axes, PCA, PLS — simplest first — and the figure
opens on the last of them.

PLS answers the question the page is actually about, which directions through
this space moved the metric. Axes needs three decisions before it says anything
at all, which is a poor thing to open on.

Declared twice, in the catalog and as `selected` in the markup, with a test
holding them together — a selector saying one thing beside a figure drawing
another is exactly the desync below, and this is where it would come from.

## Where it sits

Directly under Trial performance. The two are the same run read along its two
axes: what the search achieved over time, and where it went in space.

Parallel coordinates moved above partial dependence, so the last stretch of the
page reads at three magnifications in order — every trial at once, then one
hyperparameter at a time, then one trial at a time.

## The controls are the source of truth

A browser restores a `<select>`'s value across a reload, and a handler bound to
`change` never runs for a value the reader did not just set. So a picker could
come back saying one thing while `currentView` had never heard of it — the
method picker reading "PLS" beside a figure drawing axes, with the axis pickers
still next to it, agreeing only once it had been moved away and back.

Every view selector is now read once at startup, before anything is drawn:
the method (and the pickers it leaves standing), the importance figure's
rendering, the performance figure's two axes, and the game — which is one
selector driving six figures, so a restored one has to reach all of them and not
just the control it was restored into.

Restoring is the browser doing the reader a favour. The page's job is to agree
with it, which is why nothing is forced back to a default: a fresh load opens on
Axes in 2D because that is what the markup says, and a reload opens where you
left off because that is what you were looking at.

## Verify

```
python -m pytest -q -m "not slow"     # 987 passing
python manage.py check
python manage.py makemigrations --check --dry-run
```

Most of `tests/core/test_projection.py` is about the encoding, for the reason
above. The one test that pins why *both* methods are offered builds a result
where the configurations vary mostly along one hyperparameter while the score is
driven by another, and asserts PLS's first component correlates with the score
better than PCA's does.

In the browser, since none of the switching is exercised by the suite:

- cycle Axes → PCA → PLS and confirm the pickers swap with the method
- switch a projection between 2D and 3D, and confirm a click still selects the
  right trial in both
- confirm the colour bar and the selection ring survive every switch
- confirm the axes view is exactly as it was, including its None/None/None start
