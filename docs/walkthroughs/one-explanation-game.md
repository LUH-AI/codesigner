# One explanation game, answered by every figure

## What it was

The game selector — tunability, sensitivity, mistunability, and a fourth entry
for the selected trial's ablation — sat inside the hyperparameter importance
figure and moved only that figure. The five interaction figures read tunability
whatever it said.

So the page could show the importance figure under sensitivity beside the
heatmap under tunability: two different questions, side by side, looking like
one answer. And the selector was in the wrong place to be believed anyway — a
choice about the whole explanation, made inside one panel.

## What it is

A sidebar section, **Explanation Game**, between Experiment Evaluation and
Selected Configuration. One `<select>`, three games, and a line under it saying
what the chosen one asks — the three sound alike until they are written out:

| Game | Asks |
| --- | --- |
| Tunability | How much is there to gain by tuning this hyperparameter? |
| Sensitivity | How much does performance move as this hyperparameter moves? |
| Mistunability | How much is there to lose by getting this hyperparameter wrong? |

Changing it moves the importance figure and all five interaction figures at
once. The importance figure keeps its own rendering picker (pie / bar / table),
because how to draw a game is a different question from which game to draw.

## What it cost: nothing, twice over

**No compute.** `compute_hp_games` has always run all three games and returned
each one's importance, its order-2 interaction grid and its full Möbius
decomposition. Only tunability's grid and Möbius were kept — the other two were
computed and dropped for want of a field to put them in. The old docstring said
so in as many words. Four new fields on `OptimizationResult`
(`hyperparameter_sensitivity_interactions` and friends) is the whole change.

**No round trip.** The games are the interaction figures' `views`. A game is a
different set of numbers to draw rather than a different way to draw them, but
"one precomputed payload per option, switched in the browser" is exactly the
plumbing that needs — so they use the machinery the importance figure's
renderings already used, driven from one selector instead of six.

Field names carry their game except tunability's, which have none:
`hyperparameter_interactions` and `hyperparameter_moebius` were named back when
only tunability's had anywhere to go, and they are in every `.ihpo` ever
written. The four new ones deserialize to empty, which reads as "that game has
no interactions" — the same empty state a failed game already produces.

## The local explanation left the selector

It was the fourth option on the game selector. It is now its own figure, sitting
between partial dependence and local effects.

It never belonged with the other three: they explain the search, it explains one
trial. It has no interactions to give the five figures that read them, so a
universal selector would have had to blank five figures whenever it was chosen.
And it is the only one that cannot be precomputed — its value depends on which
trial is selected, not just the metric — so it was already the odd entry, listed
as a view whose payload was deliberately never built.

Its setting keeps the name `autocompute_local_ablation`. That is the
computation's name, not the figure's, it is stored in settings, and renaming it
would be a data migration for nothing.

## Everything follows the one seed now

The last thing that did not: each game's searcher draws 10,000 configurations to
play against, and that draw was seeded at HyperSHAP's own default of 0. Two
experiments with different seeds explained themselves from the same sample —
deterministic, but a constant's determinism rather than the experiment's.

`_shared_exact_computer` now takes the seed, and so does the facade fallback in
`_game_values`, so the two paths stay comparable. `_HPO_SIMULATION_SEED` is gone;
`_HPO_SIMULATION_SAMPLES` stays, because that one really does have to match
HyperSHAP's default or the game being computed is not the game the facade
computes.

The test asserts at the searcher rather than on the numbers. On a small space
10,000 draws find the same maximum whatever the seed, so the values coincide
here and would not on a wider one — what is being fixed is *which* determinism
it is, so what matters is that the experiment's seed is the one that arrives.

## Better trials on top in parallel coordinates

Traces were added in trial order, so with hundreds of lines crossing, the ones
drawn over the others were whichever ran latest. They are added worst-scoring
first now, so the best draw on top. Trace order is z-order and nothing can
change it after the fact, which is why it is a decision made at build time.

Trace order is therefore no longer trial order, and `meta.selection` says which
trial each trace stands for — which is exactly what that map is for, and was
already carrying the same distinction for the sampled beeswarm.

## Tooltips in the sidebar

Two things were wrong at once. `position: sticky` makes its own stacking
context, so with no `z-index` the whole sidebar painted in document order —
before `main.content`, which drew over its bubbles. And the sidebar scrolls its
own contents, which clips anything positioned outside its box, so a bubble could
not have reached over the page even with the paint order fixed.

The z-index half is an explicit one on both navs and on `main.content`.

The clipping half took three goes, and the first two were the same mistake: they
moved the bubble. Opening it leftwards traded the main column for the icon rail;
stretching it across the field kept it inside horizontally and made the sidebar
grow a horizontal scrollbar instead.

The problem was never where the bubble was. An absolutely positioned descendant
of a scroll container is part of what that container scrolls — it does not escape
the box, it widens it. So the bubble is `position: fixed` in the sidebar, which
takes it out of the scroll container altogether (`position: sticky` does not
capture fixed descendants; only transforms and their kin would). Offsets left
`auto`, so it opens at its static position beside the circled i, and the
sidebar's own `z-index` is what puts it over the page.

## Titles and pickers on one line

A figure's pickers say what its title is currently showing, so they read as part
of the heading rather than as a caption under it. Every figure now has a
`.card-head` row — title left, pickers right — and they fall onto two lines only
when the card is too narrow to hold both.

## Verify

```
python -m pytest -q -m "not slow"     # 973 passing
python manage.py check
python manage.py makemigrations --check --dry-run
```

`tests/fixtures/analytics.ihpo` was regenerated, since the fixture exists to
carry every analytics field and four of them are new. A test loads it with those
four stripped out, which is what every earlier file looks like.

In the browser:

- change the game and confirm the importance figure and all five interaction
  figures move together, and that the line under the selector changes with them
- confirm sensitivity's heatmap is not tunability's under another name
- confirm the importance figure's rendering picker still works, and independently
- select a trial and confirm the local explanation follows it, and that with its
  autocompute off it shows a Compute button instead
- open a tooltip in the sidebar and confirm it is neither painted over nor cut off
- confirm the best-scoring parallel-coordinates lines are drawn over the rest
