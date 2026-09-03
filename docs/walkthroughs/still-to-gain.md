# "Still to gain": tunability as achievable *and* banked

## The problem, stated exactly

Tunability is the Shapley value of

> ν(S) = max { f(x) : x agrees with the config-space **default** outside S }

Two parameters of that definition are made silently — the baseline, and the
range — and together they make it an *ex ante* quantity: **before you searched,
how much was being allowed to tune λ worth?**

The reader at trial 30 is asking an *ex post* question: from where I now stand,
is more budget on λ worth spending? The two coincide at trial 0 and diverge
monotonically after. Only the first is on screen, and it gets read as the second.

Measured, on wine + random forest + SMAC, this is not a subtlety. Tunability says
`min_samples_split` 98%. But SMAC had already solved it: Sobol put 7 of its first
10 trials in the dead region, and from trial 20 onward the optimizer went back
there once in forty. The figure spent 98% of itself on a settled question while
`max_depth`, where the remaining gains were, read 0.9%.

## Why the obvious fix is wrong

Move the baseline to the incumbent, and ask what is reachable from there. Tried,
measured, discarded: `min_samples_split` went **up**, 71.6% → 86.9%.

ν is a max over a *fitted surrogate*, and the maximum of a noisy function is
upward-biased — biased most along the axis with the largest surrogate variance,
which is precisely the axis that has been tuned hardest. A "remaining tunability"
built that way measures surrogate optimism, not headroom.

That rules out the family, not just the instance: **no max-based statistic can
answer "what is left", because max-of-noise rises with uncertainty instead of
falling to zero with headroom.**

## What works, using two numbers already computed

Tunability decomposes (best achievable − default). The **ablation** — the local
explanation — decomposes (this trial − default). Same baseline, same machinery,
same units. So the per-hyperparameter ratio is well posed, and the difference is
what is still on the table.

| | achievable | banked |
| --- | --- | --- |
| `min_samples_split` | 97.6% | **0.89×** |
| `max_features` | 1.1% | −0.14× |
| `max_depth` | 0.9% | −3.80× |
| `n_estimators` | 0.3% | −27.86× |

"98% achievable, 89% of it banked" is a complete sentence where "98%" alone was a
misleading one. And the negatives are a finding in their own right: the incumbent
holds values *worse* than the default on three axes — drift picked up while
chasing the one that mattered — which no figure showed before.

## The presentation

One checkbox on the importance figure: **Still to gain**. It is there because
that is where the reader is misled, and it is one control because the risk was
overloading the figure that already carries the most.

- **Clear** — today's figure, untouched. A test asserts every payload and every
  plot is byte-identical to what it was.
- **Ticked** — the same renderings, of `max(0, achievable − banked)`,
  renormalised, ordered by what is left rather than by size. `min_samples_split`
  collapses to a sliver; `max_depth` grows. Flipping it back and forth *is* the
  insight, delivered as an interaction rather than as prose.

**Both renderings divide, box clear** — `hyperparameter_progress_plot`, one hue
at two intensities like `hyperparameter_orders_plot`'s three, because these are
parts of one quantity rather than two quantities. Darker is what is still to
gain, because that is the part worth acting on.

- **Bar**: banked stacked under still-to-gain, so a bar that is nearly all light
  is a settled question and one that is nearly all dark is where the budget
  should go. Banked at the base, so it reads as a bar filling up.
- **Pie**: one flat pie with two adjacent slices per hyperparameter, so a wedge
  is literally part light and part dark. The name appears once per
  hyperparameter, on whichever of its two parts has the room.

  Two earlier drafts got this wrong in opposite directions. The first argued
  against splitting the pie at all, on the grounds that a second encoding per
  slice means comparing arcs at different radii — right about a ring whose
  *radius* encodes the fraction, wrong as a reason not to split. The second used
  a sunburst, which puts the split on an outer ring: legible, but with several
  hyperparameters there is no single root node to fill the centre, so it leaves
  a hole there and puts the labels on the innermost ring, where there is least
  room. A flat pie has neither problem and asks one less idea of the reader.

**Table**: achievable, banked and still-to-gain in the metric's own units. This
is also where magnitude finally appears at all — every rendering until now showed
a share of 100%, and a share cannot say whether the whole is worth eight accuracy
points or eight thousandths of one.

So the box now switches between two honest views rather than between a
misleading one and an honest one: divided by achievable (clear), or sized by what
is left (ticked). The split needs the ablation, so it appears once that has been
asked for — by ticking the box once, or by the local explanation, or by turning
`autocompute_local_ablation` on. Until then the figure is exactly what it was.

**Tunability only, and the control goes away with it.** Both halves are
tunability's: the achievable side is its own game, and the banked side is the
ablation, which measures against the same baseline the max game measures from.
There is no reading of the subtraction under the other two — mistunability's
achievable value is downside rather than gain, and nothing "achieves"
sensitivity's variance.

So under the other games the checkbox is hidden, the same way the cube's axis
pickers go when a projection is showing, and it is unticked on the way out. Left
in place it would draw tunability's headroom under another game's name, which is
worse than not offering it at all. The table's columns go with it for the same
reason: beside another game's shares they would read as that game's.

**Which trial** is the page's selection, which defaults to the metric's best. So
it opens on the incumbent reading with no new concept, and follows the selection
bus like everything else.

## Wedge labels are one size or absent

It takes two settings, and the first attempt used only the second — which is why
nothing changed. `textfont` fixes the size; Plotly otherwise scales each label to
its own slice, so a thin wedge gets tiny text and a fat one gets large text,
which reads as emphasis that is not there and is unreadable at the small end.
`uniformtext` with `mode: "hide"` then decides what happens to a label that no
longer fits at that size: dropped rather than shrunk back, with the hover still
carrying it. `minsize` matches the font size, since below it Plotly would be
scaling again.

The pie also states `hole=0` rather than leaving the attribute out.
`Plotly.react` diffs against what is already drawn, and an absent attribute is a
weaker instruction than one set to its default.

## The box does not come back ticked

A browser restores a checkbox across a reload the way it restores a select. But
this one is a *request* — it asks for a computation that has not been made — and
a request is not a preference to remember. Restored, it came back ticked with
nothing behind it, so the figure showed the plain view while the control claimed
otherwise: the same desync class as the method picker, and reported the same way.

It is cleared on load, alongside the other controls the page reads at startup.

## What these numbers are not

Checked after the fact, by evaluating both games' own endpoints rather than
trusting the decompositions. Two limits, both real, both reasons to read the
ordering rather than the digits.

**Achievable is a lower bound.** The tunability game's maximum comes from 10,000
*random* draws of the configuration space. A real optimizer beats random search
— that is what it is for — so a well-tuned hyperparameter can bank more than the
estimate says was achievable. Measured on a 40-trial wine run:
`min_samples_split` achievable +0.070, banked +0.081.

That is not an error and it is not noise. It means the search found something
the estimate could not see, which is the *strongest* evidence an axis is
finished. So it is marked — the table reads "—" with the reason on hover —
rather than floored to zero and left looking like a rounding artefact. An earlier
draft of this called it "overshooting", which had the direction of the error
backwards.

**Order-1 is not the whole game.** Both halves are FSII order-1 terms of an
order-2 fit, and FSII has no efficiency property at order 1. Measured: order-1
accounts for 76% of the tunability game's own `v(all) − v(none)` and 70% of the
ablation's, the rest living in order-2 interactions that neither number carries.
Both sides are deflated by the same mechanism, so the *comparison* survives it
better than either figure alone.

**What does hold**, and was checked rather than assumed: both games return
exactly 0 for the empty coalition, so both are gains over the same baseline
configuration in the same units. The subtraction has a common origin, which is
the part that would have been fatal.

## Shape

**The missing scalar.** `_compute_hp_game` computed the sum of the raw order-1
magnitudes and threw it away when it normalised. It is returned now, as a fifth
element, and stored per metric as
`OptimizationResult.hyperparameter_tunability_total` — one float, since
`share × total` recovers the rest. A file written before it has no scale, and the
view returns nothing rather than a ratio against the wrong denominator.

**The other half was also discarded.** `_local_ablation_data` computed the raw
signed effects and returned only the figure; it returns both now.

**The headroom renderings cannot be precomputed** — they depend on which trial is
selected — so they ride back with the ablation that produced them, both at once,
and switching rendering afterwards costs no second request.

**The checkbox is a request.** Ticking it fetches if the ablation has not been
asked for yet — the same fetch the Compute prompt uses, reached through a control
the reader has already touched. With the box clear the page costs nothing new.

**Two guards**, both the lesson of the existing all-zero guard:

- remaining total within 2% of the achievable total → say "this trial has
  captured essentially all of the gain the search space had to offer" rather
  than draw a pie of rounding
- banked outside 0–1 of achievable → clamped in the share, kept signed in the
  table, so an axis the incumbent made worse is visible rather than floored away

## Verify

```
python -m pytest -q -m "not slow"     # 1015 passing
python manage.py check
python manage.py makemigrations --check --dry-run
```

`tests/fixtures/analytics.ihpo` was regenerated: it exists to carry every
analytics field, and the scale is a new one. A test loads a result without it and
asserts the view declines rather than guesses.

In the browser, since none of the switching is exercised by the suite:

- tick the box once, then untick it, and confirm the pie becomes a sunburst and
  the bar becomes two stacked segments
- tick and untick across both renderings; confirm the ticked view reorders
- switch the game with the box ticked, and confirm the control disappears and
  the figure returns to that game's own shares
- select a different trial with the box ticked and confirm what is left follows
- confirm the fetch happens once per trial, and switching rendering after it is
  instant
