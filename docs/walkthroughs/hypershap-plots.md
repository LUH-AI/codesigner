# HyperSHAP's plots, in Plotly

> **Status: done.** Roadmap: [../PLAN-hypershap-plots.md](../PLAN-hypershap-plots.md).
>
> One write-up rather than six. The roadmap was drafted as six steps and they
> shipped as one unit, so six documents would have been the same story told in
> six pieces. The steps are the headings below.

## The concept

HyperSHAP explains a search as a cooperative game: each hyperparameter is a
player, each *coalition* of them has a value, and the question is how to
attribute that value back to the players. codesigner already showed two answers
— per-hyperparameter shares (the pie/bar/table) and pairwise interactions (the
heatmap). Both stop at order 2.

They stop there because of what they are computed from. FSII, the index
HyperSHAP returns by default, is the best *k*-order approximation of the game,
fitted as a whole. Ask for three orders instead of two and every term changes,
because it is a different fit — not more of the same one. Measured on a
6-hyperparameter space: the order-1 values move by 5% of the largest term, the
order-2 values by 24%.

The Möbius transform is the other way of reading the same game: one value per
coalition, defined without reference to any truncation. A coalition's Möbius
value is what that exact set contributes beyond everything its subsets already
explain — which is precisely what a picture showing pairs and triples together
needs, and is why the interaction graph is called a Möbius graph.

Both come out of one `shapiq.ExactComputer`. The coalitions are what cost
money — 2^n_hp evaluations, each a batched 10,000-row surrogate predict — and
the computer caches them, so the second index is essentially free.

## What shipped

- **One computer, two indices.** `_shared_exact_computer` builds the game the
  way `HyperSHAP.<game>()` builds it and reads FSII order-2 and the full Möbius
  transform off one `ExactComputer`. FSII 1060 ms, Möbius 1 ms after it.
- **`hyperparameter_moebius`**, a new stored field. The order-2 FSII grid is
  untouched, so no stored number and no existing view moved.
- **Interaction graph** — hyperparameters on a circle, pairs as edges, larger
  coalitions as a hub with a spoke to each member. Width is magnitude, colour is
  sign. Top 15 drawn.
- **Coalitions (UpSet)** — ranked signed bars over a membership matrix. Reads
  where the graph stops: past six hyperparameters a graph is wool, a list is
  still a list.
- **By order** — each hyperparameter's influence split into alone / in pairs /
  in larger groups. Each Möbius term divided equally among its members, which is
  the standard reading of a Harsanyi dividend.
- **Waterfall** replacing the local diverging bar: the same ablation numbers,
  walked from the config-space default to the selected trial, so the running
  total is visible instead of left to the reader.
- **Local effects (beeswarm)** — one ablation per sampled trial, showing the
  *spread* of each hyperparameter's effect rather than one trial's or an
  average. Deferred, capped, off by default.
- **`compute_hp_ablation` can share an explainer**, which is what makes the
  beeswarm affordable: 79 ms per call became 35 ms once plus 43 ms each.

## What is now possible

Three questions the page could not answer before:

- *Do these hyperparameters interact as a group, or only in pairs?* Only the
  graph and UpSet can say — everything else was order-2 by construction.
- *Does this hyperparameter matter on its own?* The by-order view separates a
  hyperparameter worth tuning alone from one that only pays off alongside
  another. Those call for different decisions and looked identical before.
- *Does this hyperparameter always help?* The beeswarm distinguishes a
  hyperparameter that reliably helps a little from one that helps enormously in
  half the space and hurts in the other. Every previous view averaged that away.

## What it cost

Nothing, for the first three. The eager stage measures **3,418 ms** at 4
hyperparameters × 4 metrics against a ~3,500 ms baseline, and an A/B on
identical inputs at 6 hyperparameters came in at **0.96×**. The Möbius terms
were being computed and discarded, the same way the order-2 terms were before
Phase 2 of the analytics roadmap.

The beeswarm is the exception and the reason it is deferred: 35 ms plus 43 ms
per trial, so 4.3 s at its 100-trial default cap. It waits to be asked.

## The risk, and what holds it

Reading two indices off one computer means building the game rather than asking
the facade for it — twenty lines against hypershap 0.0.6, whose internals may
move. Two guards, both deliberate:

1. `test_our_exact_computer_reproduces_hypershaps_own_numbers` asserts our FSII
   is bit-identical to `HyperSHAP.<game>()`'s for all three games. If they
   change a default, reorder the searcher or swap an aggregation, this fails
   rather than the page quietly showing numbers from a different game.
2. `_game_values` falls back to the facade on any exception. The Möbius-backed
   views go empty and show the caption they already show when a game fails; the
   importance numbers are unaffected.

`test_the_moebius_transform_is_truncation_independent` pins *both* halves of
finding 2 — that Möbius doesn't move and FSII does. The contrast is the whole
reason for the design, and a reader who saw only the Möbius half would be
within their rights to "simplify" it back into a bug.

## Verification

- `python -m pytest -q -m "not slow"` — 912 passing, `manage.py check` clean,
  `makemigrations --check` reports nothing (both new fields are JSON).
- Eager cost re-measured after the change, not assumed: 3,418 ms, and 0.96× on
  the A/B.
- The rendered page script re-checked for delimiter balance after the client-side
  edits, since no JS engine is available here to parse it properly.
- Browser: cycle the five interaction views on a 4-hyperparameter and a
  6-hyperparameter run; confirm the graph's spokes reach the right nodes;
  confirm the waterfall's total matches the selected trial's panel; switch
  `autocompute_local_effects` on and off and confirm a reload computes nothing
  with it off.
