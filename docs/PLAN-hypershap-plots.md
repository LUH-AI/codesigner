# Everything HyperSHAP can draw, drawn in Plotly

> Executed in one pass; see
> [walkthroughs/hypershap-plots.md](walkthroughs/hypershap-plots.md).

## Context

`docs/PLAN-analytics.md` closed out with codesigner surfacing HyperSHAP's
numbers four ways: a pie, a bar, a table, and an order-2 heatmap. HyperSHAP
itself ships five plots and shapiq beneath it nine, including the SI / Möbius
graph — the only one that shows interactions above order 2.

All of it is matplotlib and networkx. Every figure in this app is a live Plotly
chart, and Phase 2 already decided that reproducing a plot natively beats
embedding a static image of it. So this is a port, not a wrapper.

## What was and wasn't worth porting

| shapiq plot | Ported as | Why |
|---|---|---|
| `si_graph` | Interaction graph | The flagship: the only view of order 3+ |
| `upset` | Coalitions | Readable where the graph stops being |
| `stacked_bar` | By order | Answers "alone, or only in combination?" |
| `waterfall` | Local (selected trial) | Replaced the diverging bar |
| `beeswarm` | Local effects | The only one that shows spread |
| `network` | — | Order-2 only; the graph subsumes it |
| `bar` | — | Already the importance figure's bar view |
| `force` | — | The waterfall is the same reading, better |
| `sentence` | — | For text explanations; no analogue here |

## The four findings that shaped it

Measured against the installed hypershap 0.0.6 / shapiq 1.4.1, not assumed.

1. **Raising the interaction order is free.** `ExactComputer` evaluates all
   2^n_hp coalitions whatever order is asked for; `order` only selects what
   comes back. 255 ms at order 1, 2 and 3 alike at 4 hyperparameters.

2. **But FSII's low-order terms move when you raise it.** FSII is the
   *faithful* least-squares k-order approximation, fitted jointly, so the
   order-1 terms of an order-3 fit are not those of an order-2 fit. Measured
   at 6 hyperparameters: order-1 shifts 5% of the largest term, order-2 shifts
   **24%**. Raising the order in place would have silently rewritten
   `hyperparameter_importance` for every new run.

3. **The Möbius transform doesn't move.** One value per coalition, no
   truncation — measured identical at order 2 and order n. So it can be stored
   once and read at any order, which is what a graph showing 2-way and 5-way
   interactions in one picture needs. It is also the transform the graph is
   named after.

4. **One `ExactComputer` serves both indices for one set of coalitions.**
   FSII costs 1060 ms; the full Möbius transform *after it* costs **1 ms**.
   HyperSHAP's facade builds a fresh computer per call, so going through it
   twice would have paid 1110 ms twice.

Hence the design: keep FSII order-2 exactly as it was, and take the Möbius
decomposition alongside it for ~1 ms. **No stored number moved.** Measured
end to end: the eager stage is 3,418 ms against a 3,500 ms baseline, and an
A/B on identical inputs at 6 hyperparameters came in at 0.96×.

## What it cost to build it that way

Getting both indices from one computer means constructing the game the way
`HyperSHAP.<game>()` constructs it rather than calling it — about twenty lines
against a 0.0.6 dependency. Two things guard that, and both are load-bearing:

- a test asserting our FSII output is **bit-identical** to
  `HyperSHAP.<game>()`'s for all three games, which fails loudly if their
  internals move;
- a `try/except` falling back to the facade, which costs only the
  Möbius-backed views — the importance numbers cannot come down with them.

See `BaseOptimizer._shared_exact_computer`, whose docstring carries this
reasoning in full.

## What is stored, and what is not

`hyperparameter_moebius` is a new field beside the existing ones — metric →
`[{"members": [...], "value": float}, ...]`, 2^n_hp − 1 terms per metric (15 at
4 hyperparameters, 63 at 6, and the coalition budget caps it around there).
`hyperparameter_interactions` keeps the order-2 FSII grid unchanged, so the
heatmap, the top-pairs bar and every stored `.ihpo` are untouched.

The beeswarm stores nothing: one ablation game per trial, capped by
`local_effects_max_trials` and off by default, like every deferred computation
since the compute-policy epic.

## Explicitly not doing

- **A static-image fallback for the graph.** Not needed: HyperSHAP pins its own
  SI graph to a circular layout, so there is no force-directed algorithm to
  reproduce and Plotly can draw it natively.
- **shapiq's Bezier hyper-edge ribbons.** Spokes to a centroid carry the same
  information — which hyperparameters, how strongly, which sign. The ribbon is
  an aesthetic.
- **`sentence`, `force`, `network`, `bar`.** See the table above.
- **Raising the stored FSII order.** Finding 2.
