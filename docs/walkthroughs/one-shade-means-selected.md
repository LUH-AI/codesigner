# One shade means selected

**Status:** implemented, awaiting your sign-off.
**You can now:** see the same colour for "this is the one you picked" wherever
it appears — the figures, and the experiment list in the sidebar.

---

## It was the alpha

`SELECTION_COLOR` was `#EF553B`, a fairly red orange. But `performance_over_time_plot`
fades its points to 0.7 opacity — per point rather than trace-wide, so a failed
trial's cross can sit among them at full strength — and the selected point was
fading along with everything else. Over Plotly's own default `plot_bgcolor` of
`#E5ECF6`, that composited to:

```
#EF553B at 0.7 over #E5ECF6  ->  #EC8273
```

a soft coral. Every other figure painted the same constant at **full** opacity
and came out visibly redder. So "selected" was two colours depending on which
figure you were looking at, and the one people see most — trial performance is
where the click-to-select gesture started — was the softer one.

Resolved towards that one. `SELECTION_COLOR` is now `#EC8273`, the shade as it
appears, painted at full opacity everywhere; and `performance_over_time_plot`
exempts the selected point from its fade, so an already-soft colour is not faded
a second time. The point on screen there is unchanged; every other figure now
matches it.

`--selected` in `app.css` is the same value, for the parts of the page that are
not plots, and a test holds the two together.

## The sidebar

The experiment list marked the current experiment with `--primary` — `#ff4b4b`,
the brand red. That is the colour of *actions* (the Run button, a destructive
confirm), and using it for "this is the one you are looking at" made two
different jobs look alike.

It uses the selection shade now, on the left accent and the background tint.
**The label text does not**: the shade is a soft coral chosen to sit under a
scatter plot, which makes it too light to read a word in — as a foreground it
was failing contrast at `#ff4b4b` already and would fail harder at `#EC8273`. So
the accent and the tint carry the meaning and the weight carries the emphasis,
with the label in the ordinary text colour. Say the word if you would rather
have the literal reading and take the contrast hit.

The trials table's row highlight is untouched, as you said — it is a pale tint
of the old value, and at 13% alpha the two are indistinguishable.

## Verification

`tests/ui/runs/test_live_polling.py` holds three of these: that the constant is
exactly what the old one composited to (computed, not copied), that the CSS and
`plots.py` agree, and that the sidebar's active tab no longer reaches for
`--primary`. `tests/ui/results/test_plots.py`'s selection test was updated —
it pinned the selected point being translucent, which is the thing that was
wrong.
