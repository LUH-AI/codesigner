# Step 7 — Click a trial on the chart

**Status:** implemented, awaiting your sign-off.
**You can now:** click any point on the performance chart to see that trial's score and its full hyperparameter configuration, with the clicked point highlighted. Before any click, each metric's panel defaults to its best trial (matching the existing "Best configuration" panel, with no delta shown).

---

## 1. What was ported

Source: `app/analytics/selected_config.py` (the panel content — score, delta vs.
the metric's best, config table) and the `on_select` handler at the bottom of
`app/analytics/performance.py:59-70` (which curve/point a click resolves to).

**The Streamlit mechanism:** `st.plotly_chart(..., on_select="rerun",
selection_mode="points")` gives Streamlit an event object on every rerun;
the code filters for `curve_number == 0` (the trial-score markers — curve 1 is
the incumbent line, which isn't clickable-meaningful) and reads
`point_index`, which is the 0-based index into `result.trials`. If it differs
from the currently selected index, it's stored in session state and the app
reruns to show the new selection.

**The Django/JS equivalent:** there's no server round-trip per click by
default in a plain web page, so:
1. A `plotly_click` listener is attached to the chart div (once, since
   `Plotly.react` — used for the metric switcher — preserves event listeners
   on the same DOM node; only `Plotly.purge` would require re-attaching).
2. The listener does the same `curveNumber === 0` filter and reads
   `pointIndex` (JS naming, same meaning as Python's `point_index`).
3. It immediately restyles the chart's markers client-side (instant visual
   feedback — no need to wait on the network for the highlight to move), then
   fetches `GET /experiments/<pk>/trial-panel/?metric=<m>&idx=<i>`, a small
   view that renders exactly the same partial template used for the page's
   initial (default) panel, and swaps it into the DOM.

Both the initial page render and the AJAX endpoint compute the panel's data
through one shared function, `_selected_panel_data(result, metric, idx)` in
`ui/views.py` — so "what a selected trial shows" is defined once.

## 2. Feature checklist

From `app/analytics/selected_config.py` and the `on_select` block. ✅ done ·
🔄 changed.

- ✅ Subheader "Selected configuration"
- ✅ Caption "Trial N — click any point on the graph to select"
- ✅ Score for the current metric, to 4 decimal places
- ✅ Delta vs. the metric's best score, shown only when the selected trial
  isn't the best (Streamlit compared `delta != 0.0`; this port compares trial
  indices instead — equivalent in effect, immune to float-equality edge cases)
- ✅ Delta sign styled (up/down), matching the `metric-delta` component from
  Step 3's CSS
- ✅ Hyperparameter/value table, same shape as Best configuration
- ✅ Click filters to curve 0 only (ignores clicks on the incumbent line)
- ✅ Clicking restyles the marker (size 13, red) exactly like the pre-existing
  best-trial highlight from Step 4
- 🔄 **Selection resets on metric switch** rather than surviving via a query
  param (see the deviation note below)

## 3. Deviation from the plan

The plan called for "selection survives metric switches via query param."
Since Step 4 already made metric-switching fully client-side (no page reload,
no query param — a deliberate deviation at the time), carrying a per-metric
selected index through query params would have meant re-introducing page
reloads just for this one feature, working against Step 4's design.

Instead: switching metrics resets that metric's panel to its default (best
trial), via a client-side snapshot taken once at page load of each metric's
initial panel HTML. This keeps two things always consistent, which a
query-param approach would not automatically guarantee: the chart's highlight
(which already resets to the server-embedded best-trial highlight on every
`Plotly.react` call) and the panel's text. A click is remembered only while
you stay on that metric; switching away and back re-shows the best trial. If
you'd rather clicks persist per metric across switches, that's a small,
identified follow-up (cache each metric's last-fetched panel HTML client-side
instead of the page-load default) — flagging rather than assuming.

## 4. Verify it yourself

```bash
cd ~/Downloads/python/projects/codesigner
.venv/bin/python manage.py runserver
.venv/bin/python -m pytest tests/ui/results   # 13 tests
```

Open any experiment with results, click a few points on the performance
chart — the "Selected configuration" panel updates and the clicked marker
turns red/large. Click the incumbent *line* — nothing happens (by design).
Switch the metric dropdown — both the chart's highlight and the panel reset
to that metric's best trial.

I additionally verified the full path live (no headless browser available in
this environment, so via direct HTTP calls simulating what the click handler's
`fetch()` does): imported the `test2.ihpo` fixture, confirmed the detail page
wires up `plotly_click` and embeds the panel, then called the `trial-panel`
endpoint directly — trial 1 (index 0) returned a `-0.0875` delta ("down"),
trial 9 (index 8, the actual best) returned no delta, and an unknown metric /
out-of-range index both returned 400 rather than a server error.
