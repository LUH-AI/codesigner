# Step 4 — Figures of the results

**Status:** implemented, awaiting your sign-off.
**You can now:** after a run, see a performance-over-trials scatter with the running-best line, a hyperparameter-importance donut (with any warning text), and the best configuration — and switch which metric the figures show, instantly.

---

## 1. Concepts introduced

### Pure figure builders + a thin view
[ui/figures/plots.py](../../ui/figures/plots.py) holds `performance_figure` and
`importance_figure` — pure functions from an `OptimizationResult` + a metric to
a Plotly `Figure`, with no Django imports. The view
([ui/views.py](../../ui/views.py) `_results_context`) calls them and does
nothing figure-specific itself. Same separation as the run service: the
figure logic is unit-tested on its own.

### Getting Python figures into the browser
Plotly renders client-side. The bridge: `fig.to_json()` turns a figure into a
plain data structure, the view puts those in a dict, and the template ships it
with Django's `{% json_script %}` filter (which safely embeds JSON in a
`<script>` tag). Vendored [plotly.min.js](../../ui/static/ui/plotly.min.js)
(served as a static file) reads that data and draws the figures — no CDN, no
build step.

### Static files
First use of `{% load static %}` + `{% static %}`. `plotly.min.js` lives in
`ui/static/ui/` and is served at `/static/ui/plotly.min.js` in development.

## 2. Feature checklist

From `app/analytics/{performance,hp_importance,best_config,selected_config}.py`.
✅ implemented · 🔄 changed · ⏳ deferred (step).

### `performance.py`
- ✅ Scatter of each trial's score for the shown metric
- ✅ Running-best ("incumbent") line overlaid
- ✅ Selected point enlarged + recolored (defaults to the metric's best trial)
- ✅ Y-axis titled with the metric
- ⏳ Click a point to select it (`on_select`) — **Step 7**
- 🔄 Default Plotly hover instead of the custom per-hyperparameter hover (cosmetic; can restore later)

### `hp_importance.py`
- ✅ Donut (hole 0.35, label+percent) of importance for the shown metric
- ✅ Warning text shown when importance estimation fell back
- ✅ "No importance" message when a metric has none

### `best_config.py`
- ✅ Best trial for the shown metric: trial number, score, config table

### `selected_config.py`
- ⏳ **Deferred to Step 7.** Without click-to-select it would only ever repeat the best-config panel; it becomes meaningful once you can pick a trial.

### Metric switching
- 🔄 **Client-side**, not the planned `?metric=` GET param. Every metric's
  figures + best-config panel ship embedded in the results page; the dropdown
  toggles them with a small script. Rationale: a `?metric=` GET needs the result
  to survive to a second request, which requires persistence we haven't built.
  The GET approach returns when experiments persist.

## 3. What is now possible

- Run an experiment (Step 3) and the results page now shows real figures: the
  performance scatter with the incumbent line, the importance donut, and the
  best configuration — for whichever of the four metrics you select, switched
  instantly without reloading.
- This is the interpretability payoff of the tool, now manually inspectable in
  the browser and comparable side-by-side with the Streamlit app.

## 4. Verify it yourself

```bash
cd ~/Downloads/python/projects/codesigner
.venv/bin/python manage.py runserver     # New → run → results page has figures
.venv/bin/python -m pytest tests/ui/results
```

Run Grid Search on iris (fast), then change the metric dropdown and watch the
scatter, donut, and best-config panel update. Compare the incumbent line and
importance donut against the Streamlit app for the same dataset/seed/optimizer.
A metric like `recall(macro)` on a tiny run is a good way to see the importance
warning path.

## 5. Notes

- Figure builders are tested directly (`tests/ui/results/test_plots.py`) against a
  synthetic result with hand-picked numbers; the view test confirms the page
  embeds every metric's figures. The Streamlit originals are `st.*`-coupled and
  can't run under pytest, so these tests are codesigner-native (the incumbent
  logic is ported verbatim from `performance.py`).
