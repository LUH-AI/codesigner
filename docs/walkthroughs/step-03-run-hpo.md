# Step 3 — Run HPO from the browser

**Status:** implemented, awaiting your sign-off.
**You can now:** set up an experiment (model, optimizer, dataset, metric, seed, number of trials), run the optimization, and see the results — entirely in the browser. This is the first end-to-end manual test of the Step 2 engine through a GUI.

---

## 1. Concepts introduced

### Django `Form`s
[ui/forms.py](../../ui/forms.py) defines `NewExperimentForm` — a declarative
list of fields (choices, file, integers). Django renders it to HTML, and on
submit validates it: required fields, integer bounds (`n_trials` 1–1000), and a
custom `clean()` that enforces "a demo dataset **or** an upload." Invalid input
comes back as `form.errors` and re-renders with messages; the view never sees
bad data. The registry drives the dropdown choices, populated in `__init__`.

### POST-and-render, CSRF, file uploads
[ui/views.py](../../ui/views.py) `new_experiment` is the request/response
shape you'll see everywhere: GET renders an empty form; POST validates and, if
good, does the work and renders a result. The form template carries
`{% csrf_token %}` (Django rejects POSTs without it) and
`enctype="multipart/form-data"` so the file upload arrives in `request.FILES`.

### Keeping the view thin
The actual work lives in [ui/services/run.py](../../ui/services/run.py)
(`resolve_seed`, `run_experiment`) — plain functions with no Django imports,
callable and testable on their own. The view only translates HTTP ↔ those
functions. That separation is why the run logic could be tested against the
Streamlit repo before this view existed.

### Deliberately synchronous
`run_experiment` runs the optimizer to completion inside the request — the page
waits. Fine for manual testing (grid/random are quick; ~30 SMAC trials ≈ 20s).
Making runs background so the UI stays responsive is Step 6; keeping it
synchronous here keeps the step about "does HPO work at all."

### No database
Nothing is persisted. One page: configure → run → results. Uploaded CSVs go to
a temp file that's deleted after the run. Persistence (saving, listing,
revisiting) arrives when a later step first needs it.

## 2. Feature checklist

Extracted from `app/experiment_form.py` (create form) and the run path in
`app/experiment.py` (`_start_run` + run controls). ✅ implemented · 🔄 kept but
changed · ⏳ deferred (step).

### Create form (`experiment_form.py`)
- ✅ Optimizer selection (registry dropdown)
- ⏳ Optimizer parameter fields via `_render_param` int/float/select — **Step 6** (dynamic HTMX subform); Step 3 runs optimizers with default params
- 🔄 Dataset selection — demo dropdown **or** file upload (was: demo dropdown, or a typed path + tkinter Browse; the picker/path are gone by design)
- ✅ Model selection (registry dropdown)
- ⏳ Custom model `.py` (upload/mounted) — **Step 9**
- ✅ Experiment name (required — `err_name_required`)
- ❌ Name-uniqueness check (`err_name_exists`) — N/A: nothing is stored to collide with (returns when persistence lands)
- ✅ Seed field with negative → random (`resolve_seed`, mirrors `experiment_form.py:160`)
- ✅ Dataset-required validation (`err_dataset_required`, as the `clean()` rule)
- ✅ Deterministic 80/20 stratified split with fallback + separator sniffing (via `core.io._load_splits`)
- 🔄 All metrics scored; primary metric chosen in the form (Streamlit chose it at run time via the display-metric selector — same end state, one screen instead of two)

### Run path (`experiment.py`)
- ✅ `n_trials` input (1–1000, default 30; Streamlit's step=5 is cosmetic, omitted)
- ✅ Run triggers the optimizer over all metrics with the chosen primary
- ✅ Best config + best score shown; best trial by primary metric (`experiment.py:270-273`)
- ✅ "Search space exhausted" note when `len(trials) ≥ trials_limit` (`warn_trials_exhausted`)
- ✅ Per-trial scores table (all metrics)
- 🔄 Synchronous run (was a daemon thread + `cancel_event` + `st.fragment` polling) — **background/cancel/resume: Step 6**
- ⏳ Analytics panels — best-config/selected-config/hp-importance/performance charts — **Step 4** (shown as tables for now)
- ⏳ Display-metric switching + re-evaluate button — **Step 6**
- ⏳ Metric-change confirmation dialog + `primary`/`original` rules — **Step 6**
- ⏳ Load-dataset / load-model recovery pickers (read-only mode) — **Step 5**

## 3. What is now possible

- Open the app, click **New**, choose Random Forest or SVM, choose SMAC / Random
  Search / Grid Search, pick a demo dataset (iris/wine) **or upload your own CSV**,
  set the metric / seed / trial count, and hit **Run experiment**.
- See the best score, the best hyperparameter configuration, and a per-trial
  table scoring every metric.
- This makes the Step 2 engine manually verifiable: the same optimizers, splits,
  and results you saw in tests, now driven by hand through the browser.

## 4. Verify it yourself

```bash
cd ~/Downloads/python/projects/codesigner
.venv/bin/python manage.py runserver        # http://127.0.0.1:8000/ → New
.venv/bin/python -m pytest tests/ui/experiments tests/ui/runs   # one slow SMAC run
```

Try, side by side with the Streamlit app on the same dataset/seed/optimizer:
Grid Search on iris (fast, and shows the "exhausted" note), Random Search, then
a small SMAC run (a handful of trials — it blocks the page while it thinks). Feed
it an uploaded CSV. Submit with no dataset and with no name to see the form
errors. Best score and config should match what the Streamlit app produces for
the same inputs.

## 5. Notes

- The DB-backed tests written ahead of schedule (models, adapter, `import_ihpo`,
  DB list/detail) now live in [tests/deferred/](../../tests/deferred/) and are
  intentionally red; they move into whichever later step first needs a database.
  Run the scheduled suite with `pytest --ignore=tests/deferred`.
- The create-and-run contract was validated against the Streamlit repo first
  (`InteractiveHPO/tests/test_setup_and_run.py`) before porting here, per the
  TDD process.
