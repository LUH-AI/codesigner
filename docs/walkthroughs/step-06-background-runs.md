# Step 6 — Background runs: launch, poll, cancel, resume, change metric

**Status:** implemented, awaiting your sign-off.
**You can now:** creating an experiment no longer runs it — you press **Run** on its page and the optimization runs **in the background** while the page stays responsive. You watch live status (polled every 2s), **cancel** a run, **resume** it (more trials continue the history), and **change the optimized metric** with a confirmation. The sidebar shows a spinner next to a running experiment, and runs abandoned by a restart are cleared.

---

## 1. Concepts introduced

- **Separating create from run** — matches the Streamlit app (its `new_form`
  only set things up; the Run button launched the work). Creating persists an
  experiment with no result; running is a separate POST on the detail page.
- **Background work + polling** — `start_background_run` runs `execute_run` in a
  daemon thread; the detail page polls a status endpoint every 2s and refreshes
  to results when the run finishes. This is the analog of Streamlit's
  `@st.fragment(run_every=2)`.
- **DB-backed run state** — a `Run` row carries status/cancel/error/timestamps,
  so the request, the worker, and the poller coordinate through the database
  rather than shared memory (`DbCancelFlag` reads the cancel request from the DB
  in place of a `threading.Event`).
- **Pure decision logic** — the metric-change rules live in
  [run_logic.py](../../ui/services/run_logic.py) as small tested functions,
  kept out of the view.

## 2. Feature checklist

From `app/experiment.py` (run lifecycle), `app/dialogs.py` (metric-change,
delete-cancel) and `app/sidebar.py` (spinner). ✅ done · 🔄 changed · ➕ added.

### Run lifecycle (`_start_run`, the commit/poll blocks)
- ✅ Launch a run (n_trials 1–1000, default 30) → background thread
- ✅ Run states pending / running / done / error → `Run.status`
- ✅ Live polling every 2s, swapping to results on completion (`HX-Refresh`)
- ✅ Run failure surfaced as an error message on the page
- ✅ **Resume**: running again continues the trial history (`previous_result`)
- 🔄 The Streamlit "display metric" selectbox + "Reevaluate" button became two
  things: **viewing** a metric is client-side (Step 4's switcher); the Run form
  has an explicit **"optimize for"** field. This removes Streamlit's overloaded
  selectbox and its `display_metric`/`display_metric_input` inconsistency.

### Metric change (`dialogs.open_metric_change_dialog`, `experiment.py:200-240`)
- ✅ First run commits `primary = original = chosen`
- ✅ A run that would change the optimized metric (while primary still equals
  original) shows a confirmation naming both metrics
- ✅ Confirm "new" → optimize the chosen metric (→ label becomes "Inconsistent");
  "old" → keep optimizing the original; cancel → nothing
- ✅ Rules captured as `decide_run` / `resolve_metric_change` / `apply_metrics`

### Cancel & delete
- 🔄➕ **Standalone Cancel button** — Streamlit only cancelled a run *by deleting*
  the experiment; here Cancel stops the run and keeps the experiment (completed
  trials are retained — progress survives a cancel)
- ✅ Delete still stops any active run (sets the cancel flag) before removing

### Sidebar & robustness
- ✅ Spinner next to an experiment with an active run (`Experiment.is_running`)
- ➕ **Orphaned-run sweep** — a `sweep_stale_runs` management command marks runs
  left pending/running by a restart as errored (Streamlit's runs simply vanished
  with the process)

## 3. Deviations / notes

- **~10 Step 3/4/5 view tests were rewritten** to the create→run→poll flow (you
  approved this when choosing "separate create/run, all background"). They now
  test creation (step3), charts on the detail page (step4), and create-persists
  (step5); the run flow is covered in step6. No test was changed to make failing
  code pass — the product behavior changed by design.
- **No HTMX dependency**: HTMX isn't available offline, so a ~20-line vanilla
  shim ([poll.js](../../ui/static/ui/poll.js)) reads the same
  `hx-get`/`hx-trigger` attributes and the `HX-Refresh` header. Templates and
  tests stay HTMX-shaped, so swapping in real HTMX later is trivial.
- **Orphan sweep is a management command**, not `AppConfig.ready()`: querying the
  DB during app initialization is discouraged and, under the test runner, would
  touch the *development* database before the test DB exists. Step 10's container
  entrypoint runs `manage.py sweep_stale_runs` at startup.
- The synchronous `run_experiment` helper is retained (its build-and-run path is
  unit-tested); the background engine is what the UI now uses.
- **Oracle validation**: the resume and cancel *engine* semantics were already
  validated against the Streamlit app in `tests/step2/test_optimizers`
  (resume/`PreCancelled`). Step 6's DB wiring, views, polling, and metric-change
  decision are codesigner-native (no runnable Streamlit equivalent).

## 4. Verify it yourself

```bash
cd ~/Downloads/python/projects/codesigner
.venv/bin/python manage.py runserver
.venv/bin/python -m pytest              # 143 tests
```

Create an experiment → on its page press **Run** (say 30 trials, accuracy). The
page shows a spinner and updates when it finishes — the sidebar spinner too.
Press **Run** again for more trials (the count grows — resume). Change **Optimize
for** to a different metric and Run: you'll get the confirmation; "Optimize f1"
flips the metric label to *Inconsistent*. Start a longer run and **Cancel** it.
Kill the server mid-run, restart, and run `manage.py sweep_stale_runs` — the
interrupted run is marked errored.
