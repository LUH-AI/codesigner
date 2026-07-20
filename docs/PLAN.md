# Codesigner: Greenfield Django Rebuild of InteractiveHPO, to Feature Parity

> Approved 2026-07-17. Project name chosen after approval: **codesigner**
> (product branding "Codesigner" replaces "Interactive HPO" in all UI text).
> Progress is tracked in [PARITY.md](../PARITY.md); per-step write-ups live in
> [docs/walkthroughs/](walkthroughs/).

## Context

InteractiveHPO is a Streamlit workbench for interactive hyperparameter optimization (SMAC ask/tell, grid, random search over user-selectable models/datasets, with Plotly analytics and `.ihpo` save files). It was built for speed; the author has now been hired to professionalize it and wants to rebuild it on Django — **as a new, separate project**, not an in-place migration. The existing Streamlit app stays untouched and running throughout, serving as the reference implementation for feature parity. A second, equal goal: the author doesn't know Django yet, so the build order is pedagogical — each step introduces one coherent Django concept, ends in a runnable state, and gets verified against the Streamlit app before moving on.

**Workflow per step:** Claude implements the step, then delivers a written walkthrough containing:

1. A short explanation of the Django concept(s) the step introduces (what the files are, why Django structures it that way).
2. **A feature checklist**: before porting, Claude extracts every feature/behavior from the source files being ported — reading their code, comments, and docstrings line by line (e.g. every branch in `experiment.py`, every validation message in `experiment_form.py`) — and turns them into an explicit checklist. The walkthrough shows that checklist with each item marked implemented / deferred-to-step-N / intentionally changed, so nothing is silently lost.
3. A "what is now possible" list — the concrete user-facing capabilities the new app has after this step.

The user reads the walkthrough, reviews the code, runs the step's verification, and asks questions. No step starts until the previous one is signed off.

## What carries over vs. what's new

The existing repo is already well-layered: `models/`, `optimizers/`, and the serialization half of `utils/io.py` are pure Python with zero Streamlit imports. These get **copied into the new project** (as a `core/` package) essentially unchanged. Everything in `app/` (~36 KB of Streamlit view code) gets **rebuilt** in Django idioms. Key existing seams we reuse:

- `utils/io.py` — `.ihpo` format, `parse()` / `build_experiment()` / `save()`: can reconstruct a complete live experiment (model, optimizer, numpy splits, prior result) from JSON + file paths. A background worker never needs live Python objects — a task payload can be just a DB row ID.
- `optimizers/base.py` — `serialize_result()` / `deserialize_result()` define the runhistory-mirrored dict that becomes the `Experiment.result` JSONField; cancellation only requires an object with `.is_set()`, so `threading.Event` swaps for a DB-polled flag with zero changes to `optimizers/`.
- `app/analytics/*.py` — the Plotly figure-building code ports nearly verbatim once `st.*` wrappers are stripped (serve via `fig.to_json()` + plotly.js).
- `run.py:11-27` — the `MODELS`/`OPTIMIZERS`/`METRICS` registries move to a `registry.py` module.

## Decisions (rationale inline — flag any you want changed)

| Decision | Choice | Why |
|---|---|---|
| Location | New sibling directory `~/Downloads/python/projects/codesigner/` (own git repo) | Clean separation; Streamlit repo is the untouched oracle |
| Frontend | Django templates + HTMX + plotly.js (no SPA, no build toolchain) | Teaches core Django directly (views/templates/forms/CSRF); HTMX covers the Streamlit-like partial updates (2s run-status polling, modals, chart-driven panel swaps). A DRF+React SPA would double the learning surface for no benefit at this app's size |
| Database | SQLite, configured via `django-environ` so Postgres is later just a `DATABASE_URL` change | Single-box tool, low write volume; keeps the learning curve flat |
| Data model | `Experiment` row mirroring the `.ihpo` snapshot (result as JSONField in `serialize_result()` shape) + `Run` row (status, cancel_requested, error, timestamps). No `Trial` table initially — the result JSON stays canonical, keeping `.ihpo` import/export a near-passthrough | Full interop with Streamlit-produced `.ihpo` files; `test/test.ihpo`, `test/test2.ihpo` become fixtures |
| Background runs | DB-backed `Run` state from day one; execute in a `threading.Thread` first (Step 8), swap to **huey** (`SqliteHuey`, no Redis) in Step 11 | The load-bearing design is run-state-in-DB, not the executor. Views/polling/cancel are written against the DB and never change when the queue arrives. Celery is overkill; Django's built-in Tasks framework is the forward-looking alternative but its DB worker is still young |
| Live updates | HTMX polling every 2s (`hx-trigger="every 2s"`) | Direct analog of today's `@st.fragment(run_every=2)`; websockets/Channels unjustified for a spinner + status swap |
| Auth | **None during this rebuild.** The app must run locally/privately with zero auth friction, and also be deployable as a web service later. Everything is built auth-agnostic: no `owner` FKs, no `login_required`, no user references in models or views. When auth is eventually wanted, it should bolt on as an independent layer (middleware / decorator wrapping / reverse-proxy auth), touching as little of the codebase as possible | User decision: single platform, auth is a deployment concern, not a core-app concern |
| File handling | Datasets and custom-model `.py` files become `FileField` uploads under `MEDIA_ROOT`; the tkinter picker and absolute-path persistence do not carry over. Demo/mounted dropdowns (`demo_datasets()`, mounted `models/`) remain as a second source for Docker volume workflows | Fixes the headless/tkinter problem the README already acknowledges |
| Custom model uploads | Kept, but gated by an `ALLOW_CUSTOM_MODELS` env setting (default on for local use, documented as must-disable for public deployment), executed only in the worker process, trust model documented | `load_model_from_path` is arbitrary code execution by design; sandboxing is out of scope and said so explicitly |
| i18n | Django's native gettext: existing `locale/{de,es}` catalogs copy over (msgids are the English strings, matching `{% translate %}`) | First-class in Django; `utils/strings.py` shim not needed |

## The Steps

Each step = one PR-sized unit: concept walkthrough → implementation → verification → sign-off.

### Step 1 — Walking skeleton *(done)*
**You can now:** open Codesigner in a browser and see its two-panel shell.
**Teaches:** Django project vs. app anatomy, `settings.py`, `manage.py runserver`, URLconf → view → template flow, built-in migrations.
**Build:** New repo; `startproject config .` + `startapp web`; env-driven settings via `django-environ` (`SECRET_KEY`, `DEBUG`, `DATABASE_URL`, `.env.example` committed); a base template with the two-column sidebar/main shell; hardcoded home page; `migrate` built-ins to SQLite.
**Verify:** `python manage.py runserver` serves the shell on :8000 while Streamlit still runs on :8501.

### Step 2 — Bring in the domain core + test safety net *(done)*
**You can now:** *(nothing visible yet)* — the HPO engine (models, optimizers, `.ihpo` handling) runs and is covered by tests; it's the foundation every later step calls into.
**Teaches:** (not Django) — pytest discipline; establishes the parity baseline everything else is checked against.
**Build:** Copy `models/`, `optimizers/`, and the pure parts of `utils/io.py` into `core/`; copy `test/*.csv` and `test/*.ihpo` as fixtures. Write pytest suites: `.ihpo` round-trip (`save`→`parse`→`build_experiment`, both fixtures + legacy v1), `TrialCollector` resume/incumbent logic, seeded grid/random runs on iris (SMAC marked `slow`), `serialize_result`/`deserialize_result` inverse. Add a `FakeOptimizer` (instant, deterministic) for later run-lifecycle tests. CI: pytest.
**Verify:** `pytest` green with zero Django involvement.

### Step 2.5 — Inspect page *(done)*
**You can now:** upload an `.ihpo` file and read its contents in the browser (no saving, no dataset needed).
*(This also established the rule for the rest of the plan: every step ships the UI needed to manually test what it built.)*
Every migrated feature needs a way to be exercised by hand in the browser, not
just by pytest — the original plan deferred visible UI too long. This step
adds the minimum GUI to manually test Steps 2 and 3, and from here on every
step ships whatever small UI hooks are needed to manually test what it built.
**Teaches:** file uploads in views (`request.FILES`), rendering domain objects
through templates, 404 handling, the test client.
**Build:** (a) an **Inspect page** (no DB): upload any `.ihpo`, parse and
build it read-only via `core.io`, and render a summary (name, model,
optimizer, metric label, seed, trial count, best score) — parse errors render
the load-dialog's message. This exercises the Step 2 engine through the
browser. (b) **Experiment list + detail** (after Step 3 models): sidebar
lists DB experiments linking to a detail page with the header/caption fields
and metric-label rules from the Streamlit experiment view ("~" when no
original metric, "Inconsistent" when primary ≠ original), plus trial count /
best score; unknown ids 404. This makes `import_ihpo` results browsable.
**Verify:** upload both fixtures on the Inspect page and compare against the
Streamlit app; `import_ihpo` a fixture and browse to its detail page.

### Step 3 — Run HPO from the browser  ← the app becomes usable here  *(done)*
**You can now:** set up an experiment (choose a model, an optimizer, a dataset — demo or uploaded — a metric to optimize, a seed, and how many trials), run the optimization, and see the results (best config, best score, per-trial scores). **Nothing is saved: one page, configure → run → results.**
**Teaches:** Django `Form`s + validation + CSRF, file uploads (`request.FILES`), GET-shows-form / POST-runs, rendering results through templates; keeping the run logic in a Django-free service module.
**Build:** `NewExperimentForm` (registry model + optimizer + demo-or-uploaded dataset + metric-to-optimize + seed + n_trials) ported from `app/experiment_form.py`. On submit, `web/services/run.py` builds the split (`core.io._load_splits`) and **runs the optimizer synchronously** (the request blocks — fine for manual testing; grid/random are quick, ~30 SMAC trials on iris ≈ 20s). Results render as tables. All metrics are always scored; the chosen metric is the one the optimizer optimizes. Optimizer params use their defaults (editing them = Step 6). **No database, no persistence.**
**Verify:** in the browser, run each optimizer on iris/wine (demo and uploaded CSV), read the results; submit with no dataset / no name for the validation errors. Automated: `tests/step3` — run-service contract (validated first against InteractiveHPO) + Django view tests, one real SMAC run marked `slow`.

> **Deviation from the original Step 3 — read this.** The first draft of Step 3 also promised a **database** (`Experiment`/`Run` models + admin), a row↔snapshot **adapter**, an **`import_ihpo`** command, a **persisted list/detail** page, and **delete** — i.e. experiments surviving between visits. On the instruction to keep Step 3 to the bare minimum to *run* HPO, all of that was **cut and is not yet scheduled**; its tests wait in `tests/deferred/`. Persistence will be slotted when a later step first needs durable server-side state — at the latest Step 6 (background runs must record status somewhere), possibly earlier if browser save/load (Step 5) motivates it. **Until then, every experiment is ephemeral.**

### Step 4 — Charts of the results
**You can now:** see interactive charts of a run — performance over trials with the running-best line, the best configuration, and hyperparameter importance — and switch which metric the charts show.
**Teaches:** static files, the `json_script` filter, thin views calling pure chart-builder functions.
**Build:** Port `app/analytics/{best_config,selected_config,hp_importance,performance}.py` to `web/charts.py` as pure `(result, display_metric, selected_idx) → go.Figure` functions; the detail page embeds `fig.to_json()` and renders with vendored `plotly.min.js`. Display-metric switch is a `?metric=f1` GET param. Selected trial defaults to best-per-metric; click-to-select is Step 7.
**Verify:** side-by-side with Streamlit on the same experiment — identical best config, importance pie (+ its warning text), performance scatter + incumbent line; switching metric re-renders.

### Step 5 — Save and load `.ihpo` files in the browser
**You can now:** download any experiment as an `.ihpo` file and load one back in — files interoperate with the old Streamlit app — re-supplying the dataset (or loading read-only) when the original file isn't present.
**Teaches:** file-download responses (content-disposition), multi-step form flows, MEDIA file management.
**Build:** Export view (snapshot adapter → `{name}.ihpo` download). Import view porting `app/dialogs.open_load_dialog`: upload `.ihpo`; if the stored dataset is missing, prompt for a dataset upload or load read-only (browse results, can't run); registry-model substitution and name-collision handling.
**Verify:** round-trip both directions (Codesigner export → Streamlit load and vice versa); the read-only path works with no dataset.

### Step 6 — Background runs: cancel, resume, change metric
**You can now:** start a long run and keep using the app while it runs in the background, watch its progress live, cancel it, resume it (trial numbers continue), and switch the evaluation metric with a confirmation prompt.
**Teaches:** designing run state for concurrent access, `refresh_from_db`, HTMX polling, why web processes shouldn't own long work.
**Build:** Move the Step 3 synchronous run into a background `threading.Thread` writing `Run` status/result to the DB; `hx-trigger="every 2s"` status partial swapping to the results/charts on completion; sidebar running-spinner; cancel button via a DB `cancel_requested` flag (`.is_set()` shim, no `threading.Event`); resume via `trial_offset`; metric-change confirm rules from `app/experiment.py:200-240`; an `AppConfig.ready` sweep marking runs orphaned by a restart as stale (a capability Streamlit lacks).
**Verify:** launch 30 SMAC trials — page stays responsive, spinner → charts; resume continues numbering; cancel mid-run; restart the server mid-run and see the run marked stale.

### Step 7 — Click a trial on the chart
**You can now:** click a point on the performance chart to inspect that trial's configuration.
**Teaches:** HTMX partial endpoints + a little first-party JS; progressive enhancement.
**Build:** Port `app/analytics/performance.py:59-70`: a `plotly_click` handler HTMX-GETs a trial-panel partial and re-renders the scatter with the selected point restyled; selection survives metric switches via query param.
**Verify:** clicking points updates the selected-config panel without a reload; matches Streamlit.

### Step 8 — Use the app in another language
**You can now:** switch the interface between English, German, and Spanish.
**Teaches:** `LocaleMiddleware`, `{% translate %}`/`gettext`, `makemessages`/`compilemessages`, `set_language`.
**Build:** Wrap UI strings (English msgids = the values in `utils/strings.py`, so existing translations match); bring over the `de`/`es` catalogs; language dropdown; translation-completeness CI check.
**Verify:** switch to de/es — parity with the Streamlit app's translated UI.

### Step 9 — Bring your own model; production-grade runs
**You can now:** upload your own model `.py` file to optimize (when the operator enables it), with runs handled by a real background task queue instead of an in-process thread.
**Teaches:** huey (task decorator + consumer process), settings-gated features.
**Build:** Step 6's thread body becomes a `@db_task()` huey task (SqliteHuey), `manage.py run_huey` as a second process (views change only the launch line). Custom-model `.py` upload gated by `ALLOW_CUSTOM_MODELS`, loaded via `load_model_from_path` in the worker; model-reattach flow from `app/experiment.py:96-147`; trust-model note in the README (must be off on any public deployment).
**Verify:** full run through the consumer; cancel/resume still work; a custom model runs when enabled and is absent when disabled.

### Step 10 — Package for deployment + final parity sign-off
**You can now (as an operator):** deploy Codesigner as a container and run it as a web service.
**Teaches:** deployment hygiene: `DEBUG=False`, `collectstatic` + whitenoise, gunicorn, multi-process Docker.
**Build:** Dockerfile/compose (gunicorn + huey consumer; keep the `pyrfr` wheel workaround and the demo `datasets/`/`mounted_models/` volume mounts); healthcheck; README; CI = tests + translation check + docker build.
**Verify:** `docker build && docker run` on a clean, display-less container: create → run → cancel → export → import end-to-end; **walk the full parity checklist against the Streamlit app one final time**. The old repo then simply retires.

## Parity checklist (maintained as `PARITY.md` in the new repo, ticked per step)

Create experiment (registry model / custom model / demo dataset / uploaded dataset) · load `.ihpo` (incl. read-only when dataset missing) · save/export `.ihpo` (Streamlit-interoperable) · delete with confirm · run N trials (SMAC/grid/random) · resume with continuing trial numbers · cancel mid-run · metric change with confirmation rules · best-config panel · selected-config panel · HP-importance pie (+ warning text) · performance scatter with incumbent line and click-to-select · display-metric switching · sidebar with running spinner · en/de/es · demo mounts in Docker.

**Deliberate behavior changes (not bugs):** experiments persist across restarts and are shared by everyone using the same instance (new capability — there is no auth or per-user scoping by design); absolute dataset paths in old `.ihpo` files route through the re-upload / read-only flow; tkinter picker gone.

## First concrete actions on approval

1. Ask you for the new project's name/location, then scaffold the repo (Step 1).
2. Each step thereafter: concept walkthrough → implementation → your verification → sign-off → next step.
