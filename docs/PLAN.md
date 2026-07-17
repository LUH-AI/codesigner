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

### Step 1 — Walking skeleton
**Teaches:** Django project vs. app anatomy, `settings.py`, `manage.py runserver`, URLconf → view → template flow, built-in migrations.
**Build:** New repo; `startproject config .` + `startapp web`; env-driven settings via `django-environ` (`SECRET_KEY`, `DEBUG`, `DATABASE_URL`, `.env.example` committed); a base template with the two-column sidebar/main shell; hardcoded home page; `migrate` built-ins to SQLite.
**Verify:** `python manage.py runserver` serves the shell on :8000 while Streamlit still runs on :8501.

### Step 2 — Bring in the domain core + test safety net
**Teaches:** (not Django) — pytest discipline; establishes the parity baseline everything else is checked against.
**Build:** Copy `models/`, `optimizers/`, and the pure parts of `utils/io.py` into `core/`; copy `test/*.csv` and `test/*.ihpo` as fixtures. Write pytest suites: `.ihpo` round-trip (`save`→`parse`→`build_experiment`, both fixtures + legacy v1), `TrialCollector` resume/incumbent logic, seeded grid/random runs on iris (SMAC marked `slow`), `serialize_result`/`deserialize_result` inverse. Add a `FakeOptimizer` (instant, deterministic) for later run-lifecycle tests. CI: pytest.
**Verify:** `pytest` green with zero Django involvement.

### Step 2.5 — Minimal manual-testing GUI *(added mid-migration)*
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

### Step 3 — Data model, admin, and `.ihpo` import command
**Teaches:** ORM models, `makemigrations`/`migrate`, JSONField/FileField, the admin site, management commands, the Django shell.
**Build:** `Experiment` + `Run` models (per Decisions); register in admin; `createsuperuser`; management command `import_ihpo <path>` reusing `core.io.parse()` (copies the CSV into `media/datasets/`); a `web/services/snapshot.py` adapter mapping row ↔ `.ihpo` snapshot dict — the seam reused by export, runs, and loading.
**Verify:** `manage.py import_ihpo test/test.ihpo`; inspect the row in `/admin/` and via `manage.py shell`; adapter unit tests (row→snapshot equals the source file's fields).

### Step 4 — Read-only pages: experiment list + detail
**Teaches:** function-based views, URL path converters, template inheritance, context, `{% url %}`.
**Build:** Sidebar experiment list (inclusion tag), home page, experiment detail header (name/model/optimizer/metric, including the "Inconsistent" metric label logic from `app/experiment.py:14-19`) and config summary. No forms/charts/runs yet.
**Verify:** import both fixtures, click list → detail; facts match the Streamlit view of the same file.

### Step 5 — Analytics panels with Plotly
**Teaches:** static files, the `json_script` filter, thin views + pure chart-builder functions.
**Build:** Port `app/analytics/{best_config,selected_config,hp_importance,performance}.py` to `web/charts.py` as pure functions `(result, display_metric, selected_idx) → go.Figure`; detail view deserializes `Experiment.result`, embeds `fig.to_json()`, renders with vendored `plotly.min.js`. Display-metric switcher is a plain GET param (`?metric=f1`) — idempotent URL state instead of session state. Selected trial defaults to best-per-metric (`app/experiment.py:270-273`); click-to-select comes in Step 10.
**Verify:** side-by-side with Streamlit on the same `.ihpo`: identical best config, importance pie, performance scatter + incumbent line; metric switching re-renders. *Biggest pure-port step.*

### Step 6 — Forms: create and delete experiments
**Teaches:** Django `Form`s, `clean_*` validation, CSRF, POST/redirect/GET, file uploads, dynamic fields, first taste of HTMX.
**Build:** Port `app/experiment_form.py` to `NewExperimentForm`: name uniqueness (vs. DB), seed (negative → random), optimizer choice with param subform generated from each optimizer's `params_schema` (`OptimizerParam` → Integer/Float/ChoiceField), dataset = demo dropdown **or** upload. Optimizer switch re-renders the param subform via an HTMX GET. Registry models only (custom `.py` deferred to Step 11). Delete = POST-only confirm flow (ports `dialogs.open_confirm_delete`). `MODELS`/`OPTIMIZERS`/`METRICS` from `run.py` → `web/registry.py`.
**Verify:** create with uploaded CSV and with a demo dataset; duplicate-name and validation errors render; delete works; Django test-client form tests.

### Step 7 — `.ihpo` export and import through the browser
**Teaches:** non-HTML responses (content-disposition downloads), multi-step form flows, MEDIA file management.
**Build:** Export view (snapshot adapter → `{name}.ihpo` download, replacing `st.download_button`); import view porting `dialogs.open_load_dialog` — upload `.ihpo`, and if the stored `dataset_path` doesn't exist server-side, prompt for a dataset upload or load read-only (results panels only, no runs); name-collision handling.
**Verify:** cross-app round-trip both directions (Django export → Streamlit load, and vice versa); automated test asserting exported JSON == imported snapshot for the fixtures.

### Step 8 — Background runs, part 1: DB-backed state + threads ⚠ *the risky step*
**Teaches:** designing state for concurrent access, transactions/`refresh_from_db`, HTMX polling, why web processes shouldn't own long work (sets up Step 11).
**Build:** Port `_start_run` (`app/experiment.py:295-326`): Run form (n_trials, metric) → POST creates `Run(status=pending)`, launches a thread that (a) rebuilds everything from the DB via snapshot adapter + `build_experiment` — no shared live objects, (b) writes result JSON + status back to the DB, (c) gets a `DbCancelFlag` (`.is_set()` → cached read of `Run.cancel_requested`) instead of `threading.Event`. Port the metric-change confirm dialog and `primary_metric`/`original_metric` rules (`app/experiment.py:200-240`). Status partial polled via `hx-trigger="every 2s"`, swapping to result panels on completion; sidebar spinner; error rendering; cancel button. Plus a new capability Streamlit lacks: an `AppConfig.ready` sweep marking orphaned "running" rows stale after a server restart.
**Verify:** 30 SMAC trials on iris — page stays responsive, spinner → charts on completion; run again and confirm trial numbers continue (resume via `trial_offset`); cancel mid-run; kill and restart the server mid-run, confirm the Run is marked stale. Run-lifecycle tests use `FakeOptimizer`.

### Step 9 — i18n
**Teaches:** `LocaleMiddleware`, `{% translate %}`/`gettext`, `makemessages`/`compilemessages`, the `set_language` view.
**Build:** Wrap template/form strings (English msgids = the values in `utils/strings.py`, so existing translations match); copy `locale/{de,es}/LC_MESSAGES/app.po` → `django.po` + `compilemessages`; language dropdown posting to `set_language`; a translation-completeness CI check (adapting `utils/check_translations.py`).
**Verify:** switch to de/es — parity with Streamlit's translated UI; CI check green.

### Step 10 — Interactive chart selection (click a trial)
**Teaches:** HTMX partial endpoints + a little first-party JS; progressive enhancement.
**Build:** Port `on_select="rerun"` from `app/analytics/performance.py:59-70`: a `plotly_click` handler reads `point_index` (curve 0 only, as today) and HTMX-GETs `/experiments/<id>/trial/<idx>/panel/`, returning the selected-config partial + re-rendered scatter (selected marker restyled, matching lines 32-36). Selection survives metric switches via query param.
**Verify:** click points → selected-config table updates without reload; matches Streamlit side-by-side.

### Step 11 — Production task queue + custom models
**Teaches:** huey integration (task decorator, consumer process), settings-gated features.
**Build:** (a) Step 8's thread body becomes a `@db_task()` huey task (SqliteHuey); `manage.py run_huey` as a second process — views change only the launch line since cancel/polling are already DB-based. (b) Custom model `.py` upload gated by `ALLOW_CUSTOM_MODELS`, loaded via existing `load_model_from_path` in the worker; port the model-reattach flow (`app/experiment.py:96-147`); trust-model note in README (must be disabled on any public deployment).
**Verify:** full run through the consumer (`runserver` + `run_huey`); cancel/resume still work; custom model runs when enabled, absent from UI when disabled.

### Step 12 — Productionize + parity sign-off
**Teaches:** deployment hygiene: `DEBUG=False`, `collectstatic` + whitenoise, gunicorn, multi-process Docker.
**Build:** Dockerfile/compose (gunicorn + huey consumer; keep the `pyrfr` wheel workaround and `datasets/`/`models/` volume-mount dropdowns); healthcheck; README (setup, run, trust model); CI = tests + translation check + docker build.
**Verify:** `docker build && docker run` on a clean, display-less container: create → run → cancel → export → import end-to-end; **walk the full parity checklist against the Streamlit app one final time** (below). The old repo then simply retires — nothing to delete.

## Parity checklist (maintained as `PARITY.md` in the new repo, ticked per step)

Create experiment (registry model / custom model / demo dataset / uploaded dataset) · load `.ihpo` (incl. read-only when dataset missing) · save/export `.ihpo` (Streamlit-interoperable) · delete with confirm · run N trials (SMAC/grid/random) · resume with continuing trial numbers · cancel mid-run · metric change with confirmation rules · best-config panel · selected-config panel · HP-importance pie (+ warning text) · performance scatter with incumbent line and click-to-select · display-metric switching · sidebar with running spinner · en/de/es · demo mounts in Docker.

**Deliberate behavior changes (not bugs):** experiments persist across restarts and are shared by everyone using the same instance (new capability — there is no auth or per-user scoping by design); absolute dataset paths in old `.ihpo` files route through the re-upload / read-only flow; tkinter picker gone.

## First concrete actions on approval

1. Ask you for the new project's name/location, then scaffold the repo (Step 1).
2. Each step thereafter: concept walkthrough → implementation → your verification → sign-off → next step.
