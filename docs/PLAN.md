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
- `app/analytics/*.py` — the Plotly chart-building code ports nearly verbatim once `st.*` wrappers are stripped (serve via `fig.to_json()` + plotly.js).
- `run.py:11-27` — the `MODELS`/`OPTIMIZERS`/`METRICS` registries move to a `registry.py` module.

## Decisions (rationale inline — flag any you want changed)

| Decision | Choice | Why |
|---|---|---|
| Location | New sibling directory `~/Downloads/python/projects/codesigner/` (own git repo) | Clean separation; Streamlit repo is the untouched oracle |
| Frontend | Django templates + HTMX + plotly.js (no SPA, no build toolchain) | Teaches core Django directly (views/templates/forms/CSRF); HTMX covers the Streamlit-like partial updates (2s run-status polling, modals, figure-driven panel swaps). A DRF+React SPA would double the learning surface for no benefit at this app's size |
| Database | SQLite, configured via `django-environ` so Postgres is later just a `DATABASE_URL` change | Single-box tool, low write volume; keeps the learning curve flat |
| Data model | `Experiment` row mirroring the `.ihpo` snapshot (result as JSONField in `serialize_result()` shape) + `Run` row (status, cancel_requested, error, timestamps). No `Trial` table initially — the result JSON stays canonical, keeping `.ihpo` import/export a near-passthrough | Full interop with Streamlit-produced `.ihpo` files; `test/test.ihpo`, `test/test2.ihpo` become fixtures |
| Background runs | DB-backed `Run` state from day one; execute in a `threading.Thread` first (Step 6), swap to **huey** (`SqliteHuey`, no Redis) in Step 10 | The load-bearing design is run-state-in-DB, not the executor. Views/polling/cancel are written against the DB and never change when the queue arrives. Celery is overkill; Django's built-in Tasks framework is the forward-looking alternative but its DB worker is still young |
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

> **Later removed (post-Step 9).** The standalone Inspect page was dropped —
> loading an `.ihpo` (the Load/import flow) covers viewing a file, so a
> separate no-save preview was redundant. Its view/URL/template and tests were
> removed. Also dropped around then: the **unique-name constraint** on
> experiments — names are now a human label and each experiment is told apart
> by its short `identifier`, so duplicate names (and re-importing the same
> `.ihpo`) are allowed.

### Step 3 — Run HPO from the browser  ← the app becomes usable here  *(done)*
**You can now:** set up an experiment (choose a model, an optimizer, a dataset — demo or uploaded — a metric to optimize, a seed, and how many trials), run the optimization, and see the results (best config, best score, per-trial scores). **Nothing is saved: one page, configure → run → results.**
**Teaches:** Django `Form`s + validation + CSRF, file uploads (`request.FILES`), GET-shows-form / POST-runs, rendering results through templates; keeping the run logic in a Django-free service module.
**Build:** `NewExperimentForm` (registry model + optimizer + demo-or-uploaded dataset + metric-to-optimize + seed + n_trials) ported from `app/experiment_form.py`. On submit, `ui/services/run.py` builds the split (`core.io._load_splits`) and **runs the optimizer synchronously** (the request blocks — fine for manual testing; grid/random are quick, ~30 SMAC trials on iris ≈ 20s). Results render as tables. All metrics are always scored; the chosen metric is the one the optimizer optimizes. Optimizer params use their defaults (editing them = Step 6). **No database, no persistence.**
**Verify:** in the browser, run each optimizer on iris/wine (demo and uploaded CSV), read the results; submit with no dataset / no name for the validation errors. Automated: `tests/ui/experiments` + `tests/ui/runs` — run-service contract (validated first against InteractiveHPO) + Django view tests, one real SMAC run marked `slow`.

> **Deviation from the original Step 3 — read this.** The first draft of Step 3 also promised a **database** (`Experiment`/`Run` models + admin), a row↔snapshot **adapter**, an **`import_ihpo`** command, a **persisted list/detail** page, and **delete** — i.e. experiments surviving between visits. On the instruction to keep Step 3 to the bare minimum to *run* HPO, all of that was **cut and is not yet scheduled**; its tests wait in `tests/deferred/`. Persistence will be slotted when a later step first needs durable server-side state — at the latest Step 6 (background runs must record status somewhere), possibly earlier if browser save/load (Step 5) motivates it. **Until then, every experiment is ephemeral.**

### Step 4 — Figures of the results  *(done)*
**You can now:** see interactive figures of a run — performance over trials with the running-best line, the best configuration, and hyperparameter importance — and switch which metric the figures show.
**Teaches:** static files, the `json_script` filter, thin views calling pure figure-builder functions.
**Build:** Ported the performance scatter and importance pie to `ui/figures/plots.py` as pure `(result, display_metric[, selected_idx]) → go.Figure` functions; the results page embeds `fig.to_json()` and renders with vendored `plotly.min.js`. Best-configuration shown as a per-metric panel.
**Verify:** side-by-side with Streamlit on the same experiment — best config, importance donut (+ its warning text), performance scatter + incumbent line; switching metric re-renders.

> **Two deviations from the original Step 4:** (1) **Metric switching is client-side**, not the planned `?metric=f1` GET param — every metric's figures ship embedded in the results page and a dropdown swaps them with JS. The GET-param approach needs the result to survive to a second request, which requires persistence (not built yet); it returns when experiments persist. (2) The **selected-config panel is deferred to Step 7** — with no click-to-select yet, it would only ever mirror the best-config panel. Minor: the scatter uses Plotly's default hover, not the Streamlit custom per-hyperparameter hover.

### Step 5 — Persistence: save/load `.ihpo`, and experiments that stick around  *(done)*
**You can now:** your experiments are **saved** — running one stores it, it appears in the sidebar, and you can revisit its detail page any time. You can **download** any experiment as an `.ihpo` file (Streamlit-loadable) and **load** one back by uploading it (optionally attaching its dataset, else read-only). Delete removes an experiment after a confirmation.
**Teaches:** ORM models + migrations, the admin, a custom JSONField, `FileField`/`MEDIA_ROOT` uploads, a context processor (sidebar on every page), management commands, file-download responses, `get_object_or_404`.
**Build:** `Experiment` (+ `Run`) models with a `SafeJSONField` that round-trips SMAC's non-finite floats; admin; `ui/services/snapshot.py` row↔snapshot adapter; `import_ihpo` command; export/import/detail/delete views; sidebar context processor. Running an experiment now persists it via the adapter (dataset copied into `MEDIA_ROOT`).
**Verify:** create→run persists and shows the detail page; sidebar lists it; export downloads a parseable `.ihpo`; import (a Streamlit fixture) lands on the detail page; delete works. Full suite 115 green; browser smoke test of the whole flow.

> **Deviations / notes for this step.** (1) **Running now persists** the experiment (this is what "introduce the database" bought us) — this wasn't in the original Step 5 build text, which was only export/import; it came from the mid-migration decision to add the DB now. (2) `new_experiment` **renders the detail page inline (200)** after saving rather than doing a POST→redirect→GET, so the Step 3/4 view tests (which assert a 200 with results) stay valid without being edited. (3) The **`Run` model exists but is inert** — its status/cancel lifecycle is wired up in Step 6 (background runs); only the schema lands now. (4) Custom-model files are still not adopted on import (Step 9).

### Step 6 — Background runs: launch, poll, cancel, resume, change metric  *(done)*
**You can now:** creating an experiment no longer runs it — press **Run** on its page and the optimization runs in the background while the page stays responsive; watch live status (polled every 2s), cancel, resume (trials continue), and change the optimized metric with a confirmation. The sidebar shows a spinner for a running experiment, and runs abandoned by a restart are cleared.
**Teaches:** separating create from run (matching Streamlit), background threads + polling, DB-backed run state for cross-process coordination, keeping decision logic pure.
**Build:** `create_run` / `execute_run` / `start_background_run` with `DbCancelFlag` (reads `Run.cancel_requested`); the detail page's Run form + a 2s-polling status partial that refreshes to results on completion; `run_logic` (metric-change decision); standalone Cancel; sidebar spinner (`Experiment.is_running`); `sweep_stale_runs` for orphaned runs.
**Verify:** live smoke test — create (no run), launch, poll to completion, resume 4→6 trials, metric-change warning + confirm → *Inconsistent*; 143 tests pass.

> **Deviations / notes.** (1) **Create and Run are now separate** (Streamlit's model), superseding the interim synchronous `new_experiment`; ~10 Step 3/4/5 view tests were rewritten to the create→run→poll flow (you approved this). (2) **Standalone Cancel** button (Streamlit only cancelled via delete); a cancelled run keeps its completed trials. (3) **No HTMX dependency** — a ~20-line vanilla shim (`poll.js`) reads `hx-get`/`hx-trigger` + the `HX-Refresh` header, so templates/tests stay HTMX-shaped. (4) The orphan sweep is a **`sweep_stale_runs` management command** (run at startup, wired in Step 10), not `AppConfig.ready()` — querying the DB during app init is discouraged and would hit the dev DB under pytest. (5) Viewing a metric (client-side, Step 4) is separate from the Run form's explicit **"optimize for"** field, resolving Streamlit's overloaded selectbox.

### Step 7 — Click a trial on the figure  *(done)*
**You can now:** click a point on the performance figure to see that trial's score and full hyperparameter configuration in a "Selected configuration" panel, with the clicked point highlighted on the figure. Before any click, it defaults to the metric's best trial (same as Best configuration, with no delta shown).
**Teaches:** a small partial-rendering endpoint fetched by client-side JS; keeping a shared helper between the page render and the AJAX endpoint so both agree on what a "selected trial" contains.
**Build:** Ported `app/analytics/selected_config.py` (score, delta-vs-best, config table) plus the `on_select` handler in `app/analytics/performance.py:59-70` (curve 0 only, i.e. ignore clicks on the incumbent line; `point_index` → trial). `_selected_panel_data(result, metric, idx)` is shared by `_detail_context` (the default, pre-rendered panel per metric) and the new `GET /experiments/<pk>/trial-panel/?metric=&idx=` endpoint (fetched on click). The click handler restyles the figure's markers via `Plotly.restyle` and swaps the panel's inner HTML via `fetch`.
**Verify:** live-checked against the `test2.ihpo` fixture — clicking trial 1 (index 0) shows a `-0.0875` delta vs. the best (trial 9); clicking the best trial itself shows no delta; bad metric/index both 400. 13 new tests, 157 total.

> **Deviation.** Selection does **not** survive a metric switch via query param, as originally planned — switching metrics resets the selection to that metric's best (a client-side snapshot-and-restore of each metric's default panel HTML, taken at page load). This matches the metric-switching mechanism already chosen in Step 4 (fully client-side, no page reload/query params) and keeps the figure's highlight and the panel's text always consistent with each other.
**Verify:** clicking points updates the selected-config panel without a reload; matches Streamlit.

### Step 8 — Use the app in another language  *(done)*
**You can now:** switch the interface between English, German, and Spanish.
**Teaches:** `LocaleMiddleware`, `{% translate %}`/`gettext`, `makemessages`/`compilemessages`, `set_language`.
**Build:** Wrap UI strings (English msgids = the values in `utils/strings.py`, so existing translations match); bring over the `de`/`es` catalogs; language dropdown; translation-completeness CI check.
**Verify:** switch to de/es — parity with the Streamlit app's translated UI; 10 new tests, 169 total.

> **Deviations / notes.** (1) The `utils/strings.py` symbolic-key shim is not carried over — Django keys `{% translate %}`/`gettext` directly on the English source, so the indirection is unnecessary. (2) Of 70 marked strings, 30 reuse the Streamlit catalog byte-for-byte; the other 40 are Codesigner-native (different phrasing, or UI that only exists here) and were translated fresh or adapted for Django's `%(name)s` placeholder syntax. (3) `check_translations.py` is reproduced as a pytest test (completeness) plus a parity test that diffs Codesigner's catalog against the InteractiveHPO oracle for shared msgids, rather than a standalone script. (4) One Step 6 test (`test_run_and_metric_controls_share_one_card`) was adjusted to count forms in the page body only — the new sidebar language switcher is a second `<form>` and must not be conflated with the run/metric controls.

### Step 9 — Bring your own model  *(done)*
**You can now:** upload your own model `.py` file to optimize, when the operator has enabled the feature.
**Teaches:** settings-gated features; a trust boundary (arbitrary code execution) documented and flag-controlled; composing FileField uploads + form `clean()` validation + the snapshot adapter.
**Build:** `ALLOW_CUSTOM_MODELS` setting; `NewExperimentForm` gains a gated model `.py` upload validated via `load_model_from_path` (resolving `model_name` to the model's `.name`); the snapshot adapter adopts the model file and emits an absolute `model_path`; `_model_available` gates `can_run` and the run view; import gains an optional model-reattach upload; README trust note.
**Verify:** 15 new tests (form, create, run, snapshot, read-only, gating), 184 total; live-checked the gated upload fields render and a custom model runs end-to-end.

> **Scope change (agreed).** The original Step 9 also folded in the huey task queue ("production-grade runs"). We split it: **custom models landed here; huey moves to Step 10 (deployment)**, since it is invisible plumbing whose payoff is operational, and its main argument here (isolating untrusted model code in a worker) is acceptable to defer for a local, gated, single-user tool. Consequence: custom-model code currently runs in the in-process thread executor, behind the flag — noted in the README.

### Step 10 — Package for deployment + task queue + final parity sign-off  *(implemented; container run + final side-by-side pending a Docker host)*
**You can now (as an operator):** deploy Codesigner as a container and run it as a web service, with runs handled by a real background task queue.
**Teaches:** huey (task decorator + consumer process); deployment hygiene: `DEBUG=False`, `collectstatic` + whitenoise, gunicorn, multi-process Docker.
**Build:** Step 6's thread body becomes a `@db_task()` huey task (SqliteHuey), `manage.py run_huey` as a second process (views change only the launch line) — this is where custom-model code gains worker isolation. Dockerfile/compose (gunicorn + huey consumer; keep the `pyrfr` wheel workaround and the demo `datasets/`/`mounted_models/` volume mounts); healthcheck; README; CI = tests + translation check + docker build.
**Verify:** `docker build && docker run` on a clean, display-less container: create → run (through the consumer) → cancel → export → import end-to-end; cancel/resume still work; **walk the full parity checklist against the Streamlit app one final time**. The old repo then simply retires.

> **Status / deviations.** Delivered in two parts: part 1 the huey task-queue swap ([walkthroughs/step-10-task-queue.md](walkthroughs/step-10-task-queue.md)), part 2 static/healthcheck/Docker + the mounted-model source ([walkthroughs/step-10-deployment.md](walkthroughs/step-10-deployment.md)). Full suite green (209). **Not executed in the build environment (no Docker daemon):** `docker build`/`docker run` and the final one-sitting side-by-side against a live Streamlit instance — both are set up and documented, pending a Docker host. WhiteNoise uses non-manifest compressed storage (manifest/hashed storage errors in tests/runserver without a prior collectstatic). The stale-run sweep runs in the *worker* entrypoint (a "running" row is only stale when its consumer died = a worker restart).

## Parity checklist (maintained as `PARITY.md` in the new repo, ticked per step)

Create experiment (registry model / custom model / demo dataset / uploaded dataset) · load `.ihpo` (incl. read-only when dataset missing) · save/export `.ihpo` (Streamlit-interoperable) · delete with confirm · run N trials (SMAC/grid/random) · resume with continuing trial numbers · cancel mid-run · metric change with confirmation rules · best-config panel · selected-config panel · HP-importance pie (+ warning text) · performance scatter with incumbent line and click-to-select · display-metric switching · sidebar with running spinner · en/de/es · demo mounts in Docker.

**Deliberate behavior changes (not bugs):** experiments persist across restarts and are shared by everyone using the same instance (new capability — there is no auth or per-user scoping by design); absolute dataset paths in old `.ihpo` files route through the re-upload / read-only flow; tkinter picker gone.

## First concrete actions on approval

1. Ask you for the new project's name/location, then scaffold the repo (Step 1).
2. Each step thereafter: concept walkthrough → implementation → your verification → sign-off → next step.
