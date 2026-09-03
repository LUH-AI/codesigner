# Model environments (post-parity epic)

**Status:** implemented, awaiting your sign-off.
**You can now:** upload a model that needs packages Codesigner does not have.
The file declares its dependencies with a PEP 723 header, Codesigner builds and
pins an environment from them, and every trial runs in a separate process inside
that environment. Uploading no longer executes anything in the web request.

Built in six commit groups (`d858e74` … `c7907a3`), after the security fixes in
`38a96f7`.

---

## Why

A hard requirement is that this works with **arbitrary** models. The previous
design could not satisfy it: the uploaded file was imported into the app's own
interpreter, so it could only use packages the app already had. There is no
single environment that works for every use case, and every package added to
accommodate one model constrains the next.

So a user model gets its own environment and its own process. The configuration
surface is the file itself — a PEP 723 inline-metadata header, which is the
Python standard for a single-file script declaring what it needs — and there is
no env-config UI to build or store.

**This is dependency isolation, not a security boundary.** The child runs as the
same user with the same filesystem and network. `ALLOW_CUSTOM_MODELS` is still
the boundary, and the README still says to turn it off on a shared deployment.

## 1. Reading an upload instead of running it

**Concept.** `core/model_source.py` inspects the bytes with `ast` and never
executes them: size cap → UTF-8 → `ast.parse` → a top-level class whose base
name ends in `BaseModel` → a **literal** `name = "…"` → the PEP 723 block
(the PEP's own regex plus stdlib `tomllib`).

- `inspect_model_source(bytes) -> (ModelSourceInfo | None, error)`. Taking bytes
  means the tempfile dance leaves the request path entirely.
- One heuristic, only when there is no header at all: a non-stdlib import with no
  declared dependency is a guaranteed later failure, so it is named now and the
  error prints `STARTER_HEADER` — the exact block to paste, exported from this
  module so the message and the README cannot drift.
- Listing `codesigner-model` gets a friendly error; it is supplied automatically.

**Checklist:** ✅ ast-only inspection · ✅ size cap (an `ast.parse` of a 100 MB
upload was a free DoS) · ✅ literal `name` · ✅ PEP 723 parse · ✅ starter header
in the error and the README · ✅ the "No BaseModel subclass found in the file."
message kept byte-identical, so its translations survive.

## 2. Models return predictions; the app scores

**Concept.** The contract is now
`fit_predict(config, X_train, y_train, X_val, seed=0) -> y_pred`.
`train_evaluate` and its `metrics` parameter are gone from the model contract.

- `model_sdk/` ships in-repo as **`codesigner_model`**, zero dependencies,
  `requires-python >= 3.9`, and owns the `BaseModel` ABC. `core/models/base.py`
  is now one import from it — two identical ABCs would not be the same class, so
  `issubclass` would be False across the seam.
- `core/metrics.py` holds `METRICS` / `resolve` / `score_all`, with a `float()`
  wrap so `np.float64` stops travelling into `TrialResult.scores`. Users never
  define metrics.
- `core/optimizers/trial.py::evaluate_trial` replaces the three duplicated inner
  loop bodies. A crash or timeout is **recorded, not raised**: cost 0.0 with
  `status=CRASHED`/`TIMEOUT` and the message in `additional_info` — a real data
  point for the search.
- **`y_val` never leaves the parent.** That is what makes "every model is scored
  the same way, by us" a property rather than a convention.

**Checklist:** ✅ SDK package + re-export · ✅ both built-ins converted ·
✅ metrics out of `ui` · ✅ one shared trial evaluation · ✅ SMAC's `status`
passed through to `TrialValue` (it was being dropped).

## 3. The subprocess seam

**Concept.** `core/modelhost/` — JSON Lines over stdin/stdout, one request in
flight, monotonic ids echoed back: `hello` (unsolicited, first), then `init`,
`trial`, `result`, `error`, `shutdown`.

- Because `hello` is unsolicited, `--describe` — spawn, read the greeting, exit —
  is how the app learns a model's name and search space without importing user
  code. That doubles as the deferred execution validation.
- **Asymmetric serialization, deliberately.** Parent → child: the split as `.npy`
  (object-dtype string labels have no other form, and the parent wrote these
  files seconds ago into a 0700 directory). Child → parent: predictions as a
  **JSON list**. Never unpickle what the isolated process produced.
- The split **ships**; it is not re-derived child-side. Re-deriving would put
  pandas and scikit-learn in every model env, and `train_test_split`'s stratified
  shuffle is only reproducible for a *given* scikit-learn — two envs on different
  pins would train on different rows than the parent scores against.
- `RemoteModel` wears the model interface (`name`, `get_config_space`,
  `fit_predict`), so the optimizer loops cannot tell where training happened.
- `harness.py` is executed, never imported. It puts the model file's **parent**
  on `sys.path`, so a model can import siblings.

**Checklist:** ✅ protocol + errors + arrays + client + harness · ✅ `--describe`
· ✅ `build_experiment(load_model=False)` so `execute_run` builds everything but
the model · ✅ context-managed lifetime around `optimizer.optimize` + an `atexit`
sweep · ✅ `start_new_session=True` and `killpg` (a model using `n_jobs=-1`
spawns a pool that would otherwise leak) · ✅ the child times its own
`cpu_time` with `process_time` — `thread_time` is thread-local and would read ~0
across a process boundary, and `RUSAGE_CHILDREN` only accumulates for children
that have *terminated*.

## 4. The environment lifecycle

**Concept.** An environment is pinned **per experiment** (resolved and locked
once, at creation); processes are **per run** — one child for all trials.

- `Experiment` gains `env_status` (none/pending/preparing/ready/failed/skipped/
  legacy), `env_error`, `env_meta` (a JSON bag, so this never needs another
  migration) and `env_prepared_at`. The migration's data step marks existing
  custom-model rows **legacy**: they keep working in-process behind an opt-in
  *Prepare* button, rather than triggering an unattended mass download on upgrade.
- `ui/services/modelenv.py` owns it: `uv_path`, `decide_mode`, the path helpers,
  the command builders, the env denylist, `prepare_environment`. Prepare is
  `uv lock --script` → `uv run --locked --with <sdk wheel> … --describe` (which
  materialises the env, validates it, *and* reconciles the model's declared name
  against the class's own).
- uv builds an environment from the header of the script it **runs**, and
  `uv run --script model.py` would run the model. So a small **generated runner**
  (`<stem>_runner.py`) carries the header and `runpy.run_path`s the harness. The
  lock lives beside it in MEDIA, so the path is stable across runs — copying to a
  temp dir per run would rebuild the venv every time.
- The page reuses the existing polling machinery verbatim: `_env_status.html`
  shaped like `_run_status.html`, an `env_status` view returning
  `HX-Refresh: true` when settled. `poll.js` needed no change.
- The immediate-mode thread dance came out of `start_background_run` into
  `ui/services/dispatch.py` first — otherwise preparing blocks the create POST
  in local dev.

**Checklist:** ✅ fields + migration + legacy data step · ✅ modelenv service ·
✅ generated runner · ✅ polled status, retry (POST-only, refused while a run is
in flight) · ✅ `sweep_stale_model_envs` — a build interrupted by a restart
becomes *failed*, **not** re-queued, because a build that killed the worker would
restart-loop.

## 5. Shipping uv

- `COPY --from=ghcr.io/astral-sh/uv:0.9.7` — no curl, no unpinned install
  script, and uv stays independent of the app's site-packages, which matters
  because its job is managing *other* interpreters.
- The SDK **wheel** is built at image build and passed with `--with`. Handing uv
  the source directory makes it fetch a build backend: fatal air-gapped.
- A separate `uv-cache` named volume, on the **worker only** — disposable and
  unbounded, where `data` is irreplaceable. `UV_PYTHON_INSTALL_DIR` inside it so
  downloaded interpreters survive a recreate.
- Settings for offline, python-downloads and the timeouts; `MODEL_ENV_DENYLIST`
  keeps `SECRET_KEY` / `DATABASE_URL` out of the child. A denylist rather than an
  allowlist, because proxy, certificate and index settings are numerous and
  operator-specific.
- HUEY gains a `consumer` config (`HUEY_WORKERS`, default 4). It defaulted to
  **one** worker, so an environment build downloading torch held the only worker
  and nothing on the instance could run. A build also outranks a run by priority,
  so a freshly uploaded model does not wait behind a queue of optimizations
  before its page stops saying *preparing*.

## 6. Containment and ops

Proportionate only, and not a sandbox: process-group kill, three timeouts
(prepare / startup / trial), a fresh empty `cwd` per run, a 200-line stderr ring
buffer, and `RLIMIT_CORE=0` / `RLIMIT_FSIZE` / `oom_score_adj` applied **by the
child to itself** (`preexec_fn` is unsafe from a threaded huey consumer), plus a
worker `mem_limit` in compose.

Explicitly **not** `RLIMIT_AS` (torch reserves huge virtual space; a meaningful
cap breaks legitimate models) or `RLIMIT_NPROC` (per-UID, so it counts the
parent's processes). `prune_model_envs` wraps `uv cache prune`; reclamation is
manual and the docs say so.

## What is now possible

- Upload a model that needs a package Codesigner does **not** have, and run it.
- Watch an experiment page go *preparing → ready* and read uv's own error
  verbatim when a dependency does not resolve, with a retry button.
- Two models with conflicting pins, in one instance, at the same time.
- Cancel a run and have it actually stop mid-trial — a hanging model used to
  wedge the single worker forever.
- Upload without executing anything in the web request.

## Verify

```bash
python -m pytest -m "not slow"        # 413
python -m pytest                      # + slow SMAC
python -m pytest -m uv                # real uv, offline; skipped without it
python manage.py check --deploy
```

The whole subprocess path is tested with `sys.executable` as the "environment" —
the app env has numpy, ConfigSpace and the SDK — so CI needs neither uv nor
network: happy round trip, a model that prints to stdout, one that raises on a
single config, one that sleeps past a short timeout, a cancel flag flipping
mid-trial, a child SIGKILLed, garbage on stdout, and `get_config_space`
reproducibility across the wire (`tests/core/`, 148 tests).

The uv paths run against a **fake uv** — a shim in `tmp_path` that writes a
canned lock and prints canned `--describe` JSON, or exits non-zero with canned
stderr — which exercises real argv, exit codes and timeout plumbing
(`tests/ui/model_envs/`, 43 tests). Command construction is pinned by exact
argv, including *which script uv is handed*; a weaker assertion there hid a real
bug that only real uv found (`test_real_uv.py` exists for that reason).

Live-checked: a model declaring `scikit-learn` going preparing → ready → run; a
bad dependency name showing uv's error on the page; and uv removed, confirming
the in-process fallback note appears and local runs still work.

## Known ceiling, accepted

Predictions-back means **label-based metrics only**. ROC-AUC and log-loss would
need an optional second method (`predict_proba`). All four current metrics are
label-based, so nothing is lost today — but it is a real ceiling, written down
here rather than discovered later. If user-defined metrics ever land, they land
app-side, not in the model env.

Also: a `mounted_models/` file with no PEP 723 header will not build an
environment. And `cpu_time` means "the model's CPU, measured the best way each
path allows" — `process_time` in a child (which also counts BLAS threads,
strictly better than before), `thread_time` in-process.

## Deliberately not done

- **A web/worker venv split.** The blocker is code, not packaging: `ui/views.py`
  imports `ui.registry` → `core.optimizers` → `smac_optimizer` → `smac`, so a
  "web-only" image would fail to boot. The prerequisite is a lazy, metadata-only
  optimizer registry — worth doing on its own merits. Note this epic *reduces*
  the pressure: user models were the unbounded growth vector and now live in
  uv's cache.
- **Stripping `build-essential`/`swig` from the runtime image** (~400 MB): uv now
  builds *user* dependencies inside the running container.
- **Seeded default packages** — that re-creates the coupling this removes.
- **Putting the lock in the `.ihpo`** — an imported lock is pins the uploader
  chose that the recipient never reviewed. Re-lock on import instead.
- **Real sandboxing** (seccomp, bubblewrap, a separate UID, network namespaces) —
  a later epic. The docs must not let "runs in its own environment" be read as
  "sandboxed".
