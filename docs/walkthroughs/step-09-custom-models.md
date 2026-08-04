# Step 9 — Bring your own model

**Status:** implemented, awaiting your sign-off.
**You can now:** upload your own model as a `.py` file (defining a `BaseModel`
subclass) when creating an experiment, and optimize it exactly like a built-in
model — provided the operator has left the feature enabled. The upload is
validated on the spot, runs like any other experiment, and travels with the
experiment through `.ihpo` export/import.

> **Scope note.** The plan's Step 9 bundled two things: custom models *and* a
> production task queue (huey). We agreed to do custom models now and fold huey
> into the deployment step (Step 10), because huey is invisible plumbing whose
> real payoff is operational — and its most cited benefit here (isolating
> untrusted model code in a worker process) is acceptable to defer for a local,
> gated, single-user tool. So for now custom-model code runs in the same
> in-process thread executor as everything else, behind the feature flag.

---

## 1. The Django concepts this leans on

Nothing brand new — this step is mostly *composition* of pieces you've already
seen, plus one operational idea:

- **A settings-gated feature.** `ALLOW_CUSTOM_MODELS` (read from the environment
  via `django-environ`, default on) turns the whole capability on or off. The
  form field, the import adoption, and the "can this run?" check all consult it.
  This is the standard way to ship a capability that's safe locally but must be
  disabled in another deployment context — the flag is the single boundary.
- **`FileField` uploads** (as with datasets in Steps 6–7): the model `.py` is
  stored under `MEDIA_ROOT/custom_models/`, and the snapshot adapter adopts and
  emits it the same way it does the dataset.
- **Form `clean()` doing real work:** the uploaded file is *validated by loading
  it*, and a valid load rewrites `model_name` to the model's own `.name`.

## 2. The trust model (the important part)

`core.io.load_model_from_path` **executes** the uploaded file to find the
`BaseModel` subclass inside it. That is arbitrary code execution by design.
It is gated by `ALLOW_CUSTOM_MODELS`, documented in the README as *must be off
on any shared/public deployment*, and there is no sandboxing — the flag is the
boundary. With the flag off: the upload field disappears, imported `.ihpo`
model files are not adopted, and a custom-model experiment loads read-only.

One honest wrinkle: validating the upload means executing it **during the web
request** (to read the model's name and confirm it's a valid `BaseModel`), just
as the Streamlit app did. When huey lands in Step 10, *running* moves to the
worker; the create-time validation load stays in the web process unless we
decide to defer it. Flagged, not hidden.

## 3. What was ported

Source: the custom-model half of `app/experiment_form.py` (create-with-custom-
model) and the reattach flow in `app/experiment.py:96-147`.

- **Create:** `NewExperimentForm` gains a `model_file` upload (present only when
  the flag is on). `clean()` prefers an uploaded file over the registry
  dropdown; `_load_uploaded_model` writes it to a temp file, calls
  `load_model_from_path`, and on success resolves `model_name` to `model.name`.
- **Persist / round-trip:** `snapshot.experiment_from_snapshot` adopts the model
  file (from an upload, or from the snapshot's `model_path` if present on disk
  and the flag is on). `snapshot_from_experiment` now emits an **absolute**
  `model_path` (it previously emitted the relative storage name, which the run
  engine couldn't have loaded).
- **Run:** unchanged — `execute_run` already rebuilds via `build_experiment`,
  which loads the custom model from `model_path`. The absolute-path fix is what
  makes this actually work.
- **Reattach on import:** the import form gains an optional model-`.py` field
  (gated). Attach it to make an imported custom-model experiment runnable;
  omit it and it loads read-only.
- **Availability gate:** `_model_available(exp)` — registry models are always
  available; a custom model needs its file present *and* the flag on. It drives
  both the detail page's `can_run` and the run view's guard.

## 4. Feature checklist

From the Streamlit custom-model handling. ✅ done · 🔄 changed · ⏭ deferred.

- ✅ Create an experiment with an uploaded custom model `.py`
- ✅ Upload validated by loading it; invalid file → form error, not a crash
- ✅ `model_name` resolves to the model's declared `.name`
- ✅ Run a custom-model experiment (trials, results, figures — all as normal)
- ✅ Export carries `model_path`; import adopts an attached model file
- ✅ Read-only when the model file is missing or the feature is disabled
- ✅ `ALLOW_CUSTOM_MODELS` gate: no upload UI, no adoption, no runs when off
- ✅ Trust model documented in the README
- 🔄 Reattach is a simple optional upload on the import form (mirroring the
  dataset flow), not Streamlit's separate model-picker dialog
- 🔄 Validation executes the upload in the web request (as Streamlit did),
  pending the worker in Step 10
- ⏭ The no-display "mounted models" picker (`mounted_models()`) — that's a
  Docker volume-mount concern, deferred to Step 10 with the demo mounts
- ⏭ Production task queue (huey) — deferred to Step 10 (see scope note)

## 5. The tests (`tests/ui/custom_models/`, 15 tests)

- **Form** (`test_custom_model_form.py`): upload field present/absent per flag;
  valid upload resolves the model name; invalid `.py` is a form error; registry
  path still works; a model is required one way or the other.
- **Flow** (`test_custom_model_flow.py`): the create view stores the file and
  resolved name; the adapter adopts an upload and emits an absolute, loadable
  `model_path`; `execute_run` optimizes a custom model to completion; the detail
  page is runnable with the file, read-only without it or with the flag off; the
  run view refuses an unavailable custom model.

Two gaps in the Step 8 i18n tests surfaced while adding this step's strings and
are now fixed: the completeness check also rejects **fuzzy** entries (which
`makemessages` had auto-guessed for one new string, and which gettext silently
refuses to compile).

## 6. Verify it yourself

```bash
cd ~/Downloads/python/projects/codesigner
.venv/bin/python -m pytest tests/ui/custom_models tests/ui/i18n   # 25 tests
.venv/bin/python manage.py runserver
```

Create an experiment; instead of picking a model, upload a `.py` with a
`BaseModel` subclass (see `tests/ui/custom_models/conftest.py` for a minimal one) — it runs
like any built-in model. Try an invalid `.py`: you get a form error. Restart
with `ALLOW_CUSTOM_MODELS=False` in `.env`: the upload field is gone, and an
existing custom-model experiment shows read-only.

I verified the run path end-to-end via the in-process test client (a custom
model optimized to 3 trials, result persisted) and live-checked that the gated
upload fields render on the new-experiment and import pages.
