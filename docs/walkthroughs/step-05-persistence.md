# Step 5 — Persistence: saved experiments + `.ihpo` save/load

**Status:** implemented, awaiting your sign-off.
**You can now:** experiments are **saved** — running one stores it, it shows in the sidebar, and you can revisit its detail page later. You can **download** any experiment as an `.ihpo` file and **load** one back by uploading it (optionally attaching its dataset, else read-only), and **delete** with a confirmation.

---

## 1. Concepts introduced

### Models, migrations, the admin
[ui/models.py](../../ui/models.py) defines `Experiment` (the DB mirror of an
`.ihpo` snapshot) and `Run` (one run's lifecycle — schema only for now).
`makemigrations` generated [ui/migrations/0001_initial.py](../../ui/migrations/0001_initial.py);
`migrate` creates the tables. Both are registered in
[ui/admin.py](../../ui/admin.py), so `/admin/` is a full inspection surface.

### A custom field for a real-world snag
SMAC's stored scenario state contains `inf` (no-limit sentinels). SQLite's JSON
column rejects the `Infinity` token Python emits, a constraint the file-based
`.ihpo` store never had. [ui/fields.py](../../ui/fields.py)'s `SafeJSONField`
stores non-finite floats as a sentinel and restores them on read — valid JSON
in the DB, full fidelity in Python (verified: `inf` survives a write→read).

### FileField / MEDIA, context processor, downloads, get_object_or_404
Datasets are stored as `FileField`s under `MEDIA_ROOT` (served in dev via
[config/urls.py](../../config/urls.py)). The sidebar lists experiments on every
page through a **context processor**
([ui/context_processors.py](../../ui/context_processors.py)) — no per-view
plumbing. Export is a plain `HttpResponse` with a content-disposition header;
detail/delete use `get_object_or_404`.

### The adapter seam
[ui/services/snapshot.py](../../ui/services/snapshot.py) is the single
row↔snapshot converter. Import, export, running, and the detail page all go
through it, so the DB and the `.ihpo` format never drift.

## 2. Feature checklist

Ported from the Streamlit save/load/delete/sidebar flows. ✅ done · 🔄 changed · ⏳ deferred.

### Save / export (`app/experiment.py` download button)
- ✅ Download an experiment as `{name}.ihpo`
- ✅ Exported file is valid and Streamlit-loadable (round-trips through `io.parse`)
- ✅ Running an experiment now saves it (new: Streamlit kept experiments only in session state)

### Load / import (`app/dialogs.open_load_dialog`)
- ✅ Upload an `.ihpo`, validated by `core.io.parse` (same "Invalid or unreadable…" message)
- ✅ Name-collision refused ("already exists")
- ✅ Dataset re-supply: optional dataset upload adopted into `MEDIA_ROOT`
- ✅ Read-only load when no dataset is supplied (browsable, not runnable)
- ⏳ Registry-model substitution UI (choosing a replacement when the stored model is unavailable) — **not yet**; a file naming an unavailable model imports but its detail can't rebuild the result. Deferred alongside custom models (Step 9)
- ⏳ Custom-model `.py` adoption on import — **Step 9**

### Delete (`app/dialogs.open_confirm_delete`)
- ✅ Delete with a confirmation page; cancels its runs via cascade

### Sidebar (`app/sidebar.py`)
- ✅ Lists saved experiments, each linking to its detail page, active one highlighted
- ⏳ Running-spinner next to an active run — **Step 6**

### `import_ihpo` management command
- ✅ CLI import reusing `core.io.parse`; adopts an existing dataset into `MEDIA_ROOT`; rejects duplicates and invalid files

## 3. What is now possible

- Run an experiment → it's saved and appears in the sidebar; come back to it
  later via its detail page (figures and all).
- Download it as `.ihpo`, open it in the Streamlit app, and vice-versa.
- Load a colleague's `.ihpo` by uploading it; browse it read-only if you don't
  have the dataset, or attach the dataset to make it runnable.
- Delete experiments you're done with.
- Inspect everything in `/admin/`.

## 4. Verify it yourself

```bash
cd ~/Downloads/python/projects/codesigner
.venv/bin/python manage.py migrate        # first time
.venv/bin/python manage.py runserver
.venv/bin/python -m pytest                # 115 tests
```

Run an experiment → land on its detail page → see it in the sidebar. Hit
**Export**, then **Load** the downloaded file back under a new name. Load a
`tests/fixtures/*.ihpo` (Streamlit-produced) with and without a dataset. Delete
one. Cross-check an exported file loads in the Streamlit app.

## 5. Notes / deviations

- **Running persists** the experiment — this is the payoff of introducing the
  DB (the mid-migration decision), beyond the plan's original export/import
  scope. `new_experiment` renders the detail page inline (200) after saving,
  rather than POST→redirect→GET, so the Step 3/4 view tests stay valid untouched.
- The **`Run` model exists but is inert**; its lifecycle (status, cancel) is
  Step 6.
- Web-layer tests get DB access + an isolated `MEDIA_ROOT` from a shared
  `conftest.py`, so the pure-domain `tests/core` suite stays db-free.
- The DB-layer tests (models, adapter, `import_ihpo`, pages) moved out of
  `tests/deferred/`; the export/import/persist tests are new. They now live
  in `tests/ui/storage/` and `tests/ui/experiments/`.
