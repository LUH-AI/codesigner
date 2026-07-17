# Step 2 — Domain Core + Test Safety Net

**Status:** implemented, awaiting your sign-off.
**What exists now:** the framework-free HPO engine lives in `core/`, exercised by a 33-test pytest suite that runs in ~20s, plus a GitHub Actions workflow.

---

## 1. Concepts introduced

This step is deliberately Django-free. Two ideas carry the whole migration:

### The core/ boundary
[core/](../../core/) is the domain engine: ML models, optimizers, and `.ihpo`
serialization. Its import rule is absolute: **nothing in `core/` may import
Django** (enforced by taste for now; the payoff comes in Step 8 when
background workers run `core` code with no web machinery in sight). The
Streamlit repo pioneered this layering — `models/` and `optimizers/` never
imported Streamlit — which is exactly why they could move here nearly
unchanged.

### Tests as the parity instrument
There was no test suite before. From now on, "did the port lose anything?" is
answered by `pytest`, not by hope. The suite pins down the behaviors the web
layer will rely on: trial numbering across resumes, incumbent tracking,
cancellation, the `.ihpo` format, and SMAC's state embedding.

Run them yourself:
```bash
cd ~/Downloads/python/projects/codesigner
.venv/bin/python -m pytest            # full suite, ~20s
.venv/bin/python -m pytest -m "not slow"   # skip SMAC end-to-end (~2s faster here; matters more on slow machines)
```

## 2. What was ported, file by file

| Old location | New location | Changes |
|---|---|---|
| `models/base.py`, `random_forest.py`, `svm.py` | [core/models/](../../core/models/) | imports made relative; otherwise byte-identical |
| `optimizers/base.py` | [core/optimizers/base.py](../../core/optimizers/base.py) | comment wording only |
| `optimizers/smac_optimizer.py` | [core/optimizers/smac_optimizer.py](../../core/optimizers/smac_optimizer.py) | comment wording only |
| `optimizers/grid_optimizer.py` | [core/optimizers/grid_optimizer.py](../../core/optimizers/grid_optimizer.py) | fixed ConfigSpace deprecation (`get_hyperparameters()` → `list(space.values())`) |
| `optimizers/random_optimizer.py` | [core/optimizers/random_optimizer.py](../../core/optimizers/random_optimizer.py) | unchanged |
| `utils/io.py` | [core/io.py](../../core/io.py) | see checklist — the one file with real deltas |
| `utils/version.py` | [core/version.py](../../core/version.py) | Codesigner starts at `0.1.0` |
| `test/*.csv` | [datasets/](../../datasets/) | doubles as demo data + runtime mount point |
| `test/*.ihpo` | [tests/fixtures/](../../tests/fixtures/) | `version` field patched from legacy int `2` to `"0.1.0"` |

## 3. Feature checklist

Extracted line-by-line from the ported sources. ✅ = ported & covered by a
test, ✅· = ported, exercised indirectly, ❌ = intentionally dropped.

### `utils/io.py` → `core/io.py`
- ✅ `.ihpo` format: all 12 documented top-level keys written by `save()` (round-trip test)
- ✅ `parse()` validation: JSON decode error, version check, 7 required-field checks — each with its human-readable `ValueError` message (tested individually)
- ❌ **Legacy versioning** (`_LEGACY_VERSIONS`, `_dict_to_result_v1`, the `version < 2` branch) — dropped at your direction; `parse()` accepts only string versions. This also removes two latent bugs we found (int-`2` fixtures rejected by `parse()`; `"0.2.0" < 2` TypeError in `_dict_to_result`)
- ✅ `build_experiment()`: registry model resolution by key *or* display name; custom-model path branch; `model_name` override param; `read_only` mode (arrays None, custom model skipped, registry model still resolved); unknown metric/optimizer/model errors; optimizer re-instantiated from stored params; result deserialized via the optimizer
- ✅ `attach_dataset()` / `attach_model()` in-place update semantics, path resolution
- ✅ `dataset_path_ok()` / `model_path_ok()` (registry models need no file)
- ✅ `_load_splits()`: 80/20 split, deterministic per seed, stratify-with-fallback, `,`/`;` separator sniffing (tested with a synthetic semicolon CSV)
- ✅· `load_model_from_path()`: temp-copy + importlib load, BaseModel-subclass discovery, abstract-method error message, instantiation error handling (exercised via `model_path_ok`/`attach_model` paths; full custom-model tests arrive with the feature in Step 11)
- ✅ `demo_datasets()` — now scans `datasets/` only (old app also scanned `test/`); mounted files still appear automatically
- 🔄 `mounted_models()` — mount dir renamed `models/` → `mounted_models/` (a top-level `models/` directory would be importable as a Python namespace package and shadow real imports)
- ❌ `pick_file()` (tkinter) and `has_display()` — no native file picker in a web app, per plan
- 🔧 `.values` → `.to_numpy()` in `_load_splits` — **found by the new tests**: pandas 3.x backs string columns with pyarrow arrays that sklearn can't index. Note: the Streamlit app shares this venv and code path, so its dataset loading is affected by the same bug today.

### `optimizers/base.py` → `core/optimizers/base.py`
- ✅ `OptimizerParam` schema dataclass (drives form rendering in Step 6)
- ✅ `TrialResult` / `OptimizationResult` dataclasses, `trials_limit`, `metadata`
- ✅ `TrialCollector`: `done` property, incumbent tracking, `trial_offset` numbering, `initial_best_*` resume semantics (4 dedicated tests)
- ✅ `serialize_result()`: runhistory-mirrored dict — stats/data/configs/config_origins, `cost = 1 − score`, `.ihpo` extension fields (`scores`, `incumbent_score`, `incumbent_config_id`), `best_config_id` recovery (shape-pinning test)
- ✅ `deserialize_result()` inverse (synthetic + 30-trial fixture round-trips)
- ✅ `get_params()` `self._<name>` convention
- ✅· `compute_hp_importance()`: HyperSHAP tunability → RandomForest-surrogate fallback → uniform fallback, with warning-message chaining (<2 trials and happy paths hit in optimizer tests; the two fallbacks are best verified visually in Step 5's pie chart)

### `optimizers/{smac,grid,random}_optimizer.py`
- ✅ Random: uniform sampling, duplicate-skip with 200-consecutive-dupes cutoff, resume, cancellation (tested)
- ✅ Grid: discretization (`numeric_steps`, categorical/ordinal/constant/int/float handling), ConfigSpace-validity filtering, `trials_limit` = full grid size, resume-skips-evaluated (tested: 2-step RF grid = 16 configs, resumed to completion without duplicates)
- ✅ SMAC: ask/tell loop with unreachable target stub, fixed `_SMAC_MAX_TRIALS` scenario hash, tempdir working directory, resume reusing a live dir (`overwrite=False`) vs. reconstructing from embedded state, `optimizer_state` embed/extract in serialize/deserialize, scenario.json `output_directory` rewrite (2 slow-marked tests)
- ✅ Cancellation contract: optimizers only call `.is_set()` — verified with a plain object, which is exactly the shape of Step 8's DB-backed cancel flag
- ✅· Module-level `OPTIMIZER`/`MODEL` sentinels kept (custom-file loader convention, Step 11)

### New, test-only
- `FakeModel` + `FakeOptimizer` ([tests/conftest.py](../../tests/conftest.py)) — instant, deterministic; will drive Step 8's run-lifecycle tests without training real models.

## 4. What is now possible

- The full HPO engine runs headless from a Python shell against this repo —
  every optimizer, resume, cancellation, and `.ihpo` round-trip — with no UI.
- `pytest` (33 tests) guards the engine; every later step builds on the same
  suite, and CI ([.github/workflows/test.yml](../../.github/workflows/test.yml))
  runs it on every push/PR once the repo lands on GitHub.
- Both projects share one venv (`.venv` here symlinks to
  `../InteractiveHPO/.venv`), including your patched SMAC checkout
  (`automl/SMAC3` branch `dev_1320`) — documented in
  [requirements.txt](../../requirements.txt) for fresh installs.

## 5. Findings worth your attention (old repo)

1. **pandas 3.x breaks dataset loading** in the shared venv: `utils/io.py`'s
   `.values` on a pyarrow-backed label column raises `TypeError` inside
   `train_test_split`. Codesigner is fixed (`.to_numpy()`); the Streamlit app
   will hit it when creating/loading experiments until the same one-line fix
   is applied there.
2. The bundled `test/*.ihpo` fixtures (int `version: 2`) are rejected by the
   current `parse()` (`_LEGACY_VERSIONS = (1,)`) — harmless for codesigner
   since we start clean, but the old repo's fixtures don't load in its own app.
