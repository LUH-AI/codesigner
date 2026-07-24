# Feature parity vs. InteractiveHPO (Streamlit)

Ticked when the feature works in Codesigner and has been verified side-by-side
against the Streamlit app. Target step in parentheses (see docs/PLAN.md).

- [x] Create experiment with registry model (3)
- [x] Create experiment with custom uploaded model `.py` (9 — gated by ALLOW_CUSTOM_MODELS)
- [x] Create experiment with demo/mounted dataset (3)
- [x] Create experiment with uploaded dataset CSV (3)
- [x] Run N trials — SMAC (3)
- [x] Run N trials — Random Search (3)
- [x] Run N trials — Grid Search (3)
- [x] Best config + per-trial scores shown after a run (3 tables, 4 per-metric panel)
- [x] Performance scatter with incumbent line (4)
- [x] HP-importance pie + warning text (4)
- [x] Display-metric switching (4 — client-side)
- [x] Selected-config panel (7)
- [x] Click a trial point to select it (7)
- [x] Export `.ihpo`, loadable by the Streamlit app (5)
- [x] Load `.ihpo` file, dataset present (5)
- [x] Load `.ihpo` read-only when dataset missing (5)
- [x] Sidebar experiment list; delete with confirmation (5)
- [x] Resume: trial numbers continue across runs (6)
- [x] Cancel mid-run (6)
- [x] Metric change with confirmation rules (6)
- [x] Sidebar running spinner (6)
- [x] Languages: en / de / es (8)
- [ ] Demo dataset & model volume mounts in Docker (10)

## Deliberate behavior changes (not bugs)

- Legacy `.ihpo` files (integer `version` field) are not supported; only the
  current string-version format is read.

- Experiments persist across restarts and are shared by everyone on the same
  instance — there is no auth or per-user scoping by design.
- Old `.ihpo` files with dead absolute dataset paths route through the
  re-upload / read-only flow instead of a native file picker.
- The tkinter file picker does not carry over; datasets and models are
  browser uploads or server-side mounts.
- Product name is **Codesigner** (was "Interactive HPO").
