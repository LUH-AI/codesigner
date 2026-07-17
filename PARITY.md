# Feature parity vs. InteractiveHPO (Streamlit)

Ticked when the feature works in Codesigner and has been verified side-by-side
against the Streamlit app. Target step in parentheses.

- [ ] Create experiment with registry model (6)
- [ ] Create experiment with custom uploaded model `.py` (11)
- [ ] Create experiment with demo/mounted dataset (6)
- [ ] Create experiment with uploaded dataset CSV (6)
- [ ] Load `.ihpo` file, dataset present (7)
- [ ] Load `.ihpo` read-only when dataset missing (7)
- [ ] Export `.ihpo`, loadable by the Streamlit app (7)
- [ ] Delete experiment with confirmation (6)
- [ ] Run N trials — SMAC (8)
- [ ] Run N trials — Random Search (8)
- [ ] Run N trials — Grid Search (8)
- [ ] Resume: trial numbers continue across runs (8)
- [ ] Cancel mid-run (8)
- [ ] Metric change with confirmation rules (8)
- [ ] Best-config panel (5)
- [ ] Selected-config panel (5)
- [ ] HP-importance pie + warning text (5)
- [ ] Performance scatter with incumbent line (5)
- [ ] Click a trial point to select it (10)
- [ ] Display-metric switching (5)
- [ ] Sidebar experiment list with active highlight (4)
- [ ] Sidebar running spinner (8)
- [ ] Languages: en / de / es (9)
- [ ] Demo dataset & model volume mounts in Docker (12)

## Deliberate behavior changes (not bugs)

- Experiments persist across restarts and are shared by everyone on the same
  instance — there is no auth or per-user scoping by design.
- Old `.ihpo` files with dead absolute dataset paths route through the
  re-upload / read-only flow instead of a native file picker.
- The tkinter file picker does not carry over; datasets and models are
  browser uploads or server-side mounts.
- Product name is **Codesigner** (was "Interactive HPO").
