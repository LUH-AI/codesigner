# An .ihpo that can recreate its experiment

## The goal

An `.ihpo` should carry everything needed to recreate an experiment — the seed,
how the initial points were collected and how many, every optimizer setting, the
validation scheme, and the history of what changed mid-experiment — and be
readable while doing it. It is meant to be a superset of SMAC's own output.

For SMAC's own files it already was. `result` mirrors `runhistory.json`, and
`result.optimizer_state` embeds `intensifier.json`, `optimization.json`,
`scenario.json` and `configspace.json` verbatim, keyed by the relative path each
was written to. What was missing is everything *around* the optimizer: which
data, which model, how a trial was evaluated, what the settings actually
resolved to, and how the experiment got to where it is.

## What was added

Six nested sections. Every key that existed before is untouched and in the same
place, which is why there is no version bump, no migration and no compatibility
shim — a file written today opens in an older build, and a file from an older
build opens here. A test pins that directly.

```jsonc
"data": {                      // fingerprint, not contents
  "filename", "sha256", "rows", "columns", "column_names", "target_column"
},
"model": {
  "kind": "registry" | "file", "name", "sha256",
  "dependencies", "requires_python", "python", "lock_sha256"
},
"evaluation": { "scheme": "holdout" | "kfold", "folds", "test_size", "stratified" },
"optimizer": {
  "name",
  "resolved": { ... },         // every setting with the blanks answered
  "initial_design": { "kind", "use_share_cap", "share_cap",
                      "use_trial_cap", "trial_cap",
                      "per_hyperparameter", "combine" }
},
"runs": [{                     // read back on import, not only written
  "index", "status", "started_at", "finished_at",
  "trial_range": [1, 6],       // which trials this run produced
  "primary_metric", "optimizer_params", "stopping", "stopped_by",
  "budget_told",               // what sized the initial design
  "trial_seconds", "error",
  "events": [{ "kind": "metric_changed", "from": "accuracy", "to": "f1",
               "at_trial": 6, "surrogate": "rebuilt_and_replayed" }]
}],
"environment": { "codesigner", "python", "packages": { ... } }
```

## Decisions worth the paragraph

**Fingerprints, not contents.** The dataset and the model are recorded by
SHA-256 rather than embedded, so the file stays a record and not an archive.
The digest does work rather than sitting there: importing with a dataset whose
digest disagrees is **refused**. That is the one way an experiment could go on
adding trials to a history they do not belong to, and nothing downstream — not
the incumbent, not the surrogate, not the importance — could tell, because every
new trial would look exactly as valid as the ones before it.

Two things are deliberately not refusals: a file with no fingerprint (anything
exported before this existed), and an import with no dataset at all, which has
always been allowed and gives a browsable, unrunnable experiment.

**`resolved` is for the reader; `optimizer_params` is for the machine.** A blank
setting means "whatever that component already does", which is the right thing
to store and useless to read six months later — nobody knows what SMAC's random
forest uses for its leaf size. `resolved` answers the blanks by reading them out
of the installed SMAC's own signatures rather than copying them here, because a
copy is a second source of truth that goes stale silently. Reconstruction still
goes through `optimizer_params`, so a SMAC that changes a default later
reproduces the same *request* rather than freezing today's answer to it — and
`environment.packages.smac` says which version answered.

The blanks that stay blank belong to the other search strategy, which has no
default for them because it has no such component. `BlackBoxFacade` has no
forest to have a tree count of.

**`stratified` is resolved, not intended.** Both schemes ask for stratification
and fall back when scikit-learn refuses the target. Recording the request would
put a claim in the file that the run did not honour.

**One seed, still one field.** An earlier draft justified restructuring the file
by claiming `seed` meant three different things. It does not. `Experiment.seed`
drives the split, the config-space sampler, SMAC's `Scenario` — which fans it out
to the surrogate, maximizer, initial design, random design and intensifier —
`fit_predict`, and the importance computation. One source of randomness for the
whole experiment is what makes it reproducible from (dataset, seed, setup), and
splitting it across sections would have suggested otherwise.

**`runs[]` turns a flat list of trials back into a history**: which run produced
which trials, what bounded it, why it ended, and what it ran under. That last one
matters because the settings are editable *between* runs, so an experiment's
current settings are not the ones its earlier trials came out of — a record that
said they were would be wrong about every experiment anyone ever adjusted.

It was **write-only** to begin with. `experiment_from_snapshot` never looked at
it, so an imported experiment kept its trials and lost the structure over them —
a round trip was lossy in exactly the section added to stop the file being lossy.
It is read back now, and `test_exporting_what_was_imported_gives_the_same_record`
pins that export → import → export comes back identical. Two fields were added to
make that true (`trial_seconds`, `error`). `started_by` is deliberately not
recorded: an account on the exporting instance is not an account here, and
inventing a local one would put a name against work they did not do. A run that
had not finished when the file was written comes back `cancelled` rather than
`running`, or `Experiment.is_running` would be true for ever and the page would
poll a run that cannot report.

**The metric change is an event**, and the case the format could not previously
describe at all. It is recorded for what it costs: the whole history is re-read
under the new metric, and an optimizer carrying a fitted model of the objective
throws it away, because it was fitted to costs from a different question. An
optimizer that fits nothing records `"surrogate": "none"` rather than claiming
work that never happened — `BaseOptimizer.fits_surrogate` is what distinguishes
them.

**Provenance is opt-in on the way out.** `snapshot_from_experiment` is not only
the exporter: the run engine rebuilds through it on every run and the detail
page on every page load. The dataset fingerprint reads and hashes the file, so
export passes `provenance=True` and nothing else does.

## Where it lives

| | |
|---|---|
| `core/provenance.py` | Computes the fingerprints, the evaluation record and the environment; owns the digest comparison. No Django. |
| `core/optimizers/base.py` | `resolved_params()` (identity by default), `fits_surrogate`. |
| `core/optimizers/smac_optimizer.py` | `resolved_params()` reading defaults out of the facade signatures. |
| `ui/services/snapshot.py` | Assembles the sections; `runs[]` and the optimizer record. |
| `ui/services/run.py` | Records each run's settings, trial offset and metric-change event. |
| `ui/models.py` + `0014` | `Run.optimizer_params`, `Run.trial_offset`, `Run.events`. |
| `ui/services/snapshot.py` | `_restore_runs` rebuilds the history on import. |
| `ui/views.py`, `import_ihpo` | Refuse a dataset that disagrees with the record. |

## Verification

```bash
python -m pytest tests/ui/storage/test_provenance.py
python -m pytest -m slow tests/ui/storage/test_provenance.py
```

The one that matters is
`test_an_experiment_recreated_from_its_file_produces_the_same_trials`: run an
experiment, export it, import the file with the same dataset into a fresh
experiment, run that, and get the same trials in the same order. If anything the
file records is wrong or missing, that is where it shows.
