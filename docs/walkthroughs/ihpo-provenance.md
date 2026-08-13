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

## The layout

**One object per subject, each stating its subject once, and `result` last.**

```jsonc
{
  "format": 2,                   // the shape of this file
  "version": "0.1.0",            // the application that wrote it
  "name": "…",
  "seed": 7,                     // one seed for the whole experiment

  "dataset": { "path",           // "" in an exported file
               "filename", "sha256", "rows", "columns",
               "column_names", "target_column" },

  "model": { "kind": "registry" | "file", "name", "path",
             "sha256", "dependencies", "requires_python",
             "python", "lock_sha256" },

  "evaluation": { "scheme": "holdout" | "kfold", "folds",
                  "test_size", "stratified" },

  "metrics": { "names": [...], "current", "original" },

  "optimizer": { "name",
                 "params": { … },          // as asked for; null means "as it comes"
                 "defaults_used": { … } }, // only the blanks, only the answers

  "runs": [{ "index", "status", "started_at", "finished_at",
             "trial_range": [1, 6],        // which trials this run produced
             "primary_metric", "stopping", "stopped_by",
             "trial_seconds", "error",
             "events": [{ "kind": "metric_changed", "from": "accuracy",
                          "to": "f1", "at_trial": 6,
                          "surrogate": "rebuilt_and_replayed" }] }],

  "environment": { "codesigner", "python", "packages": { … } },

  "result": { … }                // strict superset of SMAC's runhistory.json
}
```

The keys above split cleanly in two, which is what the grouping is for.
`format`, `version`, `name`, `seed`, and the `path`/`name`/`folds`/`names`/
`params` fields are **read back** — they are what reconstruction uses. Everything
else is **record**: written on export, never read, there so the file can be
understood without the machine that produced it. `snapshot_from_experiment`
writes the first set always and the second only when asked, because the run
engine and the detail page rebuild through the same function on every run and
every page load, and hashing the dataset is not free.

### What format 1 was, and why this is not additive

Format 1 was one flat namespace — `model_name`, `optimizer_name`,
`optimizer_params`, `cv_folds`, `dataset_path` — and the record sections were
added *beside* it, so nothing moved and old files kept opening. The cost was
that the file said several things twice: the optimizer was named at the top and
again in `optimizer.name`; `cv_folds` and `evaluation.folds` were the same
number under two names; `optimizer.resolved` repeated all 23 settings to answer
five blanks; `optimizer.initial_design` was five settings copied out of
`optimizer_params` plus `combine`, which was `initial_points_use_max` renamed.
None of it was wrong. All of it was a second place to look, and two copies of a
fact invite the question of which one ran.

So format 2 moves things, and that is a one-way break: **reading stays
backward-compatible, writing does not.** `io.normalize` lifts a format 1 file on
the way in — those files are on other people's disks and there is nothing wrong
with them — but a file written today will not open in a build from before this.
At 0.1.0, with the format published nowhere, that is the cheaper side of the
trade. `format` exists so the next such change need not be guessed at.

Three things were dropped rather than moved, all of them derivable:
`optimizer.initial_design` (every field is a setting already in `params`),
`runs[].budget_told` (`trial_range[0] − 1 + stopping.max_trials`), and
`runs[].optimizer_params` (the settings are fixed at creation, so the one copy
at the top describes every trial in the file).

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

**`defaults_used` is for the reader; `params` is for the machine.** A blank
setting means "whatever that component already does", which is the right thing
to store and useless to read six months later — nobody knows what SMAC's random
forest uses for its leaf size. `defaults_used` answers the blanks by reading them
out of the installed SMAC's own signatures rather than copying them here, because
a copy is a second source of truth that goes stale silently. It answers **only**
the blanks: an earlier cut wrote every setting resolved, which repeated twenty of
the twenty-three values sitting directly above it. Reconstruction still goes
through `params`, so a SMAC that changes a default later reproduces the same
*request* rather than freezing today's answer to it — and
`environment.packages.smac` says which version answered.

The blanks with no answer are simply absent. They belong to the other search
strategy, which has no default for them because it has no such component:
`BlackBoxFacade` has no forest to have a tree count of.

**`stratified` is resolved, not intended.** Both schemes ask for stratification
and fall back when scikit-learn refuses the target. Recording the request would
put a claim in the file that the run did not honour.

**One seed, still one field, and still at the top.** Everything else moved into
the object for its subject; the seed did not, because it has no one subject. An
earlier draft justified restructuring the file by claiming `seed` meant three
different things. It does not. `Experiment.seed`
drives the split, the config-space sampler, SMAC's `Scenario` — which fans it out
to the surrogate, maximizer, initial design, random design and intensifier —
`fit_predict`, and the importance computation. One source of randomness for the
whole experiment is what makes it reproducible from (dataset, seed, setup), and
putting a copy in each section would have suggested otherwise.

**`runs[]` turns a flat list of trials back into a history**: which run produced
which trials, what bounded it, and why it ended. Not what it ran under — the
search settings are chosen at creation and fixed for the experiment's life, so
the one copy at the top of the file describes every trial in it. An earlier cut
let them be edited between runs and therefore had to repeat them per run; the
capability went, and the duplication with it.

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
| `core/io.py` | `SNAPSHOT_FORMAT`, `normalize()` (either format in, the current one out), `save()`, `parse()`, `build_experiment()`. |
| `core/optimizers/base.py` | `resolved_params()` (identity by default), `fits_surrogate`. |
| `core/optimizers/smac_optimizer.py` | `resolved_params()` reading defaults out of the facade signatures. |
| `ui/services/snapshot.py` | Assembles the sections; `_defaults_used` and `runs[]`. |
| `ui/services/run.py` | Records each run's trial offset and metric-change event. |
| `ui/models.py` + `0014`, `0015` | `Run.trial_offset`, `Run.events`. |
| `ui/services/snapshot.py` | `_restore_runs` rebuilds the history on import. |
| `ui/views.py`, `import_ihpo` | Refuse a dataset that disagrees with the record. |

## Verification

```bash
python -m pytest tests/ui/storage/test_provenance.py tests/core/test_snapshot_contract.py
python -m pytest -m slow tests/ui/storage/test_provenance.py
```

The one that matters is
`test_an_experiment_recreated_from_its_file_produces_the_same_trials`: run an
experiment, export it, import the file with the same dataset into a fresh
experiment, run that, and get the same trials in the same order. If anything the
file records is wrong or missing, that is where it shows.

Format 1 is pinned by the two bundled fixtures. `tests/fixtures/test.ihpo` and
`test2.ihpo` were written before any of this and are deliberately not
regenerated: every test that parses them exercises `normalize`, so the day the
lift breaks, the contract tests break with it.
