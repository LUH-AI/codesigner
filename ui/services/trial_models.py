"""Trial models kept for export as their parameters.

A run keeps a trial's fitted model — the one its last fold fitted, no extra
training — for the best trials and, if asked, every trial of the last run.
Each is stored in its library's own format beside a note of what it is (see
`ParameterFormat`), under ``MEDIA/trial_models/<experiment>/trial-<n>/``, and
recorded as a `TrialModel` so what a group stores is a sum (see
ui/services/storage.py).
"""

from __future__ import annotations

import shutil
from pathlib import Path


def discard(kept) -> None:
    """Delete a kept model: its files, then its record."""
    shutil.rmtree(kept.directory, ignore_errors=True)
    kept.delete()


#: What every kept model's note says it is.
STATUS = ("partially trained: fitted during the search on one fold's training rows only "
          "— a starting point to warm-start from, not a final model")


def root(exp) -> Path:
    """Where *exp*'s kept models live."""
    from django.conf import settings

    return Path(settings.MEDIA_ROOT) / "trial_models" / f"{exp.pk}-{exp.identifier}"


def _versions(model) -> dict:
    """The versions of what its parameters need, as installed here."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in tuple(getattr(model, "dependencies", ())) + ("scikit-learn", "skops", "numpy"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


class Keeper:
    """Keeps the fitted models a run's trials hand over — the best *best*
    trials of the experiment by the run's metric and, with *whole_run*, every
    trial of this run — and lets go of the rest.

    Called with each recorded trial and what its last fold fitted (see
    `core.optimizers.base.TrialCollector`). A model is written only while its
    trial is one to keep and the group has room (ui/services/storage.py):
    kept models never take the place of data. Whatever stops being one to keep
    — pushed out of the best by a better trial, or left from an earlier run —
    is deleted.
    """

    def __init__(self, run, model, metric, previous_trials, *, best, whole_run, folds,
                 columns=None):
        self.run, self.experiment = run, run.experiment
        self.model, self.metric = model, metric
        self.best, self.whole_run, self.folds = int(best), bool(whole_run), int(folds)
        self.columns = columns
        self.scores = {t.trial: t.score for t in previous_trials if not t.failed}
        self.this_run = set()
        #: Trials that were worth keeping but did not fit in the group's limit.
        self.no_room = 0
        # The previous run's models that are not among the best, gone before
        # this run keeps anything.
        self._evict()

    def keep_set(self) -> set:
        sign = -1 if self.metric.higher_is_better else 1
        ranked = sorted(self.scores, key=lambda n: (sign * self.scores[n], n))
        keep = set(ranked[:self.best])
        if self.whole_run:
            keep |= self.this_run
        return keep

    def _evict(self):
        from ..models import TrialModel

        keep = self.keep_set()
        for kept in TrialModel.objects.filter(experiment=self.experiment):
            if kept.trial not in keep:
                discard(kept)

    def __call__(self, trial, fitted):
        from core import parameters
        from core.optimizers.timing import STATUS_SUCCESS

        from ..models import TrialModel
        from . import storage

        if (trial.run_info or {}).get("status", STATUS_SUCCESS) != STATUS_SUCCESS:
            return
        self.scores[trial.trial] = trial.score
        self.this_run.add(trial.trial)
        if trial.trial in self.keep_set():
            folder = root(self.experiment) / f"trial-{trial.trial}"
            staging = folder.with_name(folder.name + ".new")
            shutil.rmtree(staging, ignore_errors=True)
            try:
                files = parameters.save(self.model, fitted, staging)
                (staging / "trial.json").write_text(self._note(trial, fitted, files))
            except parameters.NothingToKeep:
                shutil.rmtree(staging, ignore_errors=True)
                return
            size = sum(p.stat().st_size for p in staging.iterdir() if p.is_file())
            group = self.experiment.group
            if storage.has_room(group, size):
                shutil.rmtree(folder, ignore_errors=True)
                staging.rename(folder)
                TrialModel.objects.update_or_create(
                    experiment=self.experiment, trial=trial.trial,
                    defaults={"run": self.run, "directory": str(folder), "bytes": size})
                storage.forget(group)
            else:
                shutil.rmtree(staging, ignore_errors=True)
                self.no_room += 1
        self._evict()

    def _note(self, trial, fitted, files) -> str:
        import json

        from core.model_export import configuration
        from core.parameters import classes_of

        from ..views import _evaluation_label

        exp = self.experiment
        return json.dumps({
            "status": STATUS,
            "experiment": {"name": exp.name, "identifier": exp.identifier},
            "model": self.model.name,
            "task": exp.data.task,
            "trial": trial.trial,
            "hyperparameters": configuration(trial.config),
            "metric": self.metric.name,
            "scores": trial.scores,
            "evaluation": str(_evaluation_label(exp)),
            "fitted_on": (f"the training rows of fold {self.folds} of {self.folds}"
                          if self.folds > 1 else "the training rows of the one split"),
            "classes": classes_of(fitted),
            "columns": self.columns,
            "files": files + ["trial.json"],
            "versions": _versions(self.model),
            "how_to_load": self.model.parameter_howto(),
        }, indent=2, default=str)


def keeper_for(run, built, metric):
    """The `Keeper` for *run*, or None when it keeps nothing: the experiment
    asked to keep no models, or its model is one whose fitted state never
    reaches this process (uploads, in their own environment)."""
    from core import encoding
    from core.optimizers.trial import keeps_its_fit

    from .settings import resolve_settings

    model = built["model"]
    settings = resolve_settings(run.experiment)
    best = int(settings.get("keep_best_trial_models") or 0)
    whole_run = bool(settings.get("keep_last_run_trial_models"))
    if model is None or not keeps_its_fit(model) or not getattr(model, "has_parameters", True):
        return None
    if not best and not whole_run:
        return None
    columns = built.get("encoding")
    return Keeper(run, model, metric, list(built["result"].trials) if built["result"] else [],
                  best=best, whole_run=whole_run, folds=len(built["splits"].folds),
                  columns=columns if columns and not encoding.is_plain(columns) else None)
