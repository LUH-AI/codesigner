"""`.ExperimentData` stores the work an `.ihpo` carries; `.Experiment` stores the
local record, such as owner, group, settings and model environment.
`IN_THE_FILE` and `STAYS_HERE` assign every field to one half.
"""

import pytest

from core import io
from tests.conftest import FIXTURES_DIR
from experiments.models import Experiment, ExperimentData
from experiments.services import snapshot as adapter

pytestmark = pytest.mark.django_db

#: `IN_THE_FILE` maps each `ExperimentData` field to its `.ihpo` path.
#: `cv_folds` and `test_size` share one `evaluation` section, and `dataset`
#: carries its identity without its bytes.
IN_THE_FILE = {
    "name": ("name",),
    "created_at": ("began_at",),
    "time_basis": ("timestamps",),
    "seed": ("seed",),
    "model_name": ("model", "name"),
    "model_file": ("model", "path"),
    "dataset": ("dataset", "path"),
    "optimizer_name": ("optimizer", "name"),
    "optimizer_params": ("optimizer", "params"),
    "metric_names": ("metrics", "names"),
    "current_metric": ("metrics", "current"),
    "original_metric": ("metrics", "original"),
    "cv_folds": ("evaluation", "folds"),
    "test_size": ("evaluation", "test_size"),
    "task": ("evaluation", "task"),
    "time_column": ("evaluation", "time_column"),
    "time_gap": ("evaluation", "gap"),
    "horizon": ("evaluation", "horizon"),
    "series_column": ("evaluation", "series_column"),
    "season": ("evaluation", "season"),
    "processing": ("dataset", "processing"),
    "demo_dataset": ("dataset", "demo"),
    "target_column": ("dataset", "target_column"),
    "config_space": ("space",),
    "priors": ("priors",),
    "result": ("result",),
}

#: Each new `Experiment` column needs a written reason in `STAYS_HERE`, beside
#: every other instance-only field.
STAYS_HERE = {
    "id": "the row",
    "data": "the other half",
    "identifier": "a handle for this instance's copy; two installs' copies are two experiments",
    "created_at": "when this instance first saw it, not when the work was done",
    "settings": "what this installation was asked to draw",
    "use_default_settings": "and whether it was asked at all",
    "owner": "an .ihpo has no notion of who made it",
    "group": "the boundary is this instance's",
    "trashed_at": "a bin is a state on this instance",
    "deleted_at": "and so is somebody's having deleted it",
    "draft": "a setup in progress here; a file holds only created experiments",
    "setup_saved": "and which of its steps have been saved, on this instance",
    "deleted_by": "and who did",
    "env_status": "what this machine made of the model",
    "env_error": "and why it failed here",
    "env_meta": "and what it built",
    "env_prepared_at": "and when",
}


def _concrete(model, skip=()):
    return {f.name for f in model._meta.get_fields()
            if getattr(f, "concrete", False) and f.name not in skip}


def _experiment():
    return adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


def test_every_data_field_declares_where_it_lands():
    """A new `ExperimentData` field must state whether the file carries it."""
    declared = set(IN_THE_FILE)
    actual = _concrete(ExperimentData, skip={"id"})

    assert actual - declared == set(), "in the data half but not in the file"
    assert declared - actual == set(), "claimed for the file but not a field"


def test_every_instance_field_says_why_it_stays():
    assert _concrete(Experiment) == set(STAYS_HERE)


def test_the_file_really_has_those_places(client):
    """`IN_THE_FILE` paths are looked up in a real snapshot."""
    exp = _experiment()
    exp.data.config_space = {"hyperparameters": []}
    exp.data.priors = {"n_estimators": {"kind": "normal", "params": {}}}
    exp.data.target_column = "species"
    exp.data.save(update_fields=["config_space", "priors", "target_column"])

    snapshot = adapter.snapshot_from_experiment(exp, provenance=True)

    for field, path in IN_THE_FILE.items():
        section = snapshot
        for step in path:
            assert step in section, f"{field}: no {'.'.join(path)} in the file"
            section = section[step]


def test_nothing_from_the_instance_half_travels():
    """An exported owner or group would mean nothing on an instance with
    different people and groups."""
    exp = _experiment()
    snapshot = adapter.snapshot_from_experiment(exp, provenance=True)

    leaked = set(STAYS_HERE) & set(snapshot)
    assert leaked == set(), f"instance-only fields in the file: {sorted(leaked)}"


def test_an_experiment_cannot_exist_without_its_work():
    """`Experiment.data` is a required column because every page reads through
    it, and a missing half would raise a 500."""
    from django.db import IntegrityError, transaction

    with pytest.raises(IntegrityError):
        with transaction.atomic():
            Experiment.objects.get_queryset().create(identifier="orphan1")


def test_creating_one_writes_both_halves():
    """`ExperimentManager.create` takes flat fields and routes each to the half
    that declares it."""
    exp = Experiment.objects.create(
        name="both halves", model_name="Random Forest",
        optimizer_name="SMAC", metric_names=["accuracy"], seed=3)

    assert exp.data.name == "both halves"
    assert exp.data.seed == 3
    assert ExperimentData.objects.count() == 1


def test_refreshing_reloads_the_other_half():
    """`Experiment.refresh_from_db` drops the cached `data`, because Django
    leaves cached relations alone and `result` would stay stale."""
    exp = _experiment()
    exp.data.result  # cache `data`

    ExperimentData.objects.filter(experiment__pk=exp.pk).update(
        result={"data": [], "primary_metric": "moved"})
    exp.refresh_from_db()

    assert exp.data.result is not None
    assert exp.data.result["primary_metric"] == "moved"
