"""Which half of an experiment a field belongs to, and how that stays true.

*What:* an experiment is two rows. `ExperimentData` is the work — everything an
`.ihpo` carries — and `Experiment` is this instance's record of it: who owns it,
which group draws its boundary, what this installation was asked to draw, how
its model runs here. Neither half is allowed to drift into the other.

That boundary used to live only as prose in `snapshot_from_experiment`, so a
field added to one long model joined the file or missed it depending on whether
somebody remembered to add a line there. Splitting the table moved the decision
to where a field is *declared* — and this is what makes the decision stick.

*How:* by asserting the claim rather than restating it. `IN_THE_FILE` is not a
second copy of `ExperimentData`'s field list; it says **where each field lands
in the snapshot**, which is a claim about the file format that a test can check
against a real one. A field added to the data half with no entry here fails,
and an entry whose path is not in the file fails too — so the list cannot be
satisfied by writing it down.
"""

import pytest

from core import io
from tests.conftest import FIXTURES_DIR
from ui.models import Experiment, ExperimentData
from ui.services import snapshot as adapter

pytestmark = pytest.mark.django_db

#: Every field of `ExperimentData`, and the path it takes in the `.ihpo`.
#:
#: The mapping is not one-to-one by name, which is the reason it is written out:
#: `cv_folds` and `test_size` are two columns and one `evaluation` section, and
#: `dataset` is a file whose *identity* travels while its bytes do not.
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
    "config_space": ("space",),
    "priors": ("priors",),
    "result": ("result",),
}

#: The instance half, and why each one stays here. Named rather than counted,
#: so that adding a column to `Experiment` is a sentence somebody has to write.
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


# ── every field is on a side, and says which ────────────────────────────────

def test_every_data_field_declares_where_it_lands():
    """A field added to the portable half with no entry above has not yet said
    whether it is part of the file. It cannot be both."""
    declared = set(IN_THE_FILE)
    actual = _concrete(ExperimentData, skip={"id"})

    assert actual - declared == set(), "in the data half but not in the file"
    assert declared - actual == set(), "claimed for the file but not a field"


def test_every_instance_field_says_why_it_stays():
    """The other direction. A column added here is a claim that it means
    nothing on another instance, and that claim should be written down."""
    assert _concrete(Experiment) == set(STAYS_HERE)


# ── and the claims are true of a real file ──────────────────────────────────

def test_the_file_really_has_those_places(client):
    """`IN_THE_FILE` is checked against a snapshot rather than trusted. A path
    that is merely asserted is a comment; one that is looked up is a test."""
    exp = _experiment()
    exp.data.config_space = {"hyperparameters": []}
    exp.data.priors = {"n_estimators": {"kind": "normal", "params": {}}}
    exp.data.save(update_fields=["config_space", "priors"])

    snapshot = adapter.snapshot_from_experiment(exp, provenance=True)

    for field, path in IN_THE_FILE.items():
        section = snapshot
        for step in path:
            assert step in section, f"{field}: no {'.'.join(path)} in the file"
            section = section[step]


def test_nothing_from_the_instance_half_travels():
    """The boundary in the other direction, which is the one that leaks. An
    owner or a group in an exported file would be this instance's answer
    offered to somebody who has different people and different groups."""
    exp = _experiment()
    snapshot = adapter.snapshot_from_experiment(exp, provenance=True)

    leaked = set(STAYS_HERE) & set(snapshot)
    assert leaked == set(), f"instance-only fields in the file: {sorted(leaked)}"


# ── the two halves cannot come apart ────────────────────────────────────────

def test_an_experiment_cannot_exist_without_its_work():
    """`Experiment.data` is NOT NULL, so a half-built experiment is not a state
    the database will hold — which matters because every page reads through it,
    and one that could be missing would be a 500 on every page."""
    from django.db import IntegrityError, transaction

    with pytest.raises(IntegrityError):
        with transaction.atomic():
            Experiment.objects.get_queryset().create(identifier="orphan1")


def test_creating_one_writes_both_halves():
    """`ExperimentManager.create` takes the fields flat and sends each to the
    side that declares it, so no caller has to know the order."""
    exp = Experiment.objects.create(
        name="both halves", model_name="Random Forest",
        optimizer_name="SMAC", metric_names=["accuracy"], seed=3)

    assert exp.data.name == "both halves"
    assert exp.data.seed == 3
    assert ExperimentData.objects.count() == 1


def test_refreshing_reloads_the_other_half():
    """Django reloads an instance's own columns and leaves cached relations
    alone. Before the split `result` was one of those columns and came back
    with the row; after it, "refresh and read the result" would have gone on
    returning the copy fetched first — the same words meaning something new."""
    exp = _experiment()
    exp.data.result  # load and cache the half

    ExperimentData.objects.filter(experiment__pk=exp.pk).update(
        result={"data": [], "primary_metric": "moved"})
    exp.refresh_from_db()

    assert exp.data.result["primary_metric"] == "moved"
