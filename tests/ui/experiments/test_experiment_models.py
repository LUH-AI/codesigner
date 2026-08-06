"""Step 3 TDD: contracts for the Experiment and Run database models.

Written before the models exist — each test imports inside its body so the
suite collects cleanly while these fail. They define the store's shape:
an Experiment row mirrors an .ihpo snapshot; a Run row tracks one
optimization run's lifecycle with a DB-readable cancel flag.
"""

import pytest


def _snapshot_fields():
    """Minimal valid field values for creating an Experiment row directly."""
    return dict(
        name="exp-1",
        model_name="Random Forest",
        optimizer_name="Random Search",
        optimizer_params={},
        metric_names=["accuracy", "f1"],
        primary_metric="accuracy",
        original_metric="accuracy",
        seed=42,
    )


@pytest.mark.django_db
def test_experiments_may_share_a_name():
    """Two experiments can share a name; the identifier tells them apart.

    Names are a human label, not an identity — each row carries its own
    unique identifier, so the database allows duplicate names.
    """
    from ui.models import Experiment

    a = Experiment.objects.create(**_snapshot_fields())
    b = Experiment.objects.create(**_snapshot_fields())
    assert a.name == b.name
    assert a.identifier != b.identifier


@pytest.mark.django_db
def test_experiment_defaults_for_fresh_rows():
    """A freshly created experiment has no result, no files, and timestamps.

    Expect: result is None (JSONField, nullable — an experiment that has
    never run), dataset/model_file are falsy (no uploads yet), and
    created_at is populated automatically.
    """
    from ui.models import Experiment

    exp = Experiment.objects.create(**_snapshot_fields())
    assert exp.result is None
    assert not exp.dataset
    assert not exp.model_file
    assert exp.created_at is not None


@pytest.mark.django_db
def test_experiment_json_fields_round_trip():
    """optimizer_params and metric_names survive a save/reload cycle intact.

    JSONFields must preserve dict/list structure exactly — these values are
    fed back into optimizer constructors and metric registries.
    """
    from ui.models import Experiment

    fields = _snapshot_fields()
    fields["optimizer_params"] = {"numeric_steps": 7}
    Experiment.objects.create(**fields)

    reloaded = Experiment.objects.get(name="exp-1")
    assert reloaded.optimizer_params == {"numeric_steps": 7}
    assert reloaded.metric_names == ["accuracy", "f1"]


@pytest.mark.django_db
def test_run_lifecycle_defaults():
    """A new Run starts pending with cancellation unrequested.

    Expect: status "pending", cancel_requested False, error empty, and no
    finished_at — the state a run is in between form-submit and pickup.
    """
    from ui.models import Experiment, Run

    exp = Experiment.objects.create(**_snapshot_fields())
    run = Run.objects.create(experiment=exp, stopping={"max_trials": 10}, primary_metric="accuracy")

    assert run.status == "pending"
    assert run.cancel_requested is False
    assert not run.error
    assert run.finished_at is None


@pytest.mark.django_db
def test_runs_are_deleted_with_their_experiment():
    """Deleting an experiment removes its runs (no orphaned lifecycle rows).

    Mirrors delete-with-confirm semantics: the experiment and everything
    about it disappears together.
    """
    from ui.models import Experiment, Run

    exp = Experiment.objects.create(**_snapshot_fields())
    Run.objects.create(experiment=exp, stopping={"max_trials": 5}, primary_metric="accuracy")

    exp.delete()
    assert Run.objects.count() == 0


@pytest.mark.django_db
def test_experiment_and_run_are_registered_in_admin():
    """Both models appear in the Django admin registry.

    The admin is the Step 3 inspection surface; unregistered models would
    make the verify step impossible.
    """
    from django.contrib import admin
    from ui.models import Experiment, Run

    assert Experiment in admin.site._registry
    assert Run in admin.site._registry
