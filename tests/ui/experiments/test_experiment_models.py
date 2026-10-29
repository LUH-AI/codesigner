"""`.Experiment` exposes an `.ihpo` snapshot's fields through `exp.data`, and `Run`
tracks one run with a cancel flag the database can read.
"""

import pytest


def _snapshot_fields():
    """`_snapshot_fields` gives minimal valid fields for
    `Experiment.objects.create`."""
    return dict(
        name="exp-1",
        model_name="Random Forest",
        optimizer_name="Random Search",
        optimizer_params={},
        metric_names=["accuracy", "f1"],
        current_metric="accuracy",
        original_metric="accuracy",
        seed=42,
    )


@pytest.mark.django_db
def test_experiments_may_share_a_name():
    """The unique identifier tells same-named experiments apart.
    """
    from experiments.models import Experiment

    a = Experiment.objects.create(**_snapshot_fields())
    b = Experiment.objects.create(**_snapshot_fields())
    assert a.name == b.name
    assert a.identifier != b.identifier


@pytest.mark.django_db
def test_experiment_defaults_for_fresh_rows():
    """A new row has no result or uploads, and gets `created_at`.
    """
    from experiments.models import Experiment

    exp = Experiment.objects.create(**_snapshot_fields())
    assert exp.data.result is None
    assert not exp.data.dataset
    assert not exp.data.model_file
    assert exp.created_at is not None


@pytest.mark.django_db
def test_experiment_json_fields_round_trip():
    """The JSON values feed optimizer constructors and metric lookups, so their
    structure must survive.
    """
    from experiments.models import Experiment

    fields = _snapshot_fields()
    fields["optimizer_params"] = {"numeric_steps": 7}
    Experiment.objects.create(**fields)

    reloaded = Experiment.objects.get(data__name="exp-1")
    assert reloaded.data.optimizer_params == {"numeric_steps": 7}
    assert reloaded.data.metric_names == ["accuracy", "f1"]


@pytest.mark.django_db
def test_run_lifecycle_defaults():
    """A new Run is pending, uncancelled and unfinished until a worker picks it
    up.
    """
    from experiments.models import Experiment, Run

    exp = Experiment.objects.create(**_snapshot_fields())
    run = Run.objects.create(experiment=exp, stopping={"max_trials": 10}, primary_metric="accuracy")

    assert run.status == "pending"
    assert run.cancel_requested is False
    assert not run.error
    assert run.finished_at is None


@pytest.mark.django_db
def test_runs_are_deleted_with_their_experiment():
    from experiments.models import Experiment, Run

    exp = Experiment.objects.create(**_snapshot_fields())
    Run.objects.create(experiment=exp, stopping={"max_trials": 5}, primary_metric="accuracy")

    exp.delete()
    assert Run.objects.count() == 0


@pytest.mark.django_db
def test_experiment_and_run_are_registered_in_admin():
    from django.contrib import admin
    from experiments.models import Experiment, Run

    assert Experiment in admin.site._registry
    assert Run in admin.site._registry
