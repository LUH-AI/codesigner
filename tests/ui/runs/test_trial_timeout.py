"""`core.modelhost` enforces the trial deadline, and the tests here follow the
form's choice into the Run row and `model_session`.
"""

import pytest
from django.test import override_settings
from django.urls import reverse

from tests.ui.runs.test_run_views import _experiment, no_thread  # noqa: F401


RUN_URL = "ui:experiment_run"


def _post(client, exp, **fields):
    data = {"max_trials": 3, "optimize_metric": "accuracy"}
    data.update(fields)
    return client.post(reverse(RUN_URL, args=[exp.pk]), data)


# Form to row

@pytest.mark.django_db
def test_a_fixed_deadline_reaches_the_run(client, no_thread):
    from experiments.models import Run

    exp = _experiment()
    _post(client, exp, trial_timeout_mode="fixed", trial_timeout_seconds="120")

    assert Run.objects.get().trial_timeout == {
        "mode": "fixed", "seconds": 120.0, "factor": 5.0}


@pytest.mark.django_db
def test_a_predicted_deadline_reaches_the_run_with_its_multiplier(client, no_thread):
    from experiments.models import Run

    exp = _experiment()
    _post(client, exp, trial_timeout_mode="predicted",
          trial_timeout_seconds="600", trial_timeout_factor="3")

    assert Run.objects.get().trial_timeout == {
        "mode": "predicted", "seconds": 600.0, "factor": 3.0}


@pytest.mark.django_db
def test_the_deadline_is_not_stored_among_the_stopping_criteria(client, no_thread):
    """The deadline ends a trial, and the collector filters keys outside
    `STOPPING_CRITERIA`."""
    from experiments.models import Run

    exp = _experiment()
    _post(client, exp, trial_timeout_mode="fixed", trial_timeout_seconds="120")

    run = Run.objects.get()
    assert run.stopping == {"max_trials": 3}
    assert "trial_timeout" not in run.stopping


@pytest.mark.django_db
@override_settings(MODEL_TRIAL_TIMEOUT=90.0)
def test_a_blank_deadline_falls_back_to_the_deployment_default(client, no_thread):
    """An empty box gets the deployment default; no deadline must be asked for
    on purpose."""
    from experiments.models import Run

    exp = _experiment()
    _post(client, exp, trial_timeout_seconds="")

    assert Run.objects.get().trial_timeout["seconds"] == 90.0


@pytest.mark.django_db
def test_zero_seconds_is_stored_and_means_no_limit(client, no_thread):
    from core.modelhost.deadline import as_deadline
    from core.modelhost.deadline import NO_LIMIT
    from experiments.models import Run

    exp = _experiment()
    _post(client, exp, trial_timeout_seconds="0")

    stored = Run.objects.get().trial_timeout
    assert stored["seconds"] == 0.0
    assert as_deadline(stored).seconds_for({}) == NO_LIMIT


@pytest.mark.django_db
@override_settings(MODEL_TRIAL_TIMEOUT=90.0)
def test_an_unreadable_deadline_costs_the_field_not_the_run(client, no_thread):
    from experiments.models import Run

    exp = _experiment()
    resp = _post(client, exp, trial_timeout_seconds="soon")

    assert resp.status_code == 302
    assert Run.objects.get().trial_timeout["seconds"] == 90.0


@pytest.mark.django_db
def test_an_unknown_mode_falls_back_to_fixed(client, no_thread):
    from experiments.models import Run

    exp = _experiment()
    _post(client, exp, trial_timeout_mode="whatever")

    assert Run.objects.get().trial_timeout["mode"] == "fixed"


@pytest.mark.django_db
def test_the_deadline_survives_the_metric_change_confirmation(client, no_thread):
    from experiments.models import Run

    exp = _experiment(primary_metric="accuracy", original_metric="accuracy")
    body = client.post(reverse(RUN_URL, args=[exp.pk]),
                       {"max_trials": 3, "optimize_metric": "f1",
                        "trial_timeout_mode": "predicted",
                        "trial_timeout_seconds": "300",
                        "trial_timeout_factor": "4"}).content.decode()

    assert 'name="trial_timeout_mode" value="predicted"' in body

    client.post(reverse(RUN_URL, args=[exp.pk]),
                {"max_trials": 3, "optimize_metric": "f1", "decision": "new",
                 "trial_timeout_mode": "predicted",
                 "trial_timeout_seconds": "300",
                 "trial_timeout_factor": "4"})

    stored = Run.objects.latest("id").trial_timeout
    assert stored["mode"] == "predicted"
    assert stored["seconds"] == 300.0
    assert stored["factor"] == 4.0


# Row to session

def test_the_stored_spec_becomes_the_policy_the_session_is_given():
    from core.modelhost.deadline import FixedDeadline, PredictedDeadline
    from experiments.services import modelenv

    fixed = modelenv.session_kwargs({"mode": "fixed", "seconds": 45.0})["trial_timeout"]
    assert isinstance(fixed, FixedDeadline)
    assert fixed.seconds_for({}) == 45.0

    predicted = modelenv.session_kwargs(
        {"mode": "predicted", "seconds": 300.0, "factor": 4.0})["trial_timeout"]
    assert isinstance(predicted, PredictedDeadline)
    assert predicted.factor == 4.0
    assert predicted.ceiling == 300.0


@override_settings(MODEL_TRIAL_TIMEOUT=77.0)
def test_a_run_from_before_the_field_gets_the_deployment_number():
    """Rows that predate the column hold `{}` and get `MODEL_TRIAL_TIMEOUT`."""
    from experiments.services import modelenv

    assert modelenv.session_kwargs({})["trial_timeout"].seconds_for({}) == 77.0
    assert modelenv.session_kwargs(None)["trial_timeout"].seconds_for({}) == 77.0


def test_the_predicted_policy_is_seeded_from_the_run():
    """The predicted policy fits a forest, one more stochastic part of the
    experiment."""
    from experiments.services import modelenv

    policy = modelenv.session_kwargs(
        {"mode": "predicted", "seconds": 300.0}, seed=17)["trial_timeout"]
    assert policy.seed == 17


# Where it cannot apply

@pytest.mark.django_db
def test_the_fields_are_absent_for_a_model_that_runs_in_this_process(client):
    """An in-process model gets no deadline fields, since nothing can interrupt
    its `fit_predict`."""
    from experiments.models import Experiment

    exp = _experiment()
    exp.env_status = Experiment.ENV_SKIPPED
    exp.save(update_fields=["env_status"])

    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert exp.env_in_process
    assert 'name="trial_timeout_mode"' not in body
    assert 'name="max_failures"' in body, "the other limits are still offered"
