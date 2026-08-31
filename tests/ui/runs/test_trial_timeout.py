"""The deadline a run gives each of its trials, from the form to the session.

`docs/plan_today.md` Step 2. The number itself is enforced in
`core.modelhost` and tested there (tests/core/test_trial_deadline.py); what is
under test here is that a choice made on the run form survives the trip — the
Run row, the metric-change confirmation, and the kwargs `model_session` is
built with — and that it is not offered where it cannot be honoured.
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


# ── the form reaches the row ─────────────────────────────────────────────────

@pytest.mark.django_db
def test_a_fixed_deadline_reaches_the_run(client, no_thread):
    from ui.models import Run

    exp = _experiment()
    _post(client, exp, trial_timeout_mode="fixed", trial_timeout_seconds="120")

    assert Run.objects.get().trial_timeout == {
        "mode": "fixed", "seconds": 120.0, "factor": 5.0}


@pytest.mark.django_db
def test_a_predicted_deadline_reaches_the_run_with_its_multiplier(client, no_thread):
    from ui.models import Run

    exp = _experiment()
    _post(client, exp, trial_timeout_mode="predicted",
          trial_timeout_seconds="600", trial_timeout_factor="3")

    assert Run.objects.get().trial_timeout == {
        "mode": "predicted", "seconds": 600.0, "factor": 3.0}


@pytest.mark.django_db
def test_the_deadline_is_not_stored_among_the_stopping_criteria(client, no_thread):
    """It ends a trial, not a run. Stored among the criteria the collector would
    filter it out — it is not in STOPPING_CRITERIA — and a reader of the row
    would have been told the run stops on something it does not."""
    from ui.models import Run

    exp = _experiment()
    _post(client, exp, trial_timeout_mode="fixed", trial_timeout_seconds="120")

    run = Run.objects.get()
    assert run.stopping == {"max_trials": 3}
    assert "trial_timeout" not in run.stopping


@pytest.mark.django_db
@override_settings(MODEL_TRIAL_TIMEOUT=90.0)
def test_a_blank_deadline_falls_back_to_the_deployment_default(client, no_thread):
    """Unlike a stopping criterion, an empty box here is not "does not apply".
    No deadline at all is a thing to ask for deliberately — see the zero test —
    not a thing to arrive at by leaving a field alone."""
    from ui.models import Run

    exp = _experiment()
    _post(client, exp, trial_timeout_seconds="")

    assert Run.objects.get().trial_timeout["seconds"] == 90.0


@pytest.mark.django_db
def test_zero_seconds_is_stored_and_means_no_limit(client, no_thread):
    from core.modelhost import as_deadline
    from core.modelhost.deadline import NO_LIMIT
    from ui.models import Run

    exp = _experiment()
    _post(client, exp, trial_timeout_seconds="0")

    stored = Run.objects.get().trial_timeout
    assert stored["seconds"] == 0.0
    assert as_deadline(stored).seconds_for({}) == NO_LIMIT


@pytest.mark.django_db
@override_settings(MODEL_TRIAL_TIMEOUT=90.0)
def test_an_unreadable_deadline_costs_the_field_not_the_run(client, no_thread):
    from ui.models import Run

    exp = _experiment()
    resp = _post(client, exp, trial_timeout_seconds="soon")

    assert resp.status_code == 302
    assert Run.objects.get().trial_timeout["seconds"] == 90.0


@pytest.mark.django_db
def test_an_unknown_mode_falls_back_to_fixed(client, no_thread):
    from ui.models import Run

    exp = _experiment()
    _post(client, exp, trial_timeout_mode="whatever")

    assert Run.objects.get().trial_timeout["mode"] == "fixed"


@pytest.mark.django_db
def test_the_deadline_survives_the_metric_change_confirmation(client, no_thread):
    """The confirmation reposts the form, so anything it does not carry through
    is silently dropped — and a run would then start under a deadline nobody
    chose."""
    from ui.models import Run

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


# ── the row reaches the session ──────────────────────────────────────────────

def test_the_stored_spec_becomes_the_policy_the_session_is_given():
    from core.modelhost.deadline import FixedDeadline, PredictedDeadline
    from ui.services import modelenv

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
    """`trial_timeout` defaults to `{}` on every row that predates the column,
    and those runs behaved as `MODEL_TRIAL_TIMEOUT` said."""
    from ui.services import modelenv

    assert modelenv.session_kwargs({})["trial_timeout"].seconds_for({}) == 77.0
    assert modelenv.session_kwargs(None)["trial_timeout"].seconds_for({}) == 77.0


def test_the_predicted_policy_is_seeded_from_the_run():
    """The forest is fitted here, so it is one more stochastic part of an
    experiment that has a seed for exactly this reason."""
    from ui.services import modelenv

    policy = modelenv.session_kwargs(
        {"mode": "predicted", "seconds": 300.0}, seed=17)["trial_timeout"]
    assert policy.seed == 17


# ── not offered where it cannot be honoured ──────────────────────────────────

@pytest.mark.django_db
def test_the_fields_are_absent_for_a_model_that_runs_in_this_process(client):
    """Nothing can interrupt a `fit_predict` running in the run's own thread, so
    an in-process model gets no deadline fields rather than fields that would be
    quietly ignored. See test_trial_deadline.py for the limitation itself."""
    from ui.models import Experiment

    exp = _experiment()
    exp.env_status = Experiment.ENV_SKIPPED
    exp.save(update_fields=["env_status"])

    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert exp.env_in_process
    assert 'name="trial_timeout_mode"' not in body
    assert 'name="max_failures"' in body, "the other limits are still offered"
