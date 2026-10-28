"""The detail page launches a background run, polls its status and cancels it.
"""

import pytest
from django.urls import reverse

from tests.conftest import DATASETS_DIR, post_new_experiment


@pytest.fixture
def no_thread(monkeypatch):
    """`no_thread` keeps a launched run 'pending'."""
    from experiments.services import run as run_service
    monkeypatch.setattr(run_service, "start_background_run", lambda run_id: None)


def _create(client, **overrides):
    data = {
        "name": "exp",
        "model_name": "Random Forest",
        "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification",
        "seed": 0,
    }
    data.update(overrides)
    return post_new_experiment(client, data)


def _experiment(**overrides):
    from experiments.services import snapshot as adapter
    snapshot = {
        "version": "0.1.0", "name": "exp", "model_name": "Random Forest",
        "model_path": "", "optimizer_params": {},
        "optimizer_name": overrides.get("optimizer_name", "Random Search"),
        "primary_metric": overrides.get("primary_metric"),
        "original_metric": overrides.get("original_metric"),
        "metric_names": ["accuracy", "f1", "precision", "recall(macro)"],
        "seed": 0, "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }
    return adapter.experiment_from_snapshot(snapshot, adopt_paths=True)


@pytest.mark.django_db
def test_create_persists_without_running(client):
    from experiments.models import Experiment, Run

    resp = _create(client, name="fresh")
    exp = Experiment.objects.get(data__name="fresh")

    assert resp.status_code == 302
    assert resp["Location"] == reverse("ui:experiment_detail", args=[exp.pk])
    assert exp.data.result is None
    assert exp.data.current_metric is None
    assert Run.objects.count() == 0


@pytest.mark.django_db
def test_detail_offers_a_run_form_when_idle(client):
    """The Run form asks for trials and a metric."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert 'name="max_trials"' in html
    assert 'name="optimize_metric"' in html


# Launching a run

@pytest.mark.django_db
def test_run_launches_a_pending_run(client, no_thread):
    from experiments.models import Run

    exp = _experiment()
    resp = client.post(reverse("ui:experiment_run", args=[exp.pk]),
                       {"max_trials": 3, "optimize_metric": "accuracy"})

    assert resp.status_code == 302
    assert resp["Location"] == reverse("ui:experiment_detail", args=[exp.pk])
    run = Run.objects.get(experiment=exp)
    assert run.status == "pending"
    assert run.max_trials == 3
    assert run.primary_metric == "accuracy"


@pytest.mark.django_db
def test_run_metric_change_asks_for_confirmation(client, no_thread):
    """`experiment_run` asks for confirmation naming both metrics and creates no
    run.
    """
    from experiments.models import Run

    exp = _experiment(primary_metric="accuracy", original_metric="accuracy")
    resp = client.post(reverse("ui:experiment_run", args=[exp.pk]),
                       {"max_trials": 3, "optimize_metric": "f1"})

    assert resp.status_code == 200
    body = resp.content.decode()
    assert "f1" in body and "accuracy" in body
    assert Run.objects.count() == 0


@pytest.mark.django_db
def test_metric_change_warning_states_the_real_cost(client, no_thread):
    """`experiment_run` warns of selection bias, naming each metric once."""
    exp = _experiment(primary_metric="accuracy", original_metric="accuracy")
    body = client.post(reverse("ui:experiment_run", args=[exp.pk]),
                       {"max_trials": 3, "optimize_metric": "f1"}).content.decode()
    assert "confirm-message" in body, "not the shared confirmation page"
    warning = body.split("confirm-message", 1)[1].split("</div>", 1)[0]

    assert "bias" in warning.lower()
    assert warning.count("accuracy") == 1
    assert warning.count("f1") == 1


@pytest.mark.django_db
def test_run_metric_change_confirm_new_launches_with_chosen(client, no_thread):
    from experiments.models import Run

    exp = _experiment(primary_metric="accuracy", original_metric="accuracy")
    resp = client.post(reverse("ui:experiment_run", args=[exp.pk]),
                       {"max_trials": 3, "optimize_metric": "f1", "decision": "new"})

    assert resp.status_code == 302
    run = Run.objects.get(experiment=exp)
    assert run.primary_metric == "f1"
    exp.refresh_from_db()
    assert exp.data.current_metric == "f1"
    assert exp.data.original_metric == "accuracy"


@pytest.mark.django_db
def test_run_metric_change_confirm_old_keeps_original(client, no_thread):
    from experiments.models import Run

    exp = _experiment(primary_metric="accuracy", original_metric="accuracy")
    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 3, "optimize_metric": "f1", "decision": "old"})

    run = Run.objects.get(experiment=exp)
    assert run.primary_metric == "accuracy"


# Status polling

@pytest.mark.django_db
def test_status_partial_reports_running(client):
    """The status endpoint keeps polling and offers Cancel."""
    from experiments.models import Run

    exp = _experiment()
    Run.objects.create(experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy", status="running")
    resp = client.get(reverse("ui:run_status", args=[exp.pk]))

    assert resp.status_code == 200
    body = resp.content.decode()
    assert "unning" in body
    assert "hx-trigger" in body
    assert reverse("ui:run_cancel", args=[exp.pk]) in body


@pytest.mark.django_db
def test_status_partial_refreshes_when_finished(client):
    """An idle status endpoint sends `HX-Refresh`."""
    from experiments.models import Run

    exp = _experiment()
    Run.objects.create(experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy", status="done")
    resp = client.get(reverse("ui:run_status", args=[exp.pk]))

    assert resp.status_code == 200
    assert resp.get("HX-Refresh") == "true"


# Cancel

@pytest.mark.django_db
def test_cancel_requests_cancellation(client):
    """The worker stops soon after the flag."""
    from experiments.models import Run

    exp = _experiment()
    run = Run.objects.create(experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy", status="running")
    resp = client.post(reverse("ui:run_cancel", args=[exp.pk]))

    run.refresh_from_db()
    assert run.cancel_requested is True
    assert resp.status_code in (302, 200)


@pytest.mark.django_db
def test_delete_works_with_an_active_run(client):
    """Deleting mid-run asks the run to stop and bins the experiment with its
    runs."""
    from experiments.models import Run

    exp = _experiment()
    run = Run.objects.create(experiment=exp, stopping={"max_trials": 3},
                             primary_metric="accuracy", status="running")
    resp = client.post(reverse("ui:experiment_delete", args=[exp.pk]))

    exp.refresh_from_db()
    run.refresh_from_db()
    assert resp.status_code == 302
    assert exp.deleted_at is not None
    assert run.cancel_requested


# Sidebar spinner

@pytest.mark.django_db
def test_sidebar_marks_running_experiments(client):
    from experiments.models import Run

    exp = _experiment(primary_metric="accuracy", original_metric="accuracy")
    Run.objects.create(experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy", status="running")
    html = client.get(reverse("ui:home")).content.decode()

    assert "spinner" in html.lower()


@pytest.mark.django_db
def test_stopping_criteria_reach_the_run(client, no_thread):
    """The optimizer reads the criteria from the Run."""
    from experiments.models import Run

    exp = _experiment()
    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 3, "optimize_metric": "accuracy",
                 "target_score": "0.9", "no_improvement_trials": "5"})

    run = Run.objects.get()
    assert run.stopping == {"max_trials": 3, "target_score": 0.9,
                            "no_improvement_trials": 5}


@pytest.mark.django_db
def test_blank_criteria_are_absent_rather_than_zero(client, no_thread):
    """A criterion stored as 0 would stop the run at once."""
    from experiments.models import Run

    exp = _experiment()
    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 3, "optimize_metric": "accuracy",
                 "target_score": "", "max_seconds": "", "no_improvement_trials": ""})

    assert Run.objects.get().stopping == {"max_trials": 3}


@pytest.mark.django_db
def test_an_unparseable_criterion_costs_the_criterion_not_the_run(client, no_thread):
    """A typo costs only its criterion, and the trial cap still bounds the run."""
    from experiments.models import Run

    exp = _experiment()
    resp = client.post(reverse("ui:experiment_run", args=[exp.pk]),
                       {"max_trials": 3, "optimize_metric": "accuracy",
                        "max_seconds": "soon"})

    assert resp.status_code == 302
    assert Run.objects.get().stopping == {"max_trials": 3}


@pytest.mark.django_db
def test_criteria_survive_the_metric_change_confirmation(client, no_thread):
    """The confirmation reposts the form, dropping anything it omits."""
    from experiments.models import Run

    exp = _experiment(primary_metric="accuracy", original_metric="accuracy")
    body = client.post(reverse("ui:experiment_run", args=[exp.pk]),
                       {"max_trials": 3, "optimize_metric": "f1",
                        "target_score": "0.9"}).content.decode()

    assert 'name="target_score" value="0.9"' in body

    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 3, "optimize_metric": "f1", "decision": "new",
                 "target_score": "0.9"})

    assert Run.objects.latest("id").stopping == {"max_trials": 3, "target_score": 0.9}


@pytest.mark.django_db
def test_a_run_with_no_criteria_is_refused_with_a_reason(client, no_thread):
    """A run with no way to end never starts, and the page says why."""
    from experiments.models import Run

    exp = _experiment()
    resp = client.post(reverse("ui:experiment_run", args=[exp.pk]),
                       {"optimize_metric": "accuracy", "max_trials": "",
                        "target_score": "", "max_seconds": ""})

    assert resp.status_code == 200
    assert "at least one stopping criterion" in resp.content.decode()
    assert Run.objects.count() == 0


@pytest.mark.django_db
def test_any_single_criterion_is_enough_to_start_a_run(client, no_thread):
    """The trial cap has no special status."""
    from experiments.models import Run

    exp = _experiment()
    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"optimize_metric": "accuracy", "target_score": "0.9"})

    assert Run.objects.get().stopping == {"target_score": 0.9}


@pytest.mark.django_db
def test_the_confidence_criterion_is_offered_only_where_it_can_be_answered(client):
    """Grid search has no surrogate for the confidence criterion to read."""
    smac = _experiment(optimizer_name="SMAC")
    grid = _experiment(optimizer_name="Grid Search")

    def page(exp):
        return client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'name="incumbent_confidence"' in page(smac)
    assert 'name="incumbent_confidence"' not in page(grid)

    # Experiments created before the strategy setting still carry the old name.
    assert 'name="incumbent_confidence"' in page(_experiment(optimizer_name="SMAC (BlackBox)"))


@pytest.mark.django_db
def test_an_interrupted_run_says_it_was_interrupted(client):
    """A manual stop is reported through `stopped_by`, like the criteria."""
    from django.utils import timezone

    from experiments.models import Run

    exp = _experiment()
    Run.objects.create(experiment=exp, stopping={"max_trials": 30},
                       primary_metric="accuracy", status="cancelled",
                       stopped_by="cancelled", trial_count=4, trial_seconds=1.0,
                       started_at=timezone.now(), finished_at=timezone.now())

    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert "Stopped because it was interrupted" in html
    assert "requested trials ran" not in html


# Giving up on a run

def _running(exp, **kw):
    from experiments.models import Run
    fields = dict(experiment=exp, stopping={"max_trials": 3},
                  primary_metric="accuracy", status="running")
    fields.update(kw)
    return Run.objects.create(**fields)


@pytest.mark.django_db
def test_cancelling_comes_first_and_giving_up_only_after(client):
    """Giving up appears only after a cancel fails, since cancelling keeps
    finished trials."""
    exp = _experiment()
    run = _running(exp)

    before = client.get(reverse("ui:run_status", args=[exp.pk])).content.decode()
    run.cancel_requested = True
    run.save(update_fields=["cancel_requested"])
    after = client.get(reverse("ui:run_status", args=[exp.pk])).content.decode()

    assert reverse("ui:run_cancel", args=[exp.pk]) in before
    assert reverse("ui:run_force_stop", args=[exp.pk]) not in before
    assert reverse("ui:run_force_stop", args=[exp.pk]) in after
    assert "Cancelling" in after


@pytest.mark.django_db
def test_giving_up_finishes_the_row_and_says_why(client):
    """The row records a give-up, since the web process cannot kill the worker."""
    exp = _experiment()
    run = _running(exp, cancel_requested=True)

    client.post(reverse("ui:run_force_stop", args=[exp.pk]))
    run.refresh_from_db()

    assert run.status == "error"
    assert run.finished_at is not None
    assert "gave up" in run.error.lower() or "given up" in run.error.lower()


@pytest.mark.django_db
def test_a_run_that_was_never_asked_to_stop_is_not_given_up_on(client):
    """A direct POST still requires a cancel request first."""
    exp = _experiment()
    run = _running(exp)

    client.post(reverse("ui:run_force_stop", args=[exp.pk]))
    run.refresh_from_db()

    assert run.status == "running"


@pytest.mark.django_db
def test_giving_up_is_post_only(client):
    """A GET from a prefetcher or an `<img>` would end a run without CSRF."""
    exp = _experiment()
    _running(exp, cancel_requested=True)

    assert client.get(reverse("ui:run_force_stop", args=[exp.pk])).status_code == 405
