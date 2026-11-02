"""Between runs, the settings a later run may change (`run_safe`) sit in the
run form's "Next run" fold and are saved when Run is pressed; every other
optimizer setting stays as the first run had it. A planned trial budget fills
the run form with what is left of it.
"""

import pytest
from django.urls import reverse

from core import io
from experiments.models import Run
from experiments.services import snapshot
from tests.conftest import FIXTURES_DIR

pytestmark = pytest.mark.django_db


@pytest.fixture
def no_thread(monkeypatch):
    """`no_thread` keeps a launched run 'pending'."""
    from experiments.services import run as run_service
    monkeypatch.setattr(run_service, "start_background_run", lambda run_id: None)


@pytest.fixture
def guided():
    """SMAC on iris with 40 trials over two runs, trial cap 10."""
    return snapshot.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "smac_iris_guided.ihpo").read_bytes()))


def _request():
    from django.contrib.auth.models import AnonymousUser
    from django.test import RequestFactory

    request = RequestFactory().post("/")
    request.user = AnonymousUser()
    return request


def _run_form(client, exp):
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    start = page.index('action="' + reverse("ui:experiment_run", args=[exp.pk]))
    return page[start:page.index("</form>", start)]


def test_only_run_safe_settings_change_once_there_are_trials(guided):
    """What: with trials, a run-safe setting saves, a locked one is refused by
    name, and the optimizer cannot be swapped.
    How: `optimizer_form.save` on the guided fixture with each."""
    from ui import optimizer_form

    assert optimizer_form.save(_request(), guided, {"opt_random_probability": 1.0},
                               partial=True) is None
    guided.refresh_from_db()
    assert guided.data.optimizer_params["random_probability"] == 1.0
    assert guided.data.optimizer_params["trial_cap"] == 10, "the others keep their values"

    refused = optimizer_form.save(_request(), guided, {"opt_initial_design": "random"},
                                  partial=True)
    assert "stay as the first run had them: Sampling method" in refused
    assert "has trials" in optimizer_form.save(_request(), guided,
                                               {"optimizer_name": "Random Search"}, partial=True)


def test_nothing_changes_while_a_run_is_going(guided):
    """What: during a run even run-safe settings are refused.
    How: a pending run on the guided fixture."""
    from ui import optimizer_form

    Run.objects.create(experiment=guided, primary_metric="accuracy", status="running")

    assert "under way" in optimizer_form.save(_request(), guided,
                                              {"opt_random_probability": 1.0}, partial=True)
    assert optimizer_form.why_fixed(guided, "random_probability") is not None


def test_the_next_run_fold_offers_the_run_safe_settings(client, guided):
    """What: the dashboard's run form carries the run-safe settings, and none
    of the locked ones.
    How: the guided fixture's page."""
    form = _run_form(client, guided)

    assert 'name="next_run_settings"' in form
    assert 'id="opt_random_probability"' in form and 'id="opt_trial_cap"' in form
    assert 'id="opt_initial_design"' not in form and 'id="opt_search_strategy"' not in form


def test_run_saves_the_changed_rate_and_the_run_records_it(client, guided, no_thread):
    """What: pressing Run with a changed rate saves it, the run copies it, the
    run's optimizer is built with it, and an .ihpo round trip keeps it per run.
    How: a posted run form on the guided fixture, then an export re-imported."""
    response = client.post(reverse("ui:experiment_run", args=[guided.pk]), {
        "optimize_metric": "accuracy", "max_trials": "5", "max_failures": "1",
        "next_run_settings": "1", "opt_random_probability": "1.0", "opt_use_trial_cap": "on",
        "opt_trial_cap": "10"})
    assert response.status_code == 302
    guided.refresh_from_db()
    run = guided.runs.order_by("pk").last()

    assert run.optimizer_params["random_probability"] == 1.0
    built = io.build_experiment(snapshot.snapshot_from_experiment(guided), *_registries(),
                                read_only=True)[1]
    assert built["optimizer"]._random_probability == 1.0

    again = snapshot.experiment_from_snapshot(io.parse(io.to_bytes(
        snapshot.snapshot_from_experiment(guided, provenance=True))))
    assert again.runs.order_by("pk").last().optimizer_params["random_probability"] == 1.0


def _registries():
    from core import registry

    return registry.METRICS, registry.MODELS, registry.OPTIMIZERS


def test_a_planned_budget_fills_the_run_form_with_what_is_left(client, guided):
    """What: with a budget of 150 and 40 trials done, the run form offers 110
    trials and no target; spent, it offers none and says so; without a budget
    it offers 30 and the target to beat.
    How: the guided fixture's run form under each."""
    plain = _run_form(client, guided)
    assert 'name="max_trials"\n                               value="30"' in plain
    assert 'id="target_score" name="target_score"\n                               value=""' \
        not in plain

    guided.data.trial_budget = 150
    guided.data.save(update_fields=["trial_budget"])
    form = _run_form(client, guided)
    assert 'value="110"' in form and "What is left of the 150 trials planned." in form
    assert 'id="target_score" name="target_score"\n                               value=""' in form

    guided.data.trial_budget = 40
    guided.data.save(update_fields=["trial_budget"])
    assert "The planned 40 trials have run." in _run_form(client, guided)


def test_the_budget_survives_step_one_and_the_file():
    """What: a draft's budget outlives step 1 replacing its data, and an .ihpo
    round trip keeps it.
    How: `replace_data` with fresh data, then an export re-imported."""
    from experiments.models import ExperimentData

    exp = snapshot.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "smac_iris_guided.ihpo").read_bytes()))
    exp.data.trial_budget = 150
    exp.data.save(update_fields=["trial_budget"])
    fresh = snapshot.data_from_snapshot(snapshot.snapshot_from_experiment(exp))
    fresh.trial_budget = None
    snapshot.replace_data(exp, fresh)
    exp.refresh_from_db()

    assert exp.data.trial_budget == 150
    again = snapshot.experiment_from_snapshot(io.parse(io.to_bytes(
        snapshot.snapshot_from_experiment(exp))))
    assert again.data.trial_budget == 150
    assert ExperimentData.objects.filter(trial_budget=150).count() == 2
