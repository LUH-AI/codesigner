"""The Run form and the metric selector share one sidebar card, and choosing a
metric re-renders the figures at once. A read-only or running experiment keeps
the selector without the Run form.
"""

import pytest
from django.urls import reverse

from core import io
from tests.conftest import DATASETS_DIR, FIXTURES_DIR

_RESULT = {
    "stats": {"submitted": 1, "finished": 1, "running": 0},
    "data": [{"config_id": 1, "cost": 0.2, "scores": {
        "accuracy": 0.8, "f1": 0.7, "precision": 0.75, "recall(macro)": 0.72,
    }, "incumbent_score": 0.8, "incumbent_config_id": 1}],
    "configs": {"1": {"n_estimators": 100}}, "config_origins": {"1": "Random Search"},
    "optimizer_state": {},
    "primary_metric": "accuracy", "best_score": 0.8, "best_config_id": "1",
    "hyperparameter_importance": {}, "hyperparameter_importance_warning": {},
    "trials_limit": None,
}


def _experiment(**overrides):
    from experiments.services import snapshot as adapter

    fields = dict(
        version="0.1.0", name="reeval-exp", model_name="Random Forest",
        model_path="", optimizer_name="Random Search", optimizer_params={},
        primary_metric=None, original_metric=None,
        metric_names=["accuracy", "f1", "precision", "recall(macro)"],
        seed=0, dataset_path=str(DATASETS_DIR / "iris.csv"), result=None,
    )
    fields.update(overrides)
    return adapter.experiment_from_snapshot(fields, adopt_paths=True)


def _experiment_with_result():
    exp = _experiment(result=_RESULT)
    exp.data.current_metric = "accuracy"
    exp.data.original_metric = "accuracy"
    exp.data.save()
    return exp


def _readonly_experiment_with_result():
    """`_readonly_experiment_with_result` has a result but a foreign
    dataset_path."""
    from experiments.services import snapshot as adapter
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


@pytest.mark.django_db
def test_no_reevaluate_button_anywhere(client):
    exp = _experiment_with_result()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert "reevaluate" not in html.lower()


@pytest.mark.django_db
def test_metric_select_present_before_first_run(client):
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert 'name="max_trials"' in html
    assert 'name="optimize_metric"' in html


@pytest.mark.django_db
def test_run_and_metric_controls_share_one_form_in_the_evaluation_card(client):
    """What: the metric dropdown is the run's optimize-metric field, in one
    form with the stopping conditions in the Experiment Evaluation card, and
    the sidebar no longer has them. How: the page of an experiment with a
    result."""
    exp = _experiment_with_result()

    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    sidebar = html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]
    card = html.split('data-figure="evaluation"', 1)[1].split("</section>", 1)[0]

    assert card.count("<form") == 1
    assert 'name="max_trials"' in card and 'id="metric-select"' in card
    # dropdown doubles as optimize-metric field
    assert 'name="optimize_metric"' in card
    assert 'id="metric-select"' not in sidebar and "Experiment Evaluation" not in sidebar
    assert "run-config" not in card


@pytest.mark.django_db
def test_readonly_experiment_keeps_metric_selector_without_run_form(client):
    """A read-only experiment keeps the selector and drops the Run form."""
    exp = _readonly_experiment_with_result()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="metric-select"' in html
    assert 'name="max_trials"' not in html


@pytest.mark.django_db
def test_metric_selector_present_while_a_run_is_in_flight(client):
    """The selector must survive while a run replaces the Run form, because the
    figures read it."""
    from experiments.models import Run

    exp = _experiment_with_result()
    Run.objects.create(experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy", status="running")

    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert 'id="metric-select"' in html
    # run form gone while running
    assert 'name="max_trials"' not in html


@pytest.mark.django_db
def test_metric_switch_applies_automatically(client):
    """The dropdown's change event calls `show()` directly."""
    exp = _readonly_experiment_with_result()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'select.addEventListener("change"' in html


@pytest.mark.django_db
def test_plotly_toolbar_hides_logo_and_select_tools(client):
    exp = _readonly_experiment_with_result()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert "displaylogo: false" in html
    assert "lasso2d" in html
    assert "select2d" in html
