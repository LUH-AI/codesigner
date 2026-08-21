"""GUI polish: the merged Run/Evaluation-metric box and the metric switcher.

The Run form and the Evaluation-metric selector live in one card. Choosing a
metric in the dropdown applies immediately — the figures re-render on the
select's change event, with no separate button to press. A read-only
experiment (has a result but no dataset, so it can't run) still needs its own
metric selector, since there's no Run form to attach it to; so does an
experiment with a run in flight, whose results stay browsable while it runs.
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
    from ui.services import snapshot as adapter

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
    exp.current_metric = "accuracy"
    exp.original_metric = "accuracy"
    exp.save()
    return exp


def _readonly_experiment_with_result():
    """An experiment with a stored result but no dataset (can't run) —
    the fixture's dataset_path is a foreign machine path that doesn't exist here."""
    from ui.services import snapshot as adapter
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


@pytest.mark.django_db
def test_no_reevaluate_button_anywhere(client):
    """The Reevaluate button is gone — applying a metric is the dropdown's job."""
    exp = _experiment_with_result()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert "reevaluate" not in html.lower()


@pytest.mark.django_db
def test_metric_select_present_before_first_run(client):
    """A fresh, never-run experiment shows the Run form with its metric dropdown."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert 'name="max_trials"' in html
    assert 'name="optimize_metric"' in html


@pytest.mark.django_db
def test_run_and_metric_controls_share_one_form_in_the_sidebar(client):
    """The metric dropdown and what bounds the next run are one form, in the
    sidebar's Experiment Evaluation section.

    One form because the dropdown *is* the run's optimize-metric field; in the
    sidebar because the metric drives every figure on the page and a selector
    that scrolls away is one you have to go and find. Only what bounds the run
    folds away beneath it.
    """
    exp = _experiment_with_result()

    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    sidebar = html.split('<nav class="sidebar"', 1)[1].split("</nav>", 1)[0]
    content = html.split("<main", 1)[-1]

    assert "Experiment Evaluation" in sidebar
    assert sidebar.count("<form") == 1
    assert 'name="max_trials"' in sidebar
    assert 'id="metric-select"' in sidebar
    # the dropdown doubles as the Run form's optimize-metric field
    assert 'name="optimize_metric"' in sidebar
    # and it is not left behind in the main column as well
    assert 'id="metric-select"' not in content
    assert "<details class=\"run-config\">" in sidebar


@pytest.mark.django_db
def test_readonly_experiment_keeps_metric_selector_without_run_form(client):
    """A read-only experiment (result but no dataset) still gets a metric
    selector — just no n_trials/Run, since it can't be run."""
    exp = _readonly_experiment_with_result()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="metric-select"' in html
    assert 'name="max_trials"' not in html


@pytest.mark.django_db
def test_metric_selector_present_while_a_run_is_in_flight(client):
    """A run in flight replaces the Run form with the status panel, but the
    metric selector must survive — the figures are driven off it, so without it
    every per-metric figure renders blank until the run finishes."""
    from ui.models import Run

    exp = _experiment_with_result()
    Run.objects.create(experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy", status="running")

    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert 'id="metric-select"' in html
    # the run form itself is gone while running
    assert 'name="max_trials"' not in html


@pytest.mark.django_db
def test_metric_switch_applies_automatically(client):
    """Choosing a metric re-renders the figures on the spot — the dropdown's
    change event calls show(), with no button in between."""
    exp = _readonly_experiment_with_result()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'select.addEventListener("change"' in html


@pytest.mark.django_db
def test_plotly_toolbar_hides_logo_and_select_tools(client):
    """The embedded Plotly config removes the Plotly logo/link and the lasso
    and box-select modebar buttons from every figure."""
    exp = _readonly_experiment_with_result()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert "displaylogo: false" in html
    assert "lasso2d" in html
    assert "select2d" in html
