"""GUI polish: the merged Run/Evaluation-metric box and the Reevaluate button.

The Run form and the Evaluation-metric selector were combined into one card
(user request), and metric switching was reverted to Streamlit's original
two-step flow: choosing a metric in the dropdown does not, by itself, change
what's displayed — you press Reevaluate to apply it. A read-only experiment
(has a result but no dataset, so it can't run) still needs its own metric
selector + Reevaluate, since there's no Run form to attach it to.
"""

from django.urls import reverse

from core import io
from tests.conftest import DATASETS_DIR, FIXTURES_DIR


def _experiment(**overrides):
    from web.services import snapshot as adapter

    fields = dict(
        version="0.1.0", name="reeval-exp", model_name="Random Forest",
        model_path="", optimizer_name="Random Search", optimizer_params={},
        primary_metric=None, original_metric=None,
        metric_names=["accuracy", "f1", "precision", "recall(macro)"],
        seed=0, dataset_path=str(DATASETS_DIR / "iris.csv"), result=None,
    )
    fields.update(overrides)
    return adapter.experiment_from_snapshot(fields)


def _readonly_experiment_with_result():
    """An experiment with a stored result but no dataset (can't run) —
    the fixture's dataset_path is a foreign machine path that doesn't exist here."""
    from web.services import snapshot as adapter
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


def test_reevaluate_absent_before_first_run(client):
    """A fresh, never-run experiment shows the Run form but no Reevaluate
    button — there is nothing to reevaluate yet."""
    exp = _experiment()
    html = client.get(reverse("web:experiment_detail", args=[exp.pk])).content.decode()
    assert 'name="n_trials"' in html
    assert 'name="optimize_metric"' in html
    assert 'id="reevaluate-btn"' not in html


def test_run_and_metric_controls_share_one_card(client):
    """Once a result exists, the Run form and the metric dropdown + Reevaluate
    button live in the same card (one 'Run' subheader), not two separate boxes."""
    exp = _experiment(result={
        "stats": {"submitted": 1, "finished": 1, "running": 0},
        "data": [{"config_id": 1, "cost": 0.2, "scores": {
            "accuracy": 0.8, "f1": 0.7, "precision": 0.75, "recall(macro)": 0.72,
        }, "incumbent_score": 0.8, "incumbent_config_id": 1}],
        "configs": {"1": {"n_estimators": 100}}, "config_origins": {"1": "Random Search"},
        "optimizer_state": {},
        "primary_metric": "accuracy", "best_score": 0.8, "best_config_id": "1",
        "hyperparameter_importance": {}, "hyperparameter_importance_warning": {},
        "trials_limit": None,
    })
    exp.primary_metric = "accuracy"
    exp.original_metric = "accuracy"
    exp.save()

    html = client.get(reverse("web:experiment_detail", args=[exp.pk])).content.decode()
    assert html.count("<form") == 1
    assert 'name="n_trials"' in html
    assert 'id="metric-select"' in html
    assert 'id="reevaluate-btn"' in html
    # the dropdown and button are wired together (name=optimize_metric shared)
    assert 'name="optimize_metric"' in html


def test_readonly_experiment_keeps_metric_selector_without_run_form(client):
    """A read-only experiment (result but no dataset) still gets a metric
    selector and Reevaluate — just no n_trials/Run, since it can't be run."""
    exp = _readonly_experiment_with_result()
    html = client.get(reverse("web:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="metric-select"' in html
    assert 'id="reevaluate-btn"' in html
    assert 'name="n_trials"' not in html


def test_metric_switch_requires_reevaluate_not_automatic(client):
    """The dropdown's change event is not wired to re-render charts on its own
    — only the Reevaluate button's click handler calls show()."""
    exp = _readonly_experiment_with_result()
    html = client.get(reverse("web:experiment_detail", args=[exp.pk])).content.decode()

    assert 'select.addEventListener("change"' not in html
    assert "reevaluateBtn.addEventListener(\"click\"" in html


def test_plotly_toolbar_hides_logo_and_select_tools(client):
    """The embedded Plotly config removes the Plotly logo/link and the lasso
    and box-select modebar buttons from every chart."""
    exp = _readonly_experiment_with_result()
    html = client.get(reverse("web:experiment_detail", args=[exp.pk])).content.decode()

    assert "displaylogo: false" in html
    assert "lasso2d" in html
    assert "select2d" in html
