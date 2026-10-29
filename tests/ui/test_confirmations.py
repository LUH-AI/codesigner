"""Every confirmation extends `ui/_confirm_page.html`, stating the consequence
and offering Cancel.
"""

from django.urls import reverse

from experiments.models import Experiment
from tests.conftest import DATASETS_DIR


def _exp(**overrides):
    fields = dict(
        name="c", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy", "f1"], current_metric="accuracy",
        original_metric="accuracy", seed=0)
    fields.update(overrides)
    return Experiment.objects.create(**fields)


def _runnable():
    """The dataset lets the metric check run."""
    from experiments.services import snapshot as adapter
    return adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "c", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": "accuracy", "original_metric": "accuracy",
        "metric_names": ["accuracy", "f1"], "seed": 0,
        "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }, adopt_paths=True)


def _confirmations(client):
    """`_confirmations` gathers confirmation pages."""
    exp = _exp()
    runnable = _runnable()
    return [
        ("delete",
         client.get(reverse("ui:experiment_delete", args=[exp.pk])).content.decode()),
        ("metric change",
         client.post(reverse("ui:experiment_run", args=[runnable.pk]),
                     {"max_trials": 1, "optimize_metric": "f1"}).content.decode()),
        ("save as default",
         client.post(reverse("ui:experiment_settings", args=[exp.pk]),
                     {"save_as_default": "1"}).content.decode()),
    ]


def test_every_confirmation_uses_the_shared_page(client):
    """A warning states the consequence and one form contains the actions."""
    for name, html in _confirmations(client):
        assert "confirm-message" in html, name
        assert "confirm-actions" in html, name


def test_every_confirmation_offers_a_way_out(client):
    for name, html in _confirmations(client):
        assert "Cancel" in html, name
