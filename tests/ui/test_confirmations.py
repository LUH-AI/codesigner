"""Every "are you sure" on the site is the same page.

Anything irreversible or wide-reaching confirms first, and they all extend
`ui/_confirm_page.html` — so each one states the consequence, offers the
button(s) that carry it out, and offers Cancel back to where the user was.
These check the three that exist rather than the template in isolation, since
what matters is that the real pages share it.
"""

from django.urls import reverse

from ui.models import Experiment
from tests.conftest import DATASETS_DIR


def _exp(**overrides):
    fields = dict(
        name="c", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy", "f1"], current_metric="accuracy",
        original_metric="accuracy", seed=0)
    fields.update(overrides)
    return Experiment.objects.create(**fields)


def _runnable():
    """An experiment with a dataset, so the run view reaches the metric check."""
    from ui.services import snapshot as adapter
    return adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "c", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": "accuracy", "original_metric": "accuracy",
        "metric_names": ["accuracy", "f1"], "seed": 0,
        "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }, adopt_paths=True)


def _confirmations(client):
    """The site's confirmation pages: (name, html)."""
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
    """They share the confirm page's structure: the consequence in a warning,
    and the actions in one form."""
    for name, html in _confirmations(client):
        assert "confirm-message" in html, name
        assert "confirm-actions" in html, name


def test_every_confirmation_offers_a_way_out(client):
    for name, html in _confirmations(client):
        assert "Cancel" in html, name
