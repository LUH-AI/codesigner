"""Creating an experiment saves it and starts no run.
"""

import pytest
from django.urls import reverse

from tests.conftest import DATASETS_DIR, post_new_experiment


def _post(client, **overrides):
    data = {
        "name": "persisted",
        "model_name": "Random Forest",
        "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification",
        "seed": 0,
    }
    data.update(overrides)
    return post_new_experiment(client, data)


@pytest.mark.django_db
def test_creating_persists_the_experiment(client):
    """`dataset_path` points at the demo itself, never a copy.
    """
    from experiments.models import Experiment

    resp = _post(client, name="persisted")
    exp = Experiment.objects.get(data__name="persisted")

    assert resp.status_code == 302
    assert exp.data.demo_dataset == "iris"
    path = exp.data.dataset_path()
    assert path is not None
    assert path.resolve() == (DATASETS_DIR / "iris.csv").resolve()
    assert exp.data.result is None
    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).status_code == 200


@pytest.mark.django_db
def test_persisted_experiment_appears_in_sidebar(client):
    _post(client, name="listed-exp")
    html = client.get(reverse("ui:home")).content.decode()
    assert "listed-exp" in html
    from experiments.models import Experiment
    pk = Experiment.objects.get(data__name="listed-exp").pk
    assert reverse("ui:experiment_detail", args=[pk]) in html


@pytest.mark.django_db
def test_duplicate_name_is_allowed(client):
    """Experiments are told apart by identifier."""
    from experiments.models import Experiment

    _post(client, name="dup")
    resp = _post(client, name="dup")

    assert resp.status_code == 302  # 302 means created
    assert Experiment.objects.filter(data__name="dup").count() == 2
