"""Step 5 (updated in Step 6): creating an experiment persists it.

Creating saves the experiment (dataset stored) so it survives and can be
revisited; it does not run (running is Step 6). Names need not be unique —
each experiment is told apart by its identifier.
"""

import pytest
from django.urls import reverse

from tests.conftest import DATASETS_DIR


def _post(client, **overrides):
    data = {
        "name": "persisted",
        "model_name": "Random Forest",
        "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"),
        "seed": 0,
    }
    data.update(overrides)
    return client.post(reverse("web:new_experiment"), data)


@pytest.mark.django_db
def test_creating_persists_the_experiment(client):
    """Creating saves the experiment with a stored dataset and no result yet.

    Expect: an Experiment row with a dataset file and result None (creating
    does not run), reachable at its detail page.
    """
    from web.models import Experiment

    resp = _post(client, name="persisted")
    exp = Experiment.objects.get(name="persisted")

    assert resp.status_code == 302
    assert exp.dataset
    assert exp.result is None
    assert client.get(reverse("web:experiment_detail", args=[exp.pk])).status_code == 200


@pytest.mark.django_db
def test_persisted_experiment_appears_in_sidebar(client):
    """A saved experiment is listed in the sidebar on later page loads."""
    _post(client, name="listed-exp")
    html = client.get(reverse("web:home")).content.decode()
    assert "listed-exp" in html
    from web.models import Experiment
    pk = Experiment.objects.get(name="listed-exp").pk
    assert reverse("web:experiment_detail", args=[pk]) in html


@pytest.mark.django_db
def test_duplicate_name_is_allowed(client):
    """Creating with a name that already exists is allowed — the two rows are
    told apart by their identifiers, not their names."""
    from web.models import Experiment

    _post(client, name="dup")
    resp = _post(client, name="dup")

    assert resp.status_code == 302  # created and redirected, not re-rendered with an error
    assert Experiment.objects.filter(name="dup").count() == 2
