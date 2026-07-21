"""Step 5: running an experiment persists it.

With the database introduced, submitting the new-experiment form saves the
experiment (its chosen metric committed, its result, and a stored copy of the
dataset) so it survives and can be revisited. Unique names are enforced.
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
        "primary_metric": "accuracy",
        "seed": 0,
        "n_trials": 3,
    }
    data.update(overrides)
    return client.post(reverse("web:new_experiment"), data)


@pytest.mark.django_db
def test_running_persists_the_experiment(client):
    """A completed run is saved as a row with its result and committed metric.

    Expect: an Experiment named as submitted, carrying the run's result (3
    trials), the chosen metric recorded as both primary and original (first
    run commits the metric), and a stored dataset copy.
    """
    from web.models import Experiment

    resp = _post(client, name="persisted")
    assert resp.status_code == 200

    exp = Experiment.objects.get(name="persisted")
    assert exp.result is not None
    assert len(exp.result["data"]) == 3
    assert exp.primary_metric == "accuracy"
    assert exp.original_metric == "accuracy"
    assert exp.dataset


@pytest.mark.django_db
def test_persisted_experiment_appears_in_sidebar(client):
    """A saved experiment is listed in the sidebar on later page loads."""
    _post(client, name="listed-exp")
    html = client.get(reverse("web:home")).content.decode()
    assert "listed-exp" in html
    # and it links to its detail page
    from web.models import Experiment
    pk = Experiment.objects.get(name="listed-exp").pk
    assert reverse("web:experiment_detail", args=[pk]) in html


@pytest.mark.django_db
def test_duplicate_name_is_rejected(client):
    """Running with a name that already exists is refused, keeping names unique.

    Expect: the second submission re-renders the form (200) with an
    "already exists" error and creates no second row.
    """
    from web.models import Experiment

    _post(client, name="dup")
    resp = _post(client, name="dup")

    assert resp.status_code == 200
    assert Experiment.objects.filter(name="dup").count() == 1
    assert "already exists" in resp.content.decode()
