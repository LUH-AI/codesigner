"""Step 2.5 TDD: sidebar experiment list and the experiment detail page.

Makes Step 3's database rows browsable. Display rules come from the
Streamlit experiment view: the caption shows Model/Optimizer/Metric/Seed,
where the metric label is "~" when no original metric is recorded,
"Inconsistent" when the primary metric no longer matches the original, and
the metric name otherwise.
"""

import json

import pytest
from django.urls import reverse

from tests.conftest import FIXTURES_DIR


def _experiment(**overrides):
    """Create an Experiment row with valid defaults (import deferred so the
    suite collects while web.models doesn't exist yet)."""
    from web.models import Experiment

    fields = dict(
        name="exp-1",
        model_name="Random Forest",
        optimizer_name="Random Search",
        optimizer_params={},
        metric_names=["accuracy", "f1"],
        primary_metric="accuracy",
        original_metric="accuracy",
        seed=42,
    )
    fields.update(overrides)
    return Experiment.objects.create(**fields)


@pytest.mark.django_db
def test_sidebar_lists_experiments_with_detail_links(client):
    """Every stored experiment appears in the sidebar as a link to its detail page.

    Setup: two experiments in the database.
    Expect: the home page contains both names and both detail URLs — the
    sidebar is rendered on every page, home included.
    """
    a = _experiment(name="alpha")
    b = _experiment(name="beta")

    response = client.get(reverse("web:home"))
    html = response.content.decode()

    for exp in (a, b):
        assert exp.name in html
        assert reverse("web:experiment_detail", args=[exp.pk]) in html


@pytest.mark.django_db
def test_detail_shows_identity_and_result_summary(client):
    """The detail page shows the experiment's identity fields and result summary.

    Setup: a row carrying the 30-trial fixture result.
    Expect: name, model, optimizer, seed, the trial count, and the stored
    best score all rendered.
    """
    snapshot = json.loads((FIXTURES_DIR / "test2.ihpo").read_text(encoding="utf-8"))
    exp = _experiment(
        name=snapshot["name"],
        model_name=snapshot["model_name"],
        optimizer_name=snapshot["optimizer_name"],
        metric_names=snapshot["metric_names"],
        primary_metric=snapshot["primary_metric"],
        original_metric=snapshot["original_metric"],
        seed=snapshot["seed"],
        result=snapshot["result"],
    )

    response = client.get(reverse("web:experiment_detail", args=[exp.pk]))
    assert response.status_code == 200
    html = response.content.decode()

    assert exp.name in html
    assert exp.model_name in html
    assert exp.optimizer_name in html
    assert str(exp.seed) in html
    assert str(len(snapshot["result"]["data"])) in html


@pytest.mark.django_db
@pytest.mark.parametrize("primary,original,expected_label", [
    ("accuracy", None,       "~"),              # never run: no metric committed yet
    ("f1",       "accuracy", "Inconsistent"),   # metric changed since first run
    ("accuracy", "accuracy", "accuracy"),       # consistent: show the metric itself
])
def test_detail_metric_label_rules(client, primary, original, expected_label):
    """The caption's metric label follows the three-state rule.

    "~" when original_metric is unset, "Inconsistent" when primary and
    original disagree, else the metric name — matching the experiment
    header the user sees when switching evaluation metrics.
    """
    exp = _experiment(primary_metric=primary, original_metric=original)

    response = client.get(reverse("web:experiment_detail", args=[exp.pk]))
    assert expected_label in response.content.decode()


@pytest.mark.django_db
def test_unknown_experiment_returns_404(client):
    """A detail URL for a nonexistent experiment id returns 404, not an error page."""
    response = client.get(reverse("web:experiment_detail", args=[99999]))
    assert response.status_code == 404
