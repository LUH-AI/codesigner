"""The sidebar has a link to `ui:experiment_detail` for each experiment, and the
detail caption carries a three-state metric label.
"""

import json

import pytest
from django.urls import reverse

from tests.conftest import FIXTURES_DIR


def _experiment(**overrides):
    """`_experiment` creates an `Experiment` row with valid defaults."""
    from experiments.models import Experiment

    fields = dict(
        name="exp-1",
        model_name="Random Forest",
        optimizer_name="Random Search",
        optimizer_params={},
        metric_names=["accuracy", "f1"],
        current_metric="accuracy",
        original_metric="accuracy",
        seed=42,
    )
    fields.update(overrides)
    return Experiment.objects.create(**fields)


@pytest.mark.django_db
def test_sidebar_lists_experiments_with_detail_links(client):
    a = _experiment(name="alpha")
    b = _experiment(name="beta")

    response = client.get(reverse("ui:home"))
    html = response.content.decode()

    for exp in (a, b):
        assert exp.name in html
        assert reverse("ui:experiment_detail", args=[exp.pk]) in html


@pytest.mark.django_db
def test_detail_shows_identity_and_result_summary(client):
    """The page renders name, model, optimizer, seed and the trial count from
    `test2.ihpo`.
    """
    snapshot = json.loads((FIXTURES_DIR / "test2.ihpo").read_text(encoding="utf-8"))
    exp = _experiment(
        name=snapshot["name"],
        model_name=snapshot["model_name"],
        optimizer_name=snapshot["optimizer_name"],
        metric_names=snapshot["metric_names"],
        current_metric=snapshot["primary_metric"],
        original_metric=snapshot["original_metric"],
        seed=snapshot["seed"],
        result=snapshot["result"],
    )

    response = client.get(reverse("ui:experiment_detail", args=[exp.pk]))
    assert response.status_code == 200
    html = response.content.decode()

    assert exp.name in html
    assert exp.data.model_name in html
    assert exp.data.optimizer_name in html
    assert str(exp.data.seed) in html
    assert str(len(snapshot["result"]["data"])) in html


@pytest.mark.django_db
@pytest.mark.parametrize("primary,original,expected_label", [
    ("accuracy", None,       "~"),              # no metric committed yet
    ("f1",       "accuracy", "Inconsistent"),   # metric changed
    ("accuracy", "accuracy", "accuracy"),       # consistent
])
def test_detail_metric_label_rules(client, primary, original, expected_label):
    exp = _experiment(current_metric=primary, original_metric=original)

    response = client.get(reverse("ui:experiment_detail", args=[exp.pk]))
    assert expected_label in response.content.decode()


@pytest.mark.django_db
def test_unknown_experiment_returns_404(client):
    """A missing id gets 404."""
    response = client.get(reverse("ui:experiment_detail", args=[99999]))
    assert response.status_code == 404
