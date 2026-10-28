"""`Experiment.identifier` is a short handle that stays fixed while the name
changes; the `.ihpo` snapshot omits it.
"""

from django.urls import reverse

from experiments.models import Experiment, generate_identifier
from experiments.services import snapshot as snapshot_adapter


def _make(name):
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0,
    )


def test_identifier_generated_on_create():
    exp = _make("a")
    assert exp.identifier
    assert len(exp.identifier) >= 6


def test_identifiers_are_distinct():
    a, b = _make("a"), _make("b")
    assert a.identifier != b.identifier


def test_generate_identifier_helper_varies():
    assert generate_identifier() != generate_identifier()


def test_identifier_is_stable_across_updates():
    """Saving a result keeps the identifier."""
    exp = _make("a")
    original = exp.identifier
    exp.data.result = {"data": []}
    exp.data.save()
    exp.refresh_from_db()
    assert exp.identifier == original


def test_identifier_not_in_snapshot():
    exp = _make("a")
    assert "identifier" not in snapshot_adapter.snapshot_from_experiment(exp)


def test_identifier_shown_in_sidebar(client):
    exp = _make("sidebar-exp")
    body = client.get(reverse("ui:home")).content.decode()
    assert "({})".format(exp.identifier) in body  # in parentheses


def test_identifier_shown_on_detail_page(client):
    exp = _make("detail-exp")
    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert exp.identifier in body


def test_identifier_shown_on_delete_confirm(client):
    """The ID tells similar names apart before deleting."""
    exp = _make("delete-exp")
    body = client.get(reverse("ui:experiment_delete", args=[exp.pk])).content.decode()
    assert exp.identifier in body
