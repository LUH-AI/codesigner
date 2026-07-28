"""Each experiment carries a short, stable identifier.

Experiment names are already unique, but a compact machine/human handle lets
two similarly-named experiments be told apart at a glance and gives a stable
reference independent of the (mutable) name. It is a Codesigner-local concept:
it is NOT written into the .ihpo snapshot, so cross-app interop with the
Streamlit format is unchanged.
"""

from django.urls import reverse

from ui.models import Experiment, generate_identifier
from ui.services import snapshot as snapshot_adapter


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
    """Editing/re-saving an experiment (e.g. after a run writes its result)
    must not change the identifier — it is a fixed handle."""
    exp = _make("a")
    original = exp.identifier
    exp.result = {"data": []}
    exp.save()
    exp.refresh_from_db()
    assert exp.identifier == original


def test_identifier_not_in_snapshot():
    """The snapshot/.ihpo format is unchanged — no identifier leaks into it."""
    exp = _make("a")
    assert "identifier" not in snapshot_adapter.snapshot_from_experiment(exp)


def test_identifier_shown_in_sidebar(client):
    exp = _make("sidebar-exp")
    body = client.get(reverse("ui:home")).content.decode()
    assert "({})".format(exp.identifier) in body  # shown in parentheses


def test_identifier_shown_on_detail_page(client):
    exp = _make("detail-exp")
    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert exp.identifier in body


def test_identifier_shown_on_delete_confirm(client):
    """The delete confirmation names the experiment AND its ID, so the right
    one is being deleted even when names are similar."""
    exp = _make("delete-exp")
    body = client.get(reverse("ui:experiment_delete", args=[exp.pk])).content.decode()
    assert exp.identifier in body
