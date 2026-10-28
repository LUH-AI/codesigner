"""`Experiment.title` shows a nameless experiment as "Untitled Experiment", and
its owner names, renames or clears the name from the heading."""

import pytest
from django.urls import reverse

from experiments.models import Experiment

pytestmark = pytest.mark.django_db


def _experiment(name=""):
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)


def test_an_unnamed_experiment_is_shown_as_untitled(client):
    """Sidebar, heading and breadcrumb link all read "Untitled Experiment"."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()
    crumbs = html.split('class="breadcrumbs"', 1)[1].split("</nav>", 1)[0]

    assert "Experiment: Untitled Experiment" in html
    assert f'href="{reverse("ui:experiment_detail", args=[exp.pk])}">Untitled Experiment</a>' in crumbs
    assert html.split('<nav class="sidebar"', 1)[1].count("Untitled Experiment") >= 1


def test_renaming_from_the_heading(client):
    """`experiment_rename` follows `next`; an empty name resets the title."""
    exp = _experiment("first")
    here = reverse("ui:experiment_timeline", args=[exp.pk])
    url = reverse("ui:experiment_rename", args=[exp.pk])

    assert client.post(url, {"name": "  second  ", "next": here})["Location"] == here
    exp.refresh_from_db()
    assert exp.name == "second"

    client.post(url, {"name": ""})
    exp.refresh_from_db()
    assert exp.name == "" and exp.title == "Untitled Experiment"


def test_the_rename_form_offers_the_name_as_given(client):
    """"Untitled Experiment" is only the field's placeholder."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    field = html.split('id="rename-name"', 1)[1].split(">", 1)[0]

    assert 'value=""' in field and 'placeholder="Untitled Experiment"' in field


def test_an_unnamed_experiment_exports_under_its_identifier(client):
    from tests.conftest import export_ihpo

    exp = _experiment()
    disposition = export_ihpo(client, exp.pk)["Content-Disposition"]

    assert f'filename="{exp.identifier}.ihpo"' in disposition
