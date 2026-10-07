"""An experiment's name is optional: one without is shown as "Untitled
Experiment" (`Experiment.title`), and its owner may name it, rename it, or take
the name away again from its heading."""

import pytest
from django.urls import reverse

from ui.models import Experiment

pytestmark = pytest.mark.django_db


def _experiment(name=""):
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)


def test_an_unnamed_experiment_is_shown_as_untitled(client):
    """What: with no name, the sidebar, the heading and the breadcrumbs all say
    "Untitled Experiment", and the breadcrumb is still a link. How: makes one
    and reads its timeline page."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()
    crumbs = html.split('class="breadcrumbs"', 1)[1].split("</nav>", 1)[0]

    assert "Experiment: Untitled Experiment" in html
    assert f'href="{reverse("ui:experiment_detail", args=[exp.pk])}">Untitled Experiment</a>' in crumbs
    assert html.split('<nav class="sidebar"', 1)[1].count("Untitled Experiment") >= 1


def test_renaming_from_the_heading(client):
    """What: the heading's form names the experiment and returns to the page it
    was sent from; an empty name takes the name away. How: names it from the
    timeline, then clears it."""
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
    """What: the heading's rename field holds the name itself, empty for an
    unnamed experiment, with "Untitled Experiment" only as its placeholder.
    How: reads the field on an unnamed experiment's dashboard."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    field = html.split('id="rename-name"', 1)[1].split(">", 1)[0]

    assert 'value=""' in field and 'placeholder="Untitled Experiment"' in field


def test_an_unnamed_experiment_exports_under_its_identifier(client):
    """What: the .ihpo of an experiment with no name is named after its
    identifier, not ".ihpo". How: exports one and reads the filename."""
    from tests.conftest import export_ihpo

    exp = _experiment()
    disposition = export_ihpo(client, exp.pk)["Content-Disposition"]

    assert f'filename="{exp.identifier}.ihpo"' in disposition
