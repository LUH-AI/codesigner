"""Data Handling's pages: reachable from the experiment, under it in the trail,
with their own sidebar — and, for now, only saying what they will do."""

import re

import pytest
from django.urls import reverse

from datahandling.sections import SECTIONS
from ui.models import Experiment
from ui.permissions import OpenPolicy


class HidesEverything(OpenPolicy):
    """A policy that sees no experiment at all."""

    def experiments(self, request):
        return Experiment.objects.none()


def _exp(name="Iris tuning"):
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)


def _url(exp, section):
    if section is SECTIONS[0]:
        return reverse("datahandling:data_overview", args=[exp.pk])
    return reverse("datahandling:data_section", args=[exp.pk, section.slug])


def _trail(html):
    nav = html.split('class="breadcrumbs"', 1)[1].split("</nav>", 1)[0]
    return [re.sub(r"\s+", " ", m).strip()
            for m in re.findall(r"<(?:a|span)[^>]*>(.*?)</(?:a|span)>", nav, re.S)]


def _sidebar(html):
    return html.split('<nav class="sidebar">', 1)[1].split("</nav>", 1)[0]


@pytest.mark.parametrize("section", SECTIONS, ids=[s.slug for s in SECTIONS])
def test_every_section_renders_what_it_plans(client, section):
    """What: each section's page shows its title and every capability it lists.
    How: fetches the page and looks for each capability's title in it."""
    exp = _exp()
    response = client.get(_url(exp, section))

    assert response.status_code == 200
    html = response.content.decode()
    assert str(section.title) in html
    for capability in section.planned:
        assert str(capability.title) in html


def test_the_first_sections_slug_is_not_a_second_address(client):
    """What: the overview is only reachable at the data root.
    How: requests the overview's slug as a section and expects a 404."""
    exp = _exp()
    url = reverse("datahandling:data_section", args=[exp.pk, SECTIONS[0].slug])
    assert client.get(url).status_code == 404


def test_an_unknown_section_is_a_404(client):
    """What: a slug no section has is not a page.
    How: requests a made-up slug."""
    exp = _exp()
    url = reverse("datahandling:data_section", args=[exp.pk, "no-such-section"])
    assert client.get(url).status_code == 404


def test_an_invisible_experiment_has_no_data_pages(client, settings):
    """What: the data pages exist only for experiments the reader can see.
    How: with a policy that hides everything, both kinds of page 404."""
    exp = _exp()
    settings.EXPERIMENT_POLICY = f"{__name__}.HidesEverything"

    assert client.get(_url(exp, SECTIONS[0])).status_code == 404
    assert client.get(_url(exp, SECTIONS[1])).status_code == 404


def test_the_overview_sits_under_its_experiment(client):
    """What: the trail reads Experiments / <name> / Data.
    How: reads the breadcrumb labels off the overview page."""
    exp = _exp()
    html = client.get(_url(exp, SECTIONS[0])).content.decode()
    assert _trail(html) == ["Experiments", "Iris tuning", "Data"]


def test_a_section_sits_under_data(client):
    """What: a section's trail continues past Data, and Data links back to the
    overview. How: reads the labels and the links out of the breadcrumbs."""
    exp = _exp()
    section = SECTIONS[1]
    html = client.get(_url(exp, section)).content.decode()

    assert _trail(html) == ["Experiments", "Iris tuning", "Data", str(section.title)]
    nav = html.split('class="breadcrumbs"', 1)[1].split("</nav>", 1)[0]
    assert f'href="{_url(exp, SECTIONS[0])}"' in nav


def test_the_sidebar_is_the_sections(client):
    """What: on a data page the experiment list gives way to the sections' tabs,
    with the current one marked and a way back to the experiment.
    How: reads the inner sidebar of one section's page."""
    exp = _exp(name="list-me-not")
    section = SECTIONS[2]
    sidebar = _sidebar(client.get(_url(exp, section)).content.decode())

    assert "Data Handling" in sidebar
    for s in SECTIONS:
        assert f'href="{_url(exp, s)}"' in sidebar
    assert re.search(r'class="tab active"\s+href="%s"' % re.escape(_url(exp, section)), sidebar)
    assert reverse("ui:experiment_detail", args=[exp.pk]) in sidebar
    assert "Experiment Selection" not in sidebar


def test_the_experiment_page_links_to_its_data(client):
    """What: the experiment's own sidebar has the way in.
    How: looks for the overview's URL in the experiment page's sidebar."""
    exp = _exp()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert f'href="{_url(exp, SECTIONS[0])}"' in _sidebar(html)


def test_the_section_slugs_are_unique():
    """What: no two sections share an address.
    How: compares the number of slugs with the number of distinct ones."""
    slugs = [s.slug for s in SECTIONS]
    assert len(slugs) == len(set(slugs))
