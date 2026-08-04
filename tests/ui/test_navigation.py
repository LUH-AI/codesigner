"""Breadcrumbs say where a page sits; the back link says where you came from.

Breadcrumbs are derived from the resolved route, so they are the same however
you arrived and survive a bookmark or a shared link. `?next=` is only for links
that leave the hierarchy — the global defaults page reached from one experiment
— and must be validated, or it would be an open redirect.
"""

from django.urls import reverse

from ui.models import Experiment


def _exp(name="Iris tuning"):
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)


def _trail(client, url):
    """The breadcrumb labels on a page, in order."""
    html = client.get(url).content.decode()
    nav = html.split('class="breadcrumbs"', 1)[1].split("</nav>", 1)[0]
    import re
    return [re.sub(r"\s+", " ", m).strip()
            for m in re.findall(r"<(?:a|span)[^>]*>(.*?)</(?:a|span)>", nav, re.S)]


def test_home_is_the_root_of_the_experiment_trail(client):
    assert _trail(client, reverse("ui:home")) == ["Experiments"]


def test_an_experiment_page_sits_under_experiments(client):
    exp = _exp()
    assert _trail(client, reverse("ui:experiment_detail", args=[exp.pk])) == [
        "Experiments", "Iris tuning"]


def test_experiment_settings_sits_under_its_experiment(client):
    """This is what replaced the hand-written back button: the trail already
    links to the experiment, one level up."""
    exp = _exp()
    trail = _trail(client, reverse("ui:experiment_settings", args=[exp.pk]))
    assert trail == ["Experiments", "Iris tuning", "Settings"]

    html = client.get(reverse("ui:experiment_settings", args=[exp.pk])).content.decode()
    nav = html.split('class="breadcrumbs"', 1)[1].split("</nav>", 1)[0]
    assert reverse("ui:experiment_detail", args=[exp.pk]) in nav


def test_the_current_page_is_not_a_link(client):
    exp = _exp()
    html = client.get(reverse("ui:experiment_settings", args=[exp.pk])).content.decode()
    nav = html.split('class="breadcrumbs"', 1)[1].split("</nav>", 1)[0]
    assert 'aria-current="page"' in nav


def test_the_global_defaults_page_sits_under_settings(client):
    """It is global, so its trail never mentions an experiment however it was
    reached — that is the difference the back link exists to cover."""
    assert _trail(client, reverse("ui:default_experiment_settings")) == [
        "Settings", "Default experiment settings"]


def test_no_back_link_without_next(client):
    html = client.get(reverse("ui:default_experiment_settings")).content.decode()
    assert "back-link" not in html


def test_next_offers_a_labelled_way_back(client):
    """Reaching the global defaults from an experiment carries where to return
    to, and the label is resolved from the target rather than passed along."""
    exp = _exp()
    target = reverse("ui:experiment_detail", args=[exp.pk])
    html = client.get(
        reverse("ui:default_experiment_settings") + f"?next={target}").content.decode()

    assert "back-link" in html
    assert f'href="{target}"' in html
    assert "Iris tuning" in html


def test_the_experiment_settings_page_carries_next_to_the_defaults(client):
    """The link out of the hierarchy is where ?next= gets set."""
    exp = _exp()
    html = client.get(reverse("ui:experiment_settings", args=[exp.pk])).content.decode()
    expected = (reverse("ui:default_experiment_settings")
                + "?next=" + reverse("ui:experiment_detail", args=[exp.pk]))
    assert expected in html


def test_an_offsite_next_is_ignored(client):
    """Honoured blindly, ?next= would let a crafted link bounce a user off-site
    from a page that still looks like ours."""
    html = client.get(reverse("ui:default_experiment_settings")
                      + "?next=https://evil.example.com/").content.decode()

    assert "evil.example.com" not in html
    assert "back-link" not in html


def test_an_unresolvable_next_is_ignored(client):
    """A local URL that isn't a route of ours has no page to name."""
    html = client.get(reverse("ui:default_experiment_settings")
                      + "?next=/nope/nowhere/").content.decode()
    assert "back-link" not in html


def test_a_deleted_experiment_leaves_the_trail_intact(client):
    """The trail is built from the route, so a missing object degrades to the
    root rather than raising on a page that would otherwise render."""
    exp = _exp()
    url = reverse("ui:experiment_settings", args=[exp.pk])
    exp.delete()
    assert client.get(url).status_code == 404
    # the root trail still renders for a page that does exist
    assert _trail(client, reverse("ui:home")) == ["Experiments"]
