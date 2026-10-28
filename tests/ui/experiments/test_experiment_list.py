"""`ui:experiment_list` paginates everything the sidebar cuts off."""

import pytest
from django.urls import reverse

from ui import context_processors
from ui.views import pages
from experiments.models import Experiment

pytestmark = pytest.mark.django_db


def _experiment(**overrides):
    fields = dict(name="exp", model_name="Random Forest",
                  optimizer_name="Random Search", metric_names=["accuracy"], seed=0)
    fields.update(overrides)
    return Experiment.objects.create(**fields)


def test_the_sidebar_caps_at_the_limit(client, monkeypatch):
    monkeypatch.setattr(context_processors, "SIDEBAR_LIMIT", 3)
    for i in range(5):
        _experiment(name=f"exp-{i}")

    response = client.get(reverse("ui:home"))

    assert len(response.context["sidebar_experiments"]) == 3


def test_the_sidebar_shows_the_most_recent(client, monkeypatch):
    monkeypatch.setattr(context_processors, "SIDEBAR_LIMIT", 1)
    older = _experiment(name="older")
    newer = _experiment(name="newer")

    response = client.get(reverse("ui:home"))

    assert list(response.context["sidebar_experiments"]) == [newer]
    assert older not in response.context["sidebar_experiments"]


def test_the_full_list_paginates(client, monkeypatch):
    monkeypatch.setattr(pages, "EXPERIMENT_LIST_PAGE_SIZE", 2)
    for i in range(5):
        _experiment(name=f"exp-{i}")

    first = client.get(reverse("ui:experiment_list"))
    second = client.get(reverse("ui:experiment_list"), {"page": 2})

    assert len(first.context["page"]) == 2
    assert first.context["page"].has_next()
    assert len(second.context["page"]) == 2
    assert set(first.context["page"].object_list) & set(second.context["page"].object_list) == set()


def test_an_out_of_range_page_settles_on_the_last_one(client, monkeypatch):
    monkeypatch.setattr(pages, "EXPERIMENT_LIST_PAGE_SIZE", 2)
    for i in range(3):
        _experiment(name=f"exp-{i}")

    response = client.get(reverse("ui:experiment_list"), {"page": 999})

    assert response.status_code == 200
    assert response.context["page"].number == response.context["page"].paginator.num_pages


def test_the_full_list_is_scoped_by_the_policy(client, django_user_model, settings):
    """The default `GroupPolicy` hides another user's experiment."""
    settings.REQUIRE_LOGIN = True
    ana = django_user_model.objects.create_user(username="ana", password="pw")
    django_user_model.objects.create_user(username="ben", password="pw")
    mine = _experiment(name="mine", owner=ana)
    _experiment(name="not-mine", owner=django_user_model.objects.get(username="ben"))
    client.force_login(ana)

    response = client.get(reverse("ui:experiment_list"))

    names = {e.name for e in response.context["page"]}
    assert names == {"mine"}
    assert mine.name in names
