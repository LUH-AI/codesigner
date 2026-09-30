"""Deleting an experiment puts it in its people's bins, and it can come back.

*What:* deleting sets an experiment aside rather than destroying it. It leaves
every list and every URL, and lands in the bins of the people it belonged to —
its owner and its contributors, and nobody else. Any of them can restore it
exactly as it was or download it as an `.ihpo`; only its owner can destroy it,
and a contributor can take it out of their own bin without touching anybody
else's. Without accounts there is one bin, and it holds everything.

*How:* through the pages as each role, and at the row for what restoring has to
bring back: both halves of the experiment, its runs, its grants and its model
environment.
"""

import pytest
from django.urls import reverse

from access.models import Group, Membership
from ui.models import BinEntry, Experiment, ExperimentShare, Run

pytestmark = pytest.mark.django_db

VIEWER, CONTRIBUTOR = ExperimentShare.VIEWER, ExperimentShare.CONTRIBUTOR


@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True
    return settings


def _person(django_user_model, name, group, role=Membership.MEMBER):
    user = django_user_model.objects.create_user(username=name, password="pw")
    Membership.objects.create(user=user, group=group, role=role)
    return django_user_model.objects.get(pk=user.pk)


@pytest.fixture
def lab(django_user_model):
    """Ana's experiment, with Ben contributing, Cleo viewing and Vera leading."""
    group = Group.objects.create(name="lab", user_limit=9)
    people = {name: _person(django_user_model, name, group)
              for name in ("ana", "ben", "cleo", "dan")}
    people["vera"] = _person(django_user_model, "vera", group, Membership.LEAD)
    exp = Experiment.objects.create(
        name="wine", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=people["ana"],
        settings={"show_ice": False}, use_default_settings=False,
        env_status=Experiment.ENV_READY, env_meta={"python": "3.12"})
    exp.data.result = {"trials": [1, 2]}
    exp.data.priors = {"x": {"kind": "normal"}}
    exp.data.save(update_fields=["result", "priors"])
    ExperimentShare.objects.create(experiment=exp, user=people["ben"], level=CONTRIBUTOR)
    ExperimentShare.objects.create(experiment=exp, user=people["cleo"], level=VIEWER)
    Run.objects.create(experiment=exp, primary_metric="accuracy", status="done",
                       stopping={"max_trials": 2}, trial_offset=0,
                       priors={"x": {"kind": "normal"}})
    return {"group": group, "exp": exp, **people}


def _delete(client, lab, who="ana"):
    client.force_login(lab[who])
    client.post(reverse("ui:experiment_delete", args=[lab["exp"].pk]))
    lab["exp"].refresh_from_db()


def _in_bin(client, user):
    client.force_login(user)
    return [row["experiment"].name
            for row in client.get(reverse("ui:experiment_bin")).context["binned"]]


# ── whose bin ────────────────────────────────────────────────────────────────

def test_deleting_puts_it_in_the_owners_and_contributors_bins(client, hosted, lab):
    _delete(client, lab)

    assert set(BinEntry.objects.values_list("user__username", flat=True)) == {"ana", "ben"}
    assert _in_bin(client, lab["ana"]) == ["wine"]
    assert _in_bin(client, lab["ben"]) == ["wine"]


def test_and_nobody_elses(client, hosted, lab):
    """Not a viewer's, who could never have deleted it, and not the lead's —
    seeing a colleague's work was never the same as it being theirs."""
    _delete(client, lab)

    for who in ("cleo", "vera", "dan"):
        assert _in_bin(client, lab[who]) == [], who


def test_a_contributor_deleting_it_puts_it_in_the_owners_bin_too(client, hosted, lab):
    _delete(client, lab, who="ben")

    assert lab["exp"].deleted_by == lab["ben"]
    assert _in_bin(client, lab["ana"]) == ["wine"]


def test_the_confirmation_names_whose_bin_it_goes_to(client, hosted, lab):
    client.force_login(lab["ana"])

    body = client.get(reverse("ui:experiment_delete", args=[lab["exp"].pk])).content.decode()

    assert "ben" in body and "cleo" not in body


# ── out of sight ─────────────────────────────────────────────────────────────

def test_a_deleted_experiment_is_at_no_url(client, hosted, lab):
    _delete(client, lab)

    for who in ("ana", "ben", "cleo", "vera"):
        client.force_login(lab[who])
        assert client.get(reverse("ui:experiment_detail",
                                  args=[lab["exp"].pk])).status_code == 404, who


def test_and_in_no_list(client, hosted, lab):
    _delete(client, lab)

    for who in ("ana", "ben"):
        client.force_login(lab[who])
        listed = client.get(reverse("ui:home")).context["sidebar_experiments"]
        assert "wine" not in [e.name for e in listed], who
    client.force_login(lab["vera"])
    page = client.get(reverse("ui:group_detail", args=[lab["group"].pk])).context
    assert "wine" not in [e.name for e in page["others"]]


def test_a_deleted_groups_bin_leaves_it_to_its_people(client, hosted, lab):
    """Two bins, two audiences. What a person deleted is theirs to settle even
    when the group goes too."""
    from access.policy import GroupPolicy

    _delete(client, lab)
    Experiment.objects.filter(pk=lab["exp"].pk).update(trashed_at=lab["exp"].deleted_at,
                                                       group=None)
    site = type("R", (), {"user": lab["ana"]})()

    assert not GroupPolicy().trash(site).exists()


# ── restoring ────────────────────────────────────────────────────────────────

def test_restoring_puts_it_back_exactly(client, hosted, lab):
    """Every part of it: both halves, its runs and their priors, its grants,
    its settings and its model environment."""
    _delete(client, lab)

    client.post(reverse("ui:bin_restore", args=[lab["exp"].pk]))
    exp = Experiment.objects.get(pk=lab["exp"].pk)

    assert exp.deleted_at is None and exp.deleted_by is None
    assert exp.data.result == {"trials": [1, 2]}
    assert exp.data.priors == {"x": {"kind": "normal"}}
    assert exp.runs.get().priors == {"x": {"kind": "normal"}}
    assert dict(exp.shares.values_list("user__username", "level")) == {
        "ben": CONTRIBUTOR, "cleo": VIEWER}
    assert (exp.settings, exp.use_default_settings) == ({"show_ice": False}, False)
    assert (exp.env_status, exp.env_meta) == (Experiment.ENV_READY, {"python": "3.12"})
    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).status_code == 200


def test_restoring_takes_it_out_of_every_bin(client, hosted, lab):
    _delete(client, lab)
    client.force_login(lab["ben"])

    client.post(reverse("ui:bin_restore", args=[lab["exp"].pk]))

    assert not BinEntry.objects.exists()
    assert _in_bin(client, lab["ana"]) == []


def test_somebody_whose_bin_it_is_not_in_cannot_restore_it(client, hosted, lab):
    _delete(client, lab)
    client.force_login(lab["cleo"])

    resp = client.post(reverse("ui:bin_restore", args=[lab["exp"].pk]))

    lab["exp"].refresh_from_db()
    assert resp.status_code == 404
    assert lab["exp"].deleted_at is not None


# ── taking a copy ────────────────────────────────────────────────────────────

def test_it_can_be_downloaded_from_the_bin(client, hosted, lab):
    """The same file an export makes, asked about the same way."""
    from core import io

    _delete(client, lab, who="ben")
    url = reverse("ui:bin_download", args=[lab["exp"].pk])

    assert client.get(url).status_code == 200
    resp = client.post(url)

    assert io.parse(resp.content)["name"] == "wine"


def test_but_not_by_somebody_whose_bin_it_is_not_in(client, hosted, lab):
    _delete(client, lab)
    client.force_login(lab["cleo"])

    assert client.get(reverse("ui:bin_download", args=[lab["exp"].pk])).status_code == 404


# ── destroying it ────────────────────────────────────────────────────────────

def test_the_owner_can_delete_it_for_good(client, hosted, lab):
    _delete(client, lab)

    client.post(reverse("ui:bin_purge", args=[lab["exp"].pk]))

    assert not Experiment.objects.filter(pk=lab["exp"].pk).exists()
    assert not Run.objects.exists()


def test_a_contributor_cannot(client, hosted, lab):
    """The one step in the bin with no undoing, on work that is not theirs."""
    _delete(client, lab)
    client.force_login(lab["ben"])

    resp = client.post(reverse("ui:bin_purge", args=[lab["exp"].pk]))

    assert resp.status_code == 403
    assert Experiment.objects.filter(pk=lab["exp"].pk).exists()


def test_a_contributor_takes_it_out_of_their_own_bin_only(client, hosted, lab):
    _delete(client, lab)
    client.force_login(lab["ben"])

    client.post(reverse("ui:bin_dismiss", args=[lab["exp"].pk]))

    assert _in_bin(client, lab["ben"]) == []
    assert _in_bin(client, lab["ana"]) == ["wine"]


# ── without accounts ─────────────────────────────────────────────────────────

def test_without_accounts_there_is_one_bin(client):
    exp = Experiment.objects.create(
        name="solo", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)

    client.post(reverse("ui:experiment_delete", args=[exp.pk]))
    assert [r["experiment"].name
            for r in client.get(reverse("ui:experiment_bin")).context["binned"]] == ["solo"]

    client.post(reverse("ui:bin_restore", args=[exp.pk]))
    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).status_code == 200


def test_without_accounts_anybody_can_destroy_it(client):
    exp = Experiment.objects.create(
        name="solo", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)
    client.post(reverse("ui:experiment_delete", args=[exp.pk]))

    client.post(reverse("ui:bin_purge", args=[exp.pk]))

    assert not Experiment.objects.exists()
