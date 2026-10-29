"""Deleting puts an experiment in its owner's and contributors' bins, where any
of them can restore it. Once no person's bin contains it, the site admins' bin
keeps it until a site admin destroys it.
"""

import pytest
from django.urls import reverse

from access.models import Group, Membership
from experiments.models import BinEntry, Experiment, ExperimentShare, Run

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
    """Ana owns, Ben contributes and Cleo reads."""
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


# Whose bin

def test_deleting_puts_it_in_the_owners_and_contributors_bins(client, hosted, lab):
    _delete(client, lab)

    assert set(BinEntry.objects.values_list("user__username", flat=True)) == {"ana", "ben"}
    assert _in_bin(client, lab["ana"]) == ["wine"]
    assert _in_bin(client, lab["ben"]) == ["wine"]


def test_and_nobody_elses(client, hosted, lab):
    """The deletion stays out of the bins of Cleo the viewer, Vera the lead and
    Dan."""
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


# Out of sight

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
    """`GroupPolicy.trash` omits a group-less experiment while a person's bin
    still contains it."""
    from access.policy import GroupPolicy

    _delete(client, lab)
    Experiment.objects.filter(pk=lab["exp"].pk).update(trashed_at=lab["exp"].deleted_at,
                                                       group=None)
    site = type("R", (), {"user": lab["ana"]})()

    assert not GroupPolicy().trash(site).exists()


# Restoring

def test_restoring_puts_it_back_exactly(client, hosted, lab):
    """Restoring brings back the data, runs, priors, shares, settings and model
    environment."""
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


# Taking a copy

def test_it_can_be_downloaded_from_the_bin(client, hosted, lab):
    """`bin_download` serves the same file an export makes."""
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


# Dismissing

def _dismiss(client, user, exp):
    client.force_login(user)
    return client.post(reverse("ui:bin_dismiss", args=[exp.pk]))


def _site_bin(lab):
    from access.policy import GroupPolicy
    from django.contrib.auth.models import Permission

    site = lab.get("site")
    if site is None:
        site = type(lab["ana"]).objects.create_user(username="site")
        site.user_permissions.add(Permission.objects.get(
            content_type__app_label="access", codename="manage_site"))
        lab["site"] = site = type(site).objects.get(pk=site.pk)
    return list(GroupPolicy().trash(type("R", (), {"user": site})())
                .values_list("pk", flat=True))


def test_taking_it_out_of_your_bin_leaves_everyone_elses(client, hosted, lab):
    _delete(client, lab)

    _dismiss(client, lab["ben"], lab["exp"])

    assert _in_bin(client, lab["ben"]) == []
    assert _in_bin(client, lab["ana"]) == ["wine"]
    assert Experiment.objects.filter(pk=lab["exp"].pk).exists()


def test_the_owner_taking_it_out_destroys_nothing_either(client, hosted, lab):
    _delete(client, lab)

    _dismiss(client, lab["ana"], lab["exp"])

    assert _in_bin(client, lab["ben"]) == ["wine"]
    assert Run.objects.filter(experiment=lab["exp"]).exists()


def test_and_a_colleague_restoring_it_does_not_reshare_it(client, hosted, lab):
    """A dismissal ends the dismisser's access, even after a colleague restores
    the experiment."""
    _delete(client, lab)
    _dismiss(client, lab["ben"], lab["exp"])

    client.force_login(lab["ana"])
    client.post(reverse("ui:bin_restore", args=[lab["exp"].pk]))
    exp = Experiment.objects.get(pk=lab["exp"].pk)

    assert exp.deleted_at is None
    assert dict(exp.shares.values_list("user__username", "level")) == {"cleo": VIEWER}
    client.force_login(lab["ben"])
    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).status_code == 404


def test_if_the_owner_gave_it_up_whoever_restores_it_owns_it(client, hosted, lab):
    """Ownership goes to the restoring contributor, who could receive it anyway."""
    _delete(client, lab)
    _dismiss(client, lab["ana"], lab["exp"])

    client.force_login(lab["ben"])
    client.post(reverse("ui:bin_restore", args=[lab["exp"].pk]))
    exp = Experiment.objects.get(pk=lab["exp"].pk)

    assert exp.owner == lab["ben"]
    assert not exp.shares.filter(user__in=[lab["ana"], lab["ben"]]).exists()
    client.force_login(lab["ana"])
    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).status_code == 404


def test_the_confirmation_says_who_still_has_it(client, hosted, lab):
    _delete(client, lab)
    client.force_login(lab["ben"])

    page = client.get(reverse("ui:bin_dismiss", args=[lab["exp"].pk]))

    assert [u.username for u in page.context["others"]] == ["ana"]
    assert Experiment.objects.get(pk=lab["exp"].pk).bin_entries.count() == 2, \
        "asking is not doing it"


# Site bin

def test_once_nobody_keeps_it_it_is_the_site_admins(client, hosted, lab):
    _delete(client, lab)
    _dismiss(client, lab["ben"], lab["exp"])
    assert _site_bin(lab) == [], "Ana still has it"

    _dismiss(client, lab["ana"], lab["exp"])

    assert _site_bin(lab) == [lab["exp"].pk]
    assert Experiment.objects.filter(pk=lab["exp"].pk).exists()


def test_after_the_retention_it_leaves_its_peoples_bins(client, hosted, lab, settings):
    from datetime import timedelta

    from django.utils import timezone

    _delete(client, lab)
    Experiment.objects.filter(pk=lab["exp"].pk).update(
        deleted_at=timezone.now() - timedelta(days=settings.BIN_RETENTION_DAYS, minutes=1))

    assert _in_bin(client, lab["ana"]) == []
    assert _in_bin(client, lab["ben"]) == []
    assert _site_bin(lab) == [lab["exp"].pk]


def test_but_not_a_day_before(client, hosted, lab, settings):
    from datetime import timedelta

    from django.utils import timezone

    _delete(client, lab)
    Experiment.objects.filter(pk=lab["exp"].pk).update(
        deleted_at=timezone.now() - timedelta(days=settings.BIN_RETENTION_DAYS - 1))

    assert _in_bin(client, lab["ana"]) == ["wine"]
    assert _site_bin(lab) == []


def test_the_bin_says_when_it_leaves(client, hosted, lab, settings):
    from datetime import timedelta

    _delete(client, lab)
    client.force_login(lab["ana"])

    row = client.get(reverse("ui:experiment_bin")).context["binned"][0]

    assert row["expires_at"] == lab["exp"].deleted_at + timedelta(days=settings.BIN_RETENTION_DAYS)


def test_a_site_admin_puts_it_back(client, hosted, lab):
    """Ben's bin expired, so Ben keeps his grant."""
    from datetime import timedelta

    from django.utils import timezone

    _delete(client, lab)
    _dismiss(client, lab["ana"], lab["exp"])
    Experiment.objects.filter(pk=lab["exp"].pk).update(
        deleted_at=timezone.now() - timedelta(days=60))
    _site_bin(lab)
    client.force_login(lab["site"])

    client.post(reverse("ui:site_trash_rehome", args=[lab["exp"].pk]),
                {"group": lab["group"].pk})
    exp = Experiment.objects.get(pk=lab["exp"].pk)

    assert exp.deleted_at is None and not exp.bin_entries.exists()
    assert exp.owner == lab["ana"], "the owner of record, by the site admin's decision"
    assert dict(exp.shares.values_list("user__username", "level")) == {
        "ben": CONTRIBUTOR, "cleo": VIEWER}


def test_and_is_the_only_one_who_destroys_it(client, hosted, lab):
    _delete(client, lab)
    for who in ("ana", "ben"):
        _dismiss(client, lab[who], lab["exp"])
    _site_bin(lab)
    client.force_login(lab["site"])

    client.post(reverse("ui:site_trash_delete", args=[lab["exp"].pk]))

    assert not Experiment.objects.filter(pk=lab["exp"].pk).exists()
    assert not Run.objects.exists()


def test_the_site_bin_does_not_take_what_somebody_still_has(client, hosted, lab):
    _delete(client, lab)
    _site_bin(lab)
    client.force_login(lab["site"])

    resp = client.post(reverse("ui:site_trash_delete", args=[lab["exp"].pk]))

    assert resp.status_code == 404
    assert Experiment.objects.filter(pk=lab["exp"].pk).exists()


def test_the_site_bin_page_lists_both_kinds(client, hosted, lab, django_user_model):
    _delete(client, lab)
    for who in ("ana", "ben"):
        _dismiss(client, lab[who], lab["exp"])
    gone = Group.objects.create(name="gone", user_limit=1)
    owner = _person(django_user_model, "olaf", gone)
    other = Experiment.objects.create(
        name="left", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=owner)
    Membership.objects.filter(user=owner).delete()
    gone.delete()
    _site_bin(lab)
    client.force_login(lab["site"])

    resp = client.get(reverse("ui:site_trash"))
    body = resp.content.decode()

    assert resp.status_code == 200
    assert {e.pk for e in resp.context["experiments"]} == {lab["exp"].pk, other.pk}
    assert "Nobody kept it" in body and "Its group was deleted" in body


# Without accounts

def test_without_accounts_there_is_one_bin(client):
    exp = Experiment.objects.create(
        name="solo", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)

    client.post(reverse("ui:experiment_delete", args=[exp.pk]))
    assert [r["experiment"].name
            for r in client.get(reverse("ui:experiment_bin")).context["binned"]] == ["solo"]

    client.post(reverse("ui:bin_restore", args=[exp.pk]))
    assert client.get(reverse("ui:experiment_detail", args=[exp.pk])).status_code == 200


def test_without_accounts_removing_it_destroys_it(client):
    """With no accounts, `ui:bin_dismiss` destroys the experiment."""
    exp = Experiment.objects.create(
        name="solo", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0)
    client.post(reverse("ui:experiment_delete", args=[exp.pk]))

    client.post(reverse("ui:bin_dismiss", args=[exp.pk]))

    assert not Experiment.objects.exists()
