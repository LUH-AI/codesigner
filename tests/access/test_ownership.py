"""A viewer can read and export an experiment, and a contributor can also run,
edit and delete it. Only the owner shares the experiment or hands it to a
contributor. `test_groups.py` covers access between groups.
"""

import pytest
from django.urls import reverse

from experiments.models import Experiment, ExperimentShare
from access.policy import DELETE, EXPORT, RUN

VIEWER, CONTRIBUTOR = ExperimentShare.VIEWER, ExperimentShare.CONTRIBUTOR

from tests.conftest import export_ihpo, post_new_experiment

pytestmark = pytest.mark.django_db


@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True


@pytest.fixture
def group():
    """`group` is the module's only group."""
    from access.models import Group

    return Group.objects.create(name="lab", user_limit=10)


def _member(django_user_model, name, group, role="member"):
    from access.models import Membership

    user = django_user_model.objects.create_user(username=name, password="pw")
    Membership.objects.create(user=user, group=group, role=role)
    return django_user_model.objects.get(pk=user.pk)


@pytest.fixture
def ana(django_user_model, group):
    return _member(django_user_model, "ana", group)


@pytest.fixture
def ben(django_user_model, group):
    return _member(django_user_model, "ben", group)


@pytest.fixture
def administrator(django_user_model, group):
    """A lead's reach covers everything in the lead's group, and
    `test_groups.py` checks that bound.
    """
    return _member(django_user_model, "root", group, role="lead")


@pytest.fixture
def staff(django_user_model):
    """`staff` has `is_staff` alone, which reaches /admin/ and nothing in
    codesigner.
    """
    return django_user_model.objects.create_user(
        username="deskclerk", password="pw", is_staff=True)


def _granted(django_user_model, username, *codenames):
    """The re-fetch clears Django's per-instance permission cache.
    """
    from django.contrib.auth.models import Permission

    user = django_user_model.objects.create_user(username=username, password="pw")
    user.user_permissions.set(
        Permission.objects.filter(content_type__app_label="access",
                                  codename__in=codenames))
    return django_user_model.objects.get(pk=user.pk)


def _experiment(owner=None, name="owned", **shares) -> Experiment:
    """Shares pass as `ana=VIEWER` keywords."""
    exp = Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=owner)
    from django.contrib.auth import get_user_model
    for username, level in shares.items():
        ExperimentShare.objects.create(
            experiment=exp, level=level,
            user=get_user_model().objects.get(username=username))
    return exp


def _detail(client, exp):
    return client.get(reverse("ui:experiment_detail", args=[exp.pk]))


def test_the_default_policy_is_the_group_policy():
    """`REQUIRE_LOGIN` is the only switch."""
    from django.conf import settings as django_settings

    from access.policy import policy

    assert django_settings.EXPERIMENT_POLICY == "access.policy.GroupPolicy"
    assert type(policy()).__name__ == "GroupPolicy"


def test_the_old_policy_name_still_resolves():
    """Old `.env` files pinning `OwnerPolicy` must still start the app."""
    from django.utils.module_loading import import_string

    from access.policy import GroupPolicy

    assert import_string("access.policy.OwnerPolicy") is GroupPolicy


def test_ownership_is_invisible_without_accounts(client, settings):
    """Rows carrying an owner still show no ownership without accounts."""
    settings.REQUIRE_LOGIN = False
    exp = _experiment(owner=None)

    resp = _detail(client, exp)

    assert resp.status_code == 200
    assert resp.context["ownership"] is None


def test_creating_an_experiment_makes_you_its_owner(client, hosted, ana):
    from tests.conftest import DATASETS_DIR

    client.force_login(ana)
    post_new_experiment(client, {
        "name": "mine", "model_name": "Random Forest",
        "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification", "seed": "0",
    })

    assert Experiment.objects.get(data__name="mine").owner == ana


def test_you_can_do_everything_to_your_own(client, hosted, ana):
    exp = _experiment(owner=ana)
    client.force_login(ana)

    resp = _detail(client, exp)

    assert resp.status_code == 200
    assert resp.context["may"] == {"run": True, "edit": True,
                                   "delete": True, "export": True,
                                   "share": True, "rename": True}


def test_an_experiment_you_do_not_own_is_not_there_at_all(client, hosted, ana, ben):
    """A 403 would reveal who owns what."""
    exp = _experiment(owner=ben)
    client.force_login(ana)

    assert _detail(client, exp).status_code == 404


def test_someone_elses_experiment_is_not_in_your_sidebar(client, hosted, ana, ben):
    _experiment(owner=ben, name="bens-private-work")
    client.force_login(ana)

    assert b"bens-private-work" not in client.get(reverse("ui:home")).content


def test_a_viewer_can_read_it(client, hosted, ana, ben):
    exp = _experiment(owner=ben, ana=VIEWER)
    client.force_login(ana)

    assert _detail(client, exp).status_code == 200


def test_a_viewer_cannot_run_change_or_delete_it(client, hosted, ana, ben):
    """Reading results grants no power to overwrite them."""
    exp = _experiment(owner=ben, ana=VIEWER)
    client.force_login(ana)

    assert _detail(client, exp).context["may"] == {
        "run": False, "edit": False, "delete": False, "export": True,
        "share": False, "rename": False}


def test_a_viewer_running_it_is_refused(client, hosted, ana, ben):
    """The route checks access, since the form can be posted without the page."""
    exp = _experiment(owner=ben, ana=VIEWER)
    client.force_login(ana)

    resp = client.post(reverse("ui:experiment_run", args=[exp.pk]),
                       {"max_trials": "3", "optimize_metric": "accuracy"})

    assert resp.status_code == 403


def test_a_viewer_deleting_it_is_refused(client, hosted, ana, ben):
    exp = _experiment(owner=ben, ana=VIEWER)
    client.force_login(ana)

    resp = client.post(reverse("ui:experiment_delete", args=[exp.pk]))

    assert resp.status_code == 403
    assert Experiment.objects.filter(pk=exp.pk).exists()


def test_a_viewer_can_export_it(client, hosted, ana, ben):
    """An export strips server paths and carries only what the page already
    shows."""
    exp = _experiment(owner=ben, ana=VIEWER)
    client.force_login(ana)

    assert export_ihpo(client, exp.pk).status_code == 200


def test_a_contributor_can_do_everything_but_share_or_rename_it(client, hosted, ana, ben):
    exp = _experiment(owner=ben, ana=CONTRIBUTOR)
    client.force_login(ana)

    assert _detail(client, exp).context["may"] == {
        "run": True, "edit": True, "delete": True, "export": True,
        "share": False, "rename": False}


def test_a_contributor_can_delete_it(client, hosted, ana, ben):
    exp = _experiment(owner=ben, ana=CONTRIBUTOR)
    client.force_login(ana)

    client.post(reverse("ui:experiment_delete", args=[exp.pk]))

    exp.refresh_from_db()
    assert exp.deleted_at is not None


def test_a_contributor_cannot_share_it_on(client, hosted, ana, ben,
                                          django_user_model):
    """Only the owner decides who may reach the work."""
    cleo = django_user_model.objects.create_user(username="cleo")
    from access.models import Membership
    Membership.objects.create(user=cleo, group=ana.memberships.get().group)
    exp = _experiment(owner=ben, ana=CONTRIBUTOR)
    client.force_login(ana)

    resp = client.post(reverse("ui:experiment_share", args=[exp.pk]),
                       {"user": cleo.pk, "level": VIEWER})

    assert resp.status_code == 403
    assert not exp.shares.filter(user=cleo).exists()


def test_the_owner_sets_each_level_and_takes_it_away(client, hosted, ana, ben):
    exp = _experiment(owner=ana)
    client.force_login(ana)
    share = reverse("ui:experiment_share", args=[exp.pk])

    client.post(share, {"user": ben.pk, "level": VIEWER})
    assert exp.shares.get(user=ben).level == VIEWER

    client.post(share, {"user": ben.pk, "level": CONTRIBUTOR})
    assert exp.shares.get(user=ben).level == CONTRIBUTOR
    assert exp.shares.get(user=ben).granted_by == ana

    client.post(share, {"user": ben.pk, "level": "none"})
    assert not exp.shares.exists()


def test_nobody_else_can_share_your_experiment_out_from_under_you(client, hosted, ana, ben):
    exp = _experiment(owner=ben)
    client.force_login(ana)

    resp = client.post(reverse("ui:experiment_share", args=[exp.pk]),
                       {"user": ana.pk, "level": CONTRIBUTOR})

    assert resp.status_code == 404
    assert not exp.shares.exists()


def test_sharing_goes_back_where_it_was_asked_from(client, hosted, ana, ben):
    """Share controls live on the group page."""
    exp = _experiment(owner=ana)
    client.force_login(ana)

    resp = client.post(reverse("ui:experiment_share", args=[exp.pk]),
                       {"user": ben.pk, "level": VIEWER, "next": "/group/"})

    assert resp["Location"] == "/group/"


def test_and_never_off_the_site(client, hosted, ana, ben):
    exp = _experiment(owner=ana)
    client.force_login(ana)

    resp = client.post(reverse("ui:experiment_share", args=[exp.pk]),
                       {"user": ben.pk, "level": VIEWER,
                        "next": "https://elsewhere.example/"})

    assert resp["Location"] == reverse("ui:experiment_detail", args=[exp.pk])


def test_an_unknown_level_is_refused(client, hosted, ana, ben):
    exp = _experiment(owner=ana)
    client.force_login(ana)

    resp = client.post(reverse("ui:experiment_share", args=[exp.pk]),
                       {"user": ben.pk, "level": "owner"})

    assert resp.status_code == 400
    assert not exp.shares.exists()


def test_the_owner_cannot_be_given_a_level(client, hosted, ana):
    """The owner already has full access."""
    exp = _experiment(owner=ana)
    client.force_login(ana)

    resp = client.post(reverse("ui:experiment_share", args=[exp.pk]),
                       {"user": ana.pk, "level": VIEWER})

    assert resp.status_code == 404
    assert not exp.shares.exists()


def test_ownership_goes_to_a_contributor(client, hosted, ana, ben):
    """The previous owner stays on as a contributor."""
    exp = _experiment(owner=ana, ben=CONTRIBUTOR)
    client.force_login(ana)

    client.post(reverse("ui:experiment_transfer", args=[exp.pk]), {"user": ben.pk})
    exp.refresh_from_db()

    assert exp.owner == ben
    assert exp.shares.get(user=ana).level == CONTRIBUTOR
    assert not exp.shares.filter(user=ben).exists(), "an owner holds no grant"


def test_ownership_does_not_go_to_a_viewer(client, hosted, ana, ben):
    """The page offers only contributors, and the view answers any other request
    with a 404."""
    exp = _experiment(owner=ana, ben=VIEWER)
    client.force_login(ana)

    resp = client.post(reverse("ui:experiment_transfer", args=[exp.pk]), {"user": ben.pk})
    exp.refresh_from_db()

    assert resp.status_code == 404

    assert exp.owner == ana
    assert exp.shares.get(user=ben).level == VIEWER


def test_only_the_owner_hands_it_over(client, hosted, ana, ben):
    exp = _experiment(owner=ben, ana=CONTRIBUTOR)
    client.force_login(ana)

    resp = client.post(reverse("ui:experiment_transfer", args=[exp.pk]),
                       {"user": ana.pk})
    exp.refresh_from_db()

    assert resp.status_code == 403
    assert exp.owner == ben


def test_the_new_owner_can_share_it_and_the_old_one_cannot(client, hosted, ana, ben):
    exp = _experiment(owner=ana, ben=CONTRIBUTOR)
    client.force_login(ana)
    client.post(reverse("ui:experiment_transfer", args=[exp.pk]), {"user": ben.pk})

    assert _detail(client, exp).context["may"]["share"] is False
    client.force_login(ben)
    assert _detail(client, exp).context["may"]["share"] is True


def test_an_ownerless_experiment_belongs_to_nobody(client, hosted, ana):
    """An ownerless row is reachable by nobody, and `access/admin.py` refuses to
    delete a user who still owns experiments.
    """
    exp = _experiment(owner=None)
    client.force_login(ana)

    assert _detail(client, exp).status_code == 404


def test_deleting_a_user_keeps_their_experiments(hosted, ana):
    """`Experiment.owner` is `SET_NULL`, so removing a person keeps results
    others rely on."""
    exp = _experiment(owner=ana)

    ana.delete()
    exp.refresh_from_db()

    assert exp.owner is None


# Each administrator power is a separate permission.


def test_a_lead_sees_and_can_run_everything_in_their_group(client, hosted, ben,
                                                           administrator):
    """A lead reaches the group's work but cannot delete or share it."""
    exp = _experiment(owner=ben)
    client.force_login(administrator)

    resp = _detail(client, exp)

    assert resp.status_code == 200
    assert resp.context["may"] == {"run": True, "edit": True, "export": True,
                                   "delete": False, "share": False, "rename": False}


def test_is_staff_alone_grants_nothing_here(client, hosted, ben, staff):
    """A refactor reaching for `is_staff` again fails here.
    """
    exp = _experiment(owner=ben)
    client.force_login(staff)

    assert _detail(client, exp).status_code == 404


def test_seeing_everything_is_not_being_allowed_to_touch_it(client, hosted, ben,
                                                            django_user_model):
    """A supervisor can read results without spending compute or deleting
    anything."""
    reader = _granted(django_user_model, "reader", "view_all_experiments")
    exp = _experiment(owner=ben)
    client.force_login(reader)

    resp = _detail(client, exp)

    # Reaching the page is the view check.
    assert resp.status_code == 200, "they should be able to see it"
    assert resp.context["may"][EXPORT], "reading includes taking a copy"
    assert not resp.context["may"][RUN]
    assert not resp.context["may"][DELETE]


def test_managing_implies_seeing(client, hosted, ben, django_user_model):
    """The queryset returns everything for either permission."""
    manager = _granted(django_user_model, "manager", "manage_experiments")
    exp = _experiment(owner=ben)
    client.force_login(manager)

    resp = _detail(client, exp)

    assert resp.status_code == 200
    assert resp.context["may"][RUN]


def test_a_superuser_needs_no_grant(client, hosted, ben, django_user_model):
    """Superusers keep every power through an upgrade."""
    root = django_user_model.objects.create_superuser(username="su", password="pw")
    exp = _experiment(owner=ben)
    client.force_login(root)

    assert all(_detail(client, exp).context["may"].values())


def test_an_export_names_no_paths_on_this_server(client, ana):
    """Server paths mean nothing to the recipient and expose the instance's
    layout."""
    import json

    from django.core.files.base import ContentFile

    from tests.conftest import DATASETS_DIR

    exp = _experiment(owner=ana)
    exp.data.dataset.save("iris.csv", ContentFile((DATASETS_DIR / "iris.csv").read_bytes()))

    body = export_ihpo(client, exp.pk).content
    snapshot = json.loads(body)

    assert snapshot["dataset"]["path"] == ""
    assert snapshot["model"]["path"] == ""


def test_the_experiment_itself_still_knows_its_paths(client, ana):
    """The detail page and the run engine rebuild from the snapshot and need the
    file."""
    from django.core.files.base import ContentFile

    from tests.conftest import DATASETS_DIR
    from experiments.services import snapshot as adapter

    exp = _experiment(owner=ana)
    exp.data.dataset.save("iris.csv", ContentFile((DATASETS_DIR / "iris.csv").read_bytes()))

    assert adapter.snapshot_from_experiment(exp)["dataset"]["path"] != ""


def test_only_the_owner_may_rename_it(client, hosted, ana, ben):
    """A contributor may do everything except rename."""
    exp = _experiment(owner=ben, ana=CONTRIBUTOR)
    url = reverse("ui:experiment_rename", args=[exp.pk])

    client.force_login(ana)
    assert client.post(url, {"name": "ana's"}).status_code == 403
    exp.refresh_from_db()
    assert exp.name == "owned"

    client.force_login(ben)
    client.post(url, {"name": "ben's"})
    exp.refresh_from_db()
    assert exp.name == "ben's"
