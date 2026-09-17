"""Who owns an experiment, and what sharing one actually grants.

The distinction that carries the most weight is that **sharing is an invitation
to look, not a transfer of control**: a colleague can read and export a shared
experiment, and cannot run, edit or delete it. An owner's results should not
change because someone else pressed Run.

`GroupPolicy` is the default policy, so these also pin that it stays completely
transparent while `REQUIRE_LOGIN` is off — the same inert-by-default contract
the login wall has.

Ana and Ben are in one group here, because `shared` means *shared with my group*
and outside one it grants nothing. What happens **between** groups is
`test_groups.py`; this file is about what happens inside one.
"""

import pytest
from django.urls import reverse

from ui.models import Experiment
from ui.permissions import DELETE, EXPORT, RUN

from tests.conftest import export_ihpo

pytestmark = pytest.mark.django_db


@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True


@pytest.fixture
def group():
    """The one group everybody in this module belongs to."""
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
    """A group lead — which is what "can act on everything here" means now.

    It used to be membership of an `Administrators` group carrying instance-wide
    permissions. Roles replaced that: the reach a lead has is bounded by their
    group, which is the point, and `test_groups.py` is where that bound is
    checked.
    """
    return _member(django_user_model, "root", group, role="lead")


@pytest.fixture
def staff(django_user_model):
    """`is_staff` and nothing else — able to reach /admin/, and that is all.

    Kept as its own fixture because the point of naming the permissions was that
    this person is *not* an administrator of experiments.
    """
    return django_user_model.objects.create_user(
        username="deskclerk", password="pw", is_staff=True)


def _granted(django_user_model, username, *codenames):
    """A user holding exactly *codenames*, granted directly.

    Re-fetched because permissions are cached on the instance the moment
    anything asks, and every one of these tests asks.
    """
    from django.contrib.auth.models import Permission

    user = django_user_model.objects.create_user(username=username, password="pw")
    user.user_permissions.set(
        Permission.objects.filter(content_type__app_label="access",
                                  codename__in=codenames))
    return django_user_model.objects.get(pk=user.pk)


def _experiment(owner=None, shared=False, name="owned") -> Experiment:
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=owner, shared=shared)


def _detail(client, exp):
    return client.get(reverse("ui:experiment_detail", args=[exp.pk]))


# ── inert without accounts ───────────────────────────────────────────────────

def test_the_default_policy_is_the_group_policy():
    """One switch, not two. REQUIRE_LOGIN turning the boundary on is the whole
    configuration; there is no second variable to remember."""
    from django.conf import settings as django_settings

    from ui.permissions import policy

    assert django_settings.EXPERIMENT_POLICY == "access.policy.GroupPolicy"
    assert type(policy()).__name__ == "GroupPolicy"


def test_the_old_policy_name_still_resolves():
    """`OwnerPolicy` is an alias now. An operator who pinned EXPERIMENT_POLICY
    to the old path in their .env should not find the app refusing to start."""
    from django.utils.module_loading import import_string

    from access.policy import GroupPolicy

    assert import_string("access.policy.OwnerPolicy") is GroupPolicy


def test_ownership_is_invisible_without_accounts(client, settings):
    """An install with no accounts behaves exactly as it did before any of this
    existed, including for rows that somehow carry an owner."""
    settings.REQUIRE_LOGIN = False
    exp = _experiment(owner=None)

    resp = _detail(client, exp)

    assert resp.status_code == 200
    assert resp.context["ownership"] is None


# ── your own ─────────────────────────────────────────────────────────────────

def test_creating_an_experiment_makes_you_its_owner(client, hosted, ana):
    from tests.conftest import DATASETS_DIR

    client.force_login(ana)
    client.post(reverse("ui:new_experiment"), {
        "name": "mine", "model_name": "Random Forest",
        "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "seed": "0",
    })

    assert Experiment.objects.get(name="mine").owner == ana


def test_you_can_do_everything_to_your_own(client, hosted, ana):
    exp = _experiment(owner=ana)
    client.force_login(ana)

    resp = _detail(client, exp)

    assert resp.status_code == 200
    assert resp.context["may"] == {"run": True, "edit": True,
                                   "delete": True, "export": True}


def test_an_experiment_you_do_not_own_is_not_there_at_all(client, hosted, ana, ben):
    """404, not 403 — otherwise the URL space reports who has what."""
    exp = _experiment(owner=ben)
    client.force_login(ana)

    assert _detail(client, exp).status_code == 404


def test_someone_elses_experiment_is_not_in_your_sidebar(client, hosted, ana, ben):
    _experiment(owner=ben, name="bens-private-work")
    client.force_login(ana)

    assert b"bens-private-work" not in client.get(reverse("ui:home")).content


# ── shared with you ──────────────────────────────────────────────────────────

def test_a_shared_experiment_can_be_read(client, hosted, ana, ben):
    exp = _experiment(owner=ben, shared=True)
    client.force_login(ana)

    assert _detail(client, exp).status_code == 200


def test_a_shared_experiment_cannot_be_run_or_changed(client, hosted, ana, ben):
    """The point of the distinction. Reading someone's results should not come
    with the ability to overwrite them."""
    exp = _experiment(owner=ben, shared=True)
    client.force_login(ana)

    assert _detail(client, exp).context["may"] == {
        "run": False, "edit": False, "delete": False, "export": True}


def test_running_someone_elses_shared_experiment_is_refused(client, hosted, ana, ben):
    """Checked at the route, not only hidden in the template — the form can be
    posted without the page that renders it."""
    exp = _experiment(owner=ben, shared=True)
    client.force_login(ana)

    resp = client.post(reverse("ui:experiment_run", args=[exp.pk]),
                       {"max_trials": "3", "optimize_metric": "accuracy"})

    assert resp.status_code == 403


def test_deleting_someone_elses_shared_experiment_is_refused(client, hosted, ana, ben):
    exp = _experiment(owner=ben, shared=True)
    client.force_login(ana)

    resp = client.post(reverse("ui:experiment_delete", args=[exp.pk]))

    assert resp.status_code == 403
    assert Experiment.objects.filter(pk=exp.pk).exists()


def test_a_shared_experiment_can_be_exported(client, hosted, ana, ben):
    """It carries only what the page already shows, so refusing would be
    theatre — and server paths are stripped from every export."""
    exp = _experiment(owner=ben, shared=True)
    client.force_login(ana)

    assert export_ihpo(client, exp.pk).status_code == 200


def test_only_the_owner_sees_the_sharing_control(client, hosted, ana, ben):
    exp = _experiment(owner=ben, shared=True)

    client.force_login(ben)
    assert reverse("ui:experiment_share", args=[exp.pk]) in _detail(client, exp).content.decode()

    client.force_login(ana)
    assert reverse("ui:experiment_share", args=[exp.pk]) not in _detail(client, exp).content.decode()


def test_sharing_can_be_turned_on_and_off(client, hosted, ana):
    exp = _experiment(owner=ana)
    client.force_login(ana)

    client.post(reverse("ui:experiment_share", args=[exp.pk]), {"shared": "on"})
    exp.refresh_from_db()
    assert exp.shared is True

    client.post(reverse("ui:experiment_share", args=[exp.pk]), {})
    exp.refresh_from_db()
    assert exp.shared is False


def test_nobody_else_can_share_your_experiment_out_from_under_you(client, hosted, ana, ben):
    exp = _experiment(owner=ben)
    client.force_login(ana)

    resp = client.post(reverse("ui:experiment_share", args=[exp.pk]), {"shared": "on"})

    assert resp.status_code == 404      # not even visible, let alone editable
    exp.refresh_from_db()
    assert exp.shared is False


# ── nobody's ─────────────────────────────────────────────────────────────────

def test_an_ownerless_experiment_belongs_to_nobody(client, hosted, ana):
    """It used to belong to *everyone*, and that was right at the time.

    The only ownerless experiments were the ones predating accounts: no owner's
    wishes were being overridden, and hiding them would have swallowed an
    operator's existing work the moment they turned the switch on. Once there
    are groups the same rule is a leak by construction — every member of every
    group would see them — so it is gone, and the migration that introduced
    groups gave the existing ones an owner rather than leaving them to this.

    What is left is a row nobody can reach, which is a state to notice rather
    than to rely on: `access/admin.py` stops an account being deleted into it.
    """
    exp = _experiment(owner=None)
    client.force_login(ana)

    assert _detail(client, exp).status_code == 404


def test_deleting_a_user_keeps_their_experiments(hosted, ana):
    """SET_NULL. Removing a person from an instance must not destroy results
    other people may be relying on; an operator reassigns them in the admin."""
    exp = _experiment(owner=ana)

    ana.delete()
    exp.refresh_from_db()

    assert exp.owner is None


# ── administrators, and the powers they are made of ─────────────────────────
#
# These used to be one flag. `is_staff` meant "may open /admin/" *and* "sees
# everyone's experiments" *and* "may act on them" — so there was no way to let
# somebody administer accounts without also handing them everyone's unpublished
# results. Each is its own permission now, and these pin them apart.


def test_an_administrator_sees_and_can_act_on_everything(client, hosted, ben,
                                                         administrator):
    """The group the migration creates carries what `is_staff` used to."""
    exp = _experiment(owner=ben)
    client.force_login(administrator)

    resp = _detail(client, exp)

    assert resp.status_code == 200
    assert all(resp.context["may"].values())


def test_is_staff_alone_grants_nothing_here(client, hosted, ben, staff):
    """The whole point of the split, and the thing a later refactor could
    quietly undo by reaching for `is_staff` again because it is nearer to hand.

    Reaching /admin/ is Django's business and is untouched; what this pins is
    that it buys nothing in codesigner.
    """
    exp = _experiment(owner=ben)
    client.force_login(staff)

    assert _detail(client, exp).status_code == 404


def test_seeing_everything_is_not_being_allowed_to_touch_it(client, hosted, ben,
                                                            django_user_model):
    """A supervisor who should read results without being able to spend compute
    or delete anything. Impossible to express before."""
    reader = _granted(django_user_model, "reader", "view_all_experiments")
    exp = _experiment(owner=ben)
    client.force_login(reader)

    resp = _detail(client, exp)

    # Reaching the page at all *is* the view check — `may` carries only the
    # actions offered on it.
    assert resp.status_code == 200, "they should be able to see it"
    assert resp.context["may"][EXPORT], "reading includes taking a copy"
    assert not resp.context["may"][RUN]
    assert not resp.context["may"][DELETE]


def test_managing_implies_seeing(client, hosted, ben, django_user_model):
    """Being able to act on a row you cannot see is not a state worth having,
    so the queryset returns everything for either permission."""
    manager = _granted(django_user_model, "manager", "manage_experiments")
    exp = _experiment(owner=ben)
    client.force_login(manager)

    resp = _detail(client, exp)

    assert resp.status_code == 200
    assert resp.context["may"][RUN]


def test_a_superuser_needs_no_grant(client, hosted, ben, django_user_model):
    """Django gives an active superuser every permission, so an operator keeps
    all of this through an upgrade without touching anything."""
    root = django_user_model.objects.create_superuser(username="su", password="pw")
    exp = _experiment(owner=ben)
    client.force_login(root)

    assert all(_detail(client, exp).context["may"].values())


# ── what leaves the instance ─────────────────────────────────────────────────

def test_an_export_names_no_paths_on_this_server(client, ana):
    """A .ihpo travels. The paths in it point at a machine that is not the
    recipient's, so they are of no use to them and describe the layout of an
    instance they may not have an account on."""
    import json

    from django.core.files.base import ContentFile

    from tests.conftest import DATASETS_DIR

    exp = _experiment(owner=ana)
    exp.dataset.save("iris.csv", ContentFile((DATASETS_DIR / "iris.csv").read_bytes()))

    body = export_ihpo(client, exp.pk).content
    snapshot = json.loads(body)

    assert snapshot["dataset"]["path"] == ""
    assert snapshot["model"]["path"] == ""


def test_the_experiment_itself_still_knows_its_paths(client, ana):
    """Blanking happens on the way out only — the detail page and the run
    engine rebuild from the same snapshot and both need the file."""
    from django.core.files.base import ContentFile

    from tests.conftest import DATASETS_DIR
    from ui.services import snapshot as adapter

    exp = _experiment(owner=ana)
    exp.dataset.save("iris.csv", ContentFile((DATASETS_DIR / "iris.csv").read_bytes()))

    assert adapter.snapshot_from_experiment(exp)["dataset"]["path"] != ""
