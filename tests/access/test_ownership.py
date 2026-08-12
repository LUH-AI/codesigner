"""Who owns an experiment, and what sharing one actually grants.

Three kinds of experiment once an instance has accounts — yours, shared with
you, and nobody's — and the interesting cases are the boundaries between them.
The distinction that carries the most weight is that **sharing is an invitation
to look, not a transfer of control**: a colleague can read and export a shared
experiment, and cannot run, edit or delete it. An owner's results should not
change because someone else pressed Run.

`OwnerPolicy` is the default policy, so these also pin that it stays completely
transparent while `REQUIRE_LOGIN` is off — the same inert-by-default contract
the login wall has.
"""

import pytest
from django.urls import reverse

from ui.models import Experiment

pytestmark = pytest.mark.django_db


@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True


@pytest.fixture
def ana(django_user_model):
    return django_user_model.objects.create_user(username="ana", password="pw")


@pytest.fixture
def ben(django_user_model):
    return django_user_model.objects.create_user(username="ben", password="pw")


@pytest.fixture
def staff(django_user_model):
    return django_user_model.objects.create_user(
        username="root", password="pw", is_staff=True)


def _experiment(owner=None, shared=False, name="owned") -> Experiment:
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=owner, shared=shared)


def _detail(client, exp):
    return client.get(reverse("ui:experiment_detail", args=[exp.pk]))


# ── inert without accounts ───────────────────────────────────────────────────

def test_the_default_policy_is_the_owner_policy():
    """One switch, not two. REQUIRE_LOGIN turning ownership on is the whole
    configuration; there is no second variable to remember."""
    from django.conf import settings as django_settings

    from ui.permissions import policy

    assert django_settings.EXPERIMENT_POLICY == "access.policy.OwnerPolicy"
    assert type(policy()).__name__ == "OwnerPolicy"


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

    assert client.get(reverse("ui:experiment_export", args=[exp.pk])).status_code == 200


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

def test_an_ownerless_experiment_belongs_to_everyone(client, hosted, ana):
    """These predate accounts, so there is no owner whose wishes are being
    overridden — and hiding them would swallow an operator's existing work the
    moment they turned the switch on."""
    exp = _experiment(owner=None)
    client.force_login(ana)

    resp = _detail(client, exp)

    assert resp.status_code == 200
    assert resp.context["may"] == {"run": True, "edit": True,
                                   "delete": True, "export": True}


def test_the_page_says_an_ownerless_experiment_is_unowned(client, hosted, ana):
    exp = _experiment(owner=None)
    client.force_login(ana)

    assert "no owner" in _detail(client, exp).content.decode()


def test_deleting_a_user_keeps_their_experiments(hosted, ana):
    """SET_NULL. Removing a person from an instance must not destroy results
    other people may be relying on; an operator reassigns them in the admin."""
    exp = _experiment(owner=ana)

    ana.delete()
    exp.refresh_from_db()

    assert exp.owner is None


# ── staff ────────────────────────────────────────────────────────────────────

def test_staff_see_and_can_act_on_everything(client, hosted, ben, staff):
    exp = _experiment(owner=ben)
    client.force_login(staff)

    resp = _detail(client, exp)

    assert resp.status_code == 200
    assert all(resp.context["may"].values())


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

    body = client.get(reverse("ui:experiment_export", args=[exp.pk])).content
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
