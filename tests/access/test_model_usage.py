"""Site-wide usage per model, and deleting an account whose experiments are
already in the bin."""

import pytest

from access.models import Group, Membership
from access.services import users as user_service
from experiments.models import Experiment, Run
from experiments.services import bin as bin_service
from experiments.services import site as site_service

pytestmark = pytest.mark.django_db

def _experiment(owner, group, model):
    return Experiment.objects.create(
        name="x", model_name=model, optimizer_name="SMAC", metric_names=["accuracy"],
        seed=0, owner=owner, group=group)


@pytest.fixture
def lab(django_user_model):
    group = Group.objects.create(name="lab", user_limit=9)
    ana = django_user_model.objects.create_user("ana", password="pw")
    Membership.objects.create(user=ana, group=group)
    return group, ana


def test_usage_is_totalled_per_model(lab):
    """What: `usage_by_model` counts each model's experiments and sums its
    runs' trials and seconds, most seconds first. How: two SVM experiments
    and one Random Forest, with runs."""
    group, ana = lab
    for model, seconds in (("SVM", 30.0), ("SVM", 50.0), ("Random Forest", 10.0)):
        exp = _experiment(ana, group, model)
        Run.objects.create(experiment=exp, primary_metric="accuracy", status="done",
                           trial_count=5, trial_seconds=seconds)

    rows = site_service.usage_by_model()

    assert [r["model"] for r in rows] == ["SVM", "Random Forest"]
    assert (rows[0]["experiments"], rows[0]["runs"], rows[0]["trials"], rows[0]["seconds"]) == (
        2, 2, 10, 80.0)
    assert rows[0]["people"] == 1 and rows[0]["groups"] == 1


def test_an_account_whose_experiments_are_binned_can_be_deleted(lab):
    """What: `delete_user` deletes an account whose only experiment is in the
    bin, which then waits in the site bin, and still raises for a live one.
    How: delete ana's experiment, then her account."""
    group, ana = lab
    exp = _experiment(ana, group, "SVM")
    live = _experiment(ana, group, "SVM")
    with pytest.raises(Exception, match="still owns experiments"):
        user_service.delete_user(ana)

    bin_service.delete(live)
    bin_service.delete(exp)
    user_service.delete_user(ana)

    assert set(site_service.trash().values_list("pk", flat=True)) == {exp.pk, live.pk}


def test_each_account_has_one_status(django_user_model):
    """What: `with_status` tells an active account, one invited with a live
    link, one whose links all lapsed, one invited to choose a password at
    first sign-in, and a deactivated one apart. How: one account of each."""
    from datetime import timedelta

    from django.utils import timezone

    from access.models import Account, SetupLink
    from access.services import setup_links

    def account(name, pending=""):
        user = django_user_model.objects.create_user(name, password=None if pending else "pw")
        Account.objects.create(user=user, password_pending=pending)
        return user

    account("act")
    setup_links.issue(account("inv", Account.SETUP_LINK), 24)
    setup_links.issue(account("exp", Account.SETUP_LINK), 24)
    SetupLink.objects.filter(user__username="exp").update(
        expires_at=timezone.now() - timedelta(hours=1))
    account("first", Account.FIRST_SIGN_IN)
    gone = account("off")
    user_service.set_active(gone, False)

    statuses = dict(user_service.with_status(django_user_model.objects)
                    .values_list("username", "status"))
    assert statuses == {"act": "active", "inv": "invited", "exp": "invite expired",
                        "first": "invited", "off": "deactivated"}
