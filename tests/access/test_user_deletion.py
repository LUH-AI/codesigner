"""Deleting an account must not publish the work behind it.

*What:* `Experiment.owner` is `SET_NULL`, so deleting an account detaches its
experiments rather than destroying them — and a detached experiment is now
reachable by nobody at all. Neither outcome is what an administrator deleting a
departed colleague expects: the work does not go away, it goes silent, and the
only way back is the Django admin.

The guard predates groups, when the consequence was the opposite and worse —
ownerless meant *everyone's*, so deleting somebody published their unpublished
work across the instance. Groups closed that (`test_groups.py`), which changed
what this is protecting against without changing whether it is worth having.

*How:* through the Django admin, which is the only place users are deleted, and
against the policy itself to show what the deletion would otherwise have caused.
"""

import pytest
from django.urls import reverse

from ui.models import Experiment

pytestmark = pytest.mark.django_db


@pytest.fixture
def hosted(settings):
    settings.REQUIRE_LOGIN = True
    return settings


@pytest.fixture
def root(client, django_user_model):
    user = django_user_model.objects.create_superuser(
        username="root", password="pw", email="root@example.org")
    client.force_login(user)
    return user


def _experiment(owner=None, name="theirs") -> Experiment:
    return Experiment.objects.create(
        name=name, model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], seed=0, owner=owner)


def _delete_url(user):
    return reverse("admin:auth_user_delete", args=[user.pk])


# ── what the guard is for ────────────────────────────────────────────────────

def test_an_orphaned_experiment_is_reachable_by_nobody(client, hosted, root,
                                                      django_user_model):
    """The consequence being prevented, stated outright.

    Without this the guard looks like fussiness; with it, the reason it exists
    is a test rather than a comment. Before groups the failure ran the other
    way — an orphan was everyone's — and it is worth keeping a test on *which*
    way it fails, because the guard reads the same either way and the two
    problems want different explanations to the reader.
    """
    from access.policy import GroupPolicy

    leaver = django_user_model.objects.create_user(username="leaver", password="pw")
    exp = _experiment(owner=leaver)
    stranger = django_user_model.objects.create_user(username="nosy", password="pw")

    exp.owner = None            # exactly what SET_NULL would leave behind
    exp.save(update_fields=["owner"])

    request = type("R", (), {"user": stranger})()
    assert not GroupPolicy().experiments(request).filter(pk=exp.pk).exists()


# ── so the admin will not do it ──────────────────────────────────────────────

def test_deleting_someone_who_owns_experiments_is_refused(client, hosted, root,
                                                          django_user_model):
    leaver = django_user_model.objects.create_user(username="leaver", password="pw")
    _experiment(owner=leaver)

    resp = client.post(_delete_url(leaver), {"post": "yes"}, follow=True)

    assert django_user_model.objects.filter(pk=leaver.pk).exists()
    assert Experiment.objects.get().owner_id == leaver.pk


def test_the_account_page_says_why_there_is_no_delete_button(client, hosted, root,
                                                            django_user_model):
    """Otherwise the guard is silent — a missing button reads as a bug or a
    missing permission rather than a refusal with something to do instead.
    Deactivating is the ordinary way somebody leaves, and the administrator who
    has just been stopped is exactly who needs telling."""
    leaver = django_user_model.objects.create_user(username="leaver", password="pw")
    _experiment(owner=leaver)

    body = client.get(
        reverse("admin:auth_user_change", args=[leaver.pk])).content.decode()

    assert "cannot be deleted" in body
    assert "reassign" in body


def test_someone_who_owns_nothing_can_still_be_deleted(client, hosted, root,
                                                       django_user_model):
    """The guard is about the work, not about accounts."""
    nobody = django_user_model.objects.create_user(username="nobody", password="pw")

    client.post(_delete_url(nobody), {"post": "yes"}, follow=True)

    assert not django_user_model.objects.filter(pk=nobody.pk).exists()


def test_a_bulk_delete_including_an_owner_removes_nobody(client, hosted, root,
                                                        django_user_model):
    """Django's bulk delete is all-or-nothing — it asks per object and refuses
    the whole selection if any one is refused. That is its call, not ours, and
    the safe direction: nothing is half-deleted. The Experiments column on the
    list is how an administrator finds which one without opening each account.
    """
    owner = django_user_model.objects.create_user(username="owner", password="pw")
    _experiment(owner=owner)
    spare = django_user_model.objects.create_user(username="spare", password="pw")

    client.post(reverse("admin:auth_user_changelist"), {
        "action": "delete_selected", "post": "yes",
        "_selected_action": [str(owner.pk), str(spare.pk)],
    })

    assert django_user_model.objects.filter(pk=owner.pk).exists()
    assert django_user_model.objects.filter(pk=spare.pk).exists(), \
        "the whole selection is refused, not part of it"


# ── and deactivation is the way out ──────────────────────────────────────────

def test_deactivating_blocks_signing_in_and_keeps_the_work(client, hosted,
                                                           django_user_model):
    leaver = django_user_model.objects.create_user(username="leaver", password="pw")
    _experiment(owner=leaver)

    leaver.is_active = False
    leaver.save(update_fields=["is_active"])

    assert not client.login(username="leaver", password="pw")
    assert Experiment.objects.get().owner_id == leaver.pk


def test_the_list_shows_how_much_someone_owns(client, hosted, root,
                                              django_user_model):
    """Visible before anyone reaches for delete, rather than only in the
    refusal afterwards."""
    owner = django_user_model.objects.create_user(username="owner", password="pw")
    _experiment(owner=owner, name="a")
    _experiment(owner=owner, name="b")

    body = client.get(reverse("admin:auth_user_changelist")).content.decode()

    assert "Experiments" in body
