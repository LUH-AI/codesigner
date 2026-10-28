"""A group's usage counts datasets, model files and kept trial models, and
bundled demos count for nobody. An upload past the limit evicts the stalest kept
models, and the app refuses the upload only when datasets and model files alone
exceed the limit.
"""

from pathlib import Path

import pytest
from django.urls import reverse

from access.models import Group, Membership
from experiments.models import Experiment, TrialModel
from experiments.services import storage

from tests.conftest import DATASETS_DIR, FIXTURES_DIR, post_new_experiment

pytestmark = pytest.mark.django_db

FLATS = FIXTURES_DIR / "flats.csv"


@pytest.fixture
def lab(settings, django_user_model, client):
    """`lab` enrols Ana and Ben."""
    settings.REQUIRE_LOGIN = True
    group = Group.objects.create(name="lab", user_limit=5)
    people = {}
    for name in ("ana", "ben"):
        user = django_user_model.objects.create_user(username=name, password="pw")
        Membership.objects.create(user=user, group=group, role="member")
        people[name] = user
    client.force_login(people["ana"])
    return {"group": group, **people}


def _create(client, name="flats", upload=True, **extra):
    from django.core.files.uploadedfile import SimpleUploadedFile

    data = {"name": name, "task": "classification", "model_name": "Random Forest",
            "optimizer_name": "Random Search", "seed": 0}
    if upload:
        data["dataset_file"] = SimpleUploadedFile("flats.csv", FLATS.read_bytes(),
                                                  content_type="text/csv")
    else:
        data["demo_dataset"] = str(DATASETS_DIR / "iris.csv")
    data.update(extra)
    return post_new_experiment(client, data)


def _kept(exp, trial, size, tmp_path, minutes_ago):
    """`_kept` stores a model used *minutes_ago*."""
    from datetime import timedelta

    from django.utils import timezone

    folder = tmp_path / f"kept-{exp.pk}-{trial}"
    folder.mkdir()
    (folder / "model.bin").write_bytes(b"x" * size)
    kept = TrialModel.objects.create(experiment=exp, trial=trial, directory=str(folder), bytes=size)
    TrialModel.objects.filter(pk=kept.pk).update(
        last_used_at=timezone.now() - timedelta(minutes=minutes_ago))
    return TrialModel.objects.get(pk=kept.pk)


def test_a_demo_dataset_is_named_not_copied_and_still_runs(lab, client):
    from experiments.services.snapshot import rebuild_result

    _create(client, "iris", upload=False)
    exp = Experiment.objects.get(data__name="iris")

    assert exp.data.demo_dataset == "iris" and not exp.data.dataset
    assert storage.usage(lab["group"])["total"] == 0
    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 2, "optimize_metric": "accuracy"})
    exp.refresh_from_db()
    result = rebuild_result(exp)
    assert result is not None
    assert len(result.trials) == 2


def test_the_file_names_the_demo_and_another_instance_reads_it_as_its_own(lab, client):
    """`experiment_from_snapshot` reopens a demo-backed `.ihpo` without the
    dataset file."""
    from experiments.services import snapshot as snapshot_adapter

    _create(client, "iris", upload=False)
    snapshot = snapshot_adapter.snapshot_from_experiment(Experiment.objects.get(data__name="iris"))
    assert snapshot["dataset"]["demo"] == "iris"

    snapshot["name"], snapshot["dataset"]["path"] = "reopened", ""
    reopened = snapshot_adapter.experiment_from_snapshot(snapshot)
    assert reopened.data.demo_dataset == "iris" and reopened.data.has_dataset


def test_an_upload_counts_and_purging_deletes_its_file(lab, client):
    from experiments.services import bin as bin_service

    _create(client)
    exp = Experiment.objects.get(data__name="flats")
    path = Path(exp.data.dataset.path)
    assert storage.usage(lab["group"])["fixed"] == FLATS.stat().st_size

    bin_service.purge(exp)
    assert not path.exists()
    assert storage.usage(lab["group"])["total"] == 0


def test_a_dataset_past_the_limit_replaces_the_stalest_kept_models(lab, client, tmp_path):
    """`storage` deletes the stalest kept models until the upload fits and
    reports how many."""
    _create(client, "first")
    first = Experiment.objects.get(data__name="first")
    size = FLATS.stat().st_size
    stale = _kept(first, 1, 400, tmp_path, minutes_ago=60)
    fresh = _kept(first, 2, 400, tmp_path, minutes_ago=1)
    lab["group"].storage_limit_bytes = 2 * size + 500
    lab["group"].save()

    response = _create(client, "second")

    assert response.status_code == 302
    assert Experiment.objects.filter(data__name="second").exists()
    assert not TrialModel.objects.filter(pk=stale.pk).exists()
    assert not Path(stale.directory).exists()
    assert TrialModel.objects.filter(pk=fresh.pk).exists()
    page = client.get(response["Location"]).content.decode()
    assert "1 kept trial model was deleted" in page


def test_datasets_alone_past_the_limit_are_refused(lab, client):
    """The refused upload leaves nothing behind."""
    lab["group"].storage_limit_bytes = 100
    lab["group"].save()

    response = _create(client)

    assert not Experiment.objects.filter(data__name="flats").exists()
    assert "has no room for this" in client.get(response["Location"]).content.decode()
    assert storage.usage(lab["group"])["total"] == 0


def test_every_member_is_warned_past_ninety_percent(lab, client):
    """The warning reaches members other than the uploader."""
    from django.core.cache import cache

    _create(client)
    lab["group"].storage_limit_bytes = int(FLATS.stat().st_size / 0.95)
    lab["group"].save()
    cache.clear()

    client.force_login(lab["ben"])
    page = client.get(reverse("ui:experiment_list")).content.decode()
    assert "storage-warning" in page and "lab has used 95%" in page


def test_a_site_admin_sets_a_groups_limit(lab, client, django_user_model):
    """The group page shows its usage against the limit."""
    from django.contrib.auth.models import Permission

    admin = django_user_model.objects.create_user(username="site", password="pw")
    admin.user_permissions.add(Permission.objects.get(content_type__app_label="access",
                                                      codename="manage_site"))
    client.force_login(admin)
    client.post(reverse("ui:site_group_save"), {
        "pk": lab["group"].pk, "name": "lab", "user_limit": 5, "is_active": "on",
        "storage_gb": "2.5"})
    lab["group"].refresh_from_db()
    assert lab["group"].storage_limit == int(2.5 * 1024 ** 3)

    client.force_login(lab["ana"])
    page = client.get(reverse("ui:group_detail", args=[lab["group"].pk])).content.decode()
    assert "of 2.5 GB" in page
