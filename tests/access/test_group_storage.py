"""What a group stores, its limit, and what gives way when it is full.

A group's usage is its experiments' datasets and model files plus the trial
models kept for export; the bundled demo datasets are shared and count against
nobody. Datasets come first: an upload past the limit is accepted and the
stalest kept trial models are deleted to make room, and only datasets and
model files alone past the limit refuse it. Every member is warned on every
page past 90%.
"""

from pathlib import Path

import pytest
from django.urls import reverse

from access.models import Group, Membership
from ui.models import Experiment, TrialModel
from ui.services import storage

from tests.conftest import DATASETS_DIR, FIXTURES_DIR

pytestmark = pytest.mark.django_db

FLATS = FIXTURES_DIR / "flats.csv"


@pytest.fixture
def lab(settings, django_user_model, client):
    """A group of two, Ana signed in."""
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
    return client.post(reverse("ui:new_experiment"), data)


def _kept(exp, trial, size, tmp_path, minutes_ago):
    """A kept trial model of *size* bytes, kept *minutes_ago*."""
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
    """What: choosing a demo stores its name and no file, counts nothing
    against the group, and the experiment runs on the bundled file.
    How: creates iris from the demo, reads the row and the usage, runs it."""
    from ui.views import _rebuild_result

    _create(client, "iris", upload=False)
    exp = Experiment.objects.get(data__name="iris")

    assert exp.data.demo_dataset == "iris" and not exp.data.dataset
    assert storage.usage(lab["group"])["total"] == 0
    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 2, "optimize_metric": "accuracy"})
    exp.refresh_from_db()
    assert len(_rebuild_result(exp).trials) == 2


def test_the_file_names_the_demo_and_another_instance_reads_it_as_its_own(lab, client):
    """What: the .ihpo says which demo the dataset is, and opening it again
    uses the bundled demo rather than needing the file. How: exports the
    snapshot and builds a second experiment from it without a dataset."""
    from ui.services import snapshot as snapshot_adapter

    _create(client, "iris", upload=False)
    snapshot = snapshot_adapter.snapshot_from_experiment(Experiment.objects.get(data__name="iris"))
    assert snapshot["dataset"]["demo"] == "iris"

    snapshot["name"], snapshot["dataset"]["path"] = "reopened", ""
    reopened = snapshot_adapter.experiment_from_snapshot(snapshot)
    assert reopened.data.demo_dataset == "iris" and reopened.data.has_dataset


def test_an_upload_counts_and_purging_deletes_its_file(lab, client):
    """What: an uploaded dataset counts its bytes against the group, and
    purging the experiment deletes the file and frees them.
    How: uploads flats, reads the usage, purges, checks the disk."""
    from ui.services import bin as bin_service

    _create(client)
    exp = Experiment.objects.get(data__name="flats")
    path = Path(exp.data.dataset.path)
    assert storage.usage(lab["group"])["fixed"] == FLATS.stat().st_size

    bin_service.purge(exp)
    assert not path.exists()
    assert storage.usage(lab["group"])["total"] == 0


def test_a_dataset_past_the_limit_replaces_the_stalest_kept_models(lab, client, tmp_path):
    """What: an upload that takes the group past its limit is accepted, and
    kept trial models are deleted, the stalest first, until it fits — the
    person is told how many. How: a limit just above one file plus one kept
    model, two kept models of different ages, then a second upload."""
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
    """What: when the datasets and model files alone would be over the limit
    there is nothing to make room with, so the upload is refused and nothing
    of it is left behind. How: a limit smaller than one file."""
    lab["group"].storage_limit_bytes = 100
    lab["group"].save()

    response = _create(client)

    assert not Experiment.objects.filter(data__name="flats").exists()
    assert "has no room for this" in client.get(response["Location"]).content.decode()
    assert storage.usage(lab["group"])["total"] == 0


def test_every_member_is_warned_past_ninety_percent(lab, client):
    """What: once the group uses 90% of its limit, every member sees the
    warning on every page — not only whoever uploaded. How: fills the group
    to about 95% and signs Ben in."""
    from django.core.cache import cache

    _create(client)
    lab["group"].storage_limit_bytes = int(FLATS.stat().st_size / 0.95)
    lab["group"].save()
    cache.clear()

    client.force_login(lab["ben"])
    page = client.get(reverse("ui:experiment_list")).content.decode()
    assert "storage-warning" in page and "lab has used 95%" in page


def test_a_site_admin_sets_a_groups_limit(lab, client, django_user_model):
    """What: a site admin sets the storage limit beside the seats; the group's
    page shows its usage against it. How: posts the site form, then opens the
    group page as a member."""
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
