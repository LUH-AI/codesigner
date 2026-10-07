"""An uploaded .ihpo cannot name files on this machine.

A snapshot's `dataset_path` and `model_path` are strings inside a file that came
from wherever the user got it. The adapter used to read them as paths and copy
whatever they pointed at into the new experiment — so uploading an .ihpo naming
any readable server file put a copy in an experiment the uploader could then
export. Adoption is now something a caller opts into, and only the two callers
that produced the paths themselves do: the create form, and the operator's
import command.
"""

from pathlib import Path

from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from tests.conftest import DATASETS_DIR
from ui.services import snapshot as adapter


def _snapshot(**overrides):
    fields = {
        "version": "0.1.0", "name": "imported", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy"], "seed": 0,
        "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }
    fields.update(overrides)
    return fields


# ── The adapter's default ────────────────────────────────────────────────────

def test_a_snapshot_path_is_not_read_by_default():
    """Even a perfectly real, readable path is ignored: the caller has not said
    the snapshot is one whose paths can be trusted."""
    exp = adapter.experiment_from_snapshot(_snapshot())
    assert not exp.data.dataset
    assert not exp.data.demo_dataset


def test_a_caller_that_owns_the_paths_can_adopt_them(tmp_path):
    """What: a trusted caller's path is copied into the experiment.
    How: adopts a CSV that is not one of the bundled demos and reads the
    stored copy back."""
    own = tmp_path / "measurements.csv"
    own.write_bytes((DATASETS_DIR / "iris.csv").read_bytes())

    exp = adapter.experiment_from_snapshot(_snapshot(dataset_path=str(own)), adopt_paths=True)

    assert exp.data.dataset
    assert Path(exp.data.dataset.name).name.startswith("measurements")
    assert not exp.data.demo_dataset


def test_an_adopted_demo_is_named_rather_than_copied():
    """What: a trusted path that is a bundled demo is recorded by the demo's
    name, and no copy is stored. How: adopts iris' own path and reads back
    where the experiment's dataset is."""
    exp = adapter.experiment_from_snapshot(_snapshot(), adopt_paths=True)

    assert exp.data.demo_dataset == "iris"
    assert not exp.data.dataset
    assert exp.data.dataset_path().resolve() == (DATASETS_DIR / "iris.csv").resolve()


def test_an_attached_upload_is_still_adopted_without_the_flag():
    """Attaching the dataset is how an imported experiment becomes runnable, and
    it never depended on the snapshot's own path."""
    exp = adapter.experiment_from_snapshot(
        _snapshot(dataset_path="/nowhere/at/all.csv"),
        dataset_file=SimpleUploadedFile("mine.csv", b"a,b,t\n1,2,0\n"),
    )
    assert exp.data.dataset
    assert exp.data.dataset.read() == b"a,b,t\n1,2,0\n"


def test_a_model_path_is_not_read_by_default(tmp_path):
    """The model branch is the sharper one: adopting a path here means the file
    is later executed."""
    planted = tmp_path / "evil.py"
    planted.write_text("raise SystemExit\n")
    exp = adapter.experiment_from_snapshot(_snapshot(model_path=str(planted)))
    assert not exp.data.model_file


# ── Through the web importer ─────────────────────────────────────────────────

def test_uploading_an_ihpo_cannot_take_a_server_file(client, tmp_path):
    """The whole point: a file naming someone else's data yields an experiment
    with no dataset, not a copy of it."""
    import json

    secret = tmp_path / "someone_elses.csv"
    secret.write_text("a,b,target\n1,2,0\n3,4,1\n")
    ihpo = SimpleUploadedFile(
        "taken.ihpo",
        json.dumps(_snapshot(dataset_path=str(secret))).encode(),
        content_type="application/json",
    )

    resp = client.post(reverse("ui:import_experiment"), {"file": ihpo})
    assert resp.status_code == 302

    from ui.models import Experiment
    exp = Experiment.objects.get(data__name="imported")
    assert not exp.data.dataset, "the named server file was adopted"


def test_importing_with_an_attached_dataset_still_works(client):
    """The documented import flow — attach the dataset — is unaffected."""
    import json

    ihpo = SimpleUploadedFile(
        "exp.ihpo", json.dumps(_snapshot()).encode(), content_type="application/json")
    csv = SimpleUploadedFile("iris.csv", (DATASETS_DIR / "iris.csv").read_bytes())

    client.post(reverse("ui:import_experiment"), {"file": ihpo, "dataset": csv})

    from ui.models import Experiment
    assert Experiment.objects.get(data__name="imported").data.dataset
