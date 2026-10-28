"""`import_ihpo` reuses `core.io.parse` and copies a referenced dataset into
`MEDIA_ROOT` instead of trusting foreign paths.
"""

import json
import shutil

import pytest
from django.core.management import CommandError, call_command

from tests.conftest import DATASETS_DIR, FIXTURES_DIR


@pytest.mark.django_db
def test_import_creates_experiment_with_result():
    """A missing dataset path still imports as a datasetless row with browsable
    results.
    """
    from experiments.models import Experiment

    call_command("import_ihpo", str(FIXTURES_DIR / "test2.ihpo"))

    exp = Experiment.objects.get()
    original = json.loads((FIXTURES_DIR / "test2.ihpo").read_text(encoding="utf-8"))
    assert exp.name == original["name"]
    assert exp.data.optimizer_name == original["optimizer_name"]
    assert exp.data.result == original["result"]
    assert not exp.data.dataset


@pytest.mark.django_db
def test_import_adopts_existing_dataset_into_media(tmp_path, settings):
    """A referenced dataset is copied under `MEDIA_ROOT`, since the store owns
    its data.
    """
    from experiments.models import Experiment

    settings.MEDIA_ROOT = str(tmp_path / "media")
    own = tmp_path / "cellar.csv"
    shutil.copyfile(DATASETS_DIR / "wine.csv", own)
    snapshot = json.loads((FIXTURES_DIR / "test2.ihpo").read_text(encoding="utf-8"))
    snapshot["dataset_path"] = str(own)
    ihpo = tmp_path / "with_dataset.ihpo"
    ihpo.write_text(json.dumps(snapshot), encoding="utf-8")

    call_command("import_ihpo", str(ihpo))

    exp = Experiment.objects.get()
    assert exp.data.dataset
    assert exp.data.dataset.path.startswith(settings.MEDIA_ROOT)
    with exp.data.dataset.open("rb") as stored, open(own, "rb") as src:
        assert stored.read() == src.read()


@pytest.mark.django_db
def test_import_names_a_bundled_demo_rather_than_copying_it(tmp_path):
    from experiments.models import Experiment

    snapshot = json.loads((FIXTURES_DIR / "test2.ihpo").read_text(encoding="utf-8"))
    snapshot["dataset_path"] = str(DATASETS_DIR / "wine.csv")
    ihpo = tmp_path / "with_demo.ihpo"
    ihpo.write_text(json.dumps(snapshot), encoding="utf-8")

    call_command("import_ihpo", str(ihpo))

    exp = Experiment.objects.get()
    assert exp.data.demo_dataset == "wine"
    assert not exp.data.dataset


@pytest.mark.django_db
def test_import_allows_duplicate_names():
    """Each copy gets its own identifier.
    """
    from experiments.models import Experiment

    call_command("import_ihpo", str(FIXTURES_DIR / "test2.ihpo"))
    call_command("import_ihpo", str(FIXTURES_DIR / "test2.ihpo"))
    exps = Experiment.objects.all()
    assert exps.count() == 2
    assert exps[0].name == exps[1].name
    assert exps[0].identifier != exps[1].identifier


@pytest.mark.django_db
def test_import_rejects_invalid_file(tmp_path):
    """The command validates as strictly as the app, surfacing the parse error.
    """
    from experiments.models import Experiment

    bad = tmp_path / "bad.ihpo"
    bad.write_bytes(b"definitely not json {")
    with pytest.raises(CommandError, match="not valid JSON"):
        call_command("import_ihpo", str(bad))
    assert Experiment.objects.count() == 0


# The `--smac` flag reads a SMAC output directory, for an operator whose files
# already sit on the machine.


@pytest.mark.django_db
def test_import_smac_directory_creates_a_read_only_experiment():
    from experiments.models import Experiment

    call_command("import_ihpo", str(FIXTURES_DIR / "smac_run"), smac=True)

    exp = Experiment.objects.get()
    assert exp.name == "demo-run"
    assert exp.data.result is not None
    assert len(exp.data.result["data"]) == 25
    assert exp.data.config_space is not None
    assert not exp.data.dataset
    assert not exp.data.model_file


@pytest.mark.django_db
def test_import_smac_finds_a_run_nested_below_the_named_directory(tmp_path):
    """SMAC writes runs to `smac3_output/<name>/<seed>`, so the search is
    recursive.
    """
    from experiments.models import Experiment

    nested = tmp_path / "smac3_output" / "demo-run" / "0"
    shutil.copytree(FIXTURES_DIR / "smac_run", nested)

    call_command("import_ihpo", str(tmp_path), smac=True)

    assert Experiment.objects.get().name == "demo-run"


@pytest.mark.django_db
def test_import_smac_refuses_two_runs_at_once(tmp_path):
    """The importer matches on basenames, so two runs under one directory are
    indistinguishable.
    """
    from experiments.models import Experiment

    for seed in ("0", "1"):
        shutil.copytree(FIXTURES_DIR / "smac_run", tmp_path / "demo-run" / seed)

    with pytest.raises(CommandError, match="more than one run"):
        call_command("import_ihpo", str(tmp_path), smac=True)
    assert Experiment.objects.count() == 0


@pytest.mark.django_db
def test_import_smac_refuses_a_directory_that_is_not_one(tmp_path):
    from experiments.models import Experiment

    (tmp_path / "runhistory.json").write_text('{"data": []}')

    with pytest.raises(CommandError, match="configspace.json"):
        call_command("import_ihpo", str(tmp_path), smac=True)
    assert Experiment.objects.count() == 0


@pytest.mark.django_db
def test_the_command_prepares_the_model_environment_as_the_page_does(monkeypatch):
    """A command-line import prepares the model environment like the page."""
    from experiments.models import Experiment
    from experiments.services import modelenv

    prepared = []
    monkeypatch.setattr(modelenv, "start_preparation", lambda exp, *a, **k: prepared.append(exp.pk))

    call_command("import_ihpo", str(FIXTURES_DIR / "test2.ihpo"))

    assert prepared == [Experiment.objects.latest("pk").pk]
