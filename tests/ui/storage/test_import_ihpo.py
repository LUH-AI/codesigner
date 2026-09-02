"""Step 3 TDD: contracts for the import_ihpo management command.

Written before the command exists — call_command raises "Unknown command"
until Step 3 lands. The command is the CLI door into the store: it must
reuse core.io.parse (same validation, same error messages) and adopt any
referenced dataset into MEDIA_ROOT rather than trusting foreign paths.
"""

import json
import shutil

import pytest
from django.core.management import CommandError, call_command

from tests.conftest import DATASETS_DIR, FIXTURES_DIR


@pytest.mark.django_db
def test_import_creates_experiment_with_result():
    """Importing a fixture creates a row carrying the full stored result.

    The fixture's dataset path doesn't exist here, so the import must still
    succeed (datasetless row, browsable results — the read-only-load analog)
    rather than fail on a machine-specific path.
    """
    from ui.models import Experiment

    call_command("import_ihpo", str(FIXTURES_DIR / "test2.ihpo"))

    exp = Experiment.objects.get()
    original = json.loads((FIXTURES_DIR / "test2.ihpo").read_text(encoding="utf-8"))
    assert exp.name == original["name"]
    assert exp.optimizer_name == original["optimizer_name"]
    assert exp.result == original["result"]
    assert not exp.dataset


@pytest.mark.django_db
def test_import_adopts_existing_dataset_into_media(tmp_path, settings):
    """When the referenced dataset exists, it is copied under MEDIA_ROOT.

    Setup: a temp .ihpo whose dataset_path points at a real CSV, and
    MEDIA_ROOT redirected to tmp_path.
    Expect: the row has a dataset file stored below MEDIA_ROOT (the store
    owns its data; nothing may keep referencing the original path).
    """
    from ui.models import Experiment

    settings.MEDIA_ROOT = str(tmp_path / "media")
    snapshot = json.loads((FIXTURES_DIR / "test2.ihpo").read_text(encoding="utf-8"))
    snapshot["dataset_path"] = str(DATASETS_DIR / "wine.csv")
    ihpo = tmp_path / "with_dataset.ihpo"
    ihpo.write_text(json.dumps(snapshot), encoding="utf-8")

    call_command("import_ihpo", str(ihpo))

    exp = Experiment.objects.get()
    assert exp.dataset
    assert exp.dataset.path.startswith(settings.MEDIA_ROOT)
    with exp.dataset.open("rb") as stored, open(DATASETS_DIR / "wine.csv", "rb") as src:
        assert stored.read() == src.read()


@pytest.mark.django_db
def test_import_allows_duplicate_names():
    """Importing the same file twice creates two distinct experiments.

    Names are no longer unique — each experiment has its own identifier — so
    re-importing a file is allowed and yields a second, separately-identified
    row rather than an error.
    """
    from ui.models import Experiment

    call_command("import_ihpo", str(FIXTURES_DIR / "test2.ihpo"))
    call_command("import_ihpo", str(FIXTURES_DIR / "test2.ihpo"))
    exps = Experiment.objects.all()
    assert exps.count() == 2
    assert exps[0].name == exps[1].name
    assert exps[0].identifier != exps[1].identifier


@pytest.mark.django_db
def test_import_rejects_invalid_file(tmp_path):
    """A file that fails io.parse is rejected with parse's message.

    Setup: a .ihpo containing invalid JSON.
    Expect: CommandError surfacing the "not valid JSON" reason, no row
    created — the command must not have weaker validation than the app.
    """
    from ui.models import Experiment

    bad = tmp_path / "bad.ihpo"
    bad.write_bytes(b"definitely not json {")
    with pytest.raises(CommandError, match="not valid JSON"):
        call_command("import_ihpo", str(bad))
    assert Experiment.objects.count() == 0


# ── --smac: the same door, for somebody else's run ───────────────────────────
#
# *What:* the command reads a SMAC output directory as well as an .ihpo. For an
# operator with the files already on the machine, where a browser's directory
# picker is not the way in.
#
# *How:* against `tests/fixtures/smac_run`, a real 25-trial run, copied to a
# tmp_path so the nesting cases can be built around it.


@pytest.mark.django_db
def test_import_smac_directory_creates_a_read_only_experiment():
    """The run comes in with its trials, its space, and no model to run it."""
    from ui.models import Experiment

    call_command("import_ihpo", str(FIXTURES_DIR / "smac_run"), smac=True)

    exp = Experiment.objects.get()
    assert exp.name == "demo-run"
    assert len(exp.result["data"]) == 25
    assert exp.config_space is not None
    assert not exp.dataset
    assert not exp.model_file


@pytest.mark.django_db
def test_import_smac_finds_a_run_nested_below_the_named_directory(tmp_path):
    """`smac3_output/<name>/<seed>` is where SMAC actually puts them.

    Naming the top of that tree is at least as natural as naming the run
    directory, so the search is recursive.
    """
    from ui.models import Experiment

    nested = tmp_path / "smac3_output" / "demo-run" / "0"
    shutil.copytree(FIXTURES_DIR / "smac_run", nested)

    call_command("import_ihpo", str(tmp_path), smac=True)

    assert Experiment.objects.get().name == "demo-run"


@pytest.mark.django_db
def test_import_smac_refuses_two_runs_at_once(tmp_path):
    """A runhistory read against another run's config space is not a run.

    The importer matches on basenames — it has to, since a browser upload keeps
    nothing else — so two runs under one directory cannot be told apart.
    """
    from ui.models import Experiment

    for seed in ("0", "1"):
        shutil.copytree(FIXTURES_DIR / "smac_run", tmp_path / "demo-run" / seed)

    with pytest.raises(CommandError, match="more than one run"):
        call_command("import_ihpo", str(tmp_path), smac=True)
    assert Experiment.objects.count() == 0


@pytest.mark.django_db
def test_import_smac_refuses_a_directory_that_is_not_one(tmp_path):
    """With the reason, and no row — the same standard as the .ihpo door."""
    from ui.models import Experiment

    (tmp_path / "runhistory.json").write_text('{"data": []}')

    with pytest.raises(CommandError, match="configspace.json"):
        call_command("import_ihpo", str(tmp_path), smac=True)
    assert Experiment.objects.count() == 0
