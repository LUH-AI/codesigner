"""What an exported `.ihpo` says about how its experiment was made.

The standing goal for this format is that a file should carry everything needed
to recreate an experiment — the seed, how the initial points were collected and
how many, every optimizer setting, the validation scheme, and the history of
what changed mid-experiment. For SMAC's own state it already did: `result`
mirrors the runhistory and `result.optimizer_state` embeds the other four files
verbatim. These pin the sections that describe everything *around* the
optimizer.

Format 2 gives each subject one object and states it once. Reading stays
backward-compatible — a format 1 file is lifted by `io.normalize` rather than
refused, which the last test in this module pins directly, because those files
are on other people's disks and there is nothing wrong with them.
"""

import hashlib
import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from tests.conftest import DATASETS_DIR
from ui.models import Experiment, Run

pytestmark = pytest.mark.django_db

IRIS = DATASETS_DIR / "iris.csv"
WINE = DATASETS_DIR / "wine.csv"


@pytest.fixture
def no_thread(monkeypatch):
    """Launch a run without executing it: the row is what these read."""
    from ui.services import run as run_service
    monkeypatch.setattr(run_service, "start_background_run", lambda run_id: None)


def _create(client, **overrides):
    data = {
        "name": "recorded", "model_name": "Random Forest",
        "optimizer_name": "SMAC", "seed": "7", "cv_folds": "3",
        "demo_dataset": str(IRIS),
    }
    data.update(overrides)
    client.post(reverse("ui:new_experiment"), data)
    return Experiment.objects.get(name=data["name"])


def _exported(client, exp) -> dict:
    return json.loads(
        client.get(reverse("ui:experiment_export", args=[exp.pk])).content)


def _reimport(client, body):
    """*body* imported as a fresh experiment, and that experiment."""
    body = {**body, "name": f"{body['name']}-again"}
    client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile("e.ihpo", json.dumps(body).encode()),
        "dataset": SimpleUploadedFile("iris.csv", IRIS.read_bytes()),
    })
    return Experiment.objects.get(name=body["name"])


def _run(exp, **overrides):
    """A finished run row, without executing anything."""
    fields = {"primary_metric": "accuracy", "status": "done",
              "stopping": {"max_trials": 6}, "stopped_by": "max_trials",
              "trial_offset": 0, "trial_count": 6}
    fields.update(overrides)
    return Run.objects.create(experiment=exp, **fields)


# ── the data ─────────────────────────────────────────────────────────────────

def test_the_dataset_is_recorded_by_digest_not_embedded(client):
    """A record, not an archive. The digest is what makes it enforceable; the
    shape is what makes it readable without the file to hand."""
    record = _exported(client, _create(client))["dataset"]

    assert record["sha256"] == hashlib.sha256(IRIS.read_bytes()).hexdigest()
    assert record["filename"] == "iris.csv"
    assert record["rows"] == 150
    assert record["columns"] == 5
    assert record["target_column"] == record["column_names"][-1]


def test_an_experiment_with_no_dataset_records_nothing_to_check(client):
    """Importing without one is supported — the experiment is browsable and not
    runnable — so the section has to be able to say there is nothing to
    recognise, without a digest that would refuse every dataset offered."""
    exp = _create(client)
    exp.dataset.delete(save=True)

    record = _exported(client, exp)["dataset"]

    assert record["path"] == ""
    assert record.get("sha256") is None


# ── the model ────────────────────────────────────────────────────────────────

def test_a_registry_model_is_named_and_nothing_else(client):
    """It ships with the application, so its version is the application's —
    already in `environment`."""
    record = _exported(client, _create(client))["model"]

    assert record == {"kind": "registry", "name": "Random Forest", "path": "",
                      "sha256": None, "dependencies": None,
                      "requires_python": None, "python": None,
                      "lock_sha256": None}


# ── how a trial was evaluated ────────────────────────────────────────────────

def test_the_evaluation_scheme_is_recorded(client):
    record = _exported(client, _create(client, cv_folds="3"))["evaluation"]

    assert record["scheme"] == "kfold"
    assert record["folds"] == 3
    assert record["test_size"] is None


def test_a_holdout_records_its_split_size(client):
    record = _exported(client, _create(client, cv_folds="0"))["evaluation"]

    assert record == {"scheme": "holdout", "folds": None,
                      "test_size": 0.2, "stratified": True}


def test_stratification_is_what_happened_not_what_was_asked_for(client):
    """Both schemes ask for it and fall back when scikit-learn refuses the
    target, which in practice means a continuous one. Recording the request
    would put a claim in the file that the run did not honour."""
    rows = "\n".join(f"{i},{i * 0.37:.4f}" for i in range(40))
    continuous = SimpleUploadedFile(
        "continuous.csv", f"x,y\n{rows}\n".encode(), content_type="text/csv")

    discrete = _exported(client, _create(client, name="discrete"))["evaluation"]
    smooth = _exported(client, _create(client, name="smooth", cv_folds="0",
                                       demo_dataset="",
                                       dataset_file=continuous))["evaluation"]

    assert discrete["stratified"] is True
    assert smooth["stratified"] is False


# ── the optimizer ────────────────────────────────────────────────────────────

def test_the_blanks_are_answered_for_the_reader(client):
    """A blank setting means "whatever that component already does", which is
    the right thing to store and useless to read six months later.
    `defaults_used` answers them from the installed SMAC's own signatures — and
    the version that answered is in `environment.packages`."""
    exp = _create(client, opt_search_strategy="rf")

    record = _exported(client, exp)["optimizer"]

    assert exp.optimizer_params["rf_trees"] is None, "asked for nothing"
    assert record["defaults_used"]["rf_trees"] == 10, "and got SMAC's ten"
    assert record["defaults_used"]["retrain_after"] == 8


def test_the_other_strategys_blanks_stay_blank(client):
    """`BlackBoxFacade` has no forest to have a tree count of. A number here
    would be invented, so the key is simply absent."""
    record = _exported(client, _create(client, opt_search_strategy="gp"))["optimizer"]

    assert "rf_trees" not in record["defaults_used"]
    assert record["defaults_used"]["gp_restarts"] == 10


def test_only_the_blanks_are_answered(client):
    """`defaults_used` is the reader's copy of what the blanks became, and only
    of those. Reconstruction goes through `params`, so a SMAC that changes a
    default later reproduces the same *request* rather than freezing today's
    answer to it — and repeating the settings that were chosen would say the
    same thing twice."""
    record = _exported(client, _create(
        client, opt_search_strategy="rf", opt_rf_trees="24"))["optimizer"]

    assert record["params"]["rf_trees"] == 24, "asked for"
    assert "rf_trees" not in record["defaults_used"], "so not answered"
    assert record["params"]["rf_min_samples_leaf"] is None, "not asked for"
    assert record["defaults_used"]["rf_min_samples_leaf"] == 1, "so answered"


def test_how_the_initial_design_was_bounded_is_in_the_settings(client):
    """It has no section of its own. Every field one would hold is a setting
    already recorded under `params`, and the number they produce depends on the
    budget each run was given — so it belongs to the run, not here. SMAC records
    the number it settled on in the scenario it saves, which
    `result.optimizer_state` already embeds verbatim."""
    record = _exported(client, _create(
        client, opt_use_trial_cap="on", opt_trial_cap="4"))["optimizer"]

    assert record["params"]["initial_design"] == "sobol"
    assert record["params"]["use_trial_cap"] is True
    assert record["params"]["trial_cap"] == 4
    assert record["params"]["initial_points_use_max"] is False
    assert "initial_design" not in record, "not a second copy of the same five"


# ── the runs ─────────────────────────────────────────────────────────────────

def test_a_run_records_the_trials_it_produced(client):
    """What turns a flat list of trials back into a history: which run produced
    which of them."""
    exp = _create(client)
    _run(exp, trial_offset=0, trial_count=6)
    _run(exp, trial_offset=6, trial_count=4)

    runs = _exported(client, exp)["runs"]

    assert [r["trial_range"] for r in runs] == [[1, 6], [7, 10]]


def test_the_settings_are_recorded_once_for_the_whole_experiment(client):
    """The search settings are chosen at creation and fixed for the
    experiment's life, so they describe every trial in it. Repeating them per
    run would be the same values written once per run, and a reader finding two
    copies would reasonably assume they could differ."""
    exp = _create(client, opt_search_strategy="rf", opt_rf_trees="24")
    _run(exp)
    _run(exp)

    body = _exported(client, exp)

    assert body["optimizer"]["params"]["rf_trees"] == 24
    assert all("optimizer_params" not in run for run in body["runs"])
    assert all("params" not in run for run in body["runs"])


def test_a_run_records_what_bounded_it_and_what_ended_it(client):
    exp = _create(client)
    _run(exp, stopping={"max_trials": 6, "target_score": 0.99},
         stopped_by="target_score")

    record = _exported(client, exp)["runs"][0]

    assert record["stopping"] == {"max_trials": 6, "target_score": 0.99}
    assert record["stopped_by"] == "target_score"


def test_the_budget_smac_was_told_is_derivable_not_repeated(client):
    """It is what sized the initial design and it is not the trial cap once a
    run resumes — but it is the run's first trial plus its cap, both already
    recorded, so writing it as well would be a third copy of the same fact."""
    exp = _create(client)
    _run(exp, trial_offset=6, trial_count=4, stopping={"max_trials": 4})

    record = _exported(client, exp)["runs"][0]

    assert "budget_told" not in record
    assert record["trial_range"][0] - 1 + record["stopping"]["max_trials"] == 10


# ── the metric change ────────────────────────────────────────────────────────

def test_changing_the_metric_is_recorded_as_an_event(client, no_thread):
    """The case the format could not previously describe. It is worth recording
    for what it costs: the whole history is re-read under the new metric, and an
    optimizer carrying a fitted model of the objective throws it away, because
    it was fitted to costs from a different question."""
    exp = _create(client)
    exp.result = {"data": [{"config_id": i} for i in range(6)]}
    exp.primary_metric = exp.original_metric = "accuracy"
    exp.save()

    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"optimize_metric": "f1", "max_trials": "4", "decision": "new"})

    events = _exported(client, exp)["runs"][0]["events"]
    assert events == [{"kind": "metric_changed", "from": "accuracy", "to": "f1",
                       "at_trial": 6, "surrogate": "rebuilt_and_replayed"}]


def test_an_optimizer_that_fits_nothing_says_so(client, no_thread):
    """Random Search has no surrogate to invalidate, so the same change costs it
    a rescoring and nothing else. Recording "rebuilt" would be a claim about
    work that never happened."""
    exp = _create(client, optimizer_name="Random Search")
    exp.result = {"data": [{"config_id": 0}]}
    exp.primary_metric = exp.original_metric = "accuracy"
    exp.save()

    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"optimize_metric": "f1", "max_trials": "2", "decision": "new"})

    assert _exported(client, exp)["runs"][0]["events"][0]["surrogate"] == "none"


def test_a_run_that_changes_nothing_records_no_event(client, no_thread):
    exp = _create(client)
    exp.result = {"data": [{"config_id": 0}]}
    exp.primary_metric = exp.original_metric = "accuracy"
    exp.save()

    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"optimize_metric": "accuracy", "max_trials": "2"})

    assert _exported(client, exp)["runs"][0]["events"] == []


# ── the environment ──────────────────────────────────────────────────────────

def test_the_versions_behind_the_numbers_are_recorded(client):
    """An unset setting means whichever default the installed SMAC had, so which
    SMAC that was is part of the record rather than trivia."""
    record = _exported(client, _create(client))["environment"]

    assert record["codesigner"] and record["python"]
    assert record["packages"]["smac"]
    assert record["packages"]["scikit-learn"]


# ── the digest is enforced, not decorative ───────────────────────────────────

def test_importing_against_a_different_dataset_is_refused(client):
    """The one way an experiment could go on adding trials to a history they do
    not belong to. Nothing downstream could tell: every new trial would look
    exactly as valid as the ones before it."""
    body = _exported(client, _create(client))

    resp = client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile("e.ihpo", json.dumps(body).encode()),
        "dataset": SimpleUploadedFile("iris.csv", WINE.read_bytes()),
    })

    assert resp.status_code == 200
    assert "not the dataset the experiment was run on" in resp.content.decode()
    assert Experiment.objects.filter(name="recorded").count() == 1, "nothing imported"


def test_importing_against_the_recorded_dataset_is_allowed(client):
    body = _exported(client, _create(client))

    resp = client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile("e.ihpo", json.dumps(body).encode()),
        "dataset": SimpleUploadedFile("iris.csv", IRIS.read_bytes()),
    })

    assert resp.status_code == 302
    assert Experiment.objects.filter(name="recorded").count() == 2


def test_the_upload_survives_being_hashed(client):
    """Digesting reads the stream to the end. Whoever saves it next needs it
    back at the start, or the experiment gets an empty dataset."""
    body = _exported(client, _create(client))

    client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile("e.ihpo", json.dumps(body).encode()),
        "dataset": SimpleUploadedFile("iris.csv", IRIS.read_bytes()),
    })

    imported = Experiment.objects.filter(name="recorded").order_by("-id").first()
    assert imported.dataset.size == IRIS.stat().st_size


def test_a_file_with_no_fingerprint_is_still_accepted(client):
    """Every `.ihpo` exported before this existed. There is nothing to disagree
    with, and refusing them would make the section a breaking change."""
    body = _exported(client, _create(client))
    body["dataset"].pop("sha256")

    resp = client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile("e.ihpo", json.dumps(body).encode()),
        "dataset": SimpleUploadedFile("iris.csv", WINE.read_bytes()),
    })

    assert resp.status_code == 302


def test_importing_without_a_dataset_is_not_a_refusal(client):
    """Browsable and not runnable is what an import has always been able to be;
    the check belongs at the moment a dataset is attached."""
    body = _exported(client, _create(client))

    resp = client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile("e.ihpo", json.dumps(body).encode()),
    })

    assert resp.status_code == 302


# ── the sections are additive ────────────────────────────────────────────────

def test_a_file_from_before_any_of_this_still_opens(client):
    """A format 1 file — the flat namespace, with or without the provenance
    sections that were added beside it. `io.normalize` lifts it on the way in,
    so nothing downstream knows there were ever two shapes."""
    from core import io

    flat = {
        "version": "0.1.0", "name": "older", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "SMAC",
        "optimizer_params": {"search_strategy": "rf"},
        "primary_metric": "accuracy", "original_metric": "accuracy",
        "metric_names": ["accuracy"], "seed": 7, "cv_folds": 3,
        "dataset_path": str(IRIS), "result": None,
        # The sections format 1 grew late, in the places it put them.
        "data": {"filename": "iris.csv", "sha256": "abc"},
        "evaluation": {"scheme": "kfold", "folds": 3, "stratified": True},
        "optimizer": {"name": "SMAC", "resolved": {"rf_trees": 10}},
    }

    parsed = io.parse(json.dumps(flat).encode())

    assert parsed["format"] == io.SNAPSHOT_FORMAT
    assert parsed["name"] == "older"
    assert parsed["seed"] == 7
    assert parsed["optimizer"] == {"name": "SMAC", "params": {"search_strategy": "rf"}}
    assert parsed["metrics"] == {"names": ["accuracy"], "primary": "accuracy",
                                 "original": "accuracy"}
    assert parsed["evaluation"] == {"scheme": "kfold", "folds": 3,
                                    "test_size": None, "stratified": True}
    assert parsed["dataset"]["sha256"] == "abc"
    assert parsed["dataset"]["path"] == str(IRIS)
    assert parsed["model"] == {"kind": "registry", "name": "Random Forest",
                               "path": ""}


def test_the_run_engine_does_not_pay_for_the_record(client):
    """It rebuilds through the same function on every run, and the fingerprint
    reads and hashes the dataset. Export is the only caller that asks."""
    from ui.services import snapshot as adapter

    exp = _create(client)

    plain = adapter.snapshot_from_experiment(exp)
    recorded = adapter.snapshot_from_experiment(exp, provenance=True)

    assert "sha256" not in plain["dataset"]
    assert "environment" not in plain
    assert recorded["dataset"]["sha256"]
    assert recorded["environment"]


# ── the record survives coming back in ──────────────────────────────────────

def test_a_run_history_survives_a_round_trip(client):
    """`runs[]` was written and never read: an imported experiment kept its
    trials and lost which run produced which of them, what bounded each one and
    what settings it ran under — all of it in the file, all of it dropped."""
    exp = _create(client)
    _run(exp, trial_offset=0, trial_count=6, stopping={"max_trials": 6},
         stopped_by="max_trials", trial_seconds=1.5)
    _run(exp, trial_offset=6, trial_count=4, primary_metric="f1",
         stopping={"max_trials": 4}, stopped_by="target_score")

    imported = _reimport(client, _exported(client, exp))
    runs = list(imported.runs.order_by("id"))

    assert [(r.trial_offset, r.trial_count) for r in runs] == [(0, 6), (6, 4)]
    assert [r.primary_metric for r in runs] == ["accuracy", "f1"]
    assert [r.stopped_by for r in runs] == ["max_trials", "target_score"]
    assert runs[0].trial_seconds == 1.5


def test_a_run_that_had_not_finished_comes_back_interrupted(client):
    """A file exported mid-run records `status="running"`. Imported as-is it
    would leave the experiment permanently busy — `Experiment.is_running` reads
    exactly this — and the page would poll a run that cannot report."""
    exp = _create(client)
    _run(exp, status="running", stopped_by="", trial_offset=None, trial_count=None)

    imported = _reimport(client, _exported(client, exp))

    assert imported.runs.get().status == "cancelled"
    assert imported.is_running is False


def test_who_pressed_run_is_not_carried_across(client):
    """An account on the instance that exported this is not an account here, and
    inventing a local one would put a name against work they did not do."""
    exp = _create(client)
    _run(exp)

    body = _exported(client, exp)

    assert "started_by" not in body["runs"][0]
    assert _reimport(client, body).runs.get().started_by is None


def test_exporting_what_was_imported_gives_the_same_record(client):
    """The property the whole section exists for. Anything the importer drops
    shows up here as a difference, which is how `runs[]` being write-only would
    have been caught."""
    exp = _create(client)
    _run(exp, trial_offset=0, trial_count=6, trial_seconds=2.25,
         events=[{"kind": "metric_changed", "from": "accuracy", "to": "f1",
                  "at_trial": 6, "surrogate": "rebuilt_and_replayed"}])
    _run(exp, trial_offset=6, trial_count=2, status="error", stopped_by="",
         error="the model would not fit")

    once = _exported(client, exp)
    twice = _exported(client, _reimport(client, once))

    assert twice["runs"] == once["runs"]


# ── the whole point ──────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("label,setup", [
    ("gaussian process, holdout", {"opt_search_strategy": "gp", "cv_folds": "0"}),
    ("gaussian process, 3-fold", {"opt_search_strategy": "gp", "cv_folds": "3"}),
    ("random forest, holdout", {"opt_search_strategy": "rf", "cv_folds": "0",
                                "opt_rf_trees": "24"}),
    ("random forest, 3-fold", {"opt_search_strategy": "rf", "cv_folds": "3",
                               "opt_rf_trees": "24"}),
])
def test_an_experiment_recreated_from_its_file_produces_the_same_trials(
        client, label, setup):
    """The test the format exists for.

    Run an experiment, export it, import the file with the same dataset into a
    fresh experiment, run that — and get the same trials in the same order. If
    any of what the file records is wrong or missing, this is where it shows:
    the seed, the evaluation scheme, every optimizer setting, and the budget the
    search was given all have to come back exactly.

    Over both search strategies and both evaluation schemes, because they are
    four different paths through the same file: the strategies build different
    surrogates from different settings, and the schemes divide the data
    differently. One of them passing says little about the others.

    The second experiment is run from zero rather than resumed, so nothing is
    carried across in the result — only what the file said.
    """
    original = _create(client, opt_use_trial_cap="on", opt_trial_cap="3",
                       opt_use_share_cap="", **setup)
    client.post(reverse("ui:experiment_run", args=[original.pk]),
                {"optimize_metric": "accuracy", "max_trials": "8",
                 "target_score": ""})
    original.refresh_from_db()
    assert original.result, f"{label}: the original never ran"

    body = _exported(client, original)
    body["result"] = None          # a fresh experiment, not a resumed one
    body["name"] = "recreated"
    client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile("e.ihpo", json.dumps(body).encode()),
        "dataset": SimpleUploadedFile("iris.csv", IRIS.read_bytes()),
    })
    copy = Experiment.objects.get(name="recreated")
    client.post(reverse("ui:experiment_run", args=[copy.pk]),
                {"optimize_metric": "accuracy", "max_trials": "8",
                 "target_score": ""})
    copy.refresh_from_db()

    def trials(exp):
        result = exp.result
        return [(result["configs"][str(e["config_id"])], round(e["cost"], 12))
                for e in result["data"]]

    assert copy.seed == original.seed == 7
    assert copy.cv_folds == original.cv_folds
    assert copy.optimizer_params == original.optimizer_params
    assert trials(copy) == trials(original), label
