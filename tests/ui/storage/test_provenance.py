"""An exported `.ihpo` should carry everything needed to recreate its
experiment, and `result` already mirrors SMAC's state. Format 2 states each
subject once, and `.io.normalize` lifts format 1 files, so they still open.
"""

import hashlib
import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from tests.conftest import DATASETS_DIR, export_ihpo, post_new_experiment
from experiments.models import Experiment, Run

pytestmark = pytest.mark.django_db

IRIS = DATASETS_DIR / "iris.csv"
WINE = DATASETS_DIR / "wine.csv"


@pytest.fixture
def no_thread(monkeypatch):
    """`no_thread` launches a run row without executing it."""
    from experiments.services import run as run_service
    monkeypatch.setattr(run_service, "start_background_run", lambda run_id: None)


def _create(client, **overrides):
    data = {
        "name": "recorded", "model_name": "Random Forest",
        "optimizer_name": "SMAC", "seed": "7",
        "evaluation_scheme": "kfold", "evaluation_value": "3",
        "demo_dataset": str(IRIS), "task": "classification",
    }
    data.update(overrides)
    post_new_experiment(client, data)
    return Experiment.objects.get(data__name=data["name"])


def _exported(client, exp) -> dict:
    return json.loads(
        export_ihpo(client, exp.pk).content)


def _reimport(client, body):
    """`_reimport` turns *body* into a fresh experiment."""
    body = {**body, "name": f"{body['name']}-again"}
    client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile("e.ihpo", json.dumps(body).encode()),
        "dataset": SimpleUploadedFile("iris.csv", IRIS.read_bytes()),
    })
    return Experiment.objects.get(data__name=body["name"])


def _run(exp, **overrides):
    """`_run` creates a finished run."""
    fields = {"primary_metric": "accuracy", "status": "done",
              "stopping": {"max_trials": 6}, "stopped_by": "max_trials",
              "trial_offset": 0, "trial_count": 6}
    fields.update(overrides)
    return Run.objects.create(experiment=exp, **fields)


# Data

def test_the_dataset_is_recorded_by_digest_not_embedded(client):
    """The digest makes the record enforceable, and the shape keeps it readable
    without the file."""
    record = _exported(client, _create(client))["dataset"]

    assert record["sha256"] == hashlib.sha256(IRIS.read_bytes()).hexdigest()
    assert record["filename"] == "iris.csv"
    assert record["rows"] == 150
    assert record["columns"] == 5
    assert record["target_column"] == record["column_names"][-1]


def test_an_experiment_with_no_dataset_records_nothing_to_check(client):
    """The section records no digest for a browsable import without a dataset,
    since a digest would flag every dataset as a mismatch."""
    exp = _create(client)
    # A demo is named, never stored.
    exp.data.demo_dataset = ""
    exp.data.save(update_fields=["demo_dataset"])

    record = _exported(client, exp)["dataset"]

    assert record["path"] == ""
    assert record.get("sha256") is None


# Model

def test_a_registry_model_is_named_and_nothing_else(client):
    """A registry model's version is the application's, already in
    `environment`."""
    record = _exported(client, _create(client))["model"]

    assert record == {"kind": "registry", "name": "Random Forest", "path": "",
                      "sha256": None, "dependencies": None,
                      "requires_python": None, "python": None,
                      "lock_sha256": None}


# Evaluation

def test_the_evaluation_scheme_is_recorded(client):
    record = _exported(client, _create(client, evaluation_value="3"))["evaluation"]

    assert record["scheme"] == "kfold"
    assert record["folds"] == 3
    assert record["test_size"] is None


def test_a_holdout_records_its_split_size(client):
    record = _exported(client, _create(client, evaluation_scheme="holdout",
                                     evaluation_value="0.2"))["evaluation"]

    assert record == {"scheme": "holdout", "folds": None,
                      "test_size": 0.2, "stratified": True,
                      "task": "classification",
                      "time_column": None, "gap": None,
                      "horizon": None, "series_column": None, "season": None}


def test_stratification_is_what_happened_not_what_was_asked_for(client):
    """Both schemes fall back when scikit-learn raises a `ValueError` for a
    continuous target, so the file records what happened."""
    rows = "\n".join(f"{i},{i * 0.37:.4f}" for i in range(40))
    continuous = SimpleUploadedFile(
        "continuous.csv", f"x,y\n{rows}\n".encode(), content_type="text/csv")

    discrete = _exported(client, _create(client, name="discrete"))["evaluation"]
    smooth = _exported(client, _create(client, name="smooth",
                                       evaluation_scheme="holdout",
                                       evaluation_value="0.2",
                                       demo_dataset="",
                                       dataset_file=continuous))["evaluation"]

    assert discrete["stratified"] is True
    assert smooth["stratified"] is False


# Optimizer

def test_the_blanks_are_answered_for_the_reader(client):
    """`defaults_used` answers blank settings from the installed SMAC's
    signatures."""
    exp = _create(client, opt_search_strategy="rf")

    record = _exported(client, exp)["optimizer"]

    assert exp.data.optimizer_params["rf_trees"] is None, "asked for nothing"
    assert record["defaults_used"]["rf_trees"] == 10, "and got SMAC's ten"
    assert record["defaults_used"]["retrain_after"] == 8


def test_the_other_strategys_blanks_stay_blank(client):
    """The record lacks `rf_trees` because `BlackBoxFacade` has no forest."""
    record = _exported(client, _create(client, opt_search_strategy="gp"))["optimizer"]

    assert "rf_trees" not in record["defaults_used"]
    assert record["defaults_used"]["gp_restarts"] == 10


def test_only_the_blanks_are_answered(client):
    """Reconstruction goes through `params`, so a later SMAC reproduces the same
    request."""
    record = _exported(client, _create(
        client, opt_search_strategy="rf", opt_rf_trees="24"))["optimizer"]

    assert record["params"]["rf_trees"] == 24, "asked for"
    assert "rf_trees" not in record["defaults_used"], "so not answered"
    assert record["params"]["rf_min_samples_leaf"] is None, "not asked for"
    assert record["defaults_used"]["rf_min_samples_leaf"] == 1, "so answered"


def test_how_the_initial_design_was_bounded_is_in_the_settings(client):
    """The initial design's fields already live in `params`, and SMAC's scenario
    in `result.optimizer_state` contains the settled number."""
    record = _exported(client, _create(
        client, opt_use_trial_cap="on", opt_trial_cap="4"))["optimizer"]

    assert record["params"]["initial_design"] == "sobol"
    assert record["params"]["use_trial_cap"] is True
    assert record["params"]["trial_cap"] == 4
    assert record["params"]["initial_points_use_max"] is False
    assert "initial_design" not in record, "not a second copy of the same five"


# Runs

def test_a_run_records_the_trials_it_produced(client):
    exp = _create(client)
    _run(exp, trial_offset=0, trial_count=6)
    _run(exp, trial_offset=6, trial_count=4)

    runs = _exported(client, exp)["runs"]

    assert [r["trial_range"] for r in runs] == [[1, 6], [7, 10]]


def test_the_settings_are_recorded_once_for_the_whole_experiment(client):
    """Per-run copies would imply the settings could differ, since they are
    fixed at creation."""
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
    """The budget equals the run's first trial plus its cap, and the file
    already records both."""
    exp = _create(client)
    _run(exp, trial_offset=6, trial_count=4, stopping={"max_trials": 4})

    record = _exported(client, exp)["runs"][0]

    assert "budget_told" not in record
    assert record["trial_range"][0] - 1 + record["stopping"]["max_trials"] == 10


# Metric change

def test_changing_the_metric_is_recorded_as_an_event(client, no_thread):
    """A metric change re-reads the whole history and makes a fitted surrogate
    start over."""
    exp = _create(client)
    exp.data.result = {"data": [{"config_id": i} for i in range(6)],
                  "primary_metric": "accuracy"}
    exp.data.current_metric = exp.data.original_metric = "accuracy"
    exp.data.save()

    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"optimize_metric": "f1", "max_trials": "4", "decision": "new"})

    events = _untimed(_exported(client, exp)["runs"][0]["events"])
    assert events == [{"kind": "metric_changed", "from": "accuracy", "to": "f1",
                       "at_trial": 6, "surrogate": "rebuilt_and_replayed"}]


def test_an_optimizer_that_fits_nothing_says_so(client, no_thread):
    """The change costs Random Search only a rescoring, since Random Search has
    no surrogate."""
    exp = _create(client, optimizer_name="Random Search")
    exp.data.result = {"data": [{"config_id": 0}], "primary_metric": "accuracy"}
    exp.data.current_metric = exp.data.original_metric = "accuracy"
    exp.data.save()

    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"optimize_metric": "f1", "max_trials": "2", "decision": "new"})

    assert _exported(client, exp)["runs"][0]["events"][0]["surrogate"] == "none"


def test_a_run_that_changes_nothing_records_no_event(client, no_thread):
    exp = _create(client)
    exp.data.result = {"data": [{"config_id": 0}], "primary_metric": "accuracy"}
    exp.data.current_metric = exp.data.original_metric = "accuracy"
    exp.data.save()

    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"optimize_metric": "accuracy", "max_trials": "2"})

    assert _exported(client, exp)["runs"][0]["events"] == []


def test_a_metric_change_that_never_ran_is_not_lost_on_retry(client, no_thread):
    """`create_run` moves `current_metric` before any trial lands, so a failed
    run leaves it pointing at an unscored metric. Comparing against the stored
    result lets the retry still record the change.
    """
    exp = _create(client)
    exp.data.result = {"data": [{"config_id": 0}], "primary_metric": "accuracy"}
    exp.data.current_metric = exp.data.original_metric = "accuracy"
    exp.data.save()

    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"optimize_metric": "f1", "max_trials": "2", "decision": "new"})
    # This mirrors `execute_run`'s except-block before any trial lands.
    exp.runs.update(status="error", error="the model would not fit")

    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"optimize_metric": "f1", "max_trials": "2"})

    expected = [{"kind": "metric_changed", "from": "accuracy", "to": "f1",
                "at_trial": 1, "surrogate": "rebuilt_and_replayed"}]
    runs = _exported(client, exp)["runs"]
    assert len(runs) == 2
    assert _untimed(runs[0]["events"]) == expected
    assert _untimed(runs[1]["events"]) == expected  # kept on the retry


# Environment

def test_the_versions_behind_the_numbers_are_recorded(client):
    """Unset settings mean the installed SMAC's defaults, so its version belongs
    in the record."""
    record = _exported(client, _create(client))["environment"]

    assert record["codesigner"] and record["python"]
    assert record["packages"]["smac"]
    assert record["packages"]["scikit-learn"]


# Digest enforcement

def test_importing_against_a_different_dataset_is_refused(client):
    """A different dataset would let new trials join a history they never
    belonged to."""
    body = _exported(client, _create(client))

    resp = client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile("e.ihpo", json.dumps(body).encode()),
        "dataset": SimpleUploadedFile("iris.csv", WINE.read_bytes()),
    })

    assert resp.status_code == 200
    assert "not the dataset the experiment was run on" in resp.content.decode()
    assert Experiment.objects.filter(data__name="recorded").count() == 1, "nothing imported"


def test_importing_against_the_recorded_dataset_is_allowed(client):
    body = _exported(client, _create(client))

    resp = client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile("e.ihpo", json.dumps(body).encode()),
        "dataset": SimpleUploadedFile("iris.csv", IRIS.read_bytes()),
    })

    assert resp.status_code == 302
    assert Experiment.objects.filter(data__name="recorded").count() == 2


def test_the_upload_survives_being_hashed(client):
    """Hashing reads the stream to the end, so saving needs it rewound."""
    body = _exported(client, _create(client))

    client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile("e.ihpo", json.dumps(body).encode()),
        "dataset": SimpleUploadedFile("iris.csv", IRIS.read_bytes()),
    })

    imported = Experiment.objects.filter(data__name="recorded").order_by("-id").first()
    assert imported is not None
    assert imported.data.dataset.size == IRIS.stat().st_size


def test_a_file_with_no_fingerprint_is_still_accepted(client):
    """Older `.ihpo` files have no digest to disagree with."""
    body = _exported(client, _create(client))
    body["dataset"].pop("sha256")

    resp = client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile("e.ihpo", json.dumps(body).encode()),
        "dataset": SimpleUploadedFile("iris.csv", WINE.read_bytes()),
    })

    assert resp.status_code == 302


def test_importing_without_a_dataset_is_not_a_refusal(client):
    """The check belongs at the moment a dataset is attached."""
    body = _exported(client, _create(client))

    resp = client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile("e.ihpo", json.dumps(body).encode()),
    })

    assert resp.status_code == 302


# Additive sections

def test_a_file_from_before_any_of_this_still_opens(client):
    """`io.normalize` lifts a format 1 file, so nothing downstream sees two
    shapes."""
    from core import io

    flat = {
        "version": "0.1.0", "name": "older", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "SMAC",
        "optimizer_params": {"search_strategy": "rf"},
        "primary_metric": "accuracy", "original_metric": "accuracy",
        "metric_names": ["accuracy"], "seed": 7, "cv_folds": 3,
        "dataset_path": str(IRIS), "result": None,
        # late format 1 sections, where it put them
        "data": {"filename": "iris.csv", "sha256": "abc"},
        "evaluation": {"scheme": "kfold", "folds": 3, "stratified": True},
        "optimizer": {"name": "SMAC", "resolved": {"rf_trees": 10}},
    }

    parsed = io.parse(json.dumps(flat).encode())

    assert parsed["format"] == io.SNAPSHOT_FORMAT
    assert parsed["name"] == "older"
    assert parsed["seed"] == 7
    assert parsed["optimizer"] == {"name": "SMAC", "params": {"search_strategy": "rf"}}
    assert parsed["metrics"] == {"names": ["accuracy"], "current": "accuracy",
                                 "original": "accuracy"}
    assert parsed["evaluation"] == {"scheme": "kfold", "folds": 3,
                                    "test_size": None, "stratified": True}
    assert parsed["dataset"]["sha256"] == "abc"
    assert parsed["dataset"]["path"] == str(IRIS)
    assert parsed["model"] == {"kind": "registry", "name": "Random Forest",
                               "path": ""}


def test_the_run_engine_does_not_pay_for_the_record(client):
    """Export alone asks for the fingerprint, which hashes the dataset."""
    from experiments.services import snapshot as adapter

    exp = _create(client)

    plain = adapter.snapshot_from_experiment(exp)
    recorded = adapter.snapshot_from_experiment(exp, provenance=True)

    assert "sha256" not in plain["dataset"]
    assert "environment" not in plain
    assert recorded["dataset"]["sha256"]
    assert recorded["environment"]


# Round trip

def test_a_run_history_survives_a_round_trip(client):
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
    """A `running` status would keep the experiment busy forever, polling a run
    that cannot report."""
    exp = _create(client)
    _run(exp, status="running", stopped_by="", trial_offset=None, trial_count=None)

    imported = _reimport(client, _exported(client, exp))

    assert imported.runs.get().status == "cancelled"
    assert imported.is_running is False


def test_who_pressed_run_is_not_carried_across(client):
    """An exporting instance's account is no account here."""
    exp = _create(client)
    _run(exp)

    body = _exported(client, exp)

    assert "started_by" not in body["runs"][0]
    assert _reimport(client, body).runs.get().started_by is None


def test_exporting_what_was_imported_gives_the_same_record(client):
    """Anything the importer drops shows up as a difference."""
    exp = _create(client)
    _run(exp, trial_offset=0, trial_count=6, trial_seconds=2.25,
         events=[{"kind": "metric_changed", "from": "accuracy", "to": "f1",
                  "at_trial": 6, "surrogate": "rebuilt_and_replayed"}])
    _run(exp, trial_offset=6, trial_count=2, status="error", stopped_by="",
         error="the model would not fit")

    once = _exported(client, exp)
    twice = _exported(client, _reimport(client, once))

    assert twice["runs"] == once["runs"]


# Reproduction

@pytest.mark.slow
@pytest.mark.parametrize("label,setup", [
    ("gaussian process, holdout", {"opt_search_strategy": "gp", "evaluation_scheme": "holdout",
                                "evaluation_value": "0.2"}),
    ("gaussian process, 3-fold", {"opt_search_strategy": "gp", "evaluation_value": "3"}),
    ("random forest, holdout", {"opt_search_strategy": "rf", "evaluation_scheme": "holdout",
                                "evaluation_value": "0.2", "opt_rf_trees": "24"}),
    ("random forest, 3-fold", {"opt_search_strategy": "rf", "evaluation_value": "3",
                               "opt_rf_trees": "24"}),
])
def test_an_experiment_recreated_from_its_file_produces_the_same_trials(
        client, label, setup):
    """The recreated experiment runs from zero and must reproduce the same
    trials in order. Both strategies and both schemes take different paths
    through the file.
    """
    original = _create(client, opt_use_trial_cap="on", opt_trial_cap="3",
                       opt_use_share_cap="", **setup)
    client.post(reverse("ui:experiment_run", args=[original.pk]),
                {"optimize_metric": "accuracy", "max_trials": "8",
                 "target_score": ""})
    original.refresh_from_db()
    assert original.data.result, f"{label}: the original never ran"

    body = _exported(client, original)
    body["result"] = None          # fresh, never resumed
    body["name"] = "recreated"
    client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile("e.ihpo", json.dumps(body).encode()),
        "dataset": SimpleUploadedFile("iris.csv", IRIS.read_bytes()),
    })
    copy = Experiment.objects.get(data__name="recreated")
    client.post(reverse("ui:experiment_run", args=[copy.pk]),
                {"optimize_metric": "accuracy", "max_trials": "8",
                 "target_score": ""})
    copy.refresh_from_db()

    def trials(exp):
        result = exp.data.result
        return [(result["configs"][str(e["config_id"])], round(e["cost"], 12))
                for e in result["data"]]

    assert copy.data.seed == original.data.seed == 7
    assert copy.data.cv_folds == original.data.cv_folds
    assert copy.data.optimizer_params == original.data.optimizer_params
    assert trials(copy) == trials(original), label


# Prior events

def _prior_event(result_metadata, offset=3):
    """`_prior_event` runs `_record_prior_event` on a stand-in result and
    returns the events."""
    from experiments.models import Experiment, Run
    from experiments.services.run import _record_prior_event

    exp = Experiment.objects.create(
        name="p", model_name="rf", optimizer_name="SMAC",
        metric_names=["accuracy"], current_metric="accuracy", seed=0)
    run = Run.objects.create(experiment=exp, primary_metric="accuracy",
                             status="running", events=[])

    _record_prior_event(run.pk, result_metadata.get("prior"), offset)
    run.refresh_from_db()
    return _untimed(run.events)


def _untimed(events):
    """`_untimed` drops the unpredictable `at` field from *events*."""
    return [{k: v for k, v in e.items() if k != "at"} for e in events]


def test_a_prior_that_reached_the_search_is_recorded(db):
    """The reader needs the weighted hyperparameters and the decay to reproduce
    the run."""
    events = _prior_event({"prior": {
        "applied": True, "key": "codesigner",
        "hyperparameters": ["lr", "max_depth"], "decay": "quadratic", "beta": 3.0}})

    assert events == [{"kind": "prior_applied", "at_trial": 3, "key": "codesigner",
                       "hyperparameters": ["lr", "max_depth"],
                       "decay": "quadratic", "beta": 3.0}]


def test_a_prior_that_was_ignored_says_so(db):
    """`apply_priors` reports and continues on every failure, so a skipped
    prior would look like a weighted run.
    """
    events = _prior_event({"prior": {
        "applied": False, "reason": "this SMAC has no acquisition weight layer"}})

    assert events == [{"kind": "prior_skipped", "at_trial": 3,
                       "reason": "this SMAC has no acquisition weight layer",
                       "rejected": False, "hyperparameters": []}]


def test_no_prior_stated_writes_no_event(db):
    """One event per run without a prior would bury the real ones."""
    assert _prior_event({}) == []
    assert _prior_event({"prior": None}) == []


def test_the_prior_event_joins_the_events_already_there(db):
    """The log is append-only, so a run records both a metric change and a
    prior."""
    from experiments.models import Experiment, Run
    from experiments.services.run import _record_prior_event

    exp = Experiment.objects.create(
        name="p", model_name="rf", optimizer_name="SMAC",
        metric_names=["accuracy"], current_metric="accuracy", seed=0)
    existing = {"kind": "metric_changed", "from": "accuracy", "to": "f1",
                "at_trial": 3, "surrogate": "rebuilt_and_replayed"}
    run = Run.objects.create(experiment=exp, primary_metric="f1",
                             status="running", events=[existing])

    _record_prior_event(run.pk, {"applied": False, "reason": "no weight layer"}, 3)
    run.refresh_from_db()

    assert run.events[0] == existing
    assert run.events[1]["kind"] == "prior_skipped"
