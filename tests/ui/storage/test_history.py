"""The experiment's history: each change to how it searches, recorded where it
is made, carried in the file, and with its times in the form asked for."""

import json
import re

import pytest
from django.contrib.admin.sites import AdminSite
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import RequestFactory
from django.urls import reverse

from tests.conftest import DATASETS_DIR, export_ihpo, post_new_experiment

pytestmark = pytest.mark.django_db

ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")


def _trials(n, metrics=("accuracy",), start=1_700_000_000.0):
    return {
        "stats": {"submitted": n, "finished": n, "running": 0},
        "data": [{"config_id": i + 1, "cost": 0.2, "time": 1.0,
                  "starttime": start + 10 * i, "endtime": start + 1.0 + 10 * i,
                  "scores": {m: 0.8 for m in metrics}, "incumbent_config_id": 1}
                 for i in range(n)],
        "configs": {str(i + 1): {"max_depth": 5 + i} for i in range(n)},
        "config_origins": {}, "optimizer_state": {}, "primary_metric": "accuracy",
        "best_score": 0.8, "best_config_id": "1",
    }


def _create(client, name="recorded"):
    from ui.models import Experiment

    post_new_experiment(client, {
        "name": name, "model_name": "Random Forest", "optimizer_name": "Random Search",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "task": "classification",
        "evaluation_scheme": "holdout", "evaluation_value": 0.2, "seed": 0})
    return Experiment.objects.get(data__name=name)


def _kinds(exp):
    return [e.kind for e in exp.data.history.order_by("at", "id")]


def _import(client, body, name):
    from ui.models import Experiment

    body = {**body, "name": name}
    client.post(reverse("ui:import_experiment"), {
        "file": SimpleUploadedFile(f"{name}.ihpo", json.dumps(body).encode())})
    return Experiment.objects.get(data__name=name)


# ── where it is recorded ─────────────────────────────────────────────────────

def test_creating_an_experiment_records_how_it_was_set_up(client):
    """What: the first entry holds everything that decides the search, and
    not where the dataset lives. How: creates one and reads the entry."""
    exp = _create(client)
    (entry,) = exp.data.history.all()
    configuration = entry.payload["configuration"]

    assert entry.kind == "created" and entry.at is not None
    assert configuration["model"]["name"] == "Random Forest"
    assert configuration["evaluation"]["task"] == "classification"
    assert "path" not in configuration["dataset"]


def test_importing_records_the_import_after_the_files_own_history(client):
    """What: an imported experiment keeps the history its file carried and
    adds that it was imported here. How: exports one and imports it."""
    exp = _create(client)
    body = json.loads(export_ihpo(client, exp.pk).content)
    again = _import(client, body, "again")

    assert _kinds(again) == ["created", "imported"]
    assert again.data.history.last().payload["source"] == "ihpo"


def test_a_processing_change_is_one_entry_per_choice_changed(client):
    """What: changing two Data Handling choices records two entries, each with
    the value before and after. How: posts the Features page."""
    exp = _create(client)
    client.post(reverse("datahandling:data_section", args=[exp.pk, "features"]),
                {"labels": "categories", "scale": "standardize"})

    changes = {e.payload["step"]: (e.payload["old"], e.payload["new"])
               for e in exp.data.history.filter(kind="processing_changed")}
    assert changes == {"labels": ("auto", "categories"), "scale": ("auto", "standardize")}


def test_processing_cannot_change_while_a_run_is_under_way(client):
    """What: a run that has not written a trial yet still fixes the columns.
    How: creates a pending run, then posts a change."""
    from ui.services.run import create_run

    exp = _create(client)
    create_run(exp, {"max_trials": 3}, "accuracy")
    client.post(reverse("datahandling:data_section", args=[exp.pk, "features"]),
                {"labels": "categories"})
    exp.data.refresh_from_db()

    assert exp.data.processing.get("labels", "auto") == "auto"
    assert not exp.data.history.filter(kind="processing_changed").exists()


def _state(client, exp, hp, prior):
    client.post(reverse("ui:save_prior", args=[exp.pk]),
                json.dumps({"hp": hp, "prior": prior}), content_type="application/json")


def test_dragging_a_prior_is_one_change_of_mind(client):
    """What: a prior saved over and over — as dragging its curve does — is one
    entry holding where it started and where it ended; ending where it began
    leaves none. How: states, edits and withdraws a prior with no run between."""
    exp = _create(client)
    normal = {"kind": "normal", "params": {"mu": 0.5, "sigma": 0.1}}

    _state(client, exp, "max_depth", normal)
    _state(client, exp, "max_depth", {**normal, "params": {"mu": 0.6, "sigma": 0.1}})
    _state(client, exp, "max_depth", {**normal, "params": {"mu": 0.7, "sigma": 0.1}})
    (entry,) = exp.data.history.filter(kind__startswith="prior")
    assert entry.kind == "prior_stated" and entry.payload["old"] is None
    assert entry.payload["new"]["params"]["mu"] == 0.7

    _state(client, exp, "max_depth", None)
    assert not exp.data.history.filter(kind__startswith="prior").exists()


def test_a_run_between_two_edits_keeps_them_apart(client):
    """What: the prior a run searched under is its own entry; an edit after
    the run starts a new one. How: states a prior, starts a run, edits it."""
    from ui.services.run import create_run

    exp = _create(client)
    normal = {"kind": "normal", "params": {"mu": 0.5, "sigma": 0.1}}
    _state(client, exp, "max_depth", normal)
    create_run(exp, {"max_trials": 3}, "accuracy")
    _state(client, exp, "max_depth", {**normal, "params": {"mu": 0.8, "sigma": 0.1}})

    assert _kinds(exp)[-2:] == ["prior_stated", "prior_edited"]


def test_resetting_a_prior_is_recorded(client):
    """What: putting a prior back to what the last run searched under is an
    entry of its own. How: states a prior with no run behind it and resets."""
    exp = _create(client)
    _state(client, exp, "max_depth", {"kind": "normal", "params": {"mu": 0.5, "sigma": 0.1}})
    client.post(reverse("ui:reset_prior", args=[exp.pk]), {"hp": "max_depth"})

    entry = exp.data.history.order_by("-id").first()
    assert entry.kind == "prior_reset" and entry.payload["new"] is None


def test_an_admin_edit_of_the_optimizer_is_recorded(client):
    """What: optimizer parameters changed in the admin — the only place they
    can change — are an entry. How: saves the row through its ModelAdmin."""
    from django.contrib.auth.models import AnonymousUser

    from ui.admin import ExperimentDataAdmin
    from ui.models import ExperimentData

    exp = _create(client)
    data = ExperimentData.objects.get(pk=exp.data_id)
    data.optimizer_params = {"n_initial": 7}
    request = RequestFactory().post("/")
    request.user = AnonymousUser()
    ExperimentDataAdmin(ExperimentData, AdminSite()).save_model(request, data, None, True)

    entry = exp.data.history.get(kind="optimizer_params_changed")
    assert entry.payload["new"] == {"n_initial": 7}


def test_an_unknown_kind_is_a_mistake():
    """What: only declared kinds can be recorded, so the file format and the
    timeline know every one. How: records a made-up kind."""
    from ui.services import history

    with pytest.raises(ValueError):
        history.record(None, "made_up")


# ── in the file, and its times ───────────────────────────────────────────────

def _ran(client):
    """An experiment with a history, then a run launched now whose trials
    start a second later, ten seconds apart."""
    from django.utils import timezone

    from ui.models import ExperimentData, Run

    exp = _create(client)
    _state(client, exp, "max_depth", {"kind": "normal", "params": {"mu": 0.5, "sigma": 0.1}})
    now = timezone.now()
    ExperimentData.objects.filter(pk=exp.data_id).update(
        result=_trials(3, exp.data.metric_names, start=now.timestamp() + 1.0))
    Run.objects.create(experiment=exp, primary_metric="accuracy", status="done",
                       stopping={"max_trials": 3}, created_at=now, started_at=now,
                       finished_at=now, trial_offset=0, trial_count=3)
    return exp


def _export(client, exp, mode):
    return json.loads(client.post(reverse("ui:experiment_export", args=[exp.pk]),
                                  {"timestamps": mode, "tracebacks": "keep"}).content)


def test_the_history_travels_in_the_file(client):
    """What: every entry comes back from a file, still tied to its run.
    How: exports with absolute times, imports, and compares."""
    exp = _ran(client)
    body = _export(client, exp, "absolute")
    again = _import(client, body, "travelled")

    assert body["timestamps"] == "absolute"
    assert _kinds(again)[:2] == ["created", "prior_stated"]
    assert again.data.time_basis == "absolute"


def test_relative_times_carry_no_date(client):
    """What: in relative form no wall-clock time is left anywhere in the file,
    and how long things took survives.
    How: exports relative and looks for anything shaped like a date."""
    exp = _ran(client)
    body = _export(client, exp, "relative")
    text = json.dumps({k: v for k, v in body.items() if k != "environment"})

    assert body["timestamps"] == "relative"
    assert not ISO_DATE.search(text)
    first, second = body["result"]["data"][:2]
    assert second["starttime"] - first["starttime"] == pytest.approx(10.0)


def test_relative_times_count_from_the_first_run(client):
    """What: relative times are seconds since the first run was launched, and
    what came before it — the set-up, a prior stated first — carries no time.
    How: exports relative and reads the history's, the run's and a trial's."""
    exp = _ran(client)
    body = _export(client, exp, "relative")

    assert [e["at"] for e in body["history"]] == [None, None]
    assert body["runs"][0]["created_at"] == 0
    assert body["result"]["data"][0]["starttime"] == pytest.approx(1.0, abs=0.01)
    assert body["began_at"] is None


def test_a_relative_file_keeps_what_came_before_its_first_run_in_order(client):
    """What: opened again, a relative file's untimed set-up and first prior
    sit below its run, in the order they were made, still with no time.
    How: exports relative, imports, and reads the timeline's summaries and
    times."""
    import re

    exp = _ran(client)
    again = _import(client, _export(client, exp, "relative"), "relative again")
    html = client.get(reverse("ui:experiment_timeline", args=[again.pk])).content.decode()
    summaries = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", t)).strip()
                 for t in re.findall(r'<p class="timeline-summary">(.*?)</p>', html, re.S)]
    times = [re.sub(r"<[^>]+>", "", w).split("after")[0].strip()
             for w in re.findall(r'<div class="timeline-when">(.*?)</div>', html, re.S)]

    assert summaries[0].startswith("Imported from")
    assert summaries[1].startswith("Run 1")
    assert summaries[2].startswith("Prior stated on max_depth")
    assert summaries[-1].startswith("Set up:")
    assert times[1] == "+0 s" and all(t == "" for t in times[2:])


def test_no_times_means_none(client):
    """What: in no-times form the file keeps the order of things and how many
    trials there were, and no time at all.
    How: exports with none and reads every time field."""
    exp = _ran(client)
    body = _export(client, exp, "none")

    assert all(e["at"] is None for e in body["history"])
    assert all(run["started_at"] is None for run in body["runs"])
    assert not any("starttime" in t for t in body["result"]["data"])
    assert [e["kind"] for e in body["history"]][:2] == ["created", "prior_stated"]


def test_a_relative_file_is_shown_and_exported_as_relative(client):
    """What: an experiment opened from a relative file keeps that basis — it
    has no dates to give — and its history ends when it was imported.
    How: imports a relative export, then asks for absolute times again."""
    from django.utils import timezone

    exp = _ran(client)
    again = _import(client, _export(client, exp, "relative"), "relative")
    latest = again.data.history.exclude(at=None).order_by("-at").first()

    assert again.data.time_basis == "relative"
    assert abs((timezone.now() - latest.at).total_seconds()) < 60
    assert _export(client, again, "absolute")["timestamps"] == "relative"


def test_the_export_offers_the_experiments_choice_then_the_readers(client):
    """What: the export page preselects the experiment's own setting, else the
    reader's preference, else relative.
    How: reads the checked radio button as each is set."""
    from ui.services import timestamps

    exp = _ran(client)

    def checked():
        html = client.get(reverse("ui:experiment_export", args=[exp.pk])).content.decode()
        return re.search(r'name="timestamps" value="(\w+)" checked', html).group(1)

    assert checked() == "relative"
    client.post(reverse("ui:export_preferences"), {"export_timestamps": "none"})
    assert checked() == "none"
    exp.use_default_settings = False
    exp.settings = {"export_timestamps": "absolute"}
    exp.save(update_fields=["use_default_settings", "settings"])
    assert checked() == "absolute"
    assert timestamps.reader_preference(None) == "none"
