"""The timeline page: what was done to an experiment, newest first, each in a
sentence, with a run's own events inside it."""

import re
from datetime import timedelta

import pytest
from django.urls import reverse
from django.utils import timezone

from tests.conftest import DATASETS_DIR
from ui.models import Experiment
from ui.permissions import OpenPolicy

pytestmark = pytest.mark.django_db


class HidesEverything(OpenPolicy):
    def experiments(self, request):
        return Experiment.objects.none()


def _experiment():
    from ui.services import snapshot as adapter

    return adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "told", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": None, "original_metric": None, "metric_names": ["accuracy"],
        "seed": 0, "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }, adopt_paths=True)


def _summaries(client, exp):
    html = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()
    return [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s)).strip()
            for s in re.findall(r'<p class="timeline-summary">(.*?)</p>', html, re.S)]


def _story(exp):
    """Set up, a prior, a run with a prior event and a metric change, then a
    processing change — each a minute apart."""
    from ui.models import ExperimentEvent, Run

    t0 = timezone.now() - timedelta(hours=1)
    exp.data.created_at = t0
    exp.data.save(update_fields=["created_at"])
    ExperimentEvent.objects.create(data=exp.data, at=t0, kind="created", payload={
        "configuration": {"model": {"name": "Random Forest"}, "dataset": {"demo": "iris"},
                          "optimizer": {"name": "Random Search"}, "seed": 0}})
    ExperimentEvent.objects.create(data=exp.data, at=t0 + timedelta(minutes=1),
                                   kind="prior_stated", payload={
                                       "hyperparameter": "max_depth", "old": None,
                                       "new": {"kind": "normal", "params": {"mu": 0.5}}})
    Run.objects.create(
        experiment=exp, primary_metric="f1", status="done", stopped_by="max_trials",
        stopping={"max_trials": 4}, created_at=t0 + timedelta(minutes=2),
        started_at=t0 + timedelta(minutes=2), finished_at=t0 + timedelta(minutes=3),
        trial_offset=0, trial_count=4,
        events=[{"kind": "metric_changed", "from": "accuracy", "to": "f1", "at_trial": 0,
                 "surrogate": "none", "at": (t0 + timedelta(minutes=2)).isoformat()},
                {"kind": "prior_applied", "at_trial": 2, "hyperparameters": ["max_depth"],
                 "at": (t0 + timedelta(minutes=2, seconds=30)).isoformat()}])
    ExperimentEvent.objects.create(data=exp.data, at=t0 + timedelta(minutes=4), at_trial=4,
                                   kind="processing_changed",
                                   payload={"step": "scale", "old": "auto", "new": "standardize"})


def test_newest_first_down_to_the_set_up(client):
    """What: entries run from the latest change back to how the experiment was
    set up. How: builds a short history and reads the summaries in order."""
    exp = _experiment()
    _story(exp)
    summaries = _summaries(client, exp)

    assert summaries[0].startswith("Scaling: leave as they are (Random Forest default) → standardize")
    assert summaries[1].startswith("Run 1 ran 4 trials (1–4), optimizing f1.")
    assert summaries[2].startswith("Prior stated on max_depth: normal (μ 0.5)")
    assert summaries[3].startswith("Set up: Random Forest on iris, tuned by Random Search.")


def test_a_runs_events_are_inside_it_newest_first(client):
    """What: what happened during a run is listed under it, latest first.
    How: reads the run's nested list."""
    exp = _experiment()
    _story(exp)
    html = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()
    nested = re.findall(r'<ol class="timeline-children">(.*?)</ol>', html, re.S)[0]
    items = [re.sub(r"<[^>]+>", "", i).strip() for i in re.findall(r"<li>(.*?)</li>", nested, re.S)]

    assert items[0].startswith("Prior applied to max_depth after trial 2.")
    assert items[1].startswith("Optimized metric changed: accuracy → f1.")


def test_an_experiment_from_before_the_history_still_begins_somewhere(client):
    """What: an experiment with no recorded history gets a set-up entry made
    from what it is, below its runs. How: makes one with only a run."""
    from ui.models import Run

    exp = _experiment()
    Run.objects.create(experiment=exp, primary_metric="accuracy", status="done",
                       stopped_by="max_trials", stopping={"max_trials": 2},
                       started_at=timezone.now(), finished_at=timezone.now(),
                       trial_offset=0, trial_count=2)
    summaries = _summaries(client, exp)

    assert summaries[0].startswith("Run 1 ran 2 trials")
    assert summaries[-1] == "Set up: Random Forest, tuned by Random Search."


def test_a_failed_and_a_cancelled_run_say_so(client):
    """What: a run that failed says why; one cancelled by somebody says so.
    How: records one of each and reads their sentences."""
    from django.contrib.auth import get_user_model

    from ui.models import Run

    exp = _experiment()
    ana = get_user_model().objects.create_user("ana")
    now = timezone.now()
    Run.objects.create(experiment=exp, primary_metric="accuracy", status="error",
                       error="out of memory", created_at=now, trial_offset=0, trial_count=1)
    Run.objects.create(experiment=exp, primary_metric="accuracy", status="cancelled",
                       stopped_by="cancelled", created_at=now + timedelta(seconds=1),
                       cancel_requested=True, cancel_requested_by=ana,
                       trial_offset=1, trial_count=0)
    summaries = _summaries(client, exp)

    assert "ran no trials" in summaries[0] and "Cancelled by ana." in summaries[0]
    assert "Then it failed: out of memory" in summaries[1]


def test_relative_times_are_shown_as_time_since_the_first_run(client):
    """What: an experiment whose times are relative shows each as an offset
    from when its first run was launched, never a date; what came before the
    run shows no time. How: sets the basis on a story whose run is launched
    two minutes in, and reads each entry's time."""
    exp = _experiment()
    _story(exp)
    exp.data.time_basis = "relative"
    exp.data.save(update_fields=["time_basis"])
    html = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()
    shown = [re.sub(r"<[^>]+>", "", w).split("after")[0].strip()
             for w in re.findall(r'<div class="timeline-when">(.*?)</div>', html, re.S)]

    assert "<time" not in html
    assert shown[:2] == ["+2 min", "+0 s"]
    assert all(w == "" for w in shown[2:])


def test_without_times_the_order_still_holds(client):
    """What: an experiment with no times shows none, and its entries are still
    in order — by how many trials there were. How: clears every time."""
    from ui.models import Run

    exp = _experiment()
    _story(exp)
    exp.data.history.update(at=None)
    Run.objects.filter(experiment=exp).update(created_at=None, started_at=None,
                                              finished_at=None)
    exp.data.time_basis = "none"
    exp.data.save(update_fields=["time_basis"])
    html = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()
    summaries = _summaries(client, exp)

    assert "<time" not in html
    assert summaries[0].startswith("Scaling:") and summaries[-1].startswith("Set up:")


def test_the_timeline_is_for_those_who_can_see_the_experiment(client, settings):
    """What: an experiment nobody here may see has no timeline either.
    How: hides everything and asks for the page."""
    exp = _experiment()
    settings.EXPERIMENT_POLICY = f"{__name__}.HidesEverything"
    assert client.get(reverse("ui:experiment_timeline", args=[exp.pk])).status_code == 404


# ── priors: shown when they influenced the search ───────────────────────────

def _prior_event(exp, kind, at, new, old=None):
    from ui.models import ExperimentEvent

    return ExperimentEvent.objects.create(data=exp.data, at=at, kind=kind, payload={
        "hyperparameter": "max_depth", "old": old, "new": new})


def _run_at(exp, at, applied=False):
    from ui.models import Run

    events = ([{"kind": "prior_applied", "at_trial": 0, "hyperparameters": ["max_depth"]}]
              if applied else [])
    return Run.objects.create(experiment=exp, primary_metric="accuracy", status="done",
                              stopped_by="max_trials", stopping={"max_trials": 2},
                              created_at=at, started_at=at, finished_at=at,
                              trial_offset=0, trial_count=2, events=events)


NORMAL = {"kind": "normal", "params": {"mu": 0.5}}


def test_a_prior_withdrawn_before_it_was_applied_leaves_no_trace(client):
    """What: a prior stated and withdrawn without any run applying it — say a
    run ended inside the initial design it was waiting for — is not on the
    timeline, nor is its withdrawal. How: states one, records a run that did
    not apply it, withdraws it."""
    exp = _experiment()
    t0 = timezone.now() - timedelta(hours=1)
    _prior_event(exp, "prior_stated", t0, {**NORMAL, "delay_decay": True})
    _run_at(exp, t0 + timedelta(minutes=1))
    _prior_event(exp, "prior_withdrawn", t0 + timedelta(minutes=2), None,
                 old={**NORMAL, "delay_decay": True})

    assert not any("Prior" in s for s in _summaries(client, exp))


def test_a_scheduled_prior_is_shown_as_scheduled(client):
    """What: a prior still standing that waits for the initial design is on
    the timeline as what is to come: first, above a "Now" stop, marked
    scheduled, with no time. How: states one with a delayed decay and no run
    behind it, and reads the page."""
    exp = _experiment()
    _prior_event(exp, "prior_stated", timezone.now(), {**NORMAL, "delay_decay": True})
    html = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()
    entry = html.split('class="timeline-entry is-scheduled', 1)[1].split("</li>", 1)[0]

    assert _summaries(client, exp)[0] == (
        "Prior to be applied to max_depth when the initial design ends: normal (μ 0.5).")
    assert "<time" not in entry and "Scheduled" in entry
    assert html.index("is-scheduled") < html.index("timeline-now") < html.index("Set up:")


def test_an_applied_prior_and_its_withdrawal_are_both_shown(client):
    """What: once a run applied a prior it is part of the experiment, and
    withdrawing it is a change worth showing. How: states one, records a run
    that applied it, withdraws it."""
    exp = _experiment()
    t0 = timezone.now() - timedelta(hours=1)
    _prior_event(exp, "prior_stated", t0, NORMAL)
    _run_at(exp, t0 + timedelta(minutes=1), applied=True)
    _prior_event(exp, "prior_withdrawn", t0 + timedelta(minutes=2), None, old=NORMAL)
    summaries = _summaries(client, exp)

    assert summaries[0].startswith("Prior on max_depth withdrawn.")
    assert any(s.startswith("Prior stated on max_depth") for s in summaries)


def test_an_unapplied_change_is_folded_into_the_one_that_counted(client):
    """What: a prior applied, then changed twice with nothing applying the
    middle version, reads as one change from the applied version to the
    standing one. How: three statements around one applying run."""
    exp = _experiment()
    t0 = timezone.now() - timedelta(hours=1)
    first, middle, last = (NORMAL, {"kind": "normal", "params": {"mu": 0.6}},
                           {"kind": "normal", "params": {"mu": 0.7}})
    _prior_event(exp, "prior_stated", t0, first)
    _run_at(exp, t0 + timedelta(minutes=1), applied=True)
    _prior_event(exp, "prior_edited", t0 + timedelta(minutes=2), middle, old=first)
    _run_at(exp, t0 + timedelta(minutes=3))
    _prior_event(exp, "prior_edited", t0 + timedelta(minutes=4), last, old=middle)
    html = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()

    priors = [s for s in _summaries(client, exp) if "Prior" in s]
    assert priors == ["Prior on max_depth changed.", "Prior stated on max_depth: normal (μ 0.5)."]
    assert "<dt>μ</dt><dd>0.6</dd>" not in html and "<dt>μ</dt><dd>0.7</dd>" in html


def test_the_optimizers_settings_are_listed_by_name(client):
    """What: the set-up entry lists the optimizer's settings under the names
    the form gives them, in words, leaving out what does not apply — the other
    strategy's settings and what is left to the strategy's default.
    How: reads the details of an experiment with a Gaussian-process SMAC."""
    from ui.optimizer_labels import summary

    rows = dict(summary("SMAC", {"search_strategy": "gp", "acquisition": "ei",
                                 "rf_trees": 50, "challengers": None}))

    assert rows["Search strategy"] == "Gaussian process"
    assert rows["Acquisition function"] == "Expected improvement"
    assert not any(name.startswith("Trees") or "Candidates" in name for name in rows)


def _details(html, label):
    """The rows under *label* in the first entry whose details have it."""
    block = html.split(f"<dt>{label}</dt>", 1)[1].split("</dl>", 1)[0]
    return dict(re.findall(r"<dt>([^<]+)</dt><dd>([^<]*)</dd>", block))


def test_a_runs_stopping_and_deadline_are_listed_by_name(client):
    """What: a run's stopping criteria and its deadline on a model fit are
    rows under the names the run form gives them, seconds as durations, the
    multiple only under a predicted deadline. How: records a run with three
    criteria and a predicted deadline and reads its details."""
    from ui.models import Run

    exp = _experiment()
    Run.objects.create(experiment=exp, primary_metric="accuracy", status="done",
                       stopped_by="max_trials", created_at=timezone.now(),
                       stopping={"max_trials": 30, "max_seconds": 600, "max_failures": 1},
                       trial_timeout={"mode": "predicted", "seconds": 120, "factor": 5.0},
                       trial_offset=0, trial_count=30)
    html = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()

    assert _details(html, "Stopping") == {"Trials": "30", "Time limit": "10 min",
                                          "Failed trials allowed": "1"}
    assert _details(html, "Trial timeout") == {
        "Deadline": "A multiple of the predicted duration", "At most": "2 min",
        "Multiple": "× 5"}


def test_a_changed_prior_is_listed_as_the_figure_sets_it(client):
    """What: a prior's before and after are rows — distribution, its
    parameters under the figure's symbols, and its decay — rather than one
    run-on line. How: states a normal prior, has a run apply it, edits it
    into a decaying beta one and reads the edit's details."""
    exp = _experiment()
    t0 = timezone.now() - timedelta(hours=1)
    normal = {"kind": "normal", "params": {"mu": 0.5, "sigma": 0.1}}
    _prior_event(exp, "prior_stated", t0, normal)
    _run_at(exp, t0 + timedelta(minutes=1), applied=True)
    _prior_event(exp, "prior_edited", t0 + timedelta(minutes=2),
                 {"kind": "beta", "params": {"alpha": 2, "beta": 3},
                  "decay": {"shape": "linear", "beta_ratio": 0.1}}, old=normal)
    html = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()

    assert _details(html, "Before") == {"Distribution": "Normal", "μ": "0.5", "σ": "0.1"}
    assert _details(html, "After") == {"Distribution": "Beta", "α": "2", "β": "3",
                                       "Decay": "Linear", "β / N": "0.1"}


def test_changed_optimizer_settings_are_listed_before_and_after(client):
    """What: an admin's change to the optimizer's settings is an entry with
    both versions listed by name. How: records the event and reads the page."""
    from ui.models import ExperimentEvent

    exp = _experiment()
    ExperimentEvent.objects.create(data=exp.data, at=timezone.now(),
                                   kind="optimizer_params_changed",
                                   payload={"old": {"seed_trials": 5}, "new": {"seed_trials": 8}})
    response = client.get(reverse("ui:experiment_timeline", args=[exp.pk]))

    assert response.status_code == 200
    assert "Optimizer settings changed." in response.content.decode()


def _created_with(exp, processing):
    from ui.models import ExperimentEvent

    exp.data.processing = processing
    exp.data.save(update_fields=["processing"])
    ExperimentEvent.objects.create(data=exp.data, at=timezone.now(), kind="created", payload={
        "configuration": {"model": {"name": "Random Forest"}, "dataset": {
            "demo": "iris", "processing": processing}, "optimizer": {"name": "Random Search"}}})


def test_the_data_handling_set_up_is_an_entry_of_its_own(client):
    """What: the Data Handling an experiment was set up with is an entry just
    above its set-up, every step listed — the model's defaults too — and the
    summary naming what differs from them. It is not a change: nothing ran
    under anything else. How: sets up with scaling on and reads the page."""
    exp = _experiment()
    _created_with(exp, {"missing": "auto", "scale": "standardize", "labels": "auto"})
    html = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()
    summaries = _summaries(client, exp)

    assert summaries[-2] == ("Data handling: scaling, standardize; the rest as "
                             "Random Forest does by default.")
    assert summaries[-1].startswith("Set up:")
    assert "→" not in " ".join(summaries)
    entry = html.split('kind-data_handling', 1)[1].split("</details>", 1)[0]
    assert dict(re.findall(r"<dt>([^<]+)</dt>\s*<dd>([^<]*)</dd>", entry)) == {
        "Missing values": "Leave for the model (Random Forest default)",
        "Scaling": "Standardize", "Label columns": "One column per label (Random Forest default)"}


def test_data_handling_left_to_the_model_says_so(client):
    """What: an experiment set up on its model's defaults says that in one
    line, rather than saying nothing about its data. How: sets up on "auto"
    throughout, and once naming the default outright."""
    exp = _experiment()
    _created_with(exp, {"missing": "auto", "scale": "none", "labels": "auto"})

    assert "Data handling: as Random Forest does by default." in _summaries(client, exp)


def test_a_run_lays_out_what_its_sentence_says(client):
    """What: a run's details state outright what its summary says in words —
    status, trial range, optimized metric, what stopped it — under the fields
    its record in the file has. How: reads the run's details in the story."""
    exp = _experiment()
    _story(exp)
    html = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()
    run = html.split("kind-run", 1)[1].split("</li>\n        \n", 1)[0]
    details = run.split('<dl class="timeline-details">')[-1]
    fields = dict(re.findall(r"<dt>([^<]+)</dt>\s*<dd>([^<]*)</dd>", details))

    assert fields["Status"] == "Done" and fields["Trial range"] == "1–4"
    assert fields["Optimized metric"] == "f1" and fields["Stopped by"] == "Trials"
    assert fields["Started"].endswith("UTC")


def test_a_runs_events_have_details_of_their_own(client):
    """What: an event inside a run lays its parts out too — a metric change
    its from and to. How: reads the nested list in the story."""
    exp = _experiment()
    _story(exp)
    html = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()
    nested = html.split('<ol class="timeline-children">', 1)[1].split("</ol>", 1)[0]
    fields = dict(re.findall(r"<dt>([^<]+)</dt>\s*<dd>([^<]*)</dd>", nested))

    assert fields["From"] == "accuracy" and fields["To"] == "f1"
