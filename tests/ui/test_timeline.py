"""The timeline lists changes newest first, nesting each run's events inside it."""

import re
from datetime import timedelta

import pytest
from django.urls import reverse
from django.utils import timezone

from tests.conftest import DATASETS_DIR
from experiments.models import Experiment
from access.policy import OpenPolicy

pytestmark = pytest.mark.django_db


class HidesEverything(OpenPolicy):
    def experiments(self, request):
        return Experiment.objects.none()


def _experiment():
    from experiments.services import snapshot as adapter

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
    """`_story` records set-up, prior, run and processing events."""
    from experiments.models import ExperimentEvent, Run

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
    exp = _experiment()
    _story(exp)
    summaries = _summaries(client, exp)

    assert summaries[0].startswith("Scaling: leave as they are (Random Forest default) → standardize")
    assert summaries[1].startswith("Run 1 ran 4 trials (1–4), optimizing f1.")
    assert summaries[2].startswith("Prior stated on max_depth: normal (μ 0.5)")
    assert summaries[3].startswith("Set up: Random Forest on iris, tuned by Random Search.")


def test_a_runs_events_are_inside_it_newest_first(client):
    exp = _experiment()
    _story(exp)
    html = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()
    nested = re.findall(r'<ol class="timeline-children">(.*?)</ol>', html, re.S)[0]
    items = [re.sub(r"<[^>]+>", "", i).strip() for i in re.findall(r"<li>(.*?)</li>", nested, re.S)]

    assert items[0].startswith("Prior applied to max_depth after trial 2.")
    assert items[1].startswith("Optimized metric changed: accuracy → f1.")


def test_an_experiment_from_before_the_history_still_begins_somewhere(client):
    """A history-less experiment gets a set-up entry built from what it is."""
    from experiments.models import Run

    exp = _experiment()
    Run.objects.create(experiment=exp, primary_metric="accuracy", status="done",
                       stopped_by="max_trials", stopping={"max_trials": 2},
                       started_at=timezone.now(), finished_at=timezone.now(),
                       trial_offset=0, trial_count=2)
    summaries = _summaries(client, exp)

    assert summaries[0].startswith("Run 1 ran 2 trials")
    assert summaries[-1] == "Set up: Random Forest, tuned by Random Search."


def test_a_failed_and_a_cancelled_run_say_so(client):
    """A failed run gives its reason."""
    from django.contrib.auth import get_user_model

    from experiments.models import Run

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
    """Each relative time is an offset from the first run's launch, and earlier
    entries have none."""
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
    """Trial counts keep the order without times."""
    from experiments.models import Run

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
    exp = _experiment()
    settings.EXPERIMENT_POLICY = f"{__name__}.HidesEverything"
    assert client.get(reverse("ui:experiment_timeline", args=[exp.pk])).status_code == 404


# Priors

def _prior_event(exp, kind, at, new, old=None):
    from experiments.models import ExperimentEvent

    return ExperimentEvent.objects.create(data=exp.data, at=at, kind=kind, payload={
        "hyperparameter": "max_depth", "old": old, "new": new})


def _run_at(exp, at, applied=False):
    from experiments.models import Run

    events = ([{"kind": "prior_applied", "at_trial": 0, "hyperparameters": ["max_depth"]}]
              if applied else [])
    return Run.objects.create(experiment=exp, primary_metric="accuracy", status="done",
                              stopped_by="max_trials", stopping={"max_trials": 2},
                              created_at=at, started_at=at, finished_at=at,
                              trial_offset=0, trial_count=2, events=events)


NORMAL = {"kind": "normal", "params": {"mu": 0.5}}


def test_a_prior_withdrawn_before_it_was_applied_leaves_no_trace(client):
    """A prior no run applied leaves no trace, withdrawal included."""
    exp = _experiment()
    t0 = timezone.now() - timedelta(hours=1)
    _prior_event(exp, "prior_stated", t0, {**NORMAL, "delay_decay": True})
    _run_at(exp, t0 + timedelta(minutes=1))
    _prior_event(exp, "prior_withdrawn", t0 + timedelta(minutes=2), None,
                 old={**NORMAL, "delay_decay": True})

    assert not any("Prior" in s for s in _summaries(client, exp))


def test_a_scheduled_prior_is_shown_as_scheduled(client):
    """A waiting prior sits above a "Now" stop, marked scheduled."""
    exp = _experiment()
    _prior_event(exp, "prior_stated", timezone.now(), {**NORMAL, "delay_decay": True})
    html = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()
    entry = html.split('class="timeline-entry is-scheduled', 1)[1].split("</li>", 1)[0]

    assert _summaries(client, exp)[0] == (
        "Prior to be applied to max_depth when the initial design ends: normal (μ 0.5).")
    assert "<time" not in entry and "Scheduled" in entry
    assert html.index("is-scheduled") < html.index("timeline-now") < html.index("Set up:")


def test_an_applied_prior_and_its_withdrawal_are_both_shown(client):
    """Once a run applied a prior, its withdrawal is worth showing."""
    exp = _experiment()
    t0 = timezone.now() - timedelta(hours=1)
    _prior_event(exp, "prior_stated", t0, NORMAL)
    _run_at(exp, t0 + timedelta(minutes=1), applied=True)
    _prior_event(exp, "prior_withdrawn", t0 + timedelta(minutes=2), None, old=NORMAL)
    summaries = _summaries(client, exp)

    assert summaries[0].startswith("Prior on max_depth withdrawn.")
    assert any(s.startswith("Prior stated on max_depth") for s in summaries)


def test_an_unapplied_change_is_folded_into_the_one_that_counted(client):
    """`experiment_timeline` folds an unapplied middle version into one change."""
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
    """The entry uses the form's names and skips the other strategy's settings
    and blank defaults."""
    from ui.optimizer_labels import summary

    rows = dict(summary("SMAC", {"search_strategy": "gp", "acquisition": "ei",
                                 "rf_trees": 50, "challengers": None}))

    assert rows["Search strategy"] == "Gaussian process"
    assert rows["Acquisition function"] == "Expected improvement"
    assert not any(name.startswith("Trees") or "Candidates" in name for name in rows)


def _details(html, label):
    """`_details` reads the rows under the first *label* found."""
    block = html.split(f"<dt>{label}</dt>", 1)[1].split("</dl>", 1)[0]
    return dict(re.findall(r"<dt>([^<]+)</dt><dd>([^<]*)</dd>", block))


def test_a_runs_stopping_and_deadline_are_listed_by_name(client):
    """Rows use the run form's names, with seconds as durations."""
    from experiments.models import Run

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
    """`experiment_timeline` lists distribution, symbols and decay in the before
    and after rows."""
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
    """An admin's optimizer change lists both versions by name."""
    from experiments.models import ExperimentEvent

    exp = _experiment()
    ExperimentEvent.objects.create(data=exp.data, at=timezone.now(),
                                   kind="optimizer_params_changed",
                                   payload={"old": {"seed_trials": 5}, "new": {"seed_trials": 8}})
    response = client.get(reverse("ui:experiment_timeline", args=[exp.pk]))

    assert response.status_code == 200
    assert "Optimizer settings changed." in response.content.decode()


def _created_with(exp, processing):
    from experiments.models import ExperimentEvent

    exp.data.processing = processing
    exp.data.save(update_fields=["processing"])
    ExperimentEvent.objects.create(data=exp.data, at=timezone.now(), kind="created", payload={
        "configuration": {"model": {"name": "Random Forest"}, "dataset": {
            "demo": "iris", "processing": processing}, "optimizer": {"name": "Random Search"}}})


def test_the_data_handling_set_up_is_an_entry_of_its_own(client):
    """The initial Data Handling gets its own entry above the set-up, since
    nothing ran under anything else."""
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
    """Model defaults throughout take one line."""
    exp = _experiment()
    _created_with(exp, {"missing": "auto", "scale": "none", "labels": "auto"})

    assert "Data handling: as Random Forest does by default." in _summaries(client, exp)


def test_a_run_lays_out_what_its_sentence_says(client):
    """A run's details spell out its summary under the file's field names."""
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
    """A nested metric change lists its from and to."""
    exp = _experiment()
    _story(exp)
    html = client.get(reverse("ui:experiment_timeline", args=[exp.pk])).content.decode()
    nested = html.split('<ol class="timeline-children">', 1)[1].split("</ol>", 1)[0]
    fields = dict(re.findall(r"<dt>([^<]+)</dt>\s*<dd>([^<]*)</dd>", nested))

    assert fields["From"] == "accuracy" and fields["To"] == "f1"
