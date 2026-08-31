"""Figures that move while the run is still going.

`docs/plan_today.md` Step 3, first two parts.

**The constraint was never rendering — it was persistence.** Every trial-based
figure reads nothing but `result.trials`, and the page has always been able to
draw them; there was simply nothing to draw, because `execute_run` wrote
`experiment.result` exactly once, after `_optimize` returned. So the run writes
what it has periodically, and the run-status poll — already fetched every two
seconds, already replaced wholesale — carries whatever has arrived since the
page last said what it had.

What is deliberately absent is the analytics. Importance, interactions, partial
dependence and local effects each cost a surrogate fit or 2^n_hp coalition
evaluations, and none of them may run per trial. A partial result leaves those
fields empty, which is the state a cancelled run already produces.
"""

import json

import pytest
from django.urls import reverse

from tests.ui.runs.test_run_views import _experiment, no_thread  # noqa: F401


# ── the run writes what it has, before it is done ────────────────────────────

def test_a_partial_write_is_throttled_by_the_clock_not_the_trial_count(monkeypatch):
    """Trial durations here span three orders of magnitude. At one write per
    trial a run of 0.07s trials would rewrite the whole result a dozen times a
    second; a count that fixed that would starve a run of ten-minute trials."""
    from ui.services import run as run_service

    clock = {"now": 1000.0}
    monkeypatch.setattr(run_service.time, "monotonic", lambda: clock["now"])
    written = []
    monkeypatch.setattr(
        run_service, "OptimizationResult", _FakeResult)

    write = run_service._partial_result_writer(
        1, _FakeOptimizer(written), None, "accuracy")

    collector = _FakeCollector(trials=1)
    assert write(collector) is False, "no time has passed since the run started"

    clock["now"] += run_service.PARTIAL_RESULT_SECONDS + 0.1
    collector.results = _trials(4)
    assert write(collector) is True
    assert len(written) == 1

    collector.results = _trials(5)
    assert write(collector) is False, "throttled: the clock has not moved again"
    assert len(written) == 1


def test_a_partial_write_carries_no_analytics(monkeypatch):
    """The expensive fields stay empty by construction — that is what makes a
    per-trial write affordable at all."""
    from ui.services import run as run_service

    clock = {"now": 0.0}
    monkeypatch.setattr(run_service.time, "monotonic", lambda: clock["now"])
    seen = []
    write = run_service._partial_result_writer(1, _CaptureOptimizer(seen), None, "accuracy")

    clock["now"] += run_service.PARTIAL_RESULT_SECONDS + 1
    write(_FakeCollector(trials=3))

    partial = seen[0]
    assert partial.hyperparameter_importance == {}
    assert partial.hyperparameter_importance_warning == {}
    assert partial.hyperparameter_interactions == {}
    assert len(partial.trials) == 3


def test_the_writer_is_reached_through_the_collector_not_the_optimizers():
    """`BaseOptimizer.new_collector` is what wires it, so none of the three
    optimizers has a line about progress in it — the same reason cancellation is
    handled in `record`."""
    import inspect

    from core.optimizers import GridOptimizer, RandomOptimizer, SMACOptimizer

    for cls in (RandomOptimizer, GridOptimizer, SMACOptimizer):
        source = inspect.getsource(cls)
        assert "progress" not in source, cls.__name__
        assert "TrialCollector(" not in source, cls.__name__


def test_a_recorded_trial_calls_the_progress_callback():
    from core.optimizers.base import TrialCollector
    from core.optimizers.timing import CANCELLED, STATUS_CRASHED, STATUS_SUCCESS

    seen = []
    collector = TrialCollector(stopping={"max_trials": 10}, on_record=seen.append)

    collector.record({"a": 1}, 0.5, {"accuracy": 0.5},
                     run_info={"status": STATUS_SUCCESS})
    assert len(seen) == 1 and seen[0] is collector

    collector.record({"a": 2}, 0.0, {"accuracy": 0.0},
                     run_info={"status": STATUS_CRASHED, CANCELLED: True})
    assert len(seen) == 1, "an attempt that was not recorded is not progress"


def test_a_failing_write_does_not_take_the_run_down():
    """A snapshot is a convenience. A run that died because one could not be
    saved would be a bad trade."""
    from core.optimizers.base import TrialCollector
    from core.optimizers.timing import STATUS_SUCCESS

    def explode(collector):
        raise RuntimeError("the database went away")

    collector = TrialCollector(stopping={"max_trials": 10}, on_record=explode)
    collector.record({"a": 1}, 0.5, {"accuracy": 0.5}, run_info={"status": STATUS_SUCCESS})

    assert len(collector.results) == 1


@pytest.mark.django_db
def test_a_real_run_writes_its_trials_before_it_finishes(monkeypatch):
    """The whole point, against a real optimizer rather than a stand-in: what
    the page can read while the run is still going.

    Observed from inside the model, which is the only vantage point where "the
    run has not finished" is certainly true — checking after `execute_run`
    returns would be satisfied by the final write, which was always there.
    """
    from ui.models import Experiment
    from ui.services import run as run_service
    from ui.services.run import create_run, execute_run

    monkeypatch.setattr(run_service, "PARTIAL_RESULT_SECONDS", 0.0)

    exp = _make_experiment()
    run = create_run(exp, {"max_trials": 6}, "accuracy")

    seen = []
    model = run_service.registry.MODELS["Random Forest"]
    original = type(model).fit_predict

    def watched(self, config, X_train, y_train, X_val, seed=0):
        stored = Experiment.objects.get(pk=exp.pk).result or {}
        seen.append(len(stored.get("data") or []))
        return original(self, config, X_train, y_train, X_val, seed=seed)

    monkeypatch.setattr(type(model), "fit_predict", watched)
    execute_run(run.id)

    assert seen[-1] > 0, (
        f"the page could see nothing until the run ended; counts were {seen}")
    assert seen == sorted(seen), "and what it sees only grows"
    assert len(Experiment.objects.get(pk=exp.pk).result["data"]) == 6


@pytest.mark.django_db
def test_what_a_run_writes_mid_flight_carries_no_analytics(monkeypatch):
    """A partial result leaves those fields empty — the state the page already
    handles, because it is what a cancelled run produces."""
    from ui.models import Experiment
    from ui.services import run as run_service
    from ui.services.run import create_run, execute_run

    monkeypatch.setattr(run_service, "PARTIAL_RESULT_SECONDS", 0.0)

    exp = _make_experiment()
    run = create_run(exp, {"max_trials": 4}, "accuracy")

    seen = []
    model = run_service.registry.MODELS["Random Forest"]
    original = type(model).fit_predict

    def watched(self, config, X_train, y_train, X_val, seed=0):
        seen.append(Experiment.objects.get(pk=exp.pk).result or {})
        return original(self, config, X_train, y_train, X_val, seed=seed)

    monkeypatch.setattr(type(model), "fit_predict", watched)
    execute_run(run.id)

    partial = [s for s in seen if s.get("data")]
    assert partial, "nothing was written mid-run"
    for stored in partial:
        assert not any(stored.get("hyperparameter_importance", {}).values())
    # And the finished run does have them, so the emptiness above is the
    # partial write's doing and not the analytics being off.
    assert any(Experiment.objects.get(pk=exp.pk)
               .result["hyperparameter_importance"].values())


def _make_experiment():
    from tests.conftest import DATASETS_DIR
    from ui.services import snapshot as adapter

    return adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "live-exp", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy", "f1", "precision", "recall(macro)"],
        "seed": 0, "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }, adopt_paths=True)


# ── and the poll carries it to the page ─────────────────────────────────────

@pytest.mark.django_db
def test_the_poll_sends_fresh_plots_when_the_page_is_behind(client, ran_experiment):
    exp = ran_experiment
    stored = len(exp.result["data"])

    body = _poll(client, exp, trials=stored - 1)
    live = _live_payload(body)

    assert live["trials"] == stored
    assert set(live["static_plots"]) == {"trial_duration"}
    assert "performance_over_time" in live["metric_plots"]["accuracy"]


@pytest.mark.django_db
def test_the_poll_sends_nothing_when_the_page_is_up_to_date(client, ran_experiment):
    """The cost control. Building plots on every two-second poll would be work
    thrown away twice over — the result only moves every few seconds."""
    exp = ran_experiment
    body = _poll(client, exp, trials=len(exp.result["data"]))

    assert 'id="live-plots-data"' not in body


@pytest.mark.django_db
def test_the_poll_carries_no_analytics_figure(client, ran_experiment):
    """Whatever is in the payload is redrawn per poll, so an expensive figure
    getting in here would be an expensive figure computed every few seconds."""
    exp = ran_experiment
    live = _live_payload(_poll(client, exp, trials=0 or len(exp.result["data"]) - 1))

    drawn = set(live["static_plots"]) | set(live["metric_plots"].get("accuracy", {}))
    for expensive in ("hyperparameter_importance", "interactions_heatmap",
                      "partial_dependence", "local_effects", "local_explanation"):
        assert expensive not in drawn


@pytest.mark.django_db
def test_a_page_with_no_figures_is_sent_the_count_alone(client, ran_experiment):
    """It has no grid to merge into — it reloads into a page that has one, so
    building payloads for it would be work thrown away."""
    exp = ran_experiment
    live = _live_payload(_poll(client, exp, trials=0))

    assert live["trials"] == len(exp.result["data"])
    assert "static_plots" not in live
    assert "metric_plots" not in live


@pytest.mark.django_db
def test_the_fragment_asks_its_next_question_with_the_count_it_just_sent(
        client, ran_experiment):
    """poll.js re-reads `hx-get` off the replacement, so the swap carries the
    next question with it — that is what keeps the page and the server agreeing
    about what has already been drawn, with no state on either side."""
    exp = ran_experiment
    body = _poll(client, exp, trials=1)

    assert f"?trials={len(exp.result['data'])}" in body


@pytest.mark.django_db
def test_a_finished_run_still_refreshes_the_whole_page(client, ran_experiment):
    """Unchanged: the analytics only exist once the run has ended, so the end of
    a run is still a full reload rather than one more partial update."""
    exp = ran_experiment
    exp.runs.all().update(status="done")

    resp = client.get(reverse("ui:run_status", args=[exp.pk]) + "?trials=1")

    assert resp["HX-Refresh"] == "true"
    assert resp.content == b""


@pytest.mark.django_db
def test_the_page_declares_which_figures_move(client, ran_experiment):
    """One declaration (`Figure.live`) drives both the payload and the redraw,
    so the two cannot disagree about which figures are cheap enough."""
    body = client.get(
        reverse("ui:experiment_detail", args=[ran_experiment.pk])).content.decode()
    declared = json.loads(_json_script(body, "live-figures-data"))

    assert set(declared) == {"performance_over_time", "configuration_cube",
                             "parallel_coordinates", "trial_duration"}


# ── helpers ─────────────────────────────────────────────────────────────────

def _poll(client, exp, *, trials):
    url = reverse("ui:run_status", args=[exp.pk]) + f"?trials={trials}"
    return client.get(url).content.decode()


def _json_script(body, element_id):
    marker = f'id="{element_id}"'
    start = body.index(marker)
    start = body.index(">", start) + 1
    return body[start:body.index("</script>", start)].replace("\\u0022", '"')


def _live_payload(body):
    return json.loads(_json_script(body, "live-plots-data"))


def _trials(n):
    return [object()] * n


class _FakeCollector:
    def __init__(self, trials=0):
        self.results = _trials(trials)
        self.incumbent_score = 0.5
        self.incumbent_config = {"a": 1}


class _FakeResult:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeOptimizer:
    def __init__(self, written):
        self.written = written

    def serialize_result(self, result):
        self.written.append(result)
        return {"data": []}


class _CaptureOptimizer:
    def __init__(self, seen):
        self.seen = seen

    def serialize_result(self, result):
        self.seen.append(result)
        return {"data": []}


@pytest.fixture
def ran_experiment(client):
    """An experiment with a finished run, then put back into "running".

    The poll only answers for an active run, and what it answers with comes from
    the stored result — which is exactly the shape a partial write leaves behind,
    with more trials in it than the page has seen.
    """
    from tests.ui.storage.test_page_survives_a_round_trip import _ran_experiment

    exp = _ran_experiment(client)
    exp.runs.all().update(status="running")
    return exp
