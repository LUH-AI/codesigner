"""`execute_run` writes partial results mid-run, and the status poll sends the
cheap trial figures. Partial results leave analytics empty, like a cancelled
run.
"""

import json

import pytest
from django.urls import reverse

from tests.ui.runs.test_run_views import _experiment, no_thread  # noqa: F401


# partial writes during a run

def test_a_partial_write_is_throttled_by_the_clock_not_the_trial_count(monkeypatch):
    """The clock throttles writes instead of a trial count, because trial
    durations span three orders of magnitude."""
    from experiments.services import run as run_service

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
    """Empty analytics keep each partial write cheap."""
    from experiments.services import run as run_service

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
    """`BaseOptimizer.new_collector` wires the writer, keeping progress code out
    of every optimizer."""
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
    from core.optimizers.base import TrialCollector
    from core.optimizers.timing import STATUS_SUCCESS

    def explode(collector):
        raise RuntimeError("the database went away")

    collector = TrialCollector(stopping={"max_trials": 10}, on_record=explode)
    collector.record({"a": 1}, 0.5, {"accuracy": 0.5}, run_info={"status": STATUS_SUCCESS})

    assert len(collector.results) == 1


@pytest.mark.django_db
def test_a_real_run_writes_its_trials_before_it_finishes(monkeypatch):
    """The model checks the stored result mid-run, since a check after
    `execute_run` would only see the final write.
    """
    from experiments.models import Experiment
    from experiments.services import run as run_service
    from experiments.services.run import create_run, execute_run

    monkeypatch.setattr(run_service, "PARTIAL_RESULT_SECONDS", 0.0)

    exp = _make_experiment()
    run = create_run(exp, {"max_trials": 6}, "accuracy")

    seen = []
    model = run_service.registry.MODELS["Random Forest"]
    # `fit` covers every trial, via `fit_predict` or directly.
    original = type(model).fit

    def watched(self, *args, **kwargs):
        stored = Experiment.objects.get(pk=exp.pk).data.result or {}
        seen.append(len(stored.get("data") or []))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(model), "fit", watched)
    execute_run(run.id)

    assert seen[-1] > 0, (
        f"the page could see nothing until the run ended; counts were {seen}")
    assert seen == sorted(seen), "and what it sees only grows"
    final = Experiment.objects.get(pk=exp.pk).data.result
    assert final is not None
    assert len(final["data"]) == 6


@pytest.mark.django_db
def test_what_a_run_writes_mid_flight_carries_no_analytics(monkeypatch):
    from experiments.models import Experiment
    from experiments.services import run as run_service
    from experiments.services.run import create_run, execute_run

    monkeypatch.setattr(run_service, "PARTIAL_RESULT_SECONDS", 0.0)

    exp = _make_experiment()
    run = create_run(exp, {"max_trials": 4}, "accuracy")

    seen = []
    model = run_service.registry.MODELS["Random Forest"]
    original = type(model).fit

    def watched(self, *args, **kwargs):
        seen.append(Experiment.objects.get(pk=exp.pk).data.result or {})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(model), "fit", watched)
    execute_run(run.id)

    partial = [s for s in seen if s.get("data")]
    assert partial, "nothing was written mid-run"
    for stored in partial:
        assert not any(stored.get("hyperparameter_importance", {}).values())
    # Final analytics prove the analytics setting is on.
    final = Experiment.objects.get(pk=exp.pk).data.result
    assert final is not None
    assert any(final["hyperparameter_importance"].values())


def _make_experiment():
    from tests.conftest import DATASETS_DIR
    from experiments.services import snapshot as adapter

    return adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "live-exp", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy", "f1", "precision", "recall(macro)"],
        "seed": 0, "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }, adopt_paths=True)


# the poll

@pytest.mark.django_db
def test_the_poll_sends_fresh_plots_when_the_page_is_behind(client, ran_experiment):
    exp = ran_experiment
    stored = len(exp.data.result["data"])

    body = _poll(client, exp, trials=stored - 1)
    live = _live_payload(body)

    assert live["trials"] == stored
    assert set(live["static_plots"]) == {"trial_duration"}
    assert "performance_over_time" in live["metric_plots"]["accuracy"]


@pytest.mark.django_db
def test_the_poll_sends_nothing_when_the_page_is_up_to_date(client, ran_experiment):
    exp = ran_experiment
    body = _poll(client, exp, trials=len(exp.data.result["data"]))

    assert 'id="live-plots-data"' not in body


@pytest.mark.django_db
def test_the_poll_carries_no_analytics_figure(client, ran_experiment):
    """Expensive figures stay out, because the poll payload is redrawn every few
    seconds."""
    exp = ran_experiment
    live = _live_payload(_poll(client, exp, trials=0 or len(exp.data.result["data"]) - 1))

    drawn = set(live["static_plots"]) | set(live["metric_plots"].get("accuracy", {}))
    for expensive in ("hyperparameter_importance", "interactions_heatmap",
                      "partial_dependence", "local_effects", "local_explanation"):
        assert expensive not in drawn


@pytest.mark.django_db
def test_a_page_with_no_figures_is_sent_the_count_alone(client, ran_experiment):
    """Payloads would be wasted, because a page without a grid reloads into one."""
    exp = ran_experiment
    live = _live_payload(_poll(client, exp, trials=0))

    assert live["trials"] == len(exp.data.result["data"])
    assert "static_plots" not in live
    assert "metric_plots" not in live


@pytest.mark.django_db
def test_the_fragment_asks_its_next_question_with_the_count_it_just_sent(
        client, ran_experiment):
    """`poll.js` re-reads `hx-get` from the swap, which keeps page and server in
    agreement without stored state."""
    exp = ran_experiment
    body = _poll(client, exp, trials=1)

    assert f"?trials={len(exp.data.result['data'])}" in body


@pytest.mark.django_db
def test_a_finished_run_still_refreshes_the_whole_page(client, ran_experiment):
    """Analytics only exist after the run, so the end still triggers a full
    reload."""
    exp = ran_experiment
    exp.runs.all().update(status="done")

    resp = client.get(reverse("ui:run_status", args=[exp.pk]) + "?trials=1")

    assert resp["HX-Refresh"] == "true"
    assert resp.content == b""


@pytest.mark.django_db
def test_the_page_declares_which_figures_move(client, ran_experiment):
    body = client.get(
        reverse("ui:experiment_detail", args=[ran_experiment.pk])).content.decode()
    declared = json.loads(_json_script(body, "live-figures-data"))

    assert set(declared) == {"performance_over_time", "configuration_cube",
                             "parallel_coordinates", "trial_duration"}


# helpers

@pytest.mark.django_db
def test_the_poll_carries_the_best_configuration(client, ran_experiment):
    """Every metric's best-configuration card is sent with the plots, because
    any new best changes the card."""
    exp = ran_experiment
    body = _poll(client, exp, trials=1)
    live = _live_payload(body)

    html = live["best_html"]
    for metric in exp.data.metric_names:
        assert f'data-metric="{metric}"' in html, metric
    assert "Best overall (Trial" in html


@pytest.mark.django_db
def test_the_poll_leaves_the_best_configuration_out_when_it_is_off(client, ran_experiment):
    exp = ran_experiment
    exp.settings = {**(exp.settings or {}), "show_best_configuration": False}
    exp.use_default_settings = False
    exp.save(update_fields=["settings", "use_default_settings"])

    live = _live_payload(_poll(client, exp, trials=1))

    assert "best_html" not in live


@pytest.mark.django_db
def test_the_hypershap_figures_say_they_predate_the_run(client, ran_experiment):
    """During a run, the six HyperSHAP figures say they show earlier trials."""
    body = client.get(reverse("ui:experiment_detail", args=[ran_experiment.pk])).content.decode()

    assert body.count("before-this-run") == 6


@pytest.mark.django_db
def test_and_say_nothing_once_it_is_over(client, ran_experiment):
    ran_experiment.runs.all().update(status="done")

    body = client.get(reverse("ui:experiment_detail", args=[ran_experiment.pk])).content.decode()

    assert "before-this-run" not in body


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
    """`ran_experiment` puts a finished run back into running, with more stored
    trials than the page has seen.
    """
    from tests.ui.storage.test_page_survives_a_round_trip import _ran_experiment

    exp = _ran_experiment(client)
    exp.runs.all().update(status="running")
    return exp
