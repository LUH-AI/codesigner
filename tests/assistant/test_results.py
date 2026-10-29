"""Questions about an experiment's results answered from its stored trials
(assistant/results.py) and the tools that ask them
(assistant/tools/experiment.py): trials ranked, filtered and compared, and each
analysis, on the SMAC fixtures and on small results with known answers.
"""

import json

import pytest
from ConfigSpace import Categorical, ConfigurationSpace, Float

from assistant import results
from assistant.gateway import Context
from assistant.models import Conversation
from assistant.tools.experiment import trials_of
from assistant.turns import tools_for
from core import io
from core.optimizers.base import OptimizationResult, TrialResult
from core.optimizers.timing import STATUS_CRASHED
from experiments.services import snapshot
from tests.conftest import FIXTURES_DIR

pytestmark = pytest.mark.django_db


def _experiment(name):
    return snapshot.experiment_from_snapshot(io.parse((FIXTURES_DIR / f"{name}.ihpo").read_bytes()))


@pytest.fixture
def guided():
    """SMAC on iris over two runs (25 + 15): a 10-trial Sobol design, then 29
    model-guided trials and one random one."""
    return _experiment("smac_iris_guided")


@pytest.fixture
def design_only():
    """SMAC on iris whose 40-trial initial design used the whole budget."""
    return _experiment("smac_iris")


def _made(scores, xs, cs=None, failed=(), seconds=None):
    """Trials with *scores* (accuracy) at x = *xs* and c = *cs*, the trials in
    *failed* (0-based) crashed, by Random Search."""
    space = ConfigurationSpace({"x": Float("x", (0.0, 1.0)), "c": Categorical("c", ["a", "b", "z"])})
    cs = cs or ["a"] * len(xs)
    trials = []
    for i, (score, x, c) in enumerate(zip(scores, xs, cs)):
        info = {"time": (seconds or [1.0] * len(xs))[i]}
        if i in failed:
            info.update(status=STATUS_CRASHED, additional_info={
                "error": "ValueError: x too large", "traceback": "Traceback…\nValueError"})
        trials.append(TrialResult(trial=i + 1, config={"x": x, "c": c},
                                  scores={"accuracy": score, "f1": score / 2}, score=score,
                                  incumbent_score=score, incumbent_config={}, run_info=info,
                                  origin="Random sample"))
    result = OptimizationResult(trials=trials, primary_metric="accuracy", best_config={},
                                best_score=None, hyperparameter_importance={},
                                hyperparameter_importance_warning={})
    return results.Trials(result=result, space=space, optimizer="Random Search")


# ── phases and single trials ────────────────────────────────────────────────

def test_each_trial_is_given_the_phase_it_arose_in(guided):
    """What: SMAC's origins sort trials into initial design, model-guided and
    random; Random Search's are all search.
    How: the guided fixture's phase counts, and a made-up random search."""
    t = trials_of(guided)
    phases = [t.phase(i) for i in range(len(t.trials))]

    assert phases[:10] == [results.DESIGN] * 10
    assert phases.count(results.GUIDED) == 29 and phases.count(results.RANDOM) == 1
    assert {_made([0.5] * 3, [0.1, 0.2, 0.3]).phase(i) for i in range(3)} == {results.SEARCH}


def test_smac_under_a_former_name_is_read_as_smac():
    """What: an experiment stored under SMAC's former name has an initial
    design and its trials are not taken for a plain search.
    How: tests/fixtures/test.ihpo, stored as "SMAC (BlackBox)"."""
    t = trials_of(_experiment("test"))

    assert t.optimizer == "SMAC"
    assert results.SEARCH not in {t.phase(i) for i in range(len(t.trials))}
    assert results.analyze(t, "initial_design")["applies"] is True


def test_one_trial_is_read_with_its_rank_and_run(guided):
    """What: a trial's detail has every score, its rank on the metric, how far
    behind the best it is, whether it was a new best when it ran, and its run.
    How: the best trial of the guided fixture, and trial 30 of its second run."""
    t = trials_of(guided)
    best = t.trials[t.ranked("accuracy")[0]].trial
    top = results.trial(t, best)
    later = results.trial(t, 30)

    assert top["rank"]["place"] == 1 and top["behind_best"] == 0
    assert set(top["scores"]) >= {"accuracy", "f1", "log_loss"}
    assert later["run"] == 2 and later["rank"]["of"] == 40


def test_a_failed_trial_says_why(guided):
    """What: a failed trial's detail has its error and the end of its traceback,
    and no score or rank.
    How: a made-up search whose second trial crashed."""
    t = _made([0.5, 0.9, 0.6], [0.1, 0.9, 0.3], failed={1})
    found = results.trial(t, 2)

    assert found["status"] == "crashed" and found["failure"] == "ValueError: x too large"
    assert found["traceback_tail"].endswith("ValueError")
    assert found["scores"]["accuracy"] is None and "rank" not in found


def test_trials_are_listed_in_order_and_filtered():
    """What: a list keeps the trials matching every condition, in the order
    asked, limited, and counts every match.
    How: the best two with x above 0.2, and the failed ones."""
    t = _made([0.5, 0.9, 0.6, 0.7], [0.1, 0.9, 0.3, 0.5], failed={3})
    best = results.rows(t, "best", limit=2, where=[{"field": "x", "op": ">", "value": 0.2}])
    failed = results.rows(t, "failed")

    assert [r["trial"] for r in best["trials"]] == [2, 3] and best["matching"] == 2
    assert [r["trial"] for r in failed["trials"]] == [4]
    assert failed["trials"][0]["failure"] == "ValueError: x too large"


@pytest.mark.parametrize("ask, said", [
    (lambda t: results.rows(t, where=[{"field": "nope", "op": "=", "value": 1}]), "not a field"),
    (lambda t: results.rows(t, metric="rmse"), "no metric 'rmse'"),
    (lambda t: results.trial(t, 99), "no trial 99"),
    (lambda t: results.compare(t, [1]), "two to five"),
    (lambda t: results.analyze(t, "coverage", hyperparameter="y"), "no hyperparameter 'y'"),
])
def test_what_the_trials_do_not_have_is_named(ask, said):
    """What: a field, metric, trial or hyperparameter the trials lack is
    refused with what there is.
    How: each kind of wrong question on a made-up search."""
    with pytest.raises(results.ResultsError, match=said):
        ask(_made([0.5, 0.9], [0.1, 0.9]))


def test_trials_are_compared_by_what_they_set_differently():
    """What: compared trials list the hyperparameters that differ and those that
    do not, with every score.
    How: two trials that differ only in x."""
    found = results.compare(_made([0.5, 0.9], [0.1, 0.9]), [1, 2])

    assert found["hyperparameters_that_differ"] == ["x"]
    assert found["hyperparameters_the_same"] == ["c"]
    assert found["trials"][1]["scores"]["accuracy"] == 0.9


# ── the analyses ────────────────────────────────────────────────────────────

def test_progress_lists_every_new_best(guided):
    """What: progress lists each trial that improved the best, in order and
    improving, ending at the best score, and how long since the last.
    How: the guided fixture."""
    found = results.analyze(trials_of(guided), "progress")
    steps = found["improvements"]

    assert steps[0]["trial"] == 1 and steps[0]["gain"] is None
    assert [s["score"] for s in steps] == sorted(s["score"] for s in steps)
    assert steps[-1]["score"] == found["best"]
    assert found["trials_since_last_improvement"] == 40 - steps[-1]["trial"]
    assert [r["index"] for r in found["runs"]] == [1, 2]


def test_the_initial_design_is_measured_against_what_came_after(guided):
    """What: the initial design's analysis has its planned size and settings,
    its trials and those after it, the best of each, and how evenly it covered
    each hyperparameter.
    How: the guided fixture, whose trial cap was 10."""
    found = results.analyze(trials_of(guided), "initial_design")

    assert found["planned_size"] == 10 and found["design_trials"] == 10
    assert found["trials_after_design"] == 30
    assert found["settings"]["trial_cap"] == 10
    assert {"best_in_design", "best_after_design", "gain_after_design"} <= set(found)
    assert found["design_coverage"]["max_depth"]["bins_hit_of_10"] >= 5


def test_a_design_that_used_the_whole_budget_is_said_to(design_only):
    """What: when every trial was the initial design's, the analysis says the
    model never chose a configuration.
    How: the fixture whose design took all 40 trials."""
    found = results.analyze(trials_of(design_only), "initial_design")

    assert found["design_trials"] == 40 and found["trials_after_design"] == 0
    assert "never chose" in found["note"]


def test_random_search_has_no_initial_design():
    """What: for Random Search, the analysis says there is no initial design.
    How: a made-up random search."""
    found = results.analyze(_made([0.5, 0.9], [0.1, 0.9]), "initial_design")

    assert found["applies"] is False and "no initial design" in found["why"]


def test_a_best_value_at_the_edge_of_its_range_is_pointed_out():
    """What: coverage says when the best trials sit at a bound, and which
    choices were never tried.
    How: scores rising with x up to the top of its range; c never "z"."""
    xs = [0.1 * i for i in range(11)]
    found = results.analyze(_made([0.5 + x / 4 for x in xs], xs, ["a", "b"] * 5 + ["a"]),
                            "coverage")

    assert found["hyperparameters"]["x"]["best_at_edge"] == "upper"
    assert found["hyperparameters"]["c"]["never_tried"] == ["z"]
    assert "x" in found["note"]


def test_effects_read_which_way_a_hyperparameter_goes_with_the_score():
    """What: effects give Spearman's ρ for a number with which way is better,
    and per-value scores for a category.
    How: scores rising with x; c "b" always better."""
    xs = [0.1 * i for i in range(10)]
    cs = ["a", "b"] * 5
    scores = [0.5 + x / 4 + (0.1 if c == "b" else 0) for x, c in zip(xs, cs)]
    found = results.analyze(_made(scores, xs, cs), "effects")["hyperparameters"]

    assert found["x"]["spearman_with_score"] > 0.7
    assert found["x"]["reads"] == "higher values went with better scores"
    assert found["c"]["by_value"]["b"]["median"] > found["c"]["by_value"]["a"]["median"]


def test_importance_is_read_from_what_was_computed(guided):
    """What: the stored HyperSHAP importance is given largest first; without it
    the analysis says it was not computed.
    How: the guided fixture, and a made-up search with none."""
    found = results.analyze(trials_of(guided), "importance")
    shares = list(found["games"]["tunability"]["importance"].values())
    missing = results.analyze(_made([0.5, 0.9], [0.1, 0.9]), "importance")

    assert found["computed"] and shares == sorted(shares, key=abs, reverse=True)
    assert missing["computed"] is False


def test_failures_are_grouped_by_reason_and_place():
    """What: failures are counted by status and reason, with the ranges of
    each hyperparameter where trials failed and succeeded, and left out of
    every score.
    How: two crashed trials at large x."""
    t = _made([0.5, 0.6, 0.0, 0.0], [0.1, 0.2, 0.8, 0.9], failed={2, 3})
    found = results.analyze(t, "failures")
    scores = results.analyze(t, "scores")

    assert found["failed"] == 2 and found["by_status"] == {"crashed": 2}
    assert found["by_reason"][0] == {"reason": "ValueError: x too large", "count": 2,
                                     "trials": [3, 4]}
    assert found["where_in_the_space"]["x"] == {"failed_range": [0.8, 0.9],
                                                "succeeded_range": [0.1, 0.2]}
    assert scores["overall"]["n"] == 2 and scores["failed"] == 2


def test_metrics_and_durations_are_set_against_each_other():
    """What: the metrics analysis correlates the metrics and names each one's
    best trial; durations say what went with slow trials.
    How: f1 half of accuracy, and trials slower as x grows."""
    xs = [0.1 * i for i in range(8)]
    t = _made([0.5 + x / 4 for x in xs], xs, seconds=[1 + 10 * x for x in xs])
    metrics = results.analyze(t, "metrics")
    durations = results.analyze(t, "durations")

    assert metrics["spearman_between_metrics"]["accuracy ~ f1"] == 1.0
    assert metrics["same_best_trial"] is True
    assert durations["spearman_seconds_with_hyperparameter"]["x"] == 1.0
    assert durations["slowest_trials"][0]["trial"] == 8


def test_every_analysis_answers_on_every_fixture(guided, design_only):
    """What: every analysis gives an answer that is plain JSON on each SMAC
    fixture and on the Random Search one.
    How: each analysis on each, through json.dumps."""
    for exp in (guided, design_only, _experiment("analytics")):
        t = trials_of(exp)
        for analysis in results.ANALYSES:
            json.dumps(results.analyze(t, analysis))


# ── the tools ───────────────────────────────────────────────────────────────

def _run(conversation, tool, **args):
    return tools_for(conversation).execute(tool, args, Context(conversation))


def test_the_experiment_tools_are_offered_once_there_is_an_experiment():
    """What: the tools that read an experiment are offered with an experiment
    and not while planning.
    How: the tools of a planning conversation and of one on a dashboard."""
    names = {"read_experiment", "read_trials", "compare_trials", "analyze_experiment"}
    exp = _experiment("analytics")

    assert not names & set(tools_for(Conversation.objects.create()).names)
    assert names <= set(tools_for(Conversation.objects.create(
        experiment=exp, phase=Conversation.DASHBOARD)).names)


@pytest.mark.parametrize("tool, args, key", [
    ("read_experiment", {}, "best_trial"),
    ("read_experiment", {"aspect": "runs"}, "runs"),
    ("read_experiment", {"aspect": "settings"}, "search_space"),
    ("read_experiment", {"aspect": "priors"}, "priors"),
    ("read_experiment", {"aspect": "timeline"}, "entries"),
    ("read_trials", {"trial": 12}, "rank"),
    ("read_trials", {"order": "first", "limit": 3}, "trials"),
    ("compare_trials", {"trials": [1, 12]}, "hyperparameters_that_differ"),
    ("analyze_experiment", {"analysis": "progress"}, "improvements"),
])
def test_the_tools_read_the_conversations_experiment(guided, tool, args, key):
    """What: each tool answers about the conversation's experiment in JSON,
    and records that it touched it.
    How: every aspect and a call of each tool on the guided fixture."""
    conversation = Conversation.objects.create(experiment=guided, phase=Conversation.DASHBOARD)
    result = _run(conversation, tool, **args)

    assert key in json.loads(result.output), result.output[:300]
    assert result.experiment_id == guided.pk


def test_a_wrong_question_is_refused_with_what_there_is(guided):
    """What: a tool asked about a trial there is not is refused, saying which
    trials there are.
    How: read_trials for trial 99 of 40."""
    conversation = Conversation.objects.create(experiment=guided, phase=Conversation.DASHBOARD)

    assert "trials run from 1 to 40" in _run(conversation, "read_trials", trial=99).output
