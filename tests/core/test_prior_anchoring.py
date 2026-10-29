"""`add_prior` anchors decay at the finished-trial count and never re-anchors,
so the tests pin call order in the `core/optimizers/smac` source.
"""

from pathlib import Path

SMAC = Path(__file__).parents[2] / "core" / "optimizers" / "smac"
SOURCE = (SMAC / "optimizer.py").read_text()
FACADE = (SMAC / "facade.py").read_text()


def _after_the_facade() -> str:
    """`_after_the_facade` starts at the run loop's facade, so
    `slice_challengers` cannot satisfy a run-path assertion.
    """
    return SOURCE[SOURCE.index(
        "smac = self._facade(scenario, _unreachable, previous_result)"):]


def test_a_resumed_runs_prior_is_anchored_where_it_was_stated():
    """`add_prior` anchors a resumed run's prior at trial zero when `replay`
    runs later, and the run then weights the prior as fading since the first run
    began."""
    body = _after_the_facade()

    replay = body.index("history.replay(smac, config_space, previous_result.trials")
    applied = body.index("prior_report = stamped(apply_priors(")

    assert replay < applied, (
        "priors are applied before the history is replayed, so a resumed run "
        "anchors its prior at trial zero")


def test_the_two_paths_agree_about_it():
    """`slice_challengers` and the run both replay before `apply_priors`, so a
    re-walk and a run show the same search."""
    rebuilt = FACADE[FACADE.index("def _rebuilt"):]
    walk = SOURCE[SOURCE.index("def slice_challengers"):]
    walk = walk[:walk.index("def evaluate_prior")]

    assert rebuilt.index("replay(") < rebuilt.index("yield"), "the rebuild replays"
    assert (walk.index("self._rebuilt(")
            < walk.index("apply_priors(")), "the re-walk replays first"


def test_a_waiting_prior_is_applied_after_the_design_not_before():
    """`anchor` stays fixed once set, and `apply_priors` defers a waiting
    belief until the runhistory reaches the initial design's size.
    """
    body = _after_the_facade()

    assert "waiting = waits_for_the_design(priors)" in body
    assert "if not waiting:" in body, "applied up front only when nothing waits"
    assert "len(smac.runhistory) >= design_end" in body, (
        "the wait has to be measured against the count the anchor is taken from")


def test_the_wait_is_checked_before_the_ask():
    """After `ask`, the belief would miss its first model-chosen trial."""
    body = _after_the_facade()
    loop = body[body.index("while not collector.done:"):]

    assert (loop.index("prior_report = stamped(apply_priors(")
            < loop.index("info = smac.ask()"))


def test_one_waiting_belief_decides_for_the_statement():
    """`apply_priors` tabulates every hyperparameter into one prior, so one
    waiting belief delays all."""
    from core.optimizers.smac.priors import waits_for_the_design

    assert waits_for_the_design({"a": {"delay_decay": False},
                                  "b": {"delay_decay": True}})
    assert not waits_for_the_design({"a": {"delay_decay": False}})
    assert not waits_for_the_design({})
    assert not waits_for_the_design(None)
