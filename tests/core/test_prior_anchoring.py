"""Where a stated belief's decay is measured from.

*What:* a prior fades from the trial it was anchored at, and `add_prior` takes
that anchor from however many finished trials the runhistory holds when it is
called — then refuses to be re-anchored, so there is no second chance to get it
right. Everything the page says about a decaying prior assumes the anchor is
where the reader stated it.

*How:* at the source. Reaching this through a real optimization means running a
resumed one and reading SMAC's internal anchor back out; what is actually being
pinned is an ordering between two calls, and the ordering is right there. No
`importorskip` either — this holds whether or not the installed SMAC carries the
acquisition weight layer, because it is about the order codesigner calls things
in rather than about what SMAC does with them.
"""

from pathlib import Path

SOURCE = (Path(__file__).parents[2] / "core" / "optimizers"
          / "smac_optimizer.py").read_text()


def _after_the_facade() -> str:
    """The run loop, from where its facade is built.

    Scoped so that `slice_challengers` — which builds its own facade and has
    always replayed first — cannot satisfy an assertion about the run path.
    """
    return SOURCE[SOURCE.index(
        "smac = self._facade(scenario, _unreachable, previous_result)"):]


def test_a_resumed_runs_prior_is_anchored_where_it_was_stated():
    """Applied to a facade that has not been told the previous trials yet, a
    resumed run's prior anchors at trial zero — and is then weighted as though
    it had been fading since the first trial of the first run, while the figure
    above it draws it fading from where it was stated."""
    body = _after_the_facade()

    replay = body.index("self._replay(smac, config_space, previous_result.trials")
    applied = body.index("prior_report = _when(self._apply_priors(")

    assert replay < applied, (
        "priors are applied before the history is replayed, so a resumed run "
        "anchors its prior at trial zero")


def test_the_two_paths_agree_about_it():
    """`slice_challengers` is what the figure's re-walk asks, and the run is
    what actually spends the trials. A reader who re-walks and then runs has to
    be shown the same search twice."""
    walk = SOURCE[SOURCE.index("def slice_challengers"):]
    walk = walk[:walk.index("def _prior_grid")]

    assert (walk.index("self._replay(")
            < walk.index("self._apply_priors(")), "the re-walk replays first"


def test_a_waiting_prior_is_applied_after_the_design_not_before():
    """`anchor` refuses to move once set, so *when* `add_prior` is called is the
    only way to decide what a belief's decay is measured from.

    A belief that asked to wait is therefore held back until the runhistory has
    reached the initial design's size — at which point the anchor SMAC takes is
    the end of the design, which is what the figure has always said happens.
    """
    body = _after_the_facade()

    assert "waiting = _waits_for_the_design(priors)" in body
    assert "if not waiting:" in body, "applied up front only when nothing waits"
    assert "len(smac.runhistory) >= design_end" in body, (
        "the wait has to be measured against the count the anchor is taken from")


def test_the_wait_is_checked_before_the_ask():
    """After it, the belief would miss the very trial it waited for — the first
    one the model has any say in."""
    body = _after_the_facade()
    loop = body[body.index("while not collector.done:"):]

    assert (loop.index("prior_report = _when(self._apply_priors(")
            < loop.index("info = smac.ask()"))


def test_one_waiting_belief_decides_for_the_statement():
    """`_apply_priors` tabulates every hyperparameter into a single prior, so
    there is one moment to apply it at and one answer about when."""
    from core.optimizers.smac_optimizer import _waits_for_the_design

    assert _waits_for_the_design({"a": {"delay_decay": False},
                                  "b": {"delay_decay": True}})
    assert not _waits_for_the_design({"a": {"delay_decay": False}})
    assert not _waits_for_the_design({})
    assert not _waits_for_the_design(None)
