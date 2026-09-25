"""A belief that asked to wait, across an .ihpo file.

*What:* the initial design is drawn before any surrogate exists, so a prior
stated during it cannot reach those trials — and anchored before them it would
spend the whole phase fading while doing nothing. `delay_decay` records that the
reader asked it to start counting at the end of the design instead.

The flag is carried rather than the number it resolves to, and that is the point
of testing the file. A design's size is a fraction of the *run's* budget, so a
file written under a 13-trial history and read back into a run scheduled for 30
initial points has to wait for 30 — the reading run's design, not the writing
one's. Storing the resolved anchor would freeze the wrong number into the file.

*How:* through the snapshot seam both ways, because that is the only place the
format and the database meet, and through the allow-list that decides what a
prior is reduced to on the way out — a field missing from it is dropped in
silence, which is exactly how this would break.
"""

import pytest

from ui.models import Experiment
from ui.services import snapshot as adapter
from ui.services.snapshot import _PRIOR_FIELDS

pytestmark = pytest.mark.django_db


def _experiment(priors):
    return Experiment.objects.create(
        name="waiting", model_name="Random Forest", optimizer_name="SMAC",
        metric_names=["accuracy"], seed=0, priors=priors)


#: One belief that asked to wait, with everything else a prior carries beside
#: it, so a field lost on the way out is lost against a realistic neighbour.
WAITING = {
    "n_estimators": {
        "kind": "normal",
        "params": {"mu": 0.7, "sigma": 0.1},
        "decay": {"shape": "logarithmic", "beta_ratio": 0.1},
        "at_trial": 13,
        "delay_decay": True,
    }
}


def test_the_allow_list_carries_it():
    """`_prior_record` copies by name, so a field nobody added is dropped
    without a word. This is the line that decides it."""
    assert "delay_decay" in _PRIOR_FIELDS


def test_it_survives_the_round_trip():
    exp = _experiment(WAITING)

    written = adapter.snapshot_from_experiment(exp)
    read_back = adapter.experiment_from_snapshot(written)

    assert read_back.priors["n_estimators"]["delay_decay"] is True


def test_the_anchor_survives_with_it():
    """The two together are the whole statement: stated at 13, counting from
    the end of whatever design the reading run draws."""
    exp = _experiment(WAITING)

    read_back = adapter.experiment_from_snapshot(
        adapter.snapshot_from_experiment(exp))

    assert read_back.priors["n_estimators"]["at_trial"] == 13
    assert read_back.priors["n_estimators"]["decay"]["shape"] == "logarithmic"


def test_a_belief_that_did_not_ask_says_so():
    """`False` is an answer, not an absence. Dropped on the way out it would
    read as "never asked", which the page defaults to *on*, and the belief would
    come back from a file waiting when its author said not to."""
    said_no = {"n_estimators": dict(WAITING["n_estimators"], delay_decay=False)}
    exp = _experiment(said_no)

    read_back = adapter.experiment_from_snapshot(
        adapter.snapshot_from_experiment(exp))

    assert read_back.priors["n_estimators"]["delay_decay"] is False


def test_a_file_written_before_the_flag_existed_still_reads():
    """No `delay_decay` at all, which is every file written until now. It comes
    back without one rather than with a guess, and the page supplies the
    default — so an old file is not silently given an answer it never gave."""
    older = {"n_estimators": {k: v for k, v in WAITING["n_estimators"].items()
                              if k != "delay_decay"}}
    exp = _experiment(older)

    read_back = adapter.experiment_from_snapshot(
        adapter.snapshot_from_experiment(exp))

    assert "delay_decay" not in read_back.priors["n_estimators"]
