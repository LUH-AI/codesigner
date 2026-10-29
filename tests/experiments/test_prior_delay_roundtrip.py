"""`delay_decay` asks a prior to start decaying at the end of the initial
design, since the design runs before any surrogate exists.
`snapshot_from_experiment` writes the flag instead of the resolved anchor,
because the design size depends on the reading run's budget. `_prior_record`
only copies `_PRIOR_FIELDS`, and export drops any field missing there.
"""

import pytest

from experiments.models import Experiment
from experiments.services import snapshot as adapter
from experiments.services.snapshot import _PRIOR_FIELDS

pytestmark = pytest.mark.django_db


def _experiment(priors):
    return Experiment.objects.create(
        name="waiting", model_name="Random Forest", optimizer_name="SMAC",
        metric_names=["accuracy"], seed=0, priors=priors)


#: One waiting belief with every other prior field beside it.
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
    assert "delay_decay" in _PRIOR_FIELDS


def test_it_survives_the_round_trip():
    exp = _experiment(WAITING)

    written = adapter.snapshot_from_experiment(exp)
    read_back = adapter.experiment_from_snapshot(written)

    assert read_back.data.priors["n_estimators"]["delay_decay"] is True


def test_the_anchor_survives_with_it():
    """`at_trial` and `decay` must survive beside the flag."""
    exp = _experiment(WAITING)

    read_back = adapter.experiment_from_snapshot(
        adapter.snapshot_from_experiment(exp))

    assert read_back.data.priors["n_estimators"]["at_trial"] == 13
    assert read_back.data.priors["n_estimators"]["decay"]["shape"] == "logarithmic"


def test_a_belief_that_did_not_ask_says_so():
    """An explicit `False` must survive, since the page reads a missing flag as
    on."""
    said_no = {"n_estimators": dict(WAITING["n_estimators"], delay_decay=False)}
    exp = _experiment(said_no)

    read_back = adapter.experiment_from_snapshot(
        adapter.snapshot_from_experiment(exp))

    assert read_back.data.priors["n_estimators"]["delay_decay"] is False


def test_a_file_written_before_the_flag_existed_still_reads():
    """Old files come back without `delay_decay`, and the page supplies the
    default."""
    older = {"n_estimators": {k: v for k, v in WAITING["n_estimators"].items()
                              if k != "delay_decay"}}
    exp = _experiment(older)

    read_back = adapter.experiment_from_snapshot(
        adapter.snapshot_from_experiment(exp))

    assert "delay_decay" not in read_back.data.priors["n_estimators"]
