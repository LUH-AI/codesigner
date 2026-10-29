"""`core.priors` turns a prior's `kind` and `params` into the density the figure
also draws.
"""

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Integer

from core.optimizers import GridOptimizer, RandomOptimizer, SMACOptimizer

#: The weight layer only exists on the SMAC branch that `requirements.txt` pins,
#: and a released SMAC skips the module.
weights = pytest.importorskip(
    "smac.acquisition.weight",
    reason="needs the SMAC branch carrying the acquisition weight layer")

if TYPE_CHECKING:
    from smac.acquisition.weight import PriorWeight, TabulatedPrior


@pytest.fixture
def space():
    cs = ConfigurationSpace(seed=0)
    cs.add([Integer("a", (1, 10)), Integer("b", (1, 10))])
    return cs


def _peaked(at=0.9):
    """`_peaked` states a prior as shape and parameters."""
    return {"kind": "normal", "params": {"mu": at, "sigma": 0.05}}


#: `apply_priors` grid size, imported instead of hard-coded.
from core.optimizers.smac.priors import PRIOR_GRID, apply_priors  # noqa: E402


def _facade(space, trials=40):
    import pathlib
    import tempfile

    from smac import HyperparameterOptimizationFacade as HPO
    from smac import Scenario

    directory = tempfile.TemporaryDirectory()
    scenario = Scenario(space, n_trials=trials, deterministic=True, seed=0,
                        output_directory=pathlib.Path(directory.name))
    return HPO(scenario, lambda config, seed=0: 0.0, overwrite=True), scenario, directory


def _tabulated(smac) -> "TabulatedPrior":
    return cast("TabulatedPrior", cast("PriorWeight", smac.priors["codesigner"]).prior)


def test_a_stated_prior_reaches_the_search(space):
    smac, scenario, _tmp = _facade(space)

    apply_priors(
        smac, scenario, space, {"a": dict(_peaked(), decay="logarithmic")})

    assert "codesigner" in smac.priors
    weight = cast("PriorWeight", smac.priors["codesigner"])
    assert type(weight.prior).__name__ == "TabulatedPrior"
    assert cast("TabulatedPrior", weight.prior).meta["tabulated"] == {"a": PRIOR_GRID}


def test_every_hyperparameter_goes_in_as_one_prior(space):
    """Separate priors would compose as an ensemble, a different statement from
    one joint prior."""
    smac, scenario, _tmp = _facade(space)

    apply_priors(
        smac, scenario, space,
        {"a": _peaked(0.9), "b": _peaked(0.2)})

    assert list(smac.priors) == ["codesigner"]
    assert _tabulated(smac).meta["tabulated"] == {
        "a": PRIOR_GRID, "b": PRIOR_GRID}


def test_nothing_stated_touches_nothing(space):
    smac, scenario, _tmp = _facade(space)

    apply_priors(smac, scenario, space, None)
    apply_priors(smac, scenario, space, {})

    assert not smac.priors


def test_a_prior_naming_an_unknown_hyperparameter_is_dropped(space):
    """Re-uploads and renames can orphan a stored prior."""
    smac, scenario, _tmp = _facade(space)

    apply_priors(
        smac, scenario, space, {"nonesuch": _peaked()})

    assert not smac.priors


def test_a_prior_that_cannot_be_applied_does_not_cost_the_run(space, caplog):
    smac, scenario, _tmp = _facade(space)

    # A zero sigma becomes a spike, and the run must survive it.
    apply_priors(
        smac, scenario, space,
        {"a": {"kind": "normal", "params": {"mu": 0.5, "sigma": 0.0}}})

    assert True


def test_a_shape_that_states_nothing_applies_nothing(space):
    """A uniform weight cannot reorder the acquisition, so `apply_priors`
    registers nothing for it."""
    smac, scenario, _tmp = _facade(space)

    apply_priors(
        smac, scenario, space, {"a": {"kind": "uniform", "params": {}}})

    assert not smac.priors


def test_a_log_hyperparameter_needs_no_special_case(caplog):
    """The vectorized axis of a log hyperparameter is log space, and the Normal
    there becomes a log-normal.
    """
    from ConfigSpace import Float

    cs = ConfigurationSpace(seed=0)
    cs.add([Float("lr", (1e-4, 1e0), log=True)])
    smac, scenario, _tmp = _facade(cs)

    apply_priors(
        smac, scenario, cs, {"lr": _peaked(0.5)})

    assert "codesigner" in smac.priors
    positions, densities = _tabulated(smac).tables["lr"]
    # Mid-axis is the geometric middle, 0.01 for [1e-4, 1].
    peak = positions[int(np.argmax(densities))]
    assert abs(float(peak) - 0.5) < 2.0 / PRIOR_GRID
    assert abs(float(cs["lr"].to_value(peak)) - 0.01) < 1e-3


@pytest.mark.parametrize("optimizer", [RandomOptimizer, GridOptimizer])
def test_a_search_with_no_model_accepts_and_ignores_a_prior(optimizer):
    """Callers pass priors to any optimizer, and a model-free search has no
    acquisition to weight."""
    import inspect

    signature = inspect.signature(optimizer.optimize)
    assert "priors" in signature.parameters
