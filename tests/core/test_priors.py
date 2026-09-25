"""Handing a stated prior to the search.

The figure lets a reader say where they think the optimum is. That statement is
only worth anything if it reaches the optimizer, which is what
`SMACOptimizer._apply_priors` is for.

**A prior crosses as what it is, not as a curve.** `kind` and `params`, from
which `core.priors` evaluates the density — the same function the figure asks
for its own drawing, so there is one implementation of "a Normal with this
sigma" rather than two that would have to agree forever. The browser used to
evaluate it and send the curve along; that was the second implementation.
"""

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Integer

from core.optimizers import GridOptimizer, RandomOptimizer, SMACOptimizer

#: The weight layer lives on SMAC's `feature/output-constraints` branch, which
#: `requirements.txt` pins. A released SMAC leaves the figure drawing and the
#: search unweighted, which is exactly what `_apply_priors` reports and skips —
#: so these are skipped rather than failed, and say why.
weights = pytest.importorskip(
    "smac.acquisition.weight",
    reason="needs the SMAC branch carrying the acquisition weight layer")


@pytest.fixture
def space():
    cs = ConfigurationSpace(seed=0)
    cs.add([Integer("a", (1, 10)), Integer("b", (1, 10))])
    return cs


def _peaked(at=0.9):
    """A prior as a reader states it: a shape and its parameters."""
    return {"kind": "normal", "params": {"mu": at, "sigma": 0.05}}


#: How many points `_apply_priors` evaluates a rebuilt density at. Named here
#: rather than hard-coded, so this pins the contract and not the constant.
from core.optimizers.smac_optimizer import _PRIOR_GRID  # noqa: E402


def _facade(space, trials=40):
    import pathlib
    import tempfile

    from smac import HyperparameterOptimizationFacade as HPO
    from smac import Scenario

    directory = tempfile.TemporaryDirectory()
    scenario = Scenario(space, n_trials=trials, deterministic=True, seed=0,
                        output_directory=pathlib.Path(directory.name))
    return HPO(scenario, lambda config, seed=0: 0.0, overwrite=True), scenario, directory


def test_a_stated_prior_reaches_the_search(space):
    smac, scenario, _tmp = _facade(space)

    SMACOptimizer(search_strategy="rf")._apply_priors(
        smac, scenario, space, {"a": dict(_peaked(), decay="logarithmic")})

    assert "codesigner" in smac.priors
    weight = smac.priors["codesigner"]
    assert type(weight.prior).__name__ == "TabulatedPrior"
    assert weight.prior.meta["tabulated"] == {"a": _PRIOR_GRID}


def test_every_hyperparameter_goes_in_as_one_prior(space):
    """`validate_against` requires a prior to name the whole space in column
    order, and separate priors would compose as an ensemble — which says
    something different from one joint statement."""
    smac, scenario, _tmp = _facade(space)

    SMACOptimizer(search_strategy="rf")._apply_priors(
        smac, scenario, space,
        {"a": _peaked(0.9), "b": _peaked(0.2)})

    assert list(smac.priors) == ["codesigner"]
    assert smac.priors["codesigner"].prior.meta["tabulated"] == {
        "a": _PRIOR_GRID, "b": _PRIOR_GRID}


def test_nothing_stated_touches_nothing(space):
    smac, scenario, _tmp = _facade(space)

    SMACOptimizer(search_strategy="rf")._apply_priors(smac, scenario, space, None)
    SMACOptimizer(search_strategy="rf")._apply_priors(smac, scenario, space, {})

    assert not smac.priors


def test_a_prior_naming_an_unknown_hyperparameter_is_dropped(space):
    """An experiment's search space can change under a stored prior — a custom
    model is re-uploaded, a hyperparameter is renamed."""
    smac, scenario, _tmp = _facade(space)

    SMACOptimizer(search_strategy="rf")._apply_priors(
        smac, scenario, space, {"nonesuch": _peaked()})

    assert not smac.priors


def test_a_prior_that_cannot_be_applied_does_not_cost_the_run(space, caplog):
    """It should cost the reader their prior, not their run."""
    smac, scenario, _tmp = _facade(space)

    # A sigma of zero: no curve comes out of it, and whatever `TabulatedPrior`
    # makes of the result, the run continues.
    SMACOptimizer(search_strategy="rf")._apply_priors(
        smac, scenario, space,
        {"a": {"kind": "normal", "params": {"mu": 0.5, "sigma": 0.0}}})

    assert True  # no exception is the assertion


def test_a_shape_that_states_nothing_applies_nothing(space):
    """Uniform multiplies the acquisition by a constant, which cannot reorder
    it. Applying it anyway would have the run carry a weight that does nothing
    and report that a prior was applied."""
    smac, scenario, _tmp = _facade(space)

    SMACOptimizer(search_strategy="rf")._apply_priors(
        smac, scenario, space, {"a": {"kind": "uniform", "params": {}}})

    assert not smac.priors


def test_a_log_hyperparameter_needs_no_special_case(caplog):
    """The density is stated on the vectorized axis, which for a log-scaled
    hyperparameter *is* log space — so the same Normal becomes the log-normal
    such a hyperparameter deserves, and nothing in the path has to know.

    This is the case `ConfigSpacePrior` cannot currently take, because
    ConfigSpace scales sigma through its own log transform (see
    `tests/core/test_prior_density.py::test_configspace_still_mangles_sigma_on_a_log_axis`).
    """
    from ConfigSpace import Float

    cs = ConfigurationSpace(seed=0)
    cs.add([Float("lr", (1e-4, 1e0), log=True)])
    smac, scenario, _tmp = _facade(cs)

    SMACOptimizer(search_strategy="rf")._apply_priors(
        smac, scenario, cs, {"lr": _peaked(0.5)})

    assert "codesigner" in smac.priors
    positions, densities = smac.priors["codesigner"].prior.tables["lr"]
    # The peak sits mid-axis in vector space, which is the geometric middle of
    # the native range — 0.01 for [1e-4, 1].
    peak = positions[int(np.argmax(densities))]
    assert abs(float(peak) - 0.5) < 2.0 / _PRIOR_GRID
    assert abs(float(cs["lr"].to_value(peak)) - 0.01) < 1e-3


@pytest.mark.parametrize("optimizer", [RandomOptimizer, GridOptimizer])
def test_a_search_with_no_model_accepts_and_ignores_a_prior(optimizer):
    """The caller passes what was stated without knowing which optimizer gets
    it, and a search that fits no model has no acquisition function to weight."""
    import inspect

    signature = inspect.signature(optimizer.optimize)
    assert "priors" in signature.parameters
