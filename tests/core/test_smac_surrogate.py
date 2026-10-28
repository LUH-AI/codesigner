"""`SMACOptimizer` builds a surrogate per strategy. A feature ratio above one
makes SMAC set `max_features = 0`, and an MCMC ensemble needs an integrated
acquisition function.
"""

import tempfile
from pathlib import Path
from typing import cast

import pytest
from smac import Scenario

from core.optimizers.smac_optimizer import SMACOptimizer

#: scikit-learn names, as `_rf_opts` keys them.
FOREST = ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
          "max_features", "bootstrap")


@pytest.fixture
def space(models):
    return models["Random Forest"].get_config_space(seed=0)


def _built(space, **settings):
    """`_built` returns SMAC's model and acquisition function."""
    optimizer = SMACOptimizer(**settings)
    scenario = Scenario(space, name="ihpo", n_trials=40, deterministic=True, seed=0,
                        output_directory=Path(tempfile.mkdtemp()),
                        **optimizer._scenario_extras())
    smac = optimizer._facade(scenario, lambda config, seed=0: 0.0)
    return smac._model, smac._config_selector._acquisition_function


# random forest

def test_an_unset_forest_is_the_strategys_own(space):
    model, _ = _built(space, search_strategy="rf")

    assert model._rf_opts["n_estimators"] == 10
    assert model._rf_opts["min_samples_split"] == 2
    assert model._rf_opts["bootstrap"] is True


def test_the_forest_is_built_from_what_was_set(space):
    model, _ = _built(space, search_strategy="rf", rf_trees=64, rf_max_depth=7,
                      rf_min_samples_split=5, rf_min_samples_leaf=4,
                      rf_feature_ratio=0.5, rf_bootstrapping=False)

    assert [model._rf_opts[k] for k in FOREST] == [64, 7, 5, 4, 2, False]


def test_a_feature_ratio_above_one_would_split_on_nothing(space):
    """SMAC sets `max_features = 0` for a ratio above 1.0, and the optimizer
    caps values from stored or imported files."""
    model, _ = _built(space, search_strategy="rf", rf_feature_ratio=5.0)

    assert model._rf_opts["max_features"] == 4, "all four hyperparameters, not none"


def test_the_forest_settings_are_ignored_under_the_other_strategy(space):
    """The form hides forest settings under the GP strategy and keeps them
    stored."""
    model, _ = _built(space, search_strategy="gp", rf_trees=64)

    assert type(model).__name__ == "GaussianProcess"


# Gaussian process

def test_an_unset_process_is_the_one_the_facade_would_have_built(space):
    """`SMACOptimizer` constructs the process itself because the facade's
    `get_model` exposes only the model type and kernel."""
    from smac import BlackBoxFacade
    from smac.model.gaussian_process import GaussianProcess

    ours, _ = _built(space, search_strategy="gp")
    scenario = Scenario(space, name="ihpo", n_trials=40, deterministic=True, seed=0,
                        output_directory=Path(tempfile.mkdtemp()))
    theirs = cast(GaussianProcess, BlackBoxFacade.get_model(scenario))

    assert type(ours) is type(theirs)
    assert ours._n_restarts == theirs._n_restarts
    assert ours._normalize_y == theirs._normalize_y


def test_the_process_is_built_from_what_was_set(space):
    model, _ = _built(space, search_strategy="gp", gp_restarts=3, gp_normalize_y=False)

    assert model._n_restarts == 3
    assert model._normalize_y is False


def test_mcmc_gives_an_ensemble_instead_of_one_process(space):
    model, _ = _built(space, search_strategy="gp", gp_model_type="mcmc")

    assert type(model).__name__ == "MCMCGaussianProcess"
    assert model._n_mcmc_walkers % 2 == 0, "emcee needs an even number of walkers"


def test_mcmc_is_paired_with_an_integrated_acquisition_function(space):
    """SMAC leaves an MCMC model unpaired, and a plain acquisition function
    scores against one arbitrary member."""
    _, acquisition = _built(space, search_strategy="gp", gp_model_type="mcmc")

    assert type(acquisition).__name__ == "IntegratedAcquisitionFunction"


def test_the_wrapping_keeps_the_acquisition_function_that_was_chosen(space):
    _, acquisition = _built(space, search_strategy="gp", gp_model_type="mcmc",
                            acquisition="pi")

    assert "PI" in acquisition.name


def test_a_single_process_is_not_wrapped(space):
    """`IntegratedAcquisitionFunction` raises on a model with no members."""
    _, acquisition = _built(space, search_strategy="gp")

    assert type(acquisition).__name__ == "EI"


def test_an_unknown_fitting_method_falls_back_rather_than_failing(space):
    model, _ = _built(space, search_strategy="gp", gp_model_type="nonsense")

    assert type(model).__name__ == "GaussianProcess"


# round trip

def test_every_surrogate_setting_is_read_back_by_get_params():
    """A setting under another attribute name would reset on every save, because
    `get_params` reads `self._<name>`."""
    settings = {"rf_trees": 64, "rf_max_depth": 7, "rf_min_samples_split": 5,
                "rf_min_samples_leaf": 4, "rf_feature_ratio": 0.5,
                "rf_bootstrapping": False, "gp_model_type": "mcmc",
                "gp_restarts": 3, "gp_normalize_y": False}

    read_back = SMACOptimizer(**settings).get_params()

    assert {k: read_back[k] for k in settings} == settings


def test_each_surrogate_setting_declares_the_strategy_it_belongs_to():
    """The form needs only `depends_on` to hide settings."""
    scoped = {p.name: p.depends_on for p in SMACOptimizer.params_schema if p.depends_on}

    assert scoped == {
        "rf_trees": ("search_strategy", "rf"),
        "rf_max_depth": ("search_strategy", "rf"),
        "rf_min_samples_split": ("search_strategy", "rf"),
        "rf_min_samples_leaf": ("search_strategy", "rf"),
        "rf_feature_ratio": ("search_strategy", "rf"),
        "rf_bootstrapping": ("search_strategy", "rf"),
        "gp_model_type": ("search_strategy", "gp"),
        "gp_restarts": ("search_strategy", "gp"),
        "gp_normalize_y": ("search_strategy", "gp"),
    }
