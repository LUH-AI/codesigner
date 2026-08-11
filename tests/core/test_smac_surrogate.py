"""The model the search fits, and the settings that shape it.

Everything above this level configures *how the search is run*. These configure
the surrogate itself — the model of your model's performance that the search
consults instead of sampling blind. Each strategy has its own set, because a
random forest and a Gaussian process have nothing in common to configure.

Two things are pinned here that are easy to get wrong and silent when wrong.
A feature ratio above one makes SMAC compute `max_features = 0`, which is a
forest whose every split considers no features at all — still fitted, still
consulted, useless. And an MCMC process is an *ensemble*; an acquisition
function handed one without being told scores against a single arbitrary member
of it, and raises only in the case where there are none at all.

These build the facade rather than running a search, so they are fast. The
end-to-end runs are in `test_smac_configuration.py`.
"""

import tempfile
from pathlib import Path

import pytest
from smac import Scenario

from core.optimizers.smac_optimizer import SMACOptimizer

#: `_rf_opts` names, which are scikit-learn's rather than SMAC's argument names.
FOREST = ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
          "max_features", "bootstrap")


@pytest.fixture
def space(models):
    return models["Random Forest"].get_config_space(seed=0)


def _built(space, **settings):
    """The surrogate and acquisition function SMAC ends up holding."""
    optimizer = SMACOptimizer(**settings)
    scenario = Scenario(space, name="ihpo", n_trials=40, deterministic=True, seed=0,
                        output_directory=Path(tempfile.mkdtemp()),
                        **optimizer._scenario_extras())
    smac = optimizer._facade(scenario, lambda config, seed=0: 0.0)
    return smac._model, smac._config_selector._acquisition_function


# ── the random forest ────────────────────────────────────────────────────────

def test_an_unset_forest_is_the_strategys_own(space):
    """The convention the rest of the settings follow: blank is not a number of
    ours, it is whatever SMAC would have done."""
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
    """`max_features = 0 if ratio_features > 1.0` — no error, no warning, and a
    forest of stumps reported as a fitted surrogate. The schema caps the field
    and the optimizer caps the value, because a stored experiment or an imported
    `.ihpo` never went through the field."""
    model, _ = _built(space, search_strategy="rf", rf_feature_ratio=5.0)

    assert model._rf_opts["max_features"] == 4, "all four hyperparameters, not none"


def test_the_forest_settings_are_ignored_under_the_other_strategy(space):
    """They are stored either way — the form hides them rather than dropping
    them, so switching strategy and back does not lose them."""
    model, _ = _built(space, search_strategy="gp", rf_trees=64)

    assert type(model).__name__ == "GaussianProcess"


# ── the Gaussian process ─────────────────────────────────────────────────────

def test_an_unset_process_is_the_one_the_facade_would_have_built(space):
    """It is constructed here rather than asked for, because the facade's
    `get_model` exposes only the model type and the kernel. With nothing set the
    result has to be indistinguishable from what it would have returned."""
    from smac import BlackBoxFacade

    ours, _ = _built(space, search_strategy="gp")
    scenario = Scenario(space, name="ihpo", n_trials=40, deterministic=True, seed=0,
                        output_directory=Path(tempfile.mkdtemp()))
    theirs = BlackBoxFacade.get_model(scenario)

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
    """Marginalizing over the sampled processes is the point of sampling them.
    Nothing in SMAC pairs these up, and an unwrapped acquisition function does
    not complain — it scores against whichever member it was handed."""
    _, acquisition = _built(space, search_strategy="gp", gp_model_type="mcmc")

    assert type(acquisition).__name__ == "IntegratedAcquisitionFunction"


def test_the_wrapping_keeps_the_acquisition_function_that_was_chosen(space):
    _, acquisition = _built(space, search_strategy="gp", gp_model_type="mcmc",
                            acquisition="pi")

    assert "PI" in acquisition.name


def test_a_single_process_is_not_wrapped(space):
    """`IntegratedAcquisitionFunction` raises on a model with no members, so
    wrapping unconditionally would break the ordinary case."""
    _, acquisition = _built(space, search_strategy="gp")

    assert type(acquisition).__name__ == "EI"


def test_an_unknown_fitting_method_falls_back_rather_than_failing(space):
    """SMAC raises on an unrecognised model type. A stored experiment or an
    imported file can hold anything."""
    model, _ = _built(space, search_strategy="gp", gp_model_type="nonsense")

    assert type(model).__name__ == "GaussianProcess"


# ── the settings survive the round trip ──────────────────────────────────────

def test_every_surrogate_setting_is_read_back_by_get_params():
    """`get_params` is what `.ihpo` stores and what the settings page reloads,
    and it works off `self._<name>` — so a setting stored under a different
    attribute name would silently reset on every save."""
    settings = {"rf_trees": 64, "rf_max_depth": 7, "rf_min_samples_split": 5,
                "rf_min_samples_leaf": 4, "rf_feature_ratio": 0.5,
                "rf_bootstrapping": False, "gp_model_type": "mcmc",
                "gp_restarts": 3, "gp_normalize_y": False}

    read_back = SMACOptimizer(**settings).get_params()

    assert {k: read_back[k] for k in settings} == settings


def test_each_surrogate_setting_declares_the_strategy_it_belongs_to():
    """What lets the form hide them without knowing what any of them are."""
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
