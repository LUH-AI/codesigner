"""SMAC caps initial points at the smaller of a budget share (`max_ratio`) and a
count (`n_configs`, by default ten per hyperparameter). Each cap can be switched
off, and the larger can replace the smaller. SMAC lacks a larger form, and
`SMACOptimizer` computes the number itself and passes it with `max_ratio=1.0`.
"""

import tempfile
from pathlib import Path

import pytest
from smac import Scenario
from smac.initial_design import SobolInitialDesign

from core.optimizers import SMACOptimizer


@pytest.fixture
def space(models):
    """Four hyperparameters give `space` a bound of forty."""
    return models["Random Forest"].get_config_space(seed=0)


def _scenario(space, budget, **extras):
    return Scenario(space, name="ihpo", n_trials=budget, deterministic=True, seed=0,
                    output_directory=Path(tempfile.mkdtemp()), **extras)


def _points(space, budget=30, previous_result=None, **settings):
    """`_points` returns SMAC's resulting initial-point count."""
    optimizer = SMACOptimizer(**settings)
    scenario = _scenario(space, budget, **optimizer._scenario_extras())
    smac = optimizer._facade(scenario, lambda config, seed=0: 0.0, previous_result)
    return smac._initial_design._n_configs


@pytest.mark.parametrize("budget", [10, 30, 60, 200])
def test_untouched_it_is_the_number_smac_would_have_chosen(space, budget):
    """Untouched settings must reproduce SMAC's own count."""
    assert _points(space, budget) == SobolInitialDesign(_scenario(space, budget))._n_configs


def test_the_two_bounds_swap_over_as_the_budget_grows(space):
    """The share binds at small budgets and the count at large ones."""
    assert _points(space, 30) == 7, "a quarter of thirty"
    assert _points(space, 600) == 40, "ten per hyperparameter, four of them"


def test_the_share_cap_is_a_fraction_of_the_budget(space):
    assert _points(space, 40, share_cap=0.5, use_trial_cap=False) == 20


def test_the_trial_cap_is_a_number(space):
    assert _points(space, 40, trial_cap=12, use_share_cap=False) == 12


def test_switching_off_the_share_cap_leaves_only_the_count(space):
    """A budget of 200 gives a share of 50, which is above the count of 40."""
    assert _points(space, 200) == 40
    assert _points(space, 200, use_share_cap=False) == 40
    assert _points(space, 200, use_trial_cap=False) == 50


def test_switching_off_both_leaves_the_whole_budget(space):
    """A sample-only search is strange yet legible, and the optimizer allows it."""
    assert _points(space, 30, use_share_cap=False, use_trial_cap=False) == 30


def test_the_smaller_of_the_two_is_what_is_used(space):
    assert _points(space, 200, trial_cap=12) == 12, "against a share of fifty"


def test_the_larger_can_be_asked_for_instead(space):
    assert _points(space, 200, trial_cap=12, initial_points_use_max=True) == 50


def test_the_larger_is_ours_and_not_something_smac_offers(space):
    """SMAC writes only `min(n_configs, max_ratio * n_trials)` and has no larger
    form."""
    scenario = _scenario(space, 200)

    assert SobolInitialDesign(scenario, n_configs=50, max_ratio=0.06)._n_configs == 12


def test_with_one_cap_off_there_is_nothing_to_choose_between(space):
    for use_max in (False, True):
        assert _points(space, 200, use_share_cap=False,
                       initial_points_use_max=use_max) == 40


def test_a_count_larger_than_the_budget_is_capped_rather_than_fatal(space):
    """SMAC raises on a design exceeding the budget, and a sample-only run beats
    a refusal."""
    assert _points(space, 10, trial_cap=500, use_share_cap=False) == 10


def test_the_model_defaults_keep_their_own_place_in_the_budget(space):
    """SMAC counts the default configuration against the budget too."""
    assert _points(space, 10, trial_cap=500, use_share_cap=False,
                   use_default_config=True) == 9


def test_a_search_always_starts_somewhere(space):
    """`_points` keeps SMAC's floor of one point."""
    assert _points(space, 10, share_cap=0.01, use_trial_cap=False) == 1


def test_a_stored_share_becomes_the_share_cap():
    """`exploration_ratio` meant the share with both caps on."""
    read = SMACOptimizer.known_params({"exploration_ratio": 0.6})

    assert read["share_cap"] == 0.6
    assert read["use_share_cap"] is True
    assert read["use_trial_cap"] is True


def test_a_stored_count_becomes_the_trial_cap_with_the_share_off():
    """`exploration_trials` replaced the share and maps to the trial cap alone."""
    read = SMACOptimizer.known_params({"exploration_trials": 12})

    assert read["trial_cap"] == 12
    assert read["use_share_cap"] is False
    assert read["use_trial_cap"] is True


def test_an_old_experiment_searches_the_way_it_used_to(space):
    """Dropping the setting would have given seven points."""
    old = SMACOptimizer.known_params({"exploration_trials": 12})

    assert _points(space, 30, **old) == 12
    assert _points(space, 30) == 7, "which is what dropping it would have given"


def test_settings_written_since_the_reshape_are_left_alone():
    read = SMACOptimizer.known_params({"share_cap": 0.3, "use_trial_cap": False})

    assert read == {"share_cap": 0.3, "use_trial_cap": False}


def _resumed(name, n_configs):
    """`_resumed` builds a previous result recording SMAC's design."""
    from core.optimizers.base import OptimizationResult

    return OptimizationResult(
        trials=[], primary_metric="accuracy", best_config={}, best_score=0.0,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
        metadata={"initial_design": {"name": name, "n_configs": n_configs}})


def test_a_design_whose_points_nest_keeps_growing_with_the_budget(space):
    """Sobol draws nest, and a resumed run samples as much as an uninterrupted
    one."""
    previous = _resumed("SobolInitialDesign", 3)

    assert _points(space, 200, previous_result=previous) == 40


def test_a_design_whose_points_do_not_nest_keeps_the_size_it_had(space):
    """Changing a Latin hypercube's `n` moves every point and would resample
    mid-search."""
    previous = _resumed("LatinHypercubeInitialDesign", 3)

    assert _points(space, 200, initial_design="latin_hypercube",
                   previous_result=previous) == 3


def test_changing_the_sampling_method_rederives_rather_than_reusing(space):
    """A Sobol count would wrongly size a Latin hypercube draw."""
    previous = _resumed("SobolInitialDesign", 3)

    assert _points(space, 200, initial_design="latin_hypercube",
                   previous_result=previous) == 40


def test_a_file_that_records_no_design_falls_back_to_working_it_out(space):
    """Files without a recorded design fall back to the computed number."""
    from core.optimizers.base import OptimizationResult

    previous = OptimizationResult(
        trials=[], primary_metric="accuracy", best_config={}, best_score=0.0,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
        metadata={})

    assert _points(space, 200, initial_design="latin_hypercube",
                   previous_result=previous) == 40


def test_the_size_does_not_come_from_counting_what_the_trials_say(space):
    """The model's default configuration sits outside `n_configs`, and old files
    label every trial with the optimizer's name. Counting trial origins would
    therefore drift on every resume or collapse the design to one point.
    """
    previous = _resumed("LatinHypercubeInitialDesign", 3)
    previous.trials = []          # nothing to count

    assert _points(space, 200, initial_design="latin_hypercube",
                   previous_result=previous) == 3, "read, not reconstructed"


def test_the_default_configuration_does_not_inflate_the_size(space):
    """SMAC appends the default to the size it receives."""
    previous = _resumed("LatinHypercubeInitialDesign", 3)

    assert _points(space, 200, initial_design="latin_hypercube",
                   use_default_config=True, previous_result=previous) == 3
