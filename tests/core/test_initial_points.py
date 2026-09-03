"""How many points a search samples before its model takes over.

SMAC bounds this two ways at once and uses the smaller: a share of the budget it
was told about (`max_ratio`) and a count (`n_configs`, or ten per hyperparameter
when none is given). Both are exposed as caps that can be switched off
independently, and the smaller can be made the larger.

That last one is arithmetic here rather than a SMAC option. SMAC only ever
writes `int(max(1, min(n_configs, max_ratio * n_trials)))` — there is no maximum
form of it — but `n_configs` overrides the per-hyperparameter bound and
`max_ratio=1.0` disables the clamp, so the number can be worked out here and
handed over. The first test below is the one that keeps that honest: with
nothing configured, the result has to be the number SMAC would have reached on
its own.
"""

import tempfile
from pathlib import Path

import pytest
from smac import Scenario
from smac.initial_design import SobolInitialDesign

from core.optimizers import SMACOptimizer


@pytest.fixture
def space(models):
    """Four hyperparameters, so the per-hyperparameter bound is forty."""
    return models["Random Forest"].get_config_space(seed=0)


def _scenario(space, budget, **extras):
    return Scenario(space, name="ihpo", n_trials=budget, deterministic=True, seed=0,
                    output_directory=Path(tempfile.mkdtemp()), **extras)


def _points(space, budget=30, previous_result=None, **settings):
    """How many initial points SMAC ends up holding."""
    optimizer = SMACOptimizer(**settings)
    scenario = _scenario(space, budget, **optimizer._scenario_extras())
    smac = optimizer._facade(scenario, lambda config, seed=0: 0.0, previous_result)
    return smac._initial_design._n_configs


# ── the number is still SMAC's, when nothing is asked for ────────────────────

@pytest.mark.parametrize("budget", [10, 30, 60, 200])
def test_untouched_it_is_the_number_smac_would_have_chosen(space, budget):
    """The count is worked out here now rather than left to SMAC, which is what
    makes the larger of the two caps reachable. It also makes it possible to
    change the default sizing by accident, which this is here to catch."""
    assert _points(space, budget) == SobolInitialDesign(_scenario(space, budget))._n_configs


def test_the_two_bounds_swap_over_as_the_budget_grows(space):
    """Which one binds is not fixed. At small budgets the share does; past the
    point where a quarter of the budget exceeds ten per hyperparameter, the
    count does. Worth pinning because it is the thing that makes "how many
    initial points does it use" have no single answer."""
    assert _points(space, 30) == 7, "a quarter of thirty"
    assert _points(space, 600) == 40, "ten per hyperparameter, four of them"


# ── each cap on its own ──────────────────────────────────────────────────────

def test_the_share_cap_is_a_fraction_of_the_budget(space):
    assert _points(space, 40, share_cap=0.5, use_trial_cap=False) == 20


def test_the_trial_cap_is_a_number(space):
    assert _points(space, 40, trial_cap=12, use_share_cap=False) == 12


def test_switching_off_the_share_cap_leaves_only_the_count(space):
    """A quarter of 200 is 50 and the count is 40, so the share is what was
    binding; without it the count is."""
    assert _points(space, 200) == 40
    assert _points(space, 200, use_share_cap=False) == 40
    assert _points(space, 200, use_trial_cap=False) == 50


def test_switching_off_both_leaves_the_whole_budget(space):
    """Neither cap bounds it, so nothing does: a search that only samples. A
    strange thing to ask for and a legible one, so it is allowed rather than
    quietly reinterpreted as something else."""
    assert _points(space, 30, use_share_cap=False, use_trial_cap=False) == 30


# ── the smaller, or the larger ───────────────────────────────────────────────

def test_the_smaller_of_the_two_is_what_is_used(space):
    """Which is SMAC's own behaviour, and stays the default."""
    assert _points(space, 200, trial_cap=12) == 12, "against a share of fifty"


def test_the_larger_can_be_asked_for_instead(space):
    assert _points(space, 200, trial_cap=12, initial_points_use_max=True) == 50


def test_the_larger_is_ours_and_not_something_smac_offers(space):
    """SMAC writes `min(n_configs, max_ratio * n_trials)` and has no other form
    of it, so asking it for the larger directly is not possible: handing it
    `n_configs=50` with the default `max_ratio` would come back as 12 again."""
    scenario = _scenario(space, 200)

    assert SobolInitialDesign(scenario, n_configs=50, max_ratio=0.06)._n_configs == 12


def test_with_one_cap_off_there_is_nothing_to_choose_between(space):
    """The form greys the choice out; this is the same fact underneath."""
    for use_max in (False, True):
        assert _points(space, 200, use_share_cap=False,
                       initial_points_use_max=use_max) == 40


# ── the edges ────────────────────────────────────────────────────────────────

def test_a_count_larger_than_the_budget_is_capped_rather_than_fatal(space):
    """SMAC raises on an initial design that does not fit in the budget. Someone
    typing 500 beside a 10-trial run should get a run that only samples, not one
    that refuses to start."""
    assert _points(space, 10, trial_cap=500, use_share_cap=False) == 10


def test_the_model_defaults_keep_their_own_place_in_the_budget(space):
    """They are an extra configuration on top of the sampled ones, and SMAC
    counts both against the budget."""
    assert _points(space, 10, trial_cap=500, use_share_cap=False,
                   use_default_config=True) == 9


def test_a_search_always_starts_somewhere(space):
    """A share small enough to round to nothing still gets one point — SMAC's
    own floor, kept."""
    assert _points(space, 10, share_cap=0.01, use_trial_cap=False) == 1


# ── what an experiment configured before this reshape comes back as ──────────

def test_a_stored_share_becomes_the_share_cap():
    """`exploration_ratio` was the share, with SMAC's per-hyperparameter bound
    still applying alongside it — which is both caps on."""
    read = SMACOptimizer.known_params({"exploration_ratio": 0.6})

    assert read["share_cap"] == 0.6
    assert read["use_share_cap"] is True
    assert read["use_trial_cap"] is True


def test_a_stored_count_becomes_the_trial_cap_with_the_share_off():
    """`exploration_trials` was an exact count that *replaced* the share — it
    opened `max_ratio` right up so nothing could clamp it. The same search is
    the trial cap with the share cap switched off, not both on."""
    read = SMACOptimizer.known_params({"exploration_trials": 12})

    assert read["trial_cap"] == 12
    assert read["use_share_cap"] is False
    assert read["use_trial_cap"] is True


def test_an_old_experiment_searches_the_way_it_used_to(space):
    """The point of translating rather than dropping. Twelve exploration trials
    then, twelve initial points now — where dropping the setting would have
    silently given seven."""
    old = SMACOptimizer.known_params({"exploration_trials": 12})

    assert _points(space, 30, **old) == 12
    assert _points(space, 30) == 7, "which is what dropping it would have given"


def test_settings_written_since_the_reshape_are_left_alone():
    read = SMACOptimizer.known_params({"share_cap": 0.3, "use_trial_cap": False})

    assert read == {"share_cap": 0.3, "use_trial_cap": False}


# ── what a resume is allowed to change about it ──────────────────────────────

def _resumed(name, n_configs):
    """A previous result carrying the design SMAC recorded in its scenario."""
    from core.optimizers.base import OptimizationResult

    return OptimizationResult(
        trials=[], primary_metric="accuracy", best_config={}, best_score=0.0,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
        metadata={"initial_design": {"name": name, "n_configs": n_configs}})


def test_a_design_whose_points_nest_keeps_growing_with_the_budget(space):
    """Sobol is a sequence and `sample_configuration` draws sequentially, so a
    bigger draw contains the smaller one: SMAC skips what it has evaluated and
    tops up the rest. Nothing to protect, so nothing is pinned — which is what
    keeps a stopped-and-resumed experiment sampling as much in total as an
    uninterrupted one."""
    previous = _resumed("SobolInitialDesign", 3)

    assert _points(space, 200, previous_result=previous) == 40


def test_a_design_whose_points_do_not_nest_keeps_the_size_it_had(space):
    """A Latin hypercube stratifies each dimension into `n` bins, so asking for
    a different `n` moves every point — the whole design would be sampled again,
    in the middle of a search that had already started modelling. Measured: none
    of a 3-point draw survives into a 7-point one."""
    previous = _resumed("LatinHypercubeInitialDesign", 3)

    assert _points(space, 200, initial_design="latin_hypercube",
                   previous_result=previous) == 3


def test_changing_the_sampling_method_rederives_rather_than_reusing(space):
    """Keeping the number would size a Latin hypercube draw by a Sobol count,
    which is the resampling this exists to prevent, done deliberately."""
    previous = _resumed("SobolInitialDesign", 3)

    assert _points(space, 200, initial_design="latin_hypercube",
                   previous_result=previous) == 40


def test_a_file_that_records_no_design_falls_back_to_working_it_out(space):
    """Everything exported before SMAC's scenario was embedded. Falling through
    to the computed number is what every run did before this existed."""
    from core.optimizers.base import OptimizationResult

    previous = OptimizationResult(
        trials=[], primary_metric="accuracy", best_config={}, best_score=0.0,
        hyperparameter_importance={}, hyperparameter_importance_warning={},
        metadata={})

    assert _points(space, 200, initial_design="latin_hypercube",
                   previous_result=previous) == 40


# ── two ways of reconstructing the size that were tried and do not work ──────

def test_the_size_does_not_come_from_counting_what_the_trials_say(space):
    """Kept as a regression test for an approach that was written and rejected.

    Counting trials whose origin names an initial design cannot work, for two
    reasons this pins by construction. The model's own default configuration
    carries `Initial Design: Default configuration` while sitting *outside*
    `n_configs` — so a count grows by one on every resume, unboundedly. And a
    file written before origins were recorded labels every trial with the
    optimizer's name, which reads as "none of these were sampled" and collapses
    the design to a single point, permanently and silently.

    The recorded `n_configs` has neither problem: it excludes additional configs
    by construction and it is absent rather than wrong on an old file.
    """
    previous = _resumed("LatinHypercubeInitialDesign", 3)
    previous.trials = []          # no trials at all, so nothing to count

    assert _points(space, 200, initial_design="latin_hypercube",
                   previous_result=previous) == 3, "read, not reconstructed"


def test_the_default_configuration_does_not_inflate_the_size(space):
    """It is an additional config, outside `n_configs`, and SMAC appends it to
    whatever size it is handed. Handing back a number that already counted it
    would append it again."""
    previous = _resumed("LatinHypercubeInitialDesign", 3)

    assert _points(space, 200, initial_design="latin_hypercube",
                   use_default_config=True, previous_result=previous) == 3
