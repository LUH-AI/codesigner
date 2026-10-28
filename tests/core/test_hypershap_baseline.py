"""`compute_hp_games` measures each game from a baseline, and the baseline
defaults to the config space's default configuration.
"""

from unittest.mock import patch

import pytest

from core import analytics
from core.models import RandomForestModel
from core.optimizers import RandomOptimizer


def _trials(iris_splits, metrics):
    """`_trials` runs five seed-0 iris trials, enough for every game."""
    X_train, X_val, y_train, y_val = iris_splits
    return RandomOptimizer().optimize(
        RandomForestModel(), X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=5, seed=0).trials


def test_no_baseline_means_the_default_configuration(iris_splits, metrics):
    """Passing the default explicitly must change no field of any game.
    """
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)

    implicit = analytics.compute_hp_games(cs, trials, ["accuracy"], seed=0)
    explicit = analytics.compute_hp_games(cs, trials, ["accuracy"], seed=0,
                                          baseline=dict(cs.get_default_configuration()))

    assert implicit == explicit


def test_every_game_measures_from_the_baseline_it_is_given(iris_splits, metrics):
    """Each of the three facade games must receive the given baseline and answer
    differently from the default.
    """
    from hypershap import HyperSHAP

    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    baseline = trials[-1].config
    seen = []

    def recording(name):
        real = getattr(HyperSHAP, name)

        def wrapper(self, *args, **kwargs):
            seen.append(kwargs.get("baseline_config"))
            return real(self, *args, **kwargs)
        return wrapper

    with patch.object(HyperSHAP, "tunability", recording("tunability")), \
         patch.object(HyperSHAP, "sensitivity", recording("sensitivity")), \
         patch.object(HyperSHAP, "mistunability", recording("mistunability")):
        moved = analytics.compute_hp_games(cs, trials, ["accuracy"], seed=0, baseline=baseline)

    assert len(seen) == 3
    assert all(dict(config) == baseline for config in seen)
    default = analytics.compute_hp_games(cs, trials, ["accuracy"], seed=0)
    assert moved["mistunability"][2] != default["mistunability"][2]


def test_an_ablation_against_its_own_configuration_is_zero(iris_splits, metrics):
    """Every coalition then evaluates one configuration, leaving all effects
    zero.
    """
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    config = trials[-1].config

    effects, warning = analytics.compute_hp_ablation(
        cs, trials, "accuracy", config, seed=0, baseline=config)

    assert warning is None
    assert set(effects) == set(cs.keys())
    assert all(value == pytest.approx(0.0, abs=1e-12) for value in effects.values())


def test_an_ablation_measures_from_the_baseline_it_is_given(iris_splits, metrics):
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)

    from_default, _w = analytics.compute_hp_ablation(cs, trials, "accuracy", trials[-1].config, seed=0)
    from_first, _w = analytics.compute_hp_ablation(cs, trials, "accuracy", trials[-1].config,
                                                   seed=0, baseline=trials[0].config)

    assert from_default != from_first


def test_local_effects_measure_every_trial_from_the_baseline(iris_splits, metrics):
    """Each beeswarm row is an ablation against the baseline, which makes the
    baseline's own row zero.
    """
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)

    _names, rows, warning = analytics.compute_local_effects(
        cs, trials, "accuracy", seed=0, baseline=trials[2].config)

    assert warning is None
    own = next(row for row in rows if row["index"] == 2)
    assert all(value == pytest.approx(0.0, abs=1e-12) for value in own["effects"].values())
    others = [row for row in rows if row["index"] != 2]
    assert any(any(abs(v) > 1e-9 for v in row["effects"].values()) for row in others)
