"""Contract for the configuration the HyperSHAP games measure from
(core/analytics.py: `compute_hp_games`, `compute_hp_ablation`,
`compute_local_effects`).

Every game is a gain over some baseline configuration. That is the config
space's default unless a *baseline* is given, in which case it is that
configuration: the incumbent, say, or whichever trial is selected.
"""

from unittest.mock import patch

import pytest

from core import analytics
from core.models import RandomForestModel
from core.optimizers import RandomOptimizer


def _trials(iris_splits, metrics):
    """Five random-search trials of a random forest on iris, seed 0: enough for
    the surrogate to fit and for every game to find something."""
    X_train, X_val, y_train, y_val = iris_splits
    return RandomOptimizer().optimize(
        RandomForestModel(), X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=5, seed=0).trials


def test_no_baseline_means_the_default_configuration(iris_splits, metrics):
    """Leaving the baseline out gives exactly the games measured from the default.

    Setup: five iris trials.
    Action: compute every game once with no baseline, and once with the config
    space's default handed in explicitly.
    Expect: identical results, every field of every game.
    """
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)

    implicit = analytics.compute_hp_games(cs, trials, ["accuracy"], seed=0)
    explicit = analytics.compute_hp_games(cs, trials, ["accuracy"], seed=0,
                                          baseline=dict(cs.get_default_configuration()))

    assert implicit == explicit


def test_every_game_measures_from_the_baseline_it_is_given(iris_splits, metrics):
    """A trial's configuration handed in as the baseline is what every game is played from.

    Setup: wrap the three facade games so each records the `baseline_config`
    it was given.
    Action: compute every game with the last trial's configuration as baseline.
    Expect: three calls, each carrying that configuration; and a different
    answer from the one measured from the default, since the gains are
    measured from a different place.
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
    """Explaining a configuration against itself attributes nothing to any hyperparameter.

    Every coalition of the ablation game then evaluates the same configuration,
    so every coalition is worth the same and every effect is zero, to rounding.
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
    """The same trial explained against two different baselines is explained differently."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)

    from_default, _w = analytics.compute_hp_ablation(cs, trials, "accuracy", trials[-1].config, seed=0)
    from_first, _w = analytics.compute_hp_ablation(cs, trials, "accuracy", trials[-1].config,
                                                   seed=0, baseline=trials[0].config)

    assert from_default != from_first


def test_local_effects_measure_every_trial_from_the_baseline(iris_splits, metrics):
    """Each beeswarm row is an ablation against the baseline, so the baseline's own row is zero.

    Setup: five iris trials, the third one's configuration as the baseline.
    Action: compute every trial's local effects.
    Expect: the third trial's row attributes nothing, to rounding, and at least
    one other row does not.
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
