"""Each game plays once through HyperSHAP's facade for its Möbius terms, and
shapiq's `MoebiusConverter` derives the FSII values.
"""

from unittest.mock import patch

import pytest

from core import analytics
from core.analytics import games
from core.models import RandomForestModel
from core.optimizers import RandomOptimizer


def _trials(iris_splits, metrics):
    """`_trials` runs five seed-0 iris trials, enough for every game."""
    X_train, X_val, y_train, y_val = iris_splits
    return RandomOptimizer().optimize(
        RandomForestModel(), X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=5, seed=0).trials


@pytest.mark.parametrize("game", analytics.HP_GAMES)
def test_stored_values_are_hypershaps_own_fsii(iris_splits, metrics, game):
    """The importance, total and grid must match the facade's order-2 FSII. The
    two routes differ by about 1e-12, well inside the 1e-9 tolerance.
    """
    from hypershap import HyperSHAP

    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    explainer = games.build_explainer(cs, trials, "accuracy", 0)

    importance, warning, grid, _moebius, total = analytics.compute_hp_game(
        cs, trials, "accuracy", game, seed=7, explainer=explainer)
    hs = HyperSHAP(explainer.hs.explanation_task)
    theirs = hs.get_interaction_values_with_names(getattr(hs, game)(seed=7))

    assert warning is None
    order1 = {members[0]: abs(value) for members, value in theirs.items() if len(members) == 1}
    assert total == pytest.approx(sum(order1.values()), abs=1e-9)
    assert importance == pytest.approx({p: v / total for p, v in order1.items()}, abs=1e-9)
    for members, value in theirs.items():
        if len(members) == 1:
            assert grid[members[0]][members[0]] == pytest.approx(value, abs=1e-9), members
        elif len(members) == 2:
            a, b = members
            assert grid[a][b] == pytest.approx(value, abs=1e-9), members
            assert grid[b][a] == pytest.approx(value, abs=1e-9), members


def test_a_game_evaluates_its_coalitions_once(iris_splits, metrics):
    """A second facade call would evaluate all 2^n_hp coalitions again, the
    costliest step of a run. The test counts `__call__` invocations because
    timing is flaky.
    """
    import hypershap.games as hs_games

    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    explainer = games.build_explainer(cs, trials, "accuracy", 0)

    calls = []
    real = hs_games.TunabilityGame.__call__

    def counting(self, *args, **kwargs):
        calls.append(1)
        return real(self, *args, **kwargs)

    with patch.object(hs_games.TunabilityGame, "__call__", counting):
        _imp, warning, _grid, moebius, _total = analytics.compute_hp_game(
            cs, trials, "accuracy", "tunability", seed=0, explainer=explainer)

    assert warning is None and moebius, "the game was actually played"
    assert len(calls) == 1
