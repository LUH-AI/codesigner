"""Contract for how core.analytics plays HyperSHAP's global games
(core/analytics.py, `compute_hp_game`).

Each game is played once through HyperSHAP's public facade and asked for its
Möbius decomposition. The FSII values that the importance and the pairwise grid
are built from are derived from those terms by shapiq's `MoebiusConverter`.
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


@pytest.mark.parametrize("game", analytics.HP_GAMES)
def test_stored_values_are_hypershaps_own_fsii(iris_splits, metrics, game):
    """The importance, its total and the pairwise grid hold HyperSHAP's own FSII.

    Setup: one explainer over five iris trials.
    Action: play *game* through `compute_hp_game`, and separately call the
    facade's `HyperSHAP.<game>()`, whose default index is FSII at order 2.
    Expect: the grid's diagonal and off-diagonal terms are the facade's order-1
    and order-2 values, and the importance is the facade's order-1 magnitudes
    as shares of their sum, with that sum as the total.

    The two sides reach the same quantity by different routes (a least-squares
    fit over the game's values, and a closed-form conversion of its Möbius
    terms), so they agree to rounding rather than bit for bit. On this setup
    they differ by at most 1e-12, against terms of order 0.01; 1e-9 is well
    clear of the rounding and far below any digit the page shows.
    """
    from hypershap import HyperSHAP

    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    explainer = analytics._build_explainer(cs, trials, "accuracy", 0)

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
    """Playing a game evaluates its coalitions in one batch, not once per index.

    The Möbius terms and the FSII values both come from one game. Asking the
    facade a second time for the other index would evaluate all 2^n_hp
    coalitions again, and on a wide space that is the most expensive thing a
    run does.

    Counted rather than timed, because a wall-clock bound is flaky on a loaded
    machine. A game's `__call__` is what shapiq invokes on the whole batch of
    coalitions, so one call means one evaluation of the game.
    """
    import hypershap.games as hs_games

    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    explainer = analytics._build_explainer(cs, trials, "accuracy", 0)

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
