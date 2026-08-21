"""Contract for BaseOptimizer.compute_hp_importance (core/optimizers/base.py).

Estimates per-hyperparameter importance from completed trials, with a fallback
ladder (HyperSHAP → RandomForest surrogate → uniform) returning
(importance_dict, warning_or_None).
"""

from unittest.mock import patch

import pytest

from core.models import RandomForestModel
from core.optimizers import RandomOptimizer, TrialResult
from core.optimizers.base import BaseOptimizer, fit_surrogate


def _trials(iris_splits, metrics):
    X_train, X_val, y_train, y_val = iris_splits
    return RandomOptimizer().optimize(
        RandomForestModel(), X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=5, seed=0).trials


def test_insufficient_trials_returns_uniform_with_warning():
    """Fewer than two usable trials → uniform weights and a warning about trials."""
    cs = RandomForestModel().get_config_space(seed=0)
    params = list(cs.keys())
    one = [TrialResult(trial=1, config=dict(cs.get_default_configuration()),
                       scores={"accuracy": 0.5}, score=0.5,
                       incumbent_score=0.5, incumbent_config={})]
    imp, warning = RandomOptimizer().compute_hp_importance(cs, one, "accuracy", seed=0)
    assert set(imp) == set(params)
    assert all(abs(v - 1.0 / len(params)) < 1e-9 for v in imp.values())
    assert warning and "trials" in warning.lower()


def test_sufficient_trials_produce_normalized_importance(iris_splits, metrics):
    """Enough trials → a normalized distribution over the config-space hyperparameters."""
    cs = RandomForestModel().get_config_space(seed=0)
    imp, _ = RandomOptimizer().compute_hp_importance(
        cs, _trials(iris_splits, metrics), "accuracy", seed=0)
    assert set(imp) == set(cs.keys())
    assert all(v >= 0.0 for v in imp.values())
    assert abs(sum(imp.values()) - 1.0) < 1e-6


def test_a_game_that_finds_nothing_reports_nothing_rather_than_zeros():
    """Every contribution exactly zero → empty, with a warning saying so.

    Reachable, not theoretical: an objective that saturates (iris accuracy does
    this readily) makes the surrogate's aggregate identical whichever
    hyperparameters a coalition frees, so HyperSHAP succeeds and returns all
    zeros. Normalizing that anyway produced `{hp: 0.0, ...}` with `warning=None`
    — truthy, so the figure's own `if not importance` guard let it through and
    the pie was built from zeros and drew nothing, unexplained.

    Distinct from the uniform-weights rung on purpose: uniform means "we could
    not tell", this means "we could, and there is nothing there".
    """
    cs = RandomForestModel().get_config_space(seed=0)
    flat = [TrialResult(trial=i, config=dict(cs.sample_configuration()),
                        scores={"accuracy": 0.7}, score=0.7,
                        incumbent_score=0.7, incumbent_config={})
            for i in range(1, 7)]

    imp, warning = RandomOptimizer().compute_hp_importance(cs, flat, "accuracy", seed=0)

    assert imp == {}
    assert warning and "no signal" in warning
    uniform = 1.0 / len(list(cs.keys()))
    assert uniform not in imp.values(), "must not be confused with the uniform rung"


def test_a_game_that_finds_nothing_reports_no_interactions_either():
    """Same rung, same reasoning: there is no pairwise structure in all-zeros,
    so the interactions heatmap gets its caption rather than a grid of 0.0."""
    cs = RandomForestModel().get_config_space(seed=0)
    flat = [TrialResult(trial=i, config=dict(cs.sample_configuration()),
                        scores={"accuracy": 0.7}, score=0.7,
                        incumbent_score=0.7, incumbent_config={})
            for i in range(1, 7)]

    interactions, warning = RandomOptimizer().compute_hp_interactions(
        cs, flat, "accuracy", seed=0)

    assert interactions == {}
    assert warning and "no signal" in warning


def test_interactions_insufficient_trials_returns_empty_not_uniform():
    """Fewer than two usable trials → empty, with a warning — the same
    "not enough trials" rung `_compute_hp_game` returns for importance itself,
    since there's no `iv` to re-extract pairwise terms from."""
    cs = RandomForestModel().get_config_space(seed=0)
    one = [TrialResult(trial=1, config=dict(cs.get_default_configuration()),
                       scores={"accuracy": 0.5}, score=0.5,
                       incumbent_score=0.5, incumbent_config={})]
    interactions, warning = RandomOptimizer().compute_hp_interactions(cs, one, "accuracy", seed=0)
    assert interactions == {}
    assert warning and "trials" in warning.lower()


def test_interactions_form_a_symmetric_square_grid(iris_splits, metrics):
    """Enough trials → one entry per hyperparameter pair (plus the diagonal),
    symmetric, and the diagonal matches compute_hp_importance's own order-1
    values before normalization — same HyperSHAP call, just unnormalized and
    signed rather than abs-valued and scaled to sum to 1."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    params = list(cs.keys())
    interactions, warning = RandomOptimizer().compute_hp_interactions(cs, trials, "accuracy", seed=0)
    assert warning is None
    assert set(interactions) == set(params)
    for a in params:
        assert set(interactions[a]) == set(params)
        for b in params:
            assert interactions[a][b] == interactions[b][a]


def test_fit_surrogate_insufficient_trials_returns_none_with_warning():
    """Fewer than two usable trials → no surrogate, with a warning — the
    same condition every compute_hp_* fallback ladder treats as "nothing to
    fit from," not a fit that happens to be bad."""
    cs = RandomForestModel().get_config_space(seed=0)
    one = [TrialResult(trial=1, config=dict(cs.get_default_configuration()),
                       scores={"accuracy": 0.5}, score=0.5,
                       incumbent_score=0.5, incumbent_config={})]
    rf, warning = fit_surrogate(cs, one, "accuracy", seed=0)
    assert rf is None
    assert warning and "trials" in warning.lower()


def test_fit_surrogate_returns_a_usable_regressor(iris_splits, metrics):
    """Enough trials → a fitted RandomForestRegressor: feature_importances_
    is the exact thing _compute_hp_game's own fallback rung reads, and
    predict() is what a future surrogate-consumer (Partial Dependencies,
    Phase 6) needs that HyperSHAP's own explainer doesn't expose directly."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    rf, warning = fit_surrogate(cs, trials, "accuracy", seed=0)
    assert warning is None
    assert len(rf.feature_importances_) == len(cs.keys())
    sample = trials[0].config
    from ConfigSpace import Configuration
    prediction = rf.predict([Configuration(cs, values=sample).get_array()])
    assert isinstance(prediction[0], float)


def test_compute_hp_games_builds_the_explainer_once_per_metric_not_per_game(iris_splits, metrics):
    """tunability/sensitivity/mistunability explain the same trial history
    for a given metric — fitting the underlying surrogate 3 times over (once
    per game) instead of once was pure waste. One accuracy explainer + one
    f1 explainer, not three of each."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    opt = RandomOptimizer()

    with patch.object(BaseOptimizer, "_build_explainer",
                     wraps=opt._build_explainer) as spy:
        opt.compute_hp_games(cs, trials, ["accuracy", "f1"], seed=0)

    called_metrics = [call.args[2] for call in spy.call_args_list]
    assert called_metrics == ["accuracy", "f1"]


def test_the_fallback_surrogate_is_also_fitted_once_per_metric_not_per_game(iris_splits, metrics):
    """When HyperSHAP fails, the RandomForest stand-in is shared across games too.

    The explainer was deliberately shared per metric; this rung was not, so a
    metric whose game calls raised refit an identical forest once per game —
    the exact waste the explainer sharing existed to avoid, one rung lower.

    Two metrics × three games = 6 game calls, all failing, and 2 fits.

    There are two ways to reach a game now — our own `ExactComputer` and, if
    that construction fails, HyperSHAP's facade — so getting to the surrogate
    rung means closing both. Breaking only the facade leaves the games working,
    which is the fallback doing its job.
    """
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    opt = RandomOptimizer()

    def explode(self):
        raise RuntimeError("no shapiq today")

    def explode_shared(*a, **k):
        raise RuntimeError("no shapiq today")

    with patch("core.optimizers.base._shared_exact_computer", explode_shared), \
         patch("hypershap.HyperSHAP.tunability", explode), \
         patch("hypershap.HyperSHAP.sensitivity", explode), \
         patch("hypershap.HyperSHAP.mistunability", explode), \
         patch("core.optimizers.base.fit_surrogate",
               wraps=fit_surrogate) as fits:
        games = opt.compute_hp_games(cs, trials, ["accuracy", "f1"], seed=0)

    fitted_metrics = [call.args[2] for call in fits.call_args_list]
    assert fitted_metrics == ["accuracy", "f1"], "one fallback fit per metric"
    # And the fallback still produced real answers, not uniform weights.
    for game in opt.HP_GAMES:
        importance, warning, interactions, moebius = (games[game][0]["accuracy"],
                                                      games[game][1]["accuracy"],
                                                      games[game][2]["accuracy"],
                                                      games[game][3]["accuracy"])
        assert set(importance) == set(cs.keys())
        assert warning and "falling back" in warning
        assert interactions == {}, "a plain feature_importances_ has no pairwise structure"
        assert moebius == [], "nor any coalition structure"


# ── One ExactComputer, two indices ──────────────────────────────────────────
#
# `_shared_exact_computer` builds what `HyperSHAP.<game>()` builds rather than
# calling it, so that one set of 2^n_hp coalition evaluations can serve both the
# FSII values the importance figures use and the Möbius decomposition the graph,
# upset and by-order views need. Its own docstring carries the reasoning; these
# are the guards it names.


def test_our_exact_computer_reproduces_hypershaps_own_numbers(iris_splits, metrics):
    """The test the coupling rests on: bit-identical FSII, for all three games.

    hypershap is at 0.0.6 and its internals may move. If they do — a changed
    default, a reordered searcher, a different aggregation — this fails loudly
    rather than the page quietly showing numbers from a different game.
    """
    from hypershap import HyperSHAP
    from core.optimizers.base import _shared_exact_computer

    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    opt = RandomOptimizer()
    explainer = opt._build_explainer(cs, trials, "accuracy", 0)

    for game in opt.HP_GAMES:
        # The same seed to both, because both take one: the searcher's seed is
        # the experiment's, not HyperSHAP's default of 0.
        theirs = getattr(HyperSHAP(explainer.hs.explanation_task), game)(seed=7)
        ours, _moebius = _shared_exact_computer(explainer, game, seed=7)
        assert ours.dict_values.keys() == theirs.dict_values.keys(), game
        for key, value in theirs.dict_values.items():
            assert ours.dict_values[key] == value, f"{game} {key}"


def test_the_moebius_transform_is_truncation_independent(iris_splits, metrics):
    """Why the new views read Möbius and not a higher-order FSII.

    FSII is the *faithful* least-squares k-order approximation, fitted jointly,
    so its order-1 terms are not the order-1 terms of an order-3 fit — measured
    on a 6-hyperparameter space, order-1 moves 5% of the largest term and
    order-2 moves 24%. Raising the order in place would silently rewrite every
    stored importance number. Möbius assigns one value per coalition and does
    not move, which is what lets it be stored once and read at any order.

    Both halves are pinned, because it is the contrast that justifies the
    design and a reader who only saw the Möbius half might "simplify" it away.
    """
    from core.optimizers.base import _shared_exact_computer

    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    opt = RandomOptimizer()
    explainer = opt._build_explainer(cs, trials, "accuracy", 0)
    n = len(list(cs.keys()))

    from shapiq import ExactComputer
    import hypershap.games as hs_games
    import hypershap.task as hs_task
    from hypershap.utils import Aggregation, RandomConfigSpaceSearcher
    from core.optimizers.base import _HPO_SIMULATION_SAMPLES

    source = explainer.hs.explanation_task
    task = hs_task.TunabilityExplanationTask(
        config_space=cs, surrogate_model=source.surrogate_model,
        baseline_config=cs.get_default_configuration())
    game = hs_games.TunabilityGame(
        explanation_task=task,
        cs_searcher=RandomConfigSpaceSearcher(
            explanation_task=task, n_samples=_HPO_SIMULATION_SAMPLES,
            mode=Aggregation.MAX, seed=0))
    ec = ExactComputer(n_players=game.get_num_hyperparameters(), game=game)

    shallow = ec(index="Moebius", order=2).get_n_order(order=1).dict_values
    deep = ec(index="Moebius", order=n).get_n_order(order=1).dict_values
    assert shallow == deep, "Möbius must not move with the truncation order"

    fsii_2 = ec(index="FSII", order=2).get_n_order(order=1).dict_values
    fsii_n = ec(index="FSII", order=n).get_n_order(order=1).dict_values
    assert fsii_2 != fsii_n, "FSII does move — which is why it is not used here"


def test_moebius_covers_every_coalition_of_every_size(iris_splits, metrics):
    """2^n_hp - 1 terms: every non-empty coalition, once. The empty one is the
    game with nothing tuned — a constant offset, not attributable to any
    hyperparameter — and is dropped."""
    cs = RandomForestModel().get_config_space(seed=0)
    params = set(cs.keys())
    trials = _trials(iris_splits, metrics)

    _imp, warning, _inter, moebius, _total = RandomOptimizer()._compute_hp_game(
        cs, trials, "accuracy", "tunability", seed=0)

    assert warning is None
    assert len(moebius) == 2 ** len(params) - 1
    assert {len(row["members"]) for row in moebius} == set(range(1, len(params) + 1))
    for row in moebius:
        assert set(row["members"]) <= params
        assert isinstance(row["value"], float)
    assert len({tuple(sorted(row["members"])) for row in moebius}) == len(moebius), "no duplicates"


def test_moebius_terms_are_ranked_by_magnitude(iris_splits, metrics):
    """Every view that reads these takes a top-k slice, so the order is part of
    the contract rather than an accident of dict iteration."""
    cs = RandomForestModel().get_config_space(seed=0)
    _i, _w, _x, moebius, _total = RandomOptimizer()._compute_hp_game(
        cs, _trials(iris_splits, metrics), "accuracy", "tunability", seed=0)

    magnitudes = [abs(row["value"]) for row in moebius]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_a_broken_shared_computer_still_yields_hypershaps_own_importance(iris_splits, metrics):
    """The fallback that makes the coupling survivable: if hypershap's internals
    move, we lose the Möbius views and nothing else. The importance numbers must
    not be able to come down with them."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)

    def explode(*a, **k):
        raise RuntimeError("hypershap moved its furniture")

    with patch("core.optimizers.base._shared_exact_computer", explode):
        importance, warning, interactions, moebius, _total = RandomOptimizer()._compute_hp_game(
            cs, trials, "accuracy", "tunability", seed=0)

    assert warning is None, "the facade answered, so this is not a failure"
    assert set(importance) == set(cs.keys())
    assert interactions, "the order-2 grid still comes from HyperSHAP's own call"
    assert moebius == [], "only the Möbius-based views go without"


def test_the_shared_computer_costs_one_set_of_coalitions(iris_splits, metrics):
    """The whole reason for building the computer ourselves.

    Not timed — a wall-clock assertion would be flaky on a loaded machine.
    Counted instead: the game is called once per coalition, and asking for a
    second index must not call it again.
    """
    from core.optimizers.base import _shared_exact_computer
    import hypershap.games as hs_games

    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    explainer = RandomOptimizer()._build_explainer(cs, trials, "accuracy", 0)

    calls = []
    real = hs_games.TunabilityGame.__call__

    def counting(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)

    with patch.object(hs_games.TunabilityGame, "__call__", counting):
        fsii, moebius = _shared_exact_computer(explainer, "tunability")

    assert fsii is not None and moebius is not None
    assert len(calls) == 1, "one batched evaluation of every coalition, not one per index"


def test_ablation_insufficient_trials_returns_empty_not_uniform():
    """Fewer than two usable trials → empty, with a warning — unlike the three
    global games, there is no uniform-weights rung: a signed "how much did
    each hyperparameter help or hurt" has no honest uniform answer."""
    cs = RandomForestModel().get_config_space(seed=0)
    one = [TrialResult(trial=1, config=dict(cs.get_default_configuration()),
                       scores={"accuracy": 0.5}, score=0.5,
                       incumbent_score=0.5, incumbent_config={})]
    ablation, warning = RandomOptimizer().compute_hp_ablation(
        cs, one, "accuracy", dict(cs.get_default_configuration()), seed=0)
    assert ablation == {}
    assert warning and "trials" in warning.lower()


def test_ablation_explains_one_trial_against_the_default(iris_splits, metrics):
    """Enough trials → a signed value per hyperparameter for the trial handed
    in, not a normalized distribution (no fallback ladder to normalize)."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    ablation, warning = RandomOptimizer().compute_hp_ablation(
        cs, trials, "accuracy", trials[-1].config, seed=0)
    assert warning is None
    assert set(ablation) == set(cs.keys())
    assert all(isinstance(v, float) for v in ablation.values())


def test_the_explanation_surrogate_actually_uses_the_seed_it_is_given(iris_splits, metrics):
    """`seed` has to reach the surrogate, for both the global games and ablation.

    It didn't. Both paths called `ExplanationTask.from_data` without a
    `base_model`, and hypershap's default is `RandomForestRegressor(random_state=0)`
    — a hardcoded 0, not the experiment's seed. So every explanation was
    computed at seed 0 whatever the run was seeded with, silently. `seed` was
    even a declared parameter of `compute_hp_ablation` that the body never read.

    Two different seeds must give two different fits. (Not a claim about which
    is right — only that the argument is no longer ignored.)
    """
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    opt = RandomOptimizer()

    a, warn_a = opt.compute_hp_ablation(cs, trials, "accuracy", trials[-1].config, seed=0)
    b, warn_b = opt.compute_hp_ablation(cs, trials, "accuracy", trials[-1].config, seed=7)
    assert warn_a is None and warn_b is None
    assert a != b, "ablation ignored its seed"

    imp_a, *_ = opt._compute_hp_game(cs, trials, "accuracy", "tunability", seed=0)
    imp_b, *_ = opt._compute_hp_game(cs, trials, "accuracy", "tunability", seed=7)
    assert imp_a != imp_b, "the global games ignored their seed"


def test_seed_zero_keeps_hypershaps_own_default_estimator(iris_splits, metrics):
    """Passing the estimator explicitly must not move the numbers at seed 0.

    hypershap's default is `RandomForestRegressor(random_state=0)` and sklearn's
    own default `n_estimators` is 100, so the estimator now passed in is the
    identical object at seed 0. This pins that the seed fix changed behaviour
    only where behaviour was wrong — every stored result and pinned expectation
    from before it stays valid.
    """
    from sklearn.ensemble import RandomForestRegressor
    from hypershap import ExplanationTask, HyperSHAP
    from core.optimizers.base import _pair_trials_with_scores

    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    data = _pair_trials_with_scores(cs, trials, "accuracy")
    params = list(cs.keys())

    def order1(task):
        iv = HyperSHAP(task).tunability()
        return {params[i]: v for (i,), v in iv.get_n_order(order=1).dict_values.items()}

    theirs = order1(ExplanationTask.from_data(cs, data))
    ours = order1(ExplanationTask.from_data(
        cs, data, base_model=RandomForestRegressor(n_estimators=100, random_state=0)))
    assert theirs == ours


def test_partial_dependence_insufficient_trials_returns_empty():
    """Fewer than two usable trials → nothing to fit a surrogate from, so
    grid/ice_lines/pdp are all empty rather than some default grid with no
    predictions to show on it."""
    cs = RandomForestModel().get_config_space(seed=0)
    one = [TrialResult(trial=1, config=dict(cs.get_default_configuration()),
                       scores={"accuracy": 0.5}, score=0.5,
                       incumbent_score=0.5, incumbent_config={})]
    grid, ice_lines, pdp, warning = RandomOptimizer().compute_partial_dependence(
        cs, one, "accuracy", "n_estimators", seed=0)
    assert grid == [] and ice_lines == [] and pdp == []
    assert warning and "trials" in warning.lower()


def test_partial_dependence_integer_hp_grid_is_all_integers(iris_splits, metrics):
    """n_estimators is a UniformIntegerHyperparameter — the grid must never
    suggest a fractional trial count, unlike a plain linspace would."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    grid, ice_lines, pdp, warning = RandomOptimizer().compute_partial_dependence(
        cs, trials, "accuracy", "n_estimators", seed=0)
    assert warning is None
    assert all(isinstance(v, int) for v in grid)
    assert grid == sorted(grid)


def test_partial_dependence_categorical_hp_grid_is_its_choices(iris_splits, metrics):
    """A categorical hyperparameter's grid is its full, exact set of
    choices — there is nothing to interpolate between categories."""
    from core.models import SVMModel
    X_train, X_val, y_train, y_val = iris_splits
    cs = SVMModel().get_config_space(seed=0)
    trials = RandomOptimizer().optimize(
        SVMModel(), X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=5, seed=0).trials
    grid, ice_lines, pdp, warning = RandomOptimizer().compute_partial_dependence(
        cs, trials, "accuracy", "kernel", seed=0)
    assert warning is None
    assert grid == list(cs["kernel"].choices)


def test_partial_dependence_log_hp_grid_is_spaced_on_the_log_scale():
    """A log-scaled hyperparameter is gridded on its own scale, not on a
    native linspace between its bounds.

    `C` spans 0.01 to 100 logarithmically, so ConfigSpace samples ~69% of its
    configurations below 5.27 — which is where a native linspace puts its
    *second* of twenty points. The grid would then spend 19 points in the
    tail the surrogate has the least evidence about, and one point covering
    the two-thirds of the run that actually happened.

    Pinned as a constant *ratio* between consecutive points rather than a
    constant difference, since that is exactly what "log-spaced" means.
    """
    from core.models import SVMModel
    from core.optimizers.base import _hp_grid

    grid = _hp_grid(SVMModel().get_config_space(seed=0)["C"], 20)

    assert grid[0] == pytest.approx(0.01)
    assert grid[-1] == pytest.approx(100.0)
    assert grid[1] == pytest.approx(0.016238, abs=1e-6), "a native linspace would put 5.27 here"
    ratios = [grid[i + 1] / grid[i] for i in range(len(grid) - 1)]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))


def test_partial_dependence_linear_hp_grid_is_unchanged():
    """The same routing must not move a linear hyperparameter's grid.

    Every RandomForest hyperparameter is linear, and a great many expectations
    (this file's, the fixtures', anything pinned against a stored result) were
    taken against the plain `linspace(lower, upper)` this replaced. Mapping the
    normalized vector back through `to_value` reproduces it to within one ULP.
    """
    import numpy as np

    from core.optimizers.base import _hp_grid

    cs = RandomForestModel().get_config_space(seed=0)
    for name in cs.keys():
        hp = cs[name]
        native = np.linspace(hp.lower, hp.upper, 20)
        expected = (sorted({int(round(v)) for v in native})
                    if type(hp).__name__ == "UniformIntegerHyperparameter"
                    else [float(v) for v in native])
        assert _hp_grid(hp, 20) == pytest.approx(expected), name


def test_partial_dependence_ice_lines_and_pdp_share_the_grids_shape(iris_splits, metrics):
    """One ICE row per trial, each the same length as the grid; the PDP
    curve is the grid-wise mean of those rows, not some other reduction."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    grid, ice_lines, pdp, warning = RandomOptimizer().compute_partial_dependence(
        cs, trials, "accuracy", "max_depth", seed=0)
    assert warning is None
    assert len(ice_lines) == len(trials)
    assert all(len(row) == len(grid) for row in ice_lines)
    assert len(pdp) == len(grid)
    for i in range(len(grid)):
        column = [row[i] for row in ice_lines]
        assert pdp[i] == sum(column) / len(column)


def test_partial_dependence_cap_limits_the_curves_drawn(iris_splits, metrics):
    """`max_ice_curves` bounds how many trials appear, 0 meaning all of them."""
    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)

    _, uncapped, _, _ = RandomOptimizer().compute_partial_dependence(
        cs, trials, "accuracy", "max_depth", seed=0, max_ice_curves=0)
    _, capped, _, _ = RandomOptimizer().compute_partial_dependence(
        cs, trials, "accuracy", "max_depth", seed=0, max_ice_curves=3)

    assert len(uncapped) == len(trials)
    assert len(capped) == 3


def test_partial_dependence_cap_bounds_the_work_not_just_the_picture():
    """The cap has to reach the predictions, or it saves a payload and none of
    the seconds — which is the wrong half. The trials outside the sample are
    never predicted, so the mean is over what was sampled.

    Pinned by construction rather than by timing: with a cap of 2 the curve is
    the mean of exactly the first and last trials' ICE rows, which it could not
    be if the middle ones had been predicted and then discarded.
    """
    from ConfigSpace import ConfigurationSpace, Float
    from core.optimizers.base import fit_surrogate

    cs = ConfigurationSpace(seed=0)
    cs.add([Float("a", (0.0, 1.0), default=0.5), Float("b", (0.0, 1.0), default=0.5)])
    trials = [TrialResult(trial=i, config={"a": i / 10, "b": 1 - i / 10},
                          scores={"accuracy": i / 10}, score=i / 10,
                          incumbent_score=i / 10, incumbent_config={})
              for i in range(1, 11)]

    grid, ice, pdp, warning = RandomOptimizer().compute_partial_dependence(
        cs, trials, "accuracy", "a", seed=0, max_ice_curves=2)

    assert warning is None
    assert len(ice) == 2, "first and last only"
    for i in range(len(grid)):
        assert pdp[i] == pytest.approx((ice[0][i] + ice[1][i]) / 2)


def test_partial_dependence_cap_above_the_trial_count_changes_nothing():
    """A cap nobody reaches must not perturb the curve."""
    from ConfigSpace import ConfigurationSpace, Float

    cs = ConfigurationSpace(seed=0)
    cs.add([Float("a", (0.0, 1.0), default=0.5)])
    trials = [TrialResult(trial=i, config={"a": i / 10}, scores={"accuracy": i / 10},
                          score=i / 10, incumbent_score=i / 10, incumbent_config={})
              for i in range(1, 6)]

    _, _, uncapped, _ = RandomOptimizer().compute_partial_dependence(
        cs, trials, "accuracy", "a", seed=0, max_ice_curves=0)
    _, _, generous, _ = RandomOptimizer().compute_partial_dependence(
        cs, trials, "accuracy", "a", seed=0, max_ice_curves=500)

    assert uncapped == generous


def test_partial_dependence_batching_matches_predicting_one_at_a_time(iris_splits, metrics):
    """Batching the grid into one `rf.predict` call must not change a number.

    The one-at-a-time version this replaced spent 1.54s where the batched one
    spends 0.004s on the same 600 rows — the whole difference was sklearn's
    per-call overhead, not the forest. This pins that it was only overhead:
    the same surrogate, asked the same questions one row at a time, gives the
    same answers.
    """
    from ConfigSpace import Configuration

    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    grid, ice_lines, pdp, warning = RandomOptimizer().compute_partial_dependence(
        cs, trials, "accuracy", "max_depth", seed=0)
    assert warning is None

    rf, _ = fit_surrogate(cs, trials, "accuracy", 0)
    expected = []
    for t in trials:
        row = []
        for value in grid:
            values = dict(t.config)
            values["max_depth"] = value
            cfg = Configuration(cs, values=values)
            row.append(float(rf.predict([cfg.get_array()])[0]))
        expected.append(row)

    assert ice_lines == expected


def test_partial_dependence_holes_where_the_config_space_refuses_a_grid_value():
    """A grid value the config space rejects for a given trial contributes
    `None` at that index rather than raising or shifting the row.

    None of the registry models have forbidden clauses, so this builds a space
    that does — the branch exists for a future model, and an untested branch in
    a scatter-back loop is exactly where an off-by-one hides.
    """
    from ConfigSpace import (Categorical, ConfigurationSpace,
                             ForbiddenAndConjunction, ForbiddenEqualsClause, Integer)

    cs = ConfigurationSpace(seed=0)
    cs.add([Integer("a", (1, 3), default=1), Categorical("b", ["x", "y"], default="x")])
    cs.add(ForbiddenAndConjunction(ForbiddenEqualsClause(cs["a"], 2),
                                   ForbiddenEqualsClause(cs["b"], "y")))

    trials = [
        TrialResult(trial=1, config={"a": 1, "b": "x"}, scores={"accuracy": 0.5},
                    score=0.5, incumbent_score=0.5, incumbent_config={}),
        TrialResult(trial=2, config={"a": 3, "b": "y"}, scores={"accuracy": 0.7},
                    score=0.7, incumbent_score=0.7, incumbent_config={}),
        TrialResult(trial=3, config={"a": 1, "b": "y"}, scores={"accuracy": 0.6},
                    score=0.6, incumbent_score=0.6, incumbent_config={}),
    ]

    grid, ice_lines, pdp, warning = RandomOptimizer().compute_partial_dependence(
        cs, trials, "accuracy", "a", seed=0)

    assert warning is None
    assert grid == [1, 2, 3]
    # b="x" permits every value of a; b="y" forbids a=2 and only a=2.
    assert all(v is not None for v in ice_lines[0])
    for row in ice_lines[1:]:
        assert row[grid.index(2)] is None
        assert row[grid.index(1)] is not None and row[grid.index(3)] is not None
    # The mean at the hole skips the missing rows instead of counting them as 0.
    assert pdp[grid.index(2)] == ice_lines[0][grid.index(2)]


def test_the_explanation_games_are_drawn_with_the_experiments_own_seed(iris_splits, metrics):
    """The last thing that was not.

    Each game's searcher draws 10,000 configurations to play against, and that
    draw used to be seeded at HyperSHAP's own default of 0 — so two experiments
    with different seeds explained themselves from the same sample. Everything
    else about a run already follows the seed: the split, the model, the search,
    the surrogate. This did not.

    Asserted at the searcher rather than on the numbers, because on a small
    space 10,000 draws find the same maximum whatever the seed: the values
    coincide here and would not on a wider one. What is being fixed is which
    determinism it is, so what matters is that the experiment's seed is the one
    that arrives.
    """
    import hypershap.utils

    cs = RandomForestModel().get_config_space(seed=0)
    trials = _trials(iris_splits, metrics)
    opt = RandomOptimizer()
    seen = []
    real = hypershap.utils.RandomConfigSpaceSearcher

    def spy(*args, **kwargs):
        seen.append(kwargs.get("seed"))
        return real(*args, **kwargs)

    with patch.object(hypershap.utils, "RandomConfigSpaceSearcher", spy):
        opt.compute_hp_games(cs, trials, metrics, seed=7)

    assert seen, "the searcher is built at all"
    assert set(seen) == {7}, "and every game gets the experiment's seed"
