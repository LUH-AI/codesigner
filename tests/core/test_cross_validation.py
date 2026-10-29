"""A k-fold trial needs k fits, and the average steadies the noisy estimate of
one 80/20 split. Holdout is the one-fold case of the same fold list.
"""

import numpy as np
import pytest

from core.metrics import metrics_for, resolve
from core.optimizers.trial import evaluate_trial
from core.splits import cross_validation, holdout

#: Regression metrics would fail every trial on these class labels.
METRICS = resolve(metrics_for("classification"))

X_TRAIN = np.array([[0.0], [1.0], [2.0], [3.0]])
Y_TRAIN = np.array(["a", "b", "a", "b"], dtype=object)
X_VAL = np.array([[4.0], [5.0]])
Y_VAL = np.array(["a", "b"], dtype=object)


class Echo:
    """A fixed label keeps scores hand-computable."""

    name = "Echo"

    def __init__(self, label="a"):
        self.label = label
        self.seen = []

    def get_config_space(self, seed=0):
        return None

    def fit_predict(self, config, X_train, y_train, X_val, seed=0):
        self.seen.append((len(X_train), len(X_val)))
        return [self.label] * len(X_val)


def test_a_holdout_is_one_fold():
    splits = holdout(X_TRAIN, Y_TRAIN, X_VAL, Y_VAL)

    assert len(splits.folds) == 1


def test_a_holdout_keeps_exactly_the_split_it_was_given():
    """Re-indexing keeps the stratification from `load_splits`."""
    splits = holdout(X_TRAIN, Y_TRAIN, X_VAL, Y_VAL)
    train_idx, val_idx = splits.folds[0]

    assert np.array_equal(splits.X[train_idx], X_TRAIN)
    assert np.array_equal(splits.y[train_idx], Y_TRAIN)
    assert np.array_equal(splits.X[val_idx], X_VAL)
    assert np.array_equal(splits.y[val_idx], Y_VAL)


def test_a_holdout_trial_fits_once():
    """A one-fold trial pays for one fit."""
    model = Echo()

    evaluate_trial(model, {}, holdout(X_TRAIN, Y_TRAIN, X_VAL, Y_VAL), METRICS)

    assert model.seen == [(4, 2)]


def test_a_holdout_trial_scores_what_it_always_did():
    """One of the two "a" predictions is right."""
    scores, run_info = evaluate_trial(
        Echo(), {}, holdout(X_TRAIN, Y_TRAIN, X_VAL, Y_VAL), METRICS)

    assert scores["accuracy"] == 0.5
    assert run_info["status"] == 1


def _dataset(n=12):
    X = np.arange(n, dtype=float).reshape(-1, 1)
    y = np.array(["a" if i % 2 else "b" for i in range(n)], dtype=object)
    return X, y


def test_every_row_is_held_out_exactly_once():
    X, y = _dataset()
    splits = cross_validation(X, y, folds=4, seed=0)

    held_out = np.concatenate([val for _, val in splits.folds])

    assert sorted(held_out.tolist()) == list(range(len(X)))


def test_a_row_never_trains_and_validates_in_the_same_fold():
    X, y = _dataset()

    for train_idx, val_idx in cross_validation(X, y, folds=4, seed=0).folds:
        assert not set(train_idx) & set(val_idx)


def test_the_division_is_reproducible_from_the_seed():
    X, y = _dataset()

    first = cross_validation(X, y, folds=4, seed=7).folds
    again = cross_validation(X, y, folds=4, seed=7).folds
    other = cross_validation(X, y, folds=4, seed=8).folds

    assert all(np.array_equal(a[1], b[1]) for a, b in zip(first, again))
    assert not all(np.array_equal(a[1], b[1]) for a, b in zip(first, other))


def test_a_trial_fits_once_per_fold():
    X, y = _dataset()
    model = Echo()

    evaluate_trial(model, {}, cross_validation(X, y, folds=4, seed=0), METRICS)

    assert len(model.seen) == 4
    assert all(n_val == 3 for _, n_val in model.seen)


def test_the_trial_score_is_the_mean_across_folds():
    """Alternating labels make "a" right half the time, and each metric stays
    one number."""
    X, y = _dataset()

    scores, _ = evaluate_trial(Echo(), {}, cross_validation(X, y, folds=4, seed=0), METRICS)

    assert scores["accuracy"] == pytest.approx(0.5)
    assert set(scores) == set(METRICS)


def test_a_fold_that_fails_fails_the_trial():
    """A configuration dying on one fold gives the search no partial success."""
    X, y = _dataset()

    class DiesOnTheThirdFold(Echo):
        def fit_predict(self, config, X_train, y_train, X_val, seed=0):
            self.seen.append(1)
            if len(self.seen) == 3:
                raise RuntimeError("not on this fold")
            return ["a"] * len(X_val)

    scores, run_info = evaluate_trial(
        DiesOnTheThirdFold(), {}, cross_validation(X, y, folds=4, seed=0), METRICS)

    assert scores == {name: 0.0 for name in METRICS}
    assert "not on this fold" in run_info["additional_info"]["error"]


def test_a_continuous_target_falls_back_to_a_plain_shuffle():
    """sklearn raises when asked to stratify a regression target, and the split must
    proceed anyway."""
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.linspace(0.0, 1.0, 20)

    splits = cross_validation(X, y, folds=5, seed=0)

    assert len(splits.folds) == 5
    assert sorted(np.concatenate([v for _, v in splits.folds]).tolist()) == list(range(20))


def test_a_rare_class_is_stratified_anyway_rather_than_refused():
    """sklearn warns about a class smaller than the fold count yet still
    stratifies."""
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = np.array(["a"] * 9 + ["rare"], dtype=object)

    splits = cross_validation(X, y, folds=5, seed=0)

    assert len(splits.folds) == 5


def test_an_experiment_carries_its_folds_from_the_snapshot(tmp_path):
    from core.io import build_experiment
    from core.optimizers import RandomOptimizer
    from core.models.random_forest import RandomForestModel
    from tests.conftest import DATASETS_DIR

    snapshot = {
        "version": "0.1.0", "name": "cv", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": list(METRICS), "seed": 0, "cv_folds": 5,
        "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }
    _, exp = build_experiment(
        snapshot, METRICS, {"Random Forest": RandomForestModel()},
        {"Random Search": RandomOptimizer})

    assert exp["cv_folds"] == 5
    assert len(exp["splits"].folds) == 5


def test_an_experiment_without_the_field_is_a_holdout(tmp_path):
    """Absent `cv_folds` means one split."""
    from core.io import build_experiment
    from core.optimizers import RandomOptimizer
    from core.models.random_forest import RandomForestModel
    from tests.conftest import DATASETS_DIR

    snapshot = {
        "version": "0.1.0", "name": "old", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": list(METRICS), "seed": 0,
        "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }
    _, exp = build_experiment(
        snapshot, METRICS, {"Random Forest": RandomForestModel()},
        {"Random Search": RandomOptimizer})

    assert exp["cv_folds"] == 0
    assert len(exp["splits"].folds) == 1


def test_an_optimizer_runs_a_cross_validated_search(optimizers, models):
    from core.datasets import load_frame
    from tests.conftest import DATASETS_DIR

    X, y = load_frame(DATASETS_DIR / "iris.csv")
    splits = cross_validation(X, y, folds=3, seed=0)

    result = optimizers["Random Search"]().optimize(
        models["Random Forest"], None, None, None, None,
        metrics=METRICS, primary_metric="accuracy", n_trials=3, seed=0,
        splits=splits)

    assert len(result.trials) == 3
    # Folds average inside the trial.
    assert all(set(t.scores) == set(METRICS) for t in result.trials)
    # The status check matters because a crashed trial also scores within [0,
    # 1].
    assert all(t.run_info["status"] == 1 for t in result.trials)
    assert all(t.scores["accuracy"] > 0.5 for t in result.trials)


@pytest.mark.parametrize("dataset,kind", [("iris.csv", "string"), ("wine.csv", "integer")])
def test_the_target_keeps_the_type_the_file_gave_it(dataset, kind):
    """Object-dtype integer labels make `type_of_target` answer "unknown", and
    every classifier fails to fit."""
    from core.datasets import load_frame, load_splits
    from tests.conftest import DATASETS_DIR

    path = DATASETS_DIR / dataset
    X, y = load_frame(path)
    X_train, X_val, y_train, y_val = load_splits(path, 0)

    assert holdout(X_train, y_train, X_val, y_val).y.dtype == y.dtype, kind
    assert cross_validation(X, y, folds=3, seed=0).y.dtype == y.dtype, kind


@pytest.mark.parametrize("dataset", ["iris.csv", "wine.csv"])
@pytest.mark.parametrize("folds", [0, 5])
def test_a_real_model_fits_both_demo_datasets_either_way(dataset, folds):
    """A real trial succeeds and scores above zero."""
    from core.datasets import load_frame, load_splits
    from core.models.random_forest import RandomForestModel
    from tests.conftest import DATASETS_DIR

    path = DATASETS_DIR / dataset
    model = RandomForestModel()
    config = dict(model.get_config_space(seed=0).sample_configuration())

    if folds:
        splits = cross_validation(*load_frame(path), folds=folds, seed=0)
    else:
        X_train, X_val, y_train, y_val = load_splits(path, 0)
        splits = holdout(X_train, y_train, X_val, y_val)

    scores, run_info = evaluate_trial(model, config, splits, METRICS, seed=0)

    assert run_info["status"] == 1, run_info.get("additional_info")
    assert scores["accuracy"] > 0.0
