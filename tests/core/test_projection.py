"""Projecting configuration space, where a distance is a distance.

The configuration cube plots literal hyperparameter values, so position means
something and distance does not — the axes carry unrelated units. PCA and PLS
project the whole space instead, and the whole claim they make is that two
points near each other are two configurations near each other.

That claim rests entirely on the encoding, which is why most of this is about
the encoding: a projection of a badly encoded matrix is not a wrong number, it
is a plausible picture of nothing.
"""

import math

import pytest

from core.optimizers import TrialResult
from core.projection import encode_configurations, project


def _trials(configs, scores=None):
    scores = scores or [0.5] * len(configs)
    return [TrialResult(trial=i + 1, config=config, scores={"accuracy": score},
                        score=score, incumbent_score=score, incumbent_config={})
            for i, (config, score) in enumerate(zip(configs, scores))]


class _Hyperparameter:
    def __init__(self, log):
        self.log = log


class _Space(dict):
    """The two things a config space is asked here: membership, and `.log`."""


def _space(**logs):
    return _Space({name: _Hyperparameter(log) for name, log in logs.items()})


# ── the encoding ─────────────────────────────────────────────────────────────

def test_a_log_hyperparameter_is_encoded_logarithmically():
    """`C` on the SVM model is sampled log-uniformly over 0.01 to 100. Left
    linear, three quarters of a real run sits within a hundredth of the rest and
    the projection reports the few large values as the only structure there is."""
    trials = _trials([{"C": 0.01}, {"C": 1.0}, {"C": 100.0}])

    rows, labels = encode_configurations(_space(C=True), trials)

    assert labels == ["log10(C)"]
    # -2, 0, 2 standardised: evenly spaced, which linear encoding would not be
    assert [row[0] for row in rows] == pytest.approx([-math.sqrt(1.5), 0, math.sqrt(1.5)])


def test_a_categorical_becomes_one_column_per_value():
    """Not an integer code, which is what parallel coordinates uses because it
    needs one column per axis to draw. A code would put `linear` one unit from
    `poly` and two from `rbf` — an ordering and a distance that do not exist,
    which is exactly what this figure would then be drawing."""
    trials = _trials([{"kernel": "rbf"}, {"kernel": "linear"}, {"kernel": "rbf"}])

    rows, labels = encode_configurations(None, trials)

    assert labels == ["kernel=linear", "kernel=rbf"]
    # the two columns are each other's opposite, which is the only relationship
    # between them that is true
    assert [row[0] for row in rows] == pytest.approx([-r[1] for r in rows])


def test_a_boolean_is_a_category_not_a_number():
    """`bool` is a subclass of `int` in Python, so it would otherwise arrive as
    1 and 0 — two categories a unit apart on a scale nothing else is on."""
    trials = _trials([{"bootstrap": True}, {"bootstrap": False}])

    _rows, labels = encode_configurations(None, trials)

    assert labels == ["bootstrap=False", "bootstrap=True"]


def test_every_column_is_standardised():
    """Without it PCA reports whichever hyperparameter has the widest raw range:
    `n_estimators` over 10-500 would drown `min_samples_split` over 2-10
    whatever either did to the score."""
    trials = _trials([{"n_estimators": 10, "min_samples_split": 2},
                      {"n_estimators": 500, "min_samples_split": 10},
                      {"n_estimators": 255, "min_samples_split": 6}])

    rows, _labels = encode_configurations(None, trials)

    for column in zip(*rows):
        assert sum(column) == pytest.approx(0, abs=1e-9)
        assert math.sqrt(sum(v * v for v in column) / len(column)) == pytest.approx(1)


def test_a_hyperparameter_that_never_varied_is_left_at_zero():
    """It has no spread to divide by, and dividing anyway is a crash or an
    infinity. Zero is the honest answer: it says nothing, because it did."""
    trials = _trials([{"a": 5, "b": 1}, {"a": 5, "b": 2}, {"a": 5, "b": 3}])

    rows, _labels = encode_configurations(None, trials)

    assert [row[0] for row in rows] == [0.0, 0.0, 0.0]


# ── the projections ──────────────────────────────────────────────────────────

def _spread(n=12):
    """Trials whose configurations vary mostly along `a`, with the score driven
    entirely by `b` — so the two methods have different right answers."""
    configs = [{"a": i * 10.0, "b": float(i % 3)} for i in range(n)]
    scores = [0.5 + 0.1 * (i % 3) for i in range(n)]
    return _trials(configs, scores), scores


def test_pca_orders_its_components_by_how_much_they_explain():
    trials, _scores = _spread()

    coordinates, labels, warning = project(None, trials, _spread()[1], "pca")

    assert warning is None
    assert len(coordinates) == len(trials)
    assert len(labels) == len(coordinates[0]) == 2, "two hyperparameters, two components"
    shares = [int(label.split("(")[1].rstrip("%)")) for label in labels]
    assert shares == sorted(shares, reverse=True)
    assert labels[0].startswith("PC 1")


def test_pls_finds_the_direction_that_moves_the_metric():
    """The entire reason for offering both. PCA looks at the configurations
    alone and reports where the search spread out; PLS is fitted against the
    score, so its first component is the direction that changes it — and here
    those are deliberately different directions."""
    trials, scores = _spread()

    pca, _labels, _w = project(None, trials, scores, "pca")
    pls, _labels, _w = project(None, trials, scores, "pls")

    def correlation(coordinates):
        first = [row[0] for row in coordinates]
        mean, smean = sum(first) / len(first), sum(scores) / len(scores)
        cov = sum((a - mean) * (b - smean) for a, b in zip(first, scores))
        spread = (math.sqrt(sum((a - mean) ** 2 for a in first))
                  * math.sqrt(sum((b - smean) ** 2 for b in scores)))
        return abs(cov / spread)

    assert correlation(pls) > correlation(pca)


def test_a_projection_is_a_plane_at_most_three_deep():
    """Three is what can be drawn; a fourth component would be computed for
    nobody."""
    configs = [{f"h{j}": float((i * (j + 1)) % 7) for j in range(6)} for i in range(20)]
    coordinates, labels, warning = project(None, _trials(configs), [0.5] * 20, "pca")

    assert warning is None
    assert len(labels) == len(coordinates[0]) == 3


# ── when there is nothing to say ─────────────────────────────────────────────

def test_too_few_trials_says_so_rather_than_drawing():
    coordinates, labels, warning = project(
        None, _trials([{"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 1.0}]), [0.5, 0.6], "pca")

    assert not coordinates and not labels
    assert "few trials" in warning


def test_one_hyperparameter_is_not_a_space_to_project():
    """There is nothing to reduce: the one axis it has is already the picture,
    and the cube's own view draws it better."""
    trials = _trials([{"a": 1.0}, {"a": 2.0}, {"a": 3.0}])

    coordinates, _labels, warning = project(None, trials, [0.5, 0.6, 0.7], "pca")

    assert not coordinates
    assert "more than one hyperparameter" in warning


def test_trials_that_are_all_the_same_configuration_have_no_directions():
    """No column varied, so there is no direction to find — and sklearn does not
    say so, it returns a plane of NaNs and a divide-by-zero warning. Caught here
    rather than drawn."""
    trials = _trials([{"a": 1.0, "b": 2.0}] * 4)

    coordinates, _labels, warning = project(None, trials, [0.5] * 4, "pca")

    assert not coordinates
    assert "same configuration" in warning


def test_an_unknown_method_is_refused_rather_than_guessed_at():
    """It used to fall through to PCA, which would have drawn a picture under
    the wrong name — the one failure mode worse than no picture."""
    trials = _trials([{"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 1.0}, {"a": 3.0, "b": 4.0}])

    coordinates, _labels, warning = project(None, trials, [0.5, 0.6, 0.7], "mds")

    assert not coordinates
    assert "No such projection" in warning
