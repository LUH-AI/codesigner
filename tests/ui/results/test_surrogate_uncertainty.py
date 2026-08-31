"""How sure the surrogate is, drawn under the trials.

`docs/plan_today.md` Step 4, which asked for a design pass rather than a sketch.
What the pass decided, and what these tests hold:

- **The estimate is the spread across the forest's own trees.** No new fit and
  no second model: `fit_surrogate` already builds a `RandomForestRegressor` for
  partial dependence and for the games' fallback rung, and each tree in it
  predicts separately. Standard deviation rather than variance, so the number is
  in the metric's own units and a colour bar can be read.
- **A slice through the best trial**, not an average over every trial. The
  average costs `n_trials x n_points^2` rows through every tree instead of
  `n_points^2`, and a plane through the incumbent passes through a configuration
  that was actually evaluated — where a plane through the config-space default
  may sit somewhere the search never went.
- **The axes view at exactly two axes.** `core.projection.project` returns
  coordinates and discards the fitted PCA/PLS, so there is no inverse to carry a
  grid in component space back to configurations. Three axes would be a volume
  through a point cloud; one is a strip with no second dimension to vary in.
- **Deferred**, like partial dependence, because it is per *pair*.
"""

import pytest
from django.urls import reverse

from core.optimizers import RandomOptimizer


@pytest.fixture
def optimizer():
    return RandomOptimizer()


def _space():
    from ConfigSpace import ConfigurationSpace, Float, Integer

    cs = ConfigurationSpace(seed=0)
    cs.add([Integer("n_estimators", (1, 100), default=10),
            Float("alpha", (0.01, 1.0), default=0.5)])
    return cs


def _trials(n=25):
    """A run whose score depends on one hyperparameter and ignores the other, so
    the forest has something to be sure about and something not to be."""
    from core.optimizers.base import TrialResult

    out = []
    for i in range(n):
        config = {"n_estimators": 1 + (i * 4) % 100, "alpha": 0.01 + (i % 10) * 0.099}
        score = config["n_estimators"] / 100.0
        out.append(TrialResult(
            trial=i + 1, config=config, scores={"accuracy": score}, score=score,
            incumbent_score=score, incumbent_config=config,
            run_info={"status": 1, "time": 0.1}, origin=""))
    return out


# ── the computation ─────────────────────────────────────────────────────────

def test_it_returns_a_grid_the_shape_of_its_two_axes(optimizer):
    x, y, z, warning = optimizer.compute_surrogate_uncertainty(
        _space(), _trials(), "accuracy", "n_estimators", "alpha", n_points=8)

    assert warning is None
    assert len(x) > 1 and len(y) > 1
    assert len(z) == len(y), "one row per y value"
    assert all(len(row) == len(x) for row in z), "each row as long as x"


def test_every_value_is_a_spread_and_so_never_negative(optimizer):
    _, _, z, _ = optimizer.compute_surrogate_uncertainty(
        _space(), _trials(), "accuracy", "n_estimators", "alpha", n_points=8)

    values = [v for row in z for v in row if v is not None]
    assert values
    assert all(v >= 0.0 for v in values), "a standard deviation cannot be negative"


def test_it_is_the_trees_disagreeing_and_not_the_prediction(optimizer):
    """The distinction that makes this worth drawing: a field of predictions is
    the metric again, which the points already carry. This is a different
    quantity, and it must not track the score."""
    import numpy as np

    from core.optimizers.base import fit_surrogate

    space, trials = _space(), _trials()
    x, y, z, _ = optimizer.compute_surrogate_uncertainty(
        space, trials, "accuracy", "n_estimators", "alpha", n_points=8)

    rf, _ = fit_surrogate(space, trials, "accuracy", 0)
    from ConfigSpace import Configuration

    best = max(trials, key=lambda t: t.scores["accuracy"])
    values = dict(best.config)
    values["n_estimators"], values["alpha"] = x[0], y[0]
    row = np.array([Configuration(space, values=values).get_array()])

    mean = float(rf.predict(row)[0])
    spread = float(np.stack([t.predict(row) for t in rf.estimators_]).std(axis=0)[0])

    assert z[0][0] == pytest.approx(spread)
    assert z[0][0] != pytest.approx(mean), "this is the disagreement, not the value"


def test_the_slice_is_taken_through_the_best_trial(optimizer):
    """A plane through the incumbent passes through a configuration that was
    actually evaluated. The config-space default may be somewhere the search
    never went, and a field sliced there would report uniform ignorance."""
    space, trials = _space(), _trials()
    best = max(trials, key=lambda t: t.scores["accuracy"])

    # Only one hyperparameter is on the axes; the other is held at the
    # reference, so moving the reference must move the answer.
    _, _, held_at_best, _ = optimizer.compute_surrogate_uncertainty(
        space, trials, "accuracy", "n_estimators", "alpha", n_points=6)

    shifted = [t for t in trials if t.config != best.config]
    _, _, held_elsewhere, _ = optimizer.compute_surrogate_uncertainty(
        space, shifted, "accuracy", "n_estimators", "alpha", n_points=6)

    assert held_at_best != held_elsewhere


def test_too_few_trials_is_a_reason_rather_than_a_crash(optimizer):
    x, y, z, warning = optimizer.compute_surrogate_uncertainty(
        _space(), _trials(1), "accuracy", "n_estimators", "alpha")

    assert warning and (x, y, z) == ([], [], [])


def test_one_hyperparameter_cannot_span_a_plane(optimizer):
    _, _, _, warning = optimizer.compute_surrogate_uncertainty(
        _space(), _trials(), "accuracy", "alpha", "alpha")

    assert "different hyperparameters" in warning


def test_an_unknown_hyperparameter_is_named_in_the_reason(optimizer):
    _, _, _, warning = optimizer.compute_surrogate_uncertainty(
        _space(), _trials(), "accuracy", "n_estimators", "nonesuch")

    assert "nonesuch" in warning


# ── the endpoint ────────────────────────────────────────────────────────────

@pytest.fixture
def experiment(client):
    from tests.ui.storage.test_page_survives_a_round_trip import _ran_experiment

    return _ran_experiment(client)


def _hps(exp):
    return list(exp.result["configs"][next(iter(exp.result["configs"]))])


@pytest.mark.django_db
def test_the_endpoint_answers_with_a_field_and_its_ramp(client, experiment):
    x_hp, y_hp = _hps(experiment)[:2]

    data = client.get(
        reverse("ui:surrogate_uncertainty", args=[experiment.pk])
        + f"?metric=accuracy&x={x_hp}&y={y_hp}").json()

    assert data["warning"] is None
    assert data["z"] and len(data["z"]) == len(data["y"])
    assert data["colorscale"], "the ramp travels with the numbers"


@pytest.mark.django_db
def test_the_ramp_is_neither_the_best_so_far_green_nor_the_metric_blue(client, experiment):
    """`ACCENT_COLOR` means "the best so far" on every figure that uses it, and a
    field underneath the points is not that. `_INTENSITY_SCALE` ramps blue for
    the metric — on this very figure, on the points sitting on top of the
    field."""
    from ui.figures.plots import ACCENT_COLOR, UNCERTAINTY_SCALE

    colours = " ".join(str(stop[1]) for stop in UNCERTAINTY_SCALE)

    assert ACCENT_COLOR.lower() not in colours.lower()
    # rgba(124, 58, 237, ...) — red and blue high, green low: a violet, and one
    # the palette does not otherwise use.
    red, green, blue = 124, 58, 237
    assert green < red < blue, f"expected a violet ramp, got {colours}"
    # Translucent at both ends: it is scenery behind the trials.
    assert all(str(stop[1]).startswith("rgba") for stop in UNCERTAINTY_SCALE)


@pytest.mark.django_db
def test_an_unknown_metric_is_refused(client, experiment):
    resp = client.get(reverse("ui:surrogate_uncertainty", args=[experiment.pk])
                      + "?metric=nonesuch&x=a&y=b")
    assert resp.status_code == 400


@pytest.mark.django_db
def test_an_unusable_pair_answers_with_a_reason_not_an_error(client, experiment):
    """The page shows it as a caption. A 400 would make it look like the request
    was malformed, when what happened is that this pair has nothing to say."""
    x_hp = _hps(experiment)[0]

    resp = client.get(reverse("ui:surrogate_uncertainty", args=[experiment.pk])
                      + f"?metric=accuracy&x={x_hp}&y={x_hp}")

    assert resp.status_code == 200
    assert resp.json()["warning"]
    assert resp.json()["z"] == []


# ── and it is deferred, like partial dependence ─────────────────────────────

def test_it_waits_to_be_asked_by_default():
    """It is per pair — six hyperparameters make fifteen pairs and the reader is
    looking at one — so it gets a Compute button rather than fetching itself."""
    from ui.services.settings import SETTING_DEFAULTS

    assert SETTING_DEFAULTS["autocompute_surrogate_uncertainty"] is False


def test_the_cube_declares_it_so_it_gets_its_own_setting():
    from ui.figures import FIGURES_BY_KEY, deferred_computations

    assert "surrogate_uncertainty" in dict(deferred_computations())
    assert any(name == "surrogate_uncertainty"
               for name, _label in FIGURES_BY_KEY["configuration_cube"].deferred)


@pytest.mark.django_db
def test_the_page_offers_the_prompt_and_the_url(client, experiment):
    body = client.get(reverse("ui:experiment_detail",
                              args=[experiment.pk])).content.decode()

    assert 'id="uncertainty-compute-btn"' in body
    assert reverse("ui:surrogate_uncertainty", args=[experiment.pk]) in body
