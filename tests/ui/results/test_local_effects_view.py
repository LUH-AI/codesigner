"""The page fetches the beeswarm on request and caps it by
`local_effects_max_trials`, because each trial costs one ablation game. The
beeswarm shows how each hyperparameter's effect spreads across trials.
"""

import pytest
from django.urls import reverse

from core import io
from experiments.models import GlobalSettings
from ui.views.analytics import _local_effects_data

from tests.conftest import FIXTURES_DIR


def _experiment(**settings):
    from experiments.services import snapshot as adapter
    exp = adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "analytics.ihpo").read_bytes()))
    if settings:
        exp.use_default_settings = False
        exp.settings = settings
        exp.save(update_fields=["use_default_settings", "settings"])
    return exp


def _url(pk, metric):
    return reverse("ui:local_effects", args=[pk]) + f"?metric={metric}"


def test_no_model_reports_unavailable_instead_of_crashing():
    """Without a config space the view returns a warning instead of a 500."""
    figure, warning = _local_effects_data({"model": None}, "accuracy")
    assert figure is None
    assert warning and "model" in warning.lower()


def test_it_returns_one_row_of_points_per_hyperparameter(client):
    exp = _experiment()
    data = client.get(_url(exp.pk, "accuracy")).json()

    assert data["warning"] is None
    assert exp.data.result is not None
    hp_count = len(next(iter(exp.data.result["configs"].values())))
    assert len(data["figure"]["data"]) == hp_count
    assert data["figure"]["layout"]["yaxis"]["ticktext"], "rows are named"


def test_every_sampled_trial_becomes_a_point(client):
    exp = _experiment()
    data = client.get(_url(exp.pk, "accuracy")).json()

    assert exp.data.result is not None
    n_trials = len(exp.data.result["data"])
    for trace in data["figure"]["data"]:
        assert len(trace["x"]) == n_trials


def test_the_cap_bounds_how_many_trials_are_explained(client):
    """The cap sets the figure's runtime, since each trial costs an ablation
    game."""
    exp = _experiment(local_effects_max_trials=2)
    data = client.get(_url(exp.pk, "accuracy")).json()

    assert exp.data.result is not None
    assert len(exp.data.result["data"]) > 2, "the fixture has more than the cap"
    for trace in data["figure"]["data"]:
        assert len(trace["x"]) == 2


def test_rejects_unknown_metric(client):
    exp = _experiment()
    assert client.get(_url(exp.pk, "not-a-real-metric")).status_code == 400


def test_rejects_when_experiment_has_no_result(client):
    from experiments.services import snapshot as adapter

    exp = adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "no-result-yet", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy"], "seed": 0, "dataset_path": "", "result": None,
    })
    assert client.get(_url(exp.pk, "accuracy")).status_code == 400


def test_unknown_experiment_404s(client):
    assert client.get(_url(999999, "accuracy")).status_code == 404


def test_the_page_ships_a_compute_button_and_no_data(client):
    """Autocompute defaults off, so page load computes nothing."""
    exp = _experiment()
    body = client.get(f"/experiments/{exp.pk}/").content.decode()

    assert 'id="local-effects-compute-btn"' in body
    assert "refreshLocalEffects" in body


def test_each_row_says_which_trial_it_explains(client):
    """The figure draws a sample, and its selection contract maps each point to
    the trial's position in the result.
    """
    exp = _experiment(local_effects_max_trials=2)
    data = client.get(_url(exp.pk, "accuracy")).json()

    selection = data["figure"]["layout"]["meta"]["selection"]
    assert exp.data.result is not None
    n_trials = len(exp.data.result["data"])
    sampled = selection["trials"]["0"]

    assert len(sampled) == 2
    assert sampled == [0, n_trials - 1], "evenly sampled: the ends are kept"
    assert len(selection["trials"]) == len(data["figure"]["data"]), "one row each"
    assert all(row == sampled for row in selection["trials"].values()), \
        "the same trials on every row, so one click lights up a column"
