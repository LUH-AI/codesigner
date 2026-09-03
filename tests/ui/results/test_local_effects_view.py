"""The beeswarm of every sampled trial's local effect.

The most expensive thing an experiment page can ask for: one HyperSHAP ablation
game per trial, each 2^n_hp coalitions. So it is fetched rather than shipped,
capped by `local_effects_max_trials`, and — like every deferred computation now
— waits to be asked unless its autocompute setting says otherwise.

What it shows that nothing else does is *spread*. Every other explanation on the
page is one number per hyperparameter; this one distinguishes a hyperparameter
that reliably helps a little from one that helps enormously in half the space
and hurts in the other.
"""

import pytest
from django.urls import reverse

from core import io
from ui.models import GlobalSettings
from ui.views import _local_effects_data

from tests.conftest import FIXTURES_DIR


def _experiment(**settings):
    from ui.services import snapshot as adapter
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
    """A custom-model experiment viewed read-only has no config space to
    explain against — a warning, not a 500, like its two siblings."""
    figure, warning = _local_effects_data({"model": None}, "accuracy")
    assert figure is None
    assert warning and "model" in warning.lower()


def test_it_returns_one_row_of_points_per_hyperparameter(client):
    exp = _experiment()
    data = client.get(_url(exp.pk, "accuracy")).json()

    assert data["warning"] is None
    hp_count = len(next(iter(exp.result["configs"].values())))
    assert len(data["figure"]["data"]) == hp_count
    assert data["figure"]["layout"]["yaxis"]["ticktext"], "rows are named"


def test_every_sampled_trial_becomes_a_point(client):
    """One point per trial per hyperparameter — the spread is the whole
    figure, so a dropped trial is a dropped observation."""
    exp = _experiment()
    data = client.get(_url(exp.pk, "accuracy")).json()

    n_trials = len(exp.result["data"])
    for trace in data["figure"]["data"]:
        assert len(trace["x"]) == n_trials


def test_the_cap_bounds_how_many_trials_are_explained(client):
    """Each trial costs its own ablation game, so this is the setting that
    decides how long the figure takes rather than only how busy it looks."""
    exp = _experiment(local_effects_max_trials=2)
    data = client.get(_url(exp.pk, "accuracy")).json()

    assert len(exp.result["data"]) > 2, "the fixture has more than the cap"
    for trace in data["figure"]["data"]:
        assert len(trace["x"]) == 2


def test_rejects_unknown_metric(client):
    exp = _experiment()
    assert client.get(_url(exp.pk, "not-a-real-metric")).status_code == 400


def test_rejects_when_experiment_has_no_result(client):
    from ui.services import snapshot as adapter

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
    """Off by default, so opening the page must cost nothing for this figure."""
    exp = _experiment()
    body = client.get(f"/experiments/{exp.pk}/").content.decode()

    assert 'id="local-effects-compute-btn"' in body
    assert "refreshLocalEffects" in body


def test_each_row_says_which_trial_it_explains(client):
    """The figure draws a *sample*, so a row's place in the list says nothing
    about which trial it is — and the page's selection is a position in the
    result. Carrying the position is what lets a click anywhere else light up
    the right point on every hyperparameter's row.

    Asserted through the plot's own selection contract rather than the rows,
    because that is what the page actually reads.
    """
    exp = _experiment(local_effects_max_trials=2)
    data = client.get(_url(exp.pk, "accuracy")).json()

    selection = data["figure"]["layout"]["meta"]["selection"]
    n_trials = len(exp.result["data"])
    sampled = selection["trials"]["0"]

    assert len(sampled) == 2
    assert sampled == [0, n_trials - 1], "evenly sampled: the ends are kept"
    assert len(selection["trials"]) == len(data["figure"]["data"]), "one row each"
    assert all(row == sampled for row in selection["trials"].values()), \
        "the same trials on every row, so one click lights up a column"
