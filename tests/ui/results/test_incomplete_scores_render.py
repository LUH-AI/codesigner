"""`io.parse` refuses a trial missing one metric, but old rows hold some. That
metric loses its figures while the others still render.
"""

from core import io
from experiments.services import snapshot as adapter

from tests.conftest import FIXTURES_DIR


def _experiment_missing_one_score():
    snapshot = io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes())
    snapshot["result"]["data"][3]["scores"] = {
        k: v for k, v in snapshot["result"]["data"][3]["scores"].items() if k != "f1"
    }
    return adapter.experiment_from_snapshot(snapshot)


def test_the_detail_page_still_renders(client):
    exp = _experiment_missing_one_score()
    assert client.get(f"/experiments/{exp.pk}/").status_code == 200


def test_the_complete_metrics_still_draw(client):
    exp = _experiment_missing_one_score()
    plots = client.get(f"/experiments/{exp.pk}/").context["metric_plots"]

    assert plots["accuracy"]["performance_over_time"]["trial-score"]["data"]
    assert plots["accuracy"]["configuration_cube"]["axes"]["data"]
    assert plots["accuracy"]["parallel_coordinates"]["data"]


def test_the_incomplete_metric_draws_nothing_rather_than_a_partial_curve(client):
    """f1 comes from the switcher's endpoint, which must agree with the page."""
    exp = _experiment_missing_one_score()
    plots = client.get(f"/experiments/{exp.pk}/figures/?metric=f1").json()

    assert plots["performance_over_time"]["trial-score"] is None
    assert plots["configuration_cube"] == {"axes": None, "pca": None, "pls": None}
    assert plots["parallel_coordinates"] is None


def test_the_panel_still_describes_a_best_trial_for_the_complete_metrics(client):
    exp = _experiment_missing_one_score()
    panels = {p["metric"]: p for p in client.get(f"/experiments/{exp.pk}/").context["panels"]}

    assert panels["accuracy"]["best_idx"] is not None
    # f1's best comes from trials that have an f1.
    assert panels["f1"]["best_idx"] != 3
