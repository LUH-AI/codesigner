"""An experiment stored with a trial missing one metric still renders.

`io.parse` refuses such a file now (tests/core/test_trial_scores_validation.py),
but rows written before that check exist, and the fix has to cover them: the
detail page used to 500 on an argmax five frames down, with nothing to say
which experiment or which metric was at fault.

The incomplete metric loses its figures — a curve, a colour array and a
parallel-coordinates axis all need one value per trial, and there is no honest
partial version — and says so through the empty captions the page already has.
Every other metric is untouched.
"""

from core import io
from ui.services import snapshot as adapter

from tests.conftest import FIXTURES_DIR


def _experiment_missing_one_score():
    """The Random Forest fixture with trial 4's f1 removed, written straight to
    the adapter so it bypasses the parse check the way an old row does."""
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
    """f1 is not the metric the page opens on, so it comes from the switcher's
    own endpoint — which has to reach the same conclusion the page would."""
    exp = _experiment_missing_one_score()
    plots = client.get(f"/experiments/{exp.pk}/figures/?metric=f1").json()

    assert plots["performance_over_time"]["trial-score"] is None
    assert plots["configuration_cube"] == {"axes": None, "pca": None, "pls": None}
    assert plots["parallel_coordinates"] is None


def test_the_panel_still_describes_a_best_trial_for_the_complete_metrics(client):
    exp = _experiment_missing_one_score()
    panels = {p["metric"]: p for p in client.get(f"/experiments/{exp.pk}/").context["panels"]}

    assert panels["accuracy"]["best_idx"] is not None
    # f1's best is chosen from the trials that have an f1, not from all of them.
    assert panels["f1"]["best_idx"] != 3
