"""Priors drawn in the configuration cube: previewed, stated, edited and
withdrawn through the server, which works out everything a belief is — the
record, its weight over the cube's sample, the shapes drawn for it — so the
page computes none of it.

Uses the test.ihpo fixture (SMAC, Random Forest), whose trials give the cube a
sample and a PLS projection to draw in.
"""

import json

import pytest
from django.urls import reverse

from core import io
from experiments.services import snapshot as adapter

from tests.conftest import FIXTURES_DIR, export_ihpo


def _experiment():
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test.ihpo").read_bytes()))


def _post(client, exp, name, body):
    return client.post(reverse(f"ui:{name}", args=[exp.pk]), data=json.dumps(body),
                       content_type="application/json")


def _sample(exp):
    from ui.views import _cube_sample

    _built, config_space, sample = _cube_sample(exp, "accuracy")
    return config_space, sample


def _pls_points(exp):
    """Two points inside the PLS view: two sampled configurations' places."""
    _cs, sample = _sample(exp)
    coordinates = sample["projections"]["pls"]["coordinates"]
    return [coordinates[0][:2], coordinates[1][:2]]


def _draft(exp, **extra):
    return dict({"view": "pls", "points": _pls_points(exp)}, **extra)


def test_a_preview_weighs_the_whole_sample_and_says_what_to_draw(client):
    """A ridge drawn in PLS: one weight per sampled configuration, the defaults
    filled in, and the line, band and spread marks to draw — in the view's
    units.

    Expect: weights as long as the sample, peaking at most 1; a record labelled
    "PLS · 2D · 2 points" with a position, a spread and a width; a line and a
    two-edged band in two dimensions.
    """
    exp = _experiment()
    _cs, sample = _sample(exp)

    data = _post(client, exp, "projected_prior_preview",
                 {"metric": "accuracy", "draft": _draft(exp)}).json()

    assert len(data["weights"]) == len(sample["vectors"])
    assert 0 < max(data["weights"]) <= 1
    record = data["record"]
    assert record["label"] == "PLS · 2D · 2 points"
    assert len(record["params"]["mean"]) == 1 and record["params"]["width"] > 0
    geometry = data["geometry"]
    assert len(geometry["lines"]) == 1 and len(geometry["band"]) == 2
    assert len(geometry["ticks"]) == 2


def test_hp_space_points_go_onto_the_vectorized_axis_and_back(client):
    """In HP Space the page sends values in the hyperparameters' own units; the
    record holds them on the vectorized axis every prior is stated on, and the
    shapes come back in the units they went in.

    Expect: the record's point at the values' vectorized positions, and the
    drawn point at the values sent.
    """
    exp = _experiment()
    config_space, _sample_ = _sample(exp)
    draft = {"view": "axes", "hyperparameters": ["max_depth", "max_features"],
             "points": [[26, 0.55]]}

    data = _post(client, exp, "projected_prior_preview",
                 {"metric": "accuracy", "draft": draft}).json()

    expected = [float(config_space["max_depth"].to_vector(26)),
                float(config_space["max_features"].to_vector(0.55))]
    assert data["record"]["points"][0] == pytest.approx(expected)
    assert data["geometry"]["points"][0] == pytest.approx([26, 0.55])


def test_a_shape_that_is_not_one_is_refused_with_the_reason(client):
    exp = _experiment()
    point = _pls_points(exp)[0]
    response = _post(client, exp, "projected_prior_preview",
                     {"metric": "accuracy", "draft": {"view": "pls", "points": [point, point]}})

    assert response.status_code == 400
    assert response.json()["error"]


def test_stating_editing_and_withdrawing_one(client):
    """Stated, it is stored with an id, the trial it was stated at and its
    decay, and recorded; edited before any run, the edit folds into that one
    entry; withdrawn before any run, nothing is left in the history at all —
    the same folding a hyperparameter's prior has.
    """
    exp = _experiment()
    stated = _post(client, exp, "state_projected_prior",
                   {"metric": "accuracy", "draft": _draft(exp),
                    "decay": {"shape": "linear", "beta_ratio": 0.2}}).json()
    exp.refresh_from_db()
    record = exp.data.projected_priors[0]
    assert record["id"] and record["at_trial"] == len(exp.data.result["data"])
    assert record["decay"]["shape"] == "linear"
    assert len(stated["stated"]) == 1 and stated["weights"]
    assert [e.kind for e in exp.data.history.all()] == ["projected_prior_stated"]

    _post(client, exp, "state_projected_prior",
          {"metric": "accuracy", "id": record["id"], "params": {"width": 0.5}})
    exp.refresh_from_db()
    assert exp.data.projected_priors[0]["params"]["width"] == 0.5
    assert exp.data.projected_priors[0]["points"] == record["points"]
    assert exp.data.history.count() == 1

    _post(client, exp, "withdraw_projected_prior", {"metric": "accuracy", "id": record["id"]})
    exp.refresh_from_db()
    assert exp.data.projected_priors == []
    assert exp.data.history.count() == 0


def test_the_cube_and_the_acquisition_figure_are_weighed_by_it(client):
    """Once stated, the cube's sample and the acquisition figure's slice and
    cloud carry its weight, one number per configuration they predicted at."""
    exp = _experiment()
    layer_url = reverse("ui:surrogate_prediction", args=[exp.pk]) + "?metric=accuracy"
    assert client.get(layer_url).json()["projected_weights"] is None

    _post(client, exp, "state_projected_prior", {"metric": "accuracy", "draft": _draft(exp)})
    layer = client.get(layer_url).json()
    slice_url = reverse("ui:acquisition_slice", args=[exp.pk]) + "?metric=accuracy&hp=max_features"
    meta = client.get(slice_url).json()["meta"]

    assert len(layer["projected_weights"]) == len(layer["vectors"])
    assert len(meta["projectedWeights"]["slice"]) == len(meta["positions"])
    assert len(meta["projectedWeights"]["cloud"]) == len(meta["cloud"]["positions"])


def test_a_run_and_the_file_keep_them(client):
    """A run copies what it searched under, and the file carries both the
    standing belief and each run's copy back in."""
    from experiments.services.run import create_run

    exp = _experiment()
    _post(client, exp, "state_projected_prior", {"metric": "accuracy", "draft": _draft(exp)})
    exp.refresh_from_db()
    run = create_run(exp, {"max_trials": 1}, "accuracy")

    body = json.loads(export_ihpo(client, exp.pk).content)
    body["name"] = "reopened"
    copy = adapter.experiment_from_snapshot(io.parse(json.dumps(body).encode()))

    assert run.projected_priors == exp.data.projected_priors
    assert copy.data.projected_priors == exp.data.projected_priors
    assert copy.runs.order_by("-id").first().projected_priors == exp.data.projected_priors


def test_the_timeline_says_where_it_was_drawn(client):
    from ui import timeline

    exp = _experiment()
    _post(client, exp, "state_projected_prior", {"metric": "accuracy", "draft": _draft(exp)})
    entries = [e for e in timeline.build(exp) if e["kind"] == "projected_prior_stated"]

    assert entries and entries[0]["summary"] == "Prior drawn in PLS · 2D · 2 points."


def test_only_a_reader_who_may_edit_can_state_one(client, settings):
    """Previewing is looking; stating changes what the next run searches."""
    from ui import views

    assert views.projected_prior_preview.experiment_action == "view"
    assert views.state_projected_prior.experiment_action == "edit"
    assert views.withdraw_projected_prior.experiment_action == "edit"


def test_what_runs_next_is_asked_under_it(client):
    """Requery hands the stated projected priors to the optimizer it asks."""
    exp = _experiment()
    _post(client, exp, "state_projected_prior", {"metric": "accuracy", "draft": _draft(exp)})
    response = client.get(reverse("ui:next_trials", args=[exp.pk]))

    assert response.status_code == 200
    assert response.json()["candidates"]


def test_the_cube_offers_drawing_where_a_prior_can_steer(client):
    """Draw Prior is on the cube for an optimizer a prior can steer, and the
    card lists what is stated, served with the page."""
    exp = _experiment()
    _post(client, exp, "state_projected_prior", {"metric": "accuracy", "draft": _draft(exp)})
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="cube-draw-prior"' in html
    stated = json.loads(html.split('id="projected-priors-data"', 1)[1]
                        .split(">", 1)[1].split("</script>", 1)[0])
    assert [s["label"] for s in stated] == ["PLS · 2D · 2 points"]


def test_random_search_has_no_prior_to_draw(client):
    """A search that fits no model has no acquisition for a prior to weight."""
    exp = adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'id="cube-draw-prior"' not in html


def test_a_point_outside_the_search_space_is_refused_and_named(client):
    """A point that is no configuration at all — beyond a hyperparameter's
    range in HP Space, beyond the image of the space in a projection — is
    refused, saying which point, so the cube can take that one off.

    Expect: 400s naming point 2 in HP Space (max_depth above its upper bound)
    and point 1 in PLS (far outside the outline).
    """
    exp = _experiment()
    config_space, _sample_ = _sample(exp)
    upper = config_space["max_depth"].upper
    axes = _post(client, exp, "projected_prior_preview", {"metric": "accuracy", "draft": {
        "view": "axes", "hyperparameters": ["max_depth", "max_features"],
        "points": [[10, 0.5], [upper + 20, 0.5]]}})
    pls = _post(client, exp, "projected_prior_preview", {"metric": "accuracy", "draft": {
        "view": "pls", "points": [[1000.0, 1000.0]]}})

    assert axes.status_code == 400 and axes.json()["point"] == 1
    assert "outside" in axes.json()["error"]
    assert pls.status_code == 400 and pls.json()["point"] == 0


def test_a_preview_draws_the_prior_in_its_own_frame(client):
    exp = _experiment()
    data = _post(client, exp, "projected_prior_preview",
                 {"metric": "accuracy", "draft": _draft(exp)}).json()

    assert data["profile"]["kind"] == "ridge"


def test_a_click_is_read_once_however_often_the_cube_is_redrawn(client):
    """A change of dimensionality replots the cube and wires its clicks again
    on the same element; the last wiring's listeners come off first, or one
    click is read once per wiring — which put both of a ridge's points on the
    second. Asserted on the page script."""
    html = client.get(reverse("ui:experiment_detail", args=[_experiment().pk])).content.decode()
    wiring = html.split("function attachSelectionClicks(", 1)[1].split("(function attachTableClicks", 1)[0]

    assert "el.removeEventListener(pair[0], pair[1]);" in wiring
    assert wiring.index("removeEventListener") < wiring.index("addEventListener")
