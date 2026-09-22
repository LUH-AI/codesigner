"""The acquisition-and-beliefs figure, end to end through its endpoint.

This figure is unusual twice over: it ships no server-rendered plot (like
partial dependence), and it is drawn by its own script rather than by
experiment_detail.html's. Both mean the contract worth testing is the *payload*
— that the endpoint returns a skeleton whose belief and acquisition traces are
empty, and whose `layout.meta.acquisition` carries everything the browser needs
to fill them.

DB access + isolated media come from tests/ui/conftest.py.
"""

import json
from pathlib import Path

from django.urls import reverse

from core import io

from tests.conftest import FIXTURES_DIR


def _experiment():
    from ui.services import snapshot as adapter
    return adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))


def _slice(client, exp, hp, metric="accuracy"):
    response = client.get(reverse("ui:acquisition_slice", args=[exp.pk]),
                          {"metric": metric, "hp": hp})
    return response, (json.loads(response.content) if response.status_code == 200 else None)


def _hp_names(exp):
    """Through the same rebuild the endpoint uses, so the names are the ones it
    will actually accept rather than a guess at the stored shape."""
    from ui.views import _hp_names as names_of, _rebuild_experiment

    built = _rebuild_experiment(exp)
    return names_of(built["result"]) if built and built["result"] else []


def test_the_figure_appears_on_the_page(client):
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert 'data-figure="acquisition_slice"' in html
    assert 'id="figure-acquisition_slice"' in html
    # Its own script, deferred so it runs after the plotly tag further down.
    assert "ui/acquisition.js" in html
    assert "defer" in html


def test_the_page_announces_when_it_has_redrawn(client):
    """The figure draws itself, so the page purging it on a metric switch would
    otherwise wipe it at an unpredictable moment. One CustomEvent closes that,
    in the idiom `poll:swapped` already uses."""
    exp = _experiment()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert "figures:redrawn" in html


def test_the_skeleton_leaves_the_belief_traces_empty(client):
    """The server draws what the run supports; the browser draws what depends on
    a belief it is holding. Four empty traces is that division, in the payload."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    response, data = _slice(client, exp, hp)

    assert response.status_code == 200
    figure = data["figure"]
    assert figure is not None, data.get("warning")

    empty = [i for i, trace in enumerate(figure["data"])
             if all(v is None for v in trace.get("y", [None]))]
    meta = figure["layout"]["meta"]["acquisition"]
    assert empty == sorted(meta["traces"].values())
    assert set(meta["traces"]) == {"fiction", "belief", "acquisition", "weighted"}


def test_the_payload_carries_what_the_browser_needs(client):
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _, data = _slice(client, exp, hp)
    meta = data["figure"]["layout"]["meta"]["acquisition"]

    assert len(meta["positions"]) == len(meta["mu"]) == len(meta["sigma"]) > 1
    assert isinstance(meta["higherIsBetter"], bool)
    assert meta["eta"] is not None
    assert meta["kind"] in {"continuous", "categorical"}
    assert meta["span"][0] < meta["span"][1]


def test_the_x_axis_is_pinned(client):
    """Our own pointer handler converts a pixel to a value from the panel's
    bounding box alone, which only holds while the axis cannot move. Plotly's
    own dragging is off for the same reason."""
    exp = _experiment()
    hp = _hp_names(exp)[0]
    _, data = _slice(client, exp, hp)
    layout = data["figure"]["layout"]

    assert layout["dragmode"] is False
    for axis in ("xaxis", "xaxis2", "xaxis3"):
        assert layout[axis]["fixedrange"] is True
        assert list(layout[axis]["range"]) == list(layout["meta"]["acquisition"]["span"])


def test_every_hyperparameter_answers(client):
    exp = _experiment()
    for hp in _hp_names(exp):
        response, data = _slice(client, exp, hp)
        assert response.status_code == 200, hp
        assert data["figure"] is not None, (hp, data.get("warning"))


def test_the_script_reveals_the_div_before_drawing_into_it():
    """The page's own `draw(key, null)` sets `el.hidden = true`, and this
    figure's payload is *always* null — it ships no server-rendered plot. Every
    other figure gets un-hidden by that same `draw` on the way past; this one
    bypasses it, so its script has to do the job itself.

    Getting this wrong has no visible failure mode beyond "nothing happens":
    the Compute prompt hides, `Plotly.react` runs without error, and the plot is
    drawn into a div nobody can see. It cost an afternoon once.

    Order matters as much as presence — a hidden element has no width, so Plotly
    would size the plot to nothing and keep that size once it was revealed.
    """
    source = (Path(__file__).parents[3] / "ui" / "static" / "ui" / "acquisition.js").read_text()

    reveal = source.index("plotEl.hidden = false")
    draw = source.index("Plotly.react(plotEl")
    assert reveal < draw, "the div must be revealed before it is drawn into"


def test_an_unknown_metric_is_refused(client):
    exp = _experiment()
    response, _ = _slice(client, exp, _hp_names(exp)[0], metric="nonesuch")

    assert response.status_code == 400


def test_an_unknown_hyperparameter_is_refused(client):
    exp = _experiment()
    response, _ = _slice(client, exp, "nonesuch")

    assert response.status_code == 400


def test_an_experiment_with_no_result_does_not_500(client):
    """An empty experiment should say it has nothing, not raise."""
    from ui.models import Experiment

    exp = Experiment.objects.create(
        name="empty", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=["accuracy"], current_metric="accuracy", seed=0)
    response = client.get(reverse("ui:acquisition_slice", args=[exp.pk]),
                          {"metric": "accuracy", "hp": "n_estimators"})

    assert response.status_code in (200, 400)
    if response.status_code == 200:
        assert json.loads(response.content)["figure"] is None
