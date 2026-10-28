"""The server hands the scale toggle a custom modebar button and a per-metric
absolute range, and the click runs in the browser.
"""

import json

from django.urls import reverse

from core import io
from core.metrics import Metric, unscoreable
from ui.figures.catalog import FIGURES_BY_KEY
from experiments.services import snapshot as snapshot_adapter

from tests.conftest import FIXTURES_DIR


def _figure_options(client, exp) -> dict:
    """`_figure_options` parses the `figureOptions` payload."""
    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    start = body.index('id="figure-options-data"')
    start = body.index(">", start) + 1
    return json.loads(body[start:body.index("</script>", start)])


def _experiment_with_result():
    return snapshot_adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes())
    )


def test_absolute_scale_button_wired_on_detail(client):
    exp = _experiment_with_result()
    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    # custom modebar button
    assert "modeBarButtonsToAdd" in body

    # toggles to accuracy's full range
    scale = _figure_options(client, exp)["performance_over_time"]["absoluteScale"]
    assert scale["accuracy"]["trial-score"]["yaxis.range"] == [0.0, 1.0]
    # log error axis: log10 of 1e-3 to 1
    assert scale["accuracy"]["trial-error"]["yaxis.range"] == [-3.0, 0.0]


def test_the_scale_is_keyed_by_metric(client):
    """The page switches metric without reloading, so each metric carries its
    own range.
    """
    exp = _experiment_with_result()

    options = _figure_options(client, exp)["performance_over_time"]

    assert options["perMetric"] is True
    assert set(options["absoluteScale"]) == set(exp.data.metric_names)


def test_an_unbounded_metric_offers_no_absolute_range():
    """An unbounded metric gets an empty answer, which hides the button.
    """
    cost = Metric(name="smac:cost", fn=unscoreable, higher_is_better=False,
                  bounds=(None, None))

    assert FIGURES_BY_KEY["performance_over_time"].absolute_scale_for(cost) == {}


def test_a_lower_is_better_metric_keeps_its_score_axis():
    """A bounded lower-is-better metric uses its bounds as the full range.
    """
    loss = Metric(name="loss", fn=unscoreable, higher_is_better=False, bounds=(0.0, 4.0))

    scale = FIGURES_BY_KEY["performance_over_time"].absolute_scale_for(loss)

    assert scale is not None
    assert scale["trial-score"]["yaxis.range"] == [0.0, 4.0]
    assert "trial-error" in scale


def test_scale_button_has_two_icons_for_toggle_state(client):
    exp = _experiment_with_result()
    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert "SCALE_EXPAND" in body and "SCALE_FIT" in body
    assert "makeScaleToggle" in body
