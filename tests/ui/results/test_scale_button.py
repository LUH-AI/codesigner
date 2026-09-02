"""The performance figure gets a custom modebar button toggling the y-axis
between the metric's own full range and auto-fit-to-data.

*What:* the wiring behind the absolute/relative toggle. The behaviour itself is
client-side (a Plotly.relayout on click), so what is pinned here is what the
server hands the script: a custom modebar button, and the range to pin to —
which is per metric, since a range that is right for an accuracy is a fiction
for one that is not bounded 0-1.

*How:* render the detail page for a result-bearing experiment and read the
`figure-options-data` block the script parses, rather than matching numbers in
the page text, which pins how a float happens to be formatted.
"""

import json

from django.urls import reverse

from core import io
from core.metrics import Metric
from ui.figures.catalog import FIGURES_BY_KEY
from ui.services import snapshot as snapshot_adapter

from tests.conftest import FIXTURES_DIR


def _figure_options(client, exp) -> dict:
    """The `figureOptions` payload the detail page's script reads."""
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
    # a custom button is added to the performance figure's modebar…
    assert "modeBarButtonsToAdd" in body

    # …and it toggles the y-axis to accuracy's own full range, 0 to 1.
    scale = _figure_options(client, exp)["performance_over_time"]["absoluteScale"]
    assert scale["accuracy"]["trial-score"]["yaxis.range"] == [0.0, 1.0]
    # The error axis is logarithmic, so its range is the log10 of 1e-3 to 1.
    assert scale["accuracy"]["trial-error"]["yaxis.range"] == [-3.0, 0.0]


def test_the_scale_is_keyed_by_metric(client):
    """Every metric the experiment carries gets its own range, not one shared.

    The page switches metric without reloading, so the toggle has to resolve
    the range at the moment it is clicked rather than being handed one fixed
    range at page load.
    """
    exp = _experiment_with_result()

    options = _figure_options(client, exp)["performance_over_time"]

    assert options["perMetric"] is True
    assert set(options["absoluteScale"]) == set(exp.metric_names)


def test_an_unbounded_metric_offers_no_absolute_range():
    """A metric with no bounds has no full range, so it gets no toggle.

    "Absolute" means the metric's own range end to end. An imported optimizer
    cost has neither end, and the only range there is is the one the data
    happens to occupy — which is what the relative scale already shows. An
    empty answer is what tells the page to hide the button rather than pin the
    axis to some other metric's range.
    """
    cost = Metric(name="smac:cost", fn=None, higher_is_better=False,
                  bounds=(None, None))

    assert FIGURES_BY_KEY["performance_over_time"].absolute_scale_for(cost) == {}


def test_a_lower_is_better_metric_keeps_its_score_axis():
    """Bounded-but-inverted still has a full range: it is the bounds themselves.

    Only the *error* axis drops out, since a metric where lower wins is already
    an error and 0-to-high on a log axis is what it was going to draw anyway.
    """
    loss = Metric(name="loss", fn=None, higher_is_better=False, bounds=(0.0, 4.0))

    scale = FIGURES_BY_KEY["performance_over_time"].absolute_scale_for(loss)

    assert scale["trial-score"]["yaxis.range"] == [0.0, 4.0]
    assert "trial-error" in scale


def test_scale_button_has_two_icons_for_toggle_state(client):
    """The button swaps between two icons (expand / fit) so its current state
    is visible — both glyph definitions must be present and distinct, and the
    shared toggle factory drives them."""
    exp = _experiment_with_result()
    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert "SCALE_EXPAND" in body and "SCALE_FIT" in body
    assert "makeScaleToggle" in body
