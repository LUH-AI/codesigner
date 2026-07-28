"""The performance chart gets a custom modebar button toggling the y-axis
between its absolute 0-1 range and auto-fit-to-data.

The behaviour itself is client-side (a Plotly.relayout on click), so this pins
the wiring: the detail page must register a custom modebar button on a
result-bearing experiment.
"""

from django.urls import reverse

from core import io
from web.services import snapshot as snapshot_adapter

from tests.conftest import FIXTURES_DIR


def _experiment_with_result():
    return snapshot_adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes())
    )


def test_absolute_scale_button_wired_on_detail(client):
    exp = _experiment_with_result()
    body = client.get(reverse("web:experiment_detail", args=[exp.pk])).content.decode()
    # a custom button is added to the performance chart's modebar…
    assert "modeBarButtonsToAdd" in body
    # …and it toggles the y-axis to the absolute [0, 1] range
    assert "[0, 1]" in body


def test_scale_button_has_two_icons_for_toggle_state(client):
    """The button swaps between two icons (expand / fit) so its current state
    is visible — both glyph definitions must be present and distinct, and the
    shared toggle factory drives them."""
    exp = _experiment_with_result()
    body = client.get(reverse("web:experiment_detail", args=[exp.pk])).content.decode()
    assert "SCALE_EXPAND" in body and "SCALE_FIT" in body
    assert "makeScaleToggle" in body
