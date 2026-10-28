"""Displays use four significant figures or fewer, never four decimal places.
Storage, the optimizer and exported `.ihpo` files keep the full float.
"""

import json

from django.template import Context, Template

from ui.formatting import sigfigs

from tests.conftest import export_ihpo


def _render(value):
    return Template("{% load numbers %}{{ v|sig }}").render(Context({"v": value}))


def test_it_counts_figures_not_decimal_places():
    """A short number stays short, and a small one keeps its magnitude."""
    assert sigfigs(0.8) == "0.8"
    assert sigfigs(0.68125) == "0.6813"
    assert sigfigs(0.000123456) == "0.0001235"


def test_an_integer_keeps_every_digit():
    """Rounding 128 estimators to 130 would describe a different experiment."""
    assert sigfigs(128) == "128"
    assert sigfigs(100000) == "100,000"


def test_anything_that_is_not_a_number_is_left_alone():
    """Hyperparameter values include kernel names and booleans."""
    assert sigfigs("rbf") == "rbf"
    assert sigfigs(True) == "True"
    assert sigfigs(None) == "None"


def test_the_template_filter_is_the_same_rule():
    """Templates and plot builders share one implementation."""
    assert _render(0.68125) == sigfigs(0.68125) == "0.6813"


def test_an_exported_file_carries_the_full_float(client):
    """A recreated experiment needs every digit of what the run did."""
    from core import io
    from django.urls import reverse
    from experiments.services import snapshot as adapter

    from tests.conftest import FIXTURES_DIR

    exp = adapter.experiment_from_snapshot(
        io.parse((FIXTURES_DIR / "test2.ihpo").read_bytes()))
    exported = json.loads(
        export_ihpo(client, exp.pk).content)
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    long_ones = [row["scores"]["accuracy"] for row in exported["result"]["data"]
                 if len(repr(row["scores"]["accuracy"])) > 6]
    assert long_ones, "the fixture has a score longer than the display rule shows"

    for score in long_ones:
        assert repr(score) in json.dumps(exported), "the file keeps every digit"
        # The trials table sorts on exact `data-sort` values.
        assert f'data-sort="{score}"' in page
        assert f">{sigfigs(score)}</td>" in page, "shown to four figures"
