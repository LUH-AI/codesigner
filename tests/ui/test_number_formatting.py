"""One display rule: four significant figures, or fewer.

Not four decimal places, which is what Django's `floatformat` counts and what
this app used everywhere. The same four places that suit a score between 0 and 1
write a small hyperparameter effect as 0.0000, and write 0.8 as 0.8000 — three
digits it has not earned.

A display rule only. The full float is what is stored, what the optimizer sees
and what an exported .ihpo carries, which is the last test here.
"""

import json

from django.template import Context, Template

from ui.formatting import sigfigs

from tests.conftest import export_ihpo


def _render(value):
    return Template("{% load numbers %}{{ v|sig }}").render(Context({"v": value}))


def test_it_counts_figures_not_decimal_places():
    """The whole point, in the two directions decimal places get wrong: a short
    number stays short, and a small one keeps its magnitude."""
    assert sigfigs(0.8) == "0.8"
    assert sigfigs(0.68125) == "0.6813"
    assert sigfigs(0.000123456) == "0.0001235"


def test_an_integer_keeps_every_digit():
    """128 estimators rounded to 130 is a different experiment, and an integer
    has no rounding to hide behind."""
    assert sigfigs(128) == "128"
    assert sigfigs(100000) == "100,000"


def test_anything_that_is_not_a_number_is_left_alone():
    """This runs over hyperparameter values, which are as often a kernel name
    or a boolean as they are a float."""
    assert sigfigs("rbf") == "rbf"
    assert sigfigs(True) == "True"
    assert sigfigs(None) == "None"


def test_the_template_filter_is_the_same_rule():
    """Templates and plot builders format the same numbers, so they share the
    implementation rather than each having their own."""
    assert _render(0.68125) == sigfigs(0.68125) == "0.6813"


def test_an_exported_file_carries_the_full_float(client):
    """The rule stops at the page. What a run actually did is what the .ihpo has
    to say, to the last digit, or an experiment recreated from one is not the
    same experiment. So the same number appears two ways: shortened on the
    page, exact in the file."""
    from core import io
    from django.urls import reverse
    from ui.services import snapshot as adapter

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
        # And so does the page, where it is not being read: the trials table
        # sorts on `data-sort`, so that has to be the exact number or the order
        # is the order of the rounded ones.
        assert f'data-sort="{score}"' in page
        assert f">{sigfigs(score)}</td>" in page, "shown to four figures"
