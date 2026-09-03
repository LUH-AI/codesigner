"""The target the Run form opens on, and what "reaching" it means.

The criterion is a score to *surpass*, strictly — and the field is filled with
the incumbent's own score, so the two go together: an experiment sitting at
0.9667 is set up to run until something does better than 0.9667, and stopping on
an equal score would end the run without having improved on anything.

The number is per metric, because the incumbent is, so the field follows the
metric dropdown until someone types in it.
"""

import json
import re

import pytest
from django.urls import reverse

from tests.conftest import DATASETS_DIR
from ui.views import surpass_target

pytestmark = pytest.mark.django_db


def _result(**scores):
    """A one-trial result whose incumbent scores exactly *scores*."""
    return {
        "stats": {"submitted": 1, "finished": 1, "running": 0},
        "data": [{"config_id": 1, "cost": 0.0, "scores": scores,
                  "incumbent_score": scores["accuracy"], "incumbent_config_id": 1}],
        "configs": {"1": {"n_estimators": 100}},
        "config_origins": {"1": "Random Search"}, "optimizer_state": {},
        "primary_metric": "accuracy", "best_score": scores["accuracy"],
        "best_config_id": "1", "hyperparameter_importance": {},
        "hyperparameter_importance_warning": {}, "trials_limit": None,
    }


def _experiment(result=None, metric="accuracy"):
    from ui.services import snapshot as adapter

    exp = adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "targeted", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search", "optimizer_params": {},
        "primary_metric": metric if result else None,
        "original_metric": metric if result else None,
        "metric_names": ["accuracy", "f1"], "seed": 0,
        "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": result,
    }, adopt_paths=True)
    return exp


def _field(client, exp):
    body = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    at = body.index('name="target_score"')
    return body[body.rindex("<", 0, at):body.index(">", at)], body


# ── the wording, and the behaviour under it ──────────────────────────────────

def test_the_field_asks_to_be_surpassed(client):
    _, body = _field(client, _experiment())

    assert "…performance surpasses" in body


def test_an_equal_score_does_not_end_the_run(client):
    """The behaviour the wording promises, at the boundary that matters."""
    from core.optimizers.base import TrialCollector

    collector = TrialCollector(stopping={"target_score": 0.5})
    collector.record({"a": 1}, 0.5, {"accuracy": 0.5})

    assert collector.done is False

    collector.record({"a": 2}, 0.500001, {"accuracy": 0.500001})

    assert collector.done is True
    assert collector.stopped_by == "target_score"


# ── the number it opens on ───────────────────────────────────────────────────

def test_the_target_starts_at_the_score_there_is_to_beat(client):
    exp = _experiment(_result(accuracy=0.9667, f1=0.9))

    field, _ = _field(client, exp)

    assert 'value="0.9667"' in field


def test_an_experiment_with_no_trials_yet_leaves_it_empty(client):
    """Nothing to beat, so nothing to fill in — and a filled field is an active
    criterion, which would bound a run nobody asked to bound."""
    field, _ = _field(client, _experiment())

    assert 'value=""' in field


@pytest.mark.parametrize("score", [0.96663, 1 / 3, 0.8, 0.5000049])
def test_the_shown_target_is_never_below_the_score_it_came_from(score):
    """Rounded for the field, but rounding *down* would mean the incumbent
    already surpasses the target — the run would end after one trial without
    improving on anything."""
    assert float(surpass_target(score)) >= score


def test_the_target_is_rounded_rather_than_shown_in_full(client):
    """A raw float is fifteen digits of noise in a form field."""
    exp = _experiment(_result(accuracy=0.9666666666666667, f1=0.9))

    field, _ = _field(client, exp)

    assert 'value="0.9667"' in field


# ── following the metric ─────────────────────────────────────────────────────

def test_every_metrics_target_is_on_the_page(client):
    """The incumbent is per metric, so a number left over from the previous one
    would be nonsense against the new one."""
    exp = _experiment(_result(accuracy=0.9667, f1=0.5))

    _, body = _field(client, exp)
    targets = json.loads(
        re.search(r'id="incumbent-targets"[^>]*>(.*?)</script>', body, re.S).group(1))

    assert targets == {"accuracy": "0.9667", "f1": "0.5"}


def test_typing_a_target_stops_the_page_replacing_it(client):
    """At the point someone has chosen a target, keeping it up to date with the
    dropdown is not help."""
    exp = _experiment(_result(accuracy=0.9667, f1=0.5))

    _, body = _field(client, exp)

    assert "let untouched = true;" in body
    assert 'field.addEventListener("input", function () { untouched = false; });' in body
    assert "if (untouched) field.value = targets[metric.value]" in body


# ── what counts as out of range ──────────────────────────────────────────────
#
# The bound is the metric's, not a constant. A target is compared against a
# score, so 0-to-1 is accuracy's range rather than every metric's — and an
# unbounded one (an imported optimizer cost) bounds a target not at all.


def _posted(metric_name, **fields):
    """`_posted_stopping` over a form carrying *fields*, optimizing *metric_name*."""
    from django.test import RequestFactory

    from ui.views import _posted_stopping

    request = RequestFactory().post("/", {k: str(v) for k, v in fields.items()})
    return _posted_stopping(request, metric_name)


def test_a_target_inside_the_metrics_range_is_kept():
    assert _posted("accuracy", target_score="0.9")["target_score"] == 0.9


def test_a_target_above_the_metrics_range_is_clamped():
    assert _posted("accuracy", target_score="5")["target_score"] == 1.0


def test_a_negative_target_is_clamped_rather_than_dropped():
    """The bug this replaces: below-range meant *absent*.

    A run asking only for a target of -1 was then refused for having no
    stopping criterion at all — which reports a different problem from the one
    the reader created, and gives no hint that the number was the issue.
    """
    stopping = _posted("accuracy", target_score="-1")

    assert "target_score" in stopping
    assert stopping["target_score"] == 0.0


def test_an_unbounded_metric_accepts_a_negative_target():
    """Nothing to clamp to, so nothing is clamped.

    A metric this build has never heard of falls back to the one convention
    there was, so the case is exercised through a metric that *is* declared
    unbounded — which is what an imported run carries.
    """
    from unittest.mock import patch

    from core.metrics import Metric

    cost = Metric(name="smac:cost", fn=None, higher_is_better=False,
                  bounds=(None, None))
    with patch("ui.views.metric_for", return_value=cost):
        stopping = _posted("smac:cost", target_score="-12.5")

    assert stopping["target_score"] == -12.5


def test_the_field_takes_a_minus_sign_and_an_exponent(client):
    """A text input patterned `[0-9]*[.,]?[0-9]*` rejected both."""
    field, _ = _field(client, _experiment())

    assert 'type="number"' in field
    assert "pattern=" not in field
