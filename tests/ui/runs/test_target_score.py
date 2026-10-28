"""The target is a score to surpass strictly, and the field opens on the
incumbent's score per metric. The field follows the metric dropdown until
someone types in it.
"""

import json
import re

import pytest
from django.urls import reverse

from tests.conftest import DATASETS_DIR
from ui.views.common import surpass_target

pytestmark = pytest.mark.django_db


def _result(**scores):
    """`_result` builds one trial scoring *scores*."""
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
    from experiments.services import snapshot as adapter

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


# Wording and behaviour

def test_the_field_asks_to_be_surpassed(client):
    _, body = _field(client, _experiment())

    assert "…performance surpasses" in body


def test_an_equal_score_does_not_end_the_run(client):
    from core.optimizers.base import TrialCollector

    collector = TrialCollector(stopping={"target_score": 0.5})
    collector.record({"a": 1}, 0.5, {"accuracy": 0.5})

    assert collector.done is False

    collector.record({"a": 2}, 0.500001, {"accuracy": 0.500001})

    assert collector.done is True
    assert collector.stopped_by == "target_score"


# Starting value

def test_the_target_starts_at_the_score_there_is_to_beat(client):
    exp = _experiment(_result(accuracy=0.9667, f1=0.9))

    field, _ = _field(client, exp)

    assert 'value="0.9667"' in field


def test_an_experiment_with_no_trials_yet_leaves_it_empty(client):
    """A filled field is an active criterion and would bound the run."""
    field, _ = _field(client, _experiment())

    assert 'value=""' in field


@pytest.mark.parametrize("score", [0.96663, 1 / 3, 0.8, 0.5000049])
def test_the_shown_target_is_never_below_the_score_it_came_from(score):
    """Rounding down would end the run after one trial without improvement."""
    assert float(surpass_target(score)) >= score


def test_the_target_is_rounded_rather_than_shown_in_full(client):
    exp = _experiment(_result(accuracy=0.9666666666666667, f1=0.9))

    field, _ = _field(client, exp)

    assert 'value="0.9667"' in field


# Following the metric

def test_every_metrics_target_is_on_the_page(client):
    """Each metric gets its own target, since the incumbent is per metric."""
    exp = _experiment(_result(accuracy=0.9667, f1=0.5))

    _, body = _field(client, exp)
    found = re.search(r'id="incumbent-targets"[^>]*>(.*?)</script>', body, re.S)
    assert found is not None
    targets = json.loads(found.group(1))

    assert targets == {"accuracy": "0.9667", "f1": "0.5"}


def test_typing_a_target_stops_the_page_replacing_it(client):
    """A typed target stays when the dropdown changes."""
    exp = _experiment(_result(accuracy=0.9667, f1=0.5))

    _, body = _field(client, exp)

    assert "let untouched = true;" in body
    assert 'field.addEventListener("input", function () { untouched = false; });' in body
    assert "if (untouched) field.value = targets[metric.value]" in body


# The range is the metric's own, and an unbounded metric (an imported optimizer
# cost) leaves the target unbounded.


def _posted(metric_name, **fields):
    """`_posted` reads *fields* through `run_form.stopping_from`."""
    from django.test import RequestFactory

    from ui.run_form import stopping_from

    request = RequestFactory().post("/", {k: str(v) for k, v in fields.items()})
    return stopping_from(request.POST, metric_name)


def test_a_target_inside_the_metrics_range_is_kept():
    assert _posted("accuracy", target_score="0.9")["target_score"] == 0.9


def test_a_target_above_the_metrics_range_is_clamped():
    assert _posted("accuracy", target_score="5")["target_score"] == 1.0


def test_a_negative_target_is_clamped_rather_than_dropped():
    """Dropping a target of -1 would refuse the run for lacking any criterion.
    """
    stopping = _posted("accuracy", target_score="-1")

    assert "target_score" in stopping
    assert stopping["target_score"] == 0.0


def test_an_unbounded_metric_accepts_a_negative_target():
    """An imported run declares its metric unbounded, and nothing clamps the
    target.
    """
    from unittest.mock import patch

    from core.metrics import Metric, unscoreable

    cost = Metric(name="smac:cost", fn=unscoreable, higher_is_better=False,
                  bounds=(None, None))
    with patch("ui.run_form.metric_for", return_value=cost):
        stopping = _posted("smac:cost", target_score="-12.5")

    assert stopping["target_score"] == -12.5


def test_the_field_takes_a_minus_sign_and_an_exponent(client):
    field, _ = _field(client, _experiment())

    assert 'type="number"' in field
    assert "pattern=" not in field
