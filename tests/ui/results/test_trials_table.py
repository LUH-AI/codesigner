"""The trials table leads with each trial's outcome, then its configuration.

A trial's scores and duration are what you scan for; the hyperparameters that
produced them are the detail you look at once something stands out. So the
metric and Duration columns come first, a rule separates them from the config
columns, and the outcome columns sort on click (all numeric).
"""

import pytest
from django.urls import reverse

from ui.models import Experiment

METRICS = ["accuracy", "f1", "precision", "recall(macro)"]


def _trial(cid, acc, dur, n_estimators):
    return {
        "config_id": cid, "cost": 1 - acc, "time": dur, "cpu_time": dur,
        "starttime": 1000.0, "endtime": 1000.0 + dur, "status": 1,
        "seed": 0, "budget": None, "instance": None, "additional_info": {},
        "scores": {"accuracy": acc, "f1": 0.7, "precision": 0.75, "recall(macro)": 0.72},
        "incumbent_score": acc, "incumbent_config_id": cid,
    }


@pytest.fixture
def experiment():
    result = {
        "stats": {"submitted": 3, "finished": 3, "running": 0},
        "data": [_trial(1, 0.80, 2.5, 100), _trial(2, 0.91, 0.5, 50),
                 _trial(3, 0.72, 9.0, 200)],
        "configs": {"1": {"n_estimators": 100}, "2": {"n_estimators": 50},
                    "3": {"n_estimators": 200}},
        "config_origins": {"1": "Random Search", "2": "Random Search", "3": "Random Search"},
        "optimizer_state": {}, "primary_metric": "accuracy", "best_score": 0.91,
        "best_config_id": "2", "hyperparameter_importance": {},
        "hyperparameter_importance_warning": {}, "trials_limit": None,
    }
    return Experiment.objects.create(
        name="trials-table", model_name="Random Forest", optimizer_name="Random Search",
        metric_names=METRICS, primary_metric="accuracy", original_metric="accuracy",
        seed=0, result=result,
    )


def _parts(client, exp):
    """The whole page, plus the trials table's header and body — the
    best-config tables render first, so the trials table is isolated before
    splitting out thead/tbody."""
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    table = page.split('<table class="trials">', 1)[-1].split("</table>", 1)[0]
    return (page,
            table.split("<thead", 1)[-1].split("</thead>", 1)[0],
            table.split("<tbody", 1)[-1].split("</tbody>", 1)[0])


def test_metrics_and_duration_come_before_the_configuration(client, experiment):
    """Every outcome column precedes every hyperparameter column."""
    _, head, _ = _parts(client, experiment)

    last_outcome = max(head.index(m) for m in METRICS + ["Duration"])
    assert last_outcome < head.index("n_estimators")


def test_a_rule_separates_outcome_from_configuration(client, experiment):
    """The first configuration column is marked so it can carry the divider —
    in the header and in every body row, or the rule would not run down the
    table."""
    _, head, rows = _parts(client, experiment)

    assert "group-start" in head
    assert rows.count("group-start") == 3          # one per trial row


def test_outcome_columns_are_sortable(client, experiment):
    """Trial number, each metric and Duration are click-sortable; the
    hyperparameter columns are not."""
    page, head, _ = _parts(client, experiment)

    assert head.count('class="sortable"') == len(METRICS) + 2   # metrics + # + Duration
    # the config header is present but not sortable
    assert '<th class="group-start">n_estimators</th>' in head
    assert "aria-sort" in page                                   # sort state is exposed


def test_rows_carry_raw_values_to_sort_on(client, experiment):
    """Cells sort on an unformatted value, not their display text — otherwise
    '9.000 s' would order above '10.000 s' as a string."""
    _, _, rows = _parts(client, experiment)

    assert 'data-sort="0.91"' in rows       # a score, unrounded
    assert 'data-sort="9.0"' in rows        # a duration, without the unit
    assert "9.000 s" in rows                # still displayed formatted
