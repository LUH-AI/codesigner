"""The trials table leads with outcome columns, which sort on click, and a rule
separates them from the configuration.
"""

import pytest
from django.urls import reverse

from experiments.models import Experiment

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
        metric_names=METRICS, current_metric="accuracy", original_metric="accuracy",
        seed=0, result=result,
    )


def _parts(client, exp):
    """`_parts` returns the page plus the trials table's thead and tbody."""
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    table = page.split('<table class="trials">', 1)[-1].split("</table>", 1)[0]
    return (page,
            table.split("<thead", 1)[-1].split("</thead>", 1)[0],
            table.split("<tbody", 1)[-1].split("</tbody>", 1)[0])


def test_metrics_and_duration_come_before_the_configuration(client, experiment):
    _, head, _ = _parts(client, experiment)

    last_outcome = max(head.index(m) for m in METRICS + ["Duration"])
    assert last_outcome < head.index("n_estimators")


def test_a_rule_separates_outcome_from_configuration(client, experiment):
    """The divider class must mark the header and every row."""
    _, head, rows = _parts(client, experiment)

    assert "group-start" in head
    assert rows.count("group-start") == 3          # per row


def test_outcome_columns_are_sortable(client, experiment):
    """Hyperparameter columns stay unsortable."""
    page, head, _ = _parts(client, experiment)

    assert head.count('class="sortable"') == len(METRICS) + 2
    # config header present, unsortable
    assert '<th class="group-start">n_estimators</th>' in head
    assert "aria-sort" in page                                   # sort state


def test_rows_carry_raw_values_to_sort_on(client, experiment):
    """Raw values keep '9.000 s' from sorting above '10.000 s' as text."""
    _, _, rows = _parts(client, experiment)

    assert 'data-sort="0.91"' in rows       # unrounded score
    assert 'data-sort="9.0"' in rows        # duration without unit
    assert "9 s" in rows                    # four-figure display


def test_the_table_is_paged_with_its_page_size_on_the_page(client, experiment):
    """A long table is paged, with the page size beside the selector. Paging
    hides rows without dropping them, so a sort still sees the whole table.
    """
    from ui.figures.catalog import Trials

    page, _head, body = _parts(client, experiment)
    pager = page.split("data-trials-pager", 1)[1].split("</div>\n</section>", 1)[0]

    assert "data-page-select" in pager
    assert "data-page-prev" in pager and "data-page-next" in pager
    assert f'value="{Trials.page_size}"' in pager
    assert pager.lstrip().startswith("hidden"), \
        "no pager until the script finds more rows than fit a page"
    assert body.count("<tr") == len(experiment.data.result["data"]), \
        "every row is in the document; paging hides them, it does not drop them"
