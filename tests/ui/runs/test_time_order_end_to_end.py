"""The create form builds a time-ordered experiment, and a re-imported file
keeps its order.
"""

from pathlib import Path

import numpy as np
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from tests.conftest import post_new_experiment

FLATS = Path(__file__).parent.parent.parent / "fixtures" / "flats.csv"


def _built(exp):
    """`_built` builds *exp* with its data and folds."""
    from core import io, registry
    from experiments.services import snapshot as snapshot_adapter

    _, built = io.build_experiment(snapshot_adapter.snapshot_from_experiment(exp),
                                   registry.METRICS, registry.MODELS, registry.OPTIMIZERS,
                                   read_only=False)
    return built


def _create(client, name="flats in time", **overrides):
    data = {
        "name": name, "model_name": "Random Forest", "optimizer_name": "Random Search",
        "dataset_file": SimpleUploadedFile("flats.csv", FLATS.read_bytes(), content_type="text/csv"),
        "task": "classification",
        "evaluation_scheme": "kfold", "evaluation_value": 3, "seed": 0,
        "time_column": "listed", "time_gap": 5,
    }
    data.update(overrides)
    return post_new_experiment(client, data)


@pytest.mark.django_db
def test_every_trial_validates_on_listings_after_its_training_ones(client):
    """Every fold validates on later listings than its training rows, with a gap
    of five rows."""
    from core import encoding
    from experiments.models import Experiment
    from experiments.services.snapshot import rebuild_result

    assert _create(client).status_code == 302
    exp = Experiment.objects.get(data__name="flats in time")
    assert (exp.data.time_column, exp.data.time_gap, exp.data.cv_folds) == ("listed", 5, 3)

    built = _built(exp)
    days = built["splits"].X[:, encoding.feature_names(built["encoding"]).index("listed:days")]
    for train, val in built["splits"].folds:
        assert days[train].max() <= days[val].min()
        assert val.min() - train.max() == 6

    client.post(reverse("ui:experiment_run", args=[exp.pk]),
                {"max_trials": 3, "optimize_metric": "accuracy"})
    exp.refresh_from_db()
    result = rebuild_result(exp)
    assert result is not None
    assert not any(t.failed for t in result.trials)
    page = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    assert "3 time-ordered folds by listed" in page


@pytest.mark.django_db
def test_the_order_survives_a_round_trip_through_a_file(client):
    """The `.ihpo` records the column and the gap."""
    from experiments.models import Experiment
    from experiments.services import snapshot as snapshot_adapter

    _create(client)
    exp = Experiment.objects.get(data__name="flats in time")
    snapshot = snapshot_adapter.snapshot_from_experiment(exp)
    assert snapshot["evaluation"]["time_column"] == "listed"
    assert snapshot["evaluation"]["gap"] == 5

    snapshot["name"] = "reopened"
    copy = snapshot_adapter.experiment_from_snapshot(snapshot)
    assert (copy.data.time_column, copy.data.time_gap) == ("listed", 5)


@pytest.mark.django_db
@pytest.mark.parametrize("column, says", [("city", "holds labels"),
                                          ("nowhere", "no feature column named")])
def test_a_column_that_cannot_order_the_rows_is_refused(client, column, says):
    """The form reports a label column or a missing one as a validation error,
    naming the reason and the usable columns."""
    from experiments.models import Experiment

    resp = _create(client, name=f"bad {column}", time_column=column)
    body = resp.content.decode()

    assert resp.status_code == 200
    assert says in body and "listed, first_viewing" in body
    assert not Experiment.objects.filter(data__name=f"bad {column}").exists()


@pytest.mark.django_db
def test_left_empty_the_rows_are_divided_at_random_as_before(client):
    from core import encoding
    from experiments.models import Experiment

    _create(client, name="random", time_column="", time_gap="")
    exp = Experiment.objects.get(data__name="random")
    assert (exp.data.time_column, exp.data.time_gap) == ("", 0)

    built = _built(exp)
    days = built["splits"].X[:, encoding.feature_names(built["encoding"]).index("listed:days")]
    train, val = built["splits"].folds[0]
    assert days[train].max() > np.median(days[val])
