"""Choosing which column an experiment predicts.

Every dataset used to be read with its last column as the target. The target is
now a choice made in setup step 1, stored as `ExperimentData.target_column` and
in the file as `dataset.target_column`, and applied in one place: each reader
moves the chosen column last (`core.datasets.target_last`), so everything downstream
still reads the last column. An experiment that never chose one predicts the
last column, as every experiment made before the choice existed does.
"""

import numpy as np
import pandas as pd
import pytest
from django.contrib.auth.models import AnonymousUser
from django.test import RequestFactory

from core import datasets, io
from tests.conftest import DATASETS_DIR
from ui import drafts
from ui import setup_steps as steps
from experiments.services import snapshot as adapter

pytestmark = pytest.mark.django_db

IRIS = DATASETS_DIR / "iris.csv"


def _request():
    request = RequestFactory().post("/")
    request.user = AnonymousUser()
    return request


def _draft(**overrides):
    """A draft whose step 1 is saved from *overrides* over iris with a forest."""
    request = _request()
    exp = drafts.new(request)
    data = {"name": "target", "task": "regression", "model_name": "Random Forest",
            "demo_dataset": str(IRIS), "evaluation_scheme": "holdout",
            "evaluation_value": 0.2, "seed": 0, "target_column": "petal.width"}
    data.update(overrides)
    outcome = drafts.save_step(request, exp, steps.SETUP, data)
    exp.refresh_from_db()
    return exp, outcome


# ── where the choice is applied ──────────────────────────────────────────────

def test_the_chosen_column_is_moved_last_and_nothing_else_moves():
    """What: `target_last` puts the target at the end and keeps the order of the
    rest; no target leaves the frame alone; an unknown one is refused.
    How: a three-column frame through each case."""
    frame = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

    assert list(datasets.target_last(frame, "a").columns) == ["b", "c", "a"]
    assert list(datasets.target_last(frame, None).columns) == ["a", "b", "c"]
    assert list(datasets.target_last(frame, "c").columns) == ["a", "b", "c"]
    with pytest.raises(ValueError, match="no column named"):
        datasets.target_last(frame, "z")


def test_an_experiment_trains_on_the_column_it_chose():
    """What: a draft predicting iris's petal width is split with that column as
    y and the variety among the features.
    How: step 1 saved through the service, then the experiment rebuilt with its
    data and its y compared against the file's column."""
    exp, outcome = _draft()
    assert outcome.saved
    assert exp.data.target_column == "petal.width"

    _, built = io.build_experiment(adapter.snapshot_from_experiment(exp),
                                   *_registries(), read_only=False)
    frame = datasets.read_dataset(IRIS)
    y = np.concatenate([built["y_train"], built["y_val"]])

    assert sorted(y) == sorted(frame["petal.width"])
    assert built["X_train"].shape[1] == frame.shape[1] - 1


def test_without_a_choice_the_last_column_is_the_target():
    """What: an experiment that names no target — one made before the choice
    existed, or left empty — predicts the last column, and its file says
    nothing about one.
    How: a draft saved with an empty target, its snapshot and its rebuilt y."""
    exp, outcome = _draft(task="classification", target_column="")
    assert outcome.saved and exp.data.target_column == ""

    snapshot = adapter.snapshot_from_experiment(exp)
    assert "target_column" not in snapshot["dataset"]
    _, built = io.build_experiment(snapshot, *_registries(), read_only=False)
    assert set(built["y_train"]) <= set(datasets.read_dataset(IRIS)["variety"])


def _registries():
    from core.registry import METRICS, MODELS, OPTIMIZERS

    return METRICS, MODELS, OPTIMIZERS


# ── step 1 ───────────────────────────────────────────────────────────────────

def test_a_target_the_dataset_lacks_is_an_error_on_its_field():
    """What: naming a column the dataset does not have says so on the target.
    How: step 1 with a made-up column."""
    _, outcome = _draft(target_column="petal.colour")

    assert not outcome.saved
    assert "target_column" in outcome.form.errors


def test_regression_follows_the_chosen_column():
    """What: whether regression is possible is decided by the column chosen —
    refused on iris's varieties, which are labels, and allowed on a width.
    How: step 1 as a regression with each target."""
    _, labels = _draft(target_column="variety")
    _, numbers = _draft(target_column="sepal.length")

    assert not labels.saved and "task" in labels.form.errors
    assert "another target" in str(labels.form.errors["task"])
    assert numbers.saved


def test_changing_the_target_asks_for_the_data_step_again():
    """What: a different target is a different dataset to everything that reads
    it, so the Data step's choices have to be looked at again.
    How: saves steps 1 to 3, then step 1 with another target."""
    request = _request()
    exp, _ = _draft()
    drafts.save_step(request, exp, steps.OPTIMIZER, {"optimizer_name": "Random Search"})
    drafts.save_step(request, exp, steps.DATA, {})
    assert steps.saved(exp, steps.DATA)

    drafts.save_step(request, exp, steps.SETUP,
                     {"name": "target", "task": "regression", "model_name": "Random Forest",
                      "demo_dataset": str(IRIS), "evaluation_scheme": "holdout",
                      "evaluation_value": 0.2, "seed": 0, "target_column": "sepal.length"})
    exp.refresh_from_db()

    assert not steps.saved(exp, steps.DATA)


def test_the_form_reads_the_choice_back():
    """What: a saved draft's step 1 opens with its target filled in.
    How: `setup_initial` on a draft saved with one."""
    exp, _ = _draft()

    assert drafts.setup_initial(exp)["target_column"] == "petal.width"


# ── the file, and what is exported ──────────────────────────────────────────

def test_the_target_travels_in_the_file():
    """What: an exported experiment names its target, and one imported from the
    file predicts the same column.
    How: the snapshot of a draft with a target, read back into a new data half."""
    exp, _ = _draft()
    snapshot = adapter.snapshot_from_experiment(exp)

    assert snapshot["dataset"]["target_column"] == "petal.width"
    assert adapter.data_from_snapshot(io.normalize(snapshot)).target_column == "petal.width"


def test_an_exported_model_reads_its_target_by_name(tmp_path):
    """What: the runnable file an export writes moves its target column last
    when reading a CSV, so a CSV laid out like the dataset trains on the right
    column; with no target named it reads the CSV as it is.
    How: the generated reader run on a small CSV, with and without TARGET."""
    from core.model_export import _READ_CSV

    path = tmp_path / "rows.csv"
    path.write_text("a,b,c\n1,2,3\n")

    named = {"TARGET": "a"}
    exec(_READ_CSV, named)
    plain = {"TARGET": None}
    exec(_READ_CSV, plain)

    assert list(named["_read_csv"](path).columns) == ["b", "c", "a"]
    assert list(plain["_read_csv"](path).columns) == ["a", "b", "c"]
