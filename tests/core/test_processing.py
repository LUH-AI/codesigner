"""How an experiment's columns are processed before its model sees them.

`core.processing` turns three choices — gaps, scale, label columns — into a
fitted transformer. Left to "auto", each model gets what it needs, which is
exactly what it did for itself before processing was the experiment's to
choose; a choice the model cannot take is overruled.
"""

import numpy as np
import pandas as pd
import pytest

from core import encoding
from core.processing import DEFAULTS, Processing, choices_of, resolve
from core.registry import MODELS

from tests.conftest import FIXTURES_DIR


def _flats():
    frame = pd.read_csv(FIXTURES_DIR / "flats.csv")
    frame = frame[frame.iloc[:, -1].notna()]
    spec = encoding.fit(frame.iloc[:, :-1])
    return (encoding.encode(spec, frame.iloc[:, :-1]), frame.iloc[:, -1].to_numpy(),
            tuple(encoding.feature_kinds(spec)))


@pytest.mark.parametrize("name, plan", [
    ("SVM", {"missing": "impute", "scale": "standardize", "labels": "one_hot"}),
    ("k-Nearest Neighbors", {"missing": "impute", "scale": "standardize", "labels": "one_hot"}),
    ("Random Forest", {"missing": "keep", "scale": "none", "labels": "one_hot"}),
    ("LightGBM", {"missing": "keep", "scale": "none", "labels": "codes"}),
    ("CatBoost", {"missing": "keep", "scale": "none", "labels": "text"}),
])
def test_auto_is_what_the_model_needs(name, plan):
    """What: with nothing chosen, a model that measures distances gets its gaps
    filled and numbers scaled, trees one-hot labels only, boosters their
    categories in their own form. How: resolves the defaults for each."""
    assert resolve(DEFAULTS, MODELS[name]) == (plan, [])


def test_a_choice_the_model_cannot_take_is_overruled():
    """What: gaps left for an SVM, which refuses NaN, are filled anyway, and
    the step is named; scaling switched off is simply honoured.
    How: resolves an SVM with both choices made."""
    plan, overruled = resolve({"missing": "keep", "scale": "none"}, MODELS["SVM"])

    assert plan["missing"] == "impute" and plan["scale"] == "none"
    assert overruled == ["missing"]


def test_an_unknown_or_missing_choice_reads_as_auto():
    """What: a stored processing from an older or hand-edited file never breaks
    a run. How: reads a partial dict with a nonsense value."""
    assert choices_of({"scale": "cubic"}) == DEFAULTS
    assert choices_of(None) == DEFAULTS


def test_processing_learns_from_the_rows_it_is_fitted_on_only():
    """What: the scale is the training rows', so a validation row far outside
    them stays far outside after scaling instead of informing it.
    How: fits on 0..9, transforms 1000."""
    processing = Processing({"missing": "keep", "scale": "standardize", "labels": "one_hot"},
                            ("numeric",)).fit(np.arange(10.0).reshape(-1, 1))

    assert processing.transform(np.array([[1000.0]]))[0, 0] > 300


def test_label_columns_as_categories_keep_their_kind():
    """What: a booster's label columns stay label columns after scaling the
    numbers — moved after them — so it still splits them as categories.
    How: processes flats with scaling on and labels as codes."""
    X, _, kinds = _flats()
    processing = Processing({"missing": "keep", "scale": "standardize", "labels": "codes"}, kinds)
    out = processing.fit_transform(X)

    assert processing.kinds_out() == tuple(sorted(kinds, key=lambda k: k == "categorical"))
    assert out.shape == X.shape


def test_choosing_differently_changes_what_a_trial_scores():
    """What: the choice reaches the model — an SVM without scaling predicts
    differently from one with it. How: fits the same configuration both ways."""
    X, y, kinds = _flats()
    model = MODELS["SVM"].__class__()
    model.task, model.feature_kinds = "classification", kinds
    config = dict(model.get_config_space().get_default_configuration())

    scaled = model.fit_predict(config, X[:180], y[:180], X[180:], seed=0)
    model.processing = {"scale": "none"}
    unscaled = model.fit_predict(config, X[:180], y[:180], X[180:], seed=0)

    assert list(scaled) != list(unscaled)
