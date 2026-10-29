"""`core.processing` fits gaps, scale and label handling per experiment, and
"auto" picks what each model needs.
"""

from typing import cast

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
    assert resolve(DEFAULTS, MODELS[name]) == (plan, [])


def test_a_choice_the_model_cannot_take_is_overruled():
    """`resolve` imputes gaps for an SVM, which raises a `ValueError` for NaN,
    and reports the
    overruled step."""
    plan, overruled = resolve({"missing": "keep", "scale": "none"}, MODELS["SVM"])

    assert plan["missing"] == "impute" and plan["scale"] == "none"
    assert overruled == ["missing"]


def test_an_unknown_or_missing_choice_reads_as_auto():
    """Old or hand-edited files must never break a run."""
    assert choices_of({"scale": "cubic"}) == DEFAULTS
    assert choices_of(None) == DEFAULTS


def test_processing_learns_from_the_rows_it_is_fitted_on_only():
    processing = Processing({"missing": "keep", "scale": "standardize", "labels": "one_hot"},
                            ("numeric",)).fit(np.arange(10.0).reshape(-1, 1))

    assert cast(np.ndarray, processing.transform(np.array([[1000.0]])))[0, 0] > 300


def test_label_columns_as_categories_keep_their_kind():
    """Label codes move after the numbers and stay categorical, so boosters
    still split them as categories."""
    X, _, kinds = _flats()
    processing = Processing({"missing": "keep", "scale": "standardize", "labels": "codes"}, kinds)
    out = processing.fit_transform(X)

    assert processing.kinds_out() == tuple(sorted(kinds, key=lambda k: k == "categorical"))
    assert out.shape == X.shape


def test_choosing_differently_changes_what_a_trial_scores():
    """The choice reaches the model: an unscaled SVM predicts differently."""
    X, y, kinds = _flats()
    model = MODELS["SVM"].__class__()
    model.task, model.feature_kinds = "classification", kinds
    config = dict(model.get_config_space().get_default_configuration())

    scaled = model.fit_predict(config, X[:180], y[:180], X[180:], seed=0)
    model.processing = {"scale": "none"}
    unscaled = model.fit_predict(config, X[:180], y[:180], X[180:], seed=0)

    assert list(scaled) != list(unscaled)
