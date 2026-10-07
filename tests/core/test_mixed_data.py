"""A dataset of text, dates and gaps, end to end through `core`.

`tests/fixtures/flats.csv` has a city (with missing cities), a room count, an
area (with missing areas), a yes/no column, a listing date and a viewing
timestamp. Every built-in model is trained on it for both tasks, a model in its
own process is told what kind each column is, and an exported model predicts
from the raw CSV exactly what the model predicted inside Codesigner.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import encoding, io
from core.model_export import export_model
from core.modelhost import launch_local, model_session
from core.optimizers.trial import evaluate_trial
from core.registry import MODELS
from core.splits import holdout

from .test_modelhost import HEADER

FLATS = Path(__file__).parent.parent / "fixtures" / "flats.csv"


def _flats(target=None):
    """The fixture's encoded features and a target: its own last column, or
    *target* taken out of the features."""
    frame = pd.read_csv(FLATS)
    if target:
        frame = frame.drop(columns=frame.columns[-1])
        frame = frame[[c for c in frame.columns if c != target] + [target]]
    features = frame.iloc[:, :-1]
    spec = encoding.fit(features)
    return spec, encoding.encode(spec, features), frame.iloc[:, -1].to_numpy()


def _told(name, task, spec):
    model = MODELS[name].__class__()
    model.task = task
    model.feature_kinds = tuple(encoding.feature_kinds(spec))
    return model


def test_loading_a_mixed_file_gives_floats_and_drops_unlabelled_rows(tmp_path):
    """What: `_load_frame` encodes every feature to floats and leaves out a row
    whose target is missing. How: loads the fixture, then a copy of it with two
    targets blanked."""
    X, y = io._load_frame(FLATS)
    spec = io.dataset_encoding(FLATS)
    assert X.dtype == float and X.shape == (240, len(encoding.feature_names(spec)))
    assert np.isnan(X).any(), "the gaps survive as NaN for the model to handle"

    frame = pd.read_csv(FLATS)
    frame.iloc[[3, 7], -1] = None
    gappy = tmp_path / "gappy.csv"
    frame.to_csv(gappy, index=False)
    X, y = io._load_frame(gappy)
    assert len(X) == len(y) == 238


CLASSIFIERS = sorted(n for n, m in MODELS.items() if "classification" in m.tasks)
REGRESSORS = sorted(n for n, m in MODELS.items()
                    if "regression" in m.tasks and not getattr(m, "forecaster", False))


@pytest.mark.parametrize("name", CLASSIFIERS)
def test_every_built_in_classifies_the_mixed_file(name):
    """What: text, dates and gaps go in, labels come out — with probabilities
    where the model has them — and the pattern in the data is learnt.
    How: trains on 180 rows, predicts 60, and scores accuracy against the
    roughly even split of the classes."""
    spec, X, y = _flats()
    model = _told(name, "classification", spec)
    config = dict(model.get_config_space().get_default_configuration())

    predicted = model.fit_predict(config, X[:180], y[:180], X[180:], seed=0)
    labels, proba, classes = model.fit_predict_proba(config, X[:180], y[:180], X[180:], seed=0)

    assert set(predicted) <= {"yes", "no"} and len(predicted) == 60
    assert proba.shape == (60, 2) and sorted(classes) == ["no", "yes"]
    assert np.mean(np.asarray(predicted) == y[180:]) > 0.6


@pytest.mark.parametrize("name", REGRESSORS)
def test_every_built_in_regresses_on_the_mixed_file(name):
    """What: the same columns predict a number. How: takes the area as the
    target (dropping the rows where it is missing) and checks the predictions
    are finite and correlate with the truth."""
    spec, X, y = _flats(target="area_m2")
    known = ~pd.isna(y)
    X, y = X[known], y[known].astype(float)
    model = _told(name, "regression", spec)
    config = dict(model.get_config_space().get_default_configuration())

    predicted = np.asarray(model.fit_predict(config, X[:160], y[:160], X[160:], seed=0), dtype=float)

    assert np.isfinite(predicted).all()
    assert np.corrcoef(predicted, y[160:])[0, 1] > 0.5


def test_a_model_in_its_own_process_is_told_the_kinds(tmp_path):
    """What: `feature_kinds` reaches a model running in another process before
    its first trial. How: a model that writes what it was told to a file, run
    for one trial in a session given the kinds."""
    seen = tmp_path / "kinds.json"
    model_file = tmp_path / "model.py"
    model_file.write_text(HEADER + f'''

class Recorder(BaseModel):
    name = "Recorder"

    def get_config_space(self, seed: int = 0):
        cs = ConfigurationSpace(seed=seed)
        cs.add([Integer("k", (1, 5), default=3)])
        return cs

    def fit_predict(self, config, X_train, y_train, X_val, seed=0):
        with open({str(seen)!r}, "w") as f:
            f.write(__import__("json").dumps(list(self.feature_kinds)))
        return [y_train[0]] * len(X_val)
''', encoding="utf-8")
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    y = np.array(["a", "b", "a", "b"], dtype=object)
    splits = holdout(X[:2], y[:2], X[2:], y[2:])

    with model_session(launch_local(sys.executable, model_file), splits,
                       feature_kinds=("numeric", "categorical")) as remote:
        evaluate_trial(remote, {"k": 1}, splits, {}, seed=0)

    assert json.loads(seen.read_text()) == ["numeric", "categorical"]


def test_an_exported_model_predicts_from_the_raw_csv_what_it_predicted_here(tmp_path):
    """What: the exported file encodes a CSV of text and dates the way
    Codesigner did, so its predictions are the model's own. How: exports the
    forest with the fixture's encoding, runs it on the raw CSV in a fresh
    interpreter, and compares with fitting the same configuration here."""
    from core.processing import resolve

    spec, X, y = _flats()
    model = _told("Random Forest", "classification", spec)
    config = dict(model.get_config_space().get_default_configuration())
    plan, _ = resolve({}, model)
    script = export_model(model, config=config, task="classification", seed=0, columns=spec,
                          plan=plan, about="A forest.")
    path, out = tmp_path / "model.py", tmp_path / "predictions.csv"
    path.write_text(script)

    done = subprocess.run([sys.executable, str(path), str(FLATS), "--predict", str(FLATS),
                           "--out", str(out)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr

    here = _told("Random Forest", "classification", spec).fit_predict(config, X, y, X, seed=0)
    assert pd.read_csv(out).iloc[:, 0].tolist() == list(here)


def test_a_very_wide_datasets_kinds_reach_a_model_in_its_own_process(tmp_path):
    """What: a kinds list too long for one command-line argument (Linux caps one
    at 128 KiB) still reaches the model, through a file the harness reads, and
    the file is gone once the model is. How: tells a recording model twenty
    thousand kinds and reads back how many and which it got."""
    import glob
    import tempfile

    seen = tmp_path / "kinds.json"
    model_file = tmp_path / "model.py"
    model_file.write_text(HEADER + f'''

class Recorder(BaseModel):
    name = "Recorder"

    def get_config_space(self, seed: int = 0):
        cs = ConfigurationSpace(seed=seed)
        cs.add([Integer("k", (1, 5), default=3)])
        return cs

    def fit_predict(self, config, X_train, y_train, X_val, seed=0):
        with open({str(seen)!r}, "w") as f:
            f.write(__import__("json").dumps(list(self.feature_kinds)))
        return [y_train[0]] * len(X_val)
''', encoding="utf-8")
    kinds = ("categorical",) + ("numeric",) * 19_999
    X = np.zeros((4, 1))
    y = np.array(["a", "b", "a", "b"], dtype=object)
    splits = holdout(X[:2], y[:2], X[2:], y[2:])
    before = set(glob.glob(f"{tempfile.gettempdir()}/codesigner-arg-*"))

    with model_session(launch_local(sys.executable, model_file), splits,
                       feature_kinds=kinds) as remote:
        evaluate_trial(remote, {"k": 1}, splits, {}, seed=0)

    assert tuple(json.loads(seen.read_text())) == kinds
    assert set(glob.glob(f"{tempfile.gettempdir()}/codesigner-arg-*")) == before
