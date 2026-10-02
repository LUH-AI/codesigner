"""An exported model is the tuned model, and runs on its own.

Every tabular built-in, for every task it does, is exported with a sampled
configuration and run in a fresh interpreter that cannot import Codesigner or
its SDK, on a bundled dataset — and must predict what the same model, fitted on
the same rows here, predicts. Its file must carry nothing of the search.
"""

import ast
import copy
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from core import encoding, processing
from core.model_export import configuration, export_model, export_upload
from core.registry import MODELS

from tests.conftest import DATASETS_DIR, FIXTURES_DIR

#: The built-ins that train on a table and predict another — every one but the
#: forecasters, whose export forecasts from a history instead (see
#: tests/core/test_forecasting.py).
TABULAR = sorted(n for n, m in MODELS.items() if not getattr(m, "forecaster", False))

#: Every (model, task) a tabular built-in can be exported for.
CASES = [(name, task) for name in TABULAR for task in MODELS[name].tasks]

#: What only a search has: none of it may appear in an exported built-in.
SEARCH_NAMES = {"get_config_space", "ConfigurationSpace", "ConfigSpace", "codesigner_model",
                "Tunable", "OptimizableBaseModel", "fit_predict", "CONFIG", "TASK", "SEED",
                "FEATURE_KINDS"}


def _data(task):
    return FIXTURES_DIR / "flats.csv" if task == "classification" else DATASETS_DIR / "diabetes.csv"


def _model_for(name, task, path, processing_choices=None):
    frame = pd.read_csv(path)
    frame = frame[frame.iloc[:, -1].notna()]
    spec = encoding.fit(frame.iloc[:, :-1])
    model = copy.copy(MODELS[name])
    model.task, model.feature_kinds = task, tuple(encoding.feature_kinds(spec))
    model.processing = dict(processing_choices or {})
    X, y = encoding.encode(spec, frame.iloc[:, :-1]), frame.iloc[:, -1].to_numpy()
    return model, spec, frame, X, y


def _export(model, spec, config):
    plan, _ = processing.resolve(model.processing, model)
    return export_model(model, config=config, task=model.task, seed=0, columns=spec, plan=plan,
                        about=f"{model.name} predicting the target ({model.task}).")


def _run(script, tmp_path, *args):
    """Runs *script* in a fresh interpreter where neither Codesigner nor its
    SDK can be imported: a package of each name that refuses to import is
    found first."""
    path = tmp_path / "model.py"
    path.write_text(script)
    blockers = tmp_path / "blockers"
    for package in ("codesigner_model", "core", "ConfigSpace"):
        (blockers / package).mkdir(parents=True, exist_ok=True)
        (blockers / package / "__init__.py").write_text("raise ImportError('not here')\n")
    return subprocess.run([sys.executable, str(path), *map(str, args)], capture_output=True,
                          text=True, cwd=tmp_path, timeout=180,
                          env={"PYTHONPATH": str(blockers), "PATH": "/usr/bin:/bin"})


@pytest.mark.parametrize("name, task", CASES, ids=[f"{n}-{t}" for n, t in CASES])
def test_an_export_predicts_what_the_tuned_model_does(name, task, tmp_path):
    """What: the exported file, run without Codesigner on the dataset, predicts
    exactly what the model fitted here on the same rows does — processing and
    all. How: exports a sampled configuration, trains it on the whole file,
    predicts its first 20 rows both ways."""
    model, spec, frame, X, y = _model_for(name, task, _data(task))
    config = dict(model.get_config_space(seed=3).sample_configuration())
    new = tmp_path / "new.csv"
    frame.head(20).to_csv(new, index=False)

    done = _run(_export(model, spec, config), tmp_path, _data(task), "--predict", new,
                "--out", tmp_path / "out.csv")
    assert done.returncode == 0, done.stderr
    there = pd.read_csv(tmp_path / "out.csv").iloc[:, 0].to_numpy()
    here = model.fit_predict(config, X, y, X[:20], seed=0)

    if task == "regression":
        # Some boosters predict in single precision, which a CSV writes as such.
        np.testing.assert_allclose(there.astype(float), np.asarray(here, dtype=float), rtol=1e-6)
    else:
        assert list(map(str, there)) == list(map(str, here))


@pytest.mark.parametrize("name, task", CASES, ids=[f"{n}-{t}" for n, t in CASES])
def test_an_export_carries_nothing_of_the_search(name, task):
    """What: no search space, no SDK, no trial machinery — only the model, its
    hyperparameters and the data processing, each in its own section.
    How: reads every name the exported file defines or uses."""
    model, spec, *_ = _model_for(name, task, _data(task))
    script = _export(model, spec, dict(model.get_config_space().get_default_configuration()))
    tree = ast.parse(script)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
        a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in n.names} | {
        (n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}

    assert not names & SEARCH_NAMES, names & SEARCH_NAMES
    for section in ("The model", "The hyperparameters", "The data processing", "Running it"):
        assert f"# ── {section} " in script


def test_the_hyperparameters_are_the_trials_and_the_processing_the_experiments():
    """What: HYPERPARAMETERS is the configuration as plain values, and
    PROCESSING the experiment's choices as this model resolves them.
    How: exports an SVM told to leave its numbers unscaled, and reads both."""
    model, spec, *_ = _model_for("SVM", "classification", _data("classification"),
                                 {"scale": "none"})
    config = dict(model.get_config_space().get_default_configuration())
    assigned = {n.targets[0].id: n.value for n in ast.parse(_export(model, spec, config)).body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}

    assert ast.literal_eval(assigned["HYPERPARAMETERS"]) == configuration(config)
    assert ast.literal_eval(assigned["PROCESSING"]) == {
        "missing": "impute", "scale": "none", "labels": "one_hot"}


def test_a_dataset_of_numbers_needs_no_column_reading():
    """What: COLUMNS is None when every column is already a number — the file
    reads them as they are. How: exports for diabetes."""
    model, spec, *_ = _model_for("Elastic Net", "regression", DATASETS_DIR / "diabetes.csv")
    script = _export(model, spec, dict(model.get_config_space().get_default_configuration()))

    assert "COLUMNS = None" in script


def test_inactive_hyperparameters_are_left_out():
    """What: a conditional hyperparameter that does not apply is absent from
    HYPERPARAMETERS, not written as NaN. How: passes NaN and a numpy scalar
    through."""
    assert configuration({"a": float("nan"), "b": np.int64(3), "c": "x"}) == {"b": 3, "c": "x"}


CUSTOM = textwrap.dedent('''\
    # /// script
    # requires-python = ">=3.11"
    # dependencies = [
    #     "numpy",
    # ]
    # ///
    from __future__ import annotations

    import numpy as np
    from codesigner_model import BaseModel


    class Majority(BaseModel):
        name = "Majority"

        def get_config_space(self, seed=0):
            return None

        def fit_predict(self, config, X_train, y_train, X_val, seed=0):
            values, counts = np.unique(y_train, return_counts=True)
            return [values[counts.argmax()] + config["suffix"]] * len(X_val)
''')


def test_an_uploaded_model_is_delivered_as_written_without_its_search_space(tmp_path):
    """What: a person's own model keeps its code and its header (pandas added
    for the runner) and `from __future__` first, loses `get_config_space`, and
    runs without the SDK. How: exports a small model and runs it."""
    script = export_upload(CUSTOM, class_name="Majority", task="classification",
                           config={"suffix": "!"}, seed=0, about="Majority, tuned.")

    header = script.split("\n# ///\n", 1)[0]
    assert '"numpy"' in header and '"pandas"' in header
    body = script.split("# ///\n", 1)[1]
    assert ast.get_docstring(ast.parse(body)) == "Majority, tuned."
    assert ast.parse(body).body[1].module == "__future__"
    assert "get_config_space" not in script and "def fit_predict" in script

    out = tmp_path / "p.csv"
    done = _run(script, tmp_path, DATASETS_DIR / "iris.csv", "--predict",
                DATASETS_DIR / "iris.csv", "--out", out)
    assert done.returncode == 0, done.stderr
    assert pd.read_csv(out).iloc[0, 0].endswith("!")


def test_an_uploaded_model_without_a_header_gets_one():
    """What: a model with no PEP 723 header gets one naming what the runner
    needs. How: exports a header-less source and reads its header."""
    source = CUSTOM.split("\n# ///\n", 1)[1]
    script = export_upload(source, class_name="Majority", task="classification", config={})

    header = script.split("\n# ///\n", 1)[0]
    assert header.startswith("# /// script") and '"pandas"' in header
