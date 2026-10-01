"""An exported model is a script that runs on its own.

Every built-in is exported with its default configuration and run in a fresh
interpreter — once with the Codesigner SDK importable and once without it — on
a bundled dataset, and its predictions are read back.
"""

import ast
import inspect
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from core.model_export import configuration, export_model
from core.model_source import inspect_model_source
from core.registry import MODELS

from tests.conftest import DATASETS_DIR

BUILT_INS = sorted(MODELS)


def _export(name, config=None, task="classification"):
    model = MODELS[name]
    source = inspect.getsource(sys.modules[type(model).__module__])
    if config is None:
        config = dict(model.get_config_space().get_default_configuration())
    return export_model(source, class_name=type(model).__name__, task=task,
                        config=config, seed=0, description="A test export.")


def _run(script, tmp_path, *args, without_sdk=False):
    path = tmp_path / "model.py"
    path.write_text(script)
    env = None
    if without_sdk:
        # A package of the same name that cannot be imported, found first.
        blocker = tmp_path / "blocker" / "codesigner_model"
        blocker.mkdir(parents=True)
        (blocker / "__init__.py").write_text("raise ImportError('not installed here')\n")
        env = {"PYTHONPATH": str(blocker.parent), "PATH": "/usr/bin:/bin"}
    return subprocess.run([sys.executable, str(path), *map(str, args)],
                          capture_output=True, text=True, cwd=tmp_path, env=env, timeout=120)


@pytest.mark.parametrize("name", BUILT_INS)
def test_every_built_in_exports_a_valid_model_file(name):
    """What: the exported file parses, still reads as a Codesigner model under
    the same name, and carries the configuration at the top.
    How: parses it, runs the upload inspector over it, and reads `CONFIG` back
    out of the syntax tree."""
    model = MODELS[name]
    config = dict(model.get_config_space().get_default_configuration())
    script = _export(name, config)

    tree = ast.parse(script)
    info, error = inspect_model_source(script.encode())
    assert error is None and info.name == model.name

    assigned = {node.targets[0].id: node.value for node in tree.body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    assert ast.literal_eval(assigned["CONFIG"]) == configuration(config)
    assert ast.literal_eval(assigned["TASK"]) == "classification"


@pytest.mark.parametrize("without_sdk", [False, True], ids=["with-sdk", "without-sdk"])
@pytest.mark.parametrize("name", BUILT_INS)
def test_every_built_in_export_trains_and_predicts(name, without_sdk, tmp_path):
    """What: run as a script, the export trains on a CSV and writes one
    prediction per row of another, whether or not the SDK is installed.
    How: runs it in a subprocess on iris, predicting iris back, and reads the
    predictions CSV it wrote."""
    out = tmp_path / "predictions.csv"
    done = _run(_export(name), tmp_path, DATASETS_DIR / "iris.csv",
                "--predict", DATASETS_DIR / "iris.csv", "--out", out, without_sdk=without_sdk)

    assert done.returncode == 0, done.stderr
    predictions = pd.read_csv(out)
    iris = pd.read_csv(DATASETS_DIR / "iris.csv")
    assert len(predictions) == len(iris)
    assert set(predictions.iloc[:, 0]) <= set(iris.iloc[:, -1])


@pytest.mark.parametrize("name", BUILT_INS)
def test_every_built_in_export_saves_its_fit(name, tmp_path):
    """What: `--save` pickles a fitted estimator that predicts on its own.
    How: saves it from a subprocess, unpickles it here and predicts iris."""
    done = _run(_export(name), tmp_path, DATASETS_DIR / "iris.csv", "--save", tmp_path / "m.pkl")
    assert done.returncode == 0, done.stderr

    import pickle

    with open(tmp_path / "m.pkl", "rb") as f:
        fitted = pickle.load(f)
    iris = pd.read_csv(DATASETS_DIR / "iris.csv")
    assert len(fitted.predict(iris.iloc[:, :-1].to_numpy())) == len(iris)


def test_the_runner_refuses_to_do_nothing(tmp_path):
    """What: with neither --predict nor --save there is nothing to do, and the
    runner says so rather than training for nothing.
    How: runs the export with only a training file."""
    done = _run(_export(BUILT_INS[0]), tmp_path, DATASETS_DIR / "iris.csv")
    assert done.returncode != 0
    assert "nothing to do" in done.stderr


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


def test_a_custom_model_keeps_its_header_and_gains_the_runners(tmp_path):
    """What: an uploaded model's own header survives, with pandas added for the
    runner, and `from __future__` stays the first statement.
    How: exports a small custom model and runs it without the SDK."""
    script = export_model(CUSTOM, class_name="Majority", task="classification",
                          config={"suffix": "!"}, seed=0)

    header = script.split("\n# ///\n", 1)[0]
    assert '"numpy"' in header and '"pandas"' in header
    assert inspect_model_source(script.encode())[1] is None

    out = tmp_path / "p.csv"
    done = _run(script, tmp_path, DATASETS_DIR / "iris.csv", "--predict",
                DATASETS_DIR / "iris.csv", "--out", out, without_sdk=True)
    assert done.returncode == 0, done.stderr
    assert pd.read_csv(out).iloc[0, 0].endswith("!")


def test_a_custom_model_without_fit_cannot_be_saved(tmp_path):
    """What: --save needs a `fit` method, and a model without one is told so.
    How: asks the custom model above to save itself."""
    script = export_model(CUSTOM, class_name="Majority", task="classification",
                          config={"suffix": ""}, seed=0)
    done = _run(script, tmp_path, DATASETS_DIR / "iris.csv", "--save", tmp_path / "m.pkl",
                without_sdk=True)
    assert done.returncode != 0
    assert "no fit" in done.stderr


def test_a_model_without_a_header_gets_one():
    """What: a model with no PEP 723 header gets one naming what the runner needs.
    How: exports a header-less source and inspects the result."""
    source = CUSTOM.split("\n# ///\n", 1)[1].replace("import numpy as np\n", "")
    script = export_model(source, class_name="Majority", task="classification",
                          config={}, seed=0)
    info, error = inspect_model_source(script.encode())
    assert error is None and "pandas" in info.dependencies


def test_inactive_hyperparameters_are_left_out():
    """What: a conditional hyperparameter that does not apply is absent from
    CONFIG, not written as NaN. How: passes NaN and a numpy scalar through."""
    import numpy as np

    assert configuration({"a": float("nan"), "b": np.int64(3), "c": "x"}) == {"b": 3, "c": "x"}


@pytest.mark.parametrize("name", BUILT_INS)
def test_built_in_modules_are_self_contained(name):
    """What: a built-in's module imports nothing from Codesigner but BaseModel,
    so its source is the whole model when exported.
    How: walks the module's imports for relative or Codesigner ones."""
    tree = ast.parse(inspect.getsource(sys.modules[type(MODELS[name]).__module__]))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").startswith("core")):
            assert node.module in ("base",) and [a.name for a in node.names] == ["BaseModel"], (
                f"{name} imports {node.module}")
