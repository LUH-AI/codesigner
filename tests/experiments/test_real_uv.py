"""`test_real_uv` runs the real uv, catching flag changes that the shim-based
tests miss. The model only needs ConfigSpace.
"""

import shutil
import textwrap

import pytest
from django.core.files.base import ContentFile

from tests.conftest import DATASETS_DIR
from experiments.models import Experiment
from experiments.services import modelenv
from experiments.services.run import create_run, execute_run

pytestmark = [
    pytest.mark.uv,
    pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed"),
]

MODEL = textwrap.dedent('''
    # /// script
    # dependencies = ["ConfigSpace"]
    # ///
    from ConfigSpace import ConfigurationSpace, Integer

    from codesigner_model import BaseModel


    class Counter(BaseModel):
        """Predicts the training set's most common label, k times over."""

        name = "Counter"

        def get_config_space(self, seed: int = 0):
            cs = ConfigurationSpace(seed=seed)
            cs.add([Integer("k", (1, 3), default=1)])
            return cs

        def fit_predict(self, config, X_train, y_train, X_val, seed=0):
            counts = {}
            for label in y_train:
                counts[label] = counts.get(label, 0) + 1
            best = max(counts, key=lambda label: counts[label])
            return [best] * len(X_val)
''').encode()


@pytest.fixture
def sdk_wheel(tmp_path_factory, settings):
    """`sdk_wheel` builds the contract wheel like the image."""
    import subprocess
    import sys

    out = tmp_path_factory.mktemp("sdk-wheel")
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(out),
                    "./model_sdk"], check=True, capture_output=True)
    settings.MODEL_SDK_WHEEL = str(out)
    return out


@pytest.fixture
def experiment():
    exp = Experiment(
        name="real-uv", model_name="Counter", optimizer_name="Random Search",
        metric_names=["accuracy", "f1", "precision", "recall(macro)"], seed=0)
    exp.data.model_file.save("counter.py", ContentFile(MODEL), save=False)
    exp.data.dataset.save("iris.csv", ContentFile((DATASETS_DIR / "iris.csv").read_bytes()),
                     save=False)
    exp.save()
    return exp


def test_uv_builds_and_pins_a_real_environment(experiment, sdk_wheel):
    modelenv.prepare_environment(experiment.pk)
    experiment.refresh_from_db()

    assert experiment.env_status == Experiment.ENV_READY, experiment.env_error
    assert modelenv.lock_path(experiment).is_file()
    # imported and instantiated inside the env
    assert experiment.env_meta["model_class"] == "Counter"
    assert experiment.env_meta["python"]


def test_the_environment_is_not_this_one(experiment, sdk_wheel):
    """The model runs under an interpreter other than the app's."""
    import sys

    modelenv.prepare_environment(experiment.pk)
    experiment.refresh_from_db()

    launch, refusal = modelenv.resolve_runner(experiment)
    assert refusal == ""
    assert launch is not None
    assert launch[0].endswith("uv")
    assert sys.executable not in launch
    # uv must target the runner, which owns the lock; targeting the model file
    # fails at run time.
    assert launch[launch.index("--script") + 1] == str(modelenv.runner_path(experiment))
    assert modelenv.lock_path(experiment).name.startswith(
        modelenv.runner_path(experiment).name)


def test_a_real_run_goes_through_the_built_environment(experiment, sdk_wheel):
    modelenv.prepare_environment(experiment.pk)
    experiment.refresh_from_db()
    assert experiment.env_status == Experiment.ENV_READY, experiment.env_error

    run = create_run(experiment, {"max_trials": 3}, "accuracy")
    execute_run(run.id)

    run.refresh_from_db()
    experiment.refresh_from_db()
    assert run.status == "done", run.error
    assert run.trial_count == 3
    assert len(experiment.data.result["data"]) == 3
    # scored here, predicted there
    assert all(0.0 <= t["scores"]["accuracy"] <= 1.0 for t in experiment.data.result["data"])
