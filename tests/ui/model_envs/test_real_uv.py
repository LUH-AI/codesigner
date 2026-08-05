"""Building a model environment with the real uv.

Everything else about environments is tested against a shim, which pins our own
argv and error handling but cannot tell us whether uv accepts any of it. These
do — so they are the tests that would catch a uv release changing the flags out
from under us.

Skipped when uv is absent, and the model declares only ConfigSpace so the work
is small and cached after the first run.
"""

import shutil
import textwrap

import pytest
from django.core.files.base import ContentFile

from tests.conftest import DATASETS_DIR
from ui.models import Experiment
from ui.services import modelenv
from ui.services.run import create_run, execute_run

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
    """A built wheel of the contract, as the image makes at build time."""
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
    exp.model_file.save("counter.py", ContentFile(MODEL), save=False)
    exp.dataset.save("iris.csv", ContentFile((DATASETS_DIR / "iris.csv").read_bytes()),
                     save=False)
    exp.save()
    return exp


def test_uv_builds_and_pins_a_real_environment(experiment, sdk_wheel):
    """The commands we construct are ones uv accepts, and the lock lands where
    we look for it."""
    modelenv.prepare_environment(experiment.pk)
    experiment.refresh_from_db()

    assert experiment.env_status == Experiment.ENV_READY, experiment.env_error
    assert modelenv.lock_path(experiment).is_file()
    # answered by a model that was imported and instantiated over there
    assert experiment.env_meta["model_class"] == "Counter"
    assert experiment.env_meta["python"]


def test_the_environment_is_not_this_one(experiment, sdk_wheel):
    """The point of all of it: the model runs under an interpreter that is not
    the application's."""
    import sys

    modelenv.prepare_environment(experiment.pk)
    experiment.refresh_from_db()

    launch, refusal = modelenv.resolve_runner(experiment)
    assert refusal == ""
    assert launch[0].endswith("uv")
    assert sys.executable not in launch
    # The runner is what uv is pointed at, and the lock belongs to the runner —
    # naming the model file here resolves no lock and fails only at run time.
    assert launch[launch.index("--script") + 1] == str(modelenv.runner_path(experiment))
    assert modelenv.lock_path(experiment).name.startswith(
        modelenv.runner_path(experiment).name)


def test_a_real_run_goes_through_the_built_environment(experiment, sdk_wheel):
    """End to end: prepare, then optimize, with the model in its own
    environment for every trial."""
    modelenv.prepare_environment(experiment.pk)
    experiment.refresh_from_db()
    assert experiment.env_status == Experiment.ENV_READY, experiment.env_error

    run = create_run(experiment, n_trials=3, optimize_metric="accuracy")
    execute_run(run.id)

    run.refresh_from_db()
    experiment.refresh_from_db()
    assert run.status == "done", run.error
    assert run.trial_count == 3
    assert len(experiment.result["data"]) == 3
    # scored here, from predictions made over there
    assert all(0.0 <= t["scores"]["accuracy"] <= 1.0 for t in experiment.result["data"])
