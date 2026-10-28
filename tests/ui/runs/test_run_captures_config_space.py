"""`execute_run` stores the model's search space, since a custom model's process
ends with the run and later pages rebuild read-only.
"""

import pytest

from core import io

from tests.conftest import DATASETS_DIR


def _an_experiment():
    from experiments.services import snapshot as adapter

    return adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "space", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search",
        "optimizer_params": {}, "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy"], "seed": 0,
        "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }, adopt_paths=True)


def test_a_run_stores_the_search_space():
    """The stored space has the model's own keys."""
    from core.registry import MODELS
    from experiments.services.run import create_run, execute_run

    exp = _an_experiment()
    assert exp.data.config_space is None

    execute_run(create_run(exp, {"max_trials": 3}, "accuracy").id)
    exp.refresh_from_db()

    stored = io.config_space_from_serialized(exp.data.config_space, seed=exp.data.seed)
    expected = MODELS["Random Forest"].get_config_space(seed=exp.data.seed)
    assert stored is not None
    assert list(stored.keys()) == list(expected.keys())


def test_the_stored_space_reaches_the_figures():
    """`config_space_for` answers with the run's space through
    `rebuild_experiment`, the path every page render takes.
    """
    from experiments.services.run import create_run, execute_run
    from experiments.services.snapshot import config_space_for, rebuild_experiment

    exp = _an_experiment()
    execute_run(create_run(exp, {"max_trials": 3}, "accuracy").id)
    exp.refresh_from_db()

    built = rebuild_experiment(exp)

    assert built is not None
    assert built["config_space"] == exp.data.config_space
    assert config_space_for(built) is not None


def test_a_model_that_cannot_describe_its_space_does_not_fail_the_run():
    """`_remember_config_space` leaves the stored space unset when the model
    raises.
    """
    from experiments.services.run import _remember_config_space

    class _Mute:
        def get_config_space(self, seed=0):
            raise RuntimeError("no space here")

    exp = _an_experiment()

    _remember_config_space(exp.pk, _Mute(), 0)

    exp.refresh_from_db()
    assert exp.data.config_space is None


pytestmark = pytest.mark.django_db
