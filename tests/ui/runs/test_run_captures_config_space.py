"""A run records the search space it drew from, while it still has a model.

*What:* `execute_run` stores the model's search space on the experiment. It is
the only moment the space is certainly available — a custom model runs in its
own process and is gone afterwards, and every page rendered later rebuilds
read-only — so a run that did not capture it leaves every surrogate-backed
figure with nothing to ask.

*How:* a three-trial run over iris, then the row is re-read and its stored
space decoded and compared against what the model itself reports.
"""

import pytest

from core import io

from tests.conftest import DATASETS_DIR


def _an_experiment():
    from ui.services import snapshot as adapter

    return adapter.experiment_from_snapshot({
        "version": "0.1.0", "name": "space", "model_name": "Random Forest",
        "model_path": "", "optimizer_name": "Random Search",
        "optimizer_params": {}, "primary_metric": None, "original_metric": None,
        "metric_names": ["accuracy"], "seed": 0,
        "dataset_path": str(DATASETS_DIR / "iris.csv"), "result": None,
    }, adopt_paths=True)


def test_a_run_stores_the_search_space():
    """After a run the row carries a space, and it is the model's own."""
    from ui.registry import MODELS
    from ui.services.run import create_run, execute_run

    exp = _an_experiment()
    assert exp.config_space is None

    execute_run(create_run(exp, {"max_trials": 3}, "accuracy").id)
    exp.refresh_from_db()

    stored = io.config_space_from_serialized(exp.config_space, seed=exp.seed)
    expected = MODELS["Random Forest"].get_config_space(seed=exp.seed)
    assert stored is not None
    assert list(stored.keys()) == list(expected.keys())


def test_the_stored_space_reaches_the_figures():
    """A run's own space is what `_config_space_for` answers with afterwards.

    Going through `_rebuild_experiment` rather than asking the row directly,
    since that is the path every page render takes.
    """
    from ui.services.run import create_run, execute_run
    from ui.views import _config_space_for, _rebuild_experiment

    exp = _an_experiment()
    execute_run(create_run(exp, {"max_trials": 3}, "accuracy").id)
    exp.refresh_from_db()

    built = _rebuild_experiment(exp)

    assert built["config_space"] == exp.config_space
    assert _config_space_for(built) is not None


def test_a_model_that_cannot_describe_its_space_does_not_fail_the_run():
    """Bookkeeping must not be able to end a run.

    The capture is a convenience — without it the experiment simply goes on
    asking its model, which is where it was before — so a model that raises
    here leaves the run untouched and the stored space unset.
    """
    from ui.services.run import _remember_config_space

    class _Mute:
        def get_config_space(self, seed=0):
            raise RuntimeError("no space here")

    exp = _an_experiment()

    _remember_config_space(exp.pk, _Mute(), 0)

    exp.refresh_from_db()
    assert exp.config_space is None


pytestmark = pytest.mark.django_db
