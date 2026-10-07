import copy
from pathlib import Path

import pytest
from ConfigSpace import ConfigurationSpace, Integer

from core.io import _load_splits
from core.metrics import METRICS, metrics_for
from core.models import BaseModel
from core.registry import MODELS
from core.optimizers.trial import evaluate_trial
from core.optimizers import (
    BaseOptimizer,
    GridOptimizer,
    OptimizationResult,
    RandomOptimizer,
    SMACOptimizer,
    TrialCollector,
)
from core.optimizers.base import merge_stopping

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DATASETS_DIR = Path(__file__).parent.parent / "datasets"


class FakeModel(BaseModel):
    """Instant, deterministic model for exercising optimizer loops in tests.

    No training happens: it predicts a pattern chosen by the config, so what it
    scores is a pure function of that config and stable across runs. It cannot
    simply return a score — a model never sees the validation labels — so the
    config decides how often the pattern lines up with them instead.
    """

    name = "Fake Model"

    def get_config_space(self, seed: int = 0) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add([Integer("a", (1, 10), default=5), Integer("b", (1, 10), default=5)])
        return cs

    def fit_predict(self, config, X_train, y_train, X_val, seed: int = 0):
        labels = sorted({str(label) for label in y_train})
        a, b = int(config["a"]), int(config["b"])
        return [labels[(i * a + b) % len(labels)] for i in range(len(X_val))]


class FakeOptimizer(BaseOptimizer):
    """Instant optimizer for run-lifecycle tests: real TrialCollector bookkeeping
    (resume, incumbent, cancellation) with no actual search or importance math."""

    name = "Fake"
    params_schema = []

    def optimize(self, model, X_train, y_train, X_val, y_val,
                 metrics: dict, primary_metric: str,
                 n_trials=None, previous_result=None, seed: int = 0, cancel_event=None,
                 stopping: dict | None = None, splits=None):
        from core.splits import holdout
        splits = splits if splits is not None else holdout(X_train, y_train, X_val, y_val)
        config_space = model.get_config_space(seed=seed)
        collector = TrialCollector(
            trial_offset=len(previous_result.trials) if previous_result else 0,
            initial_best_score=previous_result.best_score if previous_result else float("-inf"),
            initial_best_config=previous_result.best_config if previous_result else None,
            stopping=merge_stopping(n_trials, stopping),
        )
        while not collector.done:
            if cancel_event and cancel_event.is_set():
                break
            cfg = dict(config_space.sample_configuration())
            all_scores, run_info = evaluate_trial(model, cfg, splits, metrics, seed=seed)
            collector.record(cfg, all_scores[primary_metric], all_scores, run_info=run_info)

        all_trials = (previous_result.trials if previous_result else []) + collector.results
        params = list(config_space.keys())
        uniform = {p: 1.0 / len(params) for p in params}
        return OptimizationResult(
            trials=all_trials,
            primary_metric=primary_metric,
            best_config=max(all_trials, key=lambda t: t.scores[primary_metric]).config
                        if all_trials else {},
            best_score=max((t.scores[primary_metric] for t in all_trials), default=0.0),
            hyperparameter_importance={m: dict(uniform) for m in metrics},
            hyperparameter_importance_warning={m: None for m in metrics},
            metadata={"stopped_by": collector.stopped_by},
        )


class PreCancelled:
    """Minimal cancel flag honouring the .is_set() contract, always cancelled.

    Deliberately not a threading.Event: it proves optimizers depend only on
    the .is_set() method, which any cancellation backend can provide.
    """

    def is_set(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def media_stays_out_of_the_repo(settings, tmp_path):
    """A throwaway MEDIA_ROOT for every test, not only the web layer's.

    Any test that creates an experiment stores what it uploads; left to the
    default, that is the developer's own media/ directory, where it outlives
    the test and counts against nobody's idea of what is stored there.
    """
    settings.MEDIA_ROOT = str(tmp_path / "media")


@pytest.fixture(autouse=True)
def runs_execute_synchronously(settings):
    """Keep inline runs synchronous for the whole suite.

    The dev server hands an immediate-mode run to a background thread so the
    request returns at once; under test that would race every assertion about a
    launched run, so launching executes in the caller instead. The tests that
    cover the threaded dispatch itself opt back in.
    """
    settings.RUN_IMMEDIATE_IN_THREAD = False


@pytest.fixture(autouse=True)
def suite_runs_in_its_own_configuration(settings):
    """Pin the settings that decide *what kind of instance this is*.

    `config/settings.py` reads `.env`, which is a developer's own file — so
    without this the suite inherits whatever they last left in it. Two of those
    settings change what the tests are even testing:

    * `REQUIRE_LOGIN` decides whether accounts exist at all. Left on, every test
      that does not ask for accounts silently becomes a test of the login wall.
    * `RUN_BACKEND` decides where a run executes. Left at `slurm`, the run
      tests would submit real jobs to a real cluster — slowly, and only on a
      machine that can reach it.

    Pinned to the defaults rather than to anything clever: these are what a
    fresh checkout has, and a test that wants the other mode says so (the
    `hosted` fixtures, and `tests/ui/runs/test_cluster_backend.py`).
    """
    settings.REQUIRE_LOGIN = False
    settings.RUN_BACKEND = "local"


@pytest.fixture
def metrics() -> dict:
    """The classification metrics the application scores with — the real ones,
    not a copy.

    They used to be redefined here. Now that scoring is the application's job
    rather than each model's, a copy could drift from what production computes
    and no test would notice. Only the classification ones: the fixtures'
    datasets have class labels, and a regression metric scored on them fails
    every trial. A regression test asks for `metrics_for("regression")` itself.
    """
    return {name: METRICS[name] for name in metrics_for("classification")}


@pytest.fixture
def models() -> dict:
    """The registry models, keyed as the app registers them — read from the
    registry, so a model added to it is tested without anyone listing it here.
    Copies, so a test that tells one its task does not tell every later test."""
    return {name: copy.copy(model) for name, model in MODELS.items()}


@pytest.fixture
def optimizers() -> dict:
    """The registry optimizers, keyed as the app registers them."""
    return {
        "SMAC": SMACOptimizer(),
        "Random Search": RandomOptimizer(),
        "Grid Search": GridOptimizer(),
    }


@pytest.fixture
def iris_splits():
    """(X_train, X_val, y_train, y_val) for datasets/iris.csv with seed 0."""
    return _load_splits(DATASETS_DIR / "iris.csv", seed=0)


@pytest.fixture
def tiny_splits():
    """A four-row split: enough for a model to predict against and be scored.

    Optimizer-contract tests used to pass None for the arrays, because the fake
    model invented its own score and never touched them. Scoring is real now, so
    there has to be something to score.
    """
    import numpy as np

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(["a", "b", "a", "b"], dtype=object)
    return X[:2], X[2:], y[:2], y[2:]     # X_train, X_val, y_train, y_val


def export_ihpo(client, pk, timestamps="keep", tracebacks="keep"):
    """Download one experiment's .ihpo, answering the questions export asks.

    Exporting is a POST rather than a GET because the page asks first what goes
    in the file: the per-trial timestamps, and a failed trial's stored traceback
    — see `ui.views.experiment_export`. Both default here to the answer that
    changes nothing about the file, for tests that only want the file.
    """
    from django.urls import reverse

    return client.post(reverse("ui:experiment_export", args=[pk]),
                       {"timestamps": timestamps, "tracebacks": tracebacks})


def start_draft(client):
    """Press New: a draft, as the sidebar's button makes one."""
    from django.urls import reverse

    from ui.models import Experiment

    response = client.post(reverse("ui:new_experiment"))
    return Experiment.objects.get(pk=int(response["Location"].strip("/").split("/")[1]))


def get_new_experiment(client):
    """The page a new experiment is set up on: a fresh draft's step 1."""
    from django.urls import reverse

    return client.get(reverse("ui:setup", args=[start_draft(client).pk]))


def get_optimizer_step(client):
    """The page an experiment's optimizer is chosen on: step 2 of a draft
    whose step 1 is saved as iris tuned with a Random Forest."""
    from django.urls import reverse

    draft = start_draft(client)
    client.post(reverse("ui:setup", args=[draft.pk]), {
        "name": "optimizer step", "task": "classification", "model_name": "Random Forest",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "evaluation_scheme": "holdout",
        "evaluation_value": 0.2, "seed": 0})
    return client.get(reverse("ui:setup_optimizer", args=[draft.pk]))


def post_new_experiment(client, data, **kwargs):
    """Set up an experiment from *data* and create it, the way the pages do:
    New, step 1 saved with *data*, step 2 with *data*'s optimizer and its
    settings (the first offered when it names none), step 3 saved as it
    stands, Create.

    The response is what a test of creating an experiment wants to read: step
    1's own when it was refused (the form, with its errors), else Create's —
    a redirect to the new experiment's dashboard."""
    from django.urls import reverse

    from ui.registry import OPTIMIZERS

    draft = start_draft(client)
    response = client.post(reverse("ui:setup", args=[draft.pk]), data, **kwargs)
    if response.status_code != 302 or not response["Location"].endswith("/setup/optimizer/"):
        return response
    optimizer = {k: v for k, v in data.items() if k == "optimizer_name" or k.startswith("opt_")}
    optimizer.setdefault("optimizer_name", next(iter(OPTIMIZERS)))
    client.post(reverse("ui:setup_optimizer", args=[draft.pk]), optimizer)
    client.post(reverse("ui:setup_data", args=[draft.pk]), {})
    return client.post(reverse("ui:setup_create", args=[draft.pk]))
