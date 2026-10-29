import copy
from pathlib import Path

import pytest
from ConfigSpace import ConfigurationSpace, Integer

from core.datasets import load_splits
from core.metrics import METRICS, metrics_for
from codesigner_model import BaseModel
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

def pytest_addoption(parser):
    """`pytest_addoption` registers the evals' options (tests/assistant/evals),
    here because only a top-level conftest may add options."""
    group = parser.getgroup("assistant evals")
    group.addoption("--eval", action="store_true",
                    help="Run the assistant's evals against ASSISTANT_LLM_*.")
    group.addoption("--eval-repeat", type=int, default=1, help="Runs of each eval case.")
    group.addoption("--eval-report", default="eval-report.json",
                    help="Where the evals' report is written.")
    group.addoption("--eval-case", action="append", default=[],
                    help="Only the eval cases whose id or group starts with this.")

DATASETS_DIR = Path(__file__).parent.parent / "datasets"


class FakeModel(BaseModel):
    """`FakeModel` trains nothing and predicts a label pattern chosen by the
    config, which makes the score a pure function of the config.
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
    """`FakeOptimizer` keeps real `TrialCollector` bookkeeping and skips search
    and importance."""

    name = "Fake"
    params_schema = []

    def optimize(self, model, X_train, y_train, X_val, y_val,
                 metrics: dict, primary_metric: str,
                 n_trials=None, previous_result=None, seed: int = 0, cancel_event=None,
                 stopping: dict | None = None, splits=None,
                 priors=None, configurations=None):
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
    """`PreCancelled` proves optimizers only rely on `.is_set()`.
    """

    def is_set(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def media_stays_out_of_the_repo(settings, tmp_path):
    """Uploads would otherwise land in the developer's own media/ directory.
    """
    settings.MEDIA_ROOT = str(tmp_path / "media")


@pytest.fixture(autouse=True)
def runs_execute_synchronously(settings):
    """A background run thread would race every assertion; tests of threaded
    dispatch opt back in.
    """
    settings.RUN_IMMEDIATE_IN_THREAD = False


@pytest.fixture(autouse=True)
def suite_runs_in_its_own_configuration(settings):
    """A developer's `.env` may set `REQUIRE_LOGIN`, which turns tests into
    login-wall tests, `RUN_BACKEND=slurm`, which submits real cluster jobs, or
    `ASSISTANT_ENABLED`, which puts a chat with an absent model on every page.
    `suite_runs_in_its_own_configuration` restores the fresh-checkout defaults,
    and the `hosted` fixtures opt out.
    """
    settings.REQUIRE_LOGIN = False
    settings.RUN_BACKEND = "local"
    settings.ASSISTANT_ENABLED = False
    settings.ASSISTANT_TURNS_INLINE = True


@pytest.fixture
def metrics() -> dict:
    """`metrics` returns the real classification metrics, since a copy could
    drift from production scoring. Regression metrics would fail every trial on
    the class-labelled fixture datasets.
    """
    return {name: METRICS[name] for name in metrics_for("classification")}


@pytest.fixture
def models() -> dict:
    """`models` copies each registry model because a test may set a model's
    task."""
    return {name: copy.copy(model) for name, model in MODELS.items()}


@pytest.fixture
def optimizers() -> dict:
    """`optimizers` maps display names to classes."""
    return {
        "SMAC": SMACOptimizer,
        "Random Search": RandomOptimizer,
        "Grid Search": GridOptimizer,
    }


@pytest.fixture
def iris_splits():
    """Seed-0 iris splits, in `load_splits` order."""
    return load_splits(DATASETS_DIR / "iris.csv", seed=0)


@pytest.fixture
def tiny_splits():
    """`tiny_splits` gives four rows, enough to fit and score a model.
    """
    import numpy as np

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(["a", "b", "a", "b"], dtype=object)
    return X[:2], X[2:], y[:2], y[2:]


def export_ihpo(client, pk, timestamps="keep", tracebacks="keep"):
    """Export is a POST that asks about timestamps and tracebacks, and
    `export_ihpo` keeps both by default.
    """
    from django.urls import reverse

    return client.post(reverse("ui:experiment_export", args=[pk]),
                       {"timestamps": timestamps, "tracebacks": tracebacks})


def start_draft(client):
    """`start_draft` presses New and returns the draft."""
    from django.urls import reverse

    from experiments.models import Experiment

    response = client.post(reverse("ui:new_experiment"))
    return Experiment.objects.get(pk=int(response["Location"].strip("/").split("/")[1]))


def get_new_experiment(client):
    """`get_new_experiment` opens step 1 of a fresh draft."""
    from django.urls import reverse

    return client.get(reverse("ui:setup", args=[start_draft(client).pk]))


def get_optimizer_step(client):
    """Step 2, after step 1 saves iris and Random Forest."""
    from django.urls import reverse

    draft = start_draft(client)
    client.post(reverse("ui:setup", args=[draft.pk]), {
        "name": "optimizer step", "task": "classification", "model_name": "Random Forest",
        "demo_dataset": str(DATASETS_DIR / "iris.csv"), "evaluation_scheme": "holdout",
        "evaluation_value": 0.2, "seed": 0})
    return client.get(reverse("ui:setup_optimizer", args=[draft.pk]))


def post_new_experiment(client, data, **kwargs):
    """`post_new_experiment` walks New and the three setup steps to Create,
    using the first optimizer when *data* names none. A step 1 that fails
    validation returns its form response, and success returns the redirect to
    the dashboard."""
    from django.urls import reverse

    from core.registry import OPTIMIZERS

    draft = start_draft(client)
    response = client.post(reverse("ui:setup", args=[draft.pk]), data, **kwargs)
    if response.status_code != 302 or not response["Location"].endswith("/setup/optimizer/"):
        return response
    optimizer = {k: v for k, v in data.items() if k == "optimizer_name" or k.startswith("opt_")}
    optimizer.setdefault("optimizer_name", next(iter(OPTIMIZERS)))
    client.post(reverse("ui:setup_optimizer", args=[draft.pk]), optimizer)
    client.post(reverse("ui:setup_data", args=[draft.pk]), {})
    return client.post(reverse("ui:setup_create", args=[draft.pk]))


def launch_local(python, model_file) -> list[str]:
    """`launch_local` builds argv running the model harness under *python*
    without an environment."""
    from core.modelhost import HARNESS

    return [str(python), "-B", str(HARNESS), "--model-file", str(model_file)]
