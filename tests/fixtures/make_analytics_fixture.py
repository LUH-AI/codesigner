"""Regenerate `analytics.ihpo`, the fixture carrying every analytics field.

`test2.ihpo` predates the sensitivity/mistunability/interactions fields, so it
deserializes them as empty and any test using it can only check that the keys
exist. This one is produced by a real run, so the interactions heatmap and the
other two games have actual numbers to render from.

Committed rather than run per-test: the three HyperSHAP games over two metrics
take about five seconds, which does not belong in the default suite. Run this
deliberately when the result format changes.

    .venv/bin/python tests/fixtures/make_analytics_fixture.py

Deliberately small and deliberately not `test2.ihpo`: nineteen test files
depend on that file, several pinning exact trial numbers, so regenerating it
would be a wide and unrelated change.
"""

from pathlib import Path

from core import io
from core.metrics import METRICS
from core.models import RandomForestModel
from core.optimizers import RandomOptimizer

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent.parent / "datasets" / "iris.csv"
OUT = HERE / "analytics.ihpo"

METRIC_NAMES = ("accuracy", "f1")
N_TRIALS = 6
SEED = 0


def main() -> None:
    X_train, X_val, y_train, y_val = io._load_splits(DATASET, SEED)
    metrics = {name: METRICS[name] for name in METRIC_NAMES}
    optimizer = RandomOptimizer()
    result = optimizer.optimize(
        RandomForestModel(), X_train, y_train, X_val, y_val,
        metrics=metrics, primary_metric="accuracy", n_trials=N_TRIALS, seed=SEED,
    )

    for field in ("hyperparameter_importance", "hyperparameter_sensitivity",
                  "hyperparameter_mistunability", "hyperparameter_interactions"):
        for name in METRIC_NAMES:
            assert getattr(result, field).get(name), f"{field}/{name} came out empty"

    OUT.write_bytes(io.save("analytics", {
        "seed": SEED,
        "dataset_path": str(DATASET),
        "model_name": RandomForestModel().name,
        "model_path": "",
        "cv_folds": 0,
        "metrics": metrics,
        "current_metric": "accuracy",
        "original_metric": "accuracy",
        "optimizer": optimizer,
        "result": result,
    }))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(result.trials)} trials)")


if __name__ == "__main__":
    main()
