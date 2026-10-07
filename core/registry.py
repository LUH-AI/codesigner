"""The models, optimizers and metrics the application offers.

Keys are the human-readable names stored in `.ihpo` files, which is what makes
this the thing `io.build_experiment` resolves a snapshot against.

Lives in `core` rather than `ui` because everything in it is a core object and
none of it needs the web layer — and because a run executed somewhere else
entirely has to resolve the same names with no Django installation to do it in
(see `cluster/run_experiment.py`). Duplicating the three dicts there instead
would work until somebody added a model, at which point the cluster would refuse
that model alone, and say only that it "is not available".

`ui.registry` re-exports this, so the application's own vocabulary is unchanged.
"""

from .metrics import METRICS
from .models import (
    CatBoostModel, ElasticNetModel, ETSModel, ExtraTreesModel, HistGradientBoostingModel,
    KNNModel, LightGBMModel, LogisticRegressionModel, RandomForestModel, SeasonalNaiveModel,
    SVMModel, XGBoostModel,
)
from .optimizers import GridOptimizer, RandomOptimizer, SMACOptimizer

#: In the order the form offers them: the two that have always been here, the
#: other tree ensembles, the boosters, the models that need their columns
#: scaled, then the two that only forecast. Every one runs on a CPU, in this
#: process — and every one that regresses can forecast too (`core.forecasting`).
MODELS = {
    "Random Forest": RandomForestModel(),
    "SVM": SVMModel(),
    "Extra Trees": ExtraTreesModel(),
    "HistGradientBoosting": HistGradientBoostingModel(),
    "LightGBM": LightGBMModel(),
    "XGBoost": XGBoostModel(),
    "CatBoost": CatBoostModel(),
    "k-Nearest Neighbors": KNNModel(),
    "Logistic Regression": LogisticRegressionModel(),
    "Elastic Net": ElasticNetModel(),
    "Seasonal Naive": SeasonalNaiveModel(),
    "ETS": ETSModel(),
}


def canonical_model_name(name: str) -> str:
    """*name* as the registry calls it now: a renamed model answers to what it
    was called before, through its `aliases`. A name the registry does not
    know is returned as it is."""
    if name in MODELS:
        return name
    return next((key for key, model in MODELS.items()
                 if name in getattr(model, "aliases", ())), name)

OPTIMIZERS = {
    "SMAC": SMACOptimizer(),
    "Random Search": RandomOptimizer(),
    "Grid Search": GridOptimizer(),
}

__all__ = ["MODELS", "OPTIMIZERS", "METRICS", "canonical_model_name"]
