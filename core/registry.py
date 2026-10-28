"""`registry` maps the names stored in `.ihpo` files to the models, optimizers
and metrics they denote.

The module lives in `core` because `remote/run_experiment.py` resolves the same
names on a machine without Django.
"""

from .metrics import METRICS
from .models import (
    CatBoostModel, ElasticNetModel, ETSModel, ExtraTreesModel, HistGradientBoostingModel,
    KNNModel, LightGBMModel, LogisticRegressionModel, RandomForestModel, SeasonalNaiveModel,
    SVMModel, XGBoostModel,
)
from .optimizers import GridOptimizer, RandomOptimizer, SMACOptimizer

#: Form order; the last two models only forecast.
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
    """`canonical_model_name` maps a former name in a model's `aliases` to its
    current key and returns an unknown name unchanged."""
    if name in MODELS:
        return name
    return next((key for key, model in MODELS.items()
                 if name in getattr(model, "aliases", ())), name)

OPTIMIZERS = {
    "SMAC": SMACOptimizer,
    "Random Search": RandomOptimizer,
    "Grid Search": GridOptimizer,
}


def optimizer_class(name: str, optimizers=OPTIMIZERS):
    """`optimizer_class` matches *name* or an alias, giving None for no match."""
    return next((kind for kind in optimizers.values()
                 if name == kind.name or name in kind.aliases), None)
